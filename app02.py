# -*- coding: utf-8 -*-
# app.py : 射出成形材料断面の結晶粒径解析（Streamlit 版）
# Author: 福田さん向け 提案版（ローカル/VSCode 実行想定）

import io
import os
import zipfile
import tempfile
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from skimage import measure, morphology, segmentation, exposure, util
from scipy import ndimage as ndi


# =========================================================
# 日本語フォント設定（文字化け対策）＋見た目（小さめ）
# =========================================================
def setup_japanese_font_and_style():
    candidates = [
        "Yu Gothic", "Yu Gothic UI",
        "Meiryo", "Meiryo UI",
        "MS Gothic", "MS PGothic",
        "BIZ UDゴシック", "BIZ UDPGothic",
        "Noto Sans CJK JP", "Noto Sans JP",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break

    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 9
    matplotlib.rcParams["axes.titlesize"] = 10
    matplotlib.rcParams["axes.labelsize"] = 9
    matplotlib.rcParams["xtick.labelsize"] = 8
    matplotlib.rcParams["ytick.labelsize"] = 8
    matplotlib.rcParams["legend.fontsize"] = 8
    matplotlib.rcParams["figure.autolayout"] = False
    matplotlib.rcParams["lines.linewidth"] = 1.5


setup_japanese_font_and_style()


# -------------------------------
# 共通ユーティリティ
# -------------------------------
def read_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    file_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("画像のデコードに失敗しました。ファイルが壊れている可能性があります。")
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    return img_gray


def compute_um_per_px(um_per_px: float,
                      scalebar_um: Optional[float],
                      scalebar_px: Optional[float]) -> float:
    if (scalebar_um and scalebar_px) and scalebar_px > 0:
        return float(scalebar_um) / float(scalebar_px)
    return float(um_per_px)


def apply_preprocess(img_gray: np.ndarray,
                     clip_limit: float,
                     gaussian_ksize: int,
                     gaussian_sigma: float) -> np.ndarray:
    img_eq = exposure.equalize_adapthist(img_gray, clip_limit=clip_limit)
    img8 = util.img_as_ubyte(img_eq)
    if gaussian_ksize > 0 and gaussian_ksize % 2 == 1:
        img8 = cv2.GaussianBlur(img8, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    return img8


def binarize(img: np.ndarray,
             method: str,
             manual_thresh: int,
             adaptive_block: int,
             adaptive_C: int) -> np.ndarray:
    if method == "otsu":
        thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bin_img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY_INV)
    elif method == "adaptive":
        block = max(3, adaptive_block | 1)
        bin_img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=block, C=adaptive_C
        )
    else:
        _, bin_img = cv2.threshold(img, manual_thresh, 255, cv2.THRESH_BINARY_INV)
    return bin_img


def morph_cleanup(bin_img: np.ndarray,
                  open_ksize: int, open_iter: int,
                  close_ksize: int, close_iter: int) -> np.ndarray:
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, open_ksize), max(1, open_ksize)))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, close_ksize), max(1, close_ksize)))
    out = bin_img.copy()
    if open_ksize > 0 and open_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=open_iter)
    if close_ksize > 0 and close_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
    return out


def split_touching_particles(bin_img: np.ndarray,
                             min_distance_px: int,
                             h_max: float) -> np.ndarray:
    distance = ndi.distance_transform_edt(bin_img)
    hmax = morphology.h_maxima(distance, h=h_max)
    markers = measure.label(hmax)
    labels = segmentation.watershed(-distance, markers, mask=bin_img.astype(bool))
    return labels


def label_by_connected_components(bin_img: np.ndarray) -> np.ndarray:
    return measure.label(bin_img, connectivity=2)


def min_area_rect_feret(coords_rc: np.ndarray) -> Tuple[float, float, float]:
    pts = np.fliplr(coords_rc).astype(np.float32)
    if len(pts) < 5:
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        w, h = (x_max - x_min), (y_max - y_min)
        feret_max, feret_min, angle = (max(w, h), min(w, h), 0.0)
    else:
        rect = cv2.minAreaRect(pts)
        (_, _), (w, h), angle = rect
        feret_max, feret_min = (max(w, h), min(w, h))
    return float(feret_max), float(feret_min), float(angle)


def extract_region_metrics(label_img: np.ndarray,
                           um_per_px: float,
                           exclude_largest: bool,
                           min_area_px: int,
                           min_area_um2: float) -> pd.DataFrame:
    props = measure.regionprops(label_img)
    if len(props) == 0:
        return pd.DataFrame()

    largest_label = None
    if exclude_largest:
        largest = max(props, key=lambda p: p.area)
        largest_label = largest.label

    rows = []
    for p in props:
        if largest_label and p.label == largest_label:
            continue

        area_px = float(p.area)
        area_um2 = area_px * (um_per_px ** 2)
        if area_px < max(0, min_area_px):
            continue
        if min_area_um2 > 0 and area_um2 < min_area_um2:
            continue

        ecd_px = float(p.equivalent_diameter)
        maj_px = float(getattr(p, "major_axis_length", 0.0))
        min_px = float(getattr(p, "minor_axis_length", 0.0))
        per_px = float(getattr(p, "perimeter", 0.0))
        cy, cx = p.centroid

        circularity = 4.0 * np.pi * area_px / (per_px ** 2 + 1e-9) if per_px > 0 else np.nan
        feret_max_px, feret_min_px, angle = min_area_rect_feret(p.coords)
        aspect = (maj_px / (min_px + 1e-9)) if (maj_px > 0 and min_px > 0) else np.nan

        rows.append({
            "label": int(p.label),
            "area_px": area_px,
            "perimeter_px": per_px,
            "equiv_diam_px": ecd_px,
            "major_axis_px": maj_px,
            "minor_axis_px": min_px,
            "aspect_ratio": aspect,
            "circularity": circularity,
            "feret_max_px": feret_max_px,
            "feret_min_px": feret_min_px,
            "orientation_deg": float(np.rad2deg(getattr(p, "orientation", 0.0))),
            "centroid_x_px": float(cx),
            "centroid_y_px": float(cy),

            "equiv_diam_um": ecd_px * um_per_px,
            "major_axis_um": maj_px * um_per_px,
            "minor_axis_um": min_px * um_per_px,
            "feret_max_um": feret_max_px * um_per_px,
            "feret_min_um": feret_min_px * um_per_px,
            "area_um2": area_um2,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("area_px", ascending=False, inplace=True, ignore_index=True)
        df.insert(0, "particle_id", np.arange(1, len(df) + 1))
    return df


def overlay_labels(img_gray: np.ndarray,
                   label_img: np.ndarray,
                   df: pd.DataFrame,
                   aspect_bins: Tuple[float, float] = (2.0, 3.0),
                   show_id: bool = True) -> np.ndarray:
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    low, high = aspect_bins

    for _, row in df.iterrows():
        lbl = int(row["label"])
        mask = (label_img == lbl)
        aspect = row["aspect_ratio"]

        if np.isnan(aspect):
            color = (200, 200, 200)
        elif aspect > high:
            color = (0, 0, 255)
        elif aspect > low:
            color = (0, 255, 255)
        else:
            color = (0, 200, 0)

        ys, xs = np.where(mask)
        img_color[ys, xs] = (0.6 * img_color[ys, xs] + 0.4 * np.array(color)).astype(np.uint8)

        if show_id:
            cx, cy = int(row["centroid_x_px"]), int(row["centroid_y_px"])
            cv2.putText(
                img_color, str(int(row["particle_id"])),
                (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA
            )
    return img_color


def plot_distributions(df: pd.DataFrame, xcols: List[str], group: Optional[str] = None):
    """
    ヒスト／箱ひげ／CDF を 2カラムで横並び表示する Streamlit 用描画関数。
    グラフが大きくなりすぎないよう figsize を小さく統一し、
    Streamlit の columns() で横並びにする。
    """
    if df.empty:
        st.info("有効な粒子がありません。しきい値・面積フィルタを調整してください。")
        return

    # グラフの基本サイズ（以前より小さめ）
    FIGSIZE = (3.5, 2.6)
    DPI = 110

    for x in xcols:
        st.markdown(f"### 指標：**{x}**")

        # === 2カラム作成 ===
        col1, col2 = st.columns(2)

        # -------------------------
        # ① ヒストグラム
        # -------------------------
        with col1:
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            if group and group in df.columns:
                for g, d in df.groupby(group):
                    ax.hist(d[x].dropna(), bins=30, alpha=0.5, label=str(g))
                ax.legend()
            else:
                ax.hist(df[x].dropna(), bins=30, color="steelblue", edgecolor="black")
            ax.grid(alpha=0.3)
            ax.set_xlabel(x)
            ax.set_ylabel("頻度")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        # -------------------------
        # ② 箱ひげ図
        # -------------------------
        with col2:
            fig2, ax2 = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            if group and group in df.columns:
                df.boxplot(column=x, by=group, ax=ax2, rot=45)
                ax2.set_title(f"{x}（group別）")
                fig2.suptitle("")
            else:
                df[[x]].boxplot(ax=ax2, vert=True)
                ax2.set_title(x)
            ax2.grid(alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2, clear_figure=True)

        # -------------------------
        # ③ さらに CDF を横全幅で表示
        # -------------------------
        st.markdown("#### CDF（累積分布）")
        fig3, ax3 = plt.subplots(figsize=(7, 2.4), dpi=DPI)
        d = df[x].dropna().sort_values()
        if len(d) > 0:
            cdf = np.arange(1, len(d) + 1) / len(d)
            ax3.plot(d, cdf, color="tomato")

        ax3.set_xlabel(x)
        ax3.set_ylabel("累積確率")
        ax3.grid(alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3, clear_figure=True)


def process_one_image(name: str,
                      file_bytes: bytes,
                      um_per_px: float,
                      threshold_method: str,
                      manual_thresh: int,
                      adaptive_block: int,
                      adaptive_C: int,
                      clahe_clip: float,
                      gauss_ksize: int,
                      gauss_sigma: float,
                      open_ksize: int, open_iter: int,
                      close_ksize: int, close_iter: int,
                      use_watershed: bool,
                      min_distance_px: int,
                      h_max: float,
                      exclude_largest: bool,
                      min_area_px: int,
                      min_area_um2: float,
                      aspect_bins: Tuple[float, float],
                      show_id: bool
                      ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:

    img_gray = read_image_from_bytes(file_bytes)
    img_pre = apply_preprocess(img_gray, clahe_clip, gauss_ksize, gauss_sigma)
    bin_img = binarize(img_pre, threshold_method, manual_thresh, adaptive_block, adaptive_C)
    bin_clean = morph_cleanup(bin_img, open_ksize, open_iter, close_ksize, close_iter)

    if use_watershed:
        label_img = split_touching_particles(bin_clean, min_distance_px, h_max)
    else:
        label_img = label_by_connected_components(bin_clean)

    df = extract_region_metrics(label_img, um_per_px, exclude_largest, min_area_px, min_area_um2)
    if not df.empty:
        df.insert(0, "source", name)

    overlay = overlay_labels(img_gray, label_img, df, aspect_bins, show_id) if not df.empty \
        else cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    return df, img_gray, bin_clean, overlay


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="結晶粒径解析（Streamlit）", layout="wide")
st.title("射出成形材料断面の結晶粒径解析")
st.caption("ローカル/VSCode 実行用のシンプル＆堅牢なアプリ。単一画像・複数画像・ZIP一括に対応。")

with st.sidebar:
    st.header("解析設定")

    st.subheader("スケール設定")
    col_scale = st.columns(2)
    with col_scale[0]:
        um_per_px_input = st.number_input("μm / px（直接）", min_value=0.0, value=1.0, step=0.01, format="%.4f")
    with col_scale[1]:
        st.caption("またはスケールバーから算出")
        scalebar_um = st.number_input("スケールバー長 [μm]", min_value=0.0, value=0.0, step=1.0)
        scalebar_px = st.number_input("スケールバー長 [px]", min_value=0.0, value=0.0, step=1.0)

    um_per_px = compute_um_per_px(
        um_per_px_input,
        None if scalebar_um == 0 else scalebar_um,
        None if scalebar_px == 0 else scalebar_px
    )

    st.subheader("前処理 & 二値化")
    clahe_clip = st.slider("CLAHE クリップ制限", 0.001, 0.050, 0.030, step=0.001)
    gauss_ksize = st.select_slider("Gaussian ksize(奇数)", options=[0, 3, 5, 7, 9], value=5)
    gauss_sigma = st.slider("Gaussian σ", 0.0, 5.0, 0.0, 0.1)

    method = st.selectbox("二値化方法", ["otsu", "adaptive", "manual"], index=0)
    manual_thresh = st.slider("手動しきい値（manual時）", 0, 255, 100, 1)
    adaptive_block = st.slider("適応（近傍）ブロックサイズ", 3, 101, 31, 2)
    adaptive_C = st.slider("適応しきい値 C", -20, 20, 0, 1)

    st.subheader("モルフォロジ")
    open_ksize = st.select_slider("Open カーネル", options=[0, 1, 2, 3, 4, 5, 6, 7], value=3)
    open_iter = st.slider("Open 回数", 0, 5, 1, 1)
    close_ksize = st.select_slider("Close カーネル", options=[0, 1, 2, 3, 4, 5, 6, 7], value=3)
    close_iter = st.slider("Close 回数", 0, 5, 1, 1)

    st.subheader("分離（Watershed）")
    use_watershed = st.toggle("接触粒子を分離する（Watershed）", value=True)
    min_distance_px = st.slider("局所極大の最小距離 [px]", 1, 50, 10, 1)
    h_max = st.slider("h-maxima（高いほど保守的）", 0.0, 10.0, 1.0, 0.1)

    st.subheader("フィルタ & 排除")
    exclude_largest = st.toggle("最大連結成分（母材）を除外", value=True)
    min_area_px = st.slider("最小面積 [px²]", 0, 1000, 20, 5)
    min_area_um2 = st.number_input("最小面積 [μm²]（0=無効）", min_value=0.0, value=0.0, step=1.0)

    st.subheader("オーバーレイ")
    aspect_low = st.slider("アスペクト比 境界1（緑→黄）", 1.0, 5.0, 2.0, 0.1)
    aspect_high = st.slider("アスペクト比 境界2（黄→赤）", 1.0, 10.0, 3.0, 0.1)
    show_id = st.toggle("粒子IDを表示", value=True)

    st.markdown("---")
    st.caption(
        "💡 チューニングのコツ：\n"
        "- 背景ムラが大きい→二値化を **adaptive**、ブロックサイズを素材の粒径より大きめに\n"
        "- 粒子が密集している→Watershed の h-max を上げつつ min_distance を粒径目安に\n"
        "- 過分割気味→h-max を上げる / min_distance を上げる / Open を弱くする"
    )

st.markdown("### 入力ファイル")
uploaded_files = st.file_uploader(
    "単一または複数の画像ファイル、または ZIP（画像入り）を選択してください。",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "zip"],
    accept_multiple_files=True
)

# -------------------------------
# 処理本体
# -------------------------------
if uploaded_files:
    to_process: List[Tuple[str, bytes]] = []
    for f in uploaded_files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    ext = os.path.splitext(info.filename.lower())[-1]
                    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
                        to_process.append((info.filename, zf.read(info)))
        else:
            to_process.append((f.name, f.read()))

    results: List[pd.DataFrame] = []
    overlays: Dict[str, np.ndarray] = {}
    previews: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    progress = st.progress(0)
    for idx, (name, bts) in enumerate(to_process, start=1):
        try:
            df, img_gray, bin_img, overlay_img = process_one_image(
                name, bts, um_per_px,
                method, manual_thresh, adaptive_block, adaptive_C,
                clahe_clip, gauss_ksize, gauss_sigma,
                open_ksize, open_iter, close_ksize, close_iter,
                use_watershed, min_distance_px, h_max,
                exclude_largest, min_area_px, min_area_um2,
                (aspect_low, aspect_high), show_id
            )
            if not df.empty:
                results.append(df)
                overlays[name] = overlay_img
                previews[name] = (img_gray, bin_img)
        except Exception as e:
            st.error(f"【{name}】の解析でエラー：{e}")

        progress.progress(int(100 * idx / max(1, len(to_process))))

    if len(results) == 0:
        st.warning("有効な結果がありませんでした。しきい値やフィルタを調整して再実行してください。")
    else:
        df_all = pd.concat(results, ignore_index=True)
        st.success(f"解析完了：{len(results)} ファイル、{len(df_all)} 粒子")

        st.markdown("### 可視化プレビュー")
        for name, (img_gray, bin_img) in previews.items():
            st.markdown(f"**{name}**")
            col = st.columns(3)
            with col[0]:
                st.image(img_gray, caption="元画像", use_column_width=True, clamp=True)
            with col[1]:
                st.image(bin_img, caption="二値（後処理込み）", use_column_width=True, clamp=True)
            with col[2]:
                st.image(cv2.cvtColor(overlays[name], cv2.COLOR_BGR2RGB),
                         caption="オーバーレイ（ID + アスペクト比色）",
                         use_column_width=True, clamp=True)

        st.markdown("### エクスポート")
        csv_bytes = df_all.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 粒子特性CSVをダウンロード",
            data=csv_bytes,
            file_name="grain_metrics.csv",
            mime="text/csv"
        )

        with tempfile.TemporaryDirectory() as tmpd:
            zip_path = os.path.join(tmpd, "overlays.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, img in overlays.items():
                    out_name = os.path.splitext(os.path.basename(name))[0] + "_overlay.png"
                    _, buf = cv2.imencode(".png", img)
                    zf.writestr(out_name, buf.tobytes())
            with open(zip_path, "rb") as fz:
                st.download_button(
                    "🖼️ 注釈画像（ZIP）をダウンロード",
                    data=fz.read(),
                    file_name="overlays.zip",
                    mime="application/zip"
                )

        st.markdown("### 統計可視化（まとめ）")
        plot_distributions(df_all, ["equiv_diam_um", "aspect_ratio", "circularity"], group="source")

        st.markdown("### 画像別サマリー")
        agg = df_all.groupby("source").agg(
            n=("particle_id", "count"),
            ecd_um_avg=("equiv_diam_um", "mean"),
            ecd_um_std=("equiv_diam_um", "std"),
            aspect_avg=("aspect_ratio", "mean"),
            circ_avg=("circularity", "mean"),
            area_um2_sum=("area_um2", "sum"),
        ).reset_index()
        st.dataframe(agg, use_container_width=True)
else:
    st.info("左下の **[Browse files]** から画像または ZIP を選択してください。")