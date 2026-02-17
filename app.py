# -*- coding: utf-8 -*-
# app.py : 射出成形材料断面の結晶粒径解析（Streamlit 版）
# Author: 福田さん向け 改善版（ローカル/VSCode 実行想定）
#
# === この版のポイント ===
# 1) 検出対象を切替可能：
#    - 白領域（材料/粒子）
#    - 黒領域（欠陥）：
#         a) 二値の黒（内部穴）方式（従来）
#         b) 元画像の深い黒点（ブラックハット）方式（NEW）
# 2) 欠陥マスク用Open/Close（サイドバー調整）
# 3) オーバーレイ輪郭＝赤（太さ調整、輪郭のみ表示可）
# 4) 欠陥率（A案）：欠陥総面積 / 材料面積（%）を画面表示＋CSV出力
# 5) use_container_width 統一
# 6) Watershed min_distance をピーク抽出に反映

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
from skimage import measure, morphology, segmentation, exposure, util
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


# =========================================================
# 日本語フォント設定（文字化け対策）＋見た目
# Streamlit Cloud 対応：fonts/ に同梱したフォントを優先的に使用
# =========================================================
def setup_japanese_font_and_style():
    import os
    from matplotlib import font_manager
    import matplotlib

    # 1) リポジトリ同梱フォント（最優先）
    #   fonts/NotoSansJP-Regular.otf などを置く想定
    local_font_candidates = [
        os.path.join("fonts", "NotoSansJP-Regular.otf"),
        os.path.join("fonts", "NotoSansJP-Regular.ttf"),
        os.path.join("fonts", "NotoSansCJKjp-Regular.otf"),
        os.path.join("fonts", "NotoSansCJKjp-Regular.ttf"),
    ]

    for fp in local_font_candidates:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                prop = font_manager.FontProperties(fname=fp)
                matplotlib.rcParams["font.family"] = prop.get_name()
                break
            except Exception:
                pass

    # 2) OSに入っているフォントから探す（ローカルWindows向け）
    if matplotlib.rcParams.get("font.family", None) is None or matplotlib.rcParams["font.family"] in [None, "sans-serif"]:
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

    # 3) 体裁（共通）
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 9
    matplotlib.rcParams["axes.titlesize"] = 10
    matplotlib.rcParams["axes.labelsize"] = 9
    matplotlib.rcParams["xtick.labelsize"] = 8
    matplotlib.rcParams["ytick.labelsize"] = 8
    matplotlib.rcParams["legend.fontsize"] = 8
    matplotlib.rcParams["figure.autolayout"] = False
    matplotlib.rcParams["lines.linewidth"] = 1.5


# =========================================================
# 共通ユーティリティ
# =========================================================
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
    # 局所コントラスト強調（ムラ補正）→二値化や暗点抽出の助け
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
    """
    出力: 0/255 uint8
    THRESH_BINARY_INV: 暗い部分を白にする
    """
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
    out = bin_img.copy()
    if open_ksize > 0 and open_iter > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=open_iter)
    if close_ksize > 0 and close_iter > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
    return out


# =========================================================
# 材料マスク（最大連結成分）
# =========================================================
def largest_component_mask(bin_u8: np.ndarray) -> np.ndarray:
    lab = measure.label(bin_u8 > 0, connectivity=2)
    if lab.max() == 0:
        return np.zeros_like(bin_u8, dtype=bool)
    props = measure.regionprops(lab)
    largest = max(props, key=lambda p: p.area)
    return lab == largest.label


# =========================================================
# 欠陥抽出（黒領域）
#   (A) 二値の黒（内部穴）方式
# =========================================================
def extract_internal_black_defects(bin_clean_u8: np.ndarray,
                                   assume_material_is_largest: bool = True) -> np.ndarray:
    if assume_material_is_largest:
        material = largest_component_mask(bin_clean_u8)
    else:
        material = (bin_clean_u8 > 0)

    filled = ndi.binary_fill_holes(material)
    holes = filled & (~material)
    return (holes.astype(np.uint8) * 255)


# =========================================================
# 欠陥抽出（黒領域）
#   (B) 元画像の深い黒点（ブラックハット）方式（NEW）
# =========================================================
def extract_dark_spots_blackhat(img_u8: np.ndarray,
                                material_mask_u8: np.ndarray,
                                bh_ksize: int,
                                thresh_mode: str,
                                manual_thr: int,
                                border_exclude_px: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    - img_u8: 元画像(0-255) または前処理後画像
    - material_mask_u8: 0/255 材料領域マスク（背景除外）
    - bh_ksize: ブラックハットのカーネルサイズ（欠陥より少し大きく）
    - thresh_mode: 'otsu' or 'manual'
    - manual_thr: 手動しきい値
    - border_exclude_px: 材料境界近傍を除外（影/縁の偽検出対策）

    return:
      defect_mask_u8 (0/255), blackhat_u8 (0/255相当の強調画像)
    """
    # 材料境界の除外（任意）
    mat = (material_mask_u8 > 0).astype(np.uint8) * 255
    if border_exclude_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*border_exclude_px+1, 2*border_exclude_px+1))
        mat = cv2.erode(mat, k, iterations=1)

    # ブラックハット：暗点を強調
    ksz = max(3, int(bh_ksize) | 1)  # 奇数化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    blackhat = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, kernel)

    # 材料領域だけに限定
    blackhat_roi = cv2.bitwise_and(blackhat, blackhat, mask=mat)

    # 二値化
    if thresh_mode == "otsu":
        # Otsuは背景が少ないROIでも効くが、状況により過剰/過小があり得る
        thr, defect = cv2.threshold(blackhat_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thr = int(manual_thr)
        _, defect = cv2.threshold(blackhat_roi, thr, 255, cv2.THRESH_BINARY)

    return defect, blackhat_roi


# =========================================================
# 欠陥率（A案）：欠陥総面積 / 材料面積
# =========================================================
def compute_area_stats_A(bin_clean_u8: np.ndarray,
                         defect_mask_u8: np.ndarray,
                         um_per_px: float,
                         assume_material_is_largest: bool = True) -> Dict[str, float]:
    if assume_material_is_largest:
        material_mask = largest_component_mask(bin_clean_u8)
    else:
        material_mask = (bin_clean_u8 > 0)

    material_area_px = float(np.count_nonzero(material_mask))
    defect_area_px = float(np.count_nonzero(defect_mask_u8 > 0))

    material_area_um2 = material_area_px * (um_per_px ** 2)
    defect_area_um2 = defect_area_px * (um_per_px ** 2)

    defect_ratio_percent = (defect_area_px / (material_area_px + 1e-9)) * 100.0

    return {
        "material_area_px": material_area_px,
        "defect_area_px": defect_area_px,
        "material_area_um2": material_area_um2,
        "defect_area_um2": defect_area_um2,
        "defect_ratio_percent": defect_ratio_percent
    }


# =========================================================
# Watershed
# =========================================================
def split_touching_particles(bin_u8: np.ndarray,
                             min_distance_px: int,
                             h_max: float) -> np.ndarray:
    mask = (bin_u8 > 0)
    distance = ndi.distance_transform_edt(mask)

    if h_max > 0:
        _ = morphology.h_maxima(distance, h=h_max)

    coords = peak_local_max(
        distance,
        min_distance=max(1, int(min_distance_px)),
        labels=mask,
        exclude_border=False
    )

    markers = np.zeros_like(distance, dtype=np.int32)
    if coords.size > 0:
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
    else:
        markers = measure.label(mask, connectivity=2).astype(np.int32)

    labels = segmentation.watershed(-distance, markers, mask=mask)
    return labels


def label_by_connected_components(bin_u8: np.ndarray) -> np.ndarray:
    return measure.label(bin_u8 > 0, connectivity=2)


# =========================================================
# 計測
# =========================================================
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
        feret_max_px, feret_min_px, _ = min_area_rect_feret(p.coords)
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


# =========================================================
# オーバーレイ（輪郭＝赤）
# =========================================================
def overlay_labels(img_gray: np.ndarray,
                   label_img: np.ndarray,
                   df: pd.DataFrame,
                   aspect_bins: Tuple[float, float] = (2.0, 3.0),
                   show_id: bool = True,
                   fill_alpha: float = 0.25,
                   draw_red_contour: bool = True,
                   contour_thickness: int = 3,
                   contour_only: bool = False) -> np.ndarray:

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if df.empty:
        return img_color

    low, high = aspect_bins

    keep_labels = df["label"].astype(int).values
    keep_mask = np.isin(label_img, keep_labels)
    label_keep = label_img.copy()
    label_keep[~keep_mask] = 0

    # 塗りつぶし（任意）
    if not contour_only:
        a = float(np.clip(fill_alpha, 0.0, 1.0))
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
            if len(xs) == 0:
                continue
            img_color[ys, xs] = ((1 - a) * img_color[ys, xs] + a * np.array(color)).astype(np.uint8)

    # 輪郭＝赤
    if draw_red_contour:
        boundary = segmentation.find_boundaries(label_keep, mode="outer")
        bnd = (boundary.astype(np.uint8) * 255)

        t = max(1, int(contour_thickness))
        if t > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
            bnd = cv2.dilate(bnd, k, iterations=1)

        ys, xs = np.where(bnd > 0)
        img_color[ys, xs] = (0, 0, 255)

    # ID表示
    if show_id:
        for _, row in df.iterrows():
            cx, cy = int(row["centroid_x_px"]), int(row["centroid_y_px"])
            cv2.putText(img_color, str(int(row["particle_id"])),
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return img_color


# =========================================================
# 統計プロット
# =========================================================
def plot_distributions(df: pd.DataFrame, xcols: List[str], group: Optional[str] = None):
    if df.empty:
        st.info("有効な粒子がありません。しきい値・面積フィルタを調整してください。")
        return

    FIGSIZE = (3.5, 2.6)
    DPI = 110

    for x in xcols:
        st.markdown(f"### 指標：**{x}**")
        col1, col2 = st.columns(2)

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


# =========================================================
# 画像1枚処理
# =========================================================
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
                      # 欠陥マスク用後処理
                      defect_open_ksize: int, defect_open_iter: int,
                      defect_close_ksize: int, defect_close_iter: int,
                      # 対象切替
                      target_mode: str,
                      assume_material_is_largest: bool,
                      defect_mode_black: str,
                      # Blackhat設定
                      bh_use_preprocessed: bool,
                      bh_ksize: int,
                      bh_thresh_mode: str,
                      bh_manual_thr: int,
                      bh_border_exclude: int,
                      # ラベリング
                      use_watershed: bool,
                      min_distance_px: int,
                      h_max: float,
                      # フィルタ等
                      exclude_largest: bool,
                      min_area_px: int,
                      min_area_um2: float,
                      # overlay
                      aspect_bins: Tuple[float, float],
                      show_id: bool,
                      fill_alpha: float,
                      draw_red_contour: bool,
                      contour_thickness: int,
                      contour_only: bool
                      ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    img_gray = read_image_from_bytes(file_bytes)
    img_pre = apply_preprocess(img_gray, clahe_clip, gauss_ksize, gauss_sigma)

    # 材料マスク用途にも使う「二値（後処理込み）」を常に作る
    bin_img = binarize(img_pre, threshold_method, manual_thresh, adaptive_block, adaptive_C)
    bin_clean = morph_cleanup(bin_img, open_ksize, open_iter, close_ksize, close_iter)

    debug_bh = np.zeros_like(img_gray, dtype=np.uint8)

    # --- 解析対象の切替 ---
    if target_mode == "黒領域（欠陥）":
        # 材料領域（背景除外）マスク：bin_clean から最大連結成分
        material_mask_u8 = (largest_component_mask(bin_clean).astype(np.uint8) * 255)

        if defect_mode_black == "二値の黒（内部穴）":
            defect_mask = extract_internal_black_defects(
                bin_clean,
                assume_material_is_largest=assume_material_is_largest
            )
            debug_bh = np.zeros_like(img_gray, dtype=np.uint8)

        else:
            # 元画像の深い黒点（ブラックハット）
            img_used = img_pre if bh_use_preprocessed else img_gray
            defect_mask, debug_bh = extract_dark_spots_blackhat(
                img_u8=img_used.astype(np.uint8),
                material_mask_u8=material_mask_u8,
                bh_ksize=bh_ksize,
                thresh_mode=bh_thresh_mode,
                manual_thr=bh_manual_thr,
                border_exclude_px=bh_border_exclude
            )

        # 欠陥マスク後処理（ノイズ除去）
        defect_mask = morph_cleanup(
            defect_mask,
            defect_open_ksize, defect_open_iter,
            defect_close_ksize, defect_close_iter
        )

        bin_target = defect_mask

    else:
        # 白領域（材料/粒子）
        bin_target = bin_clean
        debug_bh = np.zeros_like(img_gray, dtype=np.uint8)

    # --- ラベリング ---
    if use_watershed:
        label_img = split_touching_particles(bin_target, min_distance_px, h_max)
    else:
        label_img = label_by_connected_components(bin_target)

    # --- 計測 ---
    df = extract_region_metrics(label_img, um_per_px, exclude_largest, min_area_px, min_area_um2)
    if not df.empty:
        df.insert(0, "source", name)
        df.insert(1, "target_mode", target_mode)
        if target_mode == "黒領域（欠陥）":
            df.insert(2, "defect_mode", defect_mode_black)

    # --- overlay ---
    overlay = overlay_labels(
        img_gray, label_img, df, aspect_bins,
        show_id=show_id,
        fill_alpha=fill_alpha,
        draw_red_contour=draw_red_contour,
        contour_thickness=contour_thickness,
        contour_only=contour_only
    ) if not df.empty else cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    return df, img_gray, bin_clean, bin_target, debug_bh, overlay


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="欠陥（空隙）解析（Streamlit）", layout="wide")
st.title("射出成形材料断面の欠陥（空隙）解析")
st.caption("黒欠陥：二値の穴方式／元画像の深い黒点（ブラックハット）方式の両対応。欠陥率(A案)も算出。")

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

    st.subheader("前処理（コントラスト）")
    clahe_clip = st.slider("CLAHE クリップ制限", 0.001, 0.050, 0.030, step=0.001)
    gauss_ksize = st.select_slider("Gaussian ksize(奇数)", options=[0, 3, 5, 7, 9], value=5)
    gauss_sigma = st.slider("Gaussian σ", 0.0, 5.0, 0.0, 0.1)

    st.subheader("二値化（材料マスクの安定化にも利用）")
    method = st.selectbox("二値化方法", ["otsu", "adaptive", "manual"], index=0)
    manual_thresh = st.slider("手動しきい値（manual時）", 0, 255, 100, 1)
    adaptive_block = st.slider("適応（近傍）ブロックサイズ", 3, 101, 31, 2)
    adaptive_C = st.slider("適応しきい値 C", -20, 20, 0, 1)

    st.subheader("モルフォロジ（二値後処理）")
    open_ksize = st.select_slider("Open カーネル", options=[0, 1, 2, 3, 4, 5, 6, 7], value=3)
    open_iter = st.slider("Open 回数", 0, 5, 1, 1)
    close_ksize = st.select_slider("Close カーネル", options=[0, 1, 2, 3, 4, 5, 6, 7], value=3)
    close_iter = st.slider("Close 回数", 0, 5, 1, 1)

    st.subheader("解析対象（重要）")
    target_mode = st.selectbox(
        "どの領域を検出する？",
        ["黒領域（欠陥）", "白領域（材料/粒子）"],
        index=0
    )
    assume_material_is_largest = st.toggle(
        "材料は最大連結成分（白）とみなす（推奨）",
        value=True
    )

    st.subheader("黒欠陥の検出方式（黒領域モード時）")
    defect_mode_black = st.selectbox(
        "欠陥（黒）の検出方式",
        ["元画像の深い黒点（ブラックハット）", "二値の黒（内部穴）"],
        index=0
    )

    st.subheader("ブラックハット設定（深い黒点方式）")
    bh_use_preprocessed = st.toggle("前処理後画像（CLAHE+Gaussian）を使う", value=True)
    bh_ksize = st.slider("ブラックハット カーネルサイズ（欠陥より少し大きく）", 3, 51, 11, 2)
    bh_thresh_mode = st.selectbox("ブラックハットの二値化", ["otsu", "manual"], index=1)
    bh_manual_thr = st.slider("ブラックハット 手動しきい値", 1, 100, 25, 1)
    bh_border_exclude = st.slider("材料境界を除外する幅 [px]（縁の偽検出対策）", 0, 30, 3, 1)

    st.subheader("欠陥マスク用 後処理（ノイズ除去）")
    defect_open_ksize = st.select_slider("欠陥Open カーネル", options=[0, 1, 2, 3, 4, 5, 6, 7], value=0)
    defect_open_iter = st.slider("欠陥Open 回数", 0, 5, 0, 1)
    defect_close_ksize = st.select_slider("欠陥Close カーネル", options=[0, 1, 2, 3, 4, 5, 6, 7], value=0)
    defect_close_iter = st.slider("欠陥Close 回数", 0, 5, 0, 1)

    st.subheader("分離（Watershed）")
    use_watershed = st.toggle("接触欠陥/粒子を分離する（Watershed）", value=False)
    min_distance_px = st.slider("局所極大の最小距離 [px]", 1, 50, 10, 1)
    h_max = st.slider("h-maxima（高いほど保守的）", 0.0, 10.0, 1.0, 0.1)

    st.subheader("フィルタ")
    exclude_largest = st.toggle("最大連結成分を除外（通常はOFF推奨）", value=False)
    min_area_px = st.slider("最小面積 [px²]（小ノイズ除去）", 0, 5000, 10, 5)
    min_area_um2 = st.number_input("最小面積 [μm²]（0=無効）", min_value=0.0, value=0.0, step=1.0)

    st.subheader("オーバーレイ（輪郭強調）")
    aspect_low = st.slider("アスペクト比 境界1（緑→黄）", 1.0, 5.0, 2.0, 0.1)
    aspect_high = st.slider("アスペクト比 境界2（黄→赤）", 1.0, 10.0, 3.0, 0.1)
    show_id = st.toggle("ID表示", value=True)

    draw_red_contour = st.toggle("輪郭を赤で描画", value=True)
    contour_thickness = st.slider("輪郭の太さ", 1, 8, 3, 1)
    contour_only = st.toggle("輪郭のみ（塗りつぶし無し）", value=True)
    fill_alpha = st.slider("塗りつぶし透明度", 0.0, 0.8, 0.25, 0.05)

    st.markdown("---")
    st.caption(
        "💡 深い黒点（空隙）狙いの推奨：\n"
        "- 解析対象=黒領域（欠陥）\n"
        "- 欠陥方式=元画像の深い黒点（ブラックハット）\n"
        "- カーネルサイズ=欠陥径より少し大きめ\n"
        "- 手動しきい値を上げると『より深い黒』だけに絞れます\n"
        "- 欠陥率(A案)=欠陥総面積/材料面積(%)"
    )

st.markdown("### 入力ファイル")
uploaded_files = st.file_uploader(
    "単一または複数の画像ファイル、または ZIP（画像入り）を選択してください。",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "zip"],
    accept_multiple_files=True
)

# =========================================================
# 処理本体
# =========================================================
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
    previews: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    summaries: List[Dict[str, float]] = []

    progress = st.progress(0)
    for idx, (name, bts) in enumerate(to_process, start=1):
        try:
            df, img_gray, bin_clean, bin_target, debug_bh, overlay_img = process_one_image(
                name, bts, um_per_px,
                method, manual_thresh, adaptive_block, adaptive_C,
                clahe_clip, gauss_ksize, gauss_sigma,
                open_ksize, open_iter, close_ksize, close_iter,
                defect_open_ksize, defect_open_iter, defect_close_ksize, defect_close_iter,
                target_mode, assume_material_is_largest,
                defect_mode_black,
                bh_use_preprocessed, bh_ksize, bh_thresh_mode, bh_manual_thr, bh_border_exclude,
                use_watershed, min_distance_px, h_max,
                exclude_largest, min_area_px, min_area_um2,
                (aspect_low, aspect_high), show_id,
                fill_alpha, draw_red_contour, contour_thickness, contour_only
            )

            overlays[name] = overlay_img
            previews[name] = (img_gray, bin_clean, bin_target, debug_bh)

            if not df.empty:
                results.append(df)

            # 欠陥率サマリー（A案）
            defect_mask_for_ratio = bin_target if target_mode == "黒領域（欠陥）" else np.zeros_like(bin_clean)
            stats = compute_area_stats_A(
                bin_clean_u8=bin_clean,
                defect_mask_u8=defect_mask_for_ratio,
                um_per_px=um_per_px,
                assume_material_is_largest=assume_material_is_largest
            )
            stats.update({
                "source": name,
                "target_mode": target_mode,
                "defect_mode": defect_mode_black if target_mode == "黒領域（欠陥）" else "-"
            })
            summaries.append(stats)

        except Exception as e:
            st.error(f"【{name}】の解析でエラー：{e}")

        progress.progress(int(100 * idx / max(1, len(to_process))))

    df_all = pd.concat(results, ignore_index=True) if len(results) > 0 else pd.DataFrame()
    df_sum = pd.DataFrame(summaries) if len(summaries) > 0 else pd.DataFrame()

    # --- プレビュー ---
    st.markdown("### 可視化プレビュー")
    show_blackhat = (target_mode == "黒領域（欠陥）" and defect_mode_black == "元画像の深い黒点（ブラックハット）")

    for name, (img_gray, bin_clean, bin_target, debug_bh) in previews.items():
        st.markdown(f"**{name}**")
        if show_blackhat:
            cols = st.columns(5)
            with cols[0]:
                st.image(img_gray, caption="元画像", use_container_width=True, clamp=True)
            with cols[1]:
                st.image(bin_clean, caption="二値（後処理込み）", use_container_width=True, clamp=True)
            with cols[2]:
                st.image(debug_bh, caption="ブラックハット強調（ROI）", use_container_width=True, clamp=True)
            with cols[3]:
                st.image(bin_target, caption="検出対象マスク（欠陥）", use_container_width=True, clamp=True)
            with cols[4]:
                st.image(cv2.cvtColor(overlays[name], cv2.COLOR_BGR2RGB),
                         caption="オーバーレイ（輪郭=赤）",
                         use_container_width=True, clamp=True)
        else:
            cols = st.columns(4)
            with cols[0]:
                st.image(img_gray, caption="元画像", use_container_width=True, clamp=True)
            with cols[1]:
                st.image(bin_clean, caption="二値（後処理込み）", use_container_width=True, clamp=True)
            with cols[2]:
                st.image(bin_target, caption=f"検出対象マスク：{target_mode}", use_container_width=True, clamp=True)
            with cols[3]:
                st.image(cv2.cvtColor(overlays[name], cv2.COLOR_BGR2RGB),
                         caption="オーバーレイ（輪郭=赤）",
                         use_container_width=True, clamp=True)

    # --- 欠陥率サマリー ---
    if not df_sum.empty:
        st.markdown("### 欠陥率サマリー（A案：欠陥総面積 / 材料面積）")
        df_sum_disp = df_sum[[
            "source", "target_mode", "defect_mode",
            "material_area_um2", "defect_area_um2", "defect_ratio_percent",
            "material_area_px", "defect_area_px"
        ]].copy()

        df_sum_disp.rename(columns={
            "material_area_um2": "材料面積 [μm²]",
            "defect_area_um2": "欠陥総面積 [μm²]",
            "defect_ratio_percent": "欠陥率 [%]（欠陥/材料）",
            "material_area_px": "材料面積 [px²]",
            "defect_area_px": "欠陥総面積 [px²]",
        }, inplace=True)

        df_sum_disp["材料面積 [μm²]"] = df_sum_disp["材料面積 [μm²]"].round(2)
        df_sum_disp["欠陥総面積 [μm²]"] = df_sum_disp["欠陥総面積 [μm²]"].round(2)
        df_sum_disp["欠陥率 [%]（欠陥/材料）"] = df_sum_disp["欠陥率 [%]（欠陥/材料）"].round(4)

        st.dataframe(df_sum_disp, use_container_width=True)

        sum_csv = df_sum_disp.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 欠陥率サマリーCSVをダウンロード",
            data=sum_csv,
            file_name="defect_area_ratio_summary_A.csv",
            mime="text/csv"
        )

    # --- 粒子/欠陥 特性CSV ---
    if not df_all.empty:
        st.markdown("### エクスポート（欠陥/粒子 特性）")
        csv_bytes = df_all.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 欠陥/粒子 特性CSVをダウンロード",
            data=csv_bytes,
            file_name="grain_or_defect_metrics.csv",
            mime="text/csv"
        )

    # --- オーバーレイZIP ---
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

    # --- 統計可視化 ---
    if not df_all.empty:
        st.markdown("### 統計可視化（形状指標）")
        plot_distributions(df_all, ["equiv_diam_um", "aspect_ratio", "circularity"], group="source")

else:
    st.info("左下の **[Browse files]** から画像または ZIP を選択してください。")
