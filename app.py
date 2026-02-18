# -*- coding: utf-8 -*-
# app01.py : 射出成形材料断面の欠陥（空隙）解析（Streamlit 版）
#
# 統合版（完全版）
# - 母材マスク（背景黒樹脂の除外：四隅 flood fill）
# - 母材輪郭（緑）表示
# - 右下スケール表示を除外（割合指定の矩形）
# - 可視化プレビュー：選択した1枚だけ表示
# - オーバーレイ（最終抽出）を下段に拡大表示（高さ調整）
# - 欠陥率（A案：欠陥総面積 / 材料面積）
# - CSV出力、オーバーレイZIP出力、簡易統計可視化

import io
import os
import sys
import zipfile
import tempfile
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import cv2
import streamlit as st

import matplotlib
import matplotlib.pyplot as plt

# 日本語フォント（matplotlib-fontja）
# ※ requirements.txt に matplotlib-fontja が入っている前提
try:
    import matplotlib_fontja  # noqa: F401
    FONTJA_OK = True
except Exception:
    FONTJA_OK = False

from skimage import measure, morphology, segmentation, exposure, util
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


# =========================================================
# Matplotlib 体裁
# =========================================================
def setup_matplotlib_style():
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 9
    matplotlib.rcParams["axes.titlesize"] = 10
    matplotlib.rcParams["axes.labelsize"] = 9
    matplotlib.rcParams["xtick.labelsize"] = 8
    matplotlib.rcParams["ytick.labelsize"] = 8
    matplotlib.rcParams["legend.fontsize"] = 8
    matplotlib.rcParams["figure.autolayout"] = False
    matplotlib.rcParams["lines.linewidth"] = 1.5


setup_matplotlib_style()


# =========================================================
# 表示用：高さ指定でリサイズ（拡大表示用）
# =========================================================
def resize_to_height(img_bgr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h <= 0:
        return img_bgr
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    # 輪郭のにじみを抑えたいので NEAREST
    return cv2.resize(img_bgr, (new_w, int(target_h)), interpolation=cv2.INTER_NEAREST)


# =========================================================
# 右下スケール除外：割合で矩形マスク作成（白=除外）
# =========================================================
def make_bottom_right_exclude_mask(shape_hw: Tuple[int, int],
                                   w_ratio: float,
                                   h_ratio: float,
                                   pad: int = 0) -> np.ndarray:
    """
    shape_hw: (H, W)
    右下の矩形を255で塗った除外マスクを返す（0/255）
    """
    h, w = shape_hw
    ex = np.zeros((h, w), dtype=np.uint8)
    bw = int(w * float(w_ratio))
    bh = int(h * float(h_ratio))
    x0 = max(0, w - bw - int(pad))
    y0 = max(0, h - bh - int(pad))
    ex[y0:h, x0:w] = 255
    return ex


# =========================================================
# 母材（試験片）マスク：四隅から flood fill で背景を除外
# =========================================================
def compute_specimen_mask_floodfill(img_gray: np.ndarray,
                                    tol: int = 20,
                                    close_ksize: int = 21,
                                    close_iter: int = 2) -> np.ndarray:
    """
    四隅から flood fill して背景を抽出し、反転して母材マスク(0/255)を返す。
    tol: flood fill の許容差（大きいほど背景を広く拾う）
    close_ksize/iter: 母材マスクの穴埋め・連結強化
    """
    h, w = img_gray.shape[:2]
    work = img_gray.copy()

    # floodFill 用マスク（OpenCV仕様で +2）
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]

    for sx, sy in seeds:
        cv2.floodFill(work, ff_mask, (sx, sy), 0, loDiff=int(tol), upDiff=int(tol))

    bg_mask = ff_mask[1:h+1, 1:w+1] > 0  # Trueが背景
    specimen = (~bg_mask).astype(np.uint8) * 255

    # 穴埋め・連結強化
    ksz = max(3, int(close_ksize) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    if int(close_iter) > 0:
        specimen = cv2.morphologyEx(specimen, cv2.MORPH_CLOSE, k, iterations=int(close_iter))

    # 最大連結成分のみ残す（小ゴミ除外）
    lab = measure.label(specimen > 0, connectivity=2)
    if lab.max() > 0:
        props = measure.regionprops(lab)
        largest = max(props, key=lambda p: p.area)
        specimen = ((lab == largest.label).astype(np.uint8) * 255)

    return specimen


# =========================================================
# 母材輪郭を描画（緑）
# =========================================================
def draw_mask_contour_on_gray(img_gray: np.ndarray,
                              mask_u8: np.ndarray,
                              color_bgr=(0, 255, 0),
                              thickness: int = 2) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    m = (mask_u8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img_bgr, contours, -1, color_bgr, int(thickness))
    return img_bgr


# =========================================================
# 共通ユーティリティ
# =========================================================
def read_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    file_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("画像のデコードに失敗しました。ファイルが壊れている可能性があります。")
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


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
    """
    出力: 0/255 uint8
    THRESH_BINARY_INV: 暗い部分を白にする
    """
    if method == "otsu":
        thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bin_img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY_INV)
    elif method == "adaptive":
        block = max(3, int(adaptive_block) | 1)
        bin_img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=block, C=int(adaptive_C)
        )
    else:
        _, bin_img = cv2.threshold(img, int(manual_thresh), 255, cv2.THRESH_BINARY_INV)
    return bin_img


def morph_cleanup(bin_img: np.ndarray,
                  open_ksize: int, open_iter: int,
                  close_ksize: int, close_iter: int) -> np.ndarray:
    out = bin_img.copy()
    if open_ksize > 0 and open_iter > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_ksize), int(open_ksize)))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=int(open_iter))
    if close_ksize > 0 and close_iter > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=int(close_iter))
    return out


# =========================================================
# Watershed（接触分離）
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

    return segmentation.watershed(-distance, markers, mask=mask)


def label_by_connected_components(bin_u8: np.ndarray) -> np.ndarray:
    return measure.label(bin_u8 > 0, connectivity=2)


# =========================================================
# 欠陥抽出（内部穴 / ブラックハット）
# =========================================================
def largest_component_mask(bin_u8: np.ndarray) -> np.ndarray:
    lab = measure.label(bin_u8 > 0, connectivity=2)
    if lab.max() == 0:
        return np.zeros_like(bin_u8, dtype=bool)
    props = measure.regionprops(lab)
    largest = max(props, key=lambda p: p.area)
    return lab == largest.label


def extract_internal_black_defects(bin_clean_u8: np.ndarray,
                                   assume_material_is_largest: bool = True) -> np.ndarray:
    """
    二値（母材内）から内部穴を抽出
    """
    if assume_material_is_largest:
        material = largest_component_mask(bin_clean_u8)
    else:
        material = (bin_clean_u8 > 0)

    filled = ndi.binary_fill_holes(material)
    holes = filled & (~material)
    return (holes.astype(np.uint8) * 255)


def extract_dark_spots_blackhat(img_u8: np.ndarray,
                                material_mask_u8: np.ndarray,
                                bh_ksize: int,
                                thresh_mode: str,
                                manual_thr: int,
                                border_exclude_px: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    ブラックハットで「局所的暗点」を抽出
    """
    mat = (material_mask_u8 > 0).astype(np.uint8) * 255

    if border_exclude_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(border_exclude_px) + 1, 2 * int(border_exclude_px) + 1)
        )
        mat = cv2.erode(mat, k, iterations=1)

    ksz = max(3, int(bh_ksize) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    blackhat = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, kernel)
    blackhat_roi = cv2.bitwise_and(blackhat, blackhat, mask=mat)

    if thresh_mode == "otsu":
        _, defect = cv2.threshold(blackhat_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, defect = cv2.threshold(blackhat_roi, int(manual_thr), 255, cv2.THRESH_BINARY)

    return defect, blackhat_roi


# =========================================================
# 欠陥率（A案）：欠陥総面積 / 材料面積（材料=母材マスク）
# =========================================================
def compute_area_stats_A(material_mask_u8: np.ndarray,
                         defect_mask_u8: np.ndarray,
                         um_per_px: float) -> Dict[str, float]:
    material_area_px = float(np.count_nonzero(material_mask_u8 > 0))
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
# 計測（欠陥の形状指標）
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
                           min_area_px: int,
                           min_area_um2: float) -> pd.DataFrame:
    props = measure.regionprops(label_img)
    if len(props) == 0:
        return pd.DataFrame()

    rows = []
    for p in props:
        area_px = float(p.area)
        area_um2 = area_px * (um_per_px ** 2)

        if area_px < max(0, int(min_area_px)):
            continue
        if min_area_um2 > 0 and area_um2 < float(min_area_um2):
            continue

        ecd_px = float(p.equivalent_diameter)
        per_px = float(getattr(p, "perimeter", 0.0))
        maj_px = float(getattr(p, "major_axis_length", 0.0))
        min_px = float(getattr(p, "minor_axis_length", 0.0))
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
            "centroid_x_px": float(cx),
            "centroid_y_px": float(cy),
            "equiv_diam_um": ecd_px * um_per_px,
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
# オーバーレイ（輪郭=赤）
# =========================================================
def overlay_labels(img_gray: np.ndarray,
                   label_img: np.ndarray,
                   df: pd.DataFrame,
                   show_id: bool = True,
                   fill_alpha: float = 0.25,
                   draw_red_contour: bool = True,
                   contour_thickness: int = 3,
                   contour_only: bool = False) -> np.ndarray:

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if df.empty:
        return img_color

    keep_labels = df["label"].astype(int).values
    keep_mask = np.isin(label_img, keep_labels)
    label_keep = label_img.copy()
    label_keep[~keep_mask] = 0

    # 塗りつぶし
    if not contour_only:
        a = float(np.clip(fill_alpha, 0.0, 1.0))
        for _, row in df.iterrows():
            lbl = int(row["label"])
            mask = (label_img == lbl)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            color = (0, 0, 255)  # 赤
            img_color[ys, xs] = ((1 - a) * img_color[ys, xs] + a * np.array(color)).astype(np.uint8)

    # 輪郭
    if draw_red_contour:
        boundary = segmentation.find_boundaries(label_keep, mode="outer")
        bnd = (boundary.astype(np.uint8) * 255)
        t = max(1, int(contour_thickness))
        if t > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
            bnd = cv2.dilate(bnd, k, iterations=1)
        ys, xs = np.where(bnd > 0)
        img_color[ys, xs] = (0, 0, 255)

    # ID
    if show_id:
        for _, row in df.iterrows():
            cx, cy = int(row["centroid_x_px"]), int(row["centroid_y_px"])
            cv2.putText(img_color, str(int(row["particle_id"])),
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return img_color


# =========================================================
# 統計プロット（任意）
# =========================================================
def plot_distributions(df: pd.DataFrame, xcols: List[str], group: Optional[str] = None):
    if df.empty:
        st.info("有効な欠陥がありません。しきい値・面積フィルタを調整してください。")
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


# =========================================================
# 画像1枚処理（母材マスク＋スケール除外込み）
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
                      defect_open_ksize: int, defect_open_iter: int,
                      defect_close_ksize: int, defect_close_iter: int,
                      target_mode: str,
                      defect_mode_black: str,
                      bh_use_preprocessed: bool,
                      bh_ksize: int,
                      bh_thresh_mode: str,
                      bh_manual_thr: int,
                      bh_border_exclude: int,
                      use_watershed: bool,
                      min_distance_px: int,
                      h_max: float,
                      min_area_px: int,
                      min_area_um2: float,
                      show_id: bool,
                      fill_alpha: float,
                      draw_red_contour: bool,
                      contour_thickness: int,
                      contour_only: bool,
                      # 母材マスク
                      use_specimen_mask: bool,
                      ff_tol: int,
                      ff_close_ksize: int,
                      ff_close_iter: int,
                      # スケール除外
                      exclude_scalebar: bool,
                      sb_w_ratio: float,
                      sb_h_ratio: float,
                      sb_pad: int
                      ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    img_gray = read_image_from_bytes(file_bytes)

    # 右下スケール除外マスク
    ex_mask = make_bottom_right_exclude_mask(img_gray.shape[:2], sb_w_ratio, sb_h_ratio, sb_pad) \
        if exclude_scalebar else np.zeros_like(img_gray, dtype=np.uint8)

    # 母材マスク
    if use_specimen_mask:
        specimen_mask_u8 = compute_specimen_mask_floodfill(
            img_gray, tol=ff_tol, close_ksize=ff_close_ksize, close_iter=ff_close_iter
        )
    else:
        specimen_mask_u8 = np.ones_like(img_gray, dtype=np.uint8) * 255

    # スケール除外を母材マスクからも除外
    if exclude_scalebar:
        specimen_mask_u8 = cv2.bitwise_and(specimen_mask_u8, cv2.bitwise_not(ex_mask))

    # 前処理
    img_pre = apply_preprocess(img_gray, clahe_clip, gauss_ksize, gauss_sigma)

    # 二値
    bin_img = binarize(img_pre, threshold_method, manual_thresh, adaptive_block, adaptive_C)
    bin_clean = morph_cleanup(bin_img, open_ksize, open_iter, close_ksize, close_iter)

    # 母材内に限定
    bin_clean = cv2.bitwise_and(bin_clean, bin_clean, mask=specimen_mask_u8)

    debug_bh = np.zeros_like(img_gray, dtype=np.uint8)

    # 欠陥抽出
    if target_mode == "黒領域（欠陥）":
        if defect_mode_black == "二値の黒（内部穴）":
            defect_mask = extract_internal_black_defects(bin_clean, assume_material_is_largest=True)
        else:
            img_used = img_pre if bh_use_preprocessed else img_gray
            defect_mask, debug_bh = extract_dark_spots_blackhat(
                img_u8=img_used.astype(np.uint8),
                material_mask_u8=specimen_mask_u8,
                bh_ksize=bh_ksize,
                thresh_mode=bh_thresh_mode,
                manual_thr=bh_manual_thr,
                border_exclude_px=bh_border_exclude
            )

        defect_mask = morph_cleanup(defect_mask, defect_open_ksize, defect_open_iter, defect_close_ksize, defect_close_iter)

        # 母材内 + スケール除外
        defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=specimen_mask_u8)
        if exclude_scalebar:
            defect_mask = cv2.bitwise_and(defect_mask, cv2.bitwise_not(ex_mask))

        bin_target = defect_mask
    else:
        bin_target = bin_clean

    # ラベリング
    if use_watershed:
        label_img = split_touching_particles(bin_target, min_distance_px, h_max)
    else:
        label_img = label_by_connected_components(bin_target)

    # 計測
    df = extract_region_metrics(label_img, um_per_px, min_area_px, min_area_um2)
    if not df.empty:
        df.insert(0, "source", name)
        df.insert(1, "target_mode", target_mode)
        df.insert(2, "defect_mode", defect_mode_black if target_mode == "黒領域（欠陥）" else "-")

    overlay = overlay_labels(
        img_gray, label_img, df,
        show_id=show_id,
        fill_alpha=fill_alpha,
        draw_red_contour=draw_red_contour,
        contour_thickness=contour_thickness,
        contour_only=contour_only
    ) if not df.empty else cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    return df, img_gray, bin_clean, bin_target, debug_bh, specimen_mask_u8, ex_mask, overlay


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="欠陥（空隙）解析（Streamlit）", layout="wide")
st.title("射出成形材料断面の欠陥（空隙）解析")
st.caption("母材マスク（背景除外）＋スケール除外＋母材輪郭（緑）＋オーバーレイ拡大表示")

with st.sidebar:
    st.header("解析設定")
    st.caption("環境情報")
    st.write("Python:", sys.version.split()[0])
    st.write("matplotlib-fontja:", "OK" if FONTJA_OK else "NG（requirements.txt要確認）")
    st.markdown("---")

    st.subheader("母材マスク（背景黒樹脂の除外）")
    use_specimen_mask = st.toggle("母材マスクで背景を除外する（推奨）", value=True)
    ff_tol = st.slider("背景flood fill 許容差 tol", 0, 120, 25, 1)
    ff_close_ksize = st.slider("母材マスク Close カーネル（奇数推奨）", 5, 101, 21, 2)
    ff_close_iter = st.slider("母材マスク Close 回数", 0, 8, 2, 1)

    st.subheader("母材輪郭（緑）")
    show_specimen_contour = st.toggle("母材輪郭（緑）を表示（選択画像）", value=True)
    specimen_contour_thickness = st.slider("母材輪郭の太さ", 1, 10, 2, 1)

    st.subheader("右下スケール除外（ノイズ対策）")
    exclude_scalebar = st.toggle("右下スケール表示を除外する", value=True)
    sb_w_ratio = st.slider("除外幅（画像幅の割合）", 0.05, 0.80, 0.30, 0.01)
    sb_h_ratio = st.slider("除外高さ（画像高さの割合）", 0.05, 0.80, 0.22, 0.01)
    sb_pad = st.slider("除外領域 余白 [px]", 0, 80, 10, 1)
    show_ex_mask = st.toggle("除外マスクを表示（選択画像）", value=False)

    st.markdown("---")
    st.subheader("スケール設定")
    col_scale = st.columns(2)
    with col_scale[0]:
        um_per_px_input = st.number_input("μm / px（直接）", min_value=0.0, value=1.0, step=0.01, format="%.6f")
    with col_scale[1]:
        st.caption("またはスケールバーから算出")
        scalebar_um = st.number_input("スケールバー長 [μm]", min_value=0.0, value=0.0, step=1.0)
        scalebar_px = st.number_input("スケールバー長 [px]", min_value=0.0, value=0.0, step=1.0)

    um_per_px = compute_um_per_px(
        um_per_px_input,
        None if scalebar_um == 0 else scalebar_um,
        None if scalebar_px == 0 else scalebar_px
    )

    st.markdown("---")
    st.subheader("前処理（コントラスト）")
    clahe_clip = st.slider("CLAHE クリップ制限", 0.001, 0.080, 0.030, step=0.001)
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

    st.markdown("---")
    st.subheader("解析対象")
    target_mode = st.selectbox("どの領域を検出する？", ["黒領域（欠陥）", "白領域（材料/粒子）"], index=0)

    st.subheader("黒欠陥の検出方式（黒領域モード時）")
    defect_mode_black = st.selectbox("欠陥（黒）の検出方式", ["元画像の深い黒点（ブラックハット）", "二値の黒（内部穴）"], index=0)

    st.subheader("ブラックハット設定（深い黒点方式）")
    bh_use_preprocessed = st.toggle("前処理後画像（CLAHE+Gaussian）を使う", value=True)
    bh_ksize = st.slider("ブラックハット カーネルサイズ", 3, 61, 11, 2)
    bh_thresh_mode = st.selectbox("ブラックハットの二値化", ["otsu", "manual"], index=1)
    bh_manual_thr = st.slider("ブラックハット 手動しきい値", 1, 160, 25, 1)
    bh_border_exclude = st.slider("材料境界を除外する幅 [px]", 0, 50, 3, 1)

    st.subheader("欠陥マスク用 後処理（ノイズ除去）")
    defect_open_ksize = st.select_slider("欠陥Open カーネル", options=[0, 1, 2, 3, 4, 5, 6, 7], value=0)
    defect_open_iter = st.slider("欠陥Open 回数", 0, 5, 0, 1)
    defect_close_ksize = st.select_slider("欠陥Close カーネル", options=[0, 1, 2, 3, 4, 5, 6, 7], value=0)
    defect_close_iter = st.slider("欠陥Close 回数", 0, 5, 0, 1)

    st.subheader("分離（Watershed）")
    use_watershed = st.toggle("接触欠陥/粒子を分離する（Watershed）", value=False)
    min_distance_px = st.slider("局所極大の最小距離 [px]", 1, 50, 10, 1)
    h_max = st.slider("h-maxima（高いほど保守的）", 0.0, 10.0, 1.0, 0.1)

    st.markdown("---")
    st.subheader("フィルタ")
    min_area_px = st.slider("最小面積 [px²]（小ノイズ除去）", 0, 5000, 10, 5)
    min_area_um2 = st.number_input("最小面積 [μm²]（0=無効）", min_value=0.0, value=0.0, step=1.0)

    st.subheader("オーバーレイ")
    show_id = st.toggle("ID表示", value=True)
    draw_red_contour = st.toggle("欠陥輪郭を赤で描画", value=True)
    contour_thickness = st.slider("欠陥輪郭の太さ", 1, 10, 3, 1)
    contour_only = st.toggle("輪郭のみ（塗りつぶし無し）", value=True)
    fill_alpha = st.slider("塗りつぶし透明度", 0.0, 0.8, 0.25, 0.05)

    st.subheader("拡大表示")
    show_big_overlay = st.toggle("オーバーレイを大きく表示（選択した1枚）", value=True)
    big_overlay_height = st.slider("拡大表示の高さ [px]", 300, 1600, 900, 50)

st.markdown("### 入力ファイル")
uploaded_files = st.file_uploader(
    "単一または複数の画像ファイル、または ZIP（画像入り）を選択してください。",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "zip"],
    accept_multiple_files=True
)

if uploaded_files:
    # 入力展開
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

    # 実行
    results: List[pd.DataFrame] = []
    overlays: Dict[str, np.ndarray] = {}
    previews: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    summaries_list: List[Dict[str, float]] = []

    progress = st.progress(0)
    for idx, (name, bts) in enumerate(to_process, start=1):
        try:
            df, img_gray, bin_clean, bin_target, debug_bh, specimen_mask_u8, ex_mask_u8, overlay_img = process_one_image(
                name, bts, um_per_px,
                method, manual_thresh, adaptive_block, adaptive_C,
                clahe_clip, gauss_ksize, gauss_sigma,
                open_ksize, open_iter, close_ksize, close_iter,
                defect_open_ksize, defect_open_iter, defect_close_ksize, defect_close_iter,
                target_mode,
                defect_mode_black,
                bh_use_preprocessed, bh_ksize, bh_thresh_mode, bh_manual_thr, bh_border_exclude,
                use_watershed, min_distance_px, h_max,
                min_area_px, min_area_um2,
                show_id, fill_alpha, draw_red_contour, contour_thickness, contour_only,
                use_specimen_mask, ff_tol, ff_close_ksize, ff_close_iter,
                exclude_scalebar, sb_w_ratio, sb_h_ratio, sb_pad
            )

            overlays[name] = overlay_img
            previews[name] = (img_gray, bin_clean, bin_target, debug_bh, specimen_mask_u8, ex_mask_u8)

            if not df.empty:
                results.append(df)

            defect_mask_for_ratio = bin_target if target_mode == "黒領域（欠陥）" else np.zeros_like(bin_target)
            stats = compute_area_stats_A(specimen_mask_u8, defect_mask_for_ratio, um_per_px)
            stats.update({
                "source": name,
                "target_mode": target_mode,
                "defect_mode": defect_mode_black if target_mode == "黒領域（欠陥）" else "-"
            })
            summaries_list.append(stats)

        except Exception as e:
            st.error(f"【{name}】の解析でエラー：{e}")

        progress.progress(int(100 * idx / max(1, len(to_process))))

    df_all = pd.concat(results, ignore_index=True) if len(results) > 0 else pd.DataFrame()
    df_sum = pd.DataFrame(summaries_list) if len(summaries_list) > 0 else pd.DataFrame()

    # プレビュー：選択した1枚だけ表示
    st.markdown("### 可視化プレビュー（選択した1枚）")
    names = sorted(list(previews.keys()))
    selected_name = st.selectbox("表示する画像を選択してください", names, index=0)

    img_gray, bin_clean, bin_target, debug_bh, specimen_mask_u8, ex_mask_u8 = previews[selected_name]
    show_blackhat = (target_mode == "黒領域（欠陥）" and defect_mode_black == "元画像の深い黒点（ブラックハット）")

    if show_specimen_contour:
        outline_img = draw_mask_contour_on_gray(img_gray, specimen_mask_u8, (0, 255, 0), specimen_contour_thickness)
    else:
        outline_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    if show_blackhat:
        cols = st.columns(5)
        with cols[0]:
            st.image(cv2.cvtColor(outline_img, cv2.COLOR_BGR2RGB), caption="元画像 + 母材輪郭（緑）", use_container_width=True)
        with cols[1]:
            st.image(bin_clean, caption="二値（母材内）", use_container_width=True)
        with cols[2]:
            st.image(debug_bh, caption="ブラックハット（母材内）", use_container_width=True)
        with cols[3]:
            st.image(bin_target, caption="欠陥マスク（母材内）", use_container_width=True)
        with cols[4]:
            st.image(cv2.cvtColor(overlays[selected_name], cv2.COLOR_BGR2RGB), caption="オーバーレイ（赤）", use_container_width=True)
    else:
        cols = st.columns(4)
        with cols[0]:
            st.image(cv2.cvtColor(outline_img, cv2.COLOR_BGR2RGB), caption="元画像 + 母材輪郭（緑）", use_container_width=True)
        with cols[1]:
            st.image(bin_clean, caption="二値（母材内）", use_container_width=True)
        with cols[2]:
            st.image(bin_target, caption="対象マスク", use_container_width=True)
        with cols[3]:
            st.image(cv2.cvtColor(overlays[selected_name], cv2.COLOR_BGR2RGB), caption="オーバーレイ（赤）", use_container_width=True)

    if show_ex_mask and exclude_scalebar:
        st.image(ex_mask_u8, caption="右下スケール除外マスク（白=除外）", use_container_width=True)

    if show_big_overlay:
        st.markdown("#### 最終抽出結果（オーバーレイ）拡大表示")
        big = resize_to_height(overlays[selected_name], big_overlay_height)
        st.image(cv2.cvtColor(big, cv2.COLOR_BGR2RGB),
                 caption=f"オーバーレイ（拡大：高さ {big_overlay_height}px）",
                 use_container_width=True)

    # 欠陥率サマリー
    if not df_sum.empty:
        st.markdown("### 欠陥率サマリー（A案）")
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
        st.download_button("📥 欠陥率サマリーCSVをダウンロード",
                           data=df_sum_disp.to_csv(index=False).encode("utf-8-sig"),
                           file_name="defect_area_ratio_summary_A.csv",
                           mime="text/csv")

    # 欠陥特性CSV
    if not df_all.empty:
        st.markdown("### エクスポート（欠陥 特性CSV）")
        st.download_button("📥 欠陥 特性CSVをダウンロード",
                           data=df_all.to_csv(index=False).encode("utf-8-sig"),
                           file_name="defect_metrics.csv",
                           mime="text/csv")

        st.markdown("### 統計可視化（形状指標）")
        plot_distributions(df_all, ["equiv_diam_um", "aspect_ratio", "circularity"], group="source")

    # オーバーレイZIP
    with tempfile.TemporaryDirectory() as tmpd:
        zip_path = os.path.join(tmpd, "overlays.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, img in overlays.items():
                out_name = os.path.splitext(os.path.basename(fname))[0] + "_overlay.png"
                _, buf = cv2.imencode(".png", img)
                zf.writestr(out_name, buf.tobytes())
        with open(zip_path, "rb") as fz:
            st.download_button("🖼️ 注釈画像（ZIP）をダウンロード",
                               data=fz.read(),
                               file_name="overlays.zip",
                               mime="application/zip")

else:
    st.info("左下の **[Browse files]** から画像または ZIP を選択してください。")
