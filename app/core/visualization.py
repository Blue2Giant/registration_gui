from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualizationOutputs:
    matches_vis_path: str
    checkerboard_path: str
    warped_path: str


def _ensure_color(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def draw_matches_side_by_side(
    fixed_bgr: np.ndarray,
    moving_bgr: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    inlier_mask: np.ndarray,
) -> np.ndarray:
    fixed_bgr = _ensure_color(fixed_bgr)
    moving_bgr = _ensure_color(moving_bgr)

    h = max(fixed_bgr.shape[0], moving_bgr.shape[0])
    w = fixed_bgr.shape[1] + moving_bgr.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: fixed_bgr.shape[0], : fixed_bgr.shape[1]] = fixed_bgr
    canvas[: moving_bgr.shape[0], fixed_bgr.shape[1] : fixed_bgr.shape[1] + moving_bgr.shape[1]] = moving_bgr

    offset_x = fixed_bgr.shape[1]
    p1i = np.round(p1).astype(int)
    p2i = np.round(p2).astype(int)
    inlier_mask = inlier_mask.astype(bool)

    # Draw lines
    for i in range(p1i.shape[0]):
        a = (int(p1i[i, 0]), int(p1i[i, 1]))
        b = (int(p2i[i, 0] + offset_x), int(p2i[i, 1]))
        color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255) # Green for inliers, Red for outliers
        
        # Only draw outliers if there are few points, otherwise it gets too messy
        if not inlier_mask[i] and p1i.shape[0] > 100:
            continue
            
        cv2.line(canvas, a, b, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, a, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, b, 3, color, -1, lineType=cv2.LINE_AA)

    return canvas


def checkerboard_fusion(
    fixed_bgr: np.ndarray,
    moving_bgr: np.ndarray,
    H_3x3: np.ndarray,
    tile_px: int,
) -> np.ndarray:
    fixed_bgr = _ensure_color(fixed_bgr)
    moving_bgr = _ensure_color(moving_bgr)
    h, w = fixed_bgr.shape[0], fixed_bgr.shape[1]

    warped = cv2.warpPerspective(moving_bgr, H_3x3.astype(np.float64), (w, h), flags=cv2.INTER_LINEAR)

    yy, xx = np.indices((h, w))
    mask = ((xx // int(tile_px) + yy // int(tile_px)) % 2 == 0)
    out = fixed_bgr.copy()
    out[mask] = warped[mask]
    return out


def save_visualizations(
    out_dir: str,
    fixed_img_path: str,
    moving_img_path: str,
    p1: np.ndarray,
    p2: np.ndarray,
    inlier_mask: np.ndarray,
    H_3x3: np.ndarray,
    tile_px: int,
) -> VisualizationOutputs:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixed = cv2.imread(fixed_img_path, cv2.IMREAD_COLOR)
    moving = cv2.imread(moving_img_path, cv2.IMREAD_COLOR)
    if fixed is None or moving is None:
        raise ValueError("failed to read images")

    mv = draw_matches_side_by_side(fixed, moving, p1, p2, inlier_mask)
    mv_path = str((out / "matches_vis.jpg").resolve())
    cv2.imwrite(mv_path, mv)

    cb = checkerboard_fusion(fixed, moving, H_3x3, tile_px)
    cb_path = str((out / "checkerboard.jpg").resolve())
    cv2.imwrite(cb_path, cb)

    # Save full warped image for layer switching
    h, w = fixed.shape[0], fixed.shape[1]
    warped = cv2.warpPerspective(moving, H_3x3.astype(np.float64), (w, h), flags=cv2.INTER_LINEAR)
    warped_path = str((out / "warped.jpg").resolve())
    cv2.imwrite(warped_path, warped)

    return VisualizationOutputs(
        matches_vis_path=mv_path, 
        checkerboard_path=cb_path,
        warped_path=warped_path
    )
