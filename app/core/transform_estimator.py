from __future__ import annotations

import numpy as np

from .fsc import estimate_fsc_3x3
from .transform_ransac import TransformEstimation, estimate_transform_3x3_ransac


def estimate_transform_3x3(points1: np.ndarray, points2: np.ndarray, method: str, thresh_px: float) -> TransformEstimation:
    m = (method or "").strip().lower()
    if m in ("affine", "homography"):
        return estimate_transform_3x3_ransac(points1, points2, m, thresh_px)
    if m == "fsc-affine":
        return estimate_fsc_3x3(points1, points2, change_form="affine", error_t=float(thresh_px))
    if m == "fsc-perspective":
        return estimate_fsc_3x3(points1, points2, change_form="perspective", error_t=float(thresh_px))
    raise ValueError("unknown transform method")

