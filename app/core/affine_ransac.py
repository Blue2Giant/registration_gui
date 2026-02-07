from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AffineEstimation:
    H_3x3: np.ndarray
    inlier_mask: np.ndarray
    rmse: float


def estimate_affine_3x3_ransac(
    points1: np.ndarray,
    points2: np.ndarray,
    thresh_px: float,
    ransac_max_iters: int = 5000,
    ransac_confidence: float = 0.995,
    ransac_refine_iters: int = 10,
) -> AffineEstimation:
    p1 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float32).reshape(-1, 2)
    if p1.shape != p2.shape or p1.ndim != 2 or p1.shape[1] != 2:
        raise ValueError("points shape invalid")

    finite = np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
    p1 = p1[finite]
    p2 = p2[finite]
    if p1.shape[0] < 3:
        raise ValueError("not enough matches")

    p1c = np.ascontiguousarray(p1, dtype=np.float32)
    p2c = np.ascontiguousarray(p2, dtype=np.float32)

    max_iters = int(ransac_max_iters)
    if max_iters < 1:
        raise ValueError("ransac_max_iters must be >= 1")
    confidence = float(ransac_confidence)
    if not (0.0 < confidence < 1.0):
        raise ValueError("ransac_confidence must be in (0, 1)")
    refine_iters = int(ransac_refine_iters)
    if refine_iters < 0:
        raise ValueError("ransac_refine_iters must be >= 0")

    M = None
    inliers = None
    try:
        M, inliers = cv2.estimateAffine2D(
            p1c,
            p2c,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(thresh_px),
            maxIters=max_iters,
            confidence=confidence,
            refineIters=refine_iters,
        )
    except cv2.error:
        p1r = p1c.reshape(-1, 1, 2)
        p2r = p2c.reshape(-1, 1, 2)
        try:
            M, inliers = cv2.estimateAffine2D(
                p1r,
                p2r,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(thresh_px),
                maxIters=max_iters,
                confidence=confidence,
                refineIters=refine_iters,
            )
        except cv2.error:
            M, inliers = cv2.estimateAffinePartial2D(
                p1c,
                p2c,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(thresh_px),
                maxIters=max_iters,
                confidence=confidence,
                refineIters=refine_iters,
            )
    
    if M is None:
        ones = np.ones((p1c.shape[0], 1), dtype=np.float32)
        A = np.concatenate([p1c, ones], axis=1)
        bx = p2c[:, 0:1]
        by = p2c[:, 1:2]
        mx, *_ = np.linalg.lstsq(A, bx, rcond=None)
        my, *_ = np.linalg.lstsq(A, by, rcond=None)
        M = np.vstack([mx.reshape(1, 3), my.reshape(1, 3)]).astype(np.float32)
        inliers = np.ones((p1c.shape[0], 1), dtype=np.uint8)

    if M is None:
         raise ValueError("estimateAffine2D failed completely")

    inlier_mask = (inliers.reshape(-1).astype(np.uint8) > 0)

    H = np.eye(3, dtype=np.float64)
    H[0:2, 0:3] = M.astype(np.float64)

    p1h = np.concatenate([p1c.astype(np.float64), np.ones((p1c.shape[0], 1), dtype=np.float64)], axis=1)
    pred = (H @ p1h.T).T
    pred = pred[:, 0:2] / pred[:, 2:3]
    err = np.linalg.norm(pred - p2c.astype(np.float64), axis=1)
    
    if inlier_mask.any():
        rmse = float(np.sqrt(np.mean(np.square(err[inlier_mask]))))
    else:
        rmse = float(np.sqrt(np.mean(np.square(err))))

    return AffineEstimation(H_3x3=H, inlier_mask=inlier_mask, rmse=rmse)
