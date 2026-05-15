from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AffineEstimation:
    H_3x3: np.ndarray
    inlier_mask: np.ndarray
    rmse: float


def _filter_match_pairs(points1: np.ndarray, points2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float32).reshape(-1, 2)
    if p1.shape != p2.shape or p1.ndim != 2 or p1.shape[1] != 2:
        raise ValueError("points shape invalid")

    finite = np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
    if not finite.any():
        return p1[:0], p2[:0], np.zeros((p1.shape[0],), dtype=bool)

    p1f = p1[finite]
    p2f = p2[finite]
    pairs = np.concatenate([p1f, p2f], axis=1)

    uniq_pairs, uniq_idx, inverse = np.unique(pairs, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(uniq_idx)
    uniq_pairs = uniq_pairs[order]

    old_to_new = np.empty(order.shape[0], dtype=np.int32)
    old_to_new[order] = np.arange(order.shape[0], dtype=np.int32)
    inverse = old_to_new[inverse]

    p1u = uniq_pairs[:, 0:2].astype(np.float32, copy=False)
    p2u = uniq_pairs[:, 2:4].astype(np.float32, copy=False)

    expand_index = np.full((p1.shape[0],), -1, dtype=np.int32)
    expand_index[np.flatnonzero(finite)] = inverse
    return p1u, p2u, expand_index


def _expand_inlier_mask(compact_mask: np.ndarray, expand_index: np.ndarray) -> np.ndarray:
    out = np.zeros((expand_index.shape[0],), dtype=bool)
    valid = expand_index >= 0
    if valid.any():
        out[valid] = compact_mask[expand_index[valid]]
    return out


def estimate_affine_3x3_ransac(
    points1: np.ndarray,
    points2: np.ndarray,
    thresh_px: float = 3.0,
    ransac_max_iters: int = 2000,
    ransac_confidence: float = 0.99,
    ransac_refine_iters: int = 10,
) -> AffineEstimation:
    p1_all = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    p2_all = np.asarray(points2, dtype=np.float32).reshape(-1, 2)
    p1, p2, expand_index = _filter_match_pairs(p1_all, p2_all)
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

    inlier_mask_compact = (inliers.reshape(-1).astype(np.uint8) > 0)
    inlier_mask = _expand_inlier_mask(inlier_mask_compact, expand_index)

    H = np.eye(3, dtype=np.float64)
    H[0:2, 0:3] = M.astype(np.float64)

    p1h = np.concatenate([p1_all.astype(np.float64), np.ones((p1_all.shape[0], 1), dtype=np.float64)], axis=1)
    pred = (H @ p1h.T).T
    pred = pred[:, 0:2] / pred[:, 2:3]
    err = np.linalg.norm(pred - p2_all.astype(np.float64), axis=1)
    
    if inlier_mask.any():
        rmse = float(np.sqrt(np.mean(np.square(err[inlier_mask]))))
    else:
        rmse = float(np.sqrt(np.mean(np.square(err))))

    return AffineEstimation(H_3x3=H, inlier_mask=inlier_mask, rmse=rmse)
