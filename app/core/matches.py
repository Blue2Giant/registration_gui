from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchesResult:
    points1: np.ndarray
    points2: np.ndarray
    path: str
    source: str


def load_matches_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        raise ValueError("matches.txt is empty")
    rows = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
            
    if not rows:
        raise ValueError("matches.txt has no valid rows")
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("matches array shape invalid")
    p1 = arr[:, 0:2]
    p2 = arr[:, 2:4]
    return p1, p2


def save_matches_txt(path: str, points1: np.ndarray, points2: np.ndarray) -> None:
    if points1.shape != points2.shape or points1.shape[1] != 2:
        raise ValueError("points shape invalid")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for a, b in zip(points1.tolist(), points2.tolist()):
        lines.append(f"{a[0]:.4f} {a[1]:.4f} {b[0]:.4f} {b[1]:.4f}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_feature_detector() -> Tuple[cv2.Feature2D, str]:
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(), "SIFT"
    return cv2.ORB_create(nfeatures=4000), "ORB"


def generate_matches_opencv(fixed_img_path: str, moving_img_path: str, max_matches: int = 2000) -> Tuple[np.ndarray, np.ndarray, str]:
    img1 = cv2.imread(fixed_img_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(moving_img_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise ValueError("failed to read images for feature matching")
    detector, name = _create_feature_detector()
    k1, d1 = detector.detectAndCompute(img1, None)
    k2, d2 = detector.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        raise ValueError("not enough features")

    if name == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn = matcher.knnMatch(d1, d2, k=2)
        good = []
        for m, n in knn:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        good = sorted(good, key=lambda x: x.distance)[:max_matches]
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good = matcher.match(d1, d2)
        good = sorted(good, key=lambda x: x.distance)[:max_matches]

    if len(good) < 4:
        raise ValueError("not enough matches")

    p1 = np.asarray([k1[m.queryIdx].pt for m in good], dtype=np.float32)
    p2 = np.asarray([k2[m.trainIdx].pt for m in good], dtype=np.float32)
    return p1, p2, name


def resolve_matches(
    desired_matches_path: str,
    exe_fallback_paths: list[str],
    fixed_img_path: str,
    moving_img_path: str,
    generate_if_missing: bool,
) -> MatchesResult:
    candidates = [desired_matches_path] + exe_fallback_paths
    for c in candidates:
        if c and Path(c).exists():
            try:
                p1, p2 = load_matches_txt(c)
                return MatchesResult(points1=p1, points2=p2, path=str(Path(c).resolve()), source="file")
            except ValueError:
                continue

    if not generate_if_missing:
        raise FileNotFoundError("matches not found and generation disabled")

    p1, p2, method = generate_matches_opencv(fixed_img_path, moving_img_path)
    save_matches_txt(desired_matches_path, p1, p2)
    return MatchesResult(points1=p1, points2=p2, path=str(Path(desired_matches_path).resolve()), source=f"opencv:{method}")
