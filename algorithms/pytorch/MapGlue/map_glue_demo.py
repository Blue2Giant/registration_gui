import argparse
from pathlib import Path

import cv2
import numpy as np


def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_matches_txt(path: str, p_fixed: np.ndarray, p_moving: np.ndarray) -> None:
    if p_fixed.shape != p_moving.shape or p_fixed.ndim != 2 or p_fixed.shape[1] != 2:
        raise ValueError("points shape invalid")
    out = np.hstack([p_fixed, p_moving]).astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, out, fmt="%.4f")


def _create_feature_detector() -> tuple[cv2.Feature2D, str]:
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(), "SIFT"
    return cv2.ORB_create(nfeatures=4000), "ORB"


def _generate_matches_opencv(fixed_path: str, moving_path: str, max_matches: int = 2000) -> tuple[np.ndarray, np.ndarray, str]:
    img1 = cv2.imread(str(fixed_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(moving_path), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise ValueError("failed to read images")
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


def _make_checkerboard(img_ref: np.ndarray, img_warped: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = img_ref.shape[:2]
    h_warp, w_warp = img_warped.shape[:2]
    hc, wc = min(h, h_warp), min(w, w_warp)
    img_ref = img_ref[:hc, :wc]
    img_warped = img_warped[:hc, :wc]
    out = np.zeros_like(img_ref)
    for y in range(0, hc, tile_size):
        for x in range(0, wc, tile_size):
            y_end = min(y + tile_size, hc)
            x_end = min(x + tile_size, wc)
            iy = y // tile_size
            ix = x // tile_size
            if (iy + ix) % 2 == 0:
                out[y:y_end, x:x_end] = img_ref[y:y_end, x:x_end]
            else:
                out[y:y_end, x:x_end] = img_warped[y:y_end, x:x_end]
    return out


def run_mapglue(
    fixed_path: str,
    moving_path: str,
    matches_out: str,
    weights_path: str,
    num_keypoints: int = 2048,
    device: str = "cpu",
) -> int:
    try:
        import torch

        model_path = Path(weights_path)
        if not model_path.exists():
            raise FileNotFoundError(str(model_path))

        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()

        fixed_rgb = _read_rgb(fixed_path)
        moving_rgb = _read_rgb(moving_path)

        fixed = torch.from_numpy(fixed_rgb)
        moving = torch.from_numpy(moving_rgb)
        nk = torch.tensor(int(num_keypoints))

        with torch.inference_mode():
            points_tensor = model(fixed, moving, nk)

        points = points_tensor.detach().cpu().numpy().astype(np.float32)
        if points.ndim != 2 or points.shape[1] < 4:
            raise RuntimeError(f"Unexpected output shape: {tuple(points.shape)}")

        p_fixed = points[:, 0:2]
        p_moving = points[:, 2:4]

        if p_fixed.shape[0] < 4:
            raise RuntimeError(f"Not enough matches: {p_fixed.shape[0]}")

        _save_matches_txt(matches_out, p_fixed, p_moving)
        print(f"MapGlue matches saved: {matches_out}  pairs={p_fixed.shape[0]}")
        return int(p_fixed.shape[0])
    except Exception as e:
        print(f"MapGlue unavailable ({type(e).__name__}: {e}). Falling back to OpenCV matching...")
        p1, p2, method = _generate_matches_opencv(fixed_path, moving_path, max_matches=int(num_keypoints))
        _save_matches_txt(matches_out, p1, p2)
        print(f"OpenCV({method}) matches saved: {matches_out}  pairs={p1.shape[0]}")
        return int(p1.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixed", help="Fixed/target image path")
    parser.add_argument("moving", help="Moving/source image path")
    parser.add_argument("matches_out", help="Output matches txt path (x1 y1 x2 y2 per line)")
    parser.add_argument("--weights", default="", help="TorchScript weights path (.pt)")
    parser.add_argument("--num_keypoints", type=int, default=2048)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--save_chessboard", action="store_true")
    parser.add_argument("--chessboard_tile", type=int, default=64)
    parser.add_argument("--chessboard_out", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    weights = args.weights.strip() or str((base_dir / "weights" / "fastmapglue_model.pt").resolve())

    run_mapglue(
        fixed_path=args.fixed,
        moving_path=args.moving,
        matches_out=args.matches_out,
        weights_path=weights,
        num_keypoints=args.num_keypoints,
        device=args.device,
    )

    if args.save_chessboard:
        matches_path = Path(args.matches_out)
        if not matches_path.exists():
            print(f"Chessboard skipped: matches file not found: {matches_path}")
            return
        arr = np.loadtxt(str(matches_path), dtype=np.float32)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 4 or arr.shape[0] < 4:
            print("Chessboard skipped: not enough matches")
            return
        mkpts0 = arr[:, 0:2]
        mkpts1 = arr[:, 2:4]
        h_pred, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 3.0)
        if h_pred is None:
            print("Chessboard skipped: findHomography failed")
            return
        img0 = cv2.imread(str(args.fixed), cv2.IMREAD_COLOR)
        img1 = cv2.imread(str(args.moving), cv2.IMREAD_COLOR)
        if img0 is None or img1 is None:
            print("Chessboard skipped: failed to read images")
            return
        h_ref, w_ref = img0.shape[:2]
        warped = cv2.warpPerspective(img1, h_pred, (w_ref, h_ref))
        chess = _make_checkerboard(img_ref=img0, img_warped=warped, tile_size=args.chessboard_tile)
        out_path = Path(args.chessboard_out) if args.chessboard_out.strip() else matches_path.with_name("chessboard.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), chess)
        print(f"Chessboard saved: {out_path}")


if __name__ == "__main__":
    main()
