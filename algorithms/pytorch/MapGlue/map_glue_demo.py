import cv2
import numpy as np
import torch
from pathlib import Path


def _make_chessboard(height: int, width: int, square: int = 32) -> np.ndarray:
    yy, xx = np.indices((height, width))
    board = ((yy // square) + (xx // square)) % 2
    board = (board * 255).astype(np.uint8)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "weights" / "fastmapglue_model.pt"
    image0_path = base_dir / "assets" / "map-visible" / "L2.png"
    image1_path = base_dir / "assets" / "map-visible" / "R2.png"
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = torch.jit.load(str(model_path))
    model.eval()

    image0_bgr = cv2.imread(str(image0_path), cv2.IMREAD_COLOR)
    image1_bgr = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
    if image0_bgr is None:
        raise FileNotFoundError(str(image0_path))
    if image1_bgr is None:
        raise FileNotFoundError(str(image1_path))

    image0_rgb = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2RGB)
    image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)

    image0 = torch.from_numpy(image0_rgb)
    image1 = torch.from_numpy(image1_rgb)
    num_keypoints = torch.tensor(2048)

    with torch.inference_mode():
        points_tensor = model(image0, image1, num_keypoints)

    points = points_tensor.detach().cpu().numpy().astype(np.float32)
    points0 = points[:, :2]
    points1 = points[:, 2:4]

    if points0.shape[0] < 4:
        raise RuntimeError(f"Not enough matches: {points0.shape[0]}")

    H, mask = cv2.findHomography(points0, points1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None or mask is None:
        raise RuntimeError("cv2.findHomography failed")

    inlier_mask = mask.ravel().astype(bool)
    inlier_indices = np.where(inlier_mask)[0].tolist()
    if len(inlier_indices) < 4:
        raise RuntimeError(f"Not enough inliers after RANSAC: {len(inlier_indices)}")

    keypoints0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in points0]
    keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in points1]
    inlier_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in inlier_indices]
    matches_vis = cv2.drawMatches(
        image0_bgr,
        keypoints0,
        image1_bgr,
        keypoints1,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out_dir / "matches_ransac.png"), matches_vis)

    chessboard = _make_chessboard(image0_bgr.shape[0], image0_bgr.shape[1], square=32)
    warped_chessboard = cv2.warpPerspective(chessboard, H, (image1_bgr.shape[1], image1_bgr.shape[0]))
    cv2.imwrite(str(out_dir / "chessboard_warped.png"), warped_chessboard)

    overlay = cv2.addWeighted(image1_bgr, 0.7, warped_chessboard, 0.3, 0.0)
    cv2.imwrite(str(out_dir / "chessboard_overlay.png"), overlay)


if __name__ == "__main__":
    main()
