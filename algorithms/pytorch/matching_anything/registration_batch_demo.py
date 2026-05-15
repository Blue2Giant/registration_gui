"""
LoFTR 批量配准评估脚本
"""
import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.config.default import get_cfg_defaults
from src.lightning.lightning_loftr import PL_LoFTR
from tools_utils.plot import plot_matches


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="LoFTR 批量配准评估（每对耗时 + RMSE + 匹配点数 + 成功数）",
    )
    parser.add_argument("main_cfg_path", type=str, help="LoFTR 主配置文件路径（yaml）")
    parser.add_argument("--ckpt_path", type=str, default="", help="LoFTR 权重 ckpt 路径")
    parser.add_argument("--method", type=str, default="loftr@-@ransac_affine", help="方法名（前半段会写入 config.METHOD）")
    parser.add_argument("--thr", type=float, default=0.1, help="coarse-level matching threshold，会写入 config.LOFTR.MATCH_COARSE.THR")
    parser.add_argument("--pairs_dir", type=str, required=True, help="包含 pair*_1.*, pair*_2.* 以及 pair*.txt 或 gt_*.txt 的目录")
    parser.add_argument("--output_dir", type=str, default="demo_output_pairs", help="输出结果目录")
    parser.add_argument("--imgresize", type=int, default=None, help="把两张图 resize 成正方形 imgresize×imgresize 后再送进网络和评估")
    parser.add_argument("--no_cuda", action="store_true", help="强制使用 CPU（调试用）")
    parser.add_argument("--plot_matches", action="store_true", help="是否保存匹配可视化图")
    parser.add_argument("--plot_matches_alpha", type=float, default=0.2, help="plot_matches 的 alpha")
    parser.add_argument("--plot_matches_color", type=str, default="error", choices=["green", "error", "conf"], help="plot_matches 中的颜色模式")
    parser.add_argument("--save_chessboard", action="store_true", help="是否基于单应性变换保存棋盘可视化图")
    parser.add_argument("--chessboard_tile", type=int, default=64, help="棋盘格子大小（像素）")
    parser.add_argument("--num_samples", type=int, default=1000, help="计算 RMSE 时在第二张图上随机采样的点数")
    parser.add_argument("--gt_direction", type=str, default="2to1", choices=["2to1", "1to2"], help="GT 矩阵方向：2to1 或 1to2")
    parser.add_argument("--success_match_threshold", type=int, default=4, help="当匹配点数大于该阈值时，认为该 pair 匹配成功")
    parser.add_argument("--max_pairs", type=int, default=None, help="最多只处理前 N 组 pair")
    return parser.parse_args()


def load_image_as_tensor(path: str, resize: Optional[int] = None, in_channels: int = 1) -> torch.Tensor:
    if in_channels == 1:
        img = Image.open(path).convert("L")
    else:
        img = Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)[None]


def tensor_to_vis_image(tensor: torch.Tensor, in_channels: int) -> np.ndarray:
    t = tensor[0].detach().cpu()
    if in_channels == 1:
        arr = t[0].numpy()
    else:
        arr = t.permute(1, 2, 0).numpy()
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def make_checkerboard(img_ref: np.ndarray, img_warped: np.ndarray, tile_size: int) -> np.ndarray:
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
            if ((y // tile_size) + (x // tile_size)) % 2 == 0:
                out[y:y_end, x:x_end] = img_ref[y:y_end, x:x_end]
            else:
                out[y:y_end, x:x_end] = img_warped[y:y_end, x:x_end]
    return out


def load_gt_matrix_3x3(gt_path: Path) -> np.ndarray:
    arr = np.loadtxt(str(gt_path), dtype=np.float64)
    arr = np.asarray(arr)
    if arr.size == 6:
        arr = arr.reshape(2, 3)
    if arr.shape == (2, 3):
        return np.vstack([arr, np.array([0.0, 0.0, 1.0], dtype=np.float64)])
    if arr.shape == (3, 3):
        return arr.astype(np.float64)
    raise ValueError(f"无法从 {gt_path} 解析 2x3 或 3x3 矩阵，实际 shape={arr.shape}")


def adapt_gt_to_resized(
    h_gt_raw: np.ndarray,
    gt_direction: str,
    size0_orig: Tuple[int, int],
    size1_orig: Tuple[int, int],
    size0_new: Tuple[int, int],
    size1_new: Tuple[int, int],
) -> np.ndarray:
    h = h_gt_raw.copy()
    if gt_direction == "1to2":
        h = np.linalg.inv(h)

    w0_orig, h0_orig = size0_orig
    w1_orig, h1_orig = size1_orig
    w0_new, h0_new = size0_new
    w1_new, h1_new = size1_new

    s0 = np.array(
        [[w0_new / float(w0_orig), 0.0, 0.0], [0.0, h0_new / float(h0_orig), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    s1 = np.array(
        [[w1_new / float(w1_orig), 0.0, 0.0], [0.0, h1_new / float(h1_orig), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return s0 @ h @ np.linalg.inv(s1)


def compute_homography_rmse(
    h_pred: np.ndarray,
    h_gt: np.ndarray,
    width1: int,
    height1: int,
    num_samples: int = 1000,
) -> Optional[float]:
    if h_pred is None or h_gt is None or width1 <= 0 or height1 <= 0:
        return None
    xs = np.random.uniform(0, width1 - 1, size=(num_samples,))
    ys = np.random.uniform(0, height1 - 1, size=(num_samples,))
    pts_h = np.stack([xs, ys, np.ones_like(xs)], axis=0)
    pred_h = np.asarray(h_pred, dtype=np.float64) @ pts_h
    gt_h = np.asarray(h_gt, dtype=np.float64) @ pts_h
    pred_w = pred_h[2, :]
    gt_w = gt_h[2, :]
    valid = (np.abs(pred_w) > 1e-8) & (np.abs(gt_w) > 1e-8)
    if not np.any(valid):
        return None
    pred_xy = pred_h[:2, valid] / pred_w[valid]
    gt_xy = gt_h[:2, valid] / gt_w[valid]
    diff = pred_xy - gt_xy
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=0))))


def collect_pairs(pairs_dir: Path) -> List[Tuple[str, Path, Path, Path]]:
    pairs: List[Tuple[str, Path, Path, Path]] = []
    pattern = re.compile(r"pair(\d+)_1\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for p1 in pairs_dir.iterdir():
        if not p1.is_file():
            continue
        match = pattern.match(p1.name)
        if not match:
            continue
        idx = match.group(1)
        img1_path = None
        for ext in exts:
            candidate = pairs_dir / f"pair{idx}_2{ext}"
            if candidate.is_file():
                img1_path = candidate
                break
        if img1_path is None:
            print(f"[WARN] 找不到 pair{idx}_2.*，跳过该组")
            continue
        gt_path = pairs_dir / f"pair{idx}.txt"
        if not gt_path.is_file():
            gt_path = pairs_dir / f"gt_{idx}.txt"
        if not gt_path.is_file():
            print(f"[WARN] 找不到 pair{idx}.txt 或 gt_{idx}.txt，跳过该组")
            continue
        pairs.append((idx, p1, img1_path, gt_path))
    pairs.sort(key=lambda item: int(item[0]))
    return pairs


def save_pair_metrics_csv(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    csv_path = output_dir / "pair_metrics.csv"
    fieldnames = [
        "pair_id",
        "img0_path",
        "img1_path",
        "gt_path",
        "num_matches",
        "is_match_success",
        "num_inliers",
        "rmse",
        "elapsed_sec",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"[INFO] 每对结果已保存到: {csv_path}")


def save_summary_csv(output_dir: Path, summary: Dict[str, object]) -> None:
    csv_path = output_dir / "summary.csv"
    fieldnames = list(summary.keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)
    print(f"[INFO] 汇总结果已保存到: {csv_path}")


def process_single_pair(
    idx: str,
    img0_path: Path,
    img1_path: Path,
    gt_path: Path,
    args,
    matcher,
    device,
    in_channels: int,
    use_fp16: bool,
    output_dir: Path,
) -> Dict[str, object]:
    print(f"\n[PAIR {idx}] 开始处理")
    print(f"  img0 = {img0_path}")
    print(f"  img1 = {img1_path}")
    print(f"  gt   = {gt_path}")

    t0 = time.perf_counter()
    error_message = ""
    num_matches = 0
    num_inliers = None
    rmse = None

    try:
        with Image.open(str(img0_path)) as im0_orig:
            w0_orig, h0_orig = im0_orig.size
        with Image.open(str(img1_path)) as im1_orig:
            w1_orig, h1_orig = im1_orig.size

        img0_tensor = load_image_as_tensor(str(img0_path), resize=args.imgresize, in_channels=in_channels)
        img1_tensor = load_image_as_tensor(str(img1_path), resize=args.imgresize, in_channels=in_channels)
        img0_vis = tensor_to_vis_image(img0_tensor, in_channels)
        img1_vis = tensor_to_vis_image(img1_tensor, in_channels)

        h0_new, w0_new = img0_vis.shape[:2]
        h1_new, w1_new = img1_vis.shape[:2]

        img0_tensor = img0_tensor.to(device)
        img1_tensor = img1_tensor.to(device)
        _, _, h0, w0 = img0_tensor.shape
        _, _, h1, w1 = img1_tensor.shape

        batch = {
            "image0_rgb": img0_tensor,
            "image1_rgb": img1_tensor,
            "hw0_i": torch.tensor([[h0, w0]], device=device),
            "hw1_i": torch.tensor([[h1, w1]], device=device),
        }

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=use_fp16):
                    matcher(batch)
            else:
                matcher(batch)

        mkpts0 = batch["mkpts0_f"].detach().cpu().numpy()
        mkpts1 = batch["mkpts1_f"].detach().cpu().numpy()
        mconf = batch["mconf"].detach().cpu().numpy()
        num_matches = int(mkpts0.shape[0])
        print(f"[PAIR {idx}] 匹配点数量: {num_matches}")

        h_pred = None
        if num_matches >= 4:
            h_pred, inliers = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 3.0)
            if h_pred is None:
                print(f"[PAIR {idx}] [WARN] findHomography 失败")
            else:
                num_inliers = int(np.sum(inliers)) if inliers is not None else None
                print(f"[PAIR {idx}] RANSAC 内点数: {num_inliers}")
        else:
            print(f"[PAIR {idx}] [WARN] 匹配点少于 4 个，无法估计单应性")

        if args.plot_matches and num_matches > 0:
            out_match = output_dir / f"pair{idx}_matches.png"
            plot_matches(
                img0_vis,
                img1_vis,
                mkpts0,
                mkpts1,
                mconf,
                vertical=False,
                draw_match_type="corres",
                alpha=args.plot_matches_alpha,
                save_path=out_match,
                inverse=False,
                match_error=None,
                error_thr=5.0,
                color_type=args.plot_matches_color,
            )
            print(f"[PAIR {idx}] 匹配可视化已保存到: {out_match}")

        if args.save_chessboard and h_pred is not None:
            warped_img1 = cv2.warpPerspective(img1_vis, h_pred, (w0_new, h0_new))
            chessboard_img = make_checkerboard(img_ref=img0_vis, img_warped=warped_img1, tile_size=args.chessboard_tile)
            out_chess = output_dir / f"pair{idx}_chessboard.png"
            Image.fromarray(chessboard_img).save(out_chess)
            print(f"[PAIR {idx}] 棋盘可视化已保存到: {out_chess}")

        if h_pred is not None:
            h_gt_raw = load_gt_matrix_3x3(gt_path)
            h_gt_new = adapt_gt_to_resized(
                h_gt_raw=h_gt_raw,
                gt_direction=args.gt_direction,
                size0_orig=(w0_orig, h0_orig),
                size1_orig=(w1_orig, h1_orig),
                size0_new=(w0_new, h0_new),
                size1_new=(w1_new, h1_new),
            )
            rmse = compute_homography_rmse(
                h_pred=h_pred,
                h_gt=h_gt_new,
                width1=w1_new,
                height1=h1_new,
                num_samples=args.num_samples,
            )
            if rmse is not None:
                print(f"[PAIR {idx}] RMSE = {rmse:.4f}")
            else:
                print(f"[PAIR {idx}] [WARN] RMSE 计算失败")
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"[PAIR {idx}] [WARN] 处理失败: {error_message}")

    elapsed = time.perf_counter() - t0
    is_match_success = num_matches > int(args.success_match_threshold)
    print(f"[PAIR {idx}] 匹配成功: {'是' if is_match_success else '否'}")
    print(f"[PAIR {idx}] 耗时: {elapsed:.4f}s")

    return {
        "pair_id": idx,
        "img0_path": str(img0_path),
        "img1_path": str(img1_path),
        "gt_path": str(gt_path),
        "num_matches": int(num_matches),
        "is_match_success": int(is_match_success),
        "num_inliers": "" if num_inliers is None else int(num_inliers),
        "rmse": "" if rmse is None else float(rmse),
        "elapsed_sec": float(elapsed),
        "error": error_message,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)
    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.is_dir():
        raise SystemExit(f"pairs_dir 不是目录：{pairs_dir}")

    pairs = collect_pairs(pairs_dir)
    if not pairs:
        raise SystemExit(f"在 {pairs_dir} 下没有找到符合命名的 pair*_1.*, pair*_2.* 与 pair*.txt/gt_*.txt")
    if args.max_pairs is not None and args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    config = get_cfg_defaults()
    method, _ = (args.method).split("@-@")[0], (args.method).split("@-@")[1]
    if method == "None":
        raise ValueError("必须提供合法 method")
    config.merge_from_file(args.main_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)
    config.METHOD = method
    if config.LOFTR.COARSE.ROPE:
        assert config.DATASET.NPE_NAME is not None
    if config.DATASET.NPE_NAME is not None and args.imgresize is not None:
        config.LOFTR.COARSE.NPE = [832, 832, args.imgresize, args.imgresize]
    if args.thr is not None:
        config.LOFTR.MATCH_COARSE.THR = args.thr

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[INFO] 使用设备: {device}")
    pl_loftr = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, test_mode=True)
    matcher = pl_loftr.matcher.to(device)
    matcher.eval()
    in_channels = getattr(config.LOFTR, "IN_CHANNELS", 3)
    use_fp16 = bool(getattr(config.LOFTR, "FP16", False)) and device.type == "cuda"
    print(f"[INFO] IN_CHANNELS = {in_channels}")
    print(f"[INFO] 推理使用 autocast(FP16) = {use_fp16}")

    pair_rows: List[Dict[str, object]] = []
    match_list: List[float] = []
    rmse_list: List[float] = []
    time_list: List[float] = []

    for idx, img0_path, img1_path, gt_path in pairs:
        row = process_single_pair(
            idx=idx,
            img0_path=img0_path,
            img1_path=img1_path,
            gt_path=gt_path,
            args=args,
            matcher=matcher,
            device=device,
            in_channels=in_channels,
            use_fp16=use_fp16,
            output_dir=output_dir,
        )
        pair_rows.append(row)
        match_list.append(float(row["num_matches"]))
        time_list.append(float(row["elapsed_sec"]))
        if row["rmse"] != "":
            rmse_list.append(float(row["rmse"]))

    success_pairs = sum(int(row["is_match_success"]) for row in pair_rows)
    total_pairs = len(pair_rows)
    success_rate = (success_pairs / total_pairs) if total_pairs else 0.0
    mean_matches = float(np.mean(match_list)) if match_list else None
    mean_rmse = float(np.mean(rmse_list)) if rmse_list else None
    mean_time = float(np.mean(time_list)) if time_list else None

    save_pair_metrics_csv(output_dir, pair_rows)
    summary_row = {
        "pairs_dir": str(pairs_dir),
        "output_dir": str(output_dir),
        "total_pairs": total_pairs,
        "success_match_threshold": int(args.success_match_threshold),
        "successful_pairs": int(success_pairs),
        "success_rate": float(success_rate),
        "mean_num_matches": "" if mean_matches is None else float(mean_matches),
        "mean_rmse": "" if mean_rmse is None else float(mean_rmse),
        "valid_rmse_pairs": int(len(rmse_list)),
        "mean_elapsed_sec": "" if mean_time is None else float(mean_time),
    }
    save_summary_csv(output_dir, summary_row)

    print("\n================ 统计结果 ================")
    print(f"  数据目录: {pairs_dir}")
    print(f"  总 pair 数量: {total_pairs}")
    print(f"  匹配成功阈值: 匹配点数 > {args.success_match_threshold}")
    print(f"  匹配成功数量: {success_pairs}")
    print(f"  匹配成功率: {success_rate:.4f}")
    print(f"  平均匹配点数量: {mean_matches:.2f}" if mean_matches is not None else "  无法计算平均匹配点数量")
    print(f"  有效 RMSE 数量: {len(rmse_list)}")
    print(f"  平均 RMSE: {mean_rmse:.4f}" if mean_rmse is not None else "  无法计算平均 RMSE")
    print(f"  平均运行时长: {mean_time:.4f}s" if mean_time is not None else "  无法计算平均运行时长")
    print("[DONE] 所有 pair 处理完成。")


if __name__ == "__main__":
    main()
