#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoFTR 多 pair 配准评估 Demo：

输入：
- --pairs_dir 目录，内部包含若干组文件：
    pair1_1.jpg
    pair1_2.jpg
    gt_1.txt

    pair2_1.jpg
    pair2_2.jpg
    gt_2.txt
    ...

- gt_k.txt 中存的是 2x3 或 3x3 的放射/仿射矩阵参数：
    a11 a12 t1
    a21 a22 t2
  若为 2x3，会自动补一行 [0 0 1] 变成 3x3 齐次矩阵。

默认假设：
- gt 矩阵方向为 pairk_2 -> pairk_1（第二张到第一张），
  即和我们用 LoFTR + findHomography(mkpts1, mkpts0) 得到的 H_pred 同向。
- 如你的 GT 实际是 pairk_1 -> pairk_2，可以用 --gt_direction 1to2，
  脚本会自动对 GT 做矩阵求逆。

对每一对图片：
1. 用 LoFTR 提匹配，得到 mkpts0 / mkpts1 / mconf。
2. 用 RANSAC 估计单应性 H_pred（pair2 -> pair1）。
3. （可选）画匹配可视化图：pair{idx}_matches.png。
4. （可选）warp 第二张图到第一张坐标系，生成棋盘图：pair{idx}_chessboard.png。
5. 在第二张图坐标系随机采样若干点（默认 1000），分别用 H_pred 和 H_gt 变换到第一张坐标系，
   计算两组变换结果之间的 RMSE。

输出：
- output_dir/pair{idx}_matches.png
- output_dir/pair{idx}_chessboard.png
- output_dir/rmse_results.json，含每对的 RMSE 和统计平均值。
"""

import argparse
import os
import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import pytorch_lightning as pl
import cv2

# ==== 让脚本能找到 src.* / tools_utils.* ====
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.lightning.lightning_loftr import PL_LoFTR
from src.config.default import get_cfg_defaults
from tools_utils.plot import plot_matches   # 直接复用你原来的可视化


# ===================== 参数解析 =====================

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="LoFTR 多 pair 配准评估（匹配可视化 + 棋盘可视化 + RMSE）"
    )
    parser.add_argument("main_cfg_path", type=str,
                        help="LoFTR 主配置文件路径（yaml）")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="LoFTR 权重 ckpt 路径")
    parser.add_argument("--method", type=str, default="loftr@-@ransac_affine",
                        help="方法名（前半段会写入 config.METHOD）")
    parser.add_argument("--thr", type=float, default=0.1,
                        help="coarse-level matching threshold，会写入 config.LOFTR.MATCH_COARSE.THR")

    parser.add_argument("--pairs_dir", type=str, required=True,
                        help="包含 pair*_1.*, pair*_2.*, gt_*.txt 的目录")
    parser.add_argument("--output_dir", type=str, default="demo_output_pairs",
                        help="输出可视化和 RMSE 结果的目录")
    parser.add_argument("--imgresize", type=int, default=None,
                        help="（可选）把两张图 resize 成正方形 imgresize×imgresize 后再送进网络和评估")
    parser.add_argument("--no_cuda", action="store_true",
                        help="强制使用 CPU（调试用）")

    # 匹配可视化
    parser.add_argument("--plot_matches", action="store_true",
                        help="是否保存匹配可视化图")
    parser.add_argument("--plot_matches_alpha", type=float, default=0.2,
                        help="plot_matches 的 alpha")
    parser.add_argument("--plot_matches_color", type=str, default="error",
                        choices=["green", "error", "conf"],
                        help="plot_matches 中的颜色模式")

    # 棋盘可视化
    parser.add_argument("--save_chessboard", action="store_true",
                        help="是否基于单应性变换保存棋盘可视化图")
    parser.add_argument("--chessboard_tile", type=int, default=64,
                        help="棋盘格子大小（像素）")

    # RMSE 相关
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="计算 RMSE 时，在第二张图上随机采样的点数")
    parser.add_argument("--gt_direction", type=str, default="2to1",
                        choices=["2to1", "1to2"],
                        help=(
                            "GT 矩阵方向：\n"
                            "  2to1: gt 矩阵将 pair*_2 坐标变换到 pair*_1（默认，与 H_pred 一致）\n"
                            "  1to2: gt 矩阵将 pair*_1 坐标变换到 pair*_2（脚本内部会自动取逆）"
                        ))

    parser.add_argument("--max_pairs", type=int, default=None,
                        help="（可选）最多只处理前 N 组 pair，调试用")

    return parser.parse_args()


# ===================== 图像工具 =====================

def load_image_as_tensor(path: str,
                         resize: Optional[int] = None,
                         in_channels: int = 1) -> torch.Tensor:
    """
    读取图片 -> (1, C, H, W) float32 tensor, 范围 [0,1].
    - in_channels=1: 转灰度
    - in_channels=3: 保持 RGB
    """
    if in_channels == 1:
        img = Image.open(path).convert("L")
    else:
        img = Image.open(path).convert("RGB")

    if resize is not None:
        img = img.resize((resize, resize), Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0

    if arr.ndim == 2:
        # H, W -> 1, H, W
        arr = arr[None, :, :]
    else:
        # H, W, C -> C, H, W
        arr = arr.transpose(2, 0, 1)

    tensor = torch.from_numpy(arr)[None]  # -> 1, C, H, W
    return tensor


def tensor_to_vis_image(t: torch.Tensor, in_channels: int) -> np.ndarray:
    """
    将 (1, C, H, W) 的 tensor （0~1）还原成可视化用的 numpy 图像：
    - 灰度: (H, W) uint8
    - RGB : (H, W, 3) uint8
    """
    t = t[0].detach().cpu()  # [C, H, W] 或 [1, H, W]
    if in_channels == 1:
        arr = t[0].numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = t.permute(1, 2, 0).numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def make_checkerboard(img_ref: np.ndarray,
                      img_warped: np.ndarray,
                      tile_size: int) -> np.ndarray:
    """
    根据参考图 img_ref 和已变换图 img_warped 生成棋盘可视化图。
    - img_ref, img_warped: H×W 或 H×W×3, uint8，尺寸相同
    - tile_size: 棋盘格子尺寸
    返回: 与两图同尺寸的 uint8 图像
    """
    h, w = img_ref.shape[:2]
    h_warp, w_warp = img_warped.shape[:2]
    Hc, Wc = min(h, h_warp), min(w, w_warp)
    img_ref = img_ref[:Hc, :Wc]
    img_warped = img_warped[:Hc, :Wc]

    out = np.zeros_like(img_ref)
    for y in range(0, Hc, tile_size):
        for x in range(0, Wc, tile_size):
            y_end = min(y + tile_size, Hc)
            x_end = min(x + tile_size, Wc)
            iy = y // tile_size
            ix = x // tile_size
            if (iy + ix) % 2 == 0:
                out[y:y_end, x:x_end] = img_ref[y:y_end, x:x_end]
            else:
                out[y:y_end, x:x_end] = img_warped[y:y_end, x:x_end]
    return out


# ===================== GT 矩阵 & RMSE =====================

def load_gt_matrix_3x3(gt_path: Path) -> np.ndarray:
    """
    从 gt_*.txt 读取 2x3 或 3x3 矩阵，返回 3x3 numpy 数组（dtype=float64）。
    """
    arr = np.loadtxt(str(gt_path), dtype=np.float64)
    arr = np.asarray(arr)

    if arr.size == 6:
        arr = arr.reshape(2, 3)
    if arr.shape == (2, 3):
        H = np.vstack([arr, np.array([0.0, 0.0, 1.0], dtype=np.float64)])
    elif arr.shape == (3, 3):
        H = arr
    else:
        raise ValueError(f"无法从 {gt_path} 解析 2x3 或 3x3 矩阵，实际 shape={arr.shape}")
    return H.astype(np.float64)


def adapt_gt_to_resized(
    H_gt_raw: np.ndarray,
    gt_direction: str,
    size0_orig: Tuple[int, int],
    size1_orig: Tuple[int, int],
    size0_new: Tuple[int, int],
    size1_new: Tuple[int, int],
) -> np.ndarray:
    """
    将原图坐标系下的 GT 矩阵转换到 resize 后坐标系。
    - H_gt_raw: 3x3, 定义在原始坐标系
    - gt_direction:
        "2to1" : H_gt_raw 将 pair*_2 (img1) -> pair*_1 (img0)
        "1to2" : H_gt_raw 将 pair*_1 (img0) -> pair*_2 (img1) （会自动取逆）
    - size0_orig: (w0_orig, h0_orig)
    - size1_orig: (w1_orig, h1_orig)
    - size0_new : (w0_new, h0_new)
    - size1_new : (w1_new, h1_new)

    返回：定义在 resize 后坐标系的 H_gt_new（img1_new -> img0_new）。
    """
    H = H_gt_raw.copy()
    if gt_direction == "1to2":
        # 文件里是 img0->img1，我们希望 img1->img0
        H = np.linalg.inv(H)

    w0_orig, h0_orig = size0_orig
    w1_orig, h1_orig = size1_orig
    w0_new, h0_new = size0_new
    w1_new, h1_new = size1_new

    sx0 = w0_new / float(w0_orig)
    sy0 = h0_new / float(h0_orig)
    sx1 = w1_new / float(w1_orig)
    sy1 = h1_new / float(h1_orig)

    S0 = np.array([[sx0, 0.0, 0.0],
                   [0.0, sy0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    S1 = np.array([[sx1, 0.0, 0.0],
                   [0.0, sy1, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    S1_inv = np.linalg.inv(S1)

    # x0_new = S0 * x0_orig
    # x1_new = S1 * x1_orig
    # 且 x0_orig = H * x1_orig
    # => x0_new = S0 * H * S1^{-1} * x1_new
    H_new = S0 @ H @ S1_inv
    return H_new


def compute_homography_rmse(
    H_pred: np.ndarray,
    H_gt: np.ndarray,
    width1: int,
    height1: int,
    num_samples: int = 1000,
) -> Optional[float]:
    """
    在第二张图 (img1) 的坐标系中随机采样 num_samples 个点：
    - p ~ U([0, w1), [0, h1))
    - 用 H_pred 和 H_gt 变换到第一张图坐标系
    - 计算欧氏距离的 RMSE

    注意：H_pred 和 H_gt 都应为 (img1 -> img0) 的 3x3 矩阵，并且定义在同一尺度（例如都在 resize 后坐标系）。
    """
    if H_pred is None or H_gt is None:
        return None

    H_pred = np.asarray(H_pred, dtype=np.float64)
    H_gt = np.asarray(H_gt, dtype=np.float64)

    # 随机采样点（img1 坐标系）
    xs = np.random.uniform(0, width1 - 1, size=(num_samples,))
    ys = np.random.uniform(0, height1 - 1, size=(num_samples,))
    ones = np.ones_like(xs)
    pts_h = np.stack([xs, ys, ones], axis=0)  # [3, N]

    # 预测 & GT 变换
    pred_h = H_pred @ pts_h
    gt_h = H_gt @ pts_h

    # 转为笛卡尔坐标
    pred_w = pred_h[2, :]
    gt_w = gt_h[2, :]

    # 避免除 0 / 无穷
    valid = (np.abs(pred_w) > 1e-8) & (np.abs(gt_w) > 1e-8)
    if not np.any(valid):
        return None

    pred_xy = (pred_h[:2, valid] / pred_w[valid])
    gt_xy = (gt_h[:2, valid] / gt_w[valid])

    diff = pred_xy - gt_xy  # [2, M]
    dist2 = np.sum(diff ** 2, axis=0)  # [M]
    rmse = float(np.sqrt(np.mean(dist2)))
    return rmse


# ===================== pair 列表解析 =====================

def collect_pairs(pairs_dir: Path) -> List[Tuple[str, Path, Path, Path]]:
    """
    扫描 pairs_dir，寻找形如：
      pair{idx}_1.<ext>, pair{idx}_2.<ext>, gt_{idx}.txt

    返回列表：
      [(idx_str, img0_path, img1_path, gt_path), ...]
    其中 img0 对应 pair{idx}_1，img1 对应 pair{idx}_2。
    """
    pairs: List[Tuple[str, Path, Path, Path]] = []
    pattern = re.compile(r"pair(\d+)_1\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    for p1 in pairs_dir.iterdir():
        if not p1.is_file():
            continue
        m = pattern.match(p1.name)
        if not m:
            continue
        idx = m.group(1)

        # 找对应的 pair{idx}_2.*
        img1_path = None
        for ext in exts:
            cand = pairs_dir / f"pair{idx}_2{ext}"
            if cand.is_file():
                img1_path = cand
                break
        if img1_path is None:
            print(f"[WARN] 找不到 pair{idx}_2.*，跳过该组")
            continue

        gt_path = pairs_dir / f"gt_{idx}.txt"
        if not gt_path.is_file():
            print(f"[WARN] 找不到 gt_{idx}.txt，跳过该组")
            continue

        pairs.append((idx, p1, img1_path, gt_path))

    # 按 idx 排序
    pairs.sort(key=lambda x: int(x[0]))
    return pairs


# ===================== 单组 pair 处理 =====================

def process_single_pair(
    idx: str,
    img0_path: Path,
    img1_path: Path,
    gt_path: Path,
    args,
    config,
    matcher,
    device,
    in_channels: int,
    use_fp16: bool,
    output_dir: Path,
) -> Dict[str, Optional[float]]:
    """
    处理一组 pair{idx}：
      - LoFTR 匹配
      - 估计 H_pred
      - 可视化匹配 & 棋盘
      - 计算 H_pred vs H_gt 的 RMSE
    返回：
      dict，包括 rmse / num_matches / num_inliers 等。
    """
    print(f"\n[PAIR {idx}] 开始处理")
    print(f"  img0 = {img0_path}")
    print(f"  img1 = {img1_path}")
    print(f"  gt   = {gt_path}")

    # 原图尺寸
    with Image.open(str(img0_path)) as im0_orig:
        w0_orig, h0_orig = im0_orig.size
    with Image.open(str(img1_path)) as im1_orig:
        w1_orig, h1_orig = im1_orig.size

    # 读入并（可选）resize
    img0_tensor = load_image_as_tensor(str(img0_path), resize=args.imgresize, in_channels=in_channels)
    img1_tensor = load_image_as_tensor(str(img1_path), resize=args.imgresize, in_channels=in_channels)

    img0_vis = tensor_to_vis_image(img0_tensor, in_channels)
    img1_vis = tensor_to_vis_image(img1_tensor, in_channels)

    # resize 后尺寸
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

    # ---------- 前向匹配 ----------
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                matcher(batch)
        else:
            matcher(batch)

    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    mconf = batch["mconf"].cpu().numpy()

    num_matches = mkpts0.shape[0]
    print(f"[PAIR {idx}] 匹配点数量: {num_matches}")

    H_pred = None
    num_inliers = None

    if num_matches >= 4:
        H_pred, inliers = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 3.0)
        if H_pred is None:
            print(f"[PAIR {idx}] [WARN] findHomography 失败")
        else:
            num_inliers = int(inliers.sum()) if inliers is not None else None
            print(f"[PAIR {idx}] RANSAC 内点数: {num_inliers}")
    else:
        print(f"[PAIR {idx}] [WARN] 匹配点少于 4 个，无法估计单应性")

    # ---------- 可视化匹配 ----------
    if args.plot_matches and num_matches > 0:
        out_match = output_dir / f"pair{idx}_matches.png"
        print(f"[PAIR {idx}] 保存匹配可视化到: {out_match}")
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

    # ---------- 棋盘可视化 ----------
    if args.save_chessboard and H_pred is not None:
        h_ref, w_ref = img0_vis.shape[:2]
        warped_img1 = cv2.warpPerspective(img1_vis, H_pred, (w_ref, h_ref))
        chessboard_img = make_checkerboard(
            img_ref=img0_vis,
            img_warped=warped_img1,
            tile_size=args.chessboard_tile,
        )
        out_chess = output_dir / f"pair{idx}_chessboard.png"
        Image.fromarray(chessboard_img).save(out_chess)
        print(f"[PAIR {idx}] 棋盘可视化已保存到: {out_chess}")

    # ---------- 读取 GT & 计算 RMSE ----------
    rmse = None
    try:
        H_gt_raw = load_gt_matrix_3x3(gt_path)
        H_gt_new = adapt_gt_to_resized(
            H_gt_raw,
            gt_direction=args.gt_direction,
            size0_orig=(w0_orig, h0_orig),
            size1_orig=(w1_orig, h1_orig),
            size0_new=(w0_new, h0_new),
            size1_new=(w1_new, h1_new),
        )
        print(f"[PAIR {idx}] GT 矩阵 (原始)：\n{H_gt_raw}")
        print(f"[PAIR {idx}] GT 矩阵 (resize 后坐标系, img1->img0)：\n{H_gt_new}")

        if H_pred is not None:
            rmse = compute_homography_rmse(
                H_pred, H_gt_new,
                width1=w1_new,
                height1=h1_new,
                num_samples=args.num_samples,
            )
            if rmse is not None:
                print(f"[PAIR {idx}] H_pred vs H_gt 的 RMSE = {rmse:.4f}")
            else:
                print(f"[PAIR {idx}] [WARN] RMSE 计算无有效点")
    except Exception as e:
        print(f"[PAIR {idx}] [ERROR] 读取 / 适配 GT 或计算 RMSE 失败: {e}")

    return {
        "rmse": rmse,
        "num_matches": int(num_matches),
        "num_inliers": None if num_inliers is None else int(num_inliers),
        "gt_path": str(gt_path),
        "img0": str(img0_path),
        "img1": str(img1_path),
    }


# ===================== 主函数 =====================

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 配置 & 初始化 LoFTR ----------
    config = get_cfg_defaults()
    method, estimator = (args.method).split("@-@")[0], (args.method).split("@-@")[1]

    if method == "None":
        raise ValueError("必须指定合法 method（例如 'loftr@-@ransac_affine'）。")

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
    print(f"[INFO] IN_CHANNELS = {in_channels}")

    use_fp16 = bool(getattr(config.LOFTR, "FP16", False)) and device.type == "cuda"
    print(f"[INFO] 推理使用 autocast(FP16) = {use_fp16}")

    # ---------- 2. 收集所有 pair ----------
    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.is_dir():
        raise SystemExit(f"pairs_dir 不是目录：{pairs_dir}")

    pairs = collect_pairs(pairs_dir)
    if not pairs:
        raise SystemExit(f"在 {pairs_dir} 下没有找到符合命名的 pair*_1.*, pair*_2.*, gt_*.txt")

    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    print(f"[INFO] 共找到 {len(pairs)} 组 pair 将进行处理")

    # ---------- 3. 逐 pair 处理并统计 ----------
    rmse_results: Dict[str, Dict[str, Optional[float]]] = {}
    rmse_list: List[float] = []

    for idx, img0_path, img1_path, gt_path in pairs:
        metrics = process_single_pair(
            idx=idx,
            img0_path=img0_path,
            img1_path=img1_path,
            gt_path=gt_path,
            args=args,
            config=config,
            matcher=matcher,
            device=device,
            in_channels=in_channels,
            use_fp16=use_fp16,
            output_dir=output_dir,
        )
        rmse_results[idx] = metrics
        if metrics.get("rmse") is not None and np.isfinite(metrics["rmse"]):
            rmse_list.append(float(metrics["rmse"]))

    mean_rmse = float(np.mean(rmse_list)) if rmse_list else None
    print("\n================ RMSE 统计结果 ================")
    print(f"  有效 pair 数量: {len(rmse_list)}")
    if mean_rmse is not None:
        print(f"  平均 RMSE: {mean_rmse:.4f}")
    else:
        print("  无法计算平均 RMSE（可能所有 pair 都失败）")

    # ---------- 4. 写入 JSON 结果 ----------
    summary = {
        "pairs_dir": str(pairs_dir),
        "num_pairs_total": len(pairs),
        "num_pairs_valid_rmse": len(rmse_list),
        "mean_rmse": mean_rmse,
        "rmse_per_pair": rmse_results,
        "gt_direction": args.gt_direction,
        "num_samples": args.num_samples,
        "imgresize": args.imgresize,
    }
    out_json = output_dir / "rmse_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] RMSE 结果已写入: {out_json}")
    print("[DONE] 所有 pair 处理完成。")


if __name__ == "__main__":
    main()
