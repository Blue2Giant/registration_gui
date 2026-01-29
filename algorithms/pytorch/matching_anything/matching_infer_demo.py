#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最简 LoFTR Demo：
- 输入两张图片 img0 / img1
- 使用 MatchAnything 里的 PL_LoFTR + config 做初始化
- 跑一次 matcher，拿到 mkpts0 / mkpts1 / mconf
- 可选把匹配可视化保存下来
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import pytorch_lightning as pl

# ==== 让脚本能找到 src.* / tools_utils.* ====
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.lightning.lightning_loftr import PL_LoFTR
from src.config.default import get_cfg_defaults
from tools_utils.plot import plot_matches   # 直接复用你原来的可视化
# --------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="LoFTR 两张图最简 demo（不依赖 Dataset）"
    )
    parser.add_argument("main_cfg_path", type=str, help="LoFTR 主配置文件路径（如 roma_model.py 对应的 yaml）")
    parser.add_argument("--ckpt_path", type=str, default="", help="LoFTR 权重 ckpt 路径")
    parser.add_argument("--method", type=str, default="loftr@-@ransac_affine",
                        help="方法名（保持和原脚本一致，前半段会写入 config.METHOD）")
    parser.add_argument("--thr", type=float, default=0.1,
                        help="coarse-level matching threshold，会写入 config.LOFTR.MATCH_COARSE.THR")
    parser.add_argument("--img0", type=str, required=True, help="第一张图片路径")
    parser.add_argument("--img1", type=str, required=True, help="第二张图片路径")
    parser.add_argument("--imgresize", type=int, default=None,
                        help="（可选）把两张图 resize 成正方形 imgresize×imgresize 后再送进网络")
    parser.add_argument("--output_dir", type=str, default="demo_output",
                        help="输出可视化结果的目录")
    parser.add_argument("--plot_matches", action="store_true",
                        help="是否保存匹配可视化图")
    parser.add_argument("--plot_matches_alpha", type=float, default=0.2,
                        help="plot_matches 的 alpha")
    parser.add_argument("--plot_matches_color", type=str, default="error",
                        choices=["green", "error", "conf"],
                        help="plot_matches 中的颜色模式")
    parser.add_argument("--no_cuda", action="store_true",
                        help="强制使用 CPU（调试用）")

    return parser.parse_args()


def load_image_as_tensor(path: str,
                         resize: int = None,
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


def tensor_to_vis_image(t: torch.Tensor, in_channels: int):
    """
    将 (1, C, H, W) 的 tensor （0~1）还原成可视化用的 numpy 图像：
    - 灰度: (H, W) uint8
    - RGB : (H, W, 3) uint8
    """
    # 先移到 CPU，去掉 batch 维
    t = t[0].detach().cpu()  # [C, H, W] 或 [1, H, W]
    if in_channels == 1:
        # [1, H, W] -> [H, W]
        arr = t[0].numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        # [3, H, W] -> [H, W, 3]
        arr = t.permute(1, 2, 0).numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 1. 读取 & 设置 config ==========
    config = get_cfg_defaults()
    method, estimator = (args.method).split("@-@")[0], (args.method).split("@-@")[1]

    if method != "None":
        # 读 yaml / cfg 文件
        config.merge_from_file(args.main_cfg_path)
        pl.seed_everything(config.TRAINER.SEED)
        config.METHOD = method

        # 根据你原来的逻辑，NPE/ROPE 相关
        if config.LOFTR.COARSE.ROPE:
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None and args.imgresize is not None:
            # 这里简单写死成 [832, 832, imgresize, imgresize]，和你原脚本一致
            config.LOFTR.COARSE.NPE = [832, 832, args.imgresize, args.imgresize]

        # 粗匹配阈值
        if args.thr is not None:
            config.LOFTR.MATCH_COARSE.THR = args.thr

        # 是否使用 FP16 由 config.LOFTR.FP16 决定
    else:
        raise ValueError("Demo 里必须给一个合法 method（如 'loftr@-@ransac_affine'）。")

    # ========== 2. 初始化 LoFTR 模型 ==========
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 注意：PL_LoFTR 会根据 config 创建 matcher
    pl_loftr = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, test_mode=True)
    matcher = pl_loftr.matcher.to(device)
    matcher.eval()

    # ================= 3. 读取两张图 =================
    # 根据 config 判断是灰度还是 RGB（你可以按自己 config 调整）
    in_channels = getattr(config.LOFTR, "IN_CHANNELS", 3)  # 源代码是 1，这里你改成默认 3
    print(f"[INFO] IN_CHANNELS = {in_channels}")

    img0_tensor = load_image_as_tensor(args.img0, resize=args.imgresize, in_channels=in_channels)
    img1_tensor = load_image_as_tensor(args.img1, resize=args.imgresize, in_channels=in_channels)

    # —— 关键修改：直接从 tensor 还原可视化图像，保证尺寸和模型输入一致 ——
    img0_vis = tensor_to_vis_image(img0_tensor, in_channels)
    img1_vis = tensor_to_vis_image(img1_tensor, in_channels)

    # 之后再把 tensor 丢到 device 上
    img0_tensor = img0_tensor.to(device)
    img1_tensor = img1_tensor.to(device)

    # hw 信息（LoFTR 用来恢复原始分辨率）
    _, _, h0, w0 = img0_tensor.shape
    _, _, h1, w1 = img1_tensor.shape

    batch = {
        "image0_rgb": img0_tensor,  # [1, C, H, W]
        "image1_rgb": img1_tensor,  # [1, C, H, W]
        # 按 LoFTR 官方实现的习惯，加上 hw0_i / hw1_i
        "hw0_i": torch.tensor([[h0, w0]], device=device),
        "hw1_i": torch.tensor([[h1, w1]], device=device),
    }

    # ========== 4. 前向匹配 ==========
    with torch.no_grad():
        use_fp16 = bool(getattr(config.LOFTR, "FP16", False)) and device.type == "cuda"
        print(f"[INFO] 推理使用 autocast(FP16) = {use_fp16}")

        if device.type == "cuda":
            # 只在 cuda 上用 autocast
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                matcher(batch)
        else:
            matcher(batch)

    # LoFTR 会在 batch 里写入 mkpts0_f/mkpts1_f/mconf
    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    mconf = batch["mconf"].cpu().numpy()

    print(f"[INFO] 匹配得到点对数量: {mkpts0.shape[0]}")

    # ========== 5. 可选：可视化匹配 ==========
    if args.plot_matches:
        out_path = Path(args.output_dir) / "demo_matches.png"
        print(f"[INFO] 保存匹配可视化到: {out_path}")

        plot_matches(
            img0_vis,
            img1_vis,
            mkpts0,
            mkpts1,
            mconf,
            vertical=False,
            draw_match_type="corres",
            alpha=args.plot_matches_alpha,
            save_path=out_path,
            inverse=False,
            match_error=None,        # 这里只是可视化，不上色 error
            error_thr=5.0,
            color_type=args.plot_matches_color,
        )

    print("[DONE] LoFTR demo 完成。")


if __name__ == "__main__":
    main()
