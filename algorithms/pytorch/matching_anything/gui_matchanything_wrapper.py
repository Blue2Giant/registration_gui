#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import pytorch_lightning as pl

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from src.lightning.lightning_loftr import PL_LoFTR
from src.config.default import get_cfg_defaults


def _load_image_tensor(path: str, resize: int | None, in_channels: int) -> torch.Tensor:
    if in_channels == 1:
        img = Image.open(path).convert("L")
    else:
        img = Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)[None]


def write_matches_txt(out_path: str, mkpts0: np.ndarray, mkpts1: np.ndarray) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for (x0, y0), (x1, y1) in zip(mkpts0.tolist(), mkpts1.tolist()):
        lines.append(f"{float(x0):.4f} {float(y0):.4f} {float(x1):.4f} {float(y1):.4f}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="GUI wrapper for MatchAnything/LoFTR to produce matches.txt")
    ap.add_argument("--main_cfg_path", type=str, required=True, help="LoFTR main config path")
    ap.add_argument("--ckpt_path", type=str, required=True, help="LoFTR pretrained checkpoint")
    ap.add_argument("--method", type=str, default="matchanything_roma@-@ransac_affine", help="Method string")
    ap.add_argument("--img0", type=str, required=True, help="Fixed image path")
    ap.add_argument("--img1", type=str, required=True, help="Moving image path")
    ap.add_argument("--matches_out", type=str, required=True, help="Output matches.txt path")
    ap.add_argument("--imgresize", type=int, default=832, help="Optional square resize for network input")
    ap.add_argument("--output_dir", type=str, default="demo_output", help="Optional output directory for logs")
    ap.add_argument("--no_cuda", action="store_true", help="Force CPU")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = get_cfg_defaults()
    method, _est = (args.method).split("@-@")[0], (args.method).split("@-@")[1]
    if method != "None":
        config.merge_from_file(args.main_cfg_path)
        pl.seed_everything(config.TRAINER.SEED)
        config.METHOD = method
        # 若使用 ROPE/NPE，按原项目约定设置
        if getattr(config.LOFTR.COARSE, "ROPE", False):
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None and args.imgresize is not None:
            config.LOFTR.COARSE.NPE = [832, 832, args.imgresize, args.imgresize]
    else:
        raise ValueError("Invalid method")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    pl_loftr = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, test_mode=True)
    matcher = pl_loftr.matcher.to(device)
    matcher.eval()

    in_channels = getattr(config.LOFTR, "IN_CHANNELS", 3)
    img0 = _load_image_tensor(args.img0, resize=args.imgresize, in_channels=in_channels).to(device)
    img1 = _load_image_tensor(args.img1, resize=args.imgresize, in_channels=in_channels).to(device)

    _, _, h0, w0 = img0.shape
    _, _, h1, w1 = img1.shape
    batch = {
        "image0_rgb": img0,
        "image1_rgb": img1,
        "hw0_i": torch.tensor([[h0, w0]], device=device),
        "hw1_i": torch.tensor([[h1, w1]], device=device),
    }

    with torch.no_grad():
        use_fp16 = bool(getattr(config.LOFTR, "FP16", False)) and device.type == "cuda"
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                matcher(batch)
        else:
            matcher(batch)

    mkpts0 = batch["mkpts0_f"].detach().cpu().numpy()
    mkpts1 = batch["mkpts1_f"].detach().cpu().numpy()

    write_matches_txt(args.matches_out, mkpts0, mkpts1)
    print(f"[OK] Wrote matches: {args.matches_out}  count={mkpts0.shape[0]}")


if __name__ == "__main__":
    main()
