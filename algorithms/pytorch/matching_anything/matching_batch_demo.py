import argparse
import os
import re
import time
from pathlib import Path
import sys
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import pytorch_lightning as pl

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.lightning.lightning_loftr import PL_LoFTR
from src.config.default import get_cfg_defaults
from tools_utils.plot import plot_matches


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="MatchAnything 批量匹配 Demo（输出 matches.txt 与可选可视化）"
    )
    parser.add_argument("main_cfg_path", type=str, help="LoFTR 主配置文件路径（yaml）")
    parser.add_argument("--ckpt_path", type=str, default="", help="LoFTR 权重 ckpt 路径")
    parser.add_argument("--method", type=str, default="loftr@-@ransac_affine", help="方法名（前半段会写入 config.METHOD）")
    parser.add_argument("--thr", type=float, default=0.1, help="coarse-level matching threshold，会写入 config.LOFTR.MATCH_COARSE.THR")
    parser.add_argument("--pairs_dir", type=str, required=True, help="包含 pair*_1.*, pair*_2.* 的目录")
    parser.add_argument("--output_dir", type=str, default="demo_output_pairs_matching", help="输出 matches 和可视化的目录")
    parser.add_argument("--imgresize", type=int, default=None, help="把两张图 resize 成正方形 imgresize×imgresize 后再送进网络")
    parser.add_argument("--no_cuda", action="store_true", help="强制使用 CPU（调试用）")
    parser.add_argument("--plot_matches", action="store_true", help="是否保存匹配可视化图")
    parser.add_argument("--plot_matches_alpha", type=float, default=0.2, help="plot_matches 的 alpha")
    parser.add_argument("--plot_matches_color", type=str, default="error", choices=["green", "error", "conf"], help="plot_matches 中的颜色模式")
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
    tensor = torch.from_numpy(arr)[None]
    return tensor


def tensor_to_vis_image(t: torch.Tensor, in_channels: int) -> np.ndarray:
    t = t[0].detach().cpu()
    if in_channels == 1:
        arr = t[0].numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = t.permute(1, 2, 0).numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def save_matches_txt(path: Path, mkpts0: np.ndarray, mkpts1: np.ndarray) -> None:
    if mkpts0.shape[0] == 0:
        path.write_text("", encoding="utf-8")
        return
    out = np.hstack([mkpts0, mkpts1]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), out, fmt="%.4f")


def collect_pairs(pairs_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs: List[Tuple[str, Path, Path]] = []
    pattern = re.compile(r"pair(\d+)_1\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for p1 in pairs_dir.iterdir():
        if not p1.is_file():
            continue
        m = pattern.match(p1.name)
        if not m:
            continue
        idx = m.group(1)
        img1_path = None
        for ext in exts:
            cand = pairs_dir / f"pair{idx}_2{ext}"
            if cand.is_file():
                img1_path = cand
                break
        if img1_path is None:
            print(f"[WARN] 找不到 pair{idx}_2.*，跳过该组")
            continue
        pairs.append((idx, p1, img1_path))
    pairs.sort(key=lambda x: int(x[0]))
    return pairs


def process_single_pair(
    idx: str,
    img0_path: Path,
    img1_path: Path,
    args,
    matcher,
    device,
    in_channels: int,
    use_fp16: bool,
    output_dir: Path,
) -> Tuple[float, float]:
    print(f"\n[PAIR {idx}] 开始处理")
    print(f"  img0 = {img0_path}")
    print(f"  img1 = {img1_path}")
    t0 = time.perf_counter()

    img0_tensor = load_image_as_tensor(str(img0_path), resize=args.imgresize, in_channels=in_channels)
    img1_tensor = load_image_as_tensor(str(img1_path), resize=args.imgresize, in_channels=in_channels)
    img0_vis = tensor_to_vis_image(img0_tensor, in_channels)
    img1_vis = tensor_to_vis_image(img1_tensor, in_channels)
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

    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    mconf = batch["mconf"].cpu().numpy()

    num_matches = float(mkpts0.shape[0])
    print(f"[PAIR {idx}] 匹配点数量: {int(num_matches)}")

    out_txt = output_dir / f"pair{idx}_matches.txt"
    save_matches_txt(out_txt, mkpts0, mkpts1)

    if args.plot_matches and mkpts0.shape[0] > 0:
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

    elapsed = time.perf_counter() - t0
    print(f"[PAIR {idx}] 耗时: {elapsed:.4f}s")
    return num_matches, elapsed


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.is_dir():
        raise SystemExit(f"pairs_dir 不是目录：{pairs_dir}")
    pairs = collect_pairs(pairs_dir)
    if not pairs:
        raise SystemExit(f"在 {pairs_dir} 下没有找到符合命名的 pair*_1.*, pair*_2.*")
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    config = get_cfg_defaults()
    method, _ = (args.method).split("@-@")[0], (args.method).split("@-@")[1]
    if method != "None":
        config.merge_from_file(args.main_cfg_path)
        pl.seed_everything(config.TRAINER.SEED)
        config.METHOD = method
        if config.LOFTR.COARSE.ROPE:
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None and args.imgresize is not None:
            config.LOFTR.COARSE.NPE = [832, 832, args.imgresize, args.imgresize]
        if args.thr is not None:
            config.LOFTR.MATCH_COARSE.THR = args.thr
    else:
        raise ValueError("必须提供合法 method")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[INFO] 使用设备: {device}")
    pl_loftr = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, test_mode=True)
    matcher = pl_loftr.matcher.to(device)
    matcher.eval()
    in_channels = getattr(config.LOFTR, "IN_CHANNELS", 3)
    print(f"[INFO] IN_CHANNELS = {in_channels}")
    use_fp16 = bool(getattr(config.LOFTR, "FP16", False)) and device.type == "cuda"
    print(f"[INFO] 推理使用 autocast(FP16) = {use_fp16}")

    match_list: List[float] = []
    time_list: List[float] = []
    for idx, img0_path, img1_path in pairs:
        num_matches, elapsed = process_single_pair(
            idx=idx,
            img0_path=img0_path,
            img1_path=img1_path,
            args=args,
            matcher=matcher,
            device=device,
            in_channels=in_channels,
            use_fp16=use_fp16,
            output_dir=Path(args.output_dir),
        )
        match_list.append(num_matches)
        time_list.append(elapsed)

    mean_matches = float(np.mean(match_list)) if match_list else None
    mean_time = float(np.mean(time_list)) if time_list else None
    print("\n================ 统计结果 ================")
    print(f"  总 pair 数量: {len(pairs)}")
    if mean_matches is not None:
        print(f"  平均匹配点数量: {mean_matches:.2f}")
    else:
        print("  无法计算平均匹配点数量")
    if mean_time is not None:
        print(f"  平均运行时长: {mean_time:.4f}s")
    else:
        print("  无法计算平均运行时长")
    print("[DONE] 所有 pair 处理完成。")


if __name__ == "__main__":
    main()
