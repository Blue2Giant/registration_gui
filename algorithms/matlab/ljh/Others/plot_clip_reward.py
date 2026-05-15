import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _lazy_import_matplotlib():
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 16
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["axes.labelsize"] = 18
    matplotlib.rcParams["xtick.labelsize"] = 14
    matplotlib.rcParams["ytick.labelsize"] = 14
    matplotlib.rcParams["legend.fontsize"] = 14
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    return plt


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    name: str


def _generate_demo(steps: int, seed: int) -> Tuple[Series, Series]:
    rng = np.random.RandomState(int(seed))
    x = np.linspace(0.0, 100.0, int(steps), dtype=np.float32)

    clip_base = 0.745 + 0.075 * (1.0 - np.exp(-x / 22.0))
    clip_noise = rng.normal(loc=0.0, scale=0.0045, size=clip_base.shape).astype(np.float32)
    clip_y = np.clip(clip_base + clip_noise, 0.72, 0.86).astype(np.float32)

    reward_base = 0.045 + 0.19 * (1.0 - np.exp(-x / 28.0))
    reward_jitter = rng.normal(loc=0.0, scale=0.018, size=reward_base.shape).astype(np.float32)
    spikes = (rng.rand(*reward_base.shape) < 0.08).astype(np.float32) * rng.normal(
        loc=0.0, scale=0.03, size=reward_base.shape
    ).astype(np.float32)
    reward_y = np.clip(reward_base + reward_jitter + spikes, 0.0, 0.30).astype(np.float32)

    return (
        Series(x=x, y=clip_y, name="clip"),
        Series(x=x, y=reward_y, name="reward"),
    )


def _parse_floats_csv(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.replace("\n", ",").split(",")]
    parts = [p for p in parts if p]
    return np.array([float(p) for p in parts], dtype=np.float32)


def _load_series(path: str, name: str, values: str = "") -> Series:
    if values.strip():
        y = _parse_floats_csv(values)
        x = np.arange(len(y), dtype=np.float32)
        return Series(x=x, y=y, name=name)

    if not path:
        raise ValueError(f"{name} 未提供输入。请使用 --{name}-path 或 --{name}-values。")

    ext = os.path.splitext(path)[1].lower()

    if ext in {".npy"}:
        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            y = arr.astype(np.float32)
            x = np.arange(len(y), dtype=np.float32)
            return Series(x=x, y=y, name=name)
        if arr.ndim >= 2 and arr.shape[1] >= 2:
            x = arr[:, 0].astype(np.float32)
            y = arr[:, 1].astype(np.float32)
            return Series(x=x, y=y, name=name)
        raise ValueError(f"{name} npy 需要是一维(y)或二维(至少两列: x,y)数组。")

    if ext in {".npz"}:
        z = np.load(path)
        keys = list(z.keys())
        if "x" in z and "y" in z:
            x = np.asarray(z["x"], dtype=np.float32).reshape(-1)
            y = np.asarray(z["y"], dtype=np.float32).reshape(-1)
            return Series(x=x, y=y, name=name)
        if keys:
            arr = np.asarray(z[keys[0]])
            if arr.ndim == 1:
                y = arr.astype(np.float32)
                x = np.arange(len(y), dtype=np.float32)
                return Series(x=x, y=y, name=name)
            if arr.ndim >= 2 and arr.shape[1] >= 2:
                x = arr[:, 0].astype(np.float32)
                y = arr[:, 1].astype(np.float32)
                return Series(x=x, y=y, name=name)
        raise ValueError(f"{name} npz 需要包含 x/y 或至少一个数组键。")

    if ext in {".json"}:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "x" in obj and "y" in obj:
            x = np.asarray(obj["x"], dtype=np.float32).reshape(-1)
            y = np.asarray(obj["y"], dtype=np.float32).reshape(-1)
            return Series(x=x, y=y, name=name)
        if isinstance(obj, dict) and "step" in obj and "value" in obj:
            x = np.asarray(obj["step"], dtype=np.float32).reshape(-1)
            y = np.asarray(obj["value"], dtype=np.float32).reshape(-1)
            return Series(x=x, y=y, name=name)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            if "step" in obj[0] and "value" in obj[0]:
                x = np.asarray([o["step"] for o in obj], dtype=np.float32).reshape(-1)
                y = np.asarray([o["value"] for o in obj], dtype=np.float32).reshape(-1)
                return Series(x=x, y=y, name=name)
        raise ValueError(f"{name} json 需要 {{x,y}}、{{step,value}} 或由 step/value 组成的列表。")

    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    delim = "," if "," in head else None
    if delim is None and "\t" in head:
        delim = "\t"
    try:
        data = np.loadtxt(path, delimiter=delim, dtype=np.float32)
    except Exception:
        data = np.loadtxt(path, dtype=np.float32)

    data = np.asarray(data)
    if data.ndim == 0:
        y = np.array([float(data)], dtype=np.float32)
        x = np.arange(len(y), dtype=np.float32)
        return Series(x=x, y=y, name=name)
    if data.ndim == 1:
        y = data.astype(np.float32).reshape(-1)
        x = np.arange(len(y), dtype=np.float32)
        return Series(x=x, y=y, name=name)
    if data.ndim == 2 and data.shape[1] >= 2:
        x = data[:, 0].astype(np.float32).reshape(-1)
        y = data[:, 1].astype(np.float32).reshape(-1)
        return Series(x=x, y=y, name=name)
    raise ValueError(f"{name} 数据格式不支持：{data.shape}")


def _rescale_x_to_max(x: np.ndarray, x_max: float, x_src_max: Optional[float]) -> np.ndarray:
    x = x.astype(np.float32).reshape(-1)
    if len(x) == 0:
        return x
    x0 = float(np.max(x))
    if x_src_max is not None and x_src_max > 0:
        x0 = float(x_src_max)
    if x0 <= 0:
        x0 = float(max(len(x) - 1, 1))
    scale = float(x_max) / float(x0)
    return (x * scale).astype(np.float32)


def _nice_limits(ymin: float, ymax: float) -> Tuple[float, float]:
    ymin = float(ymin)
    ymax = float(ymax)
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return 0.0, 1.0
    if ymax == ymin:
        pad = 1.0 if ymax == 0 else abs(ymax) * 0.1
        return ymin - pad, ymax + pad
    r = ymax - ymin
    pad = r * 0.06
    lo = ymin - pad
    hi = ymax + pad
    return lo, hi


def _parse_ylim(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("ylim 格式应为 'min,max'")
    return float(parts[0]), float(parts[1])


def _caption(
    clip_name: str,
    reward_name: str,
    x_max: int,
    clip_ylim: Tuple[float, float],
    reward_ylim: Tuple[float, float],
    dual_axis: bool,
) -> str:
    en = (
        f"Figure: Overlaid training curves with a unified step range (0–{x_max}). "
        f"CLIP similarity ({clip_name}) and registration reward ({reward_name}) are shown with "
        f"{'dual y-axes to preserve their original numeric ranges' if dual_axis else 'a shared y-axis'}. "
        f"Y-axis ranges: CLIP [{clip_ylim[0]:.3g}, {clip_ylim[1]:.3g}], Reward [{reward_ylim[0]:.3g}, {reward_ylim[1]:.3g}]."
    )
    zh = (
        f"图：将训练曲线叠加到同一张图中，并统一横轴步数范围为 0–{x_max}。"
        f"曲线包含 CLIP 相似度（{clip_name}）与配准 reward（{reward_name}），"
        f"{'使用双纵轴以保持各自原始数值范围不变' if dual_axis else '使用共享纵轴'}。"
        f"纵轴范围：CLIP [{clip_ylim[0]:.3g}, {clip_ylim[1]:.3g}]，Reward [{reward_ylim[0]:.3g}, {reward_ylim[1]:.3g}]。"
    )
    return en + "\n\n" + zh + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-path", type=str, default="", help="CLIP 相似度数据文件路径")
    parser.add_argument("--reward-path", type=str, default="", help="配准 reward 数据文件路径")
    parser.add_argument("--clip-values", type=str, default="", help="直接粘贴 CLIP y 序列（逗号分隔）")
    parser.add_argument("--reward-values", type=str, default="", help="直接粘贴 reward y 序列（逗号分隔）")
    parser.add_argument("--demo", action="store_true", help="使用内置演示数据（可后续替换）")
    parser.add_argument("--demo-steps", type=int, default=101, help="演示数据点数（默认101，对应0-100）")
    parser.add_argument("--seed", type=int, default=0, help="演示数据随机种子")
    parser.add_argument("--out", type=str, default="clip_reward_overlay.svg")
    parser.add_argument("--format", type=str, default="", choices=["", "pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--x-max", type=int, default=10000)
    parser.add_argument("--x-src-max", type=float, default=100.0)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--xlabel", type=str, default="Step / 步数")
    parser.add_argument("--clip-ylabel", type=str, default="CLIP相似度")
    parser.add_argument("--reward-ylabel", type=str, default="配准奖励")
    parser.add_argument("--clip-name", type=str, default="CLIP相似度")
    parser.add_argument("--reward-name", type=str, default="配准奖励")
    parser.add_argument("--clip-ylim", type=str, default="")
    parser.add_argument("--reward-ylim", type=str, default="")
    parser.add_argument("--shared-y", action="store_true")
    parser.add_argument("--caption-out", type=str, default="")
    args = parser.parse_args()

    if args.demo:
        clip, reward = _generate_demo(args.demo_steps, args.seed)
    else:
        clip = _load_series(args.clip_path, "clip", values=args.clip_values)
        reward = _load_series(args.reward_path, "reward", values=args.reward_values)

    clip_x = _rescale_x_to_max(clip.x, float(args.x_max), args.x_src_max)
    reward_x = _rescale_x_to_max(reward.x, float(args.x_max), args.x_src_max)

    clip_y = clip.y.astype(np.float32).reshape(-1)
    reward_y = reward.y.astype(np.float32).reshape(-1)

    clip_ylim = _parse_ylim(args.clip_ylim) if args.clip_ylim else _nice_limits(np.min(clip_y), np.max(clip_y))
    reward_ylim = (
        _parse_ylim(args.reward_ylim) if args.reward_ylim else _nice_limits(np.min(reward_y), np.max(reward_y))
    )

    out_path = args.out
    if args.format:
        base, _ = os.path.splitext(out_path)
        out_path = base + "." + args.format

    plt = _lazy_import_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 4.8), constrained_layout=True)

    color_reward = "#1f77b4"
    color_clip = "#d62728"

    if args.shared_y:
        l1 = ax.plot(reward_x, reward_y, color=color_reward, lw=1.8, label=args.reward_name)[0]
        l2 = ax.plot(clip_x, clip_y, color=color_clip, lw=1.8, label=args.clip_name)[0]
        ax.set_ylim(_nice_limits(min(reward_ylim[0], clip_ylim[0]), max(reward_ylim[1], clip_ylim[1])))
        ax.set_ylabel(f"{args.reward_ylabel} / {args.clip_ylabel}")
        handles = [l1, l2]
    else:
        ax2 = ax.twinx()
        l1 = ax.plot(reward_x, reward_y, color=color_reward, lw=1.9, label=args.reward_name)[0]
        l2 = ax2.plot(clip_x, clip_y, color=color_clip, lw=1.9, label=args.clip_name)[0]
        ax.set_ylabel(args.reward_ylabel, color=color_reward)
        ax2.set_ylabel(args.clip_ylabel, color=color_clip)
        ax.tick_params(axis="y", labelcolor=color_reward)
        ax2.tick_params(axis="y", labelcolor=color_clip)
        ax.set_ylim(reward_ylim)
        ax2.set_ylim(clip_ylim)
        handles = [l1, l2]

    ax.set_xlim(0, int(args.x_max))
    ax.set_xlabel(args.xlabel)
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, which="major", alpha=0.25)
    ax.tick_params(axis="both", labelsize=14)
    if not args.shared_y:
        ax2.tick_params(axis="both", labelsize=14)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        borderaxespad=0.0,
        handlelength=2.6,
        columnspacing=1.6,
    )

    fmt = os.path.splitext(out_path)[1].lower().lstrip(".") or "svg"
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    caption_out = args.caption_out
    if not caption_out:
        base, _ = os.path.splitext(out_path)
        caption_out = base + "_caption.txt"
    with open(caption_out, "w", encoding="utf-8") as f:
        f.write(
            _caption(
                clip_name=args.clip_name,
                reward_name=args.reward_name,
                x_max=int(args.x_max),
                clip_ylim=clip_ylim,
                reward_ylim=reward_ylim,
                dual_axis=not args.shared_y,
            )
        )

    print("figure", os.path.abspath(out_path))
    print("caption", os.path.abspath(caption_out))


if __name__ == "__main__":
    main()
