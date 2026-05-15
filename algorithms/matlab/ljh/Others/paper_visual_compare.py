import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _lazy_import_matplotlib():
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
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


def _lazy_import_pil():
    from PIL import Image

    return Image


def _repo_root_from_this_file() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _find_default_optical_image(repo_root: str) -> str:
    candidates = [
        os.path.join(repo_root, "ht_eval_for_own_origin", "pair1_1.jpg"),
        os.path.join(repo_root, "ht_eval_for_own_origin", "pair3_1.jpg"),
        os.path.join(repo_root, "ht_eval_for_own_origin", "pair5_1.jpg"),
        os.path.join(repo_root, "save_image_origin", "matches.jpg"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    patterns = [
        os.path.join(repo_root, "ht_eval_for_own_origin", "*_1.jpg"),
        os.path.join(repo_root, "**", "*.jpg"),
        os.path.join(repo_root, "**", "*.png"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat, recursive=True))
        matches = [
            m
            for m in matches
            if os.path.isfile(m)
            and ("save_image" not in m.replace("\\", "/"))
            and ("scalespace_" not in m.replace("\\", "/"))
        ]
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "未找到可用的输入图像。请使用 --input 指定一张光学图像路径。"
    )


def _load_rgb01(path: str) -> np.ndarray:
    Image = _lazy_import_pil()
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _rgb_to_gray01(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def _rgb_to_ycbcr01(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack([y, cb, cr], axis=-1).astype(np.float32)


def _ycbcr_to_rgb01(ycbcr: np.ndarray) -> np.ndarray:
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1] - 0.5
    cr = ycbcr[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= np.sum(k)
    return k


def _convolve1d_reflect(x: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    pad = (len(k) - 1) // 2
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (pad, pad)
    xp = np.pad(x, pad_width=pad_width, mode="reflect")
    out = np.zeros_like(x, dtype=np.float32)
    slc = [slice(None)] * xp.ndim
    for i, w in enumerate(k):
        slc[axis] = slice(i, i + x.shape[axis])
        out += w * xp[tuple(slc)]
    return out


def _gaussian_blur(rgb01: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_kernel1d(sigma)
    x = rgb01.astype(np.float32)
    x = _convolve1d_reflect(x, k, axis=0)
    x = _convolve1d_reflect(x, k, axis=1)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _gaussian_blur_gray(gray01: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_kernel1d(sigma)
    x = gray01.astype(np.float32)
    x = _convolve1d_reflect(x, k, axis=0)
    x = _convolve1d_reflect(x, k, axis=1)
    return x.astype(np.float32)


def _sobel_magnitude(gray01: np.ndarray) -> np.ndarray:
    gx_k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy_k = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    def conv2(x: np.ndarray, k: np.ndarray) -> np.ndarray:
        xp = np.pad(x, [(1, 1), (1, 1)], mode="reflect").astype(np.float32)
        out = np.zeros_like(x, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                out += k[dy, dx] * xp[dy : dy + x.shape[0], dx : dx + x.shape[1]]
        return out

    gx = conv2(gray01, gx_k)
    gy = conv2(gray01, gy_k)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return mag


def _laplacian_energy(gray01: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    xp = np.pad(gray01, [(1, 1), (1, 1)], mode="reflect").astype(np.float32)
    out = np.zeros_like(gray01, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * xp[dy : dy + gray01.shape[0], dx : dx + gray01.shape[1]]
    return float(np.mean(out * out))


def _anisotropic_diffusion_gray(
    gray01: np.ndarray,
    niter: int = 20,
    kappa: float = 0.08,
    gamma: float = 0.18,
    option: int = 1,
) -> np.ndarray:
    u = gray01.astype(np.float32).copy()
    for _ in range(int(niter)):
        north = np.roll(u, -1, axis=0) - u
        south = np.roll(u, 1, axis=0) - u
        east = np.roll(u, -1, axis=1) - u
        west = np.roll(u, 1, axis=1) - u
        if option == 1:
            c_n = np.exp(-(north / kappa) ** 2)
            c_s = np.exp(-(south / kappa) ** 2)
            c_e = np.exp(-(east / kappa) ** 2)
            c_w = np.exp(-(west / kappa) ** 2)
        else:
            c_n = 1.0 / (1.0 + (north / kappa) ** 2)
            c_s = 1.0 / (1.0 + (south / kappa) ** 2)
            c_e = 1.0 / (1.0 + (east / kappa) ** 2)
            c_w = 1.0 / (1.0 + (west / kappa) ** 2)
        u = u + gamma * (c_n * north + c_s * south + c_e * east + c_w * west)
    return np.clip(u, 0.0, 1.0).astype(np.float32)


def _unsharp_mask(rgb01: np.ndarray, sigma: float = 1.2, amount: float = 0.6) -> np.ndarray:
    blur = _gaussian_blur(rgb01, sigma=sigma)
    sharp = rgb01 + float(amount) * (rgb01 - blur)
    return np.clip(sharp, 0.0, 1.0).astype(np.float32)


def _algo_gaussian_denoise(rgb01: np.ndarray) -> np.ndarray:
    return _gaussian_blur(rgb01, sigma=1.35)


def _algo_edge_preserve_diffusion(rgb01: np.ndarray) -> np.ndarray:
    ycc = _rgb_to_ycbcr01(rgb01)
    y = ycc[..., 0]
    y_d = _anisotropic_diffusion_gray(y, niter=24, kappa=0.075, gamma=0.18, option=1)
    ycc2 = ycc.copy()
    ycc2[..., 0] = y_d
    out = _ycbcr_to_rgb01(ycc2)
    out = _unsharp_mask(out, sigma=1.1, amount=0.45)
    return out


@dataclass
class Metrics:
    edge_strength_ratio: float
    noise_reduction_ratio: float
    detail_energy_ratio: float


def _compute_metrics(orig_gray: np.ndarray, out_gray: np.ndarray) -> Metrics:
    eps = 1e-8
    e0 = _sobel_magnitude(orig_gray)
    e1 = _sobel_magnitude(out_gray)
    edge_strength_ratio = float((np.mean(e1) + eps) / (np.mean(e0) + eps))

    p = 50.0
    thr = np.percentile(e0, p)
    mask = e0 <= thr
    hf0 = orig_gray - _gaussian_blur_gray(orig_gray, sigma=2.0)
    hf1 = out_gray - _gaussian_blur_gray(out_gray, sigma=2.0)
    n0 = float(np.std(hf0[mask]) + eps)
    n1 = float(np.std(hf1[mask]) + eps)
    noise_reduction_ratio = float(1.0 - (n1 / n0))

    d0 = _laplacian_energy(orig_gray) + eps
    d1 = _laplacian_energy(out_gray) + eps
    detail_energy_ratio = float(d1 / d0)

    return Metrics(
        edge_strength_ratio=edge_strength_ratio,
        noise_reduction_ratio=noise_reduction_ratio,
        detail_energy_ratio=detail_energy_ratio,
    )


def _pick_salient_points(diff01: np.ndarray, k: int = 3, radius: int = 18) -> List[Tuple[int, int, float]]:
    x = diff01.astype(np.float32).copy()
    pts: List[Tuple[int, int, float]] = []
    for _ in range(int(k)):
        idx = int(np.argmax(x))
        v = float(x.flat[idx])
        if v <= 0:
            break
        y, xx = np.unravel_index(idx, x.shape)
        pts.append((int(y), int(xx), v))
        y0 = max(0, y - radius)
        y1 = min(x.shape[0], y + radius + 1)
        x0 = max(0, xx - radius)
        x1 = min(x.shape[1], xx + radius + 1)
        x[y0:y1, x0:x1] = 0.0
    return pts


def _fmt_metrics(m: Metrics) -> str:
    return (
        f"Edge strength×{m.edge_strength_ratio:.2f}, "
        f"Noise reduction {m.noise_reduction_ratio*100:.1f}%, "
        f"Detail energy×{m.detail_energy_ratio:.2f}"
    )


def _caption_text(
    input_name: str,
    algo_a_name: str,
    algo_b_name: str,
    m_a: Metrics,
    m_b: Metrics,
    num_arrows: int,
) -> str:
    en = (
        f"Figure: Visual comparison on an optical image ({input_name}). "
        f"(a) Original. (b) {algo_a_name}. (c) {algo_b_name}. "
        f"(d) Absolute difference map |(b)-(c)| overlaid on the original (hot colors indicate larger disagreement), "
        f"with {num_arrows} annotated regions highlighting representative changes. "
        f"Quantitative cues (relative to the original; higher is better for edge/detail, lower noise is better): "
        f"{algo_a_name}: {_fmt_metrics(m_a)}; {algo_b_name}: {_fmt_metrics(m_b)}."
    )
    zh = (
        f"图：在标准光学图像（{input_name}）上的可视化对比。"
        f"(a) 原始图像。(b) {algo_a_name} 结果。(c) {algo_b_name} 结果。"
        f"(d) 叠加在原图上的绝对差异热力图 |(b)-(c)|（暖色表示差异更大），"
        f"并用 {num_arrows} 处箭头标注具有代表性的差异区域。"
        f"定量提示（相对原图；边缘/细节越大越好、噪声越低越好）："
        f"{algo_a_name}：{_fmt_metrics(m_a)}；{algo_b_name}：{_fmt_metrics(m_b)}。"
    )
    return en + "\n\n" + zh + "\n"


def _plot_and_save(
    rgb: np.ndarray,
    out_a: np.ndarray,
    out_b: np.ndarray,
    out_path: str,
    fmt: str,
    title_bilingual: str,
    algo_a_name: str,
    algo_b_name: str,
    annotate_k: int,
) -> Tuple[str, str]:
    plt = _lazy_import_matplotlib()

    orig_gray = _rgb_to_gray01(rgb)
    a_gray = _rgb_to_gray01(out_a)
    b_gray = _rgb_to_gray01(out_b)

    e0 = _sobel_magnitude(orig_gray)
    ea = _sobel_magnitude(a_gray)
    eb = _sobel_magnitude(b_gray)
    t0 = np.percentile(e0, 92.0)
    ta = np.percentile(ea, 92.0)
    tb = np.percentile(eb, 92.0)

    diff = np.abs(a_gray - b_gray).astype(np.float32)
    diff_n = diff / (np.percentile(diff, 99.5) + 1e-8)
    diff_n = np.clip(diff_n, 0.0, 1.0)
    pts = _pick_salient_points(diff_n, k=annotate_k, radius=18)

    fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    fig.suptitle(title_bilingual, fontsize=14)

    def set_axes(ax):
        ax.set_xlabel("X (pixel) / X（像素）")
        ax.set_ylabel("Y (pixel) / Y（像素）")

    axs[0, 0].imshow(rgb)
    axs[0, 0].set_title("Original / 原始图像")
    set_axes(axs[0, 0])

    axs[0, 1].imshow(out_a)
    axs[0, 1].set_title(f"{algo_a_name} / 算法A")
    set_axes(axs[0, 1])
    axs[0, 1].contour(e0, levels=[t0], colors=["#FFD54F"], linewidths=0.8)
    axs[0, 1].contour(ea, levels=[ta], colors=["#00E5FF"], linewidths=0.8)

    axs[1, 0].imshow(out_b)
    axs[1, 0].set_title(f"{algo_b_name} / 算法B")
    set_axes(axs[1, 0])
    axs[1, 0].contour(e0, levels=[t0], colors=["#FFD54F"], linewidths=0.8)
    axs[1, 0].contour(eb, levels=[tb], colors=["#00E5FF"], linewidths=0.8)

    axs[1, 1].imshow(orig_gray, cmap="gray", vmin=0.0, vmax=1.0)
    hm = axs[1, 1].imshow(diff_n, cmap="magma", alpha=0.78, vmin=0.0, vmax=1.0)
    axs[1, 1].set_title("Key differences / 关键差异标注")
    set_axes(axs[1, 1])

    for i, (yy, xx, v) in enumerate(pts, start=1):
        dx = 45 if (xx < diff_n.shape[1] // 2) else -45
        dy = -35 if (yy > diff_n.shape[0] // 2) else 35
        axs[1, 1].annotate(
            f"D{i}",
            xy=(xx, yy),
            xytext=(xx + dx, yy + dy),
            color="white",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="white", lw=1.6),
            bbox=dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.35), ec="white", lw=0.6),
        )

    cbar = fig.colorbar(hm, ax=axs[1, 1], fraction=0.046, pad=0.02)
    cbar.set_label("Abs difference / 绝对差异", rotation=90)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#FFD54F", edgecolor="none", label="Original edges / 原始边缘"),
        Patch(facecolor="#00E5FF", edgecolor="none", label="Output edges / 输出边缘"),
    ]
    axs[0, 1].legend(
        handles=legend_handles,
        loc="lower left",
        framealpha=0.85,
        fontsize=9,
        title="Overlay / 叠加",
        title_fontsize=9,
    )

    base, _ = os.path.splitext(out_path)
    fig_path = f"{base}.{fmt}"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="输入光学图像路径（可选）")
    parser.add_argument(
        "--out",
        type=str,
        default="paper_compare_figure.pdf",
        help="输出文件路径（扩展名会被 --format 覆盖）",
    )
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "svg"])
    parser.add_argument("--max-size", type=int, default=900, help="最长边最大像素（用于加速）")
    parser.add_argument("--annotate-k", type=int, default=3, help="差异箭头标注数量")
    parser.add_argument("--title", type=str, default="Algorithm Comparison / 算法效果对比")
    parser.add_argument("--algo-a-name", type=str, default="Gaussian denoise")
    parser.add_argument("--algo-b-name", type=str, default="Edge-preserving diffusion")
    args = parser.parse_args()

    repo_root = _repo_root_from_this_file()
    input_path = args.input.strip() or _find_default_optical_image(repo_root)
    rgb = _load_rgb01(input_path)

    h, w = rgb.shape[:2]
    scale = float(args.max_size) / float(max(h, w))
    if scale < 1.0:
        Image = _lazy_import_pil()
        img = Image.fromarray((rgb * 255.0).astype(np.uint8))
        rgb = np.asarray(img.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)).astype(
            np.float32
        ) / 255.0

    out_a = _algo_gaussian_denoise(rgb)
    out_b = _algo_edge_preserve_diffusion(rgb)

    orig_gray = _rgb_to_gray01(rgb)
    m_a = _compute_metrics(orig_gray, _rgb_to_gray01(out_a))
    m_b = _compute_metrics(orig_gray, _rgb_to_gray01(out_b))

    fig_path, base = _plot_and_save(
        rgb=rgb,
        out_a=out_a,
        out_b=out_b,
        out_path=args.out,
        fmt=args.format,
        title_bilingual=args.title,
        algo_a_name=args.algo_a_name,
        algo_b_name=args.algo_b_name,
        annotate_k=int(args.annotate_k),
    )

    caption = _caption_text(
        input_name=os.path.basename(input_path),
        algo_a_name=args.algo_a_name,
        algo_b_name=args.algo_b_name,
        m_a=m_a,
        m_b=m_b,
        num_arrows=int(args.annotate_k),
    )
    caption_path = base + "_caption.txt"
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption)

    print("input", input_path)
    print("figure", fig_path)
    print("caption", caption_path)


if __name__ == "__main__":
    main()
