import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class TaskResult:
    name: str
    success: bool
    return_code: int
    log_file: str
    output_dir: str
    matches_txt: Optional[str]
    matches_vis: Optional[str]
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="对单对 SAR/光学图像批量运行仓库内深度匹配算法，并统一导出匹配可视化。",
    )
    parser.add_argument("--sar", required=True, help="SAR 图像路径（将作为左图）")
    parser.add_argument("--optical", required=True, help="光学图像路径（将作为右图）")
    parser.add_argument(
        "--output_dir",
        default=r"d:\hand_craft_registration\SRIF-master\python_registration_gui\outputs\pair44_all_algorithms",
        help="总输出目录",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--imgresize", type=int, default=832, help="MatchAnything 输入尺寸")
    parser.add_argument("--pair_size", type=int, default=512, help="统一预缩放尺寸（宽高相同）")
    return parser.parse_args()


def find_pair_index(sar_path: Path) -> str:
    name = sar_path.stem
    # 支持 pair44_1 / pair44_sar 之类命名，优先提取 pair 后数字
    import re

    m = re.search(r"pair(\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return "1"


def _resize_to_square(src: Path, dst: Path, size: int) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图像: {src}")
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), resized)


def prepare_single_pair_dir(sar_path: Path, optical_path: Path, dst_dir: Path, pair_size: int) -> tuple[Path, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    pair_idx = find_pair_index(sar_path)
    # 统一预缩放为 pair_size x pair_size，避免不同算法可视化出现大片留白
    sar_dst = dst_dir / f"pair{pair_idx}_1.png"
    optical_dst = dst_dir / f"pair{pair_idx}_2.png"
    _resize_to_square(sar_path, sar_dst, pair_size)
    _resize_to_square(optical_path, optical_dst, pair_size)

    # 兼容需要 GT 的 batch 脚本：若没有真实 GT，这里提供单位阵占位，不影响匹配可视化输出
    gt_dst = dst_dir / f"pair{pair_idx}.txt"
    if not gt_dst.exists():
        gt_dst.write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
    return dst_dir, pair_idx


def run_task(name: str, command: List[str], cwd: Path, out_dir: Path, pair_idx: str) -> TaskResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[TASK] {name}\n")
        f.write(f"[CWD] {cwd}\n")
        f.write("[CMD] " + " ".join(command) + "\n\n")
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=False,
        )
    matches_txt_candidates = [
        out_dir / f"pair{pair_idx}_matches.txt",
        out_dir / "matches.txt",
    ]
    matches_vis_candidates = [
        out_dir / f"pair{pair_idx}_matches.png",
        out_dir / "matches_vis.png",
        out_dir / "matching_result.jpg",
        out_dir / "demo_matches.png",
    ]
    matches_txt = next((str(p) for p in matches_txt_candidates if p.exists()), None)
    matches_vis = next((str(p) for p in matches_vis_candidates if p.exists()), None)
    return TaskResult(
        name=name,
        success=proc.returncode == 0 and (matches_vis is not None or matches_txt is not None),
        return_code=int(proc.returncode),
        log_file=str(log_path),
        output_dir=str(out_dir),
        matches_txt=matches_txt,
        matches_vis=matches_vis,
        note="ok" if proc.returncode == 0 else "script failed, check log",
    )


def _draw_matches_image(img0: np.ndarray, img1: np.ndarray, mkpts0: np.ndarray, mkpts1: np.ndarray) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    gap = 20
    out_h = max(h0, h1)
    out_w = w0 + gap + w1
    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
    canvas[:h0, :w0] = img0
    canvas[:h1, w0 + gap:w0 + gap + w1] = img1
    if mkpts0.shape[0] == 0:
        return canvas
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(mkpts0.shape[0], 3), dtype=np.uint8)
    offset = np.array([w0 + gap, 0], dtype=np.float32)
    for i in range(mkpts0.shape[0]):
        p0 = tuple(np.round(mkpts0[i]).astype(int))
        p1 = tuple(np.round(mkpts1[i] + offset).astype(int))
        color = tuple(int(c) for c in colors[i])
        cv2.circle(canvas, p0, 2, color, -1)
        cv2.circle(canvas, p1, 2, color, -1)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    return canvas


def _save_match_vis_from_txt(
    sar_path: Path,
    optical_path: Path,
    matches_txt: Path,
    out_png: Path,
) -> bool:
    if not matches_txt.exists():
        return False
    arr = np.loadtxt(str(matches_txt), dtype=np.float32)
    arr = np.asarray(arr)
    if arr.size == 0:
        return False
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 4:
        return False
    mkpts0 = arr[:, 0:2]
    mkpts1 = arr[:, 2:4]
    img0 = cv2.imread(str(sar_path), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
    if img0 is None or img1 is None:
        return False
    out = _draw_matches_image(img0, img1, mkpts0, mkpts1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), out)
    return True


def run_legacy_exe_task(
    name: str,
    exe_path: Path,
    work_dir: Path,
    sar_resized_path: Path,
    optical_resized_path: Path,
    out_dir: Path,
    pair_idx: str,
) -> TaskResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    pair_match = out_dir / f"pair{pair_idx}_matches.txt"
    pair_vis = out_dir / f"pair{pair_idx}_matches.png"
    src1 = work_dir / "1.png"
    src2 = work_dir / "2.png"
    backup1 = work_dir / "_backup_1_png.tmp"
    backup2 = work_dir / "_backup_2_png.tmp"
    fallback_matches = [
        work_dir / "matches.txt",
        work_dir / "Matches.txt",
        work_dir / "out_matches.txt",
    ]
    code = -1
    note = "ok"
    try:
        if src1.exists():
            shutil.copy2(src1, backup1)
        if src2.exists():
            shutil.copy2(src2, backup2)
        shutil.copy2(sar_resized_path, src1)
        shutil.copy2(optical_resized_path, src2)

        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"[TASK] {name}\n")
            f.write(f"[CWD] {work_dir}\n")
            f.write(f"[EXE] {exe_path}\n\n")
            proc = subprocess.run(
                [str(exe_path)],
                cwd=str(work_dir),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=False,
            )
            code = int(proc.returncode)

        found = next((p for p in fallback_matches if p.exists()), None)
        if found is not None:
            shutil.copy2(found, pair_match)
            _save_match_vis_from_txt(sar_resized_path, optical_resized_path, pair_match, pair_vis)
        else:
            note = "未找到 matches.txt"
    except Exception as e:
        note = f"运行失败: {e}"
    finally:
        try:
            if backup1.exists():
                shutil.move(str(backup1), str(src1))
        except Exception:
            pass
        try:
            if backup2.exists():
                shutil.move(str(backup2), str(src2))
        except Exception:
            pass

    return TaskResult(
        name=name,
        success=(code == 0 and pair_match.exists()),
        return_code=code,
        log_file=str(log_path),
        output_dir=str(out_dir),
        matches_txt=str(pair_match) if pair_match.exists() else None,
        matches_vis=str(pair_vis) if pair_vis.exists() else None,
        note=note if code == 0 else "script failed, check log",
    )


def run_matlab_task(
    name: str,
    matlab_call: str,
    cwd: Path,
    out_dir: Path,
    pair_idx: str,
    sar_resized_path: Path,
    optical_resized_path: Path,
) -> TaskResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    pair_match = out_dir / f"pair{pair_idx}_matches.txt"
    pair_vis = out_dir / f"pair{pair_idx}_matches.png"
    cmd = ["matlab", "-batch", matlab_call]
    code = -1
    note = "ok"
    try:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"[TASK] {name}\n")
            f.write(f"[CWD] {cwd}\n")
            f.write("[CMD] " + " ".join(cmd) + "\n\n")
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=False,
            )
            code = int(proc.returncode)
        if pair_match.exists():
            _save_match_vis_from_txt(sar_resized_path, optical_resized_path, pair_match, pair_vis)
        else:
            note = "MATLAB 已运行但未产出 matches.txt"
    except Exception as e:
        note = f"运行失败: {e}"

    return TaskResult(
        name=name,
        success=(code == 0 and pair_match.exists()),
        return_code=code,
        log_file=str(log_path),
        output_dir=str(out_dir),
        matches_txt=str(pair_match) if pair_match.exists() else None,
        matches_vis=str(pair_vis) if pair_vis.exists() else None,
        note=note if code == 0 else "script failed, check log",
    )


def main() -> None:
    args = parse_args()
    sar = Path(args.sar).resolve()
    optical = Path(args.optical).resolve()
    if not sar.is_file():
        raise SystemExit(f"SAR 图不存在: {sar}")
    if not optical.is_file():
        raise SystemExit(f"光学图不存在: {optical}")

    repo_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_dir).resolve()
    input_pair_dir, pair_idx = prepare_single_pair_dir(
        sar, optical, output_root / "_input_pair", args.pair_size
    )
    sar_resized = input_pair_dir / f"pair{pair_idx}_1.png"
    optical_resized = input_pair_dir / f"pair{pair_idx}_2.png"

    py_exec = sys.executable
    tasks: List[TaskResult] = []

    # 1) MapGlue
    mapglue_dir = repo_root / "python_registration_gui" / "algorithms" / "pytorch" / "MapGlue"
    tasks.append(
        run_task(
            name="MapGlue",
            command=[
                py_exec,
                "map_glue_batch_demo.py",
                "--pairs_dir",
                str(input_pair_dir),
                "--output_dir",
                str(output_root / "MapGlue"),
                "--plot_matches",
                "--max_pairs",
                "1",
                "--device",
                args.device,
            ],
            cwd=mapglue_dir,
            out_dir=output_root / "MapGlue",
            pair_idx=pair_idx,
        )
    )

    # 2) MIFNet
    mifnet_dir = repo_root / "python_registration_gui" / "algorithms" / "pytorch" / "MIFNet"
    tasks.append(
        run_task(
            name="MIFNet",
            command=[
                py_exec,
                "mifnet_batch_demo.py",
                "--pairs_dir",
                str(input_pair_dir),
                "--output_dir",
                str(output_root / "MIFNet"),
                "--plot_matches",
                "--max_pairs",
                "1",
                "--device",
                args.device,
            ],
            cwd=mifnet_dir,
            out_dir=output_root / "MIFNet",
            pair_idx=pair_idx,
        )
    )

    # 3) MINIMA 子方法
    minima_dir = repo_root / "python_registration_gui" / "algorithms" / "pytorch" / "MINIMA"
    minima_methods = ["xoftr", "sp_lg", "loftr", "roma"]
    for method in minima_methods:
        subdir = output_root / f"MINIMA_{method}"
        tasks.append(
            run_task(
                name=f"MINIMA/{method}",
                command=[
                    py_exec,
                    "mmim_batch_demo.py",
                    "--method",
                    method,
                    "--pairs_dir",
                    str(input_pair_dir),
                    "--output_dir",
                    str(subdir),
                    "--plot_matches",
                    "--max_pairs",
                    "1",
                ],
                cwd=minima_dir,
                out_dir=subdir,
                pair_idx=pair_idx,
            )
        )

    # 4) MatchAnything（ROMA / ELoFTR）
    ma_dir = repo_root / "python_registration_gui" / "algorithms" / "pytorch" / "matching_anything"
    ma_cfg = repo_root / "matching_anything" / "config.py"
    ma_weights = [
        ("MatchAnything_ROMA", repo_root / "matching_anything" / "weights" / "matchanything_roma.ckpt"),
        ("MatchAnything_ELoFTR", repo_root / "matching_anything" / "weights" / "matchanything_eloftr.ckpt"),
    ]
    for name, ckpt in ma_weights:
        subdir = output_root / name
        tasks.append(
            run_task(
                name=name,
                command=[
                    py_exec,
                    "matching_batch_demo.py",
                    str(ma_cfg),
                    "--method",
                    "matchanything_roma@-@ransac_affine",
                    "--ckpt_path",
                    str(ckpt),
                    "--pairs_dir",
                    str(input_pair_dir),
                    "--imgresize",
                    str(args.imgresize),
                    "--output_dir",
                    str(subdir),
                    "--plot_matches",
                    "--max_pairs",
                    "1",
                ],
                cwd=ma_dir,
                out_dir=subdir,
                pair_idx=pair_idx,
            )
        )

    # 5) 传统 EXE 算法（含 LNIFT/SRIF 等）
    repo_alg_root = repo_root / "algorithms"
    legacy_exe_specs = [
        ("LNIFT_EXE", repo_alg_root / "LNIFT" / "LNIFT.exe", repo_alg_root / "LNIFT"),
        ("SRIF_EXE", repo_alg_root / "SRIF" / "SRIF.exe", repo_alg_root / "SRIF"),
        ("3MRS_EXE", repo_alg_root / "3MRS" / "3MRSMatcher.exe", repo_alg_root / "3MRS"),
        ("RIFT_EXE", repo_alg_root / "RIFT" / "demo_RIFT_func" / "for_testing" / "demo_RIFT_func.exe", repo_alg_root / "RIFT"),
        ("OSSIFT_EXE", repo_alg_root / "OSSIFT" / "demo_OSSIFT_func" / "for_testing" / "demo_OSSIFT_func.exe", repo_alg_root / "OSSIFT"),
        ("MS_HLMO_EXE", repo_alg_root / "MS_HLMO" / "demo_MSHLMO_func" / "for_testing" / "demo_MSHLMO_func.exe", repo_alg_root / "MS_HLMO"),
        ("CoFSM_EXE", repo_alg_root / "CoFSM" / "demo_3MRS_func" / "for_testing" / "demo_3MRS_func.exe", repo_alg_root / "CoFSM"),
        ("3MRS_DEMO_EXE", repo_alg_root / "3MRS" / "demo_3MRS_func" / "for_testing" / "demo_3MRS_func.exe", repo_alg_root / "3MRS"),
    ]
    for name, exe_path, work_dir in legacy_exe_specs:
        subdir = output_root / name
        if exe_path.exists():
            tasks.append(
                run_legacy_exe_task(
                    name=name,
                    exe_path=exe_path,
                    work_dir=work_dir,
                    sar_resized_path=sar_resized,
                    optical_resized_path=optical_resized,
                    out_dir=subdir,
                    pair_idx=pair_idx,
                )
            )
        else:
            tasks.append(
                TaskResult(
                    name=name,
                    success=False,
                    return_code=-1,
                    log_file=str((subdir / "run.log").resolve()),
                    output_dir=str(subdir.resolve()),
                    matches_txt=None,
                    matches_vis=None,
                    note=f"exe 不存在: {exe_path}",
                )
            )

    # 6) MATLAB 脚本算法
    matlab_root = repo_root / "python_registration_gui" / "algorithms" / "matlab" / "ljh"
    matlab_tasks = [
        (
            "MATLAB_WSSF",
            "cd('{}'); addpath(genpath(pwd)); WSSF_demo_func('{}','{}','{}');".format(
                str(matlab_root).replace("\\", "/"),
                str(sar_resized).replace("\\", "/"),
                str(optical_resized).replace("\\", "/"),
                str((output_root / "MATLAB_WSSF" / f"pair{pair_idx}_matches.txt")).replace("\\", "/"),
            ),
        ),
        (
            "MATLAB_WSSF_tv_logtv",
            "cd('{}'); addpath(genpath(pwd)); WSSF_demo_tv_logtv('{}','{}','{}');".format(
                str(matlab_root).replace("\\", "/"),
                str(sar_resized).replace("\\", "/"),
                str(optical_resized).replace("\\", "/"),
                str((output_root / "MATLAB_WSSF_tv_logtv" / f"pair{pair_idx}_matches.txt")).replace("\\", "/"),
            ),
        ),
    ]
    for name, matlab_call in matlab_tasks:
        subdir = output_root / name
        tasks.append(
            run_matlab_task(
                name=name,
                matlab_call=matlab_call,
                cwd=matlab_root,
                out_dir=subdir,
                pair_idx=pair_idx,
                sar_resized_path=sar_resized,
                optical_resized_path=optical_resized,
            )
        )

    summary = {
        "sar": str(sar),
        "optical": str(optical),
        "pair_index": pair_idx,
        "pair_size": int(args.pair_size),
        "output_root": str(output_root),
        "results": [t.__dict__ for t in tasks],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for t in tasks if t.success)
    print(f"[DONE] 完成。成功 {ok}/{len(tasks)}，汇总文件: {summary_path}")
    for t in tasks:
        print(
            f"- {t.name}: {'OK' if t.success else 'FAIL'} | "
            f"log={t.log_file} | vis={t.matches_vis or 'N/A'} | txt={t.matches_txt or 'N/A'}"
        )


if __name__ == "__main__":
    main()
