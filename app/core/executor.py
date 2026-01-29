from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    duration_sec: float


def run_command(
    command: str,
    cwd: str,
    on_log: Callable[[str], None],
    cancel_flag: Callable[[], bool],
) -> RunResult:
    t0 = time.time()
    cwd_path = Path(cwd)
    if not cwd_path.exists():
        cwd_path.mkdir(parents=True, exist_ok=True)

    p = subprocess.Popen(
        command,
        cwd=str(cwd_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        shell=True,
    )

    try:
        if p.stdout is not None:
            for line in p.stdout:
                on_log(line.rstrip("\n"))
                if cancel_flag():
                    break

        if cancel_flag() and p.poll() is None:
            _terminate_process_tree(p)
    finally:
        try:
            if p.stdout is not None:
                p.stdout.close()
        except Exception:
            pass

    code = p.wait()
    return RunResult(exit_code=int(code), duration_sec=float(time.time() - t0))


def run_exe(
    exe_path: str,
    fixed_img_path: str,
    moving_img_path: str,
    matches_out_path: str,
    cwd: str,
    on_log: Callable[[str], None],
    cancel_flag: Callable[[], bool],
) -> RunResult:
    # Command format: exe fixed moving matches_out
    cmd = [exe_path, fixed_img_path, moving_img_path, matches_out_path]
    t0 = time.time()
    
    # Ensure cwd exists
    cwd_path = Path(cwd)
    if not cwd_path.exists():
        cwd_path.mkdir(parents=True, exist_ok=True)

    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    try:
        if p.stdout is not None:
            for line in p.stdout:
                on_log(line.rstrip("\n"))
                if cancel_flag():
                    break

        if cancel_flag() and p.poll() is None:
            _terminate_process_tree(p)
    finally:
        try:
            if p.stdout is not None:
                p.stdout.close()
        except Exception:
            pass

    code = p.wait()
    return RunResult(exit_code=int(code), duration_sec=float(time.time() - t0))


def _terminate_process_tree(p: subprocess.Popen) -> None:
    try:
        p.terminate()
    except Exception:
        return
    try:
        p.wait(timeout=1.5)
        return
    except Exception:
        pass
    try:
        p.kill()
    except Exception:
        pass


def exe_fallback_match_paths(exe_path: str, algo_name: str) -> list[str]:
    exe = Path(exe_path)
    exe_dir = exe.parent
    return [
        str((exe_dir / "matches.txt").resolve()),
        str((exe_dir / f"{algo_name.lower()}_matches.txt").resolve()),
        str((Path.cwd() / "matches.txt").resolve()),
    ]


def workdir_fallback_match_paths(work_dir: str, algo_name: str) -> list[str]:
    d = Path(work_dir)
    return [
        str((d / "matches.txt").resolve()),
        str((d / f"{algo_name.lower()}_matches.txt").resolve()),
        str((Path.cwd() / "matches.txt").resolve()),
    ]
