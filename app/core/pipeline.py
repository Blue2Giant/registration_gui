from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .executor import run_command, workdir_fallback_match_paths
from .matches import resolve_matches
from .transform_estimator import estimate_transform_3x3
from .visualization import save_visualizations


@dataclass(frozen=True)
class TaskInputs:
    algo_name: str
    command: str
    command_cwd: str
    algorithms_root: str
    transform_model: str
    fixed_path: str
    moving_path: str
    output_dir: str
    repo_root: str
    ransac_thresh_px: float
    checker_tile_px: int
    generate_matches_if_missing: bool


@dataclass(frozen=True)
class TaskOutputs:
    transform_model: str
    H_3x3: list[list[float]]
    rmse: float
    matches_count: int
    inliers_count: int
    matches_vis_path: str
    checkerboard_path: str
    warped_path: str
    matches_path: str
    log_path: str


class RegistrationPipeline:
    def __init__(
        self,
        inputs: TaskInputs,
        on_log: Callable[[str], None],
        on_success: Callable[[TaskOutputs], None],
        on_error: Callable[[str], None],
        cancel_check: Callable[[], bool],
    ):
        self._in = inputs
        self._on_log = on_log
        self._on_success = on_success
        self._on_error = on_error
        self._is_cancelled = cancel_check

    def run(self) -> None:
        out_dir = Path(self._in.output_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._on_error(f"Failed to create output directory: {e}")
            return

        log_path = str((out_dir / "run.log").resolve())

        def log_wrapper(s: str) -> None:
            self._on_log(s)
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(s + "\n")
            except Exception:
                pass

        matches_path = str((out_dir / "matches.txt").resolve())

        try:
            log_wrapper(f"=== Task Started ===")
            log_wrapper(f"Algorithm: {self._in.algo_name}")
            log_wrapper(f"Command: {self._in.command}")
            log_wrapper(f"Fixed: {self._in.fixed_path}")
            log_wrapper(f"Moving: {self._in.moving_path}")

            # 1. Run Exe
            cmd = self._in.command.format(
                fixed=self._in.fixed_path,
                moving=self._in.moving_path,
                matches_out=matches_path,
                out_dir=str(out_dir),
                repo_root=self._in.repo_root,
                algorithms_root=self._in.algorithms_root,
            )

            cwd_template = (self._in.command_cwd or "").strip()
            if cwd_template:
                cwd = cwd_template.format(
                    fixed=self._in.fixed_path,
                    moving=self._in.moving_path,
                    matches_out=matches_path,
                    out_dir=str(out_dir),
                    repo_root=self._in.repo_root,
                    algorithms_root=self._in.algorithms_root,
                )
            else:
                cwd = self._in.repo_root
            rr = run_command(command=cmd, cwd=cwd, on_log=log_wrapper, cancel_flag=self._is_cancelled)
            log_wrapper(f"Command finished. Exit code: {rr.exit_code}, Time: {rr.duration_sec:.2f}s")

            if self._is_cancelled():
                self._on_error("Task cancelled by user.")
                return

            # 2. Resolve Matches
            fallbacks = workdir_fallback_match_paths(cwd, self._in.algo_name)
            mr = resolve_matches(
                desired_matches_path=matches_path,
                exe_fallback_paths=fallbacks,
                fixed_img_path=self._in.fixed_path,
                moving_img_path=self._in.moving_path,
                generate_if_missing=self._in.generate_matches_if_missing,
            )
            log_wrapper(f"Matches resolved: {mr.points1.shape[0]} points (Source: {mr.source})")

            # 3. Estimate Transform
            est = estimate_transform_3x3(mr.points1, mr.points2, self._in.transform_model, self._in.ransac_thresh_px)
            log_wrapper(f"{est.model} estimated. Inliers: {int(est.inlier_mask.sum())}, RMSE: {est.rmse:.4f}")

            H_path = str((out_dir / f"H_{est.model}_3x3.txt").resolve())
            np.savetxt(H_path, est.H_3x3, fmt="%.10f")

            # 4. Visualization
            vis = save_visualizations(
                out_dir=str(out_dir),
                fixed_img_path=self._in.fixed_path,
                moving_img_path=self._in.moving_path,
                p1=mr.points1,
                p2=mr.points2,
                inlier_mask=est.inlier_mask,
                H_3x3=est.H_3x3,
                tile_px=self._in.checker_tile_px,
            )
            log_wrapper(f"Visualizations saved to {out_dir}")

            out = TaskOutputs(
                transform_model=str(est.model),
                H_3x3=est.H_3x3.tolist(),
                rmse=float(est.rmse),
                matches_count=int(mr.points1.shape[0]),
                inliers_count=int(est.inlier_mask.sum()),
                matches_vis_path=vis.matches_vis_path,
                checkerboard_path=vis.checkerboard_path,
                warped_path=vis.warped_path,
                matches_path=mr.path,
                log_path=log_path,
            )
            self._on_success(out)

        except Exception as e:
            import traceback
            log_wrapper(traceback.format_exc())
            self._on_error(str(e))
