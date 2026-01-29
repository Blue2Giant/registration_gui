from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .pipeline import TaskInputs, TaskOutputs
from .transform_estimator import estimate_transform_3x3
from .visualization import save_visualizations


@dataclass(frozen=True)
class ManualInputs:
    fixed_path: str
    moving_path: str
    output_dir: str
    transform_model: str
    ransac_thresh_px: float
    checker_tile_px: int
    points_fixed: np.ndarray
    points_moving: np.ndarray


class ManualRegistrationPipeline:
    def __init__(
        self,
        inputs: ManualInputs,
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

        try:
            log_wrapper("=== Manual Task Started ===")
            log_wrapper(f"Transform model: {self._in.transform_model}")
            log_wrapper(f"RANSAC thresh: {self._in.ransac_thresh_px}")

            if self._is_cancelled():
                self._on_error("Task cancelled by user.")
                return

            p_fixed = np.asarray(self._in.points_fixed, dtype=np.float32).reshape(-1, 2)
            p_moving = np.asarray(self._in.points_moving, dtype=np.float32).reshape(-1, 2)
            if p_fixed.shape[0] != p_moving.shape[0]:
                raise ValueError("manual points length mismatch")
            if p_fixed.shape[0] < 4:
                raise ValueError("need at least 4 matched point pairs")

            matches = np.hstack([p_fixed, p_moving]).astype(np.float32)
            matches_path = str((out_dir / "matches.txt").resolve())
            np.savetxt(matches_path, matches, fmt="%.4f")
            log_wrapper(f"Manual matches saved: {matches_path}  pairs={matches.shape[0]}")

            est = estimate_transform_3x3(p_fixed, p_moving, self._in.transform_model, self._in.ransac_thresh_px)
            log_wrapper(f"{est.model} estimated. Inliers: {int(est.inlier_mask.sum())}, RMSE: {est.rmse:.4f}")

            H_path = str((out_dir / f"H_{est.model}_3x3.txt").resolve())
            np.savetxt(H_path, est.H_3x3, fmt="%.10f")

            vis = save_visualizations(
                out_dir=str(out_dir),
                fixed_img_path=self._in.fixed_path,
                moving_img_path=self._in.moving_path,
                p1=p_fixed,
                p2=p_moving,
                inlier_mask=est.inlier_mask,
                H_3x3=est.H_3x3,
                tile_px=self._in.checker_tile_px,
            )

            out = TaskOutputs(
                transform_model=str(est.model),
                H_3x3=est.H_3x3.tolist(),
                rmse=float(est.rmse),
                matches_count=int(p_fixed.shape[0]),
                inliers_count=int(est.inlier_mask.sum()),
                matches_vis_path=vis.matches_vis_path,
                checkerboard_path=vis.checkerboard_path,
                warped_path=vis.warped_path,
                matches_path=matches_path,
                log_path=log_path,
            )
            self._on_success(out)
        except Exception as e:
            import traceback
            log_wrapper(traceback.format_exc())
            self._on_error(str(e))
