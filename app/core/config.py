from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExeEntry:
    name: str
    path: str


@dataclass(frozen=True)
class AlgorithmEntry:
    name: str
    command: str
    cwd: str = ""
    env_hint: str = ""


@dataclass
class AppConfig:
    exes: list[ExeEntry]
    algorithms: list[AlgorithmEntry]
    algorithms_root: str
    last_input_mode: str
    last_folder: str
    last_pairs_txt: str
    last_fixed: str
    last_moving: str
    last_output_root: str
    last_transform_model: str
    ransac_thresh_px: float
    ransac_max_iters: int
    ransac_confidence: float
    ransac_refine_iters: int
    checker_tile_px: int
    generate_matches_if_missing: bool

    @staticmethod
    def opencv_ransac_defaults(transform_model: str = "affine") -> dict[str, float | int]:
        m = (transform_model or "").strip().lower()
        defaults: dict[str, float | int] = {
            "ransac_thresh_px": 3.0,
            "ransac_max_iters": 2000,
            "ransac_confidence": 0.99,
            "ransac_refine_iters": 10,
        }
        if m == "homography":
            defaults["ransac_confidence"] = 0.995
        return defaults

    @staticmethod
    def default() -> "AppConfig":
        ransac_defaults = AppConfig.opencv_ransac_defaults("affine")
        return AppConfig(
            exes=[],
            algorithms=[
                AlgorithmEntry(
                    name="MapGlue (PyTorch)",
                    command="python \"{algorithms_root}\\\\pytorch\\\\MapGlue\\\\map_glue_demo.py\" \"{fixed}\" \"{moving}\" \"{matches_out}\" --device cpu",
                    cwd="{algorithms_root}\\\\pytorch\\\\MapGlue",
                    env_hint="环境：需要安装 PyTorch + OpenCV；权重文件放在 algorithms\\\\pytorch\\\\MapGlue\\\\weights\\\\fastmapglue_model.pt。输出 matches.txt 用于后续 RANSAC 估计。",
                ),
            ],
            algorithms_root=str((Path(__file__).resolve().parents[2] / "algorithms").resolve()),
            last_input_mode="folder",
            last_folder="",
            last_pairs_txt="",
            last_fixed="",
            last_moving="",
            last_output_root=str((Path(__file__).resolve().parents[2] / "outputs").resolve()),
            last_transform_model="affine",
            ransac_thresh_px=float(ransac_defaults["ransac_thresh_px"]),
            ransac_max_iters=int(ransac_defaults["ransac_max_iters"]),
            ransac_confidence=float(ransac_defaults["ransac_confidence"]),
            ransac_refine_iters=int(ransac_defaults["ransac_refine_iters"]),
            checker_tile_px=48,
            generate_matches_if_missing=True,
        )


def config_path() -> Path:
    return (Path(__file__).resolve().parents[2] / "user_config.json").resolve()


def load_config() -> AppConfig:
    p = config_path()
    if not p.exists():
        return AppConfig.default()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except:
        return AppConfig.default()

    exes_raw = raw.get("exes", [])
    exes = []
    for item in exes_raw:
        if isinstance(item, dict) and "name" in item and "path" in item:
            exes.append(ExeEntry(name=str(item["name"]), path=str(item["path"])))

    algos_raw = raw.get("algorithms", [])
    algos: list[AlgorithmEntry] = []
    for item in algos_raw:
        if isinstance(item, dict) and "name" in item and "command" in item:
            algos.append(
                AlgorithmEntry(
                    name=str(item["name"]),
                    command=str(item["command"]),
                    cwd=str(item.get("cwd", "")),
                    env_hint=str(item.get("env_hint", "")),
                )
            )

    cfg = AppConfig.default()
    cfg.exes = exes
    cfg.algorithms_root = str(raw.get("algorithms_root", cfg.algorithms_root) or "").strip()
    if not cfg.algorithms_root or not Path(cfg.algorithms_root).exists():
        cfg.algorithms_root = AppConfig.default().algorithms_root

    if algos:
        cfg.algorithms = algos
    elif exes:
        # Backward compatibility: convert old "exes" list into generic algorithms
        cfg.algorithms = [
            AlgorithmEntry(
                name=e.name,
                command=f"\"{e.path}\" \"{{fixed}}\" \"{{moving}}\" \"{{matches_out}}\"",
                cwd="",
                env_hint="",
            )
            for e in exes
        ]

    cfg.last_input_mode = str(raw.get("last_input_mode", cfg.last_input_mode))
    cfg.last_folder = str(raw.get("last_folder", cfg.last_folder))
    cfg.last_pairs_txt = str(raw.get("last_pairs_txt", cfg.last_pairs_txt))
    cfg.last_fixed = str(raw.get("last_fixed", cfg.last_fixed))
    cfg.last_moving = str(raw.get("last_moving", cfg.last_moving))
    cfg.last_output_root = str(raw.get("last_output_root", cfg.last_output_root))
    cfg.last_transform_model = str(raw.get("last_transform_model", cfg.last_transform_model))
    cfg.ransac_thresh_px = float(raw.get("ransac_thresh_px", cfg.ransac_thresh_px))
    cfg.ransac_max_iters = int(raw.get("ransac_max_iters", cfg.ransac_max_iters))
    cfg.ransac_confidence = float(raw.get("ransac_confidence", cfg.ransac_confidence))
    cfg.ransac_refine_iters = int(raw.get("ransac_refine_iters", cfg.ransac_refine_iters))
    cfg.checker_tile_px = int(raw.get("checker_tile_px", cfg.checker_tile_px))
    cfg.generate_matches_if_missing = bool(raw.get("generate_matches_if_missing", cfg.generate_matches_if_missing))
    return cfg


def save_config(cfg: AppConfig) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if p.exists():
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    if "exes" not in payload:
        payload["exes"] = [asdict(x) for x in cfg.exes]
    if "algorithms" not in payload:
        payload["algorithms"] = [asdict(x) for x in cfg.algorithms]
    if "algorithms_root" not in payload:
        payload["algorithms_root"] = cfg.algorithms_root

    payload["last_input_mode"] = cfg.last_input_mode
    payload["last_folder"] = cfg.last_folder
    payload["last_pairs_txt"] = cfg.last_pairs_txt
    payload["last_fixed"] = cfg.last_fixed
    payload["last_moving"] = cfg.last_moving
    payload["last_output_root"] = cfg.last_output_root
    payload["last_transform_model"] = cfg.last_transform_model
    payload["ransac_thresh_px"] = cfg.ransac_thresh_px
    payload["ransac_max_iters"] = cfg.ransac_max_iters
    payload["ransac_confidence"] = cfg.ransac_confidence
    payload["ransac_refine_iters"] = cfg.ransac_refine_iters
    payload["checker_tile_px"] = cfg.checker_tile_px
    payload["generate_matches_if_missing"] = cfg.generate_matches_if_missing
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
