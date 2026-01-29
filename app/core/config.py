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
    ransac_thresh_px: float
    checker_tile_px: int
    generate_matches_if_missing: bool

    @staticmethod
    def default() -> "AppConfig":
        return AppConfig(
            exes=[],
            algorithms=[],
            algorithms_root=str((Path(__file__).resolve().parents[2] / "algorithms").resolve()),
            last_input_mode="folder",
            last_folder="",
            last_pairs_txt="",
            last_fixed="",
            last_moving="",
            last_output_root=str((Path(__file__).resolve().parents[2] / "outputs").resolve()),
            ransac_thresh_px=10.0,
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
    else:
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
    cfg.ransac_thresh_px = float(raw.get("ransac_thresh_px", cfg.ransac_thresh_px))
    cfg.checker_tile_px = int(raw.get("checker_tile_px", cfg.checker_tile_px))
    cfg.generate_matches_if_missing = bool(raw.get("generate_matches_if_missing", cfg.generate_matches_if_missing))
    return cfg


def save_config(cfg: AppConfig) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(cfg)
    payload["exes"] = [asdict(x) for x in cfg.exes]
    payload["algorithms"] = [asdict(x) for x in cfg.algorithms]
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
