from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    key: str
    fixed_path: str
    moving_path: str


def find_pairs(folder: str) -> list[ImagePair]:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []

    by_key: dict[str, dict[str, str]] = {}
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem
        lower = stem.lower()
        role = None
        key = None
        if lower.endswith("_1"):
            role = "fixed"
            key = stem[:-2]
        elif lower.endswith("_2"):
            role = "moving"
            key = stem[:-2]
        
        # Also support just nameA.jpg and nameB.jpg if user wants generic?
        # For now strict to requirement: pairX_1, pairX_2
        
        if role is None or key is None:
            continue
        bucket = by_key.setdefault(key, {})
        bucket[role] = str(p.resolve())

    out: list[ImagePair] = []
    for key, bucket in by_key.items():
        if "fixed" in bucket and "moving" in bucket:
            out.append(ImagePair(key=key, fixed_path=bucket["fixed"], moving_path=bucket["moving"]))

    out.sort(key=lambda x: x.key)
    return out


def parse_pairs_txt(txt_path: str) -> list[ImagePair]:
    p = Path(txt_path)
    if not p.exists() or not p.is_file():
        return []

    out: list[ImagePair] = []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except:
        try:
            lines = p.read_text(encoding="gbk").splitlines()
        except:
            return []

    idx = 0
    for raw in lines:
        s = (raw or "").strip()
        if not s:
            continue
        if s.startswith("#"):
            continue

        parts = [x.strip().strip("\"").strip("'") for x in s.split(",")]
        if len(parts) < 2:
            continue
        fixed = parts[0]
        moving = parts[1]
        if not fixed or not moving:
            continue

        fp = Path(fixed)
        mp = Path(moving)
        if fp.suffix.lower() not in IMAGE_EXTS or mp.suffix.lower() not in IMAGE_EXTS:
            continue
        if not fp.exists() or not mp.exists():
            continue

        key = f"{idx:04d}_{fp.stem}_{mp.stem}"
        out.append(ImagePair(key=key, fixed_path=str(fp.resolve()), moving_path=str(mp.resolve())))
        idx += 1

    return out
