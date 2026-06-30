"""Dependency-free base helpers shared across the ScanNet++ utility modules.

Holds label normalization (LABEL_FIXES + normalize_label) and scene-directory
resolution (get_scenes_dir). Kept import-light so loader/io/rasterization can all
import these at module top without circular imports.
"""
import json
from pathlib import Path
from typing import Dict

# Label normalization: typo/variant corrections, loaded from scannetpp_label_fixes.json
LABEL_FIXES: Dict[str, str] = json.load(
    open(Path(__file__).parent / 'scannetpp_label_fixes.json')
)


def normalize_label(label: str) -> str:
    """Fix typos and normalize labels from ScanNet++ annotations."""
    label = label.strip()
    while '  ' in label:
        label = label.replace('  ', ' ')
    label = label.rstrip(']').rstrip('[').strip()
    return LABEL_FIXES.get(label, label)


def get_scenes_dir(data_root: Path) -> Path:
    """Get the directory containing scene folders (handles nested 'data' folder)."""
    # ScanNet++ download creates: data_root/data/<scene_id>/
    nested = data_root / "data"
    if nested.exists() and nested.is_dir():
        return nested
    return data_root
