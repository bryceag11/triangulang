"""TrianguLang: Geometry-Aware Semantic Consensus for Pose-Free 3D Localization"""
__version__ = "1.0.0"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Find BPE vocab file for SAM3 tokenizer
_bpe_rel = Path("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz"
_bpe_full = Path("sam3") / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

BPE_PATH = ""
for candidate in [
    PROJECT_ROOT / _bpe_full,       # repo root: sam3/sam3/assets/...
    PROJECT_ROOT / _bpe_rel,        # repo root: sam3/assets/...
] + [Path(p) / _bpe_rel for p in sys.path] + [Path(p) / _bpe_full for p in sys.path]:
    if candidate.exists():
        BPE_PATH = str(candidate)
        break
