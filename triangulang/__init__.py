"""TrianguLang: Geometry-Aware Semantic Consensus for Pose-Free 3D Localization"""

import logging
import sys
from pathlib import Path


def get_logger(name: str = None) -> logging.Logger:
    # Get a DDP-aware logger. Only rank 0 logs; other ranks get NullHandler.
    # Usage: logger = triangulang.get_logger(__name__)
    logger = logging.getLogger(name or "triangulang")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def configure_logging(rank: int = 0, level: int = logging.INFO):
    # Call once at startup to set log level and silence non-rank-0 processes.
    root = logging.getLogger("triangulang")
    root.setLevel(level)
    if rank != 0:
        root.handlers = [logging.NullHandler()]
    elif not root.handlers or isinstance(root.handlers[0], logging.NullHandler):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.handlers = [handler]

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
