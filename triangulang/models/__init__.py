"""
TrianguLang Models

Core models for depth-aware and multi-view segmentation.

Key Models:
- TrianguLangModel: Main model with GASA (Geometry-Aware Semantic Attention)

GASA Components:
- PointmapComputer: depth + pose + K -> world XYZ
- WorldSpacePositionalEncoding: sinusoidal 3D PE
- GeometryAwareSemanticAttention: attention with geometric bias
- SymmetricCentroidHead: permutation-invariant 3D output

Fusion Heads (in simple_fusion.py):
- SimpleFusionHead, CrossAttentionFusionHead, GatedFusionHead

Experimental variants are in _experimental.py (not part of public API).
"""

from .simple_fusion import (
    SimpleFusionHead,
    CrossAttentionFusionHead,
    GatedFusionHead,
)
from .gasa import (
    PointmapComputer,
    WorldSpacePositionalEncoding,
    GeometryAwareSemanticAttention,
    GASABlock,
    GASAEncoder,
    SymmetricCentroidHead,
)

__all__ = [
    'SimpleFusionHead',
    'CrossAttentionFusionHead',
    'GatedFusionHead',
    'PointmapComputer',
    'WorldSpacePositionalEncoding',
    'GeometryAwareSemanticAttention',
    'GASABlock',
    'GASAEncoder',
    'SymmetricCentroidHead',
]
