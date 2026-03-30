"""
TrianguLang Models

Core models for depth-aware and multi-view segmentation.

Key Models:
- TrianguLangModel: Main model with GASA (Geometry-Aware Semantic Attention)
- SimpleDA3SAM3: Simpler fusion without GASA

GASA Components:
- PointmapComputer: depth + pose + K -> world XYZ
- WorldSpacePositionalEncoding: sinusoidal 3D PE
- GeometryAwareSemanticAttention: attention with geometric bias
- SymmetricCentroidHead: permutation-invariant 3D output
"""

from .simple_fusion import (
    SimpleFusionHead,
    CrossAttentionFusionHead,
    GatedFusionHead,
    CrossViewAttention,
    CrossViewAttention3D,
    SimpleDA3SAM3,
    TrianguLangGASAModel,
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
    'CrossViewAttention',
    'CrossViewAttention3D',
    'SimpleDA3SAM3',
    'TrianguLangGASAModel',
    'PointmapComputer',
    'WorldSpacePositionalEncoding',
    'GeometryAwareSemanticAttention',
    'GASABlock',
    'GASAEncoder',
    'SymmetricCentroidHead',
]
