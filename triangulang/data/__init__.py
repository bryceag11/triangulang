"""
TrianguLang data loading modules.

Supported datasets:
    - scannetpp: ScanNet++ indoor scenes (default, supervised training)
    - nvos: NVOS benchmark
    - spinnerf: SpinNeRF benchmark
    - mvimgnet: MVImgNet object-centric dataset
    - uco3d: uCO3D turntable videos
    - partimagenet: PartImageNet part-level segmentation
    - lerf_ovs: LERF open-vocabulary segmentation
"""

from .dataset_factory import get_dataset, get_dataset_config, collate_fn_universal, SUPPORTED_DATASETS

__all__ = [
    'get_dataset',
    'get_dataset_config',
    'collate_fn_universal',
    'SUPPORTED_DATASETS',
]
