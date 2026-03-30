"""Dataset factory for multi-view training.

Returns the appropriate Dataset class for a given dataset name and config.
Supported: scannetpp, nvos, spinnerf, uco3d, lerf_ovs.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset


SUPPORTED_DATASETS = ['scannetpp', 'nvos', 'spinnerf', 'uco3d', 'lerf_ovs']


def get_dataset(
    dataset_name: str,
    data_root: str,
    split: str = 'train',
    views_per_sample: int = 8,
    image_size: Tuple[int, int] = (518, 518),
    mask_size: Tuple[int, int] = (148, 148),  # SAM3 native: (518 // 14) * 4 = 148
    max_scenes: int = None,
    **kwargs
) -> Dataset:
    """
    Factory function to create the appropriate dataset.

    Args:
        dataset_name: One of 'scannetpp', 'nvos', 'spinnerf', 'mvimgnet'
        data_root: Path to dataset root directory
        split: Dataset split ('train', 'val', 'all')
        views_per_sample: Number of views per training sample
        image_size: (H, W) for image resizing
        mask_size: (H, W) for mask resizing
        max_scenes: Maximum number of scenes to load (for debugging)
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {SUPPORTED_DATASETS}")

    if dataset_name == 'scannetpp':
        from triangulang.utils.scannetpp_loader import ScanNetPPMultiViewDataset
        return ScanNetPPMultiViewDataset(
            data_root=data_root,
            split=split,
            views_per_sample=views_per_sample,
            image_size=image_size,
            mask_size=mask_size,
            max_scenes=max_scenes,
            use_undistorted=kwargs.get('use_undistorted', True),
            supervised=kwargs.get('supervised', True),
            semantic_union=kwargs.get('semantic_union', True),
            sampling_strategy=kwargs.get('sampling_strategy', 'stratified'),
            da3_chunk_size=kwargs.get('da3_chunk_size', 8),
            use_cached_depth=kwargs.get('use_cached_depth', False),
            da3_cache_name=kwargs.get('da3_cache_name', 'da3_cache'),
            min_category_samples=kwargs.get('min_category_samples', 1),
            exclude_categories=kwargs.get('exclude_categories', None),
            num_objects_per_sample=kwargs.get('num_objects_per_sample', 1),
        )

    elif dataset_name == 'nvos':
        from triangulang.data.nvos_dataset import NVOSDataset
        return NVOSDataset(
            data_root=data_root,
            split=split,
            views_per_sample=views_per_sample,
            image_size=image_size,
            mask_size=mask_size,
            num_pos_points=kwargs.get('num_pos_points', 8),
            num_neg_points=kwargs.get('num_neg_points', 2),
            samples_per_scene=kwargs.get('samples_per_scene', 1),
            use_language=kwargs.get('use_language', True),
        )

    elif dataset_name == 'spinnerf':
        from triangulang.data.spinnerf_dataset import SpinNeRFDataset
        return SpinNeRFDataset(
            data_root=data_root,
            split=split,
            views_per_sample=views_per_sample,
            image_size=image_size,
            mask_size=mask_size,
            num_pos_points=kwargs.get('num_pos_points', 8),
            num_neg_points=kwargs.get('num_neg_points', 2),
            samples_per_scene=kwargs.get('samples_per_scene', 1),
            downsample_factor=kwargs.get('downsample_factor', 4),
            use_language=kwargs.get('use_language', True),
        )

    elif dataset_name == 'uco3d':
        from triangulang.data.uco3d_dataset import UCO3DMultiViewDataset
        return UCO3DMultiViewDataset(
            data_root=data_root,
            split=split,
            num_views=views_per_sample,
            image_size=image_size,
            mask_size=mask_size,
            num_sequences=max_scenes,  # Use max_scenes as num_sequences
            frames_per_sequence=kwargs.get('frames_per_sequence', 50),
            categories=kwargs.get('categories', None),
            use_depth=kwargs.get('use_depth', True),
            samples_per_sequence=kwargs.get('samples_per_sequence', 1),
            use_cached_depth=kwargs.get('use_cached_depth', False),
            da3_cache_name=kwargs.get('da3_cache_name', 'da3_metric_cache'),
        )

    elif dataset_name == 'lerf_ovs':
        from triangulang.data.lerf_ovs_dataset import LERFOVSDataset
        return LERFOVSDataset(
            data_root=data_root,
            split=split,
            views_per_sample=views_per_sample,
            image_size=image_size,
            mask_size=mask_size,
            max_scenes=max_scenes,
            scene_filter=kwargs.get('scene_filter', None),
            context_strategy=kwargs.get('context_strategy', 'nearest'),
            use_language=kwargs.get('use_language', True),
        )



def get_dataset_config(dataset_name: str) -> Dict:
    """
    Get default configuration for a dataset.

    Returns:
        Dict with default data_root, split, and other settings
    """
    configs = {
        'scannetpp': {
            'data_root': 'data/scannetpp',
            'split': 'nvs_sem_train',
            'has_metric_scale': True,
            'has_poses': True,
            'has_gt_masks': True,
            'description': 'ScanNet++ indoor scenes with LiDAR ground truth',
        },
        'nvos': {
            'data_root': 'data/nvos',
            'split': 'all',
            'has_metric_scale': False,
            'has_poses': False,
            'has_gt_masks': True,
            'description': 'NVOS benchmark (LLFF subset) with scribble prompts',
        },
        'spinnerf': {
            'data_root': 'data/spinnerf',
            'split': 'all',
            'has_metric_scale': False,
            'has_poses': True,
            'has_gt_masks': True,
            'description': 'SpinNeRF benchmark with COLMAP poses',
        },
        'uco3d': {
            'data_root': 'data/uco3d',
            'split': 'train',
            'has_metric_scale': True,  # VGGSfM-aligned depth
            'has_poses': True,
            'has_gt_masks': True,
            'description': 'uCO3D object-centric turntable videos (~170k sequences, ~1000 categories)',
            'eval_config': {
                'num_sequences': 50,
                'frames_per_sequence': 50,
                'description': '50 sequences, 50 frames each, uniformly sampled',
            },
        },
        'lerf_ovs': {
            'data_root': 'data/lerf_ovs',
            'split': 'eval',
            'has_metric_scale': False,
            'has_poses': True,
            'has_gt_masks': True,
            'description': 'LERF-OVS open-vocabulary 3D segmentation (4 scenes, 63 queries)',
        },
    }
    return configs.get(dataset_name.lower(), {})


def print_dataset_info():
    """Print information about all supported datasets."""
    print("\nSupported Datasets\n")
    for name in SUPPORTED_DATASETS:
        config = get_dataset_config(name)
        print(f"  {name}:")
        print(f"    Description: {config.get('description', 'N/A')}")
        print(f"    Default root: {config.get('data_root', 'N/A')}")
        print(f"    Has poses: {config.get('has_poses', False)}")
        print(f"    Has metric scale: {config.get('has_metric_scale', False)}")
        print()


def collate_fn_universal(batch):
    """
    Universal collate function that works with all datasets.

    Handles differences between datasets (e.g., some have intrinsics, some don't).
    """
    import torch

    # Collect all keys across the batch
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())

    collated = {}

    # Tensor keys - stack along batch dimension
    tensor_keys = [
        'images', 'gt_masks', 'masks', 'intrinsics', 'extrinsics',
        'depths', 'prompt_points', 'prompt_labels', 'centroid_3d',
        'cached_depth'  # Pre-computed DA3 depth for faster training
    ]
    for key in tensor_keys:
        if key in all_keys:
            values = [item.get(key) for item in batch if item.get(key) is not None]
            if values:
                if all(v.shape == values[0].shape for v in values):
                    collated[key] = torch.stack(values)
                else:
                    # Variable shapes - keep as list
                    collated[key] = values

    # Scalar tensor keys
    scalar_keys = ['target_obj_id']
    for key in scalar_keys:
        if key in all_keys:
            values = [item.get(key) for item in batch if item.get(key) is not None]
            if values:
                collated[key] = torch.tensor(values)

    # String/list keys - keep as lists
    list_keys = ['prompt', 'scene_id', 'image_names', 'text_prompts', 'category', 'object_id', 'eval_frame']
    for key in list_keys:
        if key in all_keys:
            collated[key] = [item.get(key) for item in batch]

    # Alias: training script expects 'prompts' but datasets return 'prompt'
    if 'prompt' in collated:
        collated['prompts'] = collated['prompt']

    # Boolean keys - convert to tensor
    bool_keys = ['has_metric_scale', 'has_gt_mask']
    for key in bool_keys:
        if key in all_keys:
            values = [item.get(key, False) for item in batch]
            collated[key] = torch.tensor(values)

    # Tuple keys (like orig_hw) - keep as list
    tuple_keys = ['orig_hw']
    for key in tuple_keys:
        if key in all_keys:
            collated[key] = [item.get(key) for item in batch]

    return collated


if __name__ == '__main__':
    print_dataset_info()
