"""NVOS (Neural Volumetric Object Selection) dataset loader.

7 LLFF scenes (6 used, orchid excluded) with scribble prompts and binary GT masks.
20-62 images per scene at 4032x3024 resolution.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

import triangulang
logger = triangulang.get_logger(__name__)


# Scenes used by MV-SAM (orchid excluded)
NVOS_SCENES = ['fern', 'flower', 'fortress', 'horns_center', 'horns_left', 'leaves', 'trex']
NVOS_EXCLUDED = ['orchid']

# Scene-specific prompts for language-guided segmentation
# Based on LLFF dataset - each scene has a clear foreground object
NVOS_PROMPTS = {
    'fern': 'fern plant',
    'flower': 'flower',
    'fortress': 'rock fortress',
    'horns_center': 'horns',
    'horns_left': 'horns',
    'leaves': 'leaves',
    'trex': 'T-Rex skull',
    'orchid': 'orchid',  # Excluded but included for completeness
}

# Reference and target image names for each scene
# From README.md: reference has scribbles, target has GT mask
NVOS_IMAGE_IDS = {
    'fern': {'reference': 'IMG_4038', 'target': 'IMG_4027'},
    'flower': {'reference': 'IMG_2983', 'target': 'IMG_2962'},
    'fortress': {'reference': 'IMG_1821', 'target': 'IMG_1800'},
    'horns_center': {'reference': 'DJI_20200223_163055_437', 'target': 'DJI_20200223_163024_597'},
    'horns_left': {'reference': 'DJI_20200223_163055_437', 'target': 'DJI_20200223_163024_597'},
    'leaves': {'reference': 'IMG_3011', 'target': 'IMG_2997'},
    'trex': {'reference': 'DJI_20200223_163607_906', 'target': 'DJI_20200223_163619_411'},
    'orchid': {'reference': 'IMG_4479', 'target': 'IMG_4480'},
}


def parse_scribbles(scribble_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse scribble image to extract positive and negative pixel coordinates.

    Convention from NVOS:
        - Green (0, 255, 0) = positive (foreground)
        - Red (255, 0, 0) = negative (background)

    Args:
        scribble_img: (H, W, 3) RGB image with scribbles

    Returns:
        pos_coords: (N_pos, 2) array of (y, x) coordinates for positive scribbles
        neg_coords: (N_neg, 2) array of (y, x) coordinates for negative scribbles
    """
    # Green channel high, others low = positive
    green_mask = (scribble_img[:, :, 1] > 200) & (scribble_img[:, :, 0] < 100) & (scribble_img[:, :, 2] < 100)
    # Red channel high, others low = negative
    red_mask = (scribble_img[:, :, 0] > 200) & (scribble_img[:, :, 1] < 100) & (scribble_img[:, :, 2] < 100)

    pos_coords = np.argwhere(green_mask)  # (N, 2) as (y, x)
    neg_coords = np.argwhere(red_mask)    # (N, 2) as (y, x)

    return pos_coords, neg_coords


def sample_points_from_scribbles(
    pos_coords: np.ndarray,
    neg_coords: np.ndarray,
    img_height: int,
    img_width: int,
    num_positive: int = 8,
    num_negative: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample click points from scribble coordinates (MV-SAM style).

    Args:
        pos_coords: (N_pos, 2) positive scribble coordinates (y, x)
        neg_coords: (N_neg, 2) negative scribble coordinates (y, x)
        img_height, img_width: Original image dimensions
        num_positive: Number of positive points to sample
        num_negative: Number of negative points to sample

    Returns:
        points: (N, 2) normalized points (x, y) in [0, 1]
        labels: (N,) 1=positive, 0=negative
    """
    points = []
    labels = []

    # Sample positive points
    if len(pos_coords) > 0:
        indices = np.random.choice(len(pos_coords), min(num_positive, len(pos_coords)), replace=False)
        if len(indices) < num_positive:
            # Sample with replacement if not enough unique points
            extra = np.random.choice(len(pos_coords), num_positive - len(indices), replace=True)
            indices = np.concatenate([indices, extra])
        for idx in indices:
            y, x = pos_coords[idx]
            points.append([x / img_width, y / img_height])  # Normalize to [0, 1]
            labels.append(1)
    else:
        # Fallback: sample from center
        for _ in range(num_positive):
            points.append([0.5, 0.5])
            labels.append(1)

    # Sample negative points
    if len(neg_coords) > 0:
        indices = np.random.choice(len(neg_coords), min(num_negative, len(neg_coords)), replace=False)
        if len(indices) < num_negative:
            extra = np.random.choice(len(neg_coords), num_negative - len(indices), replace=True)
            indices = np.concatenate([indices, extra])
        for idx in indices:
            y, x = neg_coords[idx]
            points.append([x / img_width, y / img_height])
            labels.append(0)
    else:
        # Fallback: sample from corners
        corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        for i in range(num_negative):
            points.append(list(corners[i % len(corners)]))
            labels.append(0)

    return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class NVOSDataset(Dataset):
    """
    NVOS dataset for multi-view segmentation training/evaluation.

    Each sample contains multiple views of a scene with GT masks.
    Uses scribble prompts (8 pos + 2 neg points) as in MV-SAM.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',  # 'train' or 'val' (for NVOS, typically just 'all')
        views_per_sample: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        mask_size: Tuple[int, int] = (128, 128),
        num_pos_points: int = 8,
        num_neg_points: int = 2,
        exclude_scenes: List[str] = None,
        samples_per_scene: int = 1,  # How many samples per scene per epoch
        use_language: bool = True,  # Use language prompts (scene-specific)
    ):
        """
        Args:
            data_root: Path to NVOS data directory
            split: 'train', 'val', or 'all'
            views_per_sample: Number of views to sample per item
            image_size: (H, W) for image resizing
            mask_size: (H, W) for mask resizing
            num_pos_points: Positive points to sample from scribbles
            num_neg_points: Negative points to sample from scribbles
            exclude_scenes: Scenes to exclude (default: ['orchid'])
            samples_per_scene: Number of samples per scene per epoch
            use_language: If True, include scene-specific text prompt; if False, points only
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.mask_size = mask_size
        self.views_per_sample = views_per_sample
        self.num_pos_points = num_pos_points
        self.num_neg_points = num_neg_points
        self.samples_per_scene = samples_per_scene
        self.use_language = use_language

        if exclude_scenes is None:
            exclude_scenes = NVOS_EXCLUDED

        # Find available scenes (check llff/masks/<scene>/ since data lives there)
        self.scenes = []
        llff_masks = self.data_root / 'llff' / 'masks'
        for scene_name in NVOS_SCENES:
            if scene_name in exclude_scenes:
                continue
            scene_mask_path = llff_masks / scene_name
            if scene_mask_path.exists():
                self.scenes.append({
                    'name': scene_name,
                    'path': scene_mask_path,
                })

        # Load scene data
        self._load_scenes()

        logger.info(f"NVOS Dataset: {len(self.scenes)} scenes, {len(self)} samples")

    def _load_scenes(self):
        """Load metadata for each scene using llff directory structure."""
        llff_root = self.data_root / 'llff'

        for scene in self.scenes:
            scene_name = scene['name']
            scene_ids = NVOS_IMAGE_IDS.get(scene_name)

            if not scene_ids:
                logger.warning(f"No image IDs for scene {scene_name}")
                continue

            # Find reference image (with scribbles)
            ref_dir = llff_root / 'reference_image' / scene_name
            ref_candidates = list(ref_dir.glob(f"{scene_ids['reference']}.*"))
            scene['reference_image'] = ref_candidates[0] if ref_candidates else None

            # Find target image and mask
            mask_dir = llff_root / 'masks' / scene_name
            target_candidates = list(mask_dir.glob(f"{scene_ids['target']}.JPG")) + \
                              list(mask_dir.glob(f"{scene_ids['target']}.jpg"))
            scene['target_image'] = target_candidates[0] if target_candidates else None

            mask_candidates = list(mask_dir.glob(f"{scene_ids['target']}_mask.png"))
            scene['target_mask'] = mask_candidates[0] if mask_candidates else None

            # Load scribbles from scribbles directory
            scribble_dir = llff_root / 'scribbles' / scene_name
            scribble_files = list(scribble_dir.glob('*.png'))

            if scribble_files:
                scribble_img = np.array(Image.open(scribble_files[0]).convert('RGB'))
                pos_coords, neg_coords = parse_scribbles(scribble_img)
                scene['pos_coords'] = pos_coords
                scene['neg_coords'] = neg_coords
                scene['scribble_size'] = scribble_img.shape[:2]  # (H, W)
            else:
                scene['pos_coords'] = np.array([])
                scene['neg_coords'] = np.array([])
                scene['scribble_size'] = None

            # Debug: print what we found
            if scene['reference_image'] and scene['target_image'] and scene['target_mask']:
                logger.debug(f"{scene_name}: ref={scene['reference_image'].name}, target={scene['target_image'].name}")
            else:
                logger.warning(f"{scene_name}: Missing files (ref={scene['reference_image']}, target={scene['target_image']}, mask={scene['target_mask']})")

    def __len__(self):
        return len(self.scenes) * self.samples_per_scene

    def __getitem__(self, idx: int) -> Dict:
        scene_idx = idx % len(self.scenes)
        scene = self.scenes[scene_idx]

        # Load reference image (view 0)
        ref_img = Image.open(scene['reference_image']).convert('RGB')
        ref_img = ref_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        ref_img_tensor = torch.from_numpy(np.array(ref_img)).float() / 255.0
        ref_img_tensor = ref_img_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Load target image (view 1)
        target_img = Image.open(scene['target_image']).convert('RGB')
        target_img = target_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        target_img_tensor = torch.from_numpy(np.array(target_img)).float() / 255.0
        target_img_tensor = target_img_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Load target mask (only target has GT)
        target_mask = Image.open(scene['target_mask']).convert('L')
        target_mask = target_mask.resize((self.mask_size[1], self.mask_size[0]), Image.NEAREST)
        target_mask_tensor = torch.from_numpy(np.array(target_mask)).float() / 255.0

        # Reference has no GT mask (zeros)
        ref_mask_tensor = torch.zeros(self.mask_size)

        # Stack: [reference, target]
        images = torch.stack([ref_img_tensor, target_img_tensor])  # (2, 3, H, W)
        gt_masks = torch.stack([ref_mask_tensor, target_mask_tensor])  # (2, H, W)

        # Sample points from scribbles
        if scene['scribble_size'] is not None:
            h, w = scene['scribble_size']
            points, labels = sample_points_from_scribbles(
                scene['pos_coords'], scene['neg_coords'],
                h, w, self.num_pos_points, self.num_neg_points
            )
        else:
            # Fallback: center point
            points = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.long)

        # Get scene-specific prompt if using language
        if self.use_language:
            prompt = NVOS_PROMPTS.get(scene['name'], 'object')
        else:
            prompt = None  # Points-only mode (MV-SAM protocol)

        result = {
            'images': images,                        # (2, 3, H, W) - [reference, target]
            'gt_masks': gt_masks,                    # (2, H, W) - [zeros, target_mask]
            'prompt': prompt,                        # Scene-specific or None
            'prompt_points': points,                 # (N_points, 2) normalized (x, y)
            'prompt_labels': labels,                 # (N_points,) 1=pos, 0=neg
            'scene_id': scene['name'],
            'image_names': [scene['reference_image'].name, scene['target_image'].name],
            'has_metric_scale': False,               # NVOS doesn't have metric depth
            'has_gt_mask': True,
            'use_language': self.use_language,       # Flag for eval scripts
        }

        return result


def download_nvos(output_dir: str = 'data/nvos'):
    """Print instructions for downloading NVOS dataset."""
    print("""

1. Download from Dropbox:
   https://www.dropbox.com/sh/sdgr4mewkhjsg00/AACIKecIwzCHCGma5kkKyLTpa

2. Extract to: {output_dir}

3. Expected structure:
   {output_dir}/
   ├── fern/
   │   ├── images/
   │   ├── masks/
   │   └── scribbles.png
   ├── flower/
   ├── fortress/
   ├── horns_center/
   ├── horns_left/
   ├── leaves/
   └── trex/

Note: The 'orchid' scene is excluded from MV-SAM evaluation.

Alternative: Use wget or gdown for direct download.
""".format(output_dir=output_dir))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data/nvos')
    parser.add_argument('--download', action='store_true', help='Show download instructions')
    args = parser.parse_args()

    if args.download:
        download_nvos(args.data_root)
    else:
        # Test the dataset
        dataset = NVOSDataset(args.data_root)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Images shape: {sample['images'].shape}")
            print(f"GT masks shape: {sample['gt_masks'].shape}")
            print(f"Prompt points: {sample['prompt_points'].shape}")
            print(f"Scene: {sample['scene_id']}")
