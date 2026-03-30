"""SpinNeRF dataset loader for multi-view segmentation evaluation.

10 scenes with COLMAP poses and object masks. First 40 images are background-only;
last 60 have the target object with masks in images_4/label/.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import struct

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


# Actual scene names in SpinNeRF dataset
SPINNERF_SCENES = ['1', '2', '3', '4', '7', '9', '10', '12', 'book', 'trash']

# Scene-specific prompts for language-guided segmentation
# Named scenes have obvious objects; numbered scenes we describe generically
SPINNERF_PROMPTS = {
    'fork': 'fork',
    'truck': 'toy truck',
    'lego': 'lego figure',
    'book': 'book',
    'trash': 'trash bag',
    # Numbered scenes - based on SpinNeRF paper, these are various objects
    '1': 'object',           # Generic - check images to refine
    '2': 'object',
    '3': 'object',
    '4': 'object',
    '7': 'object',
    '9': 'object',
    '10': 'object',
    '12': 'object',
}


def load_poses_bounds(poses_bounds_path: Path) -> Dict[str, Dict]:
    """
    Load camera poses from LLFF-format poses_bounds.npy.

    Returns:
        Dict mapping image_index -> {extrinsics (c2w), bounds}
    """
    if not poses_bounds_path.exists():
        return {}

    poses_bounds = np.load(poses_bounds_path)  # (N, 17)
    # Each row: [R(3x3 flattened), t(3), h, w, f, near, far] = 17 values
    # Actually LLFF format is: [pose(3x5 flattened=15), near, far] = 17

    poses = {}
    for i, pb in enumerate(poses_bounds):
        # LLFF format: first 15 values are 3x5 pose matrix (R|t|hwf)
        pose_3x5 = pb[:15].reshape(3, 5)
        R = pose_3x5[:, :3]  # 3x3 rotation
        t = pose_3x5[:, 3]   # 3 translation
        hwf = pose_3x5[:, 4] # h, w, f (but actually this is [down, right, backwards] basis)

        # Near/far bounds
        near, far = pb[15], pb[16]

        # Build 4x4 camera-to-world matrix
        # LLFF convention: camera looks along -Z, Y is up
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = t

        poses[i] = {
            'extrinsics': c2w,
            'near': near,
            'far': far,
        }

    return poses


def load_colmap_intrinsics(sparse_dir: Path) -> Optional[Dict]:
    """Load intrinsics from COLMAP cameras.bin if available."""
    cameras_bin = sparse_dir / '0' / 'cameras.bin'
    if not cameras_bin.exists():
        cameras_bin = sparse_dir / 'cameras.bin'
    if not cameras_bin.exists():
        return None

    try:
        cameras = {}
        with open(cameras_bin, 'rb') as f:
            num_cameras = struct.unpack('Q', f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack('i', f.read(4))[0]
                model_id = struct.unpack('i', f.read(4))[0]
                width = struct.unpack('Q', f.read(8))[0]
                height = struct.unpack('Q', f.read(8))[0]

                num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
                params = struct.unpack('d' * num_params, f.read(8 * num_params))

                cameras[camera_id] = {
                    'model_id': model_id,
                    'width': width,
                    'height': height,
                    'params': params,
                }
        return cameras
    except:
        return None


def sample_points_from_mask(
    mask: np.ndarray,
    num_positive: int = 8,
    num_negative: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample click points from binary mask (MV-SAM style).

    Args:
        mask: (H, W) binary mask (1 = foreground)
        num_positive: Number of positive points to sample
        num_negative: Number of negative points to sample

    Returns:
        points: (N, 2) normalized points (x, y) in [0, 1]
        labels: (N,) 1=positive, 0=negative
    """
    H, W = mask.shape
    points = []
    labels = []

    # Sample positive points from foreground
    fg_coords = np.argwhere(mask > 0.5)  # (N, 2) as (y, x)
    if len(fg_coords) > 0:
        indices = np.random.choice(len(fg_coords), min(num_positive, len(fg_coords)), replace=False)
        if len(indices) < num_positive:
            extra = np.random.choice(len(fg_coords), num_positive - len(indices), replace=True)
            indices = np.concatenate([indices, extra])
        for idx in indices:
            y, x = fg_coords[idx]
            points.append([x / W, y / H])
            labels.append(1)
    else:
        for _ in range(num_positive):
            points.append([0.5, 0.5])
            labels.append(1)

    # Sample negative points from background
    bg_coords = np.argwhere(mask <= 0.5)  # (N, 2) as (y, x)
    if len(bg_coords) > 0:
        indices = np.random.choice(len(bg_coords), min(num_negative, len(bg_coords)), replace=False)
        if len(indices) < num_negative:
            extra = np.random.choice(len(bg_coords), num_negative - len(indices), replace=True)
            indices = np.concatenate([indices, extra])
        for idx in indices:
            y, x = bg_coords[idx]
            points.append([x / W, y / H])
            labels.append(0)
    else:
        corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        for i in range(num_negative):
            points.append(list(corners[i % len(corners)]))
            labels.append(0)

    return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class SpinNeRFDataset(Dataset):
    """
    SpinNeRF dataset for multi-view segmentation training/evaluation.

    Each sample contains multiple views of a scene with GT masks.
    Has COLMAP camera poses for 3D-aware methods.

    NOTE: Uses last 60 images (with object) from each scene.
          First 40 are GT images without object and are skipped.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        views_per_sample: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        mask_size: Tuple[int, int] = (128, 128),
        num_pos_points: int = 8,
        num_neg_points: int = 2,
        exclude_scenes: List[str] = None,
        samples_per_scene: int = 1,
        downsample_factor: int = 4,  # Use images_4/ (only option with masks)
        use_language: bool = True,  # Use language prompts (scene-specific or generic)
    ):
        """
        Args:
            data_root: Path to SpinNeRF data directory
            split: 'train', 'val', or 'all'
            views_per_sample: Number of views to sample per item
            image_size: (H, W) for image resizing
            mask_size: (H, W) for mask resizing
            num_pos_points: Positive points to sample from reference mask
            num_neg_points: Negative points to sample
            exclude_scenes: Scenes to exclude
            samples_per_scene: Number of samples per scene per epoch
            downsample_factor: Use downsampled images (must be 4 for masks)
            use_language: If True, include scene-specific text prompt; if False, points only
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.mask_size = mask_size
        self.views_per_sample = views_per_sample
        self.num_pos_points = num_pos_points
        self.num_neg_points = num_neg_points
        self.samples_per_scene = samples_per_scene
        self.downsample_factor = downsample_factor
        self.use_language = use_language

        if exclude_scenes is None:
            exclude_scenes = []

        # Handle nested folder structure (spinnerf-dataset/)
        scenes_root = self.data_root
        if (self.data_root / 'spinnerf-dataset').exists():
            scenes_root = self.data_root / 'spinnerf-dataset'

        # Find available scenes
        self.scenes = []
        for scene_name in SPINNERF_SCENES:
            if scene_name in exclude_scenes:
                continue
            scene_path = scenes_root / scene_name
            if scene_path.exists():
                self.scenes.append({
                    'name': scene_name,
                    'path': scene_path,
                })

        # Load scene data
        self._load_scenes()

        print(f"SpinNeRF Dataset: {len(self.scenes)} scenes, {len(self)} samples")

    def _load_scenes(self):
        """Load metadata for each scene."""
        for scene in self.scenes:
            scene_path = scene['path']

            # Images are in images_4/ (4x downsampled, has masks)
            images_dir = scene_path / f'images_{self.downsample_factor}'
            if not images_dir.exists():
                images_dir = scene_path / 'images_4'
            if not images_dir.exists():
                images_dir = scene_path / 'images'

            # Get all images (sorted) - last 60 are training views WITH object
            all_image_files = sorted([
                f for f in images_dir.glob('*.png')
                if f.stem not in ['label', 'test_label'] and not f.is_dir()
            ])

            # Skip first 40 (GT without object), use last 60 (with object)
            if len(all_image_files) > 60:
                image_files = all_image_files[-60:]  # Last 60 have masks
            else:
                image_files = all_image_files

            scene['images'] = image_files
            scene['images_dir'] = images_dir

            # Masks are in images_4/label/
            masks_dir = images_dir / 'label'
            scene['masks_dir'] = masks_dir if masks_dir.exists() else None

            # Load LLFF poses from poses_bounds.npy
            poses_path = scene_path / 'poses_bounds.npy'
            if poses_path.exists():
                all_poses = load_poses_bounds(poses_path)
                # Map to training images (offset by 40 if we have full 100)
                if len(all_image_files) > 60:
                    offset = len(all_image_files) - 60
                    scene['poses'] = {i: all_poses.get(i + offset, {}) for i in range(60)}
                else:
                    scene['poses'] = all_poses
            else:
                scene['poses'] = {}

            # Try to load intrinsics from COLMAP
            sparse_dir = scene_path / 'sparse'
            colmap_cameras = load_colmap_intrinsics(sparse_dir)
            if colmap_cameras:
                # Use first camera's intrinsics
                cam = list(colmap_cameras.values())[0]
                params = cam['params']
                if cam['model_id'] == 1:  # PINHOLE
                    fx, fy, cx, cy = params
                elif cam['model_id'] == 0:  # SIMPLE_PINHOLE
                    f, cx, cy = params
                    fx = fy = f
                else:
                    fx = fy = params[0] if params else 500
                    cx, cy = cam['width'] / 2, cam['height'] / 2

                scene['intrinsics'] = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                scene['orig_size'] = (cam['height'], cam['width'])
            else:
                # Estimate from image
                if image_files:
                    img = Image.open(image_files[0])
                    w, h = img.size
                    f = max(h, w)  # Rough estimate
                    scene['intrinsics'] = np.array([
                        [f, 0, w/2],
                        [0, f, h/2],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    scene['orig_size'] = (h, w)
                else:
                    scene['intrinsics'] = None
                    scene['orig_size'] = None

            # Reference mask for point sampling (first training view's mask)
            scene['reference_mask'] = None
            if scene['masks_dir'] and len(image_files) > 0:
                first_img_name = image_files[0].name
                ref_mask_path = scene['masks_dir'] / first_img_name
                if ref_mask_path.exists():
                    ref_mask = np.array(Image.open(ref_mask_path).convert('L')).astype(np.float32)
                    # SpinNeRF masks are already in [0, 1]
                    if ref_mask.max() > 1.0:
                        ref_mask = ref_mask / 255.0
                    scene['reference_mask'] = ref_mask

    def __len__(self):
        return len(self.scenes) * self.samples_per_scene

    def __getitem__(self, idx: int) -> Dict:
        scene_idx = idx % len(self.scenes)
        scene = self.scenes[scene_idx]

        # Sample views
        image_files = scene['images']
        n_images = len(image_files)

        if n_images <= self.views_per_sample:
            selected_indices = list(range(n_images))
        else:
            selected_indices = random.sample(range(n_images), self.views_per_sample)
        selected_indices = sorted(selected_indices)

        images = []
        gt_masks = []
        intrinsics_list = []
        extrinsics_list = []
        image_names = []

        for img_idx in selected_indices:
            img_path = image_files[img_idx]

            # Load image
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            img = torch.from_numpy(np.array(img)).float() / 255.0
            img = img.permute(2, 0, 1)  # HWC -> CHW
            images.append(img)
            image_names.append(img_path.name)

            # Load mask
            if scene['masks_dir'] is not None:
                mask_path = scene['masks_dir'] / img_path.name
                if mask_path.exists():
                    mask = Image.open(mask_path).convert('L')
                    mask = mask.resize((self.mask_size[1], self.mask_size[0]), Image.NEAREST)
                    mask_arr = np.array(mask).astype(np.float32)
                    # SpinNeRF masks are already in [0, 1], don't divide by 255
                    # Check if values are already normalized
                    if mask_arr.max() > 1.0:
                        mask_arr = mask_arr / 255.0
                    mask = torch.from_numpy(mask_arr)
                    gt_masks.append(mask)
                else:
                    gt_masks.append(torch.zeros(self.mask_size))
            else:
                gt_masks.append(torch.zeros(self.mask_size))

            # Get intrinsics (scale for resized image)
            if scene['intrinsics'] is not None and scene['orig_size'] is not None:
                intrinsics = torch.from_numpy(scene['intrinsics'].copy())
                scale_x = self.image_size[1] / scene['orig_size'][1]
                scale_y = self.image_size[0] / scene['orig_size'][0]
                intrinsics[0, :] *= scale_x
                intrinsics[1, :] *= scale_y
                intrinsics_list.append(intrinsics)
            else:
                intrinsics_list.append(torch.eye(3))

            # Get extrinsics from LLFF poses
            pose_data = scene['poses'].get(img_idx, {})
            if 'extrinsics' in pose_data:
                extrinsics_list.append(torch.from_numpy(pose_data['extrinsics']))
            else:
                extrinsics_list.append(torch.eye(4))

        # Sample points from reference mask
        if scene['reference_mask'] is not None:
            points, labels = sample_points_from_mask(
                scene['reference_mask'],
                self.num_pos_points, self.num_neg_points
            )
        else:
            # Fallback: sample from first GT mask
            if len(gt_masks) > 0 and gt_masks[0].sum() > 0:
                points, labels = sample_points_from_mask(
                    gt_masks[0].numpy(),
                    self.num_pos_points, self.num_neg_points
                )
            else:
                points = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
                labels = torch.tensor([1], dtype=torch.long)

        # Get scene-specific prompt if using language
        if self.use_language:
            prompt = SPINNERF_PROMPTS.get(scene['name'], 'object')
        else:
            prompt = None  # Points-only mode (MV-SAM protocol)

        result = {
            'images': torch.stack(images),                # (N, 3, H, W)
            'gt_masks': torch.stack(gt_masks),            # (N, H, W)
            'intrinsics': torch.stack(intrinsics_list),   # (N, 3, 3)
            'extrinsics': torch.stack(extrinsics_list),   # (N, 4, 4)
            'prompt': prompt,                             # Scene-specific or None
            'prompt_points': points,                      # (N_points, 2) normalized (x, y)
            'prompt_labels': labels,                      # (N_points,) 1=pos, 0=neg
            'scene_id': scene['name'],
            'image_names': image_names,
            'has_metric_scale': False,                    # SpinNeRF has arbitrary scale
            'has_gt_mask': True,
            'use_language': self.use_language,            # Flag for eval scripts
        }

        return result


def download_spinnerf(output_dir: str = 'data/spinnerf'):
    """Print instructions for downloading SpinNeRF dataset."""
    print(f"""

1. Install gdown:
   pip install gdown

2. Download from Google Drive:
   gdown --folder "https://drive.google.com/drive/folders/1N7D4-6IutYD40v9lfXGSVbWrd47UdJEC" -O {output_dir}

3. Extract the dataset:
   cd {output_dir}
   unzip spinnerf-dataset.zip

4. Expected structure after extraction:
   {output_dir}/
   ├── spinnerf-dataset/
   │   ├── 1/
   │   │   ├── images/
   │   │   ├── images_4/
   │   │   │   ├── *.png (100 images)
   │   │   │   ├── label/    (masks for last 60 training images)
   │   │   │   └── test_label/
   │   │   ├── sparse/
   │   │   └── poses_bounds.npy
   │   ├── 2/
   │   ├── 3/
   │   ├── 4/
   │   ├── 7/
   │   ├── 9/
   │   ├── 10/
   │   ├── 12/
   │   ├── book/
   │   └── trash/
   └── spinnerf-dataset.zip

NOTE: The dataset has 100 images per scene:
      - First 40 (sorted): GT captures WITHOUT the unwanted object
      - Last 60: Training views WITH the object (these have masks)
      We only use the last 60 images which have corresponding masks.

Storage: ~6 GB (spinnerf-dataset.zip)
""")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data/spinnerf')
    parser.add_argument('--download', action='store_true', help='Show download instructions')
    args = parser.parse_args()

    if args.download:
        download_spinnerf(args.data_root)
    else:
        # Test the dataset
        dataset = SpinNeRFDataset(args.data_root)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Images shape: {sample['images'].shape}")
            print(f"GT masks shape: {sample['gt_masks'].shape}")
            print(f"Intrinsics shape: {sample['intrinsics'].shape}")
            print(f"Extrinsics shape: {sample['extrinsics'].shape}")
            print(f"Prompt points: {sample['prompt_points'].shape}")
            print(f"Scene: {sample['scene_id']}")
        else:
            print("No scenes found. Run with --download for instructions.")
