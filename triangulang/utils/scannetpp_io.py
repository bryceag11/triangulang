"""ScanNet++ data loading utilities.

Lightweight I/O helpers: scene lists, camera transforms, semantic annotations,
the simple single-view dataset, and the SCANNETPP_PROMPTS vocabulary.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import triangulang
logger = triangulang.get_logger(__name__)


# Common indoor object prompts for ScanNet++
SCANNETPP_PROMPTS = {
    # Furniture
    'chair': ['chair', 'office chair', 'desk chair', 'armchair', 'seat'],
    'table': ['table', 'desk', 'dining table', 'coffee table', 'work table'],
    'sofa': ['sofa', 'couch', 'loveseat', 'settee'],
    'bed': ['bed', 'mattress', 'bedroom furniture'],
    'cabinet': ['cabinet', 'cupboard', 'storage cabinet', 'kitchen cabinet'],
    'shelf': ['shelf', 'bookshelf', 'shelving', 'rack'],
    'door': ['door', 'doorway', 'entrance'],
    'window': ['window', 'glass window', 'window frame'],
    # Electronics
    'monitor': ['monitor', 'computer screen', 'display', 'TV screen'],
    'keyboard': ['keyboard', 'computer keyboard'],
    'lamp': ['lamp', 'light', 'desk lamp', 'floor lamp', 'lighting fixture'],
    'tv': ['TV', 'television', 'flat screen', 'display'],
    # Kitchen
    'refrigerator': ['refrigerator', 'fridge', 'freezer'],
    'microwave': ['microwave', 'microwave oven'],
    'sink': ['sink', 'kitchen sink', 'bathroom sink', 'basin'],
    'toilet': ['toilet', 'bathroom toilet', 'lavatory'],
    # Objects
    'book': ['book', 'books', 'textbook', 'notebook'],
    'plant': ['plant', 'potted plant', 'houseplant', 'flower pot'],
    'bottle': ['bottle', 'water bottle', 'container'],
    'box': ['box', 'cardboard box', 'storage box'],
    'bag': ['bag', 'backpack', 'handbag', 'shopping bag'],
    'pillow': ['pillow', 'cushion', 'throw pillow'],
    'blanket': ['blanket', 'throw', 'bedding'],
    'curtain': ['curtain', 'drape', 'window covering'],
    'picture': ['picture', 'painting', 'artwork', 'photo frame'],
    'mirror': ['mirror', 'wall mirror', 'bathroom mirror'],
    # Structure
    'wall': ['wall', 'room wall'],
    'floor': ['floor', 'flooring', 'ground'],
    'ceiling': ['ceiling', 'room ceiling'],
}


def load_scene_list(data_root: Path, split: str) -> List[str]:
    """
    Load scene IDs from split file.

    Args:
        data_root: Path to scannetpp folder
        split: One of 'nvs_sem_train', 'nvs_sem_val', 'nvs_test', 'sem_test'

    Returns:
        List of scene IDs
    """
    split_file = data_root / "splits" / f"{split}.txt"
    if not split_file.exists():
        # Try alternative locations
        for alt_path in [
            data_root / f"{split}.txt",
            data_root / "metadata" / f"{split}.txt"
        ]:
            if alt_path.exists():
                split_file = alt_path
                break

    if not split_file.exists():
        logger.warning(f"Split file not found: {split_file}")
        return []

    with open(split_file) as f:
        scenes = [line.strip() for line in f if line.strip()]
    return scenes


def load_semantic_classes(data_root: Path) -> Dict[int, str]:
    """Load semantic class mapping from metadata."""
    classes_file = data_root / "metadata" / "semantic_classes.txt"
    if not classes_file.exists():
        return {}

    classes = {}
    with open(classes_file) as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name:
                classes[i] = name
    return classes


def load_nerfstudio_transforms(transforms_path: Path) -> Dict:
    """
    Load camera transforms from nerfstudio format.

    Returns dict with:
        - frames: list of {file_path, transform_matrix, ...}
        - camera intrinsics (fl_x, fl_y, cx, cy, w, h)
    """
    if not transforms_path.exists():
        return None

    with open(transforms_path) as f:
        data = json.load(f)
    return data


def load_train_test_split(scene_path: Path) -> Tuple[List[str], List[str]]:
    """Load image train/test split for a scene."""
    split_file = scene_path / "dslr" / "train_test_lists.json"
    if not split_file.exists():
        return [], []

    with open(split_file) as f:
        data = json.load(f)

    train_images = data.get('train', [])
    test_images = data.get('test', [])
    return train_images, test_images


def get_available_scenes(data_root: Path, split: str = None) -> List[str]:
    """
    Get available scene IDs with valid data.

    Args:
        data_root: Path to scannetpp folder
        split: Optional split to filter by

    Returns:
        List of valid scene IDs
    """
    from triangulang.utils.scannetpp_loader import get_scenes_dir

    data_root = Path(data_root)
    scenes_dir = get_scenes_dir(data_root)

    # Get scenes from split file if specified
    if split:
        scene_ids = load_scene_list(data_root, split)
    else:
        # Find all scene directories
        scene_ids = [d.name for d in scenes_dir.iterdir()
                     if d.is_dir() and not d.name.startswith('.')
                     and d.name not in ['splits', 'metadata', 'data']]

    # Filter to scenes with valid DSLR data
    valid_scenes = []
    for scene_id in scene_ids:
        scene_path = scenes_dir / scene_id
        dslr_dir = scene_path / "dslr"

        # Check for required data
        if not dslr_dir.exists():
            continue

        # Check for images
        images_dir = dslr_dir / "resized_images"
        if not images_dir.exists():
            images_dir = dslr_dir / "resized_undistorted_images"

        if images_dir.exists() and len(list(images_dir.glob("*.JPG"))) > 0:
            valid_scenes.append(scene_id)

    return valid_scenes


def load_semantic_annotations(scene_path: Path) -> Dict[str, List[int]]:
    """
    Load semantic annotations for a scene.

    Returns:
        Dict mapping object label -> list of segment indices
    """
    from triangulang.utils.scannetpp_loader import normalize_label

    anno_file = scene_path / "scans" / "segments_anno.json"
    if not anno_file.exists():
        return {}

    with open(anno_file) as f:
        data = json.load(f)

    # Parse annotations
    annotations = {}
    for item in data.get('segGroups', []):
        label = normalize_label(item.get('label', 'unknown'))
        segments = item.get('segments', [])
        if label not in annotations:
            annotations[label] = []
        annotations[label].extend(segments)

    return annotations


class ScanNetPPDataset(Dataset):
    """
    Dataset for ScanNet++ single-view training.
    Each sample is a single image with semantic prompts.

    Since ScanNet++ doesn't have per-image masks, we use:
    - SAM3's text prompting for segmentation
    - 3D mesh for ground truth (project to 2D if needed)
    """

    def __init__(
        self,
        data_root: Path,
        split: str = 'nvs_sem_train',
        images_per_scene: int = 10,
        image_size: Tuple[int, int] = (518, 518),
        mask_size: Tuple[int, int] = (128, 128),
        use_undistorted: bool = True,
        max_scenes: int = None
    ):
        self.data_root = Path(data_root)

        from triangulang.utils.scannetpp_loader import get_scenes_dir
        self.scenes_dir = get_scenes_dir(self.data_root)
        self.image_size = image_size
        self.mask_size = mask_size
        self.use_undistorted = use_undistorted

        # Get valid scenes
        scenes = get_available_scenes(self.data_root, split)
        if max_scenes:
            scenes = scenes[:max_scenes]

        # Build samples: (image_path, scene_id, prompt)
        self.samples = []
        self.scene_transforms = {}  # Cache transforms per scene

        logger.info(f"Loading ScanNet++ dataset ({split})...")

        from tqdm import tqdm

        for scene_id in tqdm(scenes, desc="Loading scenes"):
            scene_path = self.scenes_dir / scene_id

            # Get image directory
            if use_undistorted:
                images_dir = scene_path / "dslr" / "resized_undistorted_images"
            else:
                images_dir = scene_path / "dslr" / "resized_images"

            if not images_dir.exists():
                continue

            # Get train images only
            train_images, _ = load_train_test_split(scene_path)
            if not train_images:
                # Fall back to all images
                train_images = [f.name for f in images_dir.glob("*.JPG")]

            # Load transforms for camera parameters
            transforms_path = scene_path / "dslr" / "nerfstudio" / "transforms.json"
            if use_undistorted:
                transforms_path = scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"

            transforms = load_nerfstudio_transforms(transforms_path)
            if transforms:
                self.scene_transforms[scene_id] = transforms

            # Load semantic annotations for prompts
            annotations = load_semantic_annotations(scene_path)
            object_labels = list(annotations.keys())

            # Sample images from scene
            if len(train_images) > images_per_scene:
                selected = random.sample(train_images, images_per_scene)
            else:
                selected = train_images

            for img_name in selected:
                img_path = images_dir / img_name
                if img_path.exists():
                    # Choose a random object prompt from scene annotations
                    if object_labels:
                        label = random.choice(object_labels)
                        if label in SCANNETPP_PROMPTS:
                            prompt = SCANNETPP_PROMPTS[label][0]
                        else:
                            prompt = label
                    else:
                        # Fallback to common indoor prompts
                        prompt = random.choice(['chair', 'table', 'monitor', 'lamp'])

                    self.samples.append((img_path, scene_id, prompt))

        logger.info(f"Loaded {len(self.samples)} samples from {len(self.scene_transforms)} scenes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, scene_id, prompt = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size  # (W, H)
        image = image.resize(self.image_size, Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        # Get camera parameters if available
        intrinsics = None
        extrinsics = None

        if scene_id in self.scene_transforms:
            transforms = self.scene_transforms[scene_id]

            # Find this frame's transform
            img_name = img_path.name
            for frame in transforms.get('frames', []):
                if img_name in frame.get('file_path', ''):
                    extrinsics = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
                    break

            # Intrinsics from transforms
            intrinsics = torch.tensor([
                [transforms.get('fl_x', 500), 0, transforms.get('cx', 256)],
                [0, transforms.get('fl_y', 500), transforms.get('cy', 256)],
                [0, 0, 1]
            ], dtype=torch.float32)

        return {
            'image': image,
            'prompt': prompt,
            'scene_id': scene_id,
            'image_name': img_path.name,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'orig_size': orig_size,
            'has_metric_scale': True,  # ScanNet++ has metric GT
            'has_gt_mask': False  # No per-image masks, use SAM3 predictions
        }
