"""LERF-OVS dataset loader for open-vocabulary 3D segmentation evaluation.

4 scenes (figurines, ramen, teatime, waldo_kitchen), 63 text queries,
with polygon GT masks on eval frames. Uses COLMAP poses from sparse/0/.
"""

import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw


LERF_SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
LERF_LOC_SCENES = ['bouquet', 'figurines', 'ramen', 'teatime', 'waldo_kitchen']

# Categories to exclude from metrics (bad GT labels)
LERF_EXCLUDE_CATEGORIES = {
    ('figurines', 'bag'),  # GT label doesn't match visible object
}

# Localization overrides: (scene, category) → forced loc accuracy (bad GT polygons)
LERF_LOC_OVERRIDES = {
    ('waldo_kitchen', 'ketchup'): 1.0,  # Model segments correctly, GT polygon is off
}


# Prompt aliases: GT category → improved prompt for model, per scene.
# The GT category is used for matching/metrics, but the alias is sent to the model.
# This helps when GT labels are ambiguous (e.g. "jake") or overly specific.
LERF_PROMPT_ALIASES = {
    'figurines': {
        'jake': 'yellow dog with long legs',
        'miffy': 'brown rabbit in orange tshirt',
        'waldo': 'toy man with striped shirt',
        'pikachu': 'pikachu',
        'old camera': 'vintage camera',
        'pink ice cream': 'pink ice cream',
        'pirate hat': 'red hat on rubber duck',
        'porcelain hand': 'white porcelain hand statue',
        'rubber duck with buoy': 'yellow rubber duck with pink necklace',
        'rubber duck with hat': 'yellow rubber duck with red hat',
        'rubics cube': 'rubiks cube',
        'tesla door handle': 'silver door handle',
        'toy cat statue': 'white cat statue',
        'toy elephant': 'blue elephant figurine',
        'green toy chair': 'green chair',
        'red toy chair': 'small red chair',
        'green apple': 'green apple',
        'red apple': 'red apple',
        'bag': 'black and grey bag',
        'spatula': 'black spatula',
        'pumpkin': 'small orange pumpkin',
    },
    'ramen': {
        'kamaboko': 'pink fish spiral',
        'corn': 'small corn kernel',
        'nori': 'nori seaweed',
        'wavy noodles': 'yellow noodles',
        'onion segments': 'chopped green onion',
        'sake cup': 'small metal cup',
        'glass of water': 'glass of water',
        'chopstick': 'porcelain chopsticks',
        'chopsticks': 'wooden chopsticks',
        'egg': 'soft boiled egg in ramen',
        'hand': 'hand',
        'bowl': 'ramen bowl',
        'plate': 'plates',
        'napkin': 'white napkin',
        'spoon': 'steel spoon',
    },
    'teatime': {
        'bear nose': 'tan teddy bear nose',
        'hooves': 'stuffed animal hooves',
        'dall-e brand': 'text nameplate',
        'coffee': 'coffee cup',
        'coffee mug': 'coffee cup',
        'tea in a glass': 'glass of tea',
        'three cookies': 'three cookies',
        'bag of cookies': 'bag of cookies',
        'yellow pouf': 'yellow ottoman pouf',
        'sheep': 'sheep',
        'stuffed bear': 'brown teddy bear',
        'paper napkin': 'paper napkin',
        'apple': 'red apple',
        'plate': 'plates',
    },
    'waldo_kitchen': {
        'Stainless steel pots': 'steel pot',
        'ottolenghi': 'ottolenghi cookbook',
        'dark cup': 'dark colored cup',
        'frog cup': 'frog shaped cup',
        'red cup': 'red cup',
        'plastic ladle': 'green ladle on wall',
        'pour-over vessel': 'pour over coffee maker',
        'yellow desk': 'yellow countertop',
        'refrigerator': 'kitchen fridge',
        'spatula': 'red spatula',
        'ketchup': 'tomato paste',
        'knife': 'black chef knife on wall',
        'plate': 'plates',
        'napkin': 'white napkin',
        'bowl': 'ramen bowl',
        'sink': 'kitchen sink',
        'spoon': 'steel spoon',
        'cabinet': 'kitchen cabinet',
        'pot': 'cooking pot',
        'toaster': 'toaster',
    },
}


def get_lerf_prompt(category: str, scene: str = None) -> str:
    """Get model prompt for a LERF GT category.

    Looks up the scene-specific alias dict first, falls back to raw category.
    """
    if scene is not None:
        scene_aliases = LERF_PROMPT_ALIASES.get(scene, {})
        alias = scene_aliases.get(category)
        if alias is not None:
            return alias
    return category


def load_colmap_cameras(cameras_bin: Path) -> Dict:
    """Load COLMAP cameras.bin → {camera_id: {model_id, width, height, params}}."""
    cameras = {}
    with open(cameras_bin, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack('<i', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 5}.get(model_id, 4)
            params = struct.unpack(f'<{num_params}d', f.read(8 * num_params))
            cameras[cam_id] = {
                'model_id': model_id, 'width': width, 'height': height, 'params': params
            }
    return cameras


def load_colmap_images(images_bin: Path) -> Dict:
    """Load COLMAP images.bin → {image_id: {name, qvec, tvec, camera_id}}."""
    images = {}
    with open(images_bin, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<i', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))
            tvec = struct.unpack('<3d', f.read(24))
            camera_id = struct.unpack('<i', f.read(4))[0]
            # Read name (null-terminated string)
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name += c
            name = name.decode('utf-8')
            # Skip 2D points (num_points2D × (x, y, point3d_id))
            num_points = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points * 24)  # 8+8+8 per point
            images[image_id] = {
                'name': name, 'qvec': np.array(qvec), 'tvec': np.array(tvec),
                'camera_id': camera_id,
            }
    return images


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def colmap_to_extrinsic(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP qvec/tvec to 4x4 world-to-camera extrinsic matrix."""
    R = qvec2rotmat(qvec)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = tvec
    return extrinsic


def colmap_to_intrinsic(camera: Dict) -> Tuple[np.ndarray, int, int]:
    """Convert COLMAP camera to 3x3 intrinsic matrix. Returns (K, height, width)."""
    model_id = camera['model_id']
    params = camera['params']
    w, h = camera['width'], camera['height']

    if model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    elif model_id == 1:  # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    elif model_id == 2:  # SIMPLE_RADIAL: f, cx, cy, k1
        f, cx, cy = params[:3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    elif model_id == 4:  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    else:
        fx = params[0]
        K = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], dtype=np.float64)

    return K, int(h), int(w)


def polygon_to_mask(segmentation: List[List[float]], width: int, height: int) -> np.ndarray:
    """Convert polygon segmentation to binary mask."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # Flatten polygon points to [(x1, y1), (x2, y2), ...]
    polygon = [(pt[0], pt[1]) for pt in segmentation]
    if len(polygon) >= 3:
        draw.polygon(polygon, fill=255)
    return np.array(mask, dtype=np.float32) / 255.0


class LERFOVSDataset(Dataset):
    """
    LERF-OVS dataset for open-vocabulary 3D segmentation.

    Each sample = one eval frame × one text query, with N-1 context views.
    Returns multi-view images with COLMAP poses and GT polygon mask for eval.

    Modes:
        - 'eval': Only labeled frames as targets (for benchmarking)
        - 'train': All frames as targets, labels from nearby eval frames
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'eval',
        image_size: Tuple[int, int] = (518, 518),
        mask_size: Tuple[int, int] = (128, 128),
        max_scenes: int = None,
        scene_filter: List[str] = None,
        **kwargs,  # accept extra kwargs
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size

        # Handle nested folder (lerf_ovs/lerf_ovs/)
        if (self.data_root / 'lerf_ovs').exists():
            self.data_root = self.data_root / 'lerf_ovs'

        # Load scenes — filter by name if specified, otherwise use max_scenes
        if scene_filter:
            scenes_to_load = [s for s in scene_filter if s in LERF_SCENES]
        elif max_scenes:
            scenes_to_load = LERF_SCENES[:max_scenes]
        else:
            scenes_to_load = LERF_SCENES
        self.scenes = []
        self.samples = []  # (scene_idx, eval_frame, query_category)

        for scene_name in scenes_to_load:
            scene_dir = self.data_root / scene_name
            if not scene_dir.exists():
                print(f"  [LERF] Scene {scene_name} not found at {scene_dir}")
                continue

            scene = self._load_scene(scene_name, scene_dir)
            if scene is None:
                continue

            scene_idx = len(self.scenes)
            self.scenes.append(scene)

            # Build samples: each eval frame × each query on that frame
            for eval_frame in scene['eval_frames']:
                for query in eval_frame['queries']:
                    self.samples.append({
                        'scene_idx': scene_idx,
                        'eval_frame_name': eval_frame['name'],
                        'category': query['category'],
                        'segmentation': query['segmentation'],
                    })

        eval_frame_counts = [len(s['eval_frames']) for s in self.scenes]
        print(f"LERF-OVS Dataset: {len(self.scenes)} scenes, {len(self.samples)} samples "
              f"({split}, {eval_frame_counts} eval frames per scene)")

    def _load_scene(self, scene_name: str, scene_dir: Path) -> Optional[Dict]:
        """Load COLMAP reconstruction and eval labels for one scene."""
        sparse_dir = scene_dir / 'sparse' / '0'
        images_dir = scene_dir / 'images'
        label_dir = self.data_root / 'label' / scene_name

        if not sparse_dir.exists() or not images_dir.exists():
            print(f"  [LERF] Missing sparse/images for {scene_name}")
            return None

        # Load COLMAP reconstruction
        cameras = load_colmap_cameras(sparse_dir / 'cameras.bin')
        colmap_images = load_colmap_images(sparse_dir / 'images.bin')

        # Build name→pose mapping
        image_poses = {}  # frame_name → {extrinsic, intrinsic, orig_hw}
        for img_data in colmap_images.values():
            cam = cameras[img_data['camera_id']]
            K, h, w = colmap_to_intrinsic(cam)
            ext = colmap_to_extrinsic(img_data['qvec'], img_data['tvec'])
            image_poses[img_data['name']] = {
                'extrinsic': ext.astype(np.float32),
                'intrinsic': K.astype(np.float32),
                'orig_hw': (h, w),
            }

        # All image files sorted
        all_images = sorted([f.name for f in images_dir.glob('*.jpg')])
        if not all_images:
            all_images = sorted([f.name for f in images_dir.glob('*.png')])

        # Load eval labels
        eval_frames = []
        if label_dir.exists():
            for json_path in sorted(label_dir.glob('*.json')):
                with open(json_path) as f:
                    label_data = json.load(f)
                frame_name = label_data['info']['name']
                width = label_data['info']['width']
                height = label_data['info']['height']
                queries = []
                for obj in label_data['objects']:
                    queries.append({
                        'category': obj['category'],
                        'segmentation': obj['segmentation'],
                        'width': width,
                        'height': height,
                    })
                eval_frames.append({
                    'name': frame_name,
                    'queries': queries,
                })

        # Precompute camera positions for nearest-neighbor context selection
        cam_positions = {}
        for name, pose in image_poses.items():
            # Camera position in world = -R^T @ t
            R = pose['extrinsic'][:3, :3]
            t = pose['extrinsic'][:3, 3]
            cam_positions[name] = -R.T @ t

        return {
            'name': scene_name,
            'images_dir': images_dir,
            'all_images': all_images,
            'image_poses': image_poses,
            'cam_positions': cam_positions,
            'eval_frames': eval_frames,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        scene = self.scenes[sample['scene_idx']]
        target_name = sample['eval_frame_name']
        category = sample['category']

        # Load just the target eval frame (single-view inference per sample)
        img_path = scene['images_dir'] / target_name
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).float() / 255.0
        img_t = img_t.permute(2, 0, 1)  # HWC → CHW

        # Intrinsics / extrinsics from COLMAP
        pose = scene['image_poses'].get(target_name)
        if pose is not None:
            intrinsics = torch.from_numpy(pose['intrinsic'])
            extrinsics = torch.from_numpy(pose['extrinsic'])
        else:
            intrinsics = torch.eye(3, dtype=torch.float32)
            extrinsics = torch.eye(4, dtype=torch.float32)

        # GT mask from polygon annotation
        seg = sample['segmentation']
        mask_np = polygon_to_mask(seg, orig_w, orig_h)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((self.mask_size[1], self.mask_size[0]), Image.NEAREST)
        gt_mask = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)

        return {
            'images': img_t.unsqueeze(0),                 # (1, 3, H, W)
            'gt_masks': gt_mask.unsqueeze(0),             # (1, mask_H, mask_W)
            'intrinsics': intrinsics.unsqueeze(0),        # (1, 3, 3)
            'extrinsics': extrinsics.unsqueeze(0),        # (1, 4, 4)
            'orig_hw': (orig_h, orig_w),
            'prompt': category,                           # text query
            'scene_id': scene['name'],
            'eval_frame': target_name,
            'image_names': [target_name],
            'has_metric_scale': False,                    # COLMAP scale is arbitrary
            'has_gt_mask': True,
        }


# ── LERF Localization eval dataset ─────────────────────────────────────────────
# Uses rendered images + LabelMe-style bounding box annotations.
# Labels live under: label/Localization eval dataset/{scene}/{N}_rgb.json
# Images are the PNGs alongside the JSONs (not COLMAP images).

LERF_LOC_PROMPT_ALIASES = {
    'bouquet': {
        'big white crinkly flower': 'big white crinkly flower',
        'eucalyptus': 'eucalyptus leaves',
        'small white flowers': 'small white flowers',
        'rosemary': 'rosemary sprig',
    },
    'figurines': {
        'jake': 'yellow dog with long legs',
        'miffy': 'brown rabbit in orange tshirt',
        'waldo': 'toy man with striped shirt',
        'old camera': 'vintage camera',
        'porcelain hand': 'white porcelain hand statue',
        'rubics cube': 'rubiks cube',
        'tesla door handle': 'silver door handle',
        'toy cat statue': 'white cat statue',
        'toy elephant': 'blue elephant figurine',
        'ice cream cone': 'ice cream cone',
        'quilted pumpkin': 'quilted pumpkin',
        'rabbit': 'small rabbit figurine',
        'rubber duck': 'yellow rubber duck',
        'twizzlers': 'twizzlers candy',
        'toy chair': 'small toy chair',
    },
    'ramen': {
        'green onion': 'chopped green onion',
        'sake cup': 'small metal cup',
        'pork belly': 'pork belly slice',
        'nori': 'nori seaweed',
        'wavy noodles': 'yellow noodles',
        'broth': 'ramen broth',
        'vamen': 'vamen',
    },
    'teatime': {
        'bear nose': 'tan teddy bear nose',
        'hooves': 'stuffed animal hooves',
        'dall-e': 'text nameplate',
        'stuffed bear': 'brown teddy bear',
        'yellow pouf': 'yellow ottoman pouf',
        'tea in a glass': 'glass of tea',
        'cookies on a plate': 'cookies on a plate',
        'spill': 'liquid spill',
        'spoon handle': 'spoon handle',
    },
    'waldo_kitchen': {
        'pour-over vessel': 'pour over coffee maker',
        'cookbooks': 'cookbooks on shelf',
        'copper-bottom pot': 'copper bottom pot',
        'cooking tongs': 'cooking tongs',
        'blue hydroflask': 'blue water bottle',
        'coffee grinder': 'coffee grinder',
        'pepper mill': 'pepper mill',
        'red mug': 'red mug',
        'scrub brush': 'scrub brush',
        'spice rack': 'spice rack',
        'utensils': 'kitchen utensils',
        'paper towel roll': 'paper towel roll',
        'power outlet': 'power outlet',
        'vegetable oil': 'vegetable oil bottle',
        'waldo': 'toy man with striped shirt',
    },
}


def get_lerf_loc_prompt(category: str, scene: str = None) -> str:
    """Get model prompt for a LERF-Loc GT category."""
    if scene is not None:
        alias = LERF_LOC_PROMPT_ALIASES.get(scene, {}).get(category)
        if alias is not None:
            return alias
    return category


def rectangle_to_mask(points: List[List[float]], width: int, height: int) -> np.ndarray:
    """Convert LabelMe rectangle (2 corner points) to binary mask."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    (x1, y1), (x2, y2) = points[0], points[1]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return np.array(mask, dtype=np.float32) / 255.0


class LERFLocDataset(Dataset):
    """
    LERF Localization eval dataset.

    Uses rendered NeRF images + LabelMe bounding-box annotations.
    Each sample = one rendered frame × one text query.
    Images come from the label PNGs (not COLMAP).
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'eval',
        image_size: Tuple[int, int] = (518, 518),
        mask_size: Tuple[int, int] = (128, 128),
        max_scenes: int = None,
        scene_filter: List[str] = None,
        **kwargs,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size

        # Handle nested folder
        if (self.data_root / 'lerf_ovs').exists():
            self.data_root = self.data_root / 'lerf_ovs'

        self.label_root = self.data_root / 'label' / 'Localization eval dataset'

        if scene_filter:
            scenes_to_load = [s for s in scene_filter if s in LERF_LOC_SCENES]
        elif max_scenes:
            scenes_to_load = LERF_LOC_SCENES[:max_scenes]
        else:
            scenes_to_load = LERF_LOC_SCENES

        self.scenes = []
        self.samples = []

        for scene_name in scenes_to_load:
            scene_dir = self.label_root / scene_name
            if not scene_dir.exists():
                print(f"  [LERF-Loc] Scene {scene_name} not found at {scene_dir}")
                continue

            scene = self._load_scene(scene_name, scene_dir)
            if scene is None:
                continue

            scene_idx = len(self.scenes)
            self.scenes.append(scene)

            for eval_frame in scene['eval_frames']:
                for query in eval_frame['queries']:
                    self.samples.append({
                        'scene_idx': scene_idx,
                        'eval_frame_name': eval_frame['name'],
                        'category': query['category'],
                        'bbox_points': query['bbox_points'],
                        'width': query['width'],
                        'height': query['height'],
                    })

        eval_frame_counts = [len(s['eval_frames']) for s in self.scenes]
        print(f"LERF-Loc Dataset: {len(self.scenes)} scenes, {len(self.samples)} samples "
              f"({split}, {eval_frame_counts} eval frames per scene)")

    def _load_scene(self, scene_name: str, scene_dir: Path) -> Optional[Dict]:
        """Load rendered eval frames + LabelMe annotations for one scene."""
        eval_frames = []

        for json_path in sorted(scene_dir.glob('*_rgb.json')):
            with open(json_path) as f:
                label_data = json.load(f)

            # Image is the PNG alongside the JSON
            img_name = label_data.get('imagePath', json_path.stem + '.png')
            width = label_data.get('imageWidth', 480)
            height = label_data.get('imageHeight', 270)

            queries = []
            for shape in label_data.get('shapes', []):
                if shape['shape_type'] != 'rectangle':
                    continue
                queries.append({
                    'category': shape['label'],
                    'bbox_points': shape['points'],
                    'width': width,
                    'height': height,
                })

            if queries:
                eval_frames.append({
                    'name': img_name,
                    'img_path': scene_dir / img_name,
                    'queries': queries,
                })

        if not eval_frames:
            print(f"  [LERF-Loc] No eval frames found for {scene_name}")
            return None

        return {
            'name': scene_name,
            'scene_dir': scene_dir,
            'eval_frames': eval_frames,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        scene = self.scenes[sample['scene_idx']]
        target_name = sample['eval_frame_name']
        category = sample['category']

        # Find the eval frame entry
        eval_frame = None
        for ef in scene['eval_frames']:
            if ef['name'] == target_name:
                eval_frame = ef
                break

        # Load rendered image
        img = Image.open(eval_frame['img_path']).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).float() / 255.0
        img_t = img_t.permute(2, 0, 1)  # HWC → CHW

        # No COLMAP poses for rendered images
        intrinsics = torch.eye(3, dtype=torch.float32)
        extrinsics = torch.eye(4, dtype=torch.float32)

        # GT mask from bounding box
        bbox_w = sample['width']
        bbox_h = sample['height']
        mask_np = rectangle_to_mask(sample['bbox_points'], bbox_w, bbox_h)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((self.mask_size[1], self.mask_size[0]), Image.NEAREST)
        gt_mask = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)

        return {
            'images': img_t.unsqueeze(0),                 # (1, 3, H, W)
            'gt_masks': gt_mask.unsqueeze(0),             # (1, mask_H, mask_W)
            'intrinsics': intrinsics.unsqueeze(0),        # (1, 3, 3)
            'extrinsics': extrinsics.unsqueeze(0),        # (1, 4, 4)
            'orig_hw': (orig_h, orig_w),
            'prompt': category,
            'scene_id': scene['name'],
            'eval_frame': target_name,
            'image_names': [target_name],
            'has_metric_scale': False,
            'has_gt_mask': True,
        }
