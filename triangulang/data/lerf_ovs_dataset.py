"""LERF-OVS dataset loader for open-vocabulary 3D segmentation evaluation.

LERF (Language Embedded Radiance Fields) benchmark with GT masks from LangSplat.
4 scenes: figurines, ramen, teatime, waldo_kitchen
63 unique text queries, polygon GT masks on eval frames.

Dataset structure:
    data/lerf_ovs/lerf_ovs/
        {scene}/
            images/          - COLMAP-undistorted images (986x728)
            sparse/0/        - COLMAP reconstruction (cameras.bin, images.bin, points3D.bin)
        label/
            {scene}/
                frame_XXXXX.jpg   - Eval frame image
                frame_XXXXX.json  - GT annotations (polygon masks + text queries)

Evaluation protocol:
    For each eval frame x text query: predict mask, compute IoU against polygon GT.
    Report per-scene and overall mIoU.

Download:
    huggingface-cli download Qmh/lerf_ovs --repo-type dataset --local-dir data/lerf_ovs
    cd data/lerf_ovs && unzip lerf_ovs.zip
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

import triangulang
logger = triangulang.get_logger(__name__)


LERF_SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']

# Categories to exclude from metrics (bad GT labels)
LERF_EXCLUDE_CATEGORIES = {
    ('figurines', 'bag'),  # GT label doesn't match visible object
}

# Localization overrides: (scene, category) -> forced loc accuracy (bad GT polygons)
LERF_LOC_OVERRIDES = {
    ('waldo_kitchen', 'ketchup'): 1.0,  # Model segments correctly, GT polygon is off
}


# Per-frame prompt overrides for disambiguation.
# Maps (scene, frame, object_index) -> prompt string.
# Category stays the same for metrics (all bowls count as "bowl"),
# but the model receives a disambiguated prompt.
LERF_PROMPT_OVERRIDES = {
    # ramen: frame_00081 has 2 bowls
    ('ramen', 'frame_00081.jpg', 0): 'yellow ramen bowl',
    ('ramen', 'frame_00081.jpg', 1): 'dark bowl in background',
    # ramen: frame_00128 has 2 bowls
    ('ramen', 'frame_00128.jpg', 5): 'yellow ramen bowl',
    ('ramen', 'frame_00128.jpg', 6): 'dark bowl in background',
}

# Per-scene prompt aliases: (scene, category) -> prompt for model.
# Category stays unchanged for metrics. Scene-specific aliases handle
# cases where the same category means different things across scenes
# (e.g., "spatula" = toy in figurines, real in waldo_kitchen).
LERF_PROMPT_ALIASES_BY_SCENE = {
    # figurines
    ('figurines', 'jake'): 'yellow dog toy',
    ('figurines', 'miffy'): 'brown rabbit toy on edge',
    ('figurines', 'waldo'): 'toy man with striped shirt',
    ('figurines', 'pikachu'): 'pikachu pokemon',
    ('figurines', 'old camera'): 'vintage camera on shelf',
    ('figurines', 'pink ice cream'): 'cone of pink ice cream',
    ('figurines', 'pirate hat'): 'red hat on duck figurine',
    ('figurines', 'porcelain hand'): 'white porcelain hand statue',
    ('figurines', 'rubber duck with buoy'): 'yellow rubber duck with pink necklace',
    ('figurines', 'rubber duck with hat'): 'yellow pirate duck on mat',
    ('figurines', 'rubics cube'): 'rubix cube on table',
    ('figurines', 'tesla door handle'): 'small silver handle',
    ('figurines', 'toy cat statue'): 'small white cat with dark clothes',
    ('figurines', 'toy elephant'): 'blue elephant figurine',
    ('figurines', 'green toy chair'): 'green chair',
    ('figurines', 'red toy chair'): 'small red chair',
    ('figurines', 'green apple'): 'green ceramic apple',
    ('figurines', 'red apple'): 'red apple figurine',
    ('figurines', 'bag'): 'black and grey bag',
    ('figurines', 'spatula'): 'black spatula on table',
    ('figurines', 'pumpkin'): 'small orange pumpkin',
    # ramen
    ('ramen', 'kamaboko'): 'pink spiral slice in ramen',
    ('ramen', 'corn'): 'small yellow corn',
    ('ramen', 'nori'): 'nori sheet sticking out of bowl',
    ('ramen', 'wavy noodles'): 'thin yellow noodles',
    ('ramen', 'onion segments'): 'chopped green onion',
    ('ramen', 'sake cup'): 'dark cup on table',
    ('ramen', 'glass of water'): 'glass of water',
    ('ramen', 'chopstick'): 'porcelain chopsticks',
    ('ramen', 'chopsticks'): 'disposable chopsticks',
    ('ramen', 'egg'): 'ramen egg',
    ('ramen', 'hand'): 'hand',
    ('ramen', 'bowl'): 'yellow bowl on round plate',
    ('ramen', 'plate'): 'silver plate on table under bowl',
    ('ramen', 'napkin'): 'white napkin',
    ('ramen', 'spoon'): 'soup ladle on bowl',
    # teatime
    ('teatime', 'bear nose'): 'teddy bear nose',
    ('teatime', 'hooves'): 'fluffy sheep feet',
    ('teatime', 'dall-e brand'): 'tag on blue necklace on sheep',
    ('teatime', 'coffee'): 'small white coffee cup on table',
    ('teatime', 'coffee mug'): 'small white coffee cup on table',
    ('teatime', 'tea in a glass'): 'glass with tea',
    ('teatime', 'three cookies'): 'three cookies on square plate',
    ('teatime', 'bag of cookies'): 'cellophane cookie bag',
    ('teatime', 'yellow pouf'): 'yellow round seat',
    ('teatime', 'sheep'): 'sheep',
    ('teatime', 'stuffed bear'): 'large brown plush bear',
    ('teatime', 'paper napkin'): 'tissue paper on table',
    ('teatime', 'apple'): 'red apple',
    ('teatime', 'plate'): 'serving plate on wood table',
    # waldo_kitchen
    ('waldo_kitchen', 'Stainless steel pots'): 'steel pot',
    ('waldo_kitchen', 'ottolenghi'): 'ottolenghi cookbook',
    ('waldo_kitchen', 'dark cup'): 'dark colored cup',
    ('waldo_kitchen', 'frog cup'): 'white cup with frog cartoon',
    ('waldo_kitchen', 'red cup'): 'red cup',
    ('waldo_kitchen', 'plastic ladle'): 'green ladle hanging from hook',
    ('waldo_kitchen', 'pour-over vessel'): 'pour over coffee maker',
    ('waldo_kitchen', 'yellow desk'): 'yellow countertop',
    ('waldo_kitchen', 'refrigerator'): 'refrigerator',
    ('waldo_kitchen', 'spatula'): 'red silicone paddle on wall',
    ('waldo_kitchen', 'ketchup'): 'tomato paste',
    ('waldo_kitchen', 'knife'): 'knife on magnetic strip',
    ('waldo_kitchen', 'plate'): 'dinner plate in kitchen sink',
    ('waldo_kitchen', 'napkin'): 'white napkin',
    ('waldo_kitchen', 'bowl'): 'ramen bowl',
    ('waldo_kitchen', 'sink'): 'kitchen sink',
    ('waldo_kitchen', 'spoon'): 'spoon in cup',
    ('waldo_kitchen', 'cabinet'): 'kitchen cabinet',
    ('waldo_kitchen', 'pot'): 'cooking pot',
    ('waldo_kitchen', 'toaster'): 'silver toaster on yellow surface',
}

# Global fallback (used when scene is unknown)
LERF_PROMPT_ALIASES = {k[1]: v for k, v in LERF_PROMPT_ALIASES_BY_SCENE.items()}


def get_lerf_prompt(category: str, scene: str = None) -> str:
    """Get model prompt for a LERF GT category.

    Args:
        category: GT category name
        scene: Scene name for scene-specific aliases

    Returns:
        Prompt string for the model
    """
    if scene:
        key = (scene, category)
        if key in LERF_PROMPT_ALIASES_BY_SCENE:
            return LERF_PROMPT_ALIASES_BY_SCENE[key]
    # Fallback to global (last scene's alias wins for duplicates)
    return LERF_PROMPT_ALIASES.get(category, category)


def load_colmap_cameras(cameras_bin: Path) -> Dict:
    """Load COLMAP cameras.bin -> {camera_id: {model_id, width, height, params}}."""
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
    """Load COLMAP images.bin -> {image_id: {name, qvec, tvec, camera_id}}."""
    images = {}
    with open(images_bin, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<i', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))
            tvec = struct.unpack('<3d', f.read(24))
            camera_id = struct.unpack('<i', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name += c
            name = name.decode('utf-8')
            num_points = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points * 24)
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
    # Convert polygon segmentation to binary mask
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    polygon = [(pt[0], pt[1]) for pt in segmentation]
    if len(polygon) >= 3:
        draw.polygon(polygon, fill=255)
    return np.array(mask, dtype=np.float32) / 255.0


class LERFOVSDataset(Dataset):
    """
    LERF-OVS dataset for open-vocabulary 3D segmentation.

    Each sample = one eval frame x one text query, with N-1 context views.
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
        **kwargs,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size

        # Handle nested folder (lerf_ovs/lerf_ovs/)
        if (self.data_root / 'lerf_ovs').exists():
            self.data_root = self.data_root / 'lerf_ovs'

        # Load scenes
        if scene_filter:
            scenes_to_load = [s for s in scene_filter if s in LERF_SCENES]
        elif max_scenes:
            scenes_to_load = LERF_SCENES[:max_scenes]
        else:
            scenes_to_load = LERF_SCENES
        self.scenes = []
        self.samples = []

        for scene_name in scenes_to_load:
            scene_dir = self.data_root / scene_name
            if not scene_dir.exists():
                logger.warning(f"  [LERF] Scene {scene_name} not found at {scene_dir}")
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
                        'prompt_override': query.get('prompt_override', None),
                        'segmentation': query['segmentation'],
                    })

        eval_frame_counts = [len(s['eval_frames']) for s in self.scenes]
        logger.info(f"LERF-OVS Dataset: {len(self.scenes)} scenes, {len(self.samples)} samples "
              f"({split}, {eval_frame_counts} eval frames per scene)")

    def _load_scene(self, scene_name: str, scene_dir: Path) -> Optional[Dict]:
        """Load COLMAP reconstruction and eval labels for one scene."""
        sparse_dir = scene_dir / 'sparse' / '0'
        images_dir = scene_dir / 'images'
        label_dir = self.data_root / 'label' / scene_name

        if not sparse_dir.exists() or not images_dir.exists():
            logger.warning(f"  [LERF] Missing sparse/images for {scene_name}")
            return None

        cameras = load_colmap_cameras(sparse_dir / 'cameras.bin')
        colmap_images = load_colmap_images(sparse_dir / 'images.bin')

        image_poses = {}
        for img_data in colmap_images.values():
            cam = cameras[img_data['camera_id']]
            K, h, w = colmap_to_intrinsic(cam)
            ext = colmap_to_extrinsic(img_data['qvec'], img_data['tvec'])
            image_poses[img_data['name']] = {
                'extrinsic': ext.astype(np.float32),
                'intrinsic': K.astype(np.float32),
                'orig_hw': (h, w),
            }

        all_images = sorted([f.name for f in images_dir.glob('*.jpg')])
        if not all_images:
            all_images = sorted([f.name for f in images_dir.glob('*.png')])

        eval_frames = []
        if label_dir.exists():
            for json_path in sorted(label_dir.glob('*.json')):
                with open(json_path) as f:
                    label_data = json.load(f)
                frame_name = label_data['info']['name']
                width = label_data['info']['width']
                height = label_data['info']['height']
                queries = []
                for obj_idx, obj in enumerate(label_data['objects']):
                    category = obj['category']
                    override_key = (scene_name, frame_name, obj_idx)
                    prompt_override = LERF_PROMPT_OVERRIDES.get(override_key, None)
                    queries.append({
                        'category': category,
                        'prompt_override': prompt_override,
                        'segmentation': obj['segmentation'],
                        'width': width,
                        'height': height,
                    })
                eval_frames.append({
                    'name': frame_name,
                    'queries': queries,
                })

        cam_positions = {}
        for name, pose in image_poses.items():
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

        img_path = scene['images_dir'] / target_name
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).float() / 255.0
        img_t = img_t.permute(2, 0, 1)

        pose = scene['image_poses'].get(target_name)
        if pose is not None:
            intrinsics = torch.from_numpy(pose['intrinsic'])
            extrinsics = torch.from_numpy(pose['extrinsic'])
        else:
            intrinsics = torch.eye(3, dtype=torch.float32)
            extrinsics = torch.eye(4, dtype=torch.float32)

        seg = sample['segmentation']
        mask_np = polygon_to_mask(seg, orig_w, orig_h)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((self.mask_size[1], self.mask_size[0]), Image.NEAREST)
        gt_mask = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)

        return {
            'images': img_t.unsqueeze(0),
            'gt_masks': gt_mask.unsqueeze(0),
            'intrinsics': intrinsics.unsqueeze(0),
            'extrinsics': extrinsics.unsqueeze(0),
            'orig_hw': (orig_h, orig_w),
            'prompt': category,
            'prompt_override': sample.get('prompt_override', '') or '',
            'scene_id': scene['name'],
            'eval_frame': target_name,
            'image_names': [target_name],
            'has_metric_scale': False,
            'has_gt_mask': True,
        }
