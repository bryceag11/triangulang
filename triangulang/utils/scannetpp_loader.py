"""ScanNet++ multi-view dataset loader.

Loads RGB images, depth, masks, and camera parameters for training and evaluation.
Includes label normalization (LABEL_FIXES) and cached sample discovery.

Helpers split into sub-modules:
  - scannetpp_label_fixes.json  : LABEL_FIXES dict (label typo/variant corrections)
  - scannetpp_io.py             : I/O helpers, SCANNETPP_PROMPTS, ScanNetPPDataset
  - scannetpp_rasterization.py  : 3D geometry / mesh rasterization, SceneRasterizer
"""
import json
import pickle
import hashlib
import fcntl
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, OrderedDict, defaultdict
import random
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

import triangulang
logger = triangulang.get_logger(__name__)

def _is_main_process():
    """Check if this is the main process (rank 0) for DDP-safe printing."""
    return not dist.is_initialized() or dist.get_rank() == 0

# Import spatial context types (for GT-aware spatial augmentation)
try:
    from triangulang.utils.spatial_reasoning import (
        SpatialContext,
        InstanceSpatialInfo,
        build_spatial_context,
        compute_instance_spatial_info
    )
    HAS_SPATIAL_CONTEXT = True
except ImportError:
    HAS_SPATIAL_CONTEXT = False
    SpatialContext = None

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# Labels to skip during evaluation and training (structural/ambiguous categories)
SCANNETPP_SKIP_LABELS = frozenset({
    'remove', 'split', 'object', 'objects', 'stuff', 'unknown',
    'wall', 'floor', 'ceiling', 'door', 'window', 'doorframe',
    'windowframe', 'window frame',
    'reflection', 'mirror', 'structure',
})

# Label normalization: Fix typos and inconsistencies in ScanNet++ annotations
# Loaded from scannetpp_label_fixes.json (comments stripped; see git history for rationale)
LABEL_FIXES: Dict[str, str] = json.load(
    open(Path(__file__).parent / 'scannetpp_label_fixes.json')
)

# Bad annotations to exclude: (scene_id, object_id) pairs with incorrect masks
# Detected via segment count outlier analysis (objects with >100x median segment count)
BAD_ANNOTATIONS = {
    ('3db0a1c8f3', 86),   # 'light switch' has 24371 segs (median=142) - clearly wrong
    ('3db0a1c8f3', 54),   # 'remove' with 164922 segs
    ('ab11145646', 96),   # 'remove' with 119930 segs
    ('cf1ffd871d', 26),   # 'remove' with 116861 segs
}

# Frames to exclude per scene: {scene_id: {frame_stem, ...}}
# These frames have catastrophic DA3 depth errors (e.g. 609cm Procrustes error)
EXCLUDE_FRAMES = {
    'c4c04e6d6c': {'DSC03071'},  # 609cm Procrustes error, rest of scene is 3.7cm
}

def is_excluded_frame(scene_id: str, frame_stem: str) -> bool:
    """Check if a frame should be excluded due to known DA3 depth errors."""
    excluded = EXCLUDE_FRAMES.get(scene_id)
    return excluded is not None and frame_stem in excluded

def is_bad_annotation(scene_id: str, obj_id: int) -> bool:
    """Check if an annotation should be excluded due to known errors."""
    return (scene_id, obj_id) in BAD_ANNOTATIONS

def normalize_label(label: str) -> str:
    """Fix typos and normalize labels from ScanNet++ annotations."""
    label = label.strip()
    # Collapse double (or more) spaces to single space
    while '  ' in label:
        label = label.replace('  ', ' ')
    # Strip trailing brackets/artifacts
    label = label.rstrip(']').rstrip('[').strip()
    return LABEL_FIXES.get(label, label)

# Add scannetpp_toolkit to path for utilities
_toolkit_path = Path(__file__).parent.parent.parent / "scannetpp_toolkit"
if _toolkit_path.exists() and str(_toolkit_path) not in sys.path:
    sys.path.insert(0, str(_toolkit_path))

def get_scenes_dir(data_root: Path) -> Path:
    """Get the directory containing scene folders (handles nested 'data' folder)."""
    # ScanNet++ download creates: data_root/data/<scene_id>/
    nested = data_root / "data"
    if nested.exists() and nested.is_dir():
        return nested
    return data_root

# Deferred imports from sub-modules (avoids circular imports at module load)

def _get_io():
    """Lazy import of scannetpp_io to avoid circular imports."""
    from triangulang.utils import scannetpp_io
    return scannetpp_io

# Multi-View Dataset with Supervised Training Support

class ScanNetPPMultiViewDataset(Dataset):
    """
    Dataset for ScanNet++ multi-view training.
    Each sample is multiple views of the same indoor scene.

    Key advantage: METRIC SCALE ground truth from LiDAR.

    With supervised=True, uses pre-computed obj_ids from ScanNet++ toolkit:
    - Dense GT masks from mesh rasterization (pre-computed)
    - Objects filtered by min visibility (0.1% of image pixels)
    - Ready for focal + dice loss training

    Pre-computed obj_ids are loaded from:
        data_root/semantics_2d_train/obj_ids/{scene_id}/{image_name}.pth
    """

    def __init__(
        self,
        data_root: Path,
        split: str = 'nvs_sem_train',
        views_per_sample: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        mask_size: Tuple[int, int] = (128, 128),
        use_undistorted: bool = True,
        max_scenes: int = None,
        sampling_strategy: str = 'stratified',
        da3_chunk_size: int = 8,
        supervised: bool = True,
        min_object_pixels: float = 0.001,
        raster_cache_dir: Path = None,
        obj_ids_dir: str = None,
        samples_per_scene: int = 10,
        enumerate_all_objects: bool = True,
        semantic_union: bool = True,
        use_cached_depth: bool = False,
        da3_cache_name: str = 'da3_cache',
        min_category_samples: int = 1,
        exclude_categories: List[str] = None,
        include_categories: List[str] = None,
        num_objects_per_sample: int = 1,
        use_cached_pi3x: bool = False,
        pi3x_cache_name: str = 'ma_cache_train',
    ):
        self.data_root = Path(data_root)
        self.min_category_samples = min_category_samples
        self.exclude_categories = set(exclude_categories) if exclude_categories else set()
        self.include_categories = set(include_categories) if include_categories else None
        self.scenes_dir = get_scenes_dir(self.data_root)
        self.image_size = image_size
        self.mask_size = mask_size
        self.views_per_sample = views_per_sample
        self.use_undistorted = use_undistorted
        self.sampling_strategy = sampling_strategy
        self.da3_chunk_size = da3_chunk_size
        self.enumerate_all_objects = enumerate_all_objects
        self.supervised = supervised
        self.min_object_pixels = min_object_pixels
        self.raster_cache_dir = Path(raster_cache_dir) if raster_cache_dir else None
        self.samples_per_scene = samples_per_scene
        self.semantic_union = semantic_union
        self.num_objects_per_sample = num_objects_per_sample
        self.use_cached_depth = use_cached_depth
        self.da3_cache_name = da3_cache_name
        self.da3_cache_dir = self.data_root / da3_cache_name if use_cached_depth else None
        if use_cached_depth and self.da3_cache_dir and not self.da3_cache_dir.exists():
            if _is_main_process():
                logger.warning(f"DA3 cache directory not found: {self.da3_cache_dir}")
                logger.warning(f"  Run scripts/preprocess_da3.py or preprocess_da3_nested.py first.")
        self.use_cached_pi3x = use_cached_pi3x
        self.pi3x_cache_dir = self.data_root / pi3x_cache_name if use_cached_pi3x else None
        if use_cached_pi3x and self.pi3x_cache_dir and not self.pi3x_cache_dir.exists():
            if _is_main_process():
                logger.warning(f"PI3X cache directory not found: {self.pi3x_cache_dir}")
                logger.warning(f"  Run MapAnything caching script first.")

        self.centroid_cache = {}
        centroid_cache_path = self.data_root / "centroid_cache.json"
        if centroid_cache_path.exists():
            with open(centroid_cache_path) as f:
                self.centroid_cache = json.load(f)
            if _is_main_process():
                logger.info(f"Loaded GT centroid cache: {len(self.centroid_cache)} scenes")

        if obj_ids_dir:
            self.obj_ids_root = self.data_root / obj_ids_dir
        elif 'train' in split:
            self.obj_ids_root = self.data_root / "semantics_2d_train"
        else:
            self.obj_ids_root = self.data_root / "semantics_2d_val"

        io = _get_io()
        scenes = io.get_available_scenes(self.data_root, split)
        if max_scenes:
            scenes = scenes[:max_scenes]
        self.scenes = []
        self.rasterizers = OrderedDict()  # Legacy - kept for compatibility
        if _is_main_process():
            logger.info(f"Loading ScanNet++ multi-view dataset ({split}, supervised={supervised})...")
        from tqdm import tqdm
        skipped = 0

        for scene_id in tqdm(scenes, desc="Loading scenes", disable=not _is_main_process()):
            scene_path = self.scenes_dir / scene_id

            # Get image directory
            if use_undistorted:
                images_dir = scene_path / "dslr" / "resized_undistorted_images"
            else:
                images_dir = scene_path / "dslr" / "resized_images"

            if not images_dir.exists():
                skipped += 1
                continue

            # Get train images only
            train_images, _ = io.load_train_test_split(scene_path)
            if not train_images:
                train_images = sorted([f.name for f in images_dir.glob("*.JPG")])

            # Need enough views
            if len(train_images) < views_per_sample:
                skipped += 1
                continue

            # Load transforms
            transforms_path = scene_path / "dslr" / "nerfstudio" / "transforms.json"
            if use_undistorted:
                transforms_path = scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
            transforms = io.load_nerfstudio_transforms(transforms_path)

            # Load annotations
            annotations = io.load_semantic_annotations(scene_path)

            if supervised:
                scene_obj_ids_dir = self.obj_ids_root / scene_id
                if not scene_obj_ids_dir.exists():
                    skipped += 1
                    continue
                available_obj_ids = {p.stem for p in scene_obj_ids_dir.glob("*.pth")}
                train_images = [img for img in train_images if img in available_obj_ids]
                if len(train_images) < views_per_sample:
                    skipped += 1
                    continue

            # Filter out excluded frames (DA3 depth errors)
            if scene_id in EXCLUDE_FRAMES:
                before = len(train_images)
                train_images = [img for img in train_images
                                if (Path(img).stem if '.' in img else img) not in EXCLUDE_FRAMES[scene_id]]
                if len(train_images) < before and _is_main_process():
                    logger.debug(f"  {scene_id}: excluded {before - len(train_images)} bad frames")

            obj_to_label = {}
            label_to_obj_ids = defaultdict(list)
            anno_path = scene_path / "scans" / "segments_anno.json"
            if anno_path.exists():
                with open(anno_path) as f:
                    anno_data = json.load(f)
                for group in anno_data.get('segGroups', []):
                    obj_id = group.get('objectId') or group.get('id')
                    label = normalize_label(group.get('label', 'object'))
                    if obj_id is not None:
                        obj_to_label[obj_id] = label
                        label_to_obj_ids[label].append(obj_id)

            self.scenes.append({
                'scene_id': scene_id,
                'scene_path': scene_path,
                'images_dir': images_dir,
                'train_images': train_images,
                'transforms': transforms,
                'annotations': annotations,
                'obj_to_label': obj_to_label,
                'label_to_obj_ids': dict(label_to_obj_ids),
            })

        if _is_main_process():
            logger.info(f"Loaded {len(self.scenes)} scenes ({skipped} skipped)")

        self._scene_chunk_map = {}  # scene_id -> {stem -> chunk_idx}
        if self.use_cached_depth and self.da3_cache_dir and self.da3_cache_dir.exists():
            for scene in self.scenes:
                scene_id = scene['scene_id']
                da3_scene_dir = self.da3_cache_dir / scene_id
                if not da3_scene_dir.exists():
                    continue
                da3_cached_stems = sorted(f.stem for f in da3_scene_dir.glob("*.pt"))
                if not da3_cached_stems:
                    continue
                # Read one file to get actual chunk size
                sample_data = torch.load(
                    da3_scene_dir / f"{da3_cached_stems[0]}.pt",
                    map_location='cpu', weights_only=True, mmap=False
                )
                cframes = sample_data.get('chunk_frames', [])
                actual_chunk_size = len(cframes) if cframes else 16
                stem_to_chunk = {}
                for stem_idx, stem in enumerate(da3_cached_stems):
                    stem_to_chunk[stem] = stem_idx // actual_chunk_size
                self._scene_chunk_map[scene_id] = stem_to_chunk
            if self._scene_chunk_map and _is_main_process():
                logger.debug(f"Built DA3 chunk map for {len(self._scene_chunk_map)} scenes")

        self.scene_grouped = (num_objects_per_sample == 0)

        self.object_samples = []
        self._init_object_samples(
            enumerate_all_objects, supervised, views_per_sample,
            sampling_strategy, min_object_pixels, split, max_scenes,
            use_cached_depth, da3_cache_name,
        )
        self._mesh_cache = OrderedDict()
        self._mesh_cache_max = 20  # ~50MB per scene max

    def _get_rasterizer(self, scene_id: str, scene_path: Path):
        """Get or create rasterizer for a scene (LRU cached to limit memory)."""
        from triangulang.utils.scannetpp_rasterization import SceneRasterizer
        MAX_CACHED_RASTERIZERS = 10  # Limit to ~500MB RAM for meshes

        if scene_id in self.rasterizers:
            # Move to end (most recently used)
            self.rasterizers.move_to_end(scene_id)
        else:
            # Evict oldest if at capacity
            while len(self.rasterizers) >= MAX_CACHED_RASTERIZERS:
                oldest_id, oldest_raster = self.rasterizers.popitem(last=False)
                # Free memory
                del oldest_raster

            self.rasterizers[scene_id] = SceneRasterizer(
                scene_path,
                cache_dir=self.raster_cache_dir,
                use_undistorted=self.use_undistorted
            )
        return self.rasterizers[scene_id]

    def _get_object_centroid_lightweight(self, scene_id: str, scene_path: Path, target_obj_id: int) -> Optional[np.ndarray]:
        """
        Get 3D centroid from mesh vertex median (NOT OBB centroid).

        OBB centroid is the geometric center of the bounding box, which for
        large planar objects (walls, floors) can be meters away from the
        visible surface. Using mesh vertex median matches what triangulation
        computes from predicted masks.
        """
        from triangulang.utils.scannetpp_rasterization import (
            load_vertex_object_ids, get_object_centroid_3d
        )

        # Check cache first
        if scene_id in self._mesh_cache:
            self._mesh_cache.move_to_end(scene_id)
            vertices, vertex_obj_ids = self._mesh_cache[scene_id]
        else:
            # Load mesh vertices and object mapping
            mesh_path = scene_path / "scans" / "mesh_aligned_0.05.ply"
            if not mesh_path.exists() or not HAS_TRIMESH:
                return None

            try:
                mesh = trimesh.load(str(mesh_path), process=False)
                vertices = np.array(mesh.vertices, dtype=np.float32)

                # Load vertex-to-object mapping
                vertex_obj_ids, _ = load_vertex_object_ids(scene_path)

                # Evict oldest if at capacity
                while len(self._mesh_cache) >= self._mesh_cache_max:
                    self._mesh_cache.popitem(last=False)

                self._mesh_cache[scene_id] = (vertices, vertex_obj_ids)
            except Exception as e:
                logger.warning(f"Failed to load mesh for {scene_id}: {e}")
                return None

        return get_object_centroid_3d(vertices, vertex_obj_ids, target_obj_id)

    def _build_spatial_context(
        self,
        target_obj_id: int,
        target_label: str,
        pix_obj_ids: np.ndarray,
        depth: np.ndarray,
        obj_to_label: Dict[int, str],
        min_coverage: float = 0.001,
        max_other_objects: int = 15
    ) -> Optional['SpatialContext']:
        """Build spatial context for GT-aware spatial augmentation."""
        if not HAS_SPATIAL_CONTEXT:
            return None

        H, W = pix_obj_ids.shape
        total_pixels = H * W

        # Get target mask
        target_mask = (pix_obj_ids == target_obj_id).astype(np.float32)
        if target_mask.sum() == 0:
            return None

        unique_ids, counts = np.unique(pix_obj_ids, return_counts=True)
        scene_obj_masks = {}

        for obj_id, count in zip(unique_ids, counts):
            if obj_id <= 0:  # Skip background
                continue
            coverage = count / total_pixels
            if coverage < min_coverage:
                continue
            if obj_id not in obj_to_label:
                continue
            # Create mask for this object
            scene_obj_masks[obj_id] = (pix_obj_ids == obj_id).astype(np.float32)

        # Convert depth to numpy if needed
        if isinstance(depth, torch.Tensor):
            depth_np = depth.squeeze().cpu().numpy()
        else:
            depth_np = depth.squeeze() if hasattr(depth, 'squeeze') else depth

        return build_spatial_context(
            target_mask=target_mask,
            target_obj_id=target_obj_id,
            target_label=target_label,
            depth=depth_np,
            scene_obj_masks=scene_obj_masks,
            obj_to_label=obj_to_label,
            max_nearby_objects=max_other_objects
        )

    def __len__(self):
        if self.scene_grouped:
            return len(self.scenes) * self.samples_per_scene
        if self.enumerate_all_objects and self.object_samples:
            return len(self.object_samples)
        return len(self.scenes) * self.samples_per_scene

    def _sample_views(self, images: List[str], n_views: int, transforms: dict = None,
                       scene_id: str = None) -> List[str]:
        """Delegate to scannetpp_sampling.sample_views."""
        from triangulang.utils.scannetpp_sampling import sample_views
        if not hasattr(self, '_chunk_warning_ref'):
            self._chunk_warning_ref = [False]
        return sample_views(
            images, n_views, self.sampling_strategy,
            scene_id=scene_id, transforms=transforms,
            use_cached_depth=self.use_cached_depth,
            scene_chunk_map=self._scene_chunk_map,
            chunk_warning_ref=self._chunk_warning_ref,
        )

    def _sample_covisible_objects(self, scene, all_obj_ids, exclude_obj_id, exclude_label, num_extra,
                                   deduplicate_labels=True):
        """Sample additional objects visible in the selected views for multi-object training."""
        if num_extra == 0:
            return []

        obj_to_label = scene.get('obj_to_label', {})
        skip_labels = SCANNETPP_SKIP_LABELS

        visible_obj_ids = set()
        for pix_obj_ids in all_obj_ids:
            if pix_obj_ids is not None:
                unique_ids = np.unique(pix_obj_ids)
                for oid in unique_ids:
                    oid = int(oid)
                    if oid > 0:
                        visible_obj_ids.add(oid)

        used_labels = {exclude_label} if (exclude_label and deduplicate_labels) else set()
        candidates = []
        for oid in visible_obj_ids:
            if oid == exclude_obj_id:
                continue
            label = obj_to_label.get(oid)
            if label is None:
                continue
            label = normalize_label(label)
            if label.lower() in skip_labels:
                continue
            if label in self.exclude_categories:
                continue
            if self.include_categories and label not in self.include_categories:
                continue
            if deduplicate_labels and label in used_labels:
                continue
            has_coverage = False
            for pix_obj_ids in all_obj_ids:
                if pix_obj_ids is not None:
                    coverage = (pix_obj_ids == oid).sum() / pix_obj_ids.size
                    if coverage >= self.min_object_pixels:
                        has_coverage = True
                        break
            if has_coverage:
                candidates.append((oid, label))
                if deduplicate_labels:
                    used_labels.add(label)

        random.shuffle(candidates)
        if num_extra < 0:
            return candidates
        return candidates[:num_extra]

    def _load_gt_masks(self, scene, scene_id, selected_images, forced_obj_id, forced_label):
        """Load per-view GT masks from pre-computed obj_id .pth files.

        Returns:
            (gt_masks, gt_mask_coverages, target_obj_id, prompt, all_object_prompts)
            gt_masks is (N, H, W) for single-object or (K, N, H, W) for multi-object.
            Returns all-None tuple if no valid object found.
        """
        gt_masks = gt_mask_coverages = target_obj_id = prompt = all_object_prompts = None

        all_obj_ids = []
        for img_name in selected_images:
            obj_ids_path = self.obj_ids_root / scene_id / f"{img_name}.pth"
            if obj_ids_path.exists():
                pix_obj_ids = torch.load(obj_ids_path, weights_only=False)
                all_obj_ids.append(pix_obj_ids)
            else:
                all_obj_ids.append(None)

        has_any_obj_ids = (any(x is not None for x in all_obj_ids)
                           if self.scene_grouped else (all_obj_ids[0] is not None))
        if not has_any_obj_ids:
            return gt_masks, gt_mask_coverages, target_obj_id, prompt, all_object_prompts

        if forced_obj_id is not None:
            target_obj_id = forced_obj_id
            prompt = forced_label
        else:
            obj_to_label = scene.get('obj_to_label', {})
            skip_labels = SCANNETPP_SKIP_LABELS
            scan_views = all_obj_ids if self.scene_grouped else [all_obj_ids[0]]
            visible_objects = []
            seen_obj_ids = set()
            for pix_obj_ids in scan_views:
                if pix_obj_ids is None:
                    continue
                H, W = pix_obj_ids.shape
                total_pixels = H * W
                unique_ids, counts = np.unique(pix_obj_ids, return_counts=True)
                for obj_id, count in zip(unique_ids, counts):
                    obj_id = int(obj_id)
                    if obj_id <= 0 or obj_id in seen_obj_ids:
                        continue
                    fraction = count / total_pixels
                    if fraction >= self.min_object_pixels:
                        label = obj_to_label.get(obj_id)
                        if label and label.lower() not in skip_labels:
                            if not self.exclude_categories or label not in self.exclude_categories:
                                if not self.include_categories or label in self.include_categories:
                                    visible_objects.append((obj_id, label))
                                    seen_obj_ids.add(obj_id)
            if visible_objects:
                target_obj_id, prompt = random.choice(visible_objects)

        if target_obj_id is None:
            return gt_masks, gt_mask_coverages, target_obj_id, prompt, all_object_prompts

        all_target_objects = [(target_obj_id, prompt)]
        if self.num_objects_per_sample != 1:
            num_extra = -1 if self.num_objects_per_sample == 0 else self.num_objects_per_sample - 1
            extra_objects = self._sample_covisible_objects(
                scene, all_obj_ids, target_obj_id, prompt,
                num_extra=num_extra,
                deduplicate_labels=self.semantic_union,
            )
            all_target_objects.extend(extra_objects)

        all_object_gt_masks = []
        all_object_gt_coverages = []
        all_object_prompts = []
        label_to_obj_ids_map = scene.get('label_to_obj_ids', {})

        for obj_id, obj_label in all_target_objects:
            if self.semantic_union and obj_label is not None:
                matching_obj_ids = label_to_obj_ids_map.get(obj_label, [obj_id])
            else:
                matching_obj_ids = [obj_id]

            obj_gt_masks = []
            obj_gt_coverages = []
            for pix_obj_ids in all_obj_ids:
                if pix_obj_ids is not None:
                    mask = np.zeros_like(pix_obj_ids, dtype=np.float32)
                    for oid in matching_obj_ids:
                        mask += (pix_obj_ids == oid).astype(np.float32)
                    mask = (mask > 0).astype(np.float32)
                    orig_coverage = mask.sum() / mask.size
                    obj_gt_coverages.append(orig_coverage)
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize(
                        (self.mask_size[1], self.mask_size[0]), Image.NEAREST
                    )
                    mask = np.array(mask_pil).astype(np.float32) / 255.0
                    obj_gt_masks.append(torch.from_numpy(mask))
                else:
                    obj_gt_masks.append(torch.zeros(self.mask_size))
                    obj_gt_coverages.append(0.0)

            all_object_gt_masks.append(torch.stack(obj_gt_masks))
            all_object_gt_coverages.append(obj_gt_coverages)
            all_object_prompts.append(obj_label)

        K_actual = len(all_target_objects)
        if K_actual == 1 and self.num_objects_per_sample == 1:
            gt_masks = all_object_gt_masks[0]        # (N, H, W)
            gt_mask_coverages = all_object_gt_coverages[0]
        else:
            gt_masks = torch.stack(all_object_gt_masks)  # (K, N, H, W)
            gt_mask_coverages = all_object_gt_coverages[0]

        return gt_masks, gt_mask_coverages, target_obj_id, prompt, all_object_prompts

    def _init_object_samples(self, enumerate_all_objects, supervised, views_per_sample,
                              sampling_strategy, min_object_pixels, split, max_scenes,
                              use_cached_depth, da3_cache_name):
        """Enumerate all (scene_idx, obj_id, label) triples with file-lock-safe caching."""
        from tqdm import tqdm
        if not (enumerate_all_objects and supervised and not self.scene_grouped):
            if _is_main_process():
                logger.info(f"  {self.samples_per_scene} samples/scene = "
                      f"{len(self.scenes) * self.samples_per_scene} samples/epoch")
            return

        needs_chunk_filter = sampling_strategy in ('chunk_aware', 'overlap_30', 'overlap_50')
        chunk_tag = f"chunk_{da3_cache_name}" if (use_cached_depth and needs_chunk_filter) else "nochunk"
        cache_key_data = f"{split}_{max_scenes}_{views_per_sample}_{min_object_pixels}_{len(self.scenes)}_{chunk_tag}"
        cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()[:12]
        cache_path = self.data_root / f".object_samples_cache_{cache_key}.pkl"
        lock_path = cache_path.with_suffix('.lock')

        if cache_path.exists():
            if _is_main_process():
                logger.info(f"Loading object samples from cache: {cache_path.name}")
            try:
                with open(lock_path, 'w') as lf:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_SH)
                    try:
                        with open(cache_path, 'rb') as f:
                            self.object_samples = pickle.load(f)['object_samples']
                        if _is_main_process():
                            logger.info(f"  Loaded {len(self.object_samples)} object samples from cache")
                    finally:
                        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                logger.warning(f"  Cache load failed: {e}, re-enumerating...")
                self.object_samples = []

        if not self.object_samples:
            with open(lock_path, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    if cache_path.exists():
                        try:
                            with open(cache_path, 'rb') as f:
                                self.object_samples = pickle.load(f)['object_samples']
                            if _is_main_process():
                                logger.info(f"  Loaded {len(self.object_samples)} object samples from cache (another process)")
                        except Exception as e:
                            logger.warning(f"  Cache load failed: {e}, will enumerate...")
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

        if not self.object_samples:
            if _is_main_process():
                logger.info("Enumerating all objects across scenes (will cache for future runs)...")
            total_found = filtered_out = total_chunk_filtered = 0

            for scene_idx, scene in enumerate(tqdm(self.scenes, desc="Enumerating objects", disable=not _is_main_process())):
                scene_id = scene['scene_id']
                scene_path = scene['scene_path']
                train_images_set = set(scene['train_images'])
                anno_path = scene_path / "scans" / "segments_anno.json"
                obj_to_label = {}
                if anno_path.exists():
                    with open(anno_path) as f:
                        anno_data = json.load(f)
                    for group in anno_data.get('segGroups', []):
                        oid = group.get('objectId') or group.get('id')
                        lbl = normalize_label(group.get('label', 'object'))
                        if oid is not None:
                            obj_to_label[oid] = lbl

                obj_visibility = defaultdict(set)
                scene_obj_ids_dir = self.obj_ids_root / scene_id
                skip_set = SCANNETPP_SKIP_LABELS
                for pth_file in scene_obj_ids_dir.glob("*.pth"):
                    img_name = pth_file.stem
                    if img_name not in train_images_set:
                        continue
                    try:
                        pix_obj_ids = torch.load(pth_file, weights_only=False)
                        for oid in np.unique(pix_obj_ids):
                            if oid > 0 and oid in obj_to_label:
                                if obj_to_label[oid].lower() not in skip_set:
                                    obj_visibility[oid].add(img_name)
                    except:
                        continue

                img_to_chunk = self._scene_chunk_map.get(scene_id, {})
                bad_count = chunk_filtered = 0
                for oid, visible in obj_visibility.items():
                    total_found += 1
                    if is_bad_annotation(scene_id, oid):
                        bad_count += 1
                        continue
                    if len(visible) < views_per_sample:
                        filtered_out += 1
                        continue
                    if img_to_chunk and self.sampling_strategy in ('chunk_aware', 'overlap_30', 'overlap_50'):
                        chunk_counts = Counter()
                        for img in visible:
                            stem = Path(img).stem if '.' in img else img
                            if stem in img_to_chunk:
                                chunk_counts[img_to_chunk[stem]] += 1
                        if (max(chunk_counts.values()) if chunk_counts else 0) < views_per_sample:
                            chunk_filtered += 1
                            filtered_out += 1
                            continue
                    self.object_samples.append({
                        'scene_idx': scene_idx, 'obj_id': oid,
                        'label': obj_to_label.get(oid, 'object'),
                        'visible_images': list(visible),
                    })
                total_chunk_filtered += chunk_filtered
                if bad_count > 0:
                    logger.debug(f"    Skipped {bad_count} bad annotations in {scene_id}")

            if _is_main_process():
                logger.info(f"  Total objects found: {total_found}")
                logger.info(f"  Filtered (visible in <{views_per_sample} views): {filtered_out}")
                if total_chunk_filtered:
                    logger.info(f"    (of which {total_chunk_filtered} had no single DA3 chunk with enough views)")
                logger.info(f"  Valid object samples: {len(self.object_samples)}")
            with open(lock_path, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump({'object_samples': self.object_samples}, f)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

        if self.scene_grouped and _is_main_process():
            logger.info(f"  Scene-grouped mode: {len(self.scenes)} scenes/epoch")
        if self.exclude_categories and self.object_samples:
            before = len(self.object_samples)
            self.object_samples = [s for s in self.object_samples if s['label'] not in self.exclude_categories]
            if _is_main_process() and len(self.object_samples) < before:
                logger.info(f"  Excluded {before - len(self.object_samples)} samples: {before} -> {len(self.object_samples)}")
        if self.include_categories and self.object_samples:
            before = len(self.object_samples)
            self.object_samples = [s for s in self.object_samples if s['label'] in self.include_categories]
            if _is_main_process() and len(self.object_samples) < before:
                logger.info(f"  Whitelist: kept {len(self.object_samples)}/{before} samples")
        if self.min_category_samples > 1 and self.object_samples:
            counts = Counter(s['label'] for s in self.object_samples)
            rare = {cat for cat, n in counts.items() if n < self.min_category_samples}
            if rare:
                before = len(self.object_samples)
                self.object_samples = [s for s in self.object_samples if s['label'] not in rare]
                if _is_main_process():
                    logger.info(f"  Filtered {len(rare)} rare categories: {before} -> {len(self.object_samples)}")

    def _apply_da3_cache(self, result, image_names, scene_id, gt_masks, target_obj_id, prompt,
                          obj_to_label=None):
        """Load pre-computed DA3 depth/pose cache and add to result dict in-place."""
        cached_depths, cached_ext, cached_int = [], [], []
        all_cached = has_poses = True

        for img_name in image_names:
            cache_path = self.da3_cache_dir / scene_id / f"{Path(img_name).stem}.pt"
            if cache_path.exists():
                try:
                    d = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=False)
                    depth = d['depth'].float()
                    if depth.dim() == 4: depth = depth.squeeze(0)
                    elif depth.dim() == 2: depth = depth.unsqueeze(0)
                    cached_depths.append(depth)
                    if 'extrinsics' in d: cached_ext.append(d['extrinsics'].float())
                    else: has_poses = False
                    if 'intrinsics' in d: cached_int.append(d['intrinsics'].float())
                except Exception as e:
                    if not hasattr(self, '_cache_warning_printed'):
                        logger.warning(f"[DA3 Cache] Error loading {cache_path}: {e}")
                        self._cache_warning_printed = True
                    all_cached = False; break
            else:
                if not hasattr(self, '_cache_warning_printed'):
                    logger.warning(f"[DA3 Cache] Missing: {cache_path}  scene={scene_id} img={img_name}")
                    self._cache_warning_printed = True
                all_cached = False; break

        if not (all_cached and len(cached_depths) == len(image_names)):
            return

        cached_depth_tensor = torch.stack(cached_depths)  # (N, 1, H, W)
        cache_h, cache_w = cached_depth_tensor.shape[-2:]
        target_h, target_w = self.image_size
        scale_h = scale_w = 1.0

        if cache_h != target_h or cache_w != target_w:
            scale_h = target_h / cache_h
            scale_w = target_w / cache_w
            cached_depth_tensor = F.interpolate(
                cached_depth_tensor, size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )

        result['cached_depth'] = cached_depth_tensor

        if has_poses and len(cached_ext) == len(image_names):
            result['cached_da3_extrinsics'] = torch.stack(cached_ext)
        if len(cached_int) == len(image_names):
            intr = torch.stack(cached_int)
            if scale_h != 1.0 or scale_w != 1.0:
                orig_fy = intr[0, 1, 1].item()
                orig_cy = intr[0, 1, 2].item()
                intr = intr.clone()
                intr[:, 0, 0] *= scale_w
                intr[:, 1, 1] *= scale_h
                intr[:, 0, 2] *= scale_w
                intr[:, 1, 2] *= scale_h
                if not hasattr(self, '_intrinsics_scale_logged'):
                    self._intrinsics_scale_logged = True
                    logger.debug(f"[Dataloader] Intrinsics scaled: {cache_h}x{cache_w} -> "
                          f"{target_h}x{target_w}  fy {orig_fy:.1f} -> {intr[0,1,1].item():.1f}")
            result['cached_da3_intrinsics'] = intr

        if (HAS_SPATIAL_CONTEXT and gt_masks is not None and
                target_obj_id is not None and prompt is not None and self.supervised):
            try:
                obj_ids_path = self.obj_ids_root / scene_id / f"{image_names[0]}.pth"
                if obj_ids_path.exists():
                    first_obj_ids = torch.load(obj_ids_path, weights_only=False)
                    first_depth = cached_depth_tensor[0, 0].numpy()
                    if first_obj_ids.shape != first_depth.shape:
                        pil = Image.fromarray(first_obj_ids.astype(np.int32), mode='I')
                        pil = pil.resize((first_depth.shape[1], first_depth.shape[0]), Image.NEAREST)
                        first_obj_ids = np.array(pil)
                    ctx = self._build_spatial_context(
                        target_obj_id=target_obj_id, target_label=prompt,
                        pix_obj_ids=first_obj_ids, depth=first_depth,
                        obj_to_label=obj_to_label or {}
                    )
                    if ctx is not None:
                        result['spatial_context'] = ctx
            except Exception as e:
                if not hasattr(self, '_spatial_context_warning_printed'):
                    logger.warning(f"[Spatial Context] {e}")
                    self._spatial_context_warning_printed = True

    def __getitem__(self, idx):
        io = _get_io()
        forced_obj_id = forced_label = forced_visible_images = None

        if self.scene_grouped:
            scene_idx = idx % len(self.scenes)
        elif self.enumerate_all_objects and self.object_samples:
            obj_sample = self.object_samples[idx]
            scene_idx = obj_sample['scene_idx']
            forced_obj_id = obj_sample['obj_id']
            forced_label = obj_sample['label']
            forced_visible_images = obj_sample.get('visible_images')
        else:
            scene_idx = idx % len(self.scenes)

        scene = self.scenes[scene_idx]
        scene_id = scene['scene_id']
        scene_path = scene['scene_path']
        images_dir = scene['images_dir']

        candidate_images = forced_visible_images if forced_visible_images is not None else scene['train_images']
        selected_images = self._sample_views(
            candidate_images, self.views_per_sample, transforms=scene['transforms'],
            scene_id=scene_id
        )

        images, intrinsics_list, extrinsics_list, image_names = [], [], [], []

        transforms = scene['transforms']
        if transforms:
            fx, fy = transforms.get('fl_x', 500), transforms.get('fl_y', 500)
            cx, cy = transforms.get('cx', 256), transforms.get('cy', 256)
            shared_intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
            frame_lookup = {Path(fr.get('file_path', '')).name: fr for fr in transforms.get('frames', [])}
        else:
            shared_intrinsics = torch.eye(3)
            frame_lookup = {}

        for img_name in selected_images:
            image = Image.open(images_dir / img_name).convert('RGB')
            image = image.resize(self.image_size, Image.BILINEAR)
            images.append(torch.from_numpy(np.array(image)).float().div(255.0).permute(2, 0, 1))
            extrinsics = (torch.tensor(frame_lookup[img_name]['transform_matrix'], dtype=torch.float32)
                          if img_name in frame_lookup else torch.eye(4))
            intrinsics_list.append(shared_intrinsics)
            extrinsics_list.append(extrinsics)
            image_names.append(img_name)

        gt_masks = gt_mask_coverages = target_obj_id = prompt = all_object_prompts = None
        if self.supervised:
            gt_masks, gt_mask_coverages, target_obj_id, prompt, all_object_prompts = \
                self._load_gt_masks(scene, scene_id, selected_images, forced_obj_id, forced_label)

        if prompt is None:
            object_labels = list(scene['annotations'].keys())
            if object_labels:
                label = random.choice(object_labels)
                prompt = io.SCANNETPP_PROMPTS.get(label, [label])[0]
            else:
                prompt = random.choice(['chair', 'table', 'monitor', 'lamp'])

        orig_h = transforms.get('h', 1168) if transforms else 1168
        orig_w = transforms.get('w', 1752) if transforms else 1752

        result = {
            'images': torch.stack(images),
            'intrinsics': torch.stack(intrinsics_list),
            'extrinsics': torch.stack(extrinsics_list),
            'orig_hw': (orig_h, orig_w),
            'scene_id': scene_id,
            'prompt': prompt,
            'image_names': image_names,
            'has_metric_scale': True,
            'has_gt_mask': gt_masks is not None,
        }

        if gt_masks is not None:
            result['gt_masks'] = gt_masks
            result['target_obj_id'] = target_obj_id
            result['gt_mask_coverage'] = torch.tensor(gt_mask_coverages, dtype=torch.float32)
            if gt_masks.dim() == 4:
                result['num_objects'] = gt_masks.shape[0]
                result['multi_object_prompts'] = all_object_prompts
            if scene_id in self.centroid_cache and str(target_obj_id) in self.centroid_cache[scene_id]:
                result['centroid_3d'] = torch.tensor(
                    self.centroid_cache[scene_id][str(target_obj_id)], dtype=torch.float32)
            else:
                centroid_3d = self._get_object_centroid_lightweight(scene_id, scene_path, target_obj_id)
                if centroid_3d is not None:
                    result['centroid_3d'] = torch.from_numpy(centroid_3d).float()

        if self.use_cached_depth and self.da3_cache_dir is not None:
            self._apply_da3_cache(result, image_names, scene_id,
                                  gt_masks, target_obj_id, prompt,
                                  scene.get('obj_to_label', {}))

        # Load PI3X cached world-frame pointmaps if available
        if self.use_cached_pi3x and self.pi3x_cache_dir is not None:
            pi3x_pointmaps = []
            all_loaded = True
            for img_name in image_names:
                cache_path = self.pi3x_cache_dir / scene_id / f"{Path(img_name).stem}.pt"
                if cache_path.exists():
                    try:
                        cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
                        pointmap = torch.as_tensor(cache_data['pointmap']).float()  # [H, W, 3]
                        pi3x_pointmaps.append(pointmap)
                        # Also load depth from PI3X if not already cached
                        if 'cached_depth' not in result and 'depth' in cache_data:
                            pass  # DA3 cache takes priority for depth
                    except Exception:
                        all_loaded = False
                        break
                else:
                    all_loaded = False
                    break
            if all_loaded and len(pi3x_pointmaps) == len(image_names):
                result['cached_pi3x_pointmaps'] = torch.stack(pi3x_pointmaps)  # [N, H, W, 3]

        return result

# Backward-compatible re-exports (keep existing import statements working)
from triangulang.utils.scannetpp_io import (  # noqa: E402, F401
    SCANNETPP_PROMPTS,
    load_scene_list,
    load_semantic_classes,
    load_nerfstudio_transforms,
    load_train_test_split,
    get_available_scenes,
    load_semantic_annotations,
    ScanNetPPDataset,
)

from triangulang.utils.scannetpp_rasterization import (  # noqa: E402, F401
    load_vertex_object_ids,
    get_object_centroid_3d,
    get_vtx_prop_on_2d,
    rasterize_mesh_o3d,
    get_objects_in_image,
    SceneRasterizer,
    get_scene_3d_annotations,
    project_mesh_to_mask,
    load_segments_mapping,
)

