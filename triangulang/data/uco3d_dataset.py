"""
uCO3D Dataset Loader for Multi-View Training.

uCO3D (UnCommon Objects in 3D) provides ~170k turntable videos of objects
from ~1000 LVIS categories. Each video has:
- RGB frames (from rgb_video.mp4)
- Segmentation masks (from mask_video.mkv)
- Depth maps (from depth_maps.h5, aligned with VGGSfM)
- Camera poses (from metadata.sqlite)
- 3D Gaussian splats and point clouds

Key differences from ScanNet++:
- Object-centric (single foreground object) vs scene-level (multiple objects)
- Turntable videos (small baseline) vs wide-baseline indoor scenes
- Category name as text prompt (no instance-level naming)

Reference: Liu et al., "UnCommon Objects in 3D" (2025)
https://arxiv.org/abs/2501.07574

Usage:
    from triangulang.data.uco3d_dataset import UCO3DMultiViewDataset

    dataset = UCO3DMultiViewDataset(
        data_root='data/uco3d',
        split='train',
        num_views=8,
        num_sequences=50,  # Sample 50 representative sequences
        frames_per_sequence=50,  # Sample 50 frames uniformly
    )
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

import triangulang
logger = triangulang.get_logger(__name__)

# Try to import uCO3D package
try:
    from uco3d import UCO3DDataset as UCO3DDatasetBase, UCO3DFrameDataBuilder
    from uco3d.dataset_utils.utils import get_dataset_root
    HAS_UCO3D = True
except ImportError:
    HAS_UCO3D = False
    logger.warning("[uCO3D] uco3d package not installed. Using standalone loader.")

# Try to import h5py for depth loading
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


from triangulang.data.uco3d_utils import (
    SKIP_CATEGORIES, SKIP_SEQUENCES, PROMPT_SIMPLIFICATIONS, normalize_prompt,
)

class UCO3DMultiViewDataset(Dataset):
    """
    Multi-view dataset for uCO3D.

    Each sample returns multiple views of the same object with:
    - RGB images
    - Foreground segmentation masks (GT)
    - Depth maps (from DepthAnythingV2, scale-aligned)
    - Camera intrinsics/extrinsics
    - Category name as text prompt

    Sampling strategy (for evaluation as described in paper):
    - Sample 50 representative sequences across categories
    - Sample 50 frames uniformly from each sequence
    """

    def __init__(
        self,
        data_root: str = None,
        split: str = 'train',
        num_views: int = 8,
        image_size: Tuple[int, int] = (504, 504),
        mask_size: Tuple[int, int] = (128, 128),
        num_sequences: Optional[int] = None,  # None = all sequences
        frames_per_sequence: int = 50,  # Uniform sampling per sequence
        categories: Optional[List[str]] = None,  # Filter to specific categories
        use_depth: bool = True,
        subset_list: str = 'set_lists_all-categories.sqlite',
        seed: int = 42,
        samples_per_sequence: int = 1,  # For training: multiple samples per sequence
        normalize_prompts: bool = True,  # Simplify LVIS category names
        # K-fold cross validation
        num_folds: Optional[int] = None,  # Number of folds (e.g., 5 for 5-fold CV)
        fold: Optional[int] = None,  # Which fold to use as validation (0 to num_folds-1)
        # Cached DA3 depth
        use_cached_depth: bool = False,  # Load pre-computed DA3 depth from cache
        da3_cache_name: str = 'da3_metric_cache',  # Cache dir name under data_root
    ):
        """
        Args:
            data_root: Path to uCO3D dataset root. If None, uses UCO3D_DATASET_ROOT env var.
            split: 'train' or 'val'
            num_views: Number of views per sample
            image_size: (H, W) for image resizing
            mask_size: (H, W) for mask resizing
            num_sequences: Number of sequences to sample (None = all)
            frames_per_sequence: Number of frames to sample per sequence
            categories: List of categories to include (None = all)
            use_depth: Whether to load depth maps
            subset_list: Name of subset list file in set_lists/
            seed: Random seed for reproducible sampling
            samples_per_sequence: Number of training samples per sequence
            normalize_prompts: If True, simplify LVIS category names (e.g., "dishwasher_detergent" -> "detergent bottle")
            num_folds: For k-fold CV, number of folds. If set, ignores original train/val split.
            fold: Which fold to use as validation (0 to num_folds-1). Required if num_folds is set.
        """
        super().__init__()

        # Validate k-fold parameters
        if num_folds is not None:
            if fold is None:
                raise ValueError("fold must be specified when using num_folds")
            if fold < 0 or fold >= num_folds:
                raise ValueError(f"fold must be in [0, {num_folds-1}], got {fold}")

        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        self.mask_size = mask_size
        self.num_sequences = num_sequences
        self.frames_per_sequence = frames_per_sequence
        self.categories = categories
        self.use_depth = use_depth
        self.seed = seed
        self.samples_per_sequence = samples_per_sequence
        self.normalize_prompts = normalize_prompts
        self.num_folds = num_folds
        self.fold = fold
        self.use_cached_depth = use_cached_depth
        self.da3_cache_name = da3_cache_name

        # Get dataset root
        if data_root is not None:
                self.data_root = Path(data_root)
        elif os.environ.get('UCO3D_DATASET_ROOT'):
            self.data_root = Path(os.environ['UCO3D_DATASET_ROOT'])
        else:
            # Try common locations (including symlinks)
            for path in ['data/uco3d', '/data/uco3d', '~/data/uco3d']:
                expanded = Path(path).expanduser()
                if expanded.exists():
                    self.data_root = expanded
                    break
            else:
                raise ValueError(
                    "uCO3D dataset root not found. Set UCO3D_DATASET_ROOT env var "
                    "or pass data_root argument."
                )

        logger.info(f"[uCO3D] Loading from {self.data_root}")

        # Set up DA3 cache directory
        self.da3_cache_dir = self.data_root / da3_cache_name if use_cached_depth else None
        if use_cached_depth:
            if self.da3_cache_dir and self.da3_cache_dir.exists():
                logger.info(f"[uCO3D] Using cached depth from {self.da3_cache_dir}")
            else:
                logger.warning(f"[uCO3D] DA3 cache not found: {self.da3_cache_dir}")
                logger.warning(f"  Run scripts/preprocess_da3_uco3d.py first.")

        # Initialize sequences
        self.sequences = []
        self._init_sequences(subset_list)

        # Sample sequences if requested
        if num_sequences is not None and len(self.sequences) > num_sequences:
            self._sample_representative_sequences(num_sequences)

        logger.info(f"[uCO3D] Loaded {len(self.sequences)} sequences, "
              f"{self.frames_per_sequence} frames each, "
              f"{self.num_views} views per sample")

    # Initialize sequence list from metadata
    def _init_sequences(self, subset_list: str):
        # Try official sqlite set_lists first (has proper train/val splits)
        sqlite_path = self.data_root / "set_lists" / subset_list
        if sqlite_path.exists():
            self._init_from_sqlite(subset_list)
        elif HAS_UCO3D:
            self._init_with_uco3d_package(subset_list)
        else:
            self._init_standalone()

    # Initialize using official uCO3D package
    def _init_with_uco3d_package(self, subset_list: str):
        subset_lists_file = self.data_root / "set_lists" / subset_list

        if not subset_lists_file.exists():
            logger.warning(f"[uCO3D] Subset list not found: {subset_lists_file}")
            self._init_standalone()
            return

        # Build frame data builder for loading
        self.frame_builder = UCO3DFrameDataBuilder(
            apply_alignment=True,
            load_images=True,
            load_depths=self.use_depth,
            load_masks=True,
            load_depth_masks=False,
            load_gaussian_splats=False,
            load_point_clouds=False,
            load_segmented_point_clouds=False,
            load_sparse_point_clouds=False,
            box_crop=False,  # We handle cropping ourselves
            load_frames_from_videos=True,
            image_height=self.image_size[0],
            image_width=self.image_size[1],
            undistort_loaded_blobs=True,
        )

        # Create uCO3D dataset
        self.uco3d_dataset = UCO3DDatasetBase(
            subset_lists_file=str(subset_lists_file),
            subsets=[self.split],
            frame_data_builder=self.frame_builder,
            pick_categories=tuple(self.categories) if self.categories else (),
            n_frames_per_sequence=self.frames_per_sequence,
            seed=self.seed,
        )

        # Group frames by sequence
        sequence_frames = defaultdict(list)
        for i in range(len(self.uco3d_dataset)):
            try:
                # Get sequence name without loading full data
                meta = self.uco3d_dataset._index.iloc[i]
                seq_name = meta['sequence_name']
                sequence_frames[seq_name].append(i)
            except Exception:
                continue

        # Build sequence list
        for seq_name, frame_indices in sequence_frames.items():
            if len(frame_indices) >= self.num_views:
                category = seq_name.split('/')[0] if '/' in seq_name else 'unknown'
                # Skip categories with bad mask quality
                if category in SKIP_CATEGORIES:
                    continue
                # Skip specific mislabeled sequences
                seq_id = seq_name.split('/')[-1] if '/' in seq_name else seq_name
                if seq_id in SKIP_SEQUENCES:
                    continue
                self.sequences.append({
                    'sequence_name': seq_name,
                    'frame_indices': frame_indices,
                    'category': category,
                })

    # Initialize from official sqlite set_lists (proper train/val split or k-fold CV)
    def _init_from_sqlite(self, subset_list: str):
        import sqlite3

        sqlite_path = self.data_root / "set_lists" / subset_list
        if not sqlite_path.exists():
            logger.warning(f"[uCO3D] Set list not found: {sqlite_path}, falling back to directory scan")
            self._init_standalone()
            return

        logger.info(f"[uCO3D] Loading from sqlite: {sqlite_path}")
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()

        # K-fold cross validation: load ALL sequences and split into folds
        if self.num_folds is not None:
            cursor.execute(
                "SELECT sequence_name, category, super_category, subset FROM sequence_lengths"
            )
            all_rows = cursor.fetchall()
            conn.close()

            # Sort deterministically by sequence name for reproducible splits
            all_rows = sorted(all_rows, key=lambda x: x[0])

            # Shuffle with seed for randomized but reproducible folds
            rng = random.Random(self.seed)
            rng.shuffle(all_rows)

            # Split into folds
            fold_size = len(all_rows) // self.num_folds
            fold_starts = [i * fold_size for i in range(self.num_folds)]
            fold_starts.append(len(all_rows))  # End marker

            val_start = fold_starts[self.fold]
            val_end = fold_starts[self.fold + 1]

            if self.split == 'val':
                rows = all_rows[val_start:val_end]
            else:  # train
                rows = all_rows[:val_start] + all_rows[val_end:]

            # Strip the subset column (4th element) since we determined split via fold
            rows = [(r[0], r[1], r[2]) for r in rows]

            logger.info(f"[uCO3D] K-fold CV: {self.num_folds} folds, fold {self.fold} as val")
            logger.info(f"[uCO3D] Found {len(rows)} sequences for split '{self.split}' (fold {self.fold})")
        else:
            # Standard train/val split from dataset
            cursor.execute(
                "SELECT sequence_name, category, super_category FROM sequence_lengths WHERE subset = ?",
                (self.split,)
            )
            rows = cursor.fetchall()
            conn.close()

            logger.info(f"[uCO3D] Found {len(rows)} sequences for split '{self.split}'")

        for seq_name, category, super_category in rows:
            # Skip categories with bad mask quality
            if category in SKIP_CATEGORIES:
                continue

            # Skip specific mislabeled sequences
            if seq_name in SKIP_SEQUENCES:
                continue

            # Build path: super_category/category/sequence_name
            seq_path = self.data_root / super_category / category / seq_name
            rgb_video = seq_path / "rgb_video.mp4"
            mask_video = seq_path / "mask_video.mkv"

            if rgb_video.exists() and mask_video.exists():
                self.sequences.append({
                    'sequence_name': f"{super_category}/{category}/{seq_name}",
                    'sequence_path': seq_path,
                    'category': category,
                    'super_category': super_category,
                    'rgb_video': rgb_video,
                    'mask_video': mask_video,
                    'depth_file': seq_path / "depth_maps.h5" if self.use_depth else None,
                })

        logger.info(f"[uCO3D] Loaded {len(self.sequences)} sequences with valid videos")

    # Initialize without uCO3D package (directory scanning) - no train/val separation
    def _init_standalone(self):
        logger.info("[uCO3D] Using standalone initialization (scanning directories)")
        logger.warning("[uCO3D] No train/val separation - use sqlite set_lists for proper splits")

        # Scan for sequences in super_category/category/sequence structure
        for super_cat_dir in self.data_root.iterdir():
            if not super_cat_dir.is_dir() or super_cat_dir.name.startswith('.'):
                continue
            if super_cat_dir.name in ['set_lists', 'metadata.sqlite']:
                continue

            for cat_dir in super_cat_dir.iterdir():
                if not cat_dir.is_dir():
                    continue

                # Filter categories if specified
                if self.categories and cat_dir.name not in self.categories:
                    continue

                # Skip categories with bad mask quality
                if cat_dir.name in SKIP_CATEGORIES:
                    continue

                for seq_dir in cat_dir.iterdir():
                    if not seq_dir.is_dir():
                        continue

                    # Skip specific mislabeled sequences
                    if seq_dir.name in SKIP_SEQUENCES:
                        continue

                    # Check for required files
                    rgb_video = seq_dir / "rgb_video.mp4"
                    mask_video = seq_dir / "mask_video.mkv"

                    if rgb_video.exists() and mask_video.exists():
                        self.sequences.append({
                            'sequence_name': f"{super_cat_dir.name}/{cat_dir.name}/{seq_dir.name}",
                            'sequence_path': seq_dir,
                            'category': cat_dir.name,
                            'super_category': super_cat_dir.name,
                            'rgb_video': rgb_video,
                            'mask_video': mask_video,
                            'depth_file': seq_dir / "depth_maps.h5" if self.use_depth else None,
                        })

        logger.info(f"[uCO3D] Found {len(self.sequences)} sequences via directory scan")

    # Sample representative sequences across categories
    def _sample_representative_sequences(self, num_sequences: int):
        random.seed(self.seed)

        # Group by category
        by_category = defaultdict(list)
        for seq in self.sequences:
            by_category[seq['category']].append(seq)

        # Sample proportionally from each category
        num_categories = len(by_category)
        seqs_per_category = max(1, num_sequences // num_categories)

        sampled = []
        for cat, seqs in by_category.items():
            n_sample = min(len(seqs), seqs_per_category)
            sampled.extend(random.sample(seqs, n_sample))

        # If we need more, sample randomly from remaining
        if len(sampled) < num_sequences:
            remaining = [s for s in self.sequences if s not in sampled]
            extra_needed = num_sequences - len(sampled)
            if remaining:
                sampled.extend(random.sample(remaining, min(len(remaining), extra_needed)))

        # If we have too many, trim
        if len(sampled) > num_sequences:
            sampled = random.sample(sampled, num_sequences)

        self.sequences = sampled
        logger.info(f"[uCO3D] Sampled {len(self.sequences)} representative sequences "
              f"from {num_categories} categories")

    def __len__(self):
        return len(self.sequences) * self.samples_per_sequence

    # Return a valid but empty sample to avoid DDP hangs on bad sequences
    def _make_fallback_sample(self, category: str) -> Dict:
        prompt = normalize_prompt(category) if self.normalize_prompts else category
        return {
            'images': torch.zeros(self.num_views, 3, self.image_size[0], self.image_size[1]),
            'gt_masks': torch.zeros(self.num_views, self.mask_size[0], self.mask_size[1]),
            'intrinsics': torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1),
            'extrinsics': torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1),
            'orig_hw': self.image_size,
            'scene_id': 'fallback_bad_sequence',
            'prompt': prompt,
            'image_names': ['fallback'] * self.num_views,
            'has_metric_scale': True,
            'has_gt_mask': False,  # Signal to training loop: skip loss for this sample
            'category': category,
        }

    # Load specific frames from video file
    def _load_frames_from_video(
        self,
        video_path: Path,
        frame_indices: List[int],
        is_mask: bool = False
    ) -> List[np.ndarray]:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:
                idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                if is_mask:
                    # Mask video is grayscale, threshold to binary
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = (frame > 127).astype(np.float32)
                else:
                    # RGB video
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Fallback: return zeros
                if is_mask:
                    frames.append(np.zeros((self.mask_size[0], self.mask_size[1]), dtype=np.float32))
                else:
                    frames.append(np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8))

        cap.release()
        return frames

    # Load depth maps from HDF5 file
    def _load_depth_from_h5(
        self,
        h5_path: Path,
        frame_indices: List[int]
    ) -> List[np.ndarray]:
        if not HAS_H5PY:
            return [np.ones((self.image_size[0], self.image_size[1]), dtype=np.float32)
                    for _ in frame_indices]

        if not h5_path.exists():
            return [np.ones((self.image_size[0], self.image_size[1]), dtype=np.float32)
                    for _ in frame_indices]

        depths = []
        with h5py.File(h5_path, 'r') as f:
            # uCO3D stores 200 equidistant depth maps
            depth_keys = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            total_depths = len(depth_keys)

            for idx in frame_indices:
                # Map frame index to depth index (200 depths for full video)
                depth_idx = min(idx, total_depths - 1)
                if depth_idx < len(depth_keys):
                    depth = np.array(f[depth_keys[depth_idx]])
                    depths.append(depth)
                else:
                    depths.append(np.ones((self.image_size[0], self.image_size[1]), dtype=np.float32))

        return depths

    # Uniformly sample frame indices
    def _sample_frame_indices(self, total_frames: int, num_samples: int) -> List[int]:
        if total_frames <= num_samples:
            return list(range(total_frames))

        # Uniform sampling
        step = total_frames / num_samples
        return [int(i * step) for i in range(num_samples)]

    def __getitem__(self, idx):
        # Map index to sequence
        seq_idx = idx % len(self.sequences)
        sample_idx = idx // len(self.sequences)

        seq = self.sequences[seq_idx]
        category = seq['category']

        try:
            # Use uCO3D package if available
            if HAS_UCO3D and hasattr(self, 'uco3d_dataset'):
                return self._getitem_uco3d(seq, sample_idx)
            else:
                return self._getitem_standalone(seq, sample_idx)
        except Exception as e:
            # Bad video/sequence, return a fallback sample to avoid DDP hang
            logger.warning(f"[uCO3D] Failed to load sequence {seq.get('sequence_name', seq_idx)}: {e}")
            return self._make_fallback_sample(category)

    # Load sample using uCO3D package
    def _getitem_uco3d(self, seq: Dict, sample_idx: int) -> Dict:
        frame_indices = seq['frame_indices']

        # Sample views from available frames
        if len(frame_indices) > self.num_views:
            # Random sampling for training, uniform for eval
            if self.split == 'train':
                random.seed(sample_idx)  # Reproducible per sample
                selected_indices = random.sample(frame_indices, self.num_views)
            else:
                step = len(frame_indices) // self.num_views
                selected_indices = [frame_indices[i * step] for i in range(self.num_views)]
        else:
            selected_indices = frame_indices[:self.num_views]

        images = []
        masks = []
        depths = []
        intrinsics_list = []
        extrinsics_list = []

        for fidx in selected_indices:
            frame_data = self.uco3d_dataset[fidx]

            # RGB image: [3, H, W]
            img = frame_data.image_rgb
            if img is not None:
                images.append(img)
            else:
                images.append(torch.zeros(3, self.image_size[0], self.image_size[1]))

            # Mask: [1, H, W] -> [H, W]
            mask = frame_data.fg_mask
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.squeeze(0)
                # Resize to mask_size
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=self.mask_size,
                    mode='nearest'
                ).squeeze()
                masks.append(mask_resized)
            else:
                masks.append(torch.zeros(self.mask_size))

            # Depth
            if self.use_depth and frame_data.depth_map is not None:
                depth = frame_data.depth_map
                if depth.dim() == 3:
                    depth = depth.squeeze(0)
                depths.append(depth)

            # Camera parameters from uCO3D
            cam = frame_data.camera
            if cam is not None:
                # uCO3D uses PyTorch3D convention, convert to standard 4x4 extrinsics
                R = cam.R[0] if cam.R.dim() == 3 else cam.R  # [3, 3]
                T = cam.T[0] if cam.T.dim() == 2 else cam.T  # [3]

                extrinsics = torch.eye(4)
                extrinsics[:3, :3] = R
                extrinsics[:3, 3] = T
                extrinsics_list.append(extrinsics)

                # Intrinsics from focal length and principal point
                focal = cam.focal_length[0] if cam.focal_length.dim() == 2 else cam.focal_length
                pp = cam.principal_point[0] if cam.principal_point.dim() == 2 else cam.principal_point

                fx, fy = focal[0].item(), focal[1].item()
                cx, cy = pp[0].item(), pp[1].item()

                intrinsics = torch.tensor([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=torch.float32)
                intrinsics_list.append(intrinsics)
            else:
                intrinsics_list.append(torch.eye(3))
                extrinsics_list.append(torch.eye(4))

        # Normalize prompt if enabled
        category = seq['category']
        prompt = normalize_prompt(category) if self.normalize_prompts else category

        result = {
            'images': torch.stack(images),  # [N, 3, H, W]
            'gt_masks': torch.stack(masks),  # [N, H, W]
            'intrinsics': torch.stack(intrinsics_list),  # [N, 3, 3]
            'extrinsics': torch.stack(extrinsics_list),  # [N, 4, 4]
            'orig_hw': self.image_size,
            'scene_id': seq['sequence_name'],
            'prompt': prompt,  # Normalized category name as text prompt
            'image_names': [str(i) for i in selected_indices],
            'has_metric_scale': True,  # uCO3D depth is aligned with VGGSfM
            'has_gt_mask': True,
            'category': category,  # Keep original category for metrics
        }

        if depths:
            result['depths'] = torch.stack(depths)  # [N, H, W]

        return result

    # Load sample using direct video loading (no uCO3D package)
    def _getitem_standalone(self, seq: Dict, sample_idx: int) -> Dict:
        import cv2

        rgb_video = seq['rgb_video']
        mask_video = seq['mask_video']
        depth_file = seq.get('depth_file')

        # Get total frames from RGB video
        cap = cv2.VideoCapture(str(rgb_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Sample frame indices uniformly from the video
        candidate_frames = self._sample_frame_indices(total_frames, self.frames_per_sequence)

        # Select views from candidate frames
        if len(candidate_frames) > self.num_views:
            if self.split == 'train':
                random.seed(sample_idx + self.seed)
                selected_frames = random.sample(candidate_frames, self.num_views)
            else:
                step = len(candidate_frames) // self.num_views
                selected_frames = [candidate_frames[i * step] for i in range(self.num_views)]
        else:
            selected_frames = candidate_frames[:self.num_views]
            # Pad with duplicates if needed
            while len(selected_frames) < self.num_views:
                selected_frames.append(selected_frames[-1])

        # Load RGB frames
        rgb_frames = self._load_frames_from_video(rgb_video, selected_frames, is_mask=False)

        # Load mask frames
        mask_frames = self._load_frames_from_video(mask_video, selected_frames, is_mask=True)

        # Load depth if available
        depth_frames = None
        if self.use_depth and depth_file and depth_file.exists():
            depth_frames = self._load_depth_from_h5(depth_file, selected_frames)

        # Process frames
        images = []
        masks = []
        depths = []

        for i, (rgb, mask) in enumerate(zip(rgb_frames, mask_frames)):
            # Resize and convert RGB
            rgb_pil = Image.fromarray(rgb)
            rgb_pil = rgb_pil.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            rgb_tensor = torch.from_numpy(np.array(rgb_pil)).float() / 255.0
            rgb_tensor = rgb_tensor.permute(2, 0, 1)  # [3, H, W]
            images.append(rgb_tensor)

            # Resize mask
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((self.mask_size[1], self.mask_size[0]), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
            masks.append(mask_tensor)

            # Resize depth
            if depth_frames is not None and i < len(depth_frames):
                depth = depth_frames[i]
                depth_pil = Image.fromarray(depth.astype(np.float32), mode='F')
                depth_pil = depth_pil.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
                depth_tensor = torch.from_numpy(np.array(depth_pil)).float()
                depths.append(depth_tensor)

        # Default intrinsics/extrinsics (turntable assumption)
        # In standalone mode, we don't have camera params, use identity
        intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1)
        extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1)

        # Normalize prompt if enabled
        category = seq['category']
        prompt = normalize_prompt(category) if self.normalize_prompts else category

        result = {
            'images': torch.stack(images),  # [N, 3, H, W]
            'gt_masks': torch.stack(masks),  # [N, H, W]
            'intrinsics': intrinsics,  # [N, 3, 3]
            'extrinsics': extrinsics,  # [N, 4, 4]
            'orig_hw': self.image_size,
            'scene_id': seq['sequence_name'],
            'prompt': prompt,  # Normalized category name
            'image_names': [str(f) for f in selected_frames],
            'has_metric_scale': True,
            'has_gt_mask': True,
            'category': category,  # Keep original for metrics
        }

        if depths:
            result['depths'] = torch.stack(depths)  # [N, H, W]

        # Load cached DA3 depth if available (bypasses live DA3 inference)
        if self.use_cached_depth and self.da3_cache_dir is not None:
            seq_cache_dir = (self.da3_cache_dir / seq['super_category']
                             / seq['category'] / seq['sequence_name'])
            cached_depths = []
            cached_extrinsics = []
            cached_intrinsics = []
            all_cached = True
            has_poses = True

            for frame_idx in selected_frames:
                cache_path = seq_cache_dir / f"frame_{frame_idx:06d}.pt"
                if cache_path.exists():
                    try:
                        cache_data = torch.load(cache_path, map_location='cpu',
                                                weights_only=True, mmap=False)
                        depth = cache_data['depth']
                        if depth.dim() == 2:
                            depth = depth.unsqueeze(0)  # [H, W] -> [1, H, W]
                        cached_depths.append(depth.float())

                        if 'extrinsics' in cache_data:
                            cached_extrinsics.append(cache_data['extrinsics'].float())
                        else:
                            has_poses = False
                        if 'intrinsics' in cache_data:
                            cached_intrinsics.append(cache_data['intrinsics'].float())
                    except Exception:
                        all_cached = False
                        break
                else:
                    all_cached = False
                    break

            if all_cached and len(cached_depths) == len(selected_frames):
                cached_depth_tensor = torch.stack(cached_depths)  # (N, 1, H, W)

                # Resize cached depth to image_size if needed
                cache_h, cache_w = cached_depth_tensor.shape[-2:]
                target_h, target_w = self.image_size
                if cache_h != target_h or cache_w != target_w:
                    cached_depth_tensor = F.interpolate(
                        cached_depth_tensor,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False,
                    )

                result['cached_depth'] = cached_depth_tensor  # (N, 1, H, W)

                if has_poses and len(cached_extrinsics) == len(selected_frames):
                    result['cached_da3_extrinsics'] = torch.stack(cached_extrinsics)
                if len(cached_intrinsics) == len(selected_frames):
                    result['cached_da3_intrinsics'] = torch.stack(cached_intrinsics)

        return result

if __name__ == '__main__':
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--num-sequences', type=int, default=5)
    parser.add_argument('--num-views', type=int, default=4)
    args = parser.parse_args()

    print("Testing UCO3DMultiViewDataset...")

    dataset = UCO3DMultiViewDataset(
        data_root=args.data_root,
        split='train',
        num_views=args.num_views,
        num_sequences=args.num_sequences,
        frames_per_sequence=20,
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Images shape: {sample['images'].shape}")
        print(f"GT masks shape: {sample['gt_masks'].shape}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Scene ID: {sample['scene_id']}")
        print(f"Category: {sample['category']}")
