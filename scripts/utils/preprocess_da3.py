"""Preprocess DA3 metric depth for ScanNet++ dataset.

Caches per-frame depth maps to disk for faster training.

Usage:
    torchrun --nproc_per_node=8 scripts/utils/preprocess_da3.py \
        --data-root /path/to/scannetpp
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "sam3"))
sys.path.insert(0, str(project_root / "depth_anything_v3" / "src"))

def get_scenes_dir(data_root: Path) -> Path:
    """Get the directory containing scene folders (handles nested 'data' folder)."""
    nested = data_root / "data"
    if nested.exists() and nested.is_dir():
        return nested
    return data_root

def load_scene_list(data_root: Path, split: str) -> list:
    """Load scene IDs from split file."""
    split_file = data_root / "splits" / f"{split}.txt"
    if not split_file.exists():
        for alt_path in [data_root / f"{split}.txt", data_root / "metadata" / f"{split}.txt"]:
            if alt_path.exists():
                split_file = alt_path
                break

    if not split_file.exists():
        logger.warning(f"Split file not found: {split_file}")
        return []

    with open(split_file) as f:
        scenes = [line.strip() for line in f if line.strip()]
    return scenes

def main():
    parser = argparse.ArgumentParser(description='Preprocess DA3 depth for ScanNet++')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to ScanNet++ data root (e.g., /path/to/data/scannetpp)')
    parser.add_argument('--split', type=str, default='nvs_sem_train',
                        help='Dataset split to process')
    parser.add_argument('--da3-model', type=str, default='depth-anything/DA3METRIC-LARGE',
                        help='DA3 model to use')
    parser.add_argument('--da3-resolution', type=int, default=1008,
                        help='DA3 inference resolution (higher = more accurate, but slower)')
    parser.add_argument('--sam3-resolution', type=int, default=1008,
                        help='SAM3 resolution (depth saved at this resolution)')
    parser.add_argument('--use-undistorted', action='store_true', default=True,
                        help='Use undistorted images')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Max scenes to process (for testing)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing cache files')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Batch size per GPU (higher = faster, more VRAM)')
    args = parser.parse_args()

    # Setup distributed mode if available
    local_rank = 0
    world_size = 1

    if 'LOCAL_RANK' in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (local_rank == 0)

    if is_main:
        logger.info(f"Preprocessing DA3 depth for ScanNet++")
        logger.info(f"  Data root: {args.data_root}")
        logger.info(f"  Split: {args.split}")
        logger.info(f"  DA3 model: {args.da3_model}")
        logger.info(f"  DA3 resolution: {args.da3_resolution} -> SAM3 resolution: {args.sam3_resolution}")
        logger.info(f"  World size: {world_size}")

    # Load DA3 model
    if is_main:
        logger.info(f"[Rank {local_rank}] Loading DA3 model...")

    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained(args.da3_model).to(device)
    da3.eval()

    if is_main:
        logger.info(f"  DA3 loaded successfully")

    # Get scene list
    data_root = Path(args.data_root)
    scenes_dir = get_scenes_dir(data_root)
    scenes_from_split = load_scene_list(data_root, args.split)
    split_count = len(scenes_from_split)

    # Filter to scenes that actually exist with images
    scenes = []
    for scene_id in scenes_from_split:
        scene_path = scenes_dir / scene_id
        if args.use_undistorted:
            images_dir = scene_path / "dslr" / "resized_undistorted_images"
        else:
            images_dir = scene_path / "dslr" / "resized_images"
        if images_dir.exists():
            scenes.append(scene_id)

    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    if is_main:
        logger.info(f"Found {len(scenes)} scenes with images (from {split_count} in split file)")

    # Create cache directory
    cache_dir = data_root / "da3_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images to process
    all_images = []
    for scene_id in scenes:
        scene_path = scenes_dir / scene_id
        if args.use_undistorted:
            images_dir = scene_path / "dslr" / "resized_undistorted_images"
        else:
            images_dir = scene_path / "dslr" / "resized_images"

        if not images_dir.exists():
            continue

        scene_cache_dir = cache_dir / scene_id
        scene_cache_dir.mkdir(exist_ok=True)

        for img_path in sorted(images_dir.glob("*.JPG")):
            cache_path = scene_cache_dir / f"{img_path.stem}.pt"
            if cache_path.exists() and not args.overwrite:
                continue
            all_images.append({
                'img_path': img_path,
                'cache_path': cache_path,
                'scene_id': scene_id
            })

    if is_main:
        logger.info(f"Images to process: {len(all_images)}")

    # Distribute work across GPUs (round-robin)
    my_images = all_images[local_rank::world_size]

    # Show work distribution across all ranks
    logger.info(f"[Rank {local_rank}] Processing {len(my_images)} images on GPU {local_rank}")

    # Process images in batches for better GPU utilization
    from PIL import Image
    import numpy as np

    PATCH_SIZE = 14
    IMAGENET_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def preprocess_image(pil_img, target_res):
        """Resize preserving aspect ratio, pad to patch-aligned, ImageNet-normalize."""
        w, h = pil_img.size
        scale = target_res / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Make divisible by patch size
        new_w = (new_w // PATCH_SIZE) * PATCH_SIZE
        new_h = (new_h // PATCH_SIZE) * PATCH_SIZE
        new_w = max(new_w, PATCH_SIZE)
        new_h = max(new_h, PATCH_SIZE)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        # ToTensor converts [0,255] -> [0,1], then apply ImageNet normalization
        img_tensor = T.ToTensor()(pil_img)
        img_tensor = IMAGENET_NORMALIZE(img_tensor)
        return img_tensor, (new_h, new_w)

    da3_res = (args.da3_resolution // 14) * 14  # Patch-aligned
    batch_size = args.batch_size

    # Chunk images into batches
    batches = [my_images[i:i + batch_size] for i in range(0, len(my_images), batch_size)]
    pbar = tqdm(batches, disable=(not is_main), desc=f"Processing (batch={batch_size})")

    for batch in pbar:
        try:
            # Load and preprocess batch of images
            batch_tensors = []
            batch_meta = []

            for item in batch:
                img_path = item['img_path']
                image = Image.open(img_path).convert('RGB')
                orig_h, orig_w = image.height, image.width

                image_tensor, (proc_h, proc_w) = preprocess_image(image, da3_res)
                batch_tensors.append(image_tensor)
                batch_meta.append({
                    'cache_path': item['cache_path'],
                    'orig_hw': (orig_h, orig_w),
                    'proc_hw': (proc_h, proc_w),
                })

            # Pad batch to same spatial size (images may differ slightly after
            # aspect-ratio-preserving resize if originals have different ratios)
            max_h = max(t.shape[1] for t in batch_tensors)
            max_w = max(t.shape[2] for t in batch_tensors)
            padded_tensors = []
            for t in batch_tensors:
                pad_h = max_h - t.shape[1]
                pad_w = max_w - t.shape[2]
                if pad_h > 0 or pad_w > 0:
                    t = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
                padded_tensors.append(t)

            # Stack into batch [B, 3, H, W]
            batch_tensor = torch.stack(padded_tensors).to(device)

            with torch.no_grad():
                # Run DA3 on batch [B, 1, 3, H, W] for video format
                da3_output = da3.model.forward(
                    batch_tensor.unsqueeze(1), extrinsics=None, intrinsics=None,
                    export_feat_layers=[], infer_gs=False
                )
                depths = da3_output.depth  # [B, 1, H, W]

                # Upsample to SAM3 resolution if needed
                if depths.shape[-1] != args.sam3_resolution or depths.shape[-2] != args.sam3_resolution:
                    depths = F.interpolate(depths, size=(args.sam3_resolution, args.sam3_resolution),
                                          mode='bilinear', align_corners=False)

                # Save each depth individually
                for i, meta in enumerate(batch_meta):
                    cache_data = {
                        'depth': depths[i:i+1].cpu().half(),  # [1, 1, H, W]
                        'da3_resolution': da3_res,
                        'sam3_resolution': args.sam3_resolution,
                        'orig_hw': meta['orig_hw'],
                        'proc_hw': meta['proc_hw'],
                    }
                    torch.save(cache_data, meta['cache_path'])

        except Exception as e:
            logger.warning(f"[Rank {local_rank}] Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        print(f"\nDone! Cache saved to {cache_dir}")
        print(f"  Depth saved at {args.sam3_resolution}x{args.sam3_resolution} resolution")
        print(f"  Training will use cached depth directly (no upsampling needed)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
