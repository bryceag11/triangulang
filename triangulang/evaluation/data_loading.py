"""Model and data loading for evaluation and demos."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from triangulang.models.triangulang_model import TrianguLangModel
from triangulang.utils.lora import LoRAManager, LoRALayer
from triangulang.utils.scannetpp_loader import normalize_label, is_excluded_frame
from triangulang import BPE_PATH as _BPE_PATH


class BaselineSAM3Wrapper(torch.nn.Module):
    """Wrapper around native SAM3 for baseline comparison.

    Runs SAM3's own text-prompted segmentation (encoder + decoder + seghead)
    without GASA, depth, or cross-view fusion. Matches TrianguLangModel's
    forward interface so the benchmark loop works unchanged.
    """

    def __init__(self, sam3_model, resolution=1008):
        super().__init__()
        self.sam3 = sam3_model
        self.resolution = resolution
        self.mask_selection = 'confidence'

    @torch.no_grad()
    def forward(self, images, text_prompts, gt_masks=None,
                gt_intrinsics=None, gt_extrinsics=None, **kwargs):
        from sam3.model.data_misc import FindStage

        device = images.device
        B = images.shape[0]

        sam3_images = (images - 0.5) / 0.5
        if sam3_images.shape[-2:] != (self.resolution, self.resolution):
            sam3_images = F.interpolate(sam3_images, size=(self.resolution, self.resolution),
                                        mode='bilinear', align_corners=False)

        backbone_out = {"img_batch_all_stages": sam3_images}
        backbone_out.update(self.sam3.backbone.forward_image(sam3_images))

        text_out = self.sam3.backbone.forward_text(text_prompts, device=device)
        backbone_out.update(text_out)

        find_input = FindStage(
            img_ids=torch.arange(B, device=device, dtype=torch.long),
            text_ids=torch.arange(B, device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=B)

        outputs = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )

        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']

        scores = pred_logits.sigmoid()
        if 'presence_logit_dec' in outputs:
            presence = outputs['presence_logit_dec'].sigmoid()
            scores = scores.squeeze(-1) * presence
        else:
            scores = scores.squeeze(-1)

        best_idx = scores.argmax(dim=-1)
        batch_idx = torch.arange(B, device=device)
        best_masks = pred_masks[batch_idx, best_idx]

        return {
            'pred_masks': best_masks.unsqueeze(1),
            'all_masks': pred_masks,
        }


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_model(checkpoint_path: str, device: str = 'cuda', da3_resolution: int = None,
               num_queries: int = None, skip_trained_seghead: bool = False,
               train_config_path: str = None, resolution: int = None) -> TrianguLangModel:
    """Load trained GASA decoder model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        da3_resolution: Override DA3 resolution (default: use config or 504)
        resolution: Override model resolution/image size (default: use config or 1008)
        train_config_path: Optional explicit path to training config.json
    """
    from sam3 import build_sam3_image_model
    from depth_anything_3.api import DepthAnything3

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_dir = Path(checkpoint_path).parent

    # Try explicit config path first, then default locations
    config = {}
    config_path = None
    if train_config_path:
        config_path = Path(train_config_path)
    else:
        config_path = ckpt_dir / 'config.json'
        if not config_path.exists():
            config_path = ckpt_dir.parent.parent / 'runs' / 'train' / ckpt_dir.name / 'config.json'
        if not config_path.exists():
            config_path = ckpt_dir.parent.parent / 'runs' / 'ablations' / ckpt_dir.name / 'config.json'

    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"WARNING: Config not found, using defaults!")

    effective_da3_res = da3_resolution if da3_resolution is not None else (config.get('da3_resolution') or 504)
    print(f"Config: use_box_prompts={config.get('use_box_prompts', False)}, "
          f"use_point_prompts={config.get('use_point_prompts', False)}, "
          f"use_world_pe={config.get('use_world_pe', True)}, "
          f"use_gasa={config.get('use_gasa', True)}, "
          f"mask_selection={config.get('mask_selection', 'iou_match')}, "
          f"use_iou_head={config.get('use_iou_head', False)}, "
          f"use_spatial_tokens={config.get('use_spatial_tokens', False)}, "
          f"pe_type={config.get('pe_type', 'world')}, "
          f"per_layer_text={config.get('per_layer_text', False)}, "
          f"da3_resolution={effective_da3_res}")

    print("Loading SAM3...")
    sam3_resolution = resolution if resolution is not None else config.get('resolution', 1008)
    res_source = f"--image-size override ({resolution})" if resolution is not None else "config"
    print(f"  SAM3 img_size={sam3_resolution} ({res_source})")
    sam3_model = build_sam3_image_model(bpe_path=_BPE_PATH, img_size=sam3_resolution).to(device)

    print("Loading DA3...")
    da3_model = DepthAnything3.from_pretrained(
        config.get('da3_model', 'depth-anything/DA3METRIC-LARGE'),
        device=device
    )

    model = TrianguLangModel(
        sam3_model=sam3_model,
        da3_model=da3_model,
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        num_queries=num_queries if num_queries is not None else config.get('num_queries', 100),
        train_seghead=config.get('train_seghead', False),
        attn_map_size=config.get('attn_map_size', 64),
        use_presence_token=config.get('use_presence_token', True),
        use_box_prompts=config.get('use_box_prompts', False),
        use_point_prompts=config.get('use_point_prompts', False),
        num_pos_points=config.get('num_pos_points', 10),
        num_neg_points=config.get('num_neg_points', 2),
        use_world_pe=config.get('use_world_pe', True),
        use_gasa=config.get('use_gasa', True),
        gasa_beta_init=config.get('gasa_beta_init', 1.0),
        gasa_kernel_dim=config.get('gasa_kernel_dim', 32),
        gasa_fixed_kernel=config.get('gasa_fixed_kernel', False),
        gasa_kernel_type=config.get('gasa_kernel_type', 'learned'),
        mask_selection=config.get('mask_selection', 'iou_match'),
        use_iou_head=config.get('use_iou_head', False),
        use_spatial_tokens=config.get('use_spatial_tokens', False),
        use_spatial_points=config.get('use_spatial_points', False),
        use_object_aware_spatial=config.get('use_object_aware_spatial', False),
        use_centroid_head=config.get('use_centroid_head', False),
        use_iterative_pos=config.get('use_iterative_pos', False),
        cross_view=config.get('cross_view', True),
        pe_type=config.get('pe_type', 'world'),
        pointmap_normalize=config.get('pointmap_normalize', True),
        resolution=sam3_resolution,
        da3_resolution=da3_resolution if da3_resolution is not None else (config.get('da3_resolution') or 504),
        per_layer_text=config.get('per_layer_text', False),
        pred_logits_source=config.get('pred_logits_source', 'mask_mean'),
        use_da3_poses_for_gasa=config.get('use_da3_poses_for_gasa', False),
        use_gt_poses_for_gasa=config.get('use_gt_poses_for_gasa', False),
        da3_model_name=config.get('da3_model', 'depth-anything/DA3METRIC-LARGE').split('/')[-1],
        query_proj_mlp=config.get('query_proj_mlp', False),
        no_query_proj=config.get('no_query_proj', False),
        train_mask_embed=config.get('train_mask_embed', False),
        use_mask_refiner=config.get('use_mask_refiner', False),
        dim_feedforward=config.get('dim_feedforward', 2048),
        post_norm=config.get('post_norm', True),
        use_query_pe=config.get('use_query_pe', False),
        ffn_fp32=config.get('ffn_fp32', True),
        no_initial_text=config.get('no_initial_text', False),
        no_text_proj=config.get('no_text_proj', False),
        clean_v=config.get('clean_v', False),
        additive_pe=config.get('additive_pe', False),
        gasa_bidirectional=config.get('gasa_bidirectional', False),
        use_image_to_token=config.get('use_image_to_token', False),
        use_pos_refine=config.get('use_pos_refine', False),
        use_box_rpb=config.get('use_box_rpb', False),
        use_spatial_attn_bias=config.get('use_spatial_attn_bias', False),
        use_text_spatial_bias=config.get('use_text_spatial_bias', False),
        use_depth_crossattn=config.get('use_depth_crossattn', False),
        grouped_text_attn=config.get('grouped_text_attn', False),
        per_text_decode=config.get('per_text_decode', False),
    ).to(device)

    # Load checkpoint weights
    missing, unexpected = model.gasa_decoder.load_state_dict_compat(checkpoint['gasa_decoder'], strict=False)
    if missing:
        print(f"WARNING: Missing keys in gasa_decoder: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys in gasa_decoder: {unexpected}")
    if not config.get('no_query_proj', False):
        model.query_proj.load_state_dict(checkpoint['query_proj'], strict=False)

    if 'mask_refiner' in checkpoint and checkpoint['mask_refiner'] is not None:
        model.mask_refiner.load_state_dict(checkpoint['mask_refiner'])
        print("  Loaded mask_refiner weights")

    if 'sam3_seghead' in checkpoint and checkpoint['sam3_seghead'] is not None:
        if skip_trained_seghead:
            print("Skipping trained SAM3 seghead (--skip-trained-seghead), using default SAM3 weights")
        else:
            model.sam3.segmentation_head.load_state_dict(checkpoint['sam3_seghead'])
            print(f"Loaded trained SAM3 seghead from checkpoint")

    if 'mask_embed' in checkpoint and checkpoint['mask_embed'] is not None:
        model.sam3.segmentation_head.mask_predictor.mask_embed.load_state_dict(checkpoint['mask_embed'])
        print(f"Loaded trained mask_embed from checkpoint")

    if 'lora' in checkpoint and checkpoint['lora'] is not None:
        import torch.nn as nn
        lora_rank = config.get('lora_rank', 8)
        lora_alpha = config.get('lora_alpha', 16.0)
        lora_manager = LoRAManager(rank=lora_rank, alpha=lora_alpha)
        if config.get('lora_sam3', False):
            lora_manager.add_lora_to_model(model.sam3, "sam3")
        if config.get('lora_da3', False):
            lora_manager.add_lora_to_model(model.da3, "da3")
        if config.get('lora_mask_embed', False):
            mask_pred = model.sam3.segmentation_head.mask_predictor
            for i, layer in enumerate(mask_pred.mask_embed.layers):
                if isinstance(layer, nn.Linear):
                    adapter = LoRALayer(layer.in_features, layer.out_features,
                                        rank=lora_rank, alpha=lora_alpha)
                    adapter_name = f"mask_embed_layer{i}"
                    lora_manager.adapters[adapter_name] = adapter
                    hook = lora_manager._create_hook(adapter)
                    handle = layer.register_forward_hook(hook)
                    lora_manager.hooks.append(handle)
                    lora_manager._adapter_count += 1
        lora_manager.load_state_dict(checkpoint['lora'])
        lora_manager.to(device)
        print(f"Loaded LoRA state ({lora_manager.num_adapters} adapters, {lora_manager.num_parameters:,} params)")

    model.sam3_multi_object = config.get('sam3_multi_object', False)
    model.multi_object = config.get('multi_object', False)
    if model.sam3_multi_object:
        print(f"SAM3 multi-object mode: ENABLED (batch expansion)")

    model.config = config  # Store for downstream use
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, best_iou={checkpoint.get('best_iou', 0)*100:.2f}%")
    return model


def load_scene_data(scene_path: Path, semantics_dir: Path) -> Tuple[List[Path], Dict, List[str]]:
    """Load scene images and annotations."""
    image_dir = scene_path / "dslr" / "resized_undistorted_images"
    if not image_dir.exists():
        image_dir = scene_path / "dslr" / "resized_images"

    images = sorted(image_dir.glob("*.JPG")) + sorted(image_dir.glob("*.jpg"))

    scene_id = scene_path.name
    before = len(images)
    images = [img for img in images if not is_excluded_frame(scene_id, img.stem)]
    if len(images) < before:
        print(f"  Excluded {before - len(images)} bad frames from {scene_id}")

    anno_path = scene_path / "scans" / "segments_anno.json"
    objects = {}
    if anno_path.exists():
        with open(anno_path) as f:
            anno_data = json.load(f)
        for group in anno_data.get('segGroups', []):
            obj_id = group.get('objectId') or group.get('id')
            label = normalize_label(group.get('label', 'unknown')).lower()
            if obj_id is not None:
                objects[obj_id] = {
                    'label': label,
                    'segments': group.get('segments', []),
                }

    available_frames = []
    if semantics_dir.exists():
        for pth_file in semantics_dir.glob("*.pth"):
            frame_name = pth_file.stem
            if not is_excluded_frame(scene_id, frame_name):
                available_frames.append(frame_name)

    return images, objects, available_frames


def load_gt_masks(semantics_dir: Path, frame_name: str) -> Dict[int, np.ndarray]:
    """Load ground truth masks for a frame."""
    mask_path = semantics_dir / f"{frame_name}.pth"
    if mask_path.exists():
        try:
            instance_map = torch.load(mask_path, weights_only=False)
            if isinstance(instance_map, torch.Tensor):
                instance_map = instance_map.numpy()
            masks = {}
            for obj_id in np.unique(instance_map):
                if obj_id > 0:
                    masks[int(obj_id)] = (instance_map == obj_id).astype(np.float32)
            return masks
        except Exception as e:
            print(f"Error loading {mask_path}: {e}")
            return {}
    return {}


def load_gt_poses(scene_path: Path) -> Tuple[Optional[Dict], Optional[torch.Tensor]]:
    """Load ground truth camera poses from transforms.json (NeRFStudio format)."""
    transforms_path = scene_path / 'dslr' / 'nerfstudio' / 'transforms_undistorted.json'
    if not transforms_path.exists():
        transforms_path = scene_path / 'dslr' / 'nerfstudio' / 'transforms.json'
    if not transforms_path.exists():
        return None, None

    try:
        with open(transforms_path) as f:
            transforms = json.load(f)
        fx = transforms.get('fl_x', 500)
        fy = transforms.get('fl_y', 500)
        cx = transforms.get('cx', 256)
        cy = transforms.get('cy', 256)
        intrinsics = torch.tensor([
            [fx, 0, cx], [0, fy, cy], [0, 0, 1]
        ], dtype=torch.float32)
        return transforms, intrinsics
    except Exception as e:
        print(f"Error loading transforms from {transforms_path}: {e}")
        return None, None


def get_frame_extrinsics(transforms: Dict, frame_name: str) -> Optional[torch.Tensor]:
    """Get extrinsics (4x4 camera-to-world transform) for a specific frame."""
    if transforms is None:
        return None
    for frame in transforms.get('frames', []):
        file_path = frame.get('file_path', '')
        if Path(file_path).name == frame_name:
            transform_matrix = frame.get('transform_matrix')
            if transform_matrix:
                return torch.tensor(transform_matrix, dtype=torch.float32)
    return None


def load_cached_da3_nested(
    cache_dir: Path, scene_id: str, frame_names: List[str],
) -> Optional[Dict[str, torch.Tensor]]:
    """Load cached DA3-NESTED outputs for a scene.

    Returns dict with 'depths' [N, H, W], 'extrinsics' [N, 4, 4], 'intrinsics' [N, 3, 3],
    or None if cache not found.
    """
    scene_cache_dir = cache_dir / scene_id
    if not scene_cache_dir.exists():
        return None
    manifest_path = scene_cache_dir / 'manifest.json'
    if not manifest_path.exists():
        return None

    depths, extrinsics, intrinsics = [], [], []
    for frame_name in frame_names:
        stem = Path(frame_name).stem
        cache_path = scene_cache_dir / f'{stem}.pt'
        if not cache_path.exists():
            return None
        try:
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            d = data['depth']
            depths.append(d.float() if isinstance(d, torch.Tensor) else torch.from_numpy(d.astype(np.float32)))
            e = data['extrinsics']
            extrinsics.append(e.float() if isinstance(e, torch.Tensor) else torch.from_numpy(e).float())
            i = data['intrinsics']
            intrinsics.append(i.float() if isinstance(i, torch.Tensor) else torch.from_numpy(i).float())
        except Exception as e:
            print(f"Error loading cached data for {scene_id}/{stem}: {e}")
            return None

    return {
        'depths': torch.stack(depths),
        'extrinsics': torch.stack(extrinsics),
        'intrinsics': torch.stack(intrinsics),
    }


def load_gt_centroids(data_root: Path) -> Dict:
    """Load GT 3D centroids from centroid_cache.json (computed from mesh)."""
    centroid_path = data_root / 'centroid_cache.json'
    if not centroid_path.exists():
        return {}
    with open(centroid_path) as f:
        return json.load(f)


def load_gt_poses_for_scene(data_root: Path, scene_id: str) -> Optional[Dict]:
    """Load GT poses from nerfstudio transforms for a scene."""
    nerf_path = data_root / 'data' / scene_id / 'dslr' / 'nerfstudio' / 'transforms_undistorted.json'
    if not nerf_path.exists():
        return None

    with open(nerf_path) as f:
        data = json.load(f)

    T_nerf_to_mesh = np.array([
        [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]
    ])

    poses = {}
    for frame in data['frames']:
        frame_name = Path(frame['file_path']).stem
        T = np.array(frame['transform_matrix'])
        poses[frame_name] = T_nerf_to_mesh @ T

    return poses
