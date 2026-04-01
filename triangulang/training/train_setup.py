"""Training setup: config, environment, model building, checkpointing."""
import os
import sys
import json
import random
import time
import math
import gc
import psutil
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from sam3 import build_sam3_image_model
from depth_anything_3.api import DepthAnything3
from triangulang import BPE_PATH as _BPE_PATH

PROJECT_ROOT = Path(__file__).parent.parent.parent
from triangulang.models.triangulang_model import TrianguLangModel
from triangulang.utils.ddp_utils import DDPManager
from triangulang.utils.lora import LoRALayer, LoRAManager
from triangulang.utils.metrics import CategoryMetricsTracker
from triangulang.data.dataset_factory import get_dataset, get_dataset_config
from triangulang.training.config import TrainConfig
from torch.amp import GradScaler
from triangulang.utils.spatial_reasoning import SpatialAugmentor
from triangulang.utils.spatial_context import GTAwareSpatialAugmentor
from triangulang.losses.sheaf_losses import FeatureSheafLoss
from triangulang.training.train_helpers import set_seed, collate_fn, run_validation, visualize_predictions
from triangulang.utils.metrics import CategoryMetricsTracker
from triangulang.data.dataset_factory import get_dataset, get_dataset_config
from triangulang.utils.scannetpp_loader import ScanNetPPMultiViewDataset


def _setup_config(args):
    parser_defaults = TrainConfig.get_parser_defaults()

    if args.sam3_defaults:
        sam3_overrides = {
            'pred_logits_source': 'text_scoring',
            'init_scoring_from_sam3': True,
            'no_initial_text': True,
            'no_text_proj': True,
            'clean_v': True,
            'per_layer_align': True,
            'init_text_crossattn_from_sam3': True,
            'init_decoder_from_sam3': True,
        }
        sam3_applied = []
        for key, value in sam3_overrides.items():
            current = getattr(args, key, None)
            default = parser_defaults.get(key)
            if current == default and current != value:
                setattr(args, key, value)
                sam3_applied.append(f"{key}={value}")
        if sam3_applied:
            print(f"[--sam3-defaults] Applied: {', '.join(sam3_applied)}")
        else:
            print(f"[--sam3-defaults] All SAM3 settings already active or overridden by CLI")

    if args.pe_type == 'none':
        args.use_world_pe = False

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            run_name = resume_path.name
        else:
            run_name = resume_path.parent.name

        possible_config_paths = [
            resume_path / 'config.json' if resume_path.is_dir() else resume_path.parent / 'config.json',
            Path(__file__).parent.parent.parent / 'runs' / 'train' / run_name / 'config.json',
        ]
        config_path = None
        for p in possible_config_paths:
            if p.exists():
                config_path = p
                break

        if config_path:
            import json
            with open(config_path) as f:
                saved_config = json.load(f)

            overridden = []
            loaded = []
            for key, value in saved_config.items():
                if hasattr(args, key):
                    current_value = getattr(args, key)
                    default_value = parser_defaults.get(key)
                    if current_value == default_value and value != default_value:
                        setattr(args, key, value)
                        loaded.append(key)
                    elif current_value != default_value and current_value != value:
                        overridden.append(f"{key}={current_value}")

            compat_defaults = {
                'dim_feedforward': 1024,
                'post_norm': False,
                'ffn_fp32': False,
            }
            compat_applied = []
            for key, old_default in compat_defaults.items():
                if key not in saved_config and hasattr(args, key):
                    current = getattr(args, key)
                    parser_def = parser_defaults.get(key)
                    if current == parser_def and current != old_default:
                        setattr(args, key, old_default)
                        compat_applied.append(f"{key}={old_default}")

            print(f"[Resume] Loaded config from {config_path}")
            if compat_applied:
                print(f"[Resume]   Backward compat (old config missing keys): {', '.join(compat_applied)}")
            if loaded:
                print(f"[Resume]   Restored {len(loaded)} settings: {', '.join(loaded[:10])}" +
                      (f"... and {len(loaded)-10} more" if len(loaded) > 10 else ""))
            if overridden:
                print(f"[Resume]   CLI overrides: {', '.join(overridden)}")
        else:
            print(f"[Resume] Warning: No config.json found for run '{run_name}', using CLI args only")
            print(f"[Resume]   Searched: {[str(p) for p in possible_config_paths]}")

    return args


def _init_environment(args, ddp):
    if args.run_name is None:
        args.run_name = f"gasa_decoder_{datetime.now().strftime('%m%d_%H%M')}"

    set_seed(args.seed, ddp.rank)
    device = ddp.device
    torch.backends.cudnn.benchmark = True

    ddp.print("TRIANGULANG: GASA DECODER (REPLACES SAM3's DECODER)")
    ddp.print(f"  World size: {ddp.world_size}, Rank: {ddp.rank}")

    da3_name = args.da3_model.split('/')[-1].upper()
    da3_has_pose = 'METRIC' not in da3_name and 'MONO' not in da3_name
    da3_has_metric = 'METRIC' in da3_name or 'NESTED' in da3_name
    ddp.print(f"  DA3 model: {da3_name}")
    ddp.print(f"    - Pose estimation capability: {da3_has_pose}")
    ddp.print(f"    - Metric depth: {da3_has_metric}")
    ddp.print(f"    - NOTE: Will use GT poses from dataset if available, otherwise identity pose")
    if args.use_sheaf_loss:
        ddp.print(f"  Sheaf loss: enabled (weight={args.sheaf_weight}, type={args.sheaf_type})")
        if args.sheaf_type == 'feature':
            ddp.print(f"    - NON-CONSTANT sheaf: learned restriction maps on R^256 -> R^{args.sheaf_d_edge}")
        else:
            ddp.print(f"    - Constant sheaf: identity restriction maps")
        ddp.print(f"    - Requires GT extrinsics from dataset for world-consistent pointmaps")
        ddp.print(f"    - If GT poses unavailable, falls back to camera-frame (less effective)")

    ddp.print(f"  GASA (geometric bias): {args.use_gasa}" + (" [ABLATION: disabled]" if not args.use_gasa else ""))
    if args.use_gt_poses_for_gasa:
        ddp.print(f"  GASA pointmaps: GT COLMAP poses (globally consistent)")
    elif args.use_da3_poses_for_gasa:
        ddp.print(f"  GASA pointmaps: DA3-NESTED estimated poses (chunk-consistent)")
    else:
        ddp.print(f"  GASA pointmaps: camera-frame (identity pose) [WARNING: no cross-view consistency]")
    kernel_desc = {'learned': f'learned MLP (dim={args.gasa_kernel_dim})', 'rbf': 'RBF exp(-d²/2σ²)', 'fixed': 'fixed φ(d) = -d'}
    bidir_str = " [BIDIRECTIONAL: boost+suppress]" if args.gasa_bidirectional else " [suppress-only]"
    ddp.print(f"  GASA kernel: {kernel_desc.get(args.gasa_kernel_type, args.gasa_kernel_type)}{bidir_str}")
    ddp.print(f"  Text proj: Linear(256→{args.d_model}) for cross-attention only (scoring uses raw text)")
    if args.pred_logits_source == 'text_scoring':
        sel_str = " (text-aware scoring, eval-only selection)"
    else:
        sel_str = " (text-agnostic mask mean)"
    ddp.print(f"  pred_logits source: {args.pred_logits_source}{sel_str}")
    ddp.print(f"  Depth cross-attention: {args.use_depth_crossattn}" + (" (queries attend to 3D positions)" if args.use_depth_crossattn else ""))
    ddp.print(f"  Iterative query positions: {args.use_iterative_pos}" + (" (P_Q = attn-weighted centroid)" if args.use_iterative_pos else " (P_Q = scene centroid)"))
    ddp.print(f"  Positional encoding: {args.pe_type}" + (" [ABLATION: disabled]" if args.pe_type == 'none' else ""))
    ddp.print(f"  Presence token: {args.use_presence_token} (weight={args.presence_weight}, focal={args.presence_focal}, α={args.presence_alpha}, γ={args.presence_gamma})")
    centroid_mode = " [triangulation]" if args.use_triangulation else (" [mask-based]" if args.mask_based_centroid else " [attention-based]")
    ddp.print(f"  Centroid head: {args.use_centroid_head}" + (f" (weight={args.centroid_weight}){centroid_mode}" if args.use_centroid_head else ""))
    if args.eval_localization and not args.use_centroid_head:
        ddp.print(f"  Eval localization: {args.eval_localization} (tracking Acc@m from mask+depth, no loss)")
    ddp.print(f"  Box prompts: {args.use_box_prompts}" + (f" (dropout={args.box_prompt_dropout})" if args.box_prompt_dropout > 0 else ""))
    ddp.print(f"  Point prompts: {args.use_point_prompts} ({args.num_pos_points} pos + {args.num_neg_points} neg)" + (f" (dropout={args.point_prompt_dropout})" if args.point_prompt_dropout > 0 else ""))
    ddp.print(f"  Prompt type: {args.prompt_type}")
    ddp.print(f"  Mask selection: {args.mask_selection}")
    if args.mask_selection == 'predicted_iou' and not args.use_iou_head:
        ddp.print("  WARNING: --mask-selection predicted_iou requires --use-iou-head! Falling back to confidence.")
        args.mask_selection = 'confidence'
    ddp.print(f"  Sheaf consistency loss: {args.use_sheaf_loss}" + (f" (type={args.sheaf_type}, weight={args.sheaf_weight}, threshold={args.sheaf_threshold}m)" if args.use_sheaf_loss else " [ABLATION: disabled]"))
    ddp.print(f"  Contrastive loss: {args.contrastive_weight > 0}" + (f" (weight={args.contrastive_weight}, margin={args.contrastive_margin}, source={args.contrastive_source})" if args.contrastive_weight > 0 else ""))
    ddp.print(f"  Align loss (SAM3-style): {args.align_weight > 0}" + (f" (weight={args.align_weight}, α={args.align_alpha}, γ={args.align_gamma}, τ={args.align_tau})" if args.align_weight > 0 else ""))
    ddp.print(f"  Lovász loss: {args.lovasz_weight > 0}" + (f" (weight={args.lovasz_weight})" if args.lovasz_weight > 0 else ""))
    ddp.print(f"  Point sampling: {args.use_point_sampling}" + (f" ({args.num_sample_points} points)" if args.use_point_sampling else ""))
    ddp.print(f"  Loss at native res: {args.loss_at_native_res}")
    ddp.print(f"  IoU head: {args.use_iou_head}" + (f" (MSE weight={args.iou_head_weight})" if args.use_iou_head else ""))
    ddp.print(f"  Semantic union GT: {args.semantic_union}" + (" (text='mug' matches ALL mugs)" if args.semantic_union else " [per-instance mode]"))
    ddp.print(f"  Class-balanced sampling: {args.class_balanced}" + (f" (power={args.class_balance_power})" if args.class_balanced else ""))
    if args.min_mask_coverage > 0:
        orig_pixels = int(args.min_mask_coverage * 1752 * 1168)
        ddp.print(f"  Min mask coverage: {args.min_mask_coverage*100:.2f}% (≈{orig_pixels} pixels at ~1752×1168 original)")
    else:
        ddp.print(f"  Min mask coverage: disabled (any non-empty mask is valid)")

    return device


def _load_datasets(args, ddp):
    import json
    dataset_config = get_dataset_config(args.dataset)
    data_root = args.data_root or str(PROJECT_ROOT / dataset_config.get('data_root', f'data/{args.dataset}'))
    split = args.split or dataset_config.get('split', 'train')

    ddp.print(f"\nLoading dataset '{args.dataset}'...")
    ddp.print(f"  Data root: {data_root}")
    ddp.print(f"  Split: {split}")

    native_mask_res = (args.resolution // 14) * 4
    ddp.print(f"  Mask size: {native_mask_res}x{native_mask_res} (native for resolution={args.resolution})")

    if args.multi_object:
        args.num_objects = 0
        if args.samples_per_scene == 1:
            target_batches = 5
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            per_gpu_scenes = max(1, args.max_scenes // world_size)
            min_needed = args.batch_size * target_batches
            if per_gpu_scenes < min_needed:
                args.samples_per_scene = max(2, (min_needed + per_gpu_scenes - 1) // per_gpu_scenes)
                ddp.print(f"  Auto-set --samples-per-scene {args.samples_per_scene} for multi-object "
                         f"({args.max_scenes} scenes / {world_size} GPUs = {per_gpu_scenes}/GPU, "
                         f"target ≥{target_batches} batches of {args.batch_size})")

    if args.num_objects != 1:
        if args.num_objects == 0:
            ddp.print(f"  Multi-object training: DYNAMIC K (all visible objects per sample, like SAM3)")
        else:
            ddp.print(f"  Multi-object training: K={args.num_objects} objects per sample")
        ddp.print(f"  Hungarian matching enabled: {args.num_queries} queries competing for K GT objects")

    dataset = get_dataset(
        dataset_name=args.dataset,
        data_root=data_root,
        split=split,
        views_per_sample=args.views,
        image_size=(args.resolution, args.resolution),
        mask_size=(native_mask_res, native_mask_res),
        max_scenes=args.max_scenes,
        use_undistorted=True,
        supervised=True,
        semantic_union=args.semantic_union,
        sampling_strategy=args.sampling_strategy,
        da3_chunk_size=args.da3_chunk_size,
        use_cached_depth=args.use_cached_depth,
        da3_cache_name=args.da3_cache_name,
        min_category_samples=args.min_category_samples,
        exclude_categories=args.exclude_categories,
        include_categories=args.include_categories,
        num_pos_points=args.num_pos_points,
        num_neg_points=args.num_neg_points,
        samples_per_scene=args.samples_per_scene,
        frames_per_sequence=args.frames_per_sequence,
        samples_per_sequence=args.samples_per_sequence,
        part_query_mode=args.part_query_mode,
        augment=args.augment,
        num_objects_per_sample=args.num_objects,
    )

    if args.dataset == 'scannetpp':
        ddp.print(f"  Sampling strategy: {args.sampling_strategy}")
        if args.sampling_strategy == 'chunk_aware':
            ddp.print(f"  DA3 chunk size: {args.da3_chunk_size}")
            ddp.print(f"  Note: Views will be sampled from same DA3-NESTED chunk for world-frame consistency")

    if args.use_cached_depth:
        ddp.print(f"  da3_cache_dir: {dataset.da3_cache_dir} ({args.da3_cache_name})")
        ddp.print(f"  da3_cache_dir exists: {dataset.da3_cache_dir.exists() if dataset.da3_cache_dir else 'N/A'}")
        test_sample = dataset[0]
        ddp.print(f"  Sample keys: {list(test_sample.keys())}")
        if 'cached_depth' in test_sample:
            ddp.print(f"  Cached depth ENABLED: shape={test_sample['cached_depth'].shape}")
            if 'cached_da3_extrinsics' in test_sample:
                ddp.print(f"  Cached DA3 poses ENABLED: extrinsics={test_sample['cached_da3_extrinsics'].shape}")
        else:
            ddp.print("  WARNING: --use-cached-depth set but cache not found! DA3 will run live.")

    sample_weights = None
    if args.class_balanced and getattr(dataset, 'object_samples', None):
        ddp.print("Computing class-balanced sample weights...")
        category_counts = Counter()
        for sample in dataset.object_samples:
            category = sample.get('label', sample.get('category', 'unknown'))
            category_counts[category] += 1

        total_samples = len(dataset.object_samples)
        num_categories = len(category_counts)
        sample_weights = []
        for sample in dataset.object_samples:
            category = sample.get('label', sample.get('category', 'unknown'))
            freq = category_counts[category] / total_samples
            weight = (1.0 / (freq * num_categories)) ** args.class_balance_power
            sample_weights.append(weight)

        weight_sum = sum(sample_weights)
        sample_weights = [w * len(sample_weights) / weight_sum for w in sample_weights]

        sorted_cats = category_counts.most_common()
        ddp.print(f"  Categories: {num_categories}, Samples: {total_samples}")
        ddp.print(f"  Most common: {sorted_cats[:3]}")
        ddp.print(f"  Least common: {sorted_cats[-3:]}")
        ddp.print(f"  Weight range: [{min(sample_weights):.2f}, {max(sample_weights):.2f}]")

    collate = partial(collate_fn, max_objects=args.max_objects) if args.max_objects > 0 else collate_fn
    if args.max_objects > 0 and args.num_objects != 1:
        ddp.print(f"  Max objects per sample: {args.max_objects} (capping K)")

    dataloader = ddp.wrap_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0 and not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        sample_weights=sample_weights,
    )
    ddp.print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/gpu")

    val_dataset = None
    val_dataloader = None
    if args.val_every > 0:
        ddp.print(f"\nLoading validation dataset (split='{args.val_split}')...")
        val_max_samples = args.val_max_samples
        val_part_query_mode = 'all' if args.dataset == 'partimagenet' else args.part_query_mode

        val_dataset = get_dataset(
            dataset_name=args.dataset,
            data_root=data_root,
            split=args.val_split,
            views_per_sample=args.views,
            image_size=(args.resolution, args.resolution),
            mask_size=(native_mask_res, native_mask_res),
            max_scenes=val_max_samples,
            use_undistorted=True,
            supervised=True,
            semantic_union=args.semantic_union,
            sampling_strategy=args.sampling_strategy,
            da3_chunk_size=args.da3_chunk_size,
            use_cached_depth=args.use_cached_depth,
            da3_cache_name=args.da3_cache_name,
            min_category_samples=1,
            exclude_categories=args.exclude_categories,
            include_categories=args.include_categories,
            num_pos_points=args.num_pos_points,
            num_neg_points=args.num_neg_points,
            samples_per_scene=1,
            frames_per_sequence=args.frames_per_sequence,
            samples_per_sequence=1,
            part_query_mode=val_part_query_mode,
            augment=False,
        )

        val_dataloader = ddp.wrap_dataloader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate, pin_memory=True, drop_last=False,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        ddp.print(f"Validation dataset: {len(val_dataset)} samples, {len(val_dataloader)} batches/gpu")
        ddp.print(f"  Validation every {args.val_every} epoch(s)")
        ddp.print(f"  Save best based on: {'validation mIoU' if args.save_best_val else 'training IoU'}")

    return dataset, dataloader, val_dataset, val_dataloader


def _build_model(args, ddp, device):
    ddp.print("\nLoading models...")
    sam3_model = build_sam3_image_model(bpe_path=_BPE_PATH, img_size=args.resolution).to(device)

    needs_depth = args.use_gasa or args.use_world_pe or (args.pe_type not in ('none', None)) or args.use_centroid_head or args.use_cached_depth
    if needs_depth:
        da3_model = DepthAnything3.from_pretrained(args.da3_model).to(device)
    else:
        ddp.print("  Skipping DA3 loading (no GASA, no PE, no centroid — depth not needed)")
        da3_model = None

    ddp.barrier()
    ddp.print("All ranks synchronized after model loading")
    print(f"[R{ddp.rank}] Creating TrianguLangModel...", flush=True)

    if args.torch_compile:
        ddp.print(f"Compiling SAM3 backbone with torch.compile(mode='{args.compile_mode}')...")
        ddp.print("  Note: First epoch will be slower due to JIT compilation")
        sam3_model.backbone = torch.compile(sam3_model.backbone, mode=args.compile_mode, fullgraph=False)
        ddp.print("  SAM3 compilation registered (DA3 not compiled due to control flow)")

    use_box = args.use_box_prompts
    use_point = args.use_point_prompts
    if args.prompt_type == 'text_only':
        use_box, use_point = False, False
    elif args.prompt_type == 'box_only':
        use_point = False
    elif args.prompt_type == 'point_only':
        use_box = False
    elif args.prompt_type == 'text_box':
        use_point = False
    elif args.prompt_type == 'text_point':
        use_box = False

    model = TrianguLangModel(
        sam3_model=sam3_model, da3_model=da3_model, d_model=args.d_model, n_heads=args.n_heads,
        num_decoder_layers=args.num_decoder_layers, num_queries=args.num_queries, train_seghead=args.train_seghead,
        attn_map_size=args.attn_map_size,
        use_presence_token=args.use_presence_token, use_box_prompts=use_box, use_point_prompts=use_point,
        mask_selection=args.mask_selection,
        use_world_pe=args.use_world_pe, use_gasa=args.use_gasa,
        use_centroid_head=args.use_centroid_head,
        box_prompt_dropout=args.box_prompt_dropout, point_prompt_dropout=args.point_prompt_dropout,
        num_pos_points=args.num_pos_points, num_neg_points=args.num_neg_points,
        use_iterative_pos=args.use_iterative_pos,
        cross_view=args.cross_view, pe_type=args.pe_type,
        da3_model_name=args.da3_model.split('/')[-1],
        use_iou_head=args.use_iou_head,
        use_spatial_tokens=args.use_spatial_tokens,
        use_spatial_attn_bias=getattr(args, 'use_spatial_attn_bias', False),
        use_text_spatial_bias=getattr(args, 'use_text_spatial_bias', False),
        use_image_to_token=getattr(args, 'use_image_to_token', False),
        use_pos_refine=getattr(args, 'use_pos_refine', False),
        use_box_rpb=getattr(args, 'use_box_rpb', False),
        use_spatial_points=args.use_spatial_points,
        use_object_aware_spatial=args.use_object_aware_spatial,
        da3_resolution=args.da3_resolution,
        pointmap_normalize=not args.no_pointmap_normalize,
        resolution=args.resolution,
        gasa_beta_init=args.gasa_beta_init,
        use_da3_poses_for_gasa=args.use_da3_poses_for_gasa,
        use_gt_poses_for_gasa=args.use_gt_poses_for_gasa,
        sheaf_use_gt_poses=args.sheaf_use_gt_poses,
        gasa_kernel_dim=args.gasa_kernel_dim,
        gasa_fixed_kernel=args.gasa_fixed_kernel,
        gasa_kernel_type=args.gasa_kernel_type,
        use_depth_crossattn=args.use_depth_crossattn,
        per_layer_text=args.per_layer_text,
        pred_logits_source=args.pred_logits_source,
        gasa_bidirectional=args.gasa_bidirectional,
        query_proj_mlp=args.query_proj_mlp,
        no_query_proj=getattr(args, 'no_query_proj', False),
        train_mask_embed=args.train_mask_embed,
        use_mask_refiner=getattr(args, 'use_mask_refiner', False),
        dim_feedforward=args.dim_feedforward,
        post_norm=args.post_norm,
        use_query_pe=args.use_query_pe,
        ffn_fp32=args.ffn_fp32,
        no_initial_text=args.no_initial_text,
        no_text_proj=args.no_text_proj,
        clean_v=args.clean_v,
        additive_pe=args.additive_pe,
        grouped_text_attn=args.grouped_text_attn,
    ).to(device)
    model.per_text_decode = getattr(args, 'per_text_decode', False)
    model.sam3_multi_object = getattr(args, 'sam3_multi_object', False)
    if args.init_decoder_from_sam3:
        args.init_text_crossattn_from_sam3 = True
        args.init_scoring_from_sam3 = True

    if args.init_scoring_from_sam3 and hasattr(model.sam3, 'dot_prod_scoring'):
        dps = model.sam3.dot_prod_scoring
        if dps.prompt_mlp is not None:
            model.gasa_decoder.scoring_prompt_mlp.load_state_dict(dps.prompt_mlp.state_dict())
            ddp.print("  Initialized scoring_prompt_mlp from SAM3 DotProductScoring")
        model.gasa_decoder.scoring_prompt_proj.load_state_dict(dps.prompt_proj.state_dict())
        model.gasa_decoder.scoring_hs_proj.load_state_dict(dps.hs_proj.state_dict())
        ddp.print("  Initialized scoring_prompt_proj + scoring_hs_proj from SAM3 DotProductScoring")
    elif args.init_scoring_from_sam3:
        ddp.print("  WARNING: --init-scoring-from-sam3 requested but SAM3 has no dot_prod_scoring module")

    if args.init_text_crossattn_from_sam3 and hasattr(model.sam3, 'transformer'):
        sam3_decoder_layers = model.sam3.transformer.decoder.layers
        num_sam3_layers = len(sam3_decoder_layers)
        transferred = 0

        if model.gasa_decoder.per_layer_text:
            for i, gasa_layer in enumerate(model.gasa_decoder.layers):
                if hasattr(gasa_layer, 'text_cross_attn') and i < num_sam3_layers:
                    sam3_ca = sam3_decoder_layers[i].ca_text
                    gasa_layer.text_cross_attn.in_proj_weight.data.copy_(sam3_ca.in_proj_weight.data)
                    gasa_layer.text_cross_attn.in_proj_bias.data.copy_(sam3_ca.in_proj_bias.data)
                    gasa_layer.text_cross_attn.out_proj.weight.data.copy_(sam3_ca.out_proj.weight.data)
                    gasa_layer.text_cross_attn.out_proj.bias.data.copy_(sam3_ca.out_proj.bias.data)
                    if hasattr(gasa_layer, 'text_norm') and hasattr(sam3_decoder_layers[i], 'catext_norm'):
                        gasa_layer.text_norm.weight.data.copy_(sam3_decoder_layers[i].catext_norm.weight.data)
                        gasa_layer.text_norm.bias.data.copy_(sam3_decoder_layers[i].catext_norm.bias.data)
                    transferred += 1

        if hasattr(model.gasa_decoder, 'text_cross_attn') and num_sam3_layers > 0:
            sam3_ca0 = sam3_decoder_layers[0].ca_text
            model.gasa_decoder.text_cross_attn.in_proj_weight.data.copy_(sam3_ca0.in_proj_weight.data)
            model.gasa_decoder.text_cross_attn.in_proj_bias.data.copy_(sam3_ca0.in_proj_bias.data)
            model.gasa_decoder.text_cross_attn.out_proj.weight.data.copy_(sam3_ca0.out_proj.weight.data)
            model.gasa_decoder.text_cross_attn.out_proj.bias.data.copy_(sam3_ca0.out_proj.bias.data)
            if hasattr(model.gasa_decoder, 'text_norm') and hasattr(sam3_decoder_layers[0], 'catext_norm'):
                model.gasa_decoder.text_norm.weight.data.copy_(sam3_decoder_layers[0].catext_norm.weight.data)
                model.gasa_decoder.text_norm.bias.data.copy_(sam3_decoder_layers[0].catext_norm.bias.data)
            transferred += 1

        if hasattr(model.gasa_decoder, 'text_proj'):
            nn.init.eye_(model.gasa_decoder.text_proj.weight)
            nn.init.zeros_(model.gasa_decoder.text_proj.bias)
            ddp.print("  Initialized text_proj as identity (SAM3 uses no text projection)")

        ddp.print(f"  Initialized {transferred} text cross-attention modules from SAM3 decoder")
    elif args.init_text_crossattn_from_sam3:
        ddp.print("  WARNING: --init-text-crossattn-from-sam3 requested but SAM3 has no transformer module")

    if args.init_decoder_from_sam3 and hasattr(model.sam3, 'transformer'):
        sam3_decoder = model.sam3.transformer.decoder
        sam3_layers = sam3_decoder.layers
        num_sam3_layers = len(sam3_layers)
        num_gasa_layers = len(model.gasa_decoder.layers)
        transferred = []

        def _copy_mha(dst, src, name):
            dst.in_proj_weight.data.copy_(src.in_proj_weight.data)
            dst.in_proj_bias.data.copy_(src.in_proj_bias.data)
            dst.out_proj.weight.data.copy_(src.out_proj.weight.data)
            dst.out_proj.bias.data.copy_(src.out_proj.bias.data)
            transferred.append(name)

        def _copy_ln(dst, src, name):
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)
            transferred.append(name)

        for i in range(min(num_gasa_layers, num_sam3_layers)):
            gasa_l = model.gasa_decoder.layers[i]
            sam3_l = sam3_layers[i]
            _copy_mha(gasa_l.self_attn, sam3_l.self_attn, f"layers[{i}].self_attn")
            _copy_ln(gasa_l.norm1, sam3_l.norm2, f"layers[{i}].norm1←norm2 (self-attn)")
            _copy_ln(gasa_l.norm2, sam3_l.norm1, f"layers[{i}].norm2←norm1 (cross-attn)")
            _copy_ln(gasa_l.norm3, sam3_l.norm3, f"layers[{i}].norm3 (FFN)")
            gasa_l.ffn[0].weight.data.copy_(sam3_l.linear1.weight.data)
            gasa_l.ffn[0].bias.data.copy_(sam3_l.linear1.bias.data)
            gasa_l.ffn[3].weight.data.copy_(sam3_l.linear2.weight.data)
            gasa_l.ffn[3].bias.data.copy_(sam3_l.linear2.bias.data)
            transferred.append(f"layers[{i}].ffn (linear1→ffn[0], linear2→ffn[3])")

        if hasattr(sam3_decoder, 'norm') and hasattr(model.gasa_decoder, 'norm'):
            _copy_ln(model.gasa_decoder.norm, sam3_decoder.norm, "output norm")

        if (hasattr(model.gasa_decoder, 'presence_token') and
                hasattr(sam3_decoder, 'presence_token') and
                sam3_decoder.presence_token is not None):
            model.gasa_decoder.presence_token.weight.data.copy_(sam3_decoder.presence_token.weight.data)
            transferred.append("presence_token")
            if hasattr(model.gasa_decoder, 'presence_norm') and hasattr(sam3_decoder, 'presence_token_out_norm'):
                _copy_ln(model.gasa_decoder.presence_norm, sam3_decoder.presence_token_out_norm, "presence_norm")
            if hasattr(model.gasa_decoder, 'presence_head') and hasattr(sam3_decoder, 'presence_token_head'):
                sam3_ph = sam3_decoder.presence_token_head
                our_ph = model.gasa_decoder.presence_head
                for j, sam3_linear in enumerate(sam3_ph.layers):
                    our_linear = our_ph[j * 2]
                    our_linear.weight.data.copy_(sam3_linear.weight.data)
                    our_linear.bias.data.copy_(sam3_linear.bias.data)
                transferred.append("presence_head (3 layers)")

        if hasattr(sam3_decoder, 'query_embed') and hasattr(model.gasa_decoder, 'query_embed'):
            sam3_nq = sam3_decoder.query_embed.weight.shape[0]
            our_nq = model.gasa_decoder.query_embed.weight.shape[0]
            n_transfer = min(our_nq, sam3_nq)
            model.gasa_decoder.query_embed.weight.data[:n_transfer].copy_(
                sam3_decoder.query_embed.weight.data[:n_transfer]
            )
            transferred.append(f"query_embed ({n_transfer}/{our_nq} queries from SAM3's {sam3_nq})")

        ddp.print(f"  --init-decoder-from-sam3: Transferred {len(transferred)} module groups:")
        for t in transferred:
            ddp.print(f"    {t}")
    elif args.init_decoder_from_sam3:
        ddp.print("  WARNING: --init-decoder-from-sam3 requested but SAM3 has no transformer module")

    print(f"[R{ddp.rank}] TrianguLangModel created and moved to device", flush=True)

    if args.profile:
        ddp.print("\n  Profiling enabled - will print timing summary after first epoch")

    use_find_unused = False
    print(f"[R{ddp.rank}] Wrapping with DDP (find_unused_parameters={use_find_unused})...", flush=True)
    model = ddp.wrap_model(model, find_unused_parameters=use_find_unused)
    print(f"[R{ddp.rank}] DDP wrap complete", flush=True)
    base_model = ddp.get_model(model)

    if args.profile:
        base_model.set_profile(True)

    return model, base_model, use_box, use_point


def _setup_training(model, base_model, args, device, ddp):
    spatial_augmentor = None
    gt_aware_spatial = None
    if args.spatial_augment_prob > 0:
        if args.spatial_gt_aware:
            gt_aware_spatial = GTAwareSpatialAugmentor(
                augment_prob=args.spatial_augment_prob,
                relational_prob=args.spatial_relational_prob,
                multi_instance_only=args.spatial_multi_instance_only,
                qualifier_diversity=True
            )
            ddp.print(f"GT-aware spatial augmentation enabled (prob={args.spatial_augment_prob}, "
                      f"relational={args.spatial_relational_prob})")
            if not args.use_cached_depth:
                ddp.print(f"  WARNING: --spatial-gt-aware requires --use-cached-depth for accurate depth!")
        else:
            spatial_augmentor = SpatialAugmentor(augment_prob=args.spatial_augment_prob)
            ddp.print(f"Spatial augmentation enabled (prob={args.spatial_augment_prob}) [RANDOM - labels may be wrong!]")

    if args.use_spatial_tokens or args.use_spatial_points or args.use_object_aware_spatial or args.spatial_augment_prob > 0:
        ddp.print(f"\n  Spatial Reasoning:")
        ddp.print(f"    Spatial tokens: {args.use_spatial_tokens}")
        ddp.print(f"    Spatial-as-points: {args.use_spatial_points}")
        ddp.print(f"    Object-aware spatial: {args.use_object_aware_spatial}" + (" (uses mask+depth for 'nearest chair')" if args.use_object_aware_spatial else ""))
        ddp.print(f"    Spatial augmentation: {args.spatial_augment_prob > 0} (prob={args.spatial_augment_prob})")
        ddp.print(f"    GT-aware augmentation: {args.spatial_gt_aware}" + (" (uses qualifiers from GT masks)" if args.spatial_gt_aware else ""))
        if args.spatial_gt_aware and args.spatial_relational_prob > 0:
            ddp.print(f"    Relational queries: {args.spatial_relational_prob} (e.g., 'chair next to table')")
        if args.spatial_ranking_weight > 0:
            ddp.print(f"    Spatial ranking loss: weight={args.spatial_ranking_weight}, margin={args.spatial_ranking_margin}")
    if args.mask_smooth_kernel > 0:
        ddp.print(f"  Mask smoothing: {args.mask_smooth_kernel}x{args.mask_smooth_kernel} avg_pool (matches eval-time LangSplat)")

    sheaf_loss_fn = None
    feature_sheaf_loss_fn = None
    if args.use_sheaf_loss and args.sheaf_weight > 0:
        if args.sheaf_type == 'feature':
            feature_sheaf_loss_fn = FeatureSheafLoss(
                d_stalk=256,
                d_edge=args.sheaf_d_edge,
                context_dim=5,
            ).to(device)
            n_sheaf_params = sum(p.numel() for p in feature_sheaf_loss_fn.parameters())
            ddp.print(f"Feature sheaf loss initialized (non-constant, learned restriction maps)")
            ddp.print(f"  Stalks: R^256 (SAM3 features), Edge space: R^{args.sheaf_d_edge}")
            ddp.print(f"  Trainable params: {n_sheaf_params:,}")
            ddp.print(f"  Threshold: {args.sheaf_threshold}m, max_frame_distance: {args.sheaf_max_frame_distance}")
        else:
            sheaf_loss_fn = SheafConsistencyLoss(
                threshold=args.sheaf_threshold,
                use_soft_correspondences=args.sheaf_soft_correspondences,
                sigma=args.sheaf_sigma,
                detach_target=args.sheaf_detach_target,
                max_frame_distance=args.sheaf_max_frame_distance,
                symmetric_detach=args.sheaf_symmetric_detach,
                mutual_nn=args.sheaf_mutual_nn,
            )
            if args.sheaf_soft_correspondences:
                soft_str = f", Gaussian sigma={args.sheaf_sigma}m (cutoff at 3σ={3*args.sheaf_sigma:.2f}m)"
            else:
                soft_str = f", hard NN (threshold={args.sheaf_threshold}m)"
            detach_str = ", detach_target=True" if args.sheaf_detach_target else ", detach_target=False (CAUTION: may cause fighting)"
            sym_str = " (symmetric)" if args.sheaf_symmetric_detach and args.sheaf_detach_target else ""
            mutual_str = ", mutual_nn=True" if args.sheaf_mutual_nn else ""
            ddp.print(f"Sheaf consistency loss initialized (constant sheaf){soft_str}{detach_str}{sym_str}{mutual_str}")

    lora_manager = None
    if args.use_lora or args.lora_mask_embed:
        if not args.use_lora:
            args.use_lora = True
        lora_manager = LoRAManager(rank=args.lora_rank, alpha=args.lora_alpha)
        ddp.print(f"\nLoRA enabled (rank={args.lora_rank}, alpha={args.lora_alpha}):")
        if args.lora_sam3:
            sam3_count = lora_manager.add_lora_to_model(base_model.sam3, "sam3")
            ddp.print(f"  SAM3: {sam3_count} adapters")
        if args.lora_da3:
            da3_count = lora_manager.add_lora_to_model(base_model.da3, "da3")
            ddp.print(f"  DA3: {da3_count} adapters")
        if args.lora_mask_embed:
            mask_pred = base_model.sam3.segmentation_head.mask_predictor
            me_count = 0
            for i, layer in enumerate(mask_pred.mask_embed.layers):
                if isinstance(layer, nn.Linear):
                    adapter = LoRALayer(layer.in_features, layer.out_features,
                                        rank=args.lora_rank, alpha=args.lora_alpha)
                    adapter_name = f"mask_embed_layer{i}"
                    lora_manager.adapters[adapter_name] = adapter
                    hook = lora_manager._create_hook(adapter)
                    handle = layer.register_forward_hook(hook)
                    lora_manager.hooks.append(handle)
                    me_count += 1
            lora_manager._adapter_count += me_count
            ddp.print(f"  mask_embed: {me_count} adapters")
        lora_manager.to(device)
        ddp.print(f"  Total LoRA params: {lora_manager.num_parameters:,}")

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    gasa_params = sum(p.numel() for p in base_model.gasa_decoder.parameters())
    lora_params = lora_manager.num_parameters if lora_manager else 0
    ddp.print(f"\nParameters: Total={total_params:,}, Trainable={trainable_params + lora_params:,} ({100*(trainable_params + lora_params)/total_params:.2f}%), GASA={gasa_params:,}" + (f", LoRA={lora_params:,}" if lora_params > 0 else ""))

    trainable_model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    all_params = trainable_model_params
    if lora_manager:
        all_params = all_params + list(lora_manager.parameters())
    if feature_sheaf_loss_fn is not None:
        feature_sheaf_params = list(feature_sheaf_loss_fn.parameters())
        all_params = all_params + feature_sheaf_params
        ddp.print(f"Feature sheaf params: {sum(p.numel() for p in feature_sheaf_params):,}")
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)
    ddp.print(f"Learning rate: {args.lr} (no scaling, world_size={ddp.world_size})")
    scaler = GradScaler()

    scheduler = None
    if args.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_epochs = min(args.lr_warmup_epochs, args.epochs - 1)
        cosine_epochs = max(1, args.epochs - warmup_epochs)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.lr_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        ddp.print(f"LR Scheduler: Cosine annealing with {warmup_epochs} warmup epochs, min_lr={args.lr_min}")
    elif args.lr_scheduler == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        ddp.print(f"LR Scheduler: Step decay every {args.lr_step_size} epochs, gamma={args.lr_gamma}")
    else:
        ddp.print("LR Scheduler: None (flat LR)")

    run_dir = PROJECT_ROOT / 'runs' / 'train' / args.run_name
    checkpoint_dir = Path(args.checkpoint_dir) / args.run_name
    best_iou = 0.0
    best_val_miou = 0.0
    start_epoch = 0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            if (resume_path / 'last.pt').exists():
                resume_path = resume_path / 'last.pt'
            else:
                resume_path = resume_path / 'best.pt'
        if resume_path.exists():
            ddp.print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            base_model.gasa_decoder.load_state_dict_compat(checkpoint['gasa_decoder'], strict=False)
            base_model.query_proj.load_state_dict(checkpoint['query_proj'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_iou = checkpoint.get('best_iou', 0.0)
            best_val_miou = checkpoint.get('best_val_miou', 0.0)

            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    ddp.print(f"  Restored optimizer state")
                except (ValueError, RuntimeError) as e:
                    ddp.print(f"  WARNING: Could not restore optimizer state ({e}). "
                              f"Re-initializing optimizer (new params added since checkpoint).")

            if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
                ddp.print(f"  Restored scheduler state")

            if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                scaler.load_state_dict(checkpoint['scaler'])
                ddp.print(f"  Restored scaler state (scale={scaler.get_scale():.0f})")

            if lora_manager is not None and 'lora' in checkpoint and checkpoint['lora'] is not None:
                lora_manager.load_state_dict(checkpoint['lora'])
                ddp.print(f"  Restored LoRA state ({lora_manager.num_adapters} adapters)")

            if 'sam3_seghead' in checkpoint and checkpoint['sam3_seghead'] is not None:
                base_model.sam3.segmentation_head.load_state_dict(checkpoint['sam3_seghead'])
                ddp.print(f"  Restored SAM3 seghead state")

            if 'mask_embed' in checkpoint and checkpoint['mask_embed'] is not None:
                base_model.sam3.segmentation_head.mask_predictor.mask_embed.load_state_dict(checkpoint['mask_embed'])
                ddp.print(f"  Restored mask_embed state")

            try:
                if 'rng_state' in checkpoint:
                    random.setstate(checkpoint['rng_state'])
                if 'np_rng_state' in checkpoint:
                    np.random.set_state(checkpoint['np_rng_state'])
                if 'torch_rng_state' in checkpoint:
                    rng_state = checkpoint['torch_rng_state']
                    if not isinstance(rng_state, torch.ByteTensor):
                        rng_state = torch.ByteTensor(rng_state)
                    torch.set_rng_state(rng_state)
                if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None and torch.cuda.is_available():
                    cuda_states = [torch.ByteTensor(s) if not isinstance(s, torch.ByteTensor) else s for s in checkpoint['cuda_rng_state']]
                    torch.cuda.set_rng_state_all(cuda_states)
                ddp.print("  Restored RNG states")
            except Exception as e:
                ddp.print(f"  Skipping RNG restore (non-critical): {e}")

            val_str = f", best_val_miou={100*best_val_miou:.2f}%" if best_val_miou > 0 else ""
            ddp.print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', -1) + 1}, best_iou={100*best_iou:.2f}%{val_str}")
            ddp.print(f"  Resuming training from epoch {start_epoch + 1}")
            del checkpoint
        else:
            ddp.print(f"WARNING: Checkpoint not found at {resume_path}, starting fresh")

    elif args.load_weights:
        weights_path = Path(args.load_weights)
        if weights_path.is_dir():
            weights_path = weights_path / 'best.pt'
        if weights_path.exists():
            ddp.print(f"Loading weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            missing, unexpected = base_model.gasa_decoder.load_state_dict_compat(checkpoint['gasa_decoder'], strict=False)
            if missing:
                ddp.print(f"  Missing keys (will be randomly initialized): {len(missing)} keys")
            if unexpected:
                ddp.print(f"  Unexpected keys (ignored): {len(unexpected)} keys")
            base_model.query_proj.load_state_dict(checkpoint['query_proj'], strict=False)
            if 'sam3_seghead' in checkpoint and checkpoint['sam3_seghead'] is not None:
                base_model.sam3.segmentation_head.load_state_dict(checkpoint['sam3_seghead'])
                ddp.print(f"  Loaded SAM3 seghead weights")
            if 'mask_embed' in checkpoint and checkpoint['mask_embed'] is not None:
                base_model.sam3.segmentation_head.mask_predictor.mask_embed.load_state_dict(checkpoint['mask_embed'])
                ddp.print(f"  Loaded mask_embed weights")
            ddp.print(f"  Loaded model weights (epoch {checkpoint.get('epoch', '?')}, iou={100*checkpoint.get('best_iou', 0):.2f}%)")
            ddp.print(f"  Starting fresh from epoch 0 with new optimizer/scheduler")
        else:
            ddp.print(f"WARNING: Weights not found at {weights_path}, starting fresh")

    return optimizer, scaler, scheduler, lora_manager, sheaf_loss_fn, feature_sheaf_loss_fn, \
           spatial_augmentor, gt_aware_spatial, run_dir, checkpoint_dir, best_iou, best_val_miou, start_epoch


def _save_checkpoint(base_model, optimizer, scheduler, scaler, lora_manager, args,
                     epoch, best_iou, best_val_miou, checkpoint_dir, include_rng=False):
    ckpt = {
        'epoch': epoch,
        'gasa_decoder': base_model.gasa_decoder.state_dict(),
        'query_proj': base_model.query_proj.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict(),
        'lora': lora_manager.state_dict() if lora_manager else None,
        'best_iou': best_iou,
        'best_val_miou': best_val_miou,
        'sam3_seghead': base_model.sam3.segmentation_head.state_dict() if args.train_seghead else None,
        'mask_embed': base_model.sam3.segmentation_head.mask_predictor.mask_embed.state_dict() if args.train_mask_embed else None,
        'mask_refiner': base_model.mask_refiner.state_dict() if getattr(base_model, 'use_mask_refiner', False) else None,
    }
    if include_rng:
        ckpt['rng_state'] = random.getstate()
        ckpt['np_rng_state'] = np.random.get_state()
        ckpt['torch_rng_state'] = torch.get_rng_state()
        ckpt['cuda_rng_state'] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return ckpt


def _finalize_epoch(epoch, args, ddp, device, optimizer, cat_metrics, epoch_loss, epoch_iou,
                    epoch_macc, epoch_recall, epoch_sheaf_loss, epoch_centroid_errors,
                    num_samples, start_epoch, base_model, last_vis_data, run_dir):
    import json
    if num_samples > 0:
        avg_loss = epoch_loss / num_samples
        avg_iou = epoch_iou / num_samples
        avg_macc = epoch_macc / num_samples
        avg_recall = epoch_recall / num_samples
        avg_sheaf_loss = epoch_sheaf_loss / num_samples
    else:
        avg_loss = avg_iou = avg_macc = avg_recall = avg_sheaf_loss = 0.0
    current_lr = optimizer.param_groups[0]['lr']

    if ddp.is_distributed:
        torch.cuda.synchronize()
        metrics_tensor = torch.tensor([avg_loss * num_samples, avg_iou * num_samples,
                                       avg_macc * num_samples, avg_recall * num_samples,
                                       avg_sheaf_loss * num_samples,
                                       float(num_samples)], device=device)
        metrics_tensor = ddp.all_reduce(metrics_tensor, op="sum")
        total_samples = metrics_tensor[5].item()
        if total_samples > 0:
            avg_loss = metrics_tensor[0].item() / total_samples
            avg_iou = metrics_tensor[1].item() / total_samples
            avg_macc = metrics_tensor[2].item() / total_samples
            avg_recall = metrics_tensor[3].item() / total_samples
            avg_sheaf_loss = metrics_tensor[4].item() / total_samples
        num_samples = int(total_samples)
        cat_metrics.sync_across_ranks(ddp)

    miou = cat_metrics.get_miou()
    num_cats = len(cat_metrics.get_per_category_iou())

    acc_5cm = acc_10cm = acc_50cm = mean_dist_error = 0.0
    if (args.use_centroid_head or args.eval_localization) and len(epoch_centroid_errors) > 0:
        errors = np.array(epoch_centroid_errors)
        acc_5cm = (errors < 0.05).mean() * 100
        acc_10cm = (errors < 0.10).mean() * 100
        acc_50cm = (errors < 0.50).mean() * 100
        mean_dist_error = errors.mean()

    if num_samples > 0:
        sheaf_str = f", Sheaf={avg_sheaf_loss:.4f}" if args.use_sheaf_loss else ""
        acc_str = f", Acc@5cm={acc_5cm:.1f}%, Acc@10cm={acc_10cm:.1f}%, MDE={mean_dist_error*100:.1f}cm" if (args.use_centroid_head or args.eval_localization) and len(epoch_centroid_errors) > 0 else ""
        ddp.print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}{sheaf_str}, IoU={100*avg_iou:.2f}%, mIoU={100*miou:.2f}% ({num_cats} cats), mAcc={100*avg_macc:.2f}%, Recall={100*avg_recall:.2f}%{acc_str}, LR={current_lr:.2e}")

        if args.profile and epoch == start_epoch and ddp.is_main:
            ddp.print(f"\n{base_model.get_profile_summary()}\n")

        if args.vis_every_epochs > 0 and (epoch + 1) % args.vis_every_epochs == 0 and last_vis_data and ddp.is_main:
            try:
                visualize_predictions(run_dir, epoch + 1, last_vis_data['images'], last_vis_data['gt_masks'],
                                      last_vis_data['outputs'], last_vis_data['prompts'])
            except Exception as e:
                print(f"  Visualization failed: {e}")

    return avg_loss, avg_iou, avg_macc, avg_recall, avg_sheaf_loss, miou, num_cats, num_samples, \
           acc_5cm, acc_10cm, acc_50cm, mean_dist_error


def _run_validation_and_save(model, val_dataloader, base_model, optimizer, scheduler, scaler,
                              lora_manager, args, device, ddp, epoch, best_iou, best_val_miou,
                              checkpoint_dir):
    import json
    val_metrics = None
    if args.val_every > 0 and val_dataloader is not None and (epoch + 1) % args.val_every == 0:
        ddp.print(f"  Running validation...")
        val_metrics = run_validation(model, val_dataloader, device, ddp, args, scaler)
        val_str = f"  Val: Loss={val_metrics['val_loss']:.4f}, IoU={100*val_metrics['val_iou']:.2f}%, mIoU={100*val_metrics['val_miou']:.2f}% ({val_metrics['val_num_categories']} cats)"
        ddp.print(val_str)

        if val_metrics['val_miou'] > best_val_miou:
            best_val_miou = val_metrics['val_miou']
            if args.save_best_val and ddp.is_main:
                ckpt = _save_checkpoint(base_model, optimizer, scheduler, scaler, lora_manager,
                                        args, epoch, best_iou, best_val_miou, checkpoint_dir)
                torch.save(ckpt, checkpoint_dir / 'best.pt')
                print(f"  -> New best val mIoU! Saved to {checkpoint_dir / 'best.pt'}")

    return val_metrics, best_val_miou
