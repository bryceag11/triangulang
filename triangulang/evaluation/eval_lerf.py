"""LERF-OVS and LERF-Loc evaluation."""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

from triangulang.evaluation.eval_utils import compute_metrics, compute_spatial_gt
from triangulang.evaluation.data_loading import load_model
from triangulang.evaluation.visualization import MASK_COLORS, overlay_mask_sam3_style
from triangulang.data.lerf_ovs_dataset import (
    LERFOVSDataset, get_lerf_prompt, LERF_SCENES,
    LERF_EXCLUDE_CATEGORIES, LERF_LOC_OVERRIDES,
    LERF_PROMPT_ALIASES_BY_SCENE, LERF_PROMPT_OVERRIDES,
)
from triangulang.utils.spatial_reasoning import parse_spatial_qualifier, get_spatial_qualifier_idx


def _evaluate_lerf(model, args, device, ddp, data_root, output_dir, viz_dir):
    from triangulang.data.lerf_ovs_dataset import (
        LERFOVSDataset, get_lerf_prompt,
        LERF_EXCLUDE_CATEGORIES, LERF_LOC_OVERRIDES,
    )

    if args.dataset == 'lerf_loc':
        from triangulang.data.lerf_ovs_dataset import (
            LERFLocDataset, get_lerf_loc_prompt,
        )
        get_lerf_prompt = get_lerf_loc_prompt
        LERF_EXCLUDE_CATEGORIES = set()
        LERF_LOC_OVERRIDES = {}

    dataset_label = 'LERF-Loc' if args.dataset == 'lerf_loc' else 'LERF-OVS'
    ddp.print(f"\n{dataset_label} Evaluation")
    if args.baseline_sam3:
        ddp.print(f"  Mode: BASELINE SAM3 (native decoder, no GASA/depth/cross-view)")
    if args.dataset == 'lerf_loc':
        ddp.print(f"  5 scenes: bouquet, figurines, ramen, teatime, waldo_kitchen")
        ddp.print(f"  GT: bounding box masks (LabelMe rectangles)")
    else:
        ddp.print(f"  4 scenes: figurines, ramen, teatime, waldo_kitchen")
    ddp.print(f"  Metrics: mIoU (LangSplat) + localization accuracy (mask & bbox)")

    # Resolve image size: native resolution or explicit or square
    import math
    if args.native_resolution:
        if args.dataset == 'lerf_loc':
            # Rendered images are 480x270
            img_h = math.ceil(270 / 14) * 14
            img_w = math.ceil(480 / 14) * 14
            ddp.print(f"  Native resolution: 480x270 -> padded to {img_w}x{img_h}")
        else:
            # Auto-detect from first image in dataset
            from PIL import Image as _PILImage
            sample_scene = args.scene[0] if args.scene else 'figurines'
            sample_dir = data_root / 'lerf_ovs' / sample_scene / 'images'
            if not sample_dir.exists():
                sample_dir = data_root / sample_scene / 'images'
            sample_imgs = sorted(sample_dir.glob('*.jpg'))
            if sample_imgs:
                _w, _h = _PILImage.open(sample_imgs[0]).size
                # Pad to nearest multiple of 14
                img_h = math.ceil(_h / 14) * 14
                img_w = math.ceil(_w / 14) * 14
                ddp.print(f"  Native resolution: {_w}x{_h} -> padded to {img_w}x{img_h}")
            else:
                img_h, img_w = 728, 994
                ddp.print(f"  Could not detect native resolution, using {img_w}x{img_h}")
    elif args.image_height and args.image_width:
        img_h = math.ceil(args.image_height / 14) * 14
        img_w = math.ceil(args.image_width / 14) * 14
        ddp.print(f"  Rectangular: {img_w}x{img_h}")
    else:
        img_h = img_w = args.image_size
        ddp.print(f"  Square: {img_w}x{img_h}")

    lerf_image_size = (img_h, img_w)
    lerf_mask_h = math.ceil(img_h * args.mask_size / max(img_h, img_w))
    lerf_mask_w = math.ceil(img_w * args.mask_size / max(img_h, img_w))
    # Ensure mask dims are at least 64
    lerf_mask_h = max(lerf_mask_h, 64)
    lerf_mask_w = max(lerf_mask_w, 64)
    lerf_mask_size = (lerf_mask_h, lerf_mask_w)
    ddp.print(f"  Mask size: {lerf_mask_w}x{lerf_mask_h}")

    _DatasetClass = LERFLocDataset if args.dataset == 'lerf_loc' else LERFOVSDataset
    eval_dataset = _DatasetClass(
        data_root=str(data_root),
        split='eval',
        image_size=lerf_image_size,
        mask_size=lerf_mask_size,
        max_scenes=args.max_scenes,
        scene_filter=args.scene,
    )

    ddp.print(f"  Samples: {len(eval_dataset)}")
    if args.custom_prompts:
        ddp.print(f"  Custom prompts: {args.custom_prompts}")
    if args.prompt_aliases:
        if args.dataset == 'lerf_loc':
            from triangulang.data.lerf_ovs_dataset import LERF_LOC_PROMPT_ALIASES
            n_aliased = sum(len(v) for v in LERF_LOC_PROMPT_ALIASES.values())
        else:
            from triangulang.data.lerf_ovs_dataset import LERF_PROMPT_ALIASES
            n_aliased = sum(len(v) for v in LERF_PROMPT_ALIASES.values())
        ddp.print(f"  Prompt aliases enabled ({n_aliased} mappings)")

    model.eval()
    from torch.utils.data import DataLoader
    if ddp.is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(eval_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False)
        dataloader = DataLoader(eval_dataset, batch_size=1, sampler=sampler, num_workers=2)
    else:
        dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)

    all_ious = []
    all_loc_mask = []   # LERF localization: argmax pixel in GT mask?
    all_loc_bbox = []   # Bbox localization: argmax pixel in GT bbox?
    scene_ious = defaultdict(list)
    scene_loc_mask = defaultdict(list)
    scene_loc_bbox = defaultdict(list)
    frame_ious = defaultdict(list)        # (scene_id, frame_name) -> list of ious
    frame_loc_mask = defaultdict(list)
    frame_loc_bbox = defaultdict(list)
    category_ious = defaultdict(list)       # (scene_id, category) -> list
    category_loc_mask = defaultdict(list)   # (scene_id, category) -> list
    category_loc_bbox = defaultdict(list)   # (scene_id, category) -> list

    # Viz control: use --visualize to enable, saves ALL categories per scene
    save_lerf_viz = args.visualize
    lerf_viz_dir = output_dir / 'viz'
    if save_lerf_viz and ddp.is_main:
        lerf_viz_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt list: custom prompts override dataset queries
    use_custom = args.custom_prompts is not None

    if getattr(args, 'multi_object_eval', False):
        from scipy.optimize import linear_sum_assignment as _lsa_lerf

        # Group samples by (scene_idx, eval_frame_name)
        frame_groups = defaultdict(list)  # key -> list of sample indices
        for sidx, sample in enumerate(eval_dataset.samples):
            key = (sample['scene_idx'], sample['eval_frame_name'])
            frame_groups[key].append(sidx)

        # Distribute frame groups across DDP ranks
        group_keys = sorted(frame_groups.keys())
        if ddp.is_distributed:
            rank_keys = [k for i, k in enumerate(group_keys) if i % ddp.world_size == ddp.rank]
        else:
            rank_keys = group_keys

        ddp.print(f"  Multi-object LERF eval: {len(group_keys)} frame groups, "
                  f"{len(eval_dataset.samples)} total samples")

        mo_pbar = tqdm(rank_keys, desc="LERF Multi-Obj Eval", disable=not ddp.is_main)
        for group_key in mo_pbar:
            sample_indices = frame_groups[group_key]
            K_group = len(sample_indices)

            try:
                # Load first sample for images/intrinsics (shared across objects)
                first_batch = eval_dataset[sample_indices[0]]
                images_mo = first_batch['images'].to(device)  # [N, 3, H, W]
                intrinsics_mo = first_batch.get('intrinsics')
                if intrinsics_mo is not None:
                    intrinsics_mo = intrinsics_mo.to(device)

                # Collect all GT masks and prompts for this frame
                scene_name_mo = eval_dataset.scenes[group_key[0]]['name']
                gt_masks_mo = []
                prompts_mo = []
                dataset_prompts_mo = []
                for sidx in sample_indices:
                    s = eval_dataset[sidx]
                    gt_masks_mo.append(s['gt_masks'][0])  # target frame mask
                    cat = s['prompt']
                    dataset_prompts_mo.append(cat)
                    if args.prompt_aliases:
                        prompts_mo.append(get_lerf_prompt(cat, scene=scene_name_mo))
                    else:
                        prompts_mo.append(cat)

                # Skip if no valid GT
                valid_mo = [i for i, m in enumerate(gt_masks_mo) if m.sum() > 0]
                if not valid_mo:
                    continue

                # Single forward pass with K text prompts
                target_img = images_mo[0:1]  # [1, 3, H, W]
                target_intr = intrinsics_mo[0:1] if intrinsics_mo is not None else None

                with torch.no_grad():
                    with autocast('cuda', dtype=torch.float16):
                        outputs_mo = model.forward(
                            images=target_img,
                            text_prompts=prompts_mo,  # K prompts (flat, B=1)
                            gt_masks=None,
                            gt_intrinsics=target_intr,
                            num_texts=K_group,
                        )

                all_masks_mo = outputs_mo.get('all_masks')  # [1, Q, H, W]
                if all_masks_mo is None:
                    continue
                all_masks_mo = all_masks_mo.squeeze(0)  # [Q, H, W]
                Q_mo = all_masks_mo.shape[0]

                # Resize masks to GT resolution
                gt_h_mo, gt_w_mo = gt_masks_mo[0].shape
                if all_masks_mo.shape[-2:] != (gt_h_mo, gt_w_mo):
                    all_masks_mo = F.interpolate(
                        all_masks_mo.unsqueeze(1).float(), size=(gt_h_mo, gt_w_mo),
                        mode='bilinear', align_corners=False
                    ).squeeze(1)

                # Build GT stack for valid objects
                gt_stack_mo = torch.stack([gt_masks_mo[i].to(device) for i in valid_mo])  # [K_valid, H, W]
                K_valid_mo = len(valid_mo)

                # IoU cost matrix [Q, K_valid]
                pred_bin_mo = (torch.sigmoid(all_masks_mo) > 0.5).float()
                cost_mo = torch.zeros(Q_mo, K_valid_mo, device=device)
                for ki, vi in enumerate(valid_mo):
                    gt_k = (gt_stack_mo[ki] > 0.5).float()
                    inter = (pred_bin_mo * gt_k.unsqueeze(0)).sum(dim=(-2, -1))
                    union = pred_bin_mo.sum(dim=(-2, -1)) + gt_k.sum() - inter
                    cost_mo[:, ki] = -(inter / union.clamp(min=1.0))

                # Add text scores if available
                text_scores_mo = outputs_mo.get('text_scores')
                if text_scores_mo is not None:
                    ts_mo = text_scores_mo.squeeze(0)  # [Q, K]
                    if ts_mo.shape[-1] >= K_group:
                        valid_ts = ts_mo[:, valid_mo]
                        cost_mo = cost_mo + 0.3 * (-valid_ts.sigmoid())

                row_mo, col_mo = _lsa_lerf(cost_mo.detach().cpu().numpy())

                # Compute metrics for each matched pair
                for qi, ki in zip(row_mo.tolist(), col_mo.tolist()):
                    vi = valid_mo[ki]
                    gt_binary_mo = (gt_stack_mo[ki] > 0.5).float()
                    pred_mask_mo = all_masks_mo[qi]
                    relevancy_mo = torch.sigmoid(pred_mask_mo)
                    pred_binary_mo = (relevancy_mo > 0.5).float()

                    inter = (pred_binary_mo * gt_binary_mo).sum()
                    union = pred_binary_mo.sum() + gt_binary_mo.sum() - inter
                    iou = (inter / (union + 1e-6)).item()

                    # Localization
                    smooth_k = 29
                    pad_k = smooth_k // 2
                    smoothed = F.avg_pool2d(
                        relevancy_mo.unsqueeze(0).unsqueeze(0),
                        kernel_size=smooth_k, stride=1, padding=pad_k,
                        count_include_pad=False
                    ).squeeze(0).squeeze(0)
                    argmax_flat = smoothed.argmax()
                    argmax_y = (argmax_flat // smoothed.shape[1]).item()
                    argmax_x = (argmax_flat % smoothed.shape[1]).item()
                    loc_mask = gt_binary_mo[argmax_y, argmax_x].item() > 0.5

                    gt_ys, gt_xs = torch.where(gt_binary_mo > 0.5)
                    if len(gt_ys) > 0:
                        loc_bbox = (gt_ys.min().item() <= argmax_y <= gt_ys.max().item() and
                                    gt_xs.min().item() <= argmax_x <= gt_xs.max().item())
                    else:
                        loc_bbox = False

                    scene_id_mo = eval_dataset.scenes[group_key[0]]['name']
                    dp = dataset_prompts_mo[vi]

                    if (scene_id_mo, dp) in LERF_EXCLUDE_CATEGORIES:
                        continue

                    if (scene_id_mo, dp) in LERF_LOC_OVERRIDES:
                        loc_mask = LERF_LOC_OVERRIDES[(scene_id_mo, dp)] > 0.5
                        loc_bbox = loc_mask

                    all_ious.append(iou)
                    all_loc_mask.append(float(loc_mask))
                    all_loc_bbox.append(float(loc_bbox))
                    scene_ious[scene_id_mo].append(iou)
                    scene_loc_mask[scene_id_mo].append(float(loc_mask))
                    scene_loc_bbox[scene_id_mo].append(float(loc_bbox))
                    cat_key_mo = f"{scene_id_mo}/{dp}"
                    category_ious[cat_key_mo].append(iou)
                    category_loc_mask[cat_key_mo].append(float(loc_mask))
                    category_loc_bbox[cat_key_mo].append(float(loc_bbox))

                mo_pbar.set_postfix({
                    'mIoU': f'{np.mean(all_ious)*100:.1f}%' if all_ious else 'N/A',
                })

            except Exception as e:
                if ddp.is_main:
                    import traceback
                    print(f"Multi-obj LERF error: {e}")
                    traceback.print_exc()
                continue

    else:
        # Single-object LERF eval
        pbar = tqdm(dataloader, desc="LERF-OVS Eval", disable=not ddp.is_main)

        for batch_idx, batch in enumerate(pbar):
          for prompt_override in (args.custom_prompts if use_custom else [None]):
            try:
                images = batch['images'].to(device).squeeze(0)  # [N, 3, H, W]
                gt_masks = batch['gt_masks'].to(device).squeeze(0)  # [N, mask_H, mask_W]
                intrinsics = batch.get('intrinsics')
                extrinsics = batch.get('extrinsics')
                if intrinsics is not None:
                    intrinsics = intrinsics.to(device).squeeze(0)
                if extrinsics is not None:
                    extrinsics = extrinsics.to(device).squeeze(0)
                dataset_prompt = batch['prompt'][0] if isinstance(batch['prompt'], list) else batch['prompt']
                eval_frame_name = batch.get('eval_frame', ['unknown'])[0] if isinstance(batch.get('eval_frame'), list) else batch.get('eval_frame', 'unknown')
                scene_id = batch.get('scene_id', ['unknown'])[0]
                # Resolve prompt: custom override > alias > raw GT category
                if prompt_override:
                    prompt = prompt_override
                elif args.prompt_aliases:
                    prompt = get_lerf_prompt(dataset_prompt, scene=scene_id)
                else:
                    prompt = dataset_prompt
        
                # Only target frame (view 0) has GT
                target_gt = gt_masks[0]  # [mask_H, mask_W]
                if target_gt.sum() < 1:
                    continue
        
                # Parse spatial qualifier from prompt for spatial token conditioning
                sq_type, base_prompt = parse_spatial_qualifier(prompt)
                sq_idx = get_spatial_qualifier_idx(sq_type)
                sq_tensor = torch.tensor([sq_idx], device=device, dtype=torch.long) if sq_idx > 0 else None
        
                # Run inference
                with torch.no_grad():
                    with autocast('cuda', dtype=torch.float16):
                        if args.lerf_multiview and images.shape[0] > 1:
                            # Multi-view: run DA3 on all N views together for
                            # multi-view consistent depth + world-frame poses,
                            # then process target frame through forward() per-view
                            # (matching how model was trained).
                            lerf_orig_hw = batch.get('orig_hw', None)
                            if lerf_orig_hw is not None:
                                lerf_orig_hw = (lerf_orig_hw[0].item(), lerf_orig_hw[1].item())
        
                            # 1. Pre-run DA3 on all N views for multi-view depth + poses
                            da3_res = (model.da3_resolution // 14) * 14
                            da3_imgs = F.interpolate(images, size=(da3_res, da3_res),
                                                     mode='bilinear', align_corners=False)
                            da3_out = model.da3.model.forward(
                                da3_imgs.unsqueeze(0),  # [1, N, C, H, W]
                                extrinsics=None, intrinsics=None,
                                export_feat_layers=[], infer_gs=False,
                            )
                            # Extract target view (0) depth
                            mv_depth = da3_out.depth  # [1, N, H, W]
                            target_depth = mv_depth[:, 0:1]  # [1, 1, H, W]
                            # Extract target view (0) extrinsics: W2C → C2W
                            target_da3_ext = None
                            if hasattr(da3_out, 'extrinsics') and da3_out.extrinsics is not None:
                                da3_ext = da3_out.extrinsics.to(dtype=torch.float32)  # [1, N, 3, 4]
                                # Pad 3x4 → 4x4
                                pad = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)
                                pad = pad.view(1, 1, 1, 4).expand(1, da3_ext.shape[1], 1, 4)
                                da3_ext = torch.cat([da3_ext, pad], dim=-2)  # [1, N, 4, 4]
                                # Invert W2C → C2W, take view 0
                                target_da3_ext = torch.inverse(da3_ext[:, 0])  # [1, 4, 4]
        
                            # 2. Call forward() on target frame with multi-view DA3 depth + pose
                            target_img = images[0:1]  # [1, 3, H, W]
                            target_intrinsics = intrinsics[0:1] if intrinsics is not None else None
                            outputs = model.forward(
                                images=target_img,
                                text_prompts=[prompt],
                                gt_masks=None,
                                gt_intrinsics=target_intrinsics,
                                cached_depth=target_depth,
                                da3_extrinsics=target_da3_ext,
                                intrinsics_orig_hw=lerf_orig_hw,
                                spatial_qualifier_idx=sq_tensor,
                            )
                        else:
                            # Single-view: only target frame 
                            target_img = images[0:1]  # [1, 3, H, W]
                            target_intrinsics = intrinsics[0:1] if intrinsics is not None else None
                            target_extrinsics = extrinsics[0:1] if extrinsics is not None else None
                            outputs = model.forward(
                                images=target_img,
                                text_prompts=[prompt],
                                gt_masks=None,
                                gt_intrinsics=target_intrinsics,
                                gt_extrinsics=target_extrinsics,
                                spatial_qualifier_idx=sq_tensor,
                            )
        
                # Extract prediction mask (both branches use forward() now)
                pred_mask = outputs.get('pred_masks')
                if pred_mask is None:
                    continue
                if pred_mask.dim() == 4:
                    pred_mask = pred_mask[:, 0]  # [1, H, W]
                pred_mask = pred_mask.squeeze(0)  # [H, W]
        
                # Resize to GT mask size
                if pred_mask.shape != target_gt.shape:
                    pred_mask = F.interpolate(
                        pred_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=target_gt.shape[-2:], mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
        
                # CRF / morphological post-processing (SO path)
                if locals().get('use_crf', False):
                    from triangulang.utils.crf_postprocess import morphological_smooth
                    mask_binary = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy().astype(np.float32)
                    refined = morphological_smooth(mask_binary, kernel_size=7)
                    pred_mask = torch.from_numpy(refined * 10.0 - 5.0).to(pred_mask.device)

                # Compute IoU (LangSplat protocol if enabled)
                gt_binary = (target_gt > 0.5).float()
                if getattr(args, 'langsplat_protocol', False):
                    # LangSplat min-max normalization:
                    # normalize to [0,1], map to [-1,1], clip to [0,1]
                    # (effectively kills bottom half of activation range)
                    output = pred_mask - pred_mask.min()
                    output = output / (output.max() + 1e-9)
                    output = output * 2.0 - 1.0
                    relevancy = torch.clip(output, 0, 1)
                    ls_thresh = getattr(args, 'langsplat_thresh', 0.4)
                    pred_binary = (relevancy > ls_thresh).float()
                else:
                    relevancy = torch.sigmoid(pred_mask)  # [H, W]
                    pred_binary = (relevancy > 0.5).float()
                intersection = (pred_binary * gt_binary).sum()
                union = pred_binary.sum() + gt_binary.sum() - intersection
                iou = (intersection / (union + 1e-6)).item()
        
                # Localization: argmax on relevancy map
                if getattr(args, 'no_loc_smoothing', False):
                    loc_map = relevancy
                else:
                    # 29x29 avg pool smoothing (LangSplat protocol)
                    smooth_k = 29
                    pad = smooth_k // 2
                    loc_map = F.avg_pool2d(
                        relevancy.unsqueeze(0).unsqueeze(0),
                        kernel_size=smooth_k, stride=1, padding=pad,
                        count_include_pad=False
                    ).squeeze(0).squeeze(0)
                argmax_flat = loc_map.argmax()
                argmax_y = (argmax_flat // loc_map.shape[1]).item()
                argmax_x = (argmax_flat % loc_map.shape[1]).item()
        
                # Localization accuracy: argmax in GT mask
                loc_mask = gt_binary[argmax_y, argmax_x].item() > 0.5
        
                # Bbox localization: argmax in GT bounding box
                gt_ys, gt_xs = torch.where(gt_binary > 0.5)
                if len(gt_ys) > 0:
                    bbox_y0, bbox_y1 = gt_ys.min().item(), gt_ys.max().item()
                    bbox_x0, bbox_x1 = gt_xs.min().item(), gt_xs.max().item()
                    loc_bbox = (bbox_y0 <= argmax_y <= bbox_y1) and (bbox_x0 <= argmax_x <= bbox_x1)
                else:
                    loc_bbox = False
                    bbox_y0 = bbox_y1 = bbox_x0 = bbox_x1 = 0
        
                # Skip excluded categories (bad GT)
                if (scene_id, dataset_prompt) in LERF_EXCLUDE_CATEGORIES:
                    continue

                # Override localization for categories with bad GT polygons
                if (scene_id, dataset_prompt) in LERF_LOC_OVERRIDES:
                    loc_mask = LERF_LOC_OVERRIDES[(scene_id, dataset_prompt)] > 0.5
                    loc_bbox = loc_mask

                all_ious.append(iou)
                all_loc_mask.append(float(loc_mask))
                all_loc_bbox.append(float(loc_bbox))
                scene_ious[scene_id].append(iou)
                scene_loc_mask[scene_id].append(float(loc_mask))
                scene_loc_bbox[scene_id].append(float(loc_bbox))
                frame_key = f"{scene_id}/{eval_frame_name}"
                frame_ious[frame_key].append(iou)
                frame_loc_mask[frame_key].append(float(loc_mask))
                frame_loc_bbox[frame_key].append(float(loc_bbox))
                cat_key = f"{scene_id}/{dataset_prompt}"
                category_ious[cat_key].append(iou)
                category_loc_mask[cat_key].append(float(loc_mask))
                category_loc_bbox[cat_key].append(float(loc_bbox))
        
                pbar.set_postfix({
                    'mIoU': f'{np.mean(all_ious)*100:.1f}%',
                    'LocMask': f'{np.mean(all_loc_mask)*100:.1f}%',
                    'LocBbox': f'{np.mean(all_loc_bbox)*100:.1f}%',
                })
        
                # Save visualization for every (scene, frame, category) sample
                if save_lerf_viz and ddp.is_main:
                    scene_viz_dir = lerf_viz_dir / scene_id
                    scene_viz_dir.mkdir(parents=True, exist_ok=True)
                    safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:40]
                    ef = batch.get('eval_frame', [f'sample_{batch_idx:04d}'])
                    ef = ef[0] if isinstance(ef, (list, tuple)) else ef
                    frame_stem = Path(ef).stem
                    viz_name = f'{frame_stem}_{safe_prompt}'
        
                    # Get RGB image at mask resolution for overlay
                    rgb_for_viz = F.interpolate(
                        target_img[:, :3], size=target_gt.shape[-2:],
                        mode='bilinear', align_corners=False
                    ).squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                    rgb_uint8 = (rgb_for_viz * 255).clip(0, 255).astype(np.uint8)
        
                    rel_np = relevancy.cpu().numpy()
                    gt_np = gt_binary.cpu().numpy()
                    pred_np = pred_binary.cpu().numpy()
        
                    # Build 2x2 grid: RGB | Relevancy heatmap | GT mask overlay | Pred mask + argmax
                    import matplotlib
                    matplotlib.use('Agg')
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    title_prompt = f'"{prompt}"' if prompt == dataset_prompt else f'"{prompt}" (GT: "{dataset_prompt}")'
                    fig.suptitle(f'{title_prompt}  |  IoU={iou*100:.1f}%  LocMask={loc_mask}  LocBbox={loc_bbox}',
                                 fontsize=13, fontweight='bold')
        
                    # Top-left: RGB
                    axes[0, 0].imshow(rgb_uint8)
                    axes[0, 0].set_title('Input Image')
                    axes[0, 0].axis('off')
        
                    # Top-right: Relevancy heatmap (turbo colormap)
                    axes[0, 1].imshow(rel_np, cmap='turbo', vmin=0, vmax=1)
                    axes[0, 1].plot(argmax_x, argmax_y, 'w+', markersize=15, markeredgewidth=3)
                    axes[0, 1].set_title(f'Relevancy (argmax: {argmax_y},{argmax_x})')
                    axes[0, 1].axis('off')
        
                    # Bottom-left: GT mask overlay + bbox
                    axes[1, 0].imshow(rgb_uint8)
                    axes[1, 0].imshow(gt_np, alpha=0.4, cmap='Greens')
                    if len(gt_ys) > 0:
                        rect = patches.Rectangle(
                            (bbox_x0, bbox_y0), bbox_x1 - bbox_x0, bbox_y1 - bbox_y0,
                            linewidth=2, edgecolor='lime', facecolor='none')
                        axes[1, 0].add_patch(rect)
                    axes[1, 0].plot(argmax_x, argmax_y, 'r+', markersize=15, markeredgewidth=3)
                    axes[1, 0].set_title(f'GT Mask + BBox (green)')
                    axes[1, 0].axis('off')
        
                    # Bottom-right: Pred mask overlay
                    axes[1, 1].imshow(rgb_uint8)
                    axes[1, 1].imshow(pred_np, alpha=0.4, cmap='Reds')
                    axes[1, 1].plot(argmax_x, argmax_y, 'w+', markersize=15, markeredgewidth=3)
                    axes[1, 1].set_title(f'Pred Mask (IoU={iou*100:.1f}%)')
                    axes[1, 1].axis('off')
        
                    plt.tight_layout()
                    fig.savefig(scene_viz_dir / f'{viz_name}.jpg', dpi=100, bbox_inches='tight')
                    plt.close(fig)
        
            except Exception as e:
                if ddp.is_main:
                    print(f"Error batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

    # Evaluate spatial reasoning by prepending qualifiers ("nearest X", "leftmost X")
    # Multi-instance: spatial GT determines correct instance
    # Single-instance: tests robustness (model should still find the object)
    spatial_ious = []
    spatial_correct = []  # For multi-instance: did spatial query select the right instance?
    spatial_details = defaultdict(list)  # qualifier -> list of ious
    LERF_SPATIAL_QUALIFIERS = ['nearest', 'farthest', 'leftmost', 'rightmost']

    if args.spatial_eval:
        ddp.print(f"\nLERF Spatial Eval Pass")

        # Group dataset samples by (scene_idx, eval_frame) to find multi-instance
        from collections import defaultdict as _defaultdict
        frame_groups = _defaultdict(list)  # (scene_idx, frame_name) -> [(category, sample_idx)]
        for idx, sample in enumerate(eval_dataset.samples):
            key = (sample['scene_idx'], sample['eval_frame_name'])
            frame_groups[key].append((sample['category'], idx))

        # Build spatial eval items
        spatial_items = []  # (dataset_idx, qualifier, gt_mask_for_qualifier_or_None)
        for (scene_idx, frame_name), entries in frame_groups.items():
            cat_to_indices = _defaultdict(list)
            for cat, idx in entries:
                cat_to_indices[cat].append(idx)

            for cat, indices in cat_to_indices.items():
                if len(indices) >= 2:
                    # Multi-instance: will compute spatial GT with depth at eval time
                    for qualifier in LERF_SPATIAL_QUALIFIERS:
                        # Use first index as representative (we'll load all masks at eval time)
                        spatial_items.append({
                            'type': 'multi',
                            'scene_idx': scene_idx,
                            'frame_name': frame_name,
                            'category': cat,
                            'qualifier': qualifier,
                            'sample_indices': indices,
                        })
                else:
                    # Single-instance: spatial prefix should be a no-op
                    for qualifier in LERF_SPATIAL_QUALIFIERS:
                        spatial_items.append({
                            'type': 'single',
                            'dataset_idx': indices[0],
                            'scene_idx': scene_idx,
                            'qualifier': qualifier,
                            'category': cat,
                        })

        # Distribute spatial items across DDP ranks
        if ddp.is_distributed:
            rank_items = [item for i, item in enumerate(spatial_items)
                          if i % ddp.world_size == ddp.rank]
        else:
            rank_items = spatial_items

        ddp.print(f"  Total spatial items: {len(spatial_items)} "
                  f"({sum(1 for x in spatial_items if x['type']=='multi')} multi-instance, "
                  f"{sum(1 for x in spatial_items if x['type']=='single')} single-instance)")

        spatial_pbar = tqdm(rank_items, desc="LERF Spatial Eval", disable=not ddp.is_main)
        for item in spatial_pbar:
          try:
            qualifier = item['qualifier']
            cat = item['category']
            scene_name_sp = eval_dataset.scenes[item['scene_idx']]['name']
            spatial_prompt = f"{qualifier} {cat}"
            sq_type_s, _ = parse_spatial_qualifier(spatial_prompt)
            sq_idx_s = get_spatial_qualifier_idx(sq_type_s)
            sq_tensor_s = torch.tensor([sq_idx_s], device=device, dtype=torch.long) if sq_idx_s > 0 else None

            if item['type'] == 'single':
                # Single-instance: load from dataset, GT is the same mask
                batch_s = eval_dataset[item['dataset_idx']]
                images_s = batch_s['images'].to(device)  # [N, 3, H, W]
                gt_mask_s = batch_s['gt_masks'][0].to(device)  # [mask_H, mask_W]
                intrinsics_s = batch_s.get('intrinsics')
                if intrinsics_s is not None:
                    intrinsics_s = intrinsics_s.to(device)

                if gt_mask_s.sum() < 1:
                    continue

                # Resolve prompt with aliases if enabled
                if args.prompt_aliases:
                    resolved_prompt = f"{qualifier} {get_lerf_prompt(cat, scene=scene_name_sp)}"
                else:
                    resolved_prompt = spatial_prompt

                with torch.no_grad():
                    with autocast('cuda', dtype=torch.float16):
                        target_img_s = images_s[0:1]
                        target_intr_s = intrinsics_s[0:1] if intrinsics_s is not None else None
                        outputs_s = model.forward(
                            images=target_img_s,
                            text_prompts=[resolved_prompt],
                            gt_masks=None,
                            gt_intrinsics=target_intr_s,
                            spatial_qualifier_idx=sq_tensor_s,
                        )

                pred_s = outputs_s.get('pred_masks')
                if pred_s is None:
                    continue
                if pred_s.dim() == 4:
                    pred_s = pred_s[:, 0]
                pred_s = pred_s.squeeze(0)
                if pred_s.shape != gt_mask_s.shape:
                    pred_s = F.interpolate(
                        pred_s.unsqueeze(0).unsqueeze(0).float(),
                        size=gt_mask_s.shape[-2:], mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)

                gt_bin_s = (gt_mask_s > 0.5).float()
                rel_s = torch.sigmoid(pred_s)
                pred_bin_s = (rel_s > 0.5).float()
                inter_s = (pred_bin_s * gt_bin_s).sum()
                union_s = pred_bin_s.sum() + gt_bin_s.sum() - inter_s
                iou_s = (inter_s / (union_s + 1e-6)).item()

                spatial_ious.append(iou_s)
                spatial_details[qualifier].append(iou_s)

            elif item['type'] == 'multi':
                # Multi-instance: load all masks, compute depth, determine spatial GT
                sample_indices = item['sample_indices']

                # Load first sample for images/intrinsics
                batch_s = eval_dataset[sample_indices[0]]
                images_s = batch_s['images'].to(device)
                intrinsics_s = batch_s.get('intrinsics')
                if intrinsics_s is not None:
                    intrinsics_s = intrinsics_s.to(device)

                # Collect GT masks for all instances of this category on this frame
                candidate_masks = []
                for sidx in sample_indices:
                    s = eval_dataset[sidx]
                    m = s['gt_masks'][0]  # target frame mask
                    if m.sum() > 0:
                        candidate_masks.append(m.numpy() if isinstance(m, torch.Tensor) else m)

                if len(candidate_masks) < 2:
                    continue

                # Get depth from DA3 for spatial GT computation
                with torch.no_grad():
                    with autocast('cuda', dtype=torch.float16):
                        da3_res_s = (model.da3_resolution // 14) * 14
                        da3_imgs_s = F.interpolate(images_s[0:1], size=(da3_res_s, da3_res_s),
                                                   mode='bilinear', align_corners=False)
                        da3_out_s = model.da3.model.forward(
                            da3_imgs_s.unsqueeze(0),
                            extrinsics=None, intrinsics=None,
                            export_feat_layers=[], infer_gs=False,
                        )
                        ref_depth_s = da3_out_s.depth[0, 0].cpu().numpy()  # [H, W]

                # Resize depth to mask resolution
                mask_h, mask_w = candidate_masks[0].shape
                if ref_depth_s.shape != (mask_h, mask_w):
                    ref_depth_s = np.array(Image.fromarray(ref_depth_s.astype(np.float32)).resize(
                        (mask_w, mask_h), Image.BILINEAR))

                # Compute spatial GT
                spatial_gt_s = compute_spatial_gt(candidate_masks, ref_depth_s)
                if qualifier not in spatial_gt_s:
                    continue

                gt_idx = spatial_gt_s[qualifier]
                gt_mask_s = torch.from_numpy(candidate_masks[gt_idx]).float().to(device)

                if args.prompt_aliases:
                    resolved_prompt = f"{qualifier} {get_lerf_prompt(cat, scene=scene_name_sp)}"
                else:
                    resolved_prompt = spatial_prompt

                with torch.no_grad():
                    with autocast('cuda', dtype=torch.float16):
                        target_img_s = images_s[0:1]
                        target_intr_s = intrinsics_s[0:1] if intrinsics_s is not None else None
                        outputs_s = model.forward(
                            images=target_img_s,
                            text_prompts=[resolved_prompt],
                            gt_masks=None,
                            gt_intrinsics=target_intr_s,
                            spatial_qualifier_idx=sq_tensor_s,
                        )

                pred_s = outputs_s.get('pred_masks')
                if pred_s is None:
                    continue
                if pred_s.dim() == 4:
                    pred_s = pred_s[:, 0]
                pred_s = pred_s.squeeze(0)
                if pred_s.shape != gt_mask_s.shape:
                    pred_s = F.interpolate(
                        pred_s.unsqueeze(0).unsqueeze(0).float(),
                        size=gt_mask_s.shape[-2:], mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)

                gt_bin_s = (gt_mask_s > 0.5).float()
                rel_s = torch.sigmoid(pred_s)
                pred_bin_s = (rel_s > 0.5).float()
                inter_s = (pred_bin_s * gt_bin_s).sum()
                union_s = pred_bin_s.sum() + gt_bin_s.sum() - inter_s
                iou_s = (inter_s / (union_s + 1e-6)).item()

                # Check if spatial query selected the correct instance
                # by comparing IoU with GT for the spatially-selected instance
                # vs IoU with GT for other instances
                best_other_iou = 0.0
                for oidx, omask in enumerate(candidate_masks):
                    if oidx == gt_idx:
                        continue
                    omask_t = torch.from_numpy(omask).float().to(device)
                    obin = (omask_t > 0.5).float()
                    oi = (pred_bin_s * obin).sum()
                    ou = pred_bin_s.sum() + obin.sum() - oi
                    best_other_iou = max(best_other_iou, (oi / (ou + 1e-6)).item())

                correct = iou_s > best_other_iou
                spatial_ious.append(iou_s)
                spatial_correct.append(float(correct))
                spatial_details[qualifier].append(iou_s)

          except Exception as e:
            if ddp.is_main:
                import traceback
                print(f"Spatial eval error: {e}")
                traceback.print_exc()
            continue

    # DDP gather results from all ranks
    if ddp.is_distributed:
        # dist is imported at top of file (line 30)
        # Gather per-rank data to rank 0
        local_data = {
            'ious': all_ious,
            'loc_mask': all_loc_mask,
            'loc_bbox': all_loc_bbox,
            'scene_ious': dict(scene_ious),
            'scene_loc_mask': dict(scene_loc_mask),
            'scene_loc_bbox': dict(scene_loc_bbox),
            'frame_ious': dict(frame_ious),
            'frame_loc_mask': dict(frame_loc_mask),
            'frame_loc_bbox': dict(frame_loc_bbox),
            'category_ious': dict(category_ious),
            'category_loc_mask': dict(category_loc_mask),
            'category_loc_bbox': dict(category_loc_bbox),
            'spatial_ious': spatial_ious,
            'spatial_correct': spatial_correct,
            'spatial_details': {k: list(v) for k, v in spatial_details.items()},
        }
        gathered = [None] * ddp.world_size if ddp.is_main else None
        dist.gather_object(local_data, gathered, dst=0)

        if ddp.is_main:
            # Merge all ranks' data
            all_ious = []
            all_loc_mask = []
            all_loc_bbox = []
            scene_ious = defaultdict(list)
            scene_loc_mask = defaultdict(list)
            scene_loc_bbox = defaultdict(list)
            frame_ious = defaultdict(list)
            frame_loc_mask = defaultdict(list)
            frame_loc_bbox = defaultdict(list)
            category_ious = defaultdict(list)
            category_loc_mask = defaultdict(list)
            category_loc_bbox = defaultdict(list)
            spatial_ious = []
            spatial_correct = []
            spatial_details = defaultdict(list)
            for rank_data in gathered:
                all_ious.extend(rank_data['ious'])
                all_loc_mask.extend(rank_data['loc_mask'])
                all_loc_bbox.extend(rank_data['loc_bbox'])
                for s, v in rank_data['scene_ious'].items():
                    scene_ious[s].extend(v)
                for s, v in rank_data['scene_loc_mask'].items():
                    scene_loc_mask[s].extend(v)
                for s, v in rank_data['scene_loc_bbox'].items():
                    scene_loc_bbox[s].extend(v)
                for f, v in rank_data.get('frame_ious', {}).items():
                    frame_ious[f].extend(v)
                for f, v in rank_data.get('frame_loc_mask', {}).items():
                    frame_loc_mask[f].extend(v)
                for f, v in rank_data.get('frame_loc_bbox', {}).items():
                    frame_loc_bbox[f].extend(v)
                for c, v in rank_data['category_ious'].items():
                    category_ious[c].extend(v)
                for c, v in rank_data['category_loc_mask'].items():
                    category_loc_mask[c].extend(v)
                for c, v in rank_data['category_loc_bbox'].items():
                    category_loc_bbox[c].extend(v)
                spatial_ious.extend(rank_data.get('spatial_ious', []))
                spatial_correct.extend(rank_data.get('spatial_correct', []))
                for q, v in rank_data.get('spatial_details', {}).items():
                    spatial_details[q].extend(v)

    # Aggregate
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_loc_mask = np.mean(all_loc_mask) if all_loc_mask else 0.0
    mean_loc_bbox = np.mean(all_loc_bbox) if all_loc_bbox else 0.0
    per_scene_iou = {s: np.mean(v) for s, v in scene_ious.items()}
    per_scene_loc_mask = {s: np.mean(v) for s, v in scene_loc_mask.items()}
    per_scene_loc_bbox = {s: np.mean(v) for s, v in scene_loc_bbox.items()}
    per_frame_iou = {f: np.mean(v) for f, v in frame_ious.items()}
    per_frame_loc_mask_agg = {f: np.mean(v) for f, v in frame_loc_mask.items()}
    per_frame_loc_bbox_agg = {f: np.mean(v) for f, v in frame_loc_bbox.items()}
    per_category_iou = {c: np.mean(v) for c, v in category_ious.items()}
    per_category_loc_mask = {c: np.mean(v) for c, v in category_loc_mask.items()}
    per_category_loc_bbox = {c: np.mean(v) for c, v in category_loc_bbox.items()}
    global_miou = np.mean(list(per_category_iou.values())) if per_category_iou else 0.0

    # Spatial metrics
    spatial_mean_iou = np.mean(spatial_ious) if spatial_ious else None
    spatial_accuracy = np.mean(spatial_correct) if spatial_correct else None
    spatial_per_qualifier = {q: np.mean(v) for q, v in spatial_details.items()} if spatial_details else {}

    results = {
        'dataset': args.dataset,
        'model': 'baseline_sam3' if args.baseline_sam3 else 'triangulang',
        'num_samples': len(all_ious),
        'num_categories': len(per_category_iou),
        'prompt_aliases': args.prompt_aliases,
        'sample_iou': mean_iou,
        'global_miou': global_miou,
        'scene_miou': global_miou,
        'localization_accuracy_mask': mean_loc_mask,
        'localization_accuracy_bbox': mean_loc_bbox,
        'per_scene_iou': per_scene_iou,
        'per_scene_loc_mask': per_scene_loc_mask,
        'per_scene_loc_bbox': per_scene_loc_bbox,
        'per_frame_iou': per_frame_iou,
        'per_frame_loc_mask': per_frame_loc_mask_agg,
        'per_frame_loc_bbox': per_frame_loc_bbox_agg,
        'per_category_iou': per_category_iou,
        'per_category_loc_mask': per_category_loc_mask,
        'per_category_loc_bbox': per_category_loc_bbox,
    }
    if spatial_mean_iou is not None:
        results['spatial_eval'] = {
            'num_samples': len(spatial_ious),
            'num_multi_instance_correct': len(spatial_correct),
            'spatial_miou': spatial_mean_iou,
            'spatial_instance_accuracy': spatial_accuracy,
            'per_qualifier_iou': {q: float(v) for q, v in spatial_per_qualifier.items()},
        }

    if ddp.is_main:
        model_label = "BASELINE SAM3 (native)" if args.baseline_sam3 else "TrianguLang"
        ddp.print("\n" + "="*70)
        ddp.print(f"LERF-OVS EVALUATION RESULTS  [{model_label}]")
        ddp.print("="*70)
        ddp.print(f"Samples: {len(all_ious)}  |  Categories: {len(per_category_iou)}")
        ddp.print("-"*70)
        ddp.print(f"{'Sample-avg IoU:':<30} {100*mean_iou:>10.2f}%")
        ddp.print(f"{'Global mIoU (per-category):':<30} {100*global_miou:>10.2f}%")
        ddp.print(f"{'Loc Accuracy (mask):':<30} {100*mean_loc_mask:>10.2f}%")
        ddp.print(f"{'Loc Accuracy (bbox):':<30} {100*mean_loc_bbox:>10.2f}%")
        ddp.print("-"*70)
        ddp.print("Per-scene breakdown:")
        for scene in sorted(per_scene_iou.keys()):
            n = len(scene_ious[scene])
            ddp.print(f"  {scene:<20} IoU={100*per_scene_iou[scene]:5.1f}%  "
                      f"LocMask={100*per_scene_loc_mask[scene]:5.1f}%  "
                      f"LocBbox={100*per_scene_loc_bbox[scene]:5.1f}%  (n={n})")
        ddp.print("-"*70)
        per_frame_iou = {f: np.mean(v) for f, v in frame_ious.items()}
        per_frame_loc_mask_agg = {f: np.mean(v) for f, v in frame_loc_mask.items()}
        per_frame_loc_bbox_agg = {f: np.mean(v) for f, v in frame_loc_bbox.items()}
        ddp.print("Per-frame breakdown:")
        for fk in sorted(per_frame_iou.keys()):
            n = len(frame_ious[fk])
            ddp.print(f"  {fk:<40} IoU={100*per_frame_iou[fk]:5.1f}%  "
                      f"LocM={100*per_frame_loc_mask_agg[fk]:5.1f}%  "
                      f"LocB={100*per_frame_loc_bbox_agg[fk]:5.1f}%  (n={n})")
        ddp.print("-"*70)
        ddp.print("Per-category breakdown (grouped by scene):")
        ddp.print(f"  {'Category':<30} {'IoU':>6}  {'LocM':>6}  {'LocB':>6}  {'n':>3}")
        # Group categories by scene (keys are "scene/category")
        scene_to_cats = defaultdict(list)
        for cat_key in per_category_iou.keys():
            scene, cat = cat_key.split('/', 1)
            scene_to_cats[scene].append((cat, cat_key))
        for scene in sorted(scene_to_cats.keys()):
            entries = sorted(scene_to_cats[scene], key=lambda x: x[0])
            scene_cat_ious = [per_category_iou[ck] for _, ck in entries]
            scene_cat_miou = np.mean(scene_cat_ious)
            ddp.print(f"  {scene} ({len(entries)} queries, mIoU={100*scene_cat_miou:.1f}%)")
            for cat, cat_key in entries:
                n = len(category_ious[cat_key])
                ddp.print(f"    {cat:<28} {100*per_category_iou[cat_key]:5.1f}%  "
                          f"{100*per_category_loc_mask.get(cat_key,0):5.1f}%  "
                          f"{100*per_category_loc_bbox.get(cat_key,0):5.1f}%  {n:>3}")
        ddp.print("="*70)
        if spatial_mean_iou is not None:
            ddp.print(f"\nSpatial Reasoning")
            ddp.print(f"  Spatial samples: {len(spatial_ious)}")
            ddp.print(f"  Spatial mIoU: {100*spatial_mean_iou:.2f}%")
            if spatial_accuracy is not None:
                ddp.print(f"  Multi-instance accuracy: {100*spatial_accuracy:.1f}% "
                          f"({sum(1 for x in spatial_correct if x > 0.5)}/{len(spatial_correct)})")
            ddp.print(f"  Per-qualifier breakdown:")
            for q in LERF_SPATIAL_QUALIFIERS:
                if q in spatial_per_qualifier:
                    n_q = len(spatial_details[q])
                    ddp.print(f"    {q:<12} IoU={100*spatial_per_qualifier[q]:5.1f}%  (n={n_q})")
            ddp.print("="*70)
        if save_lerf_viz:
            ddp.print(f"\nVisualizations ({lerf_viz_count} saved) at: {lerf_viz_dir}")
        else:
            ddp.print(f"\n(Use --visualize to save viz images, --viz-samples N to control count)")

        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        ddp.print(f"Results saved to: {results_file}")

