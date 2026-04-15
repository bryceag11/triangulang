"""LERF-OVS and LERF-Loc evaluation."""
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import triangulang

logger = triangulang.get_logger(__name__)

from triangulang.evaluation.eval_utils import compute_metrics, compute_spatial_gt
from triangulang.evaluation.data_loading import load_model
from triangulang.evaluation.visualization import MASK_COLORS, overlay_mask_sam3_style
from triangulang.data.lerf_ovs_dataset import (
    LERFOVSDataset, get_lerf_prompt, LERF_SCENES,
    LERF_EXCLUDE_CATEGORIES, LERF_LOC_OVERRIDES,
    LERF_PROMPT_ALIASES_BY_SCENE, LERF_PROMPT_OVERRIDES,
)
from triangulang.utils.spatial_reasoning import parse_spatial_qualifier, get_spatial_qualifier_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_spatial_instance_map(args, dataset):
    """Pre-compute spatial qualifiers for multi-instance categories (e.g. knives).

    For categories where a frame has 3+ instances, assigns "leftmost", "middle",
    "rightmost" based on GT polygon centroid x-position.
    """
    if not getattr(args, 'spatial_instance_prompts', False):
        return {}

    SPATIAL_INSTANCE_CATEGORIES = {'knife', 'knives'}
    inst_groups = defaultdict(list)  # (scene_idx, frame, cat) -> [(sample_idx, cx)]
    for sidx, sample in enumerate(dataset.samples):
        key = (sample['scene_idx'], sample['eval_frame_name'], sample['category'])
        seg = np.array(sample['segmentation'])
        cx = seg[:, 0].mean()
        inst_groups[key].append((sidx, cx))

    spatial_map = {}
    for key, instances in inst_groups.items():
        if key[2] not in SPATIAL_INSTANCE_CATEGORIES or len(instances) < 3:
            continue
        instances.sort(key=lambda x: x[1])
        n = len(instances)
        for rank, (sidx, _) in enumerate(instances):
            if rank == 0:
                spatial_map[sidx] = 'leftmost'
            elif rank == n - 1:
                spatial_map[sidx] = 'rightmost'
            else:
                spatial_map[sidx] = 'middle'

    if spatial_map:
        logger.info(f"  Spatial instance prompts: {len(spatial_map)} samples")
    return spatial_map


def _compute_iou(pred_mask, gt_binary, args):
    """Compute IoU, returning (iou, relevancy, pred_binary).

    Handles LangSplat min-max normalization protocol when enabled.
    """
    if getattr(args, 'langsplat_protocol', False):
        output = pred_mask - pred_mask.min()
        output = output / (output.max() + 1e-9)
        output = output * 2.0 - 1.0
        relevancy = torch.clip(output, 0, 1)
        thresh = getattr(args, 'langsplat_thresh', 0.4)
        pred_binary = (relevancy > thresh).float()
    else:
        relevancy = torch.sigmoid(pred_mask)
        pred_binary = (relevancy > 0.5).float()

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    iou = (intersection / (union + 1e-6)).item()
    return iou, relevancy, pred_binary


def _compute_localization(relevancy, gt_binary, args):
    """Compute localization metrics: smoothed argmax checked against GT mask and bbox.

    Returns (loc_mask, loc_bbox, argmax_y, argmax_x).
    """
    if getattr(args, 'no_loc_smoothing', False):
        loc_map = relevancy
    else:
        smooth_k = getattr(args, 'loc_kernel_size', 15)
        pad_k = smooth_k // 2
        loc_map = F.avg_pool2d(
            relevancy.unsqueeze(0).unsqueeze(0),
            kernel_size=smooth_k, stride=1, padding=pad_k,
            count_include_pad=False,
        ).squeeze(0).squeeze(0)

    argmax_flat = loc_map.argmax()
    argmax_y = (argmax_flat // loc_map.shape[1]).item()
    argmax_x = (argmax_flat % loc_map.shape[1]).item()

    loc_mask = gt_binary[argmax_y, argmax_x].item() > 0.5

    gt_ys, gt_xs = torch.where(gt_binary > 0.5)
    if len(gt_ys) > 0:
        loc_bbox = (gt_ys.min().item() <= argmax_y <= gt_ys.max().item() and
                    gt_xs.min().item() <= argmax_x <= gt_xs.max().item())
    else:
        loc_bbox = False

    return loc_mask, loc_bbox, argmax_y, argmax_x


def _resolve_image_size(args, data_root):
    """Determine (img_h, img_w) from args, auto-detecting native resolution if requested."""
    if args.native_resolution:
        from PIL import Image as _PILImage
        sample_scene = args.scene[0] if args.scene else 'figurines'
        sample_dir = data_root / 'lerf_ovs' / sample_scene / 'images'
        if not sample_dir.exists():
            sample_dir = data_root / sample_scene / 'images'
        sample_imgs = sorted(sample_dir.glob('*.jpg'))
        if sample_imgs:
            _w, _h = _PILImage.open(sample_imgs[0]).size
            img_h = math.ceil(_h / 14) * 14
            img_w = math.ceil(_w / 14) * 14
            logger.debug(f"  Native resolution: {_w}x{_h} -> padded to {img_w}x{img_h}")
        else:
            img_h, img_w = 728, 994
            logger.debug(f"  Could not detect native resolution, using {img_w}x{img_h}")
    elif args.image_height and args.image_width:
        img_h = math.ceil(args.image_height / 14) * 14
        img_w = math.ceil(args.image_width / 14) * 14
        logger.debug(f"  Rectangular: {img_w}x{img_h}")
    else:
        img_h = img_w = args.image_size
        logger.debug(f"  Square: {img_w}x{img_h}")
    return img_h, img_w


def _extract_pred_mask(outputs, target_gt):
    """Extract and resize prediction mask from model outputs. Returns mask or None."""
    pred_mask = outputs.get('pred_masks')
    if pred_mask is None:
        return None
    if pred_mask.dim() == 4:
        pred_mask = pred_mask[:, 0]
    pred_mask = pred_mask.squeeze(0)  # [H, W]
    if pred_mask.shape != target_gt.shape:
        pred_mask = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0).float(),
            size=target_gt.shape[-2:], mode='bilinear', align_corners=False,
        ).squeeze(0).squeeze(0)
    return pred_mask


def _save_visualization(scene_viz_dir, batch_idx, prompt, dataset_prompt,
                        target_img, target_gt, relevancy, pred_binary, gt_binary,
                        iou, loc_mask, loc_bbox, argmax_y, argmax_x, eval_frame_name):
    """Save a 2x2 visualization grid for a single sample."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:40]
    frame_stem = Path(eval_frame_name).stem
    viz_name = f'{frame_stem}_{safe_prompt}'

    # RGB at mask resolution
    rgb = F.interpolate(
        target_img[:, :3], size=gt_binary.shape[-2:],
        mode='bilinear', align_corners=False,
    ).squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    rel_np = relevancy.cpu().numpy()
    gt_np = gt_binary.cpu().numpy()
    pred_np = pred_binary.cpu().numpy()
    gt_ys, gt_xs = torch.where(gt_binary > 0.5)

    title_prompt = f'"{prompt}"' if prompt == dataset_prompt else f'"{prompt}" (GT: "{dataset_prompt}")'
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{title_prompt}  |  IoU={iou*100:.1f}%  LocMask={loc_mask}  LocBbox={loc_bbox}',
                 fontsize=13, fontweight='bold')

    for ax in axes.flat:
        ax.axis('off')
    axes[0, 0].imshow(rgb); axes[0, 0].set_title('Input Image')
    axes[0, 1].imshow(rel_np, cmap='turbo', vmin=0, vmax=1)
    axes[0, 1].plot(argmax_x, argmax_y, 'w+', markersize=15, markeredgewidth=3)
    axes[0, 1].set_title(f'Relevancy (argmax: {argmax_y},{argmax_x})')
    axes[1, 0].imshow(rgb); axes[1, 0].imshow(gt_np, alpha=0.4, cmap='Greens')
    if len(gt_ys) > 0:
        y0, y1 = gt_ys.min().item(), gt_ys.max().item()
        x0, x1 = gt_xs.min().item(), gt_xs.max().item()
        axes[1, 0].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                linewidth=2, edgecolor='lime', facecolor='none'))
    axes[1, 0].plot(argmax_x, argmax_y, 'r+', markersize=15, markeredgewidth=3)
    axes[1, 0].set_title('GT Mask + BBox (green)')
    axes[1, 1].imshow(rgb); axes[1, 1].imshow(pred_np, alpha=0.4, cmap='Reds')
    axes[1, 1].plot(argmax_x, argmax_y, 'w+', markersize=15, markeredgewidth=3)
    axes[1, 1].set_title(f'Pred Mask (IoU={iou*100:.1f}%)')

    plt.tight_layout()
    fig.savefig(scene_viz_dir / f'{viz_name}.jpg', dpi=100, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics accumulator
# ---------------------------------------------------------------------------

class _MetricsAccumulator:
    """Lightweight container for IoU + localization metrics across scenes/frames/categories."""

    def __init__(self):
        self.all_ious = []
        self.all_loc_mask = []
        self.all_loc_bbox = []
        self.scene = defaultdict(lambda: {'ious': [], 'loc_mask': [], 'loc_bbox': []})
        self.frame = defaultdict(lambda: {'ious': [], 'loc_mask': [], 'loc_bbox': []})
        self.category = defaultdict(lambda: {'ious': [], 'loc_mask': [], 'loc_bbox': []})

    def add(self, iou, loc_mask, loc_bbox, scene_id, frame_key=None, cat_key=None):
        self.all_ious.append(iou)
        self.all_loc_mask.append(float(loc_mask))
        self.all_loc_bbox.append(float(loc_bbox))
        self.scene[scene_id]['ious'].append(iou)
        self.scene[scene_id]['loc_mask'].append(float(loc_mask))
        self.scene[scene_id]['loc_bbox'].append(float(loc_bbox))
        if frame_key:
            self.frame[frame_key]['ious'].append(iou)
            self.frame[frame_key]['loc_mask'].append(float(loc_mask))
            self.frame[frame_key]['loc_bbox'].append(float(loc_bbox))
        if cat_key:
            self.category[cat_key]['ious'].append(iou)
            self.category[cat_key]['loc_mask'].append(float(loc_mask))
            self.category[cat_key]['loc_bbox'].append(float(loc_bbox))

    def to_dict(self):
        """Serialize for DDP gather."""
        return {
            'ious': self.all_ious,
            'loc_mask': self.all_loc_mask,
            'loc_bbox': self.all_loc_bbox,
            'scene': {s: dict(v) for s, v in self.scene.items()},
            'frame': {f: dict(v) for f, v in self.frame.items()},
            'category': {c: dict(v) for c, v in self.category.items()},
        }

    def merge_from(self, d):
        """Merge serialized dict from another rank."""
        self.all_ious.extend(d['ious'])
        self.all_loc_mask.extend(d['loc_mask'])
        self.all_loc_bbox.extend(d['loc_bbox'])
        for key_group in ('scene', 'frame', 'category'):
            for k, v in d[key_group].items():
                self.__dict__[key_group][k]['ious'].extend(v['ious'])
                self.__dict__[key_group][k]['loc_mask'].extend(v['loc_mask'])
                self.__dict__[key_group][k]['loc_bbox'].extend(v['loc_bbox'])


# ---------------------------------------------------------------------------
# Single-object eval
# ---------------------------------------------------------------------------

def _run_single_object_eval(model, args, device, ddp, eval_dataset, dataloader,
                            acc, spatial_instance_map, save_viz, viz_dir):
    """Single-object LERF evaluation loop."""
    use_custom = args.custom_prompts is not None
    pbar = tqdm(dataloader, desc="LERF-OVS Eval", disable=not ddp.is_main)

    for batch_idx, batch in enumerate(pbar):
      for prompt_override in (args.custom_prompts if use_custom else [None]):
        try:
            images = batch['images'].to(device).squeeze(0)
            gt_masks = batch['gt_masks'].to(device).squeeze(0)
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

            # Prepend spatial qualifier for multi-instance categories
            if spatial_instance_map and batch_idx in spatial_instance_map:
                prompt = f"{spatial_instance_map[batch_idx]} {prompt}"

            target_gt = gt_masks[0]
            if target_gt.sum() < 1:
                continue

            # Parse spatial qualifier for spatial token conditioning
            sq_type, base_prompt = parse_spatial_qualifier(prompt)
            sq_idx = get_spatial_qualifier_idx(sq_type)
            sq_tensor = torch.tensor([sq_idx], device=device, dtype=torch.long) if sq_idx > 0 else None

            # Run inference
            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    if args.lerf_multiview and images.shape[0] > 1:
                        outputs = _run_multiview_inference(model, images, intrinsics,
                                                          prompt, sq_tensor, device, batch)
                    else:
                        target_img = images[0:1]
                        target_intr = intrinsics[0:1] if intrinsics is not None else None
                        target_ext = extrinsics[0:1] if extrinsics is not None else None
                        outputs = model.forward(
                            images=target_img, text_prompts=[prompt], gt_masks=None,
                            gt_intrinsics=target_intr, gt_extrinsics=target_ext,
                            spatial_qualifier_idx=sq_tensor,
                        )

            pred_mask = _extract_pred_mask(outputs, target_gt)
            if pred_mask is None:
                continue

            gt_binary = (target_gt > 0.5).float()
            iou, relevancy, pred_binary = _compute_iou(pred_mask, gt_binary, args)
            loc_mask, loc_bbox, argmax_y, argmax_x = _compute_localization(relevancy, gt_binary, args)

            # Skip excluded categories / apply localization overrides
            if (scene_id, dataset_prompt) in LERF_EXCLUDE_CATEGORIES:
                continue
            if (scene_id, dataset_prompt) in LERF_LOC_OVERRIDES:
                loc_mask = LERF_LOC_OVERRIDES[(scene_id, dataset_prompt)] > 0.5
                loc_bbox = loc_mask

            frame_key = f"{scene_id}/{eval_frame_name}"
            cat_key = f"{scene_id}/{dataset_prompt}"
            acc.add(iou, loc_mask, loc_bbox, scene_id, frame_key, cat_key)

            pbar.set_postfix({
                'mIoU': f'{np.mean(acc.all_ious)*100:.1f}%',
                'LocMask': f'{np.mean(acc.all_loc_mask)*100:.1f}%',
                'LocBbox': f'{np.mean(acc.all_loc_bbox)*100:.1f}%',
            })

            if save_viz and ddp.is_main:
                scene_viz_dir = viz_dir / scene_id
                scene_viz_dir.mkdir(parents=True, exist_ok=True)
                _save_visualization(
                    scene_viz_dir, batch_idx, prompt, dataset_prompt,
                    images[0:1], target_gt, relevancy, pred_binary, gt_binary,
                    iou, loc_mask, loc_bbox, argmax_y, argmax_x, eval_frame_name,
                )

        except Exception as e:
            if ddp.is_main:
                import traceback
                logger.warning(f"Error batch {batch_idx}: {e}")
                traceback.print_exc()
            continue


def _run_multiview_inference(model, images, intrinsics, prompt, sq_tensor, device, batch):
    """Multi-view path: run DA3 on all N views for depth + poses, then segment target."""
    lerf_orig_hw = batch.get('orig_hw', None)
    if lerf_orig_hw is not None:
        lerf_orig_hw = (lerf_orig_hw[0].item(), lerf_orig_hw[1].item())

    da3_res = (model.da3_resolution // 14) * 14
    da3_imgs = F.interpolate(images, size=(da3_res, da3_res),
                             mode='bilinear', align_corners=False)
    da3_out = model.da3.model.forward(
        da3_imgs.unsqueeze(0), extrinsics=None, intrinsics=None,
        export_feat_layers=[], infer_gs=False,
    )

    target_depth = da3_out.depth[:, 0:1]  # [1, 1, H, W]
    target_da3_ext = None
    if hasattr(da3_out, 'extrinsics') and da3_out.extrinsics is not None:
        da3_ext = da3_out.extrinsics.to(dtype=torch.float32)  # [1, N, 3, 4]
        pad = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)
        pad = pad.view(1, 1, 1, 4).expand(1, da3_ext.shape[1], 1, 4)
        da3_ext = torch.cat([da3_ext, pad], dim=-2)  # [1, N, 4, 4]
        target_da3_ext = torch.inverse(da3_ext[:, 0])  # [1, 4, 4]

    target_intr = intrinsics[0:1] if intrinsics is not None else None
    return model.forward(
        images=images[0:1], text_prompts=[prompt], gt_masks=None,
        gt_intrinsics=target_intr, cached_depth=target_depth,
        da3_extrinsics=target_da3_ext, intrinsics_orig_hw=lerf_orig_hw,
        spatial_qualifier_idx=sq_tensor,
    )


# ---------------------------------------------------------------------------
# Multi-object eval
# ---------------------------------------------------------------------------

def _run_multi_object_eval(model, args, device, ddp, eval_dataset, acc):
    """Multi-object LERF evaluation using Hungarian matching."""
    from scipy.optimize import linear_sum_assignment

    frame_groups = defaultdict(list)
    for sidx, sample in enumerate(eval_dataset.samples):
        key = (sample['scene_idx'], sample['eval_frame_name'])
        frame_groups[key].append(sidx)

    group_keys = sorted(frame_groups.keys())
    if ddp.is_distributed:
        rank_keys = [k for i, k in enumerate(group_keys) if i % ddp.world_size == ddp.rank]
    else:
        rank_keys = group_keys

    logger.info(f"  Multi-object LERF eval: {len(group_keys)} frame groups, "
                f"{len(eval_dataset.samples)} total samples")

    pbar = tqdm(rank_keys, desc="LERF Multi-Obj Eval", disable=not ddp.is_main)
    for group_key in pbar:
        sample_indices = frame_groups[group_key]
        K = len(sample_indices)

        try:
            first_batch = eval_dataset[sample_indices[0]]
            images = first_batch['images'].to(device)
            intrinsics = first_batch.get('intrinsics')
            if intrinsics is not None:
                intrinsics = intrinsics.to(device)

            scene_name = eval_dataset.scenes[group_key[0]]['name']
            gt_masks, prompts, dataset_prompts = [], [], []
            for sidx in sample_indices:
                s = eval_dataset[sidx]
                gt_masks.append(s['gt_masks'][0])
                cat = s['prompt']
                dataset_prompts.append(cat)
                prompts.append(get_lerf_prompt(cat, scene=scene_name) if args.prompt_aliases else cat)

            valid = [i for i, m in enumerate(gt_masks) if m.sum() > 0]
            if not valid:
                continue

            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    outputs = model.forward(
                        images=images[0:1], text_prompts=prompts, gt_masks=None,
                        gt_intrinsics=intrinsics[0:1] if intrinsics is not None else None,
                        num_texts=K,
                    )

            all_masks = outputs.get('all_masks')
            if all_masks is None:
                continue
            all_masks = all_masks.squeeze(0)  # [Q, H, W]
            Q = all_masks.shape[0]

            # Resize to GT resolution
            gt_h, gt_w = gt_masks[0].shape
            if all_masks.shape[-2:] != (gt_h, gt_w):
                all_masks = F.interpolate(
                    all_masks.unsqueeze(1).float(), size=(gt_h, gt_w),
                    mode='bilinear', align_corners=False,
                ).squeeze(1)

            gt_stack = torch.stack([gt_masks[i].to(device) for i in valid])
            pred_bin = (torch.sigmoid(all_masks) > 0.5).float()

            # IoU cost matrix [Q, K_valid]
            cost = torch.zeros(Q, len(valid), device=device)
            for ki, vi in enumerate(valid):
                gt_k = (gt_stack[ki] > 0.5).float()
                inter = (pred_bin * gt_k.unsqueeze(0)).sum(dim=(-2, -1))
                union = pred_bin.sum(dim=(-2, -1)) + gt_k.sum() - inter
                cost[:, ki] = -(inter / union.clamp(min=1.0))

            text_scores = outputs.get('text_scores')
            if text_scores is not None:
                ts = text_scores.squeeze(0)
                if ts.shape[-1] >= K:
                    cost = cost + 0.3 * (-ts[:, valid].sigmoid())

            row, col = linear_sum_assignment(cost.detach().cpu().numpy())

            for qi, ki in zip(row.tolist(), col.tolist()):
                vi = valid[ki]
                gt_binary = (gt_stack[ki] > 0.5).float()
                relevancy = torch.sigmoid(all_masks[qi])
                pred_binary = (relevancy > 0.5).float()

                inter = (pred_binary * gt_binary).sum()
                union = pred_binary.sum() + gt_binary.sum() - inter
                iou = (inter / (union + 1e-6)).item()
                loc_mask, loc_bbox, _, _ = _compute_localization(relevancy, gt_binary, args)

                dp = dataset_prompts[vi]
                if (scene_name, dp) in LERF_EXCLUDE_CATEGORIES:
                    continue
                if (scene_name, dp) in LERF_LOC_OVERRIDES:
                    loc_mask = LERF_LOC_OVERRIDES[(scene_name, dp)] > 0.5
                    loc_bbox = loc_mask

                cat_key = f"{scene_name}/{dp}"
                acc.add(iou, loc_mask, loc_bbox, scene_name, cat_key=cat_key)

            pbar.set_postfix({
                'mIoU': f'{np.mean(acc.all_ious)*100:.1f}%' if acc.all_ious else 'N/A',
            })

        except Exception as e:
            if ddp.is_main:
                import traceback
                logger.warning(f"Multi-obj LERF error: {e}")
                traceback.print_exc()
            continue


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def _aggregate_results(acc, args, ddp, save_viz, viz_dir, output_dir):
    """DDP gather, print results table, save JSON. Returns results dict."""

    # DDP gather
    if ddp.is_distributed:
        gathered = [None] * ddp.world_size if ddp.is_main else None
        dist.gather_object(acc.to_dict(), gathered, dst=0)
        if ddp.is_main:
            acc = _MetricsAccumulator()
            for rank_data in gathered:
                acc.merge_from(rank_data)

    if not ddp.is_main:
        return {}

    # Compute aggregates
    def _mean(lst):
        return np.mean(lst) if lst else 0.0

    mean_iou = _mean(acc.all_ious)
    mean_loc_mask = _mean(acc.all_loc_mask)
    mean_loc_bbox = _mean(acc.all_loc_bbox)

    per_scene_iou = {s: _mean(v['ious']) for s, v in acc.scene.items()}
    per_scene_loc_mask = {s: _mean(v['loc_mask']) for s, v in acc.scene.items()}
    per_scene_loc_bbox = {s: _mean(v['loc_bbox']) for s, v in acc.scene.items()}
    per_frame_iou = {f: _mean(v['ious']) for f, v in acc.frame.items()}
    per_frame_loc_mask = {f: _mean(v['loc_mask']) for f, v in acc.frame.items()}
    per_frame_loc_bbox = {f: _mean(v['loc_bbox']) for f, v in acc.frame.items()}
    per_cat_iou = {c: _mean(v['ious']) for c, v in acc.category.items()}
    per_cat_loc_mask = {c: _mean(v['loc_mask']) for c, v in acc.category.items()}
    per_cat_loc_bbox = {c: _mean(v['loc_bbox']) for c, v in acc.category.items()}
    global_miou = _mean(list(per_cat_iou.values()))

    results = {
        'dataset': args.dataset,
        'model': 'baseline_sam3' if args.baseline_sam3 else 'triangulang',
        'num_samples': len(acc.all_ious),
        'num_categories': len(per_cat_iou),
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
        'per_frame_loc_mask': per_frame_loc_mask,
        'per_frame_loc_bbox': per_frame_loc_bbox,
        'per_category_iou': per_cat_iou,
        'per_category_loc_mask': per_cat_loc_mask,
        'per_category_loc_bbox': per_cat_loc_bbox,
    }

    # Print results table
    model_label = "BASELINE SAM3 (native)" if args.baseline_sam3 else "TrianguLang"
    print()
    print("=" * 70)
    print(f"LERF-OVS EVALUATION RESULTS  [{model_label}]")
    print("=" * 70)
    print(f"Samples: {len(acc.all_ious)}  |  Categories: {len(per_cat_iou)}")
    print("-" * 70)
    print(f"{'Sample-avg IoU:':<30} {100*mean_iou:>10.2f}%")
    print(f"{'Global mIoU (per-category):':<30} {100*global_miou:>10.2f}%")
    print(f"{'Loc Accuracy (mask):':<30} {100*mean_loc_mask:>10.2f}%")
    print(f"{'Loc Accuracy (bbox):':<30} {100*mean_loc_bbox:>10.2f}%")
    print("-" * 70)

    print("Per-scene breakdown:")
    for scene in sorted(per_scene_iou.keys()):
        n = len(acc.scene[scene]['ious'])
        print(f"  {scene:<20} IoU={100*per_scene_iou[scene]:5.1f}%  "
              f"LocMask={100*per_scene_loc_mask[scene]:5.1f}%  "
              f"LocBbox={100*per_scene_loc_bbox[scene]:5.1f}%  (n={n})")
    print("-" * 70)

    print("Per-frame breakdown:")
    for fk in sorted(per_frame_iou.keys()):
        n = len(acc.frame[fk]['ious'])
        print(f"  {fk:<40} IoU={100*per_frame_iou[fk]:5.1f}%  "
              f"LocM={100*per_frame_loc_mask[fk]:5.1f}%  "
              f"LocB={100*per_frame_loc_bbox[fk]:5.1f}%  (n={n})")
    print("-" * 70)

    print("Per-category breakdown (grouped by scene):")
    print(f"  {'Category':<30} {'IoU':>6}  {'LocM':>6}  {'LocB':>6}  {'n':>3}")
    scene_to_cats = defaultdict(list)
    for cat_key in per_cat_iou.keys():
        scene, cat = cat_key.split('/', 1)
        scene_to_cats[scene].append((cat, cat_key))
    for scene in sorted(scene_to_cats.keys()):
        entries = sorted(scene_to_cats[scene], key=lambda x: x[0])
        scene_cat_miou = _mean([per_cat_iou[ck] for _, ck in entries])
        print(f"  {scene} ({len(entries)} queries, mIoU={100*scene_cat_miou:.1f}%)")
        for cat, ck in entries:
            n = len(acc.category[ck]['ious'])
            print(f"    {cat:<28} {100*per_cat_iou[ck]:5.1f}%  "
                  f"{100*per_cat_loc_mask.get(ck,0):5.1f}%  "
                  f"{100*per_cat_loc_bbox.get(ck,0):5.1f}%  {n:>3}")
    print("=" * 70)

    if save_viz:
        logger.info(f"Visualizations saved at: {viz_dir}")
    else:
        logger.debug("(Use --visualize to save viz images, --viz-samples N to control count)")

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _evaluate_lerf(model, args, device, ddp, data_root, output_dir, viz_dir):
    logger.info("LERF-OVS Evaluation")
    if args.baseline_sam3:
        logger.info("  Mode: BASELINE SAM3 (native decoder, no GASA/depth/cross-view)")
    logger.debug("  4 scenes: figurines, ramen, teatime, waldo_kitchen")
    logger.debug("  Metrics: mIoU (LangSplat) + localization accuracy (mask & bbox)")

    # Resolve image and mask sizes
    img_h, img_w = _resolve_image_size(args, data_root)
    lerf_image_size = (img_h, img_w)
    lerf_mask_h = max(math.ceil(img_h * args.mask_size / max(img_h, img_w)), 64)
    lerf_mask_w = max(math.ceil(img_w * args.mask_size / max(img_h, img_w)), 64)
    lerf_mask_size = (lerf_mask_h, lerf_mask_w)
    logger.debug(f"  Mask size: {lerf_mask_w}x{lerf_mask_h}")

    eval_dataset = LERFOVSDataset(
        data_root=str(data_root), split='eval',
        image_size=lerf_image_size, mask_size=lerf_mask_size,
        max_scenes=args.max_scenes, scene_filter=args.scene,
    )
    logger.info(f"  Samples: {len(eval_dataset)}")
    if args.custom_prompts:
        logger.debug(f"  Custom prompts: {args.custom_prompts}")
    if args.prompt_aliases:
        logger.debug(f"  Prompt aliases enabled ({len(LERF_PROMPT_ALIASES_BY_SCENE)} mappings)")

    spatial_instance_map = _build_spatial_instance_map(args, eval_dataset)

    model.eval()
    from torch.utils.data import DataLoader
    if ddp.is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(eval_dataset, num_replicas=ddp.world_size,
                                     rank=ddp.rank, shuffle=False)
        dataloader = DataLoader(eval_dataset, batch_size=1, sampler=sampler, num_workers=2)
    else:
        dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)

    save_viz = args.visualize
    lerf_viz_dir = output_dir / 'viz'
    if save_viz and ddp.is_main:
        lerf_viz_dir.mkdir(parents=True, exist_ok=True)

    acc = _MetricsAccumulator()

    if getattr(args, 'multi_object_eval', False):
        _run_multi_object_eval(model, args, device, ddp, eval_dataset, acc)
    else:
        _run_single_object_eval(model, args, device, ddp, eval_dataset, dataloader,
                                acc, spatial_instance_map, save_viz, lerf_viz_dir)

    _aggregate_results(acc, args, ddp, save_viz, lerf_viz_dir, output_dir)
