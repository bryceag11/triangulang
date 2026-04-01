"""Scene-level evaluation: single-object, multi-object, backbone precomputation."""
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

from triangulang.models.triangulang_model import TrianguLangModel
from triangulang.utils.prompt_augmentor import PromptAugmentor
from triangulang.utils.scannetpp_loader import normalize_label, is_excluded_frame, SCANNETPP_SKIP_LABELS
from triangulang.utils.spatial_reasoning import (
    get_mask_centroid, get_depth_at_centroid,
    parse_spatial_qualifier, get_spatial_qualifier_idx,
)
from triangulang.models.sheaf_embeddings import compute_3d_localization, format_localization_text
from triangulang.evaluation.eval_utils import (
    create_prompts_from_gt, compute_metrics, compute_oracle_iou,
    compute_3d_centroid, compute_centroid_error, umeyama_alignment,
    compute_cross_view_consistency, compute_spatial_gt,
)
from triangulang.evaluation.data_loading import (
    load_scene_data, load_gt_masks, load_gt_poses, get_frame_extrinsics,
    load_cached_da3_nested, load_gt_centroids, load_gt_poses_for_scene,
)
from triangulang.evaluation.visualization import (
    MASK_COLORS, overlay_mask_sam3_style,
    save_visualization, generate_paper_visualizations, generate_single_object_viz,
)
from triangulang.data.dataset_factory import get_dataset, get_dataset_config
from triangulang.utils.metrics import compute_gt_centroid as compute_gt_centroid_util



from triangulang.evaluation.eval_single_prompt import (
    evaluate_multiview_single_prompt,
    evaluate_scene_single_prompt,
)
from triangulang.evaluation.eval_scene_helpers import (
    _prepare_spatial_queries,
    _prepare_multi_instance_eval,
    _evaluate_single_object,
    _aggregate_scene_results,
)

def _evaluate_scene_multi_object(
    model, scene_path, semantics_dir, device, images, image_size,
    eval_items, min_pixel_fraction, da3_cache_dir, precomputed_backbone,
    save_viz=False, viz_dir=None, viz_samples=5,
    temporal_smooth_alpha=0.0,
    use_crf=False,
):
    """Multi-object eval: batch all K objects per frame in a single forward pass.

    Instead of K separate forward passes per frame (one per object),
    passes all K text prompts at once. The model produces Q masks and
    text scores [Q, K], allowing us to assign the best query to each object.

    Speedup: ~Kx on decoder time (backbone already cached).
    """
    from scipy.optimize import linear_sum_assignment

    results = defaultdict(list)
    category_metrics = defaultdict(lambda: {'iou': [], 'oracle_iou': [], 'pixel_acc': [], 'recall': [], 'precision': [], 'f1': []})
    preprocess_times = []
    inference_times = []
    viz_data = []

    # Temporal EMA smoothing: maintain per-object logit history across frames
    prev_logits = {}  # k_idx -> [H, W] tensor on CPU
    use_temporal_smooth = temporal_smooth_alpha > 0.0

    # Build list of (label, obj_ids, prompt, cat_label) -- use spatial override as model prompt when available
    # cat_label: for spatial items, use the spatial prompt (e.g. "nearest chair") so reporting can find them;
    # for regular items, use the base label.
    object_list = []
    for label, oids, spatial_override in eval_items:
        prompt = spatial_override if spatial_override else label
        cat_label = spatial_override if spatial_override else label
        object_list.append((label, oids, prompt, cat_label))
    K = len(object_list)
    if K == 0:
        return {'error': 'no valid objects'}

    all_labels = [prompt for _, _, prompt, _ in object_list]

    for frame_idx, img_path in enumerate(images):
        frame_name = img_path.name
        frame_stem = img_path.stem
        gt_masks_dict = load_gt_masks(semantics_dir, frame_name)

        # Build GT masks for all K objects on this frame
        gt_per_object = []  # K entries: numpy mask or None
        valid_objects = []   # indices of objects with valid GT on this frame
        for k, (label, obj_ids, _prompt, _cat) in enumerate(object_list):
            gt_mask = None
            for oid in obj_ids:
                if oid in gt_masks_dict:
                    if gt_mask is None:
                        gt_mask = gt_masks_dict[oid].copy()
                    else:
                        gt_mask = np.maximum(gt_mask, gt_masks_dict[oid])
            if gt_mask is not None and gt_mask.sum() / gt_mask.size >= min_pixel_fraction:
                gt_per_object.append(gt_mask)
                valid_objects.append(k)
            else:
                gt_per_object.append(None)

        if not valid_objects:
            continue

        # Load image
        t_preprocess_start = time.perf_counter()
        img = Image.open(img_path).convert('RGB')
        img = img.resize(image_size, Image.BILINEAR)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

        # Load cached depth
        cached_depth = None
        da3_intrinsics = None
        da3_extrinsics = None
        if da3_cache_dir is not None:
            scene_id = scene_path.name
            cache_path = da3_cache_dir / scene_id / f"{frame_stem}.pt"
            if cache_path.exists():
                try:
                    cache_data = torch.load(cache_path, map_location='cpu', weights_only=True)
                    depth = cache_data['depth'].float()
                    if depth.dim() == 4:
                        depth = depth.squeeze(0)
                    elif depth.dim() == 2:
                        depth = depth.unsqueeze(0)
                    cached_depth = depth.unsqueeze(0).to(device)
                    if 'intrinsics' in cache_data:
                        da3_intrinsics = cache_data['intrinsics'].float().unsqueeze(0).to(device)
                    if 'extrinsics' in cache_data:
                        da3_extrinsics = cache_data['extrinsics'].float().unsqueeze(0).to(device)
                except Exception:
                    pass

        # Get precomputed backbone
        _frame_backbone = precomputed_backbone.get(frame_stem)

        # Build K text prompts (for all valid objects)
        # We pass ALL K labels so query-text alignment works correctly
        text_prompts_k = all_labels  # K prompts

        # Build per-object GT tensor [1, K, H, W] for oracle mask selection (iou_match)
        # Without this, the model falls back to confidence selection which is much worse.
        # This matches single-object eval which also passes GT for oracle selection.
        gt_for_model = None
        sam3_mo = getattr(model, 'sam3_multi_object', False)
        if sam3_mo:
            gt_list = []
            for k in range(K):
                if gt_per_object[k] is not None:
                    gt_t = torch.from_numpy(gt_per_object[k]).float()
                    gt_t = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), size=image_size,
                                         mode='nearest').squeeze(0).squeeze(0)
                else:
                    gt_t = torch.zeros(image_size, dtype=torch.float32)
                gt_list.append(gt_t)
            gt_for_model = torch.stack(gt_list).unsqueeze(0).to(device)  # [1, K, H, W]

        t_preprocess_end = time.perf_counter()
        preprocess_times.append(t_preprocess_end - t_preprocess_start)

        t_start = time.perf_counter()
        with autocast('cuda', dtype=torch.float16):
            outputs = model(
                img_tensor, text_prompts_k, gt_for_model,
                cached_depth=cached_depth,
                da3_intrinsics=da3_intrinsics,
                da3_extrinsics=da3_extrinsics,
                precomputed_sam3=_frame_backbone,
                num_texts=K,
            )
        t_end = time.perf_counter()
        inference_times.append(t_end - t_start)

        # SAM3-MO mode: pred_masks is [K, 1, H, W] -- one mask per object, already matched
        # Non-SAM3: all_masks is [1, Q, H, W] -- need Hungarian matching
        sam3_mo_K = outputs.get('sam3_mo_K')
        if sam3_mo and sam3_mo_K is not None:
            # SAM3-style: pred_masks[k] is the prediction for object k
            pred_masks_k = outputs['pred_masks']  # [K, 1, H, W]
            pred_masks_k = pred_masks_k.squeeze(1)  # [K, H, W]

            # Get all candidate masks for oracle computation: [K, Q, H_mask, W_mask]
            all_masks_mo = outputs.get('all_masks')  # [K, Q, H_mask, W_mask] in SAM3-MO mode

            # Resize to GT resolution
            gt_h, gt_w = gt_per_object[valid_objects[0]].shape
            mask_h, mask_w = pred_masks_k.shape[-2:]
            if (mask_h, mask_w) != (gt_h, gt_w):
                pred_masks_k = F.interpolate(
                    pred_masks_k.unsqueeze(1).float(), size=(gt_h, gt_w),
                    mode='bilinear', align_corners=False
                ).squeeze(1)

            # Temporal EMA smoothing on logits before thresholding
            if use_temporal_smooth:
                alpha = temporal_smooth_alpha
                for k_idx in range(pred_masks_k.shape[0]):
                    logit_cpu = pred_masks_k[k_idx].cpu().float()
                    if k_idx in prev_logits:
                        logit_cpu = alpha * logit_cpu + (1 - alpha) * prev_logits[k_idx]
                    prev_logits[k_idx] = logit_cpu
                    pred_masks_k[k_idx] = logit_cpu.to(device)

            # CRF / morphological post-processing to refine mask boundaries
            if use_crf:
                from triangulang.utils.crf_postprocess import morphological_smooth
                for k_idx in range(pred_masks_k.shape[0]):
                    mask_binary = (torch.sigmoid(pred_masks_k[k_idx]) > 0.5).cpu().numpy().astype(np.float32)
                    refined = morphological_smooth(mask_binary, kernel_size=7)
                    refined_logit = torch.from_numpy(refined * 10.0 - 5.0).to(device)
                    pred_masks_k[k_idx] = refined_logit

            # Direct 1:1 matching -- object k's mask is pred_masks_k[k]
            matched_pairs = [(k_idx, k_idx) for k_idx in valid_objects]  # (obj_idx, pred_idx)
            pred_source = pred_masks_k
            gt_source = gt_per_object
        else:
            # Original path: Hungarian matching on IoU
            all_masks = outputs.get('all_masks')  # [1, Q, H, W]
            if all_masks is None:
                continue
            all_masks = all_masks.squeeze(0)  # [Q, H, W]
            Q = all_masks.shape[0]

            # Resize masks to GT resolution
            gt_h, gt_w = gt_per_object[valid_objects[0]].shape
            mask_h, mask_w = all_masks.shape[-2:]
            if (mask_h, mask_w) != (gt_h, gt_w):
                all_masks = F.interpolate(
                    all_masks.unsqueeze(1).float(), size=(gt_h, gt_w),
                    mode='bilinear', align_corners=False
                ).squeeze(1)

            # Build GT tensor for valid objects
            gt_tensors = []
            for k_idx in valid_objects:
                gt_t = torch.from_numpy(gt_per_object[k_idx]).float().to(device)
                gt_tensors.append(gt_t)
            gt_stack = torch.stack(gt_tensors)  # [K_valid, H, W]
            K_valid = len(valid_objects)

            # IoU cost matrix [Q, K_valid]
            pred_binary = (torch.sigmoid(all_masks) > 0.5).float()
            cost_matrix = torch.zeros(Q, K_valid, device=device)
            for ki, _ in enumerate(valid_objects):
                gt_k = (gt_stack[ki] > 0.5).float()
                intersection = (pred_binary * gt_k.unsqueeze(0)).sum(dim=(-2, -1))
                union = pred_binary.sum(dim=(-2, -1)) + gt_k.sum() - intersection
                ious = intersection / union.clamp(min=1.0)
                cost_matrix[:, ki] = -ious

            # Add text score cost if available
            text_scores = outputs.get('text_scores')
            if text_scores is not None:
                ts = text_scores.squeeze(0)  # [Q, K]
                if ts.shape[-1] >= K:
                    valid_ts = ts[:, valid_objects]
                    cost_matrix = cost_matrix + 0.3 * (-valid_ts.sigmoid())

            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            # Build matched pairs: (obj_idx, query_idx)
            matched_pairs = [(valid_objects[ki], qi) for qi, ki in zip(row_ind.tolist(), col_ind.tolist())]
            pred_source = all_masks
            gt_source = None  # Use gt_stack indexed by col_ind
            # Remap: for Hungarian path, pred is query idx, gt is from gt_stack
            matched_pairs_hungarian = list(zip(row_ind.tolist(), col_ind.tolist()))

        # Spatial postprocessing: after getting all masks, assign spatial labels via geometry
        # This uses depth at predicted mask centroids to determine nearest/farthest/left/right
        # among same-label predicted masks -- leverages good segmentation + geometric reasoning
        spatial_postprocess_labels = {}  # obj_k -> list of spatial cat_labels
        if sam3_mo and sam3_mo_K is not None and cached_depth is not None:
            # Group predictions by base label
            from collections import defaultdict as _dd
            label_groups = _dd(list)
            for obj_k, pred_k in matched_pairs:
                base_label = object_list[obj_k][0]  # base label (not spatial prompt)
                label_groups[base_label].append((obj_k, pred_k))

            for base_label, group in label_groups.items():
                if len(group) < 2:
                    continue
                # Compute centroids and depth for predicted masks
                pred_centroids = []
                for obj_k, pred_k in group:
                    pm = torch.sigmoid(pred_source[pred_k])
                    if pm.sum() < 1:
                        pred_centroids.append((0.5, 0.5, 0.0))
                        continue
                    ys, xs = torch.where(pm > 0.5)
                    if len(xs) == 0:
                        pred_centroids.append((0.5, 0.5, 0.0))
                        continue
                    cx, cy = float(xs.float().mean()), float(ys.float().mean())
                    # Get depth at centroid
                    depth_map = cached_depth.squeeze()  # [H, W]
                    if depth_map.shape != pm.shape:
                        depth_map = F.interpolate(depth_map.unsqueeze(0).unsqueeze(0).float(),
                                                   size=pm.shape, mode='bilinear',
                                                   align_corners=False).squeeze()
                    cy_int = min(max(int(cy), 0), depth_map.shape[0] - 1)
                    cx_int = min(max(int(cx), 0), depth_map.shape[1] - 1)
                    d = float(depth_map[cy_int, cx_int])
                    pred_centroids.append((cx, cy, d))

                # Assign spatial labels
                depths = [c[2] for c in pred_centroids]
                xs = [c[0] for c in pred_centroids]
                for qualifier, values, fn in [
                    ('nearest', depths, min), ('farthest', depths, max),
                    ('leftmost', xs, min), ('rightmost', xs, max),
                ]:
                    if all(v == 0 for v in values):
                        continue
                    best_idx = values.index(fn(v for v in values if v != 0) if any(v != 0 for v in values) else 0)
                    obj_k = group[best_idx][0]
                    spatial_cat = f"{qualifier} {base_label}"
                    if obj_k not in spatial_postprocess_labels:
                        spatial_postprocess_labels[obj_k] = []
                    spatial_postprocess_labels[obj_k].append(spatial_cat)

        # Compute metrics for matched pairs
        if sam3_mo and sam3_mo_K is not None:
            for obj_k, pred_k in matched_pairs:
                cat_label = object_list[obj_k][3]  # cat_label: spatial prompt for spatial items, base label otherwise
                pred_mask = pred_source[pred_k]  # [H, W]
                gt_mask = torch.from_numpy(gt_per_object[obj_k]).float().to(device)

                metrics = compute_metrics(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
                results['iou'].append(metrics['iou'])

                # Compute true oracle IoU from all Q candidate masks
                oracle_iou = metrics['iou']  # fallback
                if all_masks_mo is not None and obj_k < all_masks_mo.shape[0]:
                    oracle_result = compute_oracle_iou(all_masks_mo[obj_k], gt_mask)
                    oracle_iou = oracle_result['oracle_iou']
                results['oracle_iou'].append(oracle_iou)

                results['pixel_acc'].append(metrics['pixel_acc'])
                results['recall'].append(metrics['recall'])
                results['precision'].append(metrics['precision'])
                results['f1'].append(metrics['f1'])
                results['tp'].append(metrics['tp'])
                results['fp'].append(metrics['fp'])
                results['fn'].append(metrics['fn'])
                results['tn'].append(metrics['tn'])
                category_metrics[cat_label]['iou'].append(metrics['iou'])
                category_metrics[cat_label]['oracle_iou'].append(oracle_iou)
                category_metrics[cat_label]['pixel_acc'].append(metrics['pixel_acc'])
                category_metrics[cat_label]['recall'].append(metrics['recall'])

                # Spatial postprocess: also record this mask's IoU under geometric spatial labels
                if obj_k in spatial_postprocess_labels:
                    for sp_cat in spatial_postprocess_labels[obj_k]:
                        sp_cat_pp = f"pp_{sp_cat}"  # prefix with pp_ to distinguish from model-predicted spatial
                        category_metrics[sp_cat_pp]['iou'].append(metrics['iou'])
                        category_metrics[sp_cat_pp]['oracle_iou'].append(oracle_iou)
                        category_metrics[sp_cat_pp]['recall'].append(metrics['recall'])
        else:
            for qi, ki in matched_pairs_hungarian:
                obj_k = valid_objects[ki]
                cat_label = object_list[obj_k][3]  # cat_label: spatial prompt for spatial items, base label otherwise
                pred_mask = all_masks[qi]
                gt_mask = gt_stack[ki]

                metrics = compute_metrics(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
                results['iou'].append(metrics['iou'])

                # Oracle: find best of Q masks for this GT object
                oracle_result = compute_oracle_iou(all_masks.unsqueeze(0), gt_mask)
                oracle_iou = oracle_result['oracle_iou']
                results['oracle_iou'].append(oracle_iou)

                results['pixel_acc'].append(metrics['pixel_acc'])
                results['recall'].append(metrics['recall'])
                results['precision'].append(metrics['precision'])
                results['f1'].append(metrics['f1'])
                results['tp'].append(metrics['tp'])
                results['fp'].append(metrics['fp'])
                results['fn'].append(metrics['fn'])
                results['tn'].append(metrics['tn'])
                category_metrics[cat_label]['iou'].append(metrics['iou'])
                category_metrics[cat_label]['oracle_iou'].append(oracle_iou)
                category_metrics[cat_label]['pixel_acc'].append(metrics['pixel_acc'])
                category_metrics[cat_label]['recall'].append(metrics['recall'])

        # Collect viz data: group all objects per frame for multi-object overlay
        if save_viz and len(viz_data) < viz_samples:
            frame_objects = []
            img_h, img_w = img_np.shape[:2]  # image_size resolution
            if sam3_mo and sam3_mo_K is not None:
                for obj_k, pred_k in matched_pairs:
                    # Resize pred mask to image resolution
                    pred_t = torch.sigmoid(pred_source[pred_k]).unsqueeze(0).unsqueeze(0).float()
                    pred_resized = F.interpolate(pred_t, size=(img_h, img_w), mode='bilinear', align_corners=False)
                    pred_np = (pred_resized.squeeze().cpu().numpy() > 0.5).astype(np.float32)
                    # Resize GT mask to image resolution
                    gt_raw = gt_per_object[obj_k]
                    if gt_raw.shape[:2] != (img_h, img_w):
                        gt_t = torch.from_numpy(gt_raw).unsqueeze(0).unsqueeze(0).float()
                        gt_resized = F.interpolate(gt_t, size=(img_h, img_w), mode='nearest')
                        gt_np_viz = gt_resized.squeeze().numpy()
                    else:
                        gt_np_viz = gt_raw
                    frame_objects.append({
                        'label': all_labels[obj_k],
                        'gt_mask': gt_np_viz,
                        'pred_mask': pred_np,
                        'iou': results['iou'][-1] if results['iou'] else 0,
                    })
            if frame_objects:
                viz_data.append({
                    'frame_name': frame_name,
                    'image': img_np,
                    'objects': frame_objects,
                })

    # Aggregate results (same format as single-object eval)
    if not results['iou']:
        return {'error': 'no valid predictions'}

    total_tp = sum(results['tp'])
    total_fp = sum(results['fp'])
    total_fn = sum(results['fn'])
    total_tn = sum(results['tn'])
    total_pixels = total_tp + total_fp + total_fn + total_tn

    scene_result = {
        'scene_id': scene_path.name,
        'miou': np.mean(results['iou']),
        'mean_iou': np.mean(results['iou']),
        'sample_iou': np.mean(results['iou']),
        'oracle_miou': np.mean(results['oracle_iou']),
        'oracle_iou': np.mean(results['oracle_iou']),
        'pixel_acc': np.mean(results['pixel_acc']),
        'recall': np.mean(results['recall']),
        'precision': np.mean(results['precision']),
        'f1': np.mean(results['f1']),
        'num_samples': len(results['iou']),
        'avg_preprocess_ms': np.mean(preprocess_times) * 1000 if preprocess_times else 0,
        'avg_inference_ms': np.mean(inference_times) * 1000 if inference_times else 0,
        'multi_object_eval': True,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_tn': total_tn,
        'global_pixel_acc': (total_tp + total_tn) / max(total_pixels, 1),
    }

    # Per-category metrics
    scene_result['per_category_iou'] = {cat: np.mean(m['iou']) for cat, m in category_metrics.items()}
    scene_result['per_category_oracle_iou'] = {cat: np.mean(m['oracle_iou']) for cat, m in category_metrics.items()}
    scene_result['per_category_pixel_acc'] = {cat: np.mean(m['pixel_acc']) for cat, m in category_metrics.items()}
    scene_result['per_category_recall'] = {cat: np.mean(m['recall']) for cat, m in category_metrics.items()}

    # Spatial eval metrics: count items whose cat_label starts with a spatial qualifier
    spatial_qualifiers = {'nearest', 'farthest', 'leftmost', 'rightmost'}
    spatial_cats = {cat for cat in category_metrics if cat.split()[0].lower() in spatial_qualifiers}
    scene_result['spatial_num_queries'] = sum(len(category_metrics[c]['iou']) for c in spatial_cats)
    if spatial_cats:
        scene_result['spatial_miou'] = np.mean([np.mean(category_metrics[c]['iou']) for c in spatial_cats])
    else:
        scene_result['spatial_miou'] = None

    # Cross-view consistency (mean IoU of same object across frames)
    scene_result['consistency_iou'] = 0.0

    # Save multi-object visualization: all objects overlaid per frame
    if save_viz and viz_data and viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        frame_groups = {v['frame_name']: v for v in viz_data}
        fig = create_multi_object_viz(frame_groups, scene_path.name, max_frames=viz_samples)
        if fig is not None:
            fig.savefig(viz_dir / f"{scene_path.name}_multi_obj.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

    return scene_result


def _precompute_backbone(
    model: TrianguLangModel,
    images: List[Path],
    image_size: Tuple[int, int],
    device: str,
) -> Dict:
    # Pre-compute SAM3 backbone (ViT) features for all frames in batches.
    # The backbone is text-independent and the most expensive part (~37ms/frame).
    # By batching 4 frames and caching, we avoid re-running the ViT for every
    # objectxframe combination. E.g., 5 objects x 100 frames = 500 -> 100 backbone calls.
    # Text encoder + SAM3 encoder still run per-object (text-conditioned).
    precomputed_backbone = {}  # frame_stem -> dict of backbone output tensors (on CPU)
    _precompute_batch_size = 4  # 4 frames at 1008 res fits comfortably on A100 80GB
    _all_frame_stems = [img_path.stem for img_path in images]
    _all_frame_imgs = []

    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(image_size, Image.BILINEAR)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        _all_frame_imgs.append(img_tensor)

    with torch.no_grad():
        for batch_start in range(0, len(_all_frame_imgs), _precompute_batch_size):
            batch_end = min(batch_start + _precompute_batch_size, len(_all_frame_imgs))
            batch_imgs = torch.stack(_all_frame_imgs[batch_start:batch_end]).to(device)

            if batch_imgs.shape[-2:] != (model.resolution, model.resolution):
                batch_imgs = F.interpolate(batch_imgs, size=(model.resolution, model.resolution),
                                           mode='bilinear', align_corners=False)

            with autocast('cuda', dtype=torch.float16):
                bb_out = model.sam3.backbone.forward_image(batch_imgs)

            # Slice per frame -- keep on GPU (plenty of VRAM, avoids CPU↔GPU transfer)
            for j in range(batch_end - batch_start):
                frame_stem = _all_frame_stems[batch_start + j]
                frame_data = {
                    'backbone_fpn': [f[j:j+1] for f in bb_out['backbone_fpn']],
                    'vision_features': bb_out['vision_features'][j:j+1],
                    'vision_pos_enc': [p[j:j+1] for p in bb_out['vision_pos_enc']],
                }
                sam2_out = bb_out.get('sam2_backbone_out')
                if sam2_out is not None:
                    frame_data['sam2_backbone_out'] = {
                        'vision_features': sam2_out['vision_features'][j:j+1],
                        'vision_pos_enc': [p[j:j+1] for p in sam2_out['vision_pos_enc']],
                        'backbone_fpn': [f[j:j+1] for f in sam2_out['backbone_fpn']],
                    }
                else:
                    frame_data['sam2_backbone_out'] = None
                precomputed_backbone[frame_stem] = frame_data

    del _all_frame_imgs  # Free memory
    return precomputed_backbone

@torch.no_grad()
def evaluate_scene(
    model: TrianguLangModel,
    scene_path: Path,
    semantics_dir: Path,
    device: str = 'cuda',
    num_frames: int = 100,
    objects_per_scene: int = 5,
    min_pixel_fraction: float = 0.001,
    image_size: Tuple[int, int] = (1008, 1008),
    save_viz: bool = False,
    viz_dir: Optional[Path] = None,
    viz_samples: int = 10,
    prompt_type: str = 'text_only',
    num_pos_points: int = 10,
    num_neg_points: int = 2,
    sparse_prompts: bool = True,
    num_prompted_frames: int = 3,
    output_localization: bool = False,
    output_depth: bool = False,
    prompt_augmentor: Optional[PromptAugmentor] = None,
    semantic_union: bool = False,
    da3_cache_dir: Optional[Path] = None,
    # Procrustes evaluation parameters
    procrustes: bool = False,
    procrustes_with_scale: bool = True,
    gt_centroids_cache: Optional[Dict] = None,
    data_root: Optional[Path] = None,
    # Category filtering
    allowed_categories: Optional[set] = None,
    # Spatial query filtering: maps prompt -> (qualifier, base_prompt)
    spatial_query_map: Optional[Dict[str, Tuple[Optional[str], str]]] = None,
    # Automatic spatial evaluation for multi-instance labels
    spatial_eval: bool = False,
    # Paper visualization collector
    paper_viz_collector: Optional[List] = None,
    # Frame selection
    frame_names: Optional[List[str]] = None,
    eval_sampling: str = 'stratified',
    # Multi-object eval: batch all objects per frame in one forward pass
    multi_object_eval: bool = False,
    # Temporal EMA smoothing: blend mask logits across consecutive frames
    temporal_smooth_alpha: float = 0.0,
    # CRF post-processing for mask boundary refinement
    use_crf: bool = False,
) -> Dict:
    """Evaluate model on a single scene.

    Args:
        prompt_type: One of 'text_only', 'text_box', 'text_point', 'text_box_point', 'all'
        num_pos_points: Total positive points (distributed across prompted frames in sparse mode)
        num_neg_points: Total negative points (distributed across prompted frames in sparse mode)
        sparse_prompts: If True, distribute points across num_prompted_frames (MV-SAM protocol).
                       If False, give all points to every frame (dense prompting).
        num_prompted_frames: Number of frames to receive prompts in sparse mode.
        output_localization: If True, compute 3D localization for each prediction.
        output_depth: If True, save depth maps for each prediction.
        semantic_union: If True, merge all instances of same label into one GT mask.
                       This matches training behavior when --semantic-union is used.
        da3_cache_dir: Path to DA3 cache directory (da3_cache or da3_nested_cache).
        spatial_query_map: Dict mapping original prompt -> (qualifier, base_prompt) for
                          spatial queries like "leftmost towel" -> ("leftmost", "towel").
                      If provided, loads cached depth instead of running DA3 live.
        paper_viz_collector: If not None, append paper viz data dicts here for grid generation.
    """


    images, objects, available_frames = load_scene_data(scene_path, semantics_dir)
    available_frames_set = set(available_frames)

    if len(images) == 0:
        return {'error': 'No images found'}

    images = [img for img in images if img.name in available_frames_set]

    if len(images) == 0:
        return {'error': 'No images with GT masks'}

    # Filter to specific frames if requested
    if frame_names is not None:
        frame_names_set = set(frame_names)
        frame_names_set.update(f + '.JPG' for f in frame_names if not f.endswith('.JPG'))
        frame_names_set.update(f.replace('.JPG', '') for f in frame_names if f.endswith('.JPG'))
        images = [img for img in images if img.name in frame_names_set or img.stem in frame_names_set]
        if len(images) == 0:
            return {'error': f'No matching frames found for: {frame_names}'}
    elif len(images) > num_frames:
        if eval_sampling == 'sequential':
            images = images[:num_frames]
        elif eval_sampling == 'random':
            rng = np.random.default_rng(42)
            indices = rng.choice(len(images), size=num_frames, replace=False)
            images = [images[i] for i in sorted(indices)]
        elif eval_sampling == 'overlap':
            start_idx = max(0, (len(images) - num_frames) // 2)
            images = images[start_idx:start_idx + num_frames]
        else:  # stratified (default)
            indices = np.linspace(0, len(images) - 1, num_frames, dtype=int)
            images = [images[i] for i in indices]

    skip_labels = SCANNETPP_SKIP_LABELS
    valid_objects = [(obj_id, obj) for obj_id, obj in objects.items()
                     if obj['label'] and obj['label'].lower() not in skip_labels]

    # Filter by allowed categories if specified (supports substring matching)
    if allowed_categories is not None:
        def matches_allowed(label):
            label_lower = label.lower()
            for allowed in allowed_categories:
                if allowed == label_lower or allowed in label_lower or label_lower in allowed:
                    return True
            return False
        valid_objects = [(obj_id, obj) for obj_id, obj in valid_objects
                         if matches_allowed(obj['label'])]

    if len(valid_objects) == 0:
        return {'error': 'No valid objects'}

    # Pre-filter to objects visible in the sampled frames
    if len(images) > 0:
        check_indices = sorted(set(min(i, len(images) - 1)
                                   for i in [0, len(images) // 2, len(images) - 1]))
        visible_obj_ids = set()
        for idx in check_indices:
            frame_masks = load_gt_masks(semantics_dir, images[idx].name)
            for obj_id in frame_masks:
                if frame_masks[obj_id].sum() / frame_masks[obj_id].size >= min_pixel_fraction:
                    visible_obj_ids.add(obj_id)
        visible_objects = [(oid, od) for oid, od in valid_objects if oid in visible_obj_ids]
        if visible_objects:
            valid_objects = visible_objects

    # Filter to spatially-qualified instance (if spatial_query_map provided)
    valid_objects = _prepare_spatial_queries(
        valid_objects, spatial_query_map, da3_cache_dir, scene_path, semantics_dir, images)

    # Build eval_items (label, obj_ids) list
    scene_rng = random.Random(int.from_bytes(scene_path.name.encode(), 'big') % (2**31))
    if semantic_union:
        label_to_obj_ids = defaultdict(list)
        for obj_id, obj_data in valid_objects:
            label_to_obj_ids[obj_data['label']].append(obj_id)
        eval_items = list(label_to_obj_ids.items())
        scene_rng.shuffle(eval_items)
        eval_items = eval_items[:objects_per_scene]
    else:
        scene_rng.shuffle(valid_objects)
        eval_items = [(obj_data['label'], [obj_id])
                      for obj_id, obj_data in valid_objects[:objects_per_scene]]

    # Auto-generate spatial items for multi-instance labels
    spatial_eval_items = _prepare_multi_instance_eval(
        valid_objects, spatial_eval, da3_cache_dir, scene_path, semantics_dir, images)

    # Normalise eval_items to 3-tuples: (label, obj_ids, spatial_prompt_override)
    eval_items = [(label, oids, None) for label, oids in eval_items] + spatial_eval_items

    _precomputed_backbone = _precompute_backbone(model, images, image_size, device)

    if multi_object_eval:
        return _evaluate_scene_multi_object(
            model=model, scene_path=scene_path, semantics_dir=semantics_dir,
            device=device, images=images, image_size=image_size,
            eval_items=eval_items, min_pixel_fraction=min_pixel_fraction,
            da3_cache_dir=da3_cache_dir,
            precomputed_backbone=_precomputed_backbone,
            save_viz=save_viz, viz_dir=viz_dir, viz_samples=viz_samples,
            temporal_smooth_alpha=temporal_smooth_alpha,
            use_crf=use_crf,
        )

    results = defaultdict(list)
    category_metrics = defaultdict(lambda: {'iou': [], 'oracle_iou': [], 'pixel_acc': [],
                                             'recall': [], 'precision': [], 'f1': []})
    preprocess_times = []
    inference_times = []
    viz_data = []
    consistency_ious = []
    raw_centroids_per_object = defaultdict(list)

    # Load GT poses and DA3 extrinsics for Procrustes alignment
    scene_id = scene_path.name
    gt_poses = None
    scene_da3_extrinsics = {}
    procrustes_alignment = None

    if procrustes and gt_centroids_cache and data_root and scene_id in gt_centroids_cache:
        gt_poses = load_gt_poses_for_scene(data_root, scene_id)
        if gt_poses is None:
            print(f"    [Procrustes] WARNING: GT poses not found for {scene_id}")

        if da3_cache_dir is not None:
            scene_cache_dir = da3_cache_dir / scene_id
            if scene_cache_dir.exists():
                pt_files = sorted(scene_cache_dir.glob('*.pt'))
                max_procrustes_frames = 300
                if len(pt_files) > max_procrustes_frames:
                    import random as _rng
                    _rng_state = _rng.getstate()
                    _rng.seed(42)
                    pt_files = sorted(_rng.sample(pt_files, max_procrustes_frames))
                    _rng.setstate(_rng_state)
                for pt_file in pt_files:
                    try:
                        cache_data = torch.load(pt_file, map_location='cpu', weights_only=True)
                        if 'extrinsics' in cache_data:
                            scene_da3_extrinsics[pt_file.stem] = cache_data['extrinsics'].numpy()
                    except Exception:
                        pass
            else:
                print(f"    [Procrustes] WARNING: scene cache dir not found: {scene_cache_dir}")
        else:
            print(f"    [Procrustes] WARNING: da3_cache_dir is None")

        if gt_poses and scene_da3_extrinsics:
            gt_points = []
            da3_points = []
            ORIENT_OFFSET = 0.5
            for frame_name in scene_da3_extrinsics.keys():
                if frame_name in gt_poses:
                    gt_c2w = gt_poses[frame_name]
                    da3_c2w = scene_da3_extrinsics[frame_name]
                    gt_pos = gt_c2w[:3, 3]
                    da3_pos = da3_c2w[:3, 3]
                    gt_points.append(gt_pos)
                    da3_points.append(da3_pos)
                    gt_fwd = -gt_c2w[:3, 2]
                    da3_fwd = -da3_c2w[:3, 2]
                    gt_points.append(gt_pos + ORIENT_OFFSET * gt_fwd)
                    da3_points.append(da3_pos + ORIENT_OFFSET * da3_fwd)

            n_frames = len(gt_points) // 2
            image_stems = {img.stem for img in images}
            gt_pose_frames = set(gt_poses.keys())
            da3_frames = set(scene_da3_extrinsics.keys())
            in_gt = len(image_stems & gt_pose_frames)
            in_da3 = len(image_stems & da3_frames)
            in_both = len(image_stems & gt_pose_frames & da3_frames)
            print(f"    [Procrustes] eval_images={len(images)}, in_gt_poses={in_gt}, "
                  f"in_da3_cache={in_da3}, in_both={in_both}, "
                  f"frames_for_alignment={n_frames} (pts={len(gt_points)})")

            if n_frames >= 3:
                try:
                    R, t, s = umeyama_alignment(
                        np.array(da3_points),
                        np.array(gt_points),
                        with_scale=procrustes_with_scale,
                    )
                    procrustes_alignment = (R, t, s)
                    if n_frames >= 1:
                        da3_fwd0 = da3_points[1] - da3_points[0]
                        transformed_fwd0 = s * R @ da3_fwd0
                        gt_fwd0 = gt_points[1] - gt_points[0]
                        cos_sim = (np.dot(transformed_fwd0, gt_fwd0)
                                   / (np.linalg.norm(transformed_fwd0) * np.linalg.norm(gt_fwd0) + 1e-8))
                        print(f"    [Procrustes] Orientation check: "
                              f"cos_sim(DA3_fwd_aligned, GT_fwd)={cos_sim:.4f}")
                except Exception as e:
                    print(f"    [Procrustes] Alignment failed: {e}")

    # Sparse prompting setup
    use_sparse = sparse_prompts and prompt_type in ['text_point', 'text_box_point', 'all']
    points_per_prompted_frame = {}
    if use_sparse:
        n_prompted = min(num_prompted_frames, len(images))
        prompted_indices = set(np.linspace(0, len(images) - 1, n_prompted, dtype=int).tolist())
        total_pos = num_pos_points
        total_neg = num_neg_points
        pos_per_frame = [total_pos // n_prompted] * n_prompted
        neg_per_frame = [total_neg // n_prompted] * n_prompted
        for i in range(total_pos % n_prompted):
            pos_per_frame[i] += 1
        for i in range(total_neg % n_prompted):
            neg_per_frame[i] += 1
        prompted_frame_list = sorted(prompted_indices)
        points_per_prompted_frame = {
            idx: (pos_per_frame[i], neg_per_frame[i])
            for i, idx in enumerate(prompted_frame_list)
        }
    else:
        prompted_indices = set()

    # Per-object evaluation loop
    for label, obj_ids, item_spatial_override in tqdm(
            eval_items, desc=f"Objects in {scene_path.name}", leave=False):

        obj_result = _evaluate_single_object(
            model=model,
            label=label,
            obj_ids=obj_ids,
            images=images,
            semantics_dir=semantics_dir,
            image_size=image_size,
            device=device,
            spatial_prompt_override=item_spatial_override,
            spatial_query_map=spatial_query_map,
            prompt_type=prompt_type,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            use_sparse=use_sparse,
            prompted_indices=prompted_indices,
            points_per_prompted_frame=points_per_prompted_frame,
            da3_cache_dir=da3_cache_dir,
            scene_path=scene_path,
            precomputed_backbone=_precomputed_backbone,
            output_localization=output_localization,
            save_viz=save_viz,
            viz_samples=viz_samples,
            paper_viz_collector=paper_viz_collector,
            min_pixel_fraction=min_pixel_fraction,
            procrustes=procrustes,
            scene_da3_extrinsics=scene_da3_extrinsics,
            prompt_augmentor=prompt_augmentor,
            current_viz_count=len(viz_data),
        )

        # Accumulate results
        cat_label = obj_result['category_label']
        for rec in obj_result['metrics_list']:
            results['iou'].append(rec['iou'])
            results['oracle_iou'].append(rec['oracle_iou'])
            results['pixel_acc'].append(rec['pixel_acc'])
            results['recall'].append(rec['recall'])
            results['precision'].append(rec['precision'])
            results['f1'].append(rec['f1'])
            results['tp'].append(rec['tp'])
            results['fp'].append(rec['fp'])
            results['fn'].append(rec['fn'])
            results['tn'].append(rec['tn'])
            if 'centroid_error' in rec:
                results['centroid_errors'].append(rec['centroid_error'])

            category_metrics[cat_label]['iou'].append(rec['iou'])
            category_metrics[cat_label]['oracle_iou'].append(rec['oracle_iou'])
            category_metrics[cat_label]['pixel_acc'].append(rec['pixel_acc'])
            category_metrics[cat_label]['recall'].append(rec['recall'])
            category_metrics[cat_label]['precision'].append(rec['precision'])
            category_metrics[cat_label]['f1'].append(rec['f1'])

        preprocess_times.extend(obj_result['preprocess_times'])
        inference_times.extend(obj_result['inference_times'])
        viz_data.extend(obj_result['viz_entries'])
        results.setdefault('localizations', []).extend(obj_result['localization_entries'])

        if obj_result['consistency_entry'] is not None:
            consistency_ious.append(obj_result['consistency_entry'])

        for obj_id, cents in obj_result['raw_centroids'].items():
            raw_centroids_per_object[obj_id].extend(cents)

    return _aggregate_scene_results(
        results=results,
        category_metrics=category_metrics,
        preprocess_times=preprocess_times,
        inference_times=inference_times,
        consistency_ious=consistency_ious,
        procrustes=procrustes,
        procrustes_alignment=procrustes_alignment,
        gt_centroids_cache=gt_centroids_cache,
        scene_id=scene_id,
        raw_centroids_per_object=raw_centroids_per_object,
        eval_items=eval_items,
        save_viz=save_viz,
        viz_data=viz_data,
        viz_dir=viz_dir,
        scene_path=scene_path,
    )


# Generic Dataset Evaluation (for uCO3D and other dataset-based benchmarks)

