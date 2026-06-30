"""Batched forward pass for TrianguLang training (all views in one model call).

Public entry points are re-exported here so they keep their original import
paths (triangulang.training.forward_passes):
- ``_forward_cross_view``  -> forward_passes_cross_view
- ``_forward_sequential``  -> forward_passes_seq
- ``_compute_sheaf_loss``  -> forward_passes_seq
"""
import traceback

import triangulang
import torch

logger = triangulang.get_logger(__name__)
import torch.nn.functional as F
from torch.amp import autocast
from scipy.optimize import linear_sum_assignment
from triangulang.losses.segmentation import (
    focal_loss, dice_loss, align_loss, contrastive_mask_loss, centroid_loss,
)
from triangulang.utils.metrics import (
    compute_iou, compute_recall, compute_mean_accuracy,
    compute_per_mask_ious, compute_gt_centroid,
)
from triangulang.utils.matching import hungarian_match, text_greedy_match
from triangulang.utils.geometry import triangulate_centroid
from triangulang.training.forward_passes_common import (
    connect_aux_heads_to_graph, connect_trainable_params_to_graph, smooth_mask_logits,
)

# Re-exports (keep importable from triangulang.training.forward_passes).
from triangulang.training.forward_passes_cross_view import _forward_cross_view
from triangulang.training.forward_passes_seq import _compute_sheaf_loss, _forward_sequential


def _forward_batch_views(model, base_model, images, gt_masks, prompts, batch, args, device, ddp,
                         N_views, B, gt_extrinsics, gt_intrinsics, intrinsics_orig_hw,
                         cached_depth, cached_da3_extrinsics, cached_da3_intrinsics,
                         spatial_qualifier_idx, epoch, start_epoch, batch_idx,
                         cat_metrics, epoch_centroid_errors,
                         batch_iou_tensor, batch_macc_tensor, batch_recall_tensor,
                         batch_sheaf_loss_tensor):
    accumulated_loss = None
    valid = 0
    batch_loss_tensor = torch.tensor(0.0, device=device)
    last_vis_data = None
    # This is much faster than sequential processing
    B, N_views = images.shape[:2]

    # Detect multi-object mode from batch
    multi_object_K = 1
    multi_object_prompts_list = None
    all_gt_multi = None  # [B*N, K, H, W] for multi-object
    num_objects_per_item = None  # [B] actual K per item (may vary if padded)

    if 'num_objects' in batch and batch['num_objects'] is not None:
        num_objects_per_item = batch['num_objects'].to(device)  # [B]
        multi_object_K = int(num_objects_per_item.max().item())

    if multi_object_K > 1:
        # Multi-object: gt_masks is [B, K, N, H, W]
        # Reshape to [B*N, K, H, W] by permuting N and K
        gt_multi_raw = gt_masks  # [B, K, N, H, W]
        all_gt_multi = gt_multi_raw.permute(0, 2, 1, 3, 4).reshape(
            B * N_views, multi_object_K, *gt_multi_raw.shape[3:]
        ).float()  # [B*N, K, H, W]
        # For valid_mask: use ANY object's coverage (not just primary)
        # In scene_grouped mode, the primary object may only be visible in a few views,
        # but other objects ARE visible -> must not zero out their loss
        all_gt = all_gt_multi.max(dim=1).values  # [B*N, H, W] union of all objects
        # Flatten multi-object prompts: each view gets K prompts
        multi_object_prompts_list = batch['multi_object_prompts']  # List[List[str]] [B][K]
    else:
        all_gt = gt_masks.reshape(B * N_views, *gt_masks.shape[2:]).float()

    # Reshape: [B, N, C, H, W] -> [B*N, C, H, W]
    all_views = images.reshape(B * N_views, *images.shape[2:])

    # SAM3-style multi-object mode flag
    sam3_mo = False  # Will be set True after forward if sam3_mo_K in outputs

    # Build prompts for model forward
    if multi_object_K > 1:
        # Multi-object: flatten K*B*N prompts for text encoding
        # For each view, repeat the batch item's K prompts
        # all_prompts_flat = K*B*N strings, ordered as:
        #   [b0_text0, b0_text1, ..., b0_textK-1, b1_text0, ..., bB-1_textK-1] * N_views
        flat_prompts_per_batch = []
        for b_idx in range(B):
            for k in range(multi_object_K):
                flat_prompts_per_batch.append(multi_object_prompts_list[b_idx][k])
        all_prompts = flat_prompts_per_batch * N_views  # Repeat for each view
    else:
        # Single-object: repeat prompts for each view
        all_prompts = prompts * N_views  # Repeat N times

    # Check which views have valid GT masks (non-empty with sufficient coverage)
    if 'gt_mask_coverage' in batch:
        # Coverage computed at original resolution (e.g., 1752x1168)
        mask_coverage = batch['gt_mask_coverage'].to(device).reshape(B * N_views)  # [B*N]
        if args.min_mask_coverage > 0:
            valid_mask = mask_coverage >= args.min_mask_coverage
        else:
            mask_pixels = all_gt.sum(dim=(-2, -1))
            valid_mask = mask_pixels > 0
    else:
        # Fallback: compute on resized mask (less accurate for small objects)
        mask_pixels = all_gt.sum(dim=(-2, -1))  # [B*N]
        mask_coverage = mask_pixels / all_gt[0].numel()  # fraction
        if args.min_mask_coverage > 0:
            valid_mask = mask_coverage >= args.min_mask_coverage
        else:
            valid_mask = mask_pixels > 0
    # NOTE: Don't use 'continue' even if all GTs empty - backward() must be
    # called on ALL ranks for DDP gradient sync. Skipping causes deadlock!
    # The loss will be zero but all ranks must participate in all_reduce.

    # Get extrinsics/intrinsics for all views
    all_extrinsics = gt_extrinsics.reshape(B * N_views, 4, 4) if gt_extrinsics is not None else None
    all_intrinsics = gt_intrinsics.reshape(B * N_views, 3, 3) if gt_intrinsics is not None else None

    # Reshape cached depth for all views
    all_cached_depth = cached_depth.reshape(B * N_views, *cached_depth.shape[2:]) if cached_depth is not None else None

    # Reshape cached DA3-NESTED poses for all views (for world-frame GASA)
    all_da3_extrinsics = cached_da3_extrinsics.reshape(B * N_views, 4, 4) if cached_da3_extrinsics is not None else None
    all_da3_intrinsics = cached_da3_intrinsics.reshape(B * N_views, 3, 3) if cached_da3_intrinsics is not None else None

    # Repeat spatial qualifiers if used
    all_spatial_idx = None
    if spatial_qualifier_idx is not None:
        all_spatial_idx = spatial_qualifier_idx.repeat(N_views)  # [B] -> [B*N]


    try:
        with autocast('cuda'):
            # For per-text decode, pass multi-object GT [B*N, K, H, W] so each
            # text gets oracle mask selection against its own GT
            fwd_gt = all_gt_multi if ((args.per_text_decode or getattr(args, 'sam3_multi_object', False)) and all_gt_multi is not None) else all_gt
            # Load PI3X cached pointmaps if available
            cached_pi3x = batch.get('cached_pi3x_pointmaps')
            if cached_pi3x is not None:
                cached_pi3x = cached_pi3x.reshape(B * N_views, *cached_pi3x.shape[2:]).to(device, non_blocking=True)
            outputs = model(all_views, all_prompts, fwd_gt,
                          gt_extrinsics=all_extrinsics,
                          gt_intrinsics=all_intrinsics,
                          spatial_qualifier_idx=all_spatial_idx,
                          intrinsics_orig_hw=intrinsics_orig_hw,
                          cached_depth=all_cached_depth,
                          da3_extrinsics=all_da3_extrinsics,
                          da3_intrinsics=all_da3_intrinsics,
                          num_texts=multi_object_K,
                          cached_pi3x_pointmaps=cached_pi3x)

            # SAM3-MO: outputs are [B*N*K, ...], reshape GT and valid_mask
            if outputs.get('sam3_mo_K') is not None:
                sam3_K = outputs['sam3_mo_K']
                # Reshape GT: [B*N, K, H, W] -> [B*N*K, H, W]
                all_gt = all_gt_multi.reshape(-1, *all_gt_multi.shape[2:])
                # Per-object valid mask
                valid_mask = (all_gt.sum(dim=(-2, -1)) > 0)
                # Override to single-object loss path
                multi_object_K = 1
                sam3_mo = True
                # Build per-item prompts for category tracking
                sam3_mo_prompts = []
                for v_idx in range(N_views):
                    for b_idx in range(B):
                        K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else sam3_K
                        for k in range(sam3_K):
                            if k < K_i and multi_object_prompts_list is not None:
                                sam3_mo_prompts.append(multi_object_prompts_list[b_idx][k])
                            else:
                                sam3_mo_prompts.append("padding")

            # Compute loss for ALL views (but multiply by 0 for invalid ones to keep graph connected)
            # This ensures all trainable params are used for DDP gradient sync
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            n_valid = 0

            if multi_object_K > 1 and 'per_text_masks' in outputs:
                per_text_masks = outputs['per_text_masks']  # [B*N, K, H, W]
                grad_text_indices = outputs.get('grad_text_indices', list(range(multi_object_K)))
                if per_text_masks.shape[-2:] != all_gt_multi.shape[-2:]:
                    per_text_masks = F.interpolate(per_text_masks, size=all_gt_multi.shape[-2:],
                                                  mode='bilinear', align_corners=False)

                for i in range(B * N_views):
                    b_idx = i % B
                    K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                    view_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    n_k = 0
                    for k_idx in range(K_i):
                        gt_k = all_gt_multi[i, k_idx:k_idx+1]  # [1, H, W]
                        if gt_k.sum() > 0:
                            pred_k = per_text_masks[i, k_idx:k_idx+1]  # [1, H, W]
                            # Only compute loss for texts with gradients
                            if k_idx in grad_text_indices:
                                pair_loss = (
                                    args.focal_weight * focal_loss(pred_k, gt_k, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                    args.dice_weight * dice_loss(pred_k.unsqueeze(1), gt_k.unsqueeze(1))
                                )
                                view_loss = view_loss + pair_loss
                                n_k += 1

                            # Track metrics for ALL texts (no gradients needed)
                            with torch.no_grad():
                                batch_iou_tensor = batch_iou_tensor + compute_iou(pred_k.unsqueeze(1), gt_k.unsqueeze(1), return_tensor=True)
                                batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(pred_k.unsqueeze(1), gt_k.unsqueeze(1), return_tensor=True)
                                batch_recall_tensor = batch_recall_tensor + compute_recall(pred_k.unsqueeze(1), gt_k.unsqueeze(1), return_tensor=True)
                            n_valid += 1
                            mo_prompts = multi_object_prompts_list[b_idx] if multi_object_prompts_list else None
                            category = mo_prompts[k_idx] if mo_prompts and k_idx < len(mo_prompts) else "unknown"
                            cat_metrics.update(pred_k.detach(), gt_k, category)

                    if n_k > 0:
                        view_loss = view_loss / n_k
                    if not valid_mask[i]:
                        view_loss = view_loss * 0.0
                    loss = loss + view_loss

            elif multi_object_K > 1:
                all_masks = outputs['all_masks']  # [B*N, Q, H, W]
                text_scores_multi = outputs.get('text_scores', None)  # [B*N, Q, K] or None

                # Resize all_masks to match GT if needed
                if all_masks.shape[-2:] != all_gt_multi.shape[-2:]:
                    all_masks_resized = F.interpolate(
                        all_masks, size=all_gt_multi.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                else:
                    all_masks_resized = all_masks

                # Pre-compute matching ONCE per batch item (consistent across views)
                batch_matched_pairs = {}  # b_idx -> (matched_pairs, unmatched)
                for b_idx in range(B):
                    K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K

                    if args.match_strategy == 'text_greedy' and text_scores_multi is not None and text_scores_multi.dim() == 3:
                        # Text-greedy: stable assignment based on text scoring head
                        first_valid = next((v_idx * B + b_idx for v_idx in range(N_views) if valid_mask[v_idx * B + b_idx]), 0)
                        ts = text_scores_multi[first_valid, :, :K_i]
                        matched, unmatched = text_greedy_match(ts, K_i)
                        batch_matched_pairs[b_idx] = (matched, unmatched)
                    else:
                        # Hungarian: IoU-based bipartite matching averaged across views
                        avg_cost = torch.zeros(all_masks_resized.shape[1], K_i, device=device)
                        n_views_for_match = 0
                        for v_idx in range(N_views):
                            i = v_idx * B + b_idx
                            if valid_mask[i]:
                                view_masks = all_masks_resized[i]
                                view_gt = all_gt_multi[i, :K_i]
                                pred_binary = (torch.sigmoid(view_masks) > 0.5).float()
                                for k in range(K_i):
                                    gt_k = (view_gt[k] > 0.5).float()
                                    inter = (pred_binary * gt_k.unsqueeze(0)).sum(dim=(-2, -1))
                                    union = pred_binary.sum(dim=(-2, -1)) + gt_k.sum() - inter
                                    avg_cost[:, k] += -(inter / union.clamp(min=1.0))
                                n_views_for_match += 1
                        if n_views_for_match > 0:
                            avg_cost /= n_views_for_match
                        if text_scores_multi is not None and text_scores_multi.dim() == 3:
                            first_valid = next((v_idx * B + b_idx for v_idx in range(N_views) if valid_mask[v_idx * B + b_idx]), 0)
                            ts = text_scores_multi[first_valid, :, :K_i]
                            avg_cost = avg_cost + 0.5 * (-ts.sigmoid())
                        row_ind, col_ind = linear_sum_assignment(avg_cost.detach().cpu().numpy())
                        matched = list(zip(row_ind.tolist(), col_ind.tolist()))
                        unmatched = [q for q in range(all_masks_resized.shape[1]) if q not in set(row_ind.tolist())]
                        batch_matched_pairs[b_idx] = (matched, unmatched)

                for i in range(B * N_views):
                    b_idx = i % B
                    K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                    view_gt_k = all_gt_multi[i, :K_i]  # [K_i, H, W]

                    # Use pre-computed consistent matching
                    matched_pairs, unmatched = batch_matched_pairs[b_idx]

                    # Per-matched-pair loss
                    view_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    n_matched = 0
                    for q_idx, k_idx in matched_pairs:
                        if k_idx < K_i and view_gt_k[k_idx].sum() > 0:
                            pred_k = all_masks_resized[i, q_idx:q_idx+1]  # [1, H, W]
                            gt_k = view_gt_k[k_idx:k_idx+1]  # [1, H, W]
                            pair_loss = (
                                args.focal_weight * focal_loss(pred_k, gt_k, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                args.dice_weight * dice_loss(pred_k.unsqueeze(1), gt_k.unsqueeze(1))
                            )
                            view_loss = view_loss + pair_loss
                            n_matched += 1

                    if n_matched > 0:
                        view_loss = view_loss / n_matched

                    # No-object loss: force unmatched queries to predict empty masks
                    if args.no_object_weight > 0 and len(unmatched) > 0:
                        empty_gt = torch.zeros(1, all_masks_resized.shape[-2], all_masks_resized.shape[-1],
                                              device=device)
                        no_obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        for q_idx in unmatched:
                            pred_q = all_masks_resized[i, q_idx:q_idx+1]
                            # Sigmoid BCE against empty target (penalize any positive predictions)
                            no_obj_loss = no_obj_loss + F.binary_cross_entropy_with_logits(
                                pred_q, empty_gt, reduction='mean')
                        no_obj_loss = no_obj_loss / len(unmatched)
                        view_loss = view_loss + args.no_object_weight * no_obj_loss

                    # For invalid views, zero out loss but keep graph connected
                    if not valid_mask[i]:
                        view_loss = view_loss * 0.0

                    loss = loss + view_loss

                    # Metrics: track ALL matched objects (not just primary)
                    if valid_mask[i] and matched_pairs:
                        b_idx_metric = i % B
                        mo_prompts = multi_object_prompts_list[b_idx_metric] if multi_object_prompts_list else None
                        for q_idx, k_idx in matched_pairs:
                            if k_idx < K_i and all_gt_multi[i, k_idx].sum() > 0:
                                view_pred = all_masks_resized[i, q_idx:q_idx+1]
                                view_gt_k = all_gt_multi[i, k_idx:k_idx+1]
                                batch_iou_tensor = batch_iou_tensor + compute_iou(view_pred.unsqueeze(1), view_gt_k.unsqueeze(1), return_tensor=True)
                                batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(view_pred.unsqueeze(1), view_gt_k.unsqueeze(1), return_tensor=True)
                                batch_recall_tensor = batch_recall_tensor + compute_recall(view_pred.unsqueeze(1), view_gt_k.unsqueeze(1), return_tensor=True)
                                n_valid += 1
                                category = mo_prompts[k_idx] if mo_prompts and k_idx < len(mo_prompts) else "unknown"
                                cat_metrics.update(view_pred, view_gt_k, category)

                # IoU head loss for multi-object
                if n_valid > 0 and args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs:
                    # Compute IoU targets for ALL queries against their matched GTs
                    # Unmatched queries get target IoU = 0
                    for i in range(B * N_views):
                        if valid_mask[i]:
                            b_idx = i % B
                            K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                            view_masks = all_masks_resized[i]  # [Q, H, W]
                            view_gt_k = all_gt_multi[i, :K_i]
                            view_text_scores = None
                            if text_scores_multi is not None and text_scores_multi.dim() == 3:
                                view_text_scores = text_scores_multi[i, :, :K_i]
                            matched_pairs_iou, _ = hungarian_match(view_masks, view_gt_k, K_i, view_text_scores)
                            iou_targets = torch.zeros(view_masks.shape[0], device=device)
                            for q_idx, k_idx in matched_pairs_iou:
                                if k_idx < K_i:
                                    pred_bin = (torch.sigmoid(view_masks[q_idx]) > 0.5).float()
                                    gt_bin = (view_gt_k[k_idx] > 0.5).float()
                                    inter = (pred_bin * gt_bin).sum()
                                    union = pred_bin.sum() + gt_bin.sum() - inter
                                    iou_targets[q_idx] = inter / union.clamp(min=1.0)
                            iou_pred_loss = F.mse_loss(outputs['iou_pred'][i], iou_targets.detach())
                            loss = loss + args.iou_head_weight * iou_pred_loss / n_valid

                # Align loss for multi-object
                if n_valid > 0 and args.align_weight > 0:
                    for i in range(B * N_views):
                        if valid_mask[i]:
                            b_idx = i % B
                            K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                            view_masks = all_masks_resized[i]  # [Q, H, W]
                            view_gt_k = all_gt_multi[i, :K_i]
                            # Compute IoU of each query against ALL GT objects, take max
                            actual_ious = torch.zeros(1, view_masks.shape[0], device=device)
                            for k in range(K_i):
                                per_mask_ious = compute_per_mask_ious(view_masks.unsqueeze(0), view_gt_k[k:k+1])
                                actual_ious = torch.max(actual_ious, per_mask_ious)
                            logits = outputs['pred_logits'][i:i+1]
                            align_l = align_loss(logits, actual_ious,
                                                alpha=args.align_alpha,
                                                gamma=args.align_gamma,
                                                tau=args.align_tau)
                            loss = loss + args.align_weight * align_l / n_valid

                # Text scoring loss: train text_scores to predict query-text assignment
                # This is essential for text_greedy matching to work
                if n_valid > 0 and text_scores_multi is not None and text_scores_multi.dim() == 3:
                    text_score_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    n_ts_valid = 0
                    for i in range(B * N_views):
                        if valid_mask[i]:
                            b_idx = i % B
                            K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                            matched_pairs, _ = batch_matched_pairs[b_idx]
                            # Target: 1.0 for matched (query, text) pairs, 0.0 for rest
                            ts_target = torch.zeros(all_masks_resized.shape[1], K_i, device=device)
                            for q_idx, k_idx in matched_pairs:
                                if k_idx < K_i and all_gt_multi[i, k_idx].sum() > 0:
                                    ts_target[q_idx, k_idx] = 1.0
                            ts_pred = text_scores_multi[i, :, :K_i]
                            text_score_loss = text_score_loss + F.binary_cross_entropy_with_logits(
                                ts_pred, ts_target, reduction='mean')
                            n_ts_valid += 1
                    if n_ts_valid > 0:
                        loss = loss + text_score_loss / n_ts_valid

            else:
                # For SAM3-MO: batch is expanded to B*N*K, each item = 1 object
                pred = outputs['pred_masks'][:, 0] if outputs['pred_masks'].dim() == 4 else outputs['pred_masks']

                if args.loss_at_native_res:
                    # Downsample GT to native mask resolution (288x288)
                    # Avoids blurring gradients through bilinear upsampling
                    if all_gt.shape[-2:] != pred.shape[-2:]:
                        all_gt_for_loss = F.interpolate(
                            all_gt.unsqueeze(1).float(), size=pred.shape[-2:],
                            mode='nearest'
                        ).squeeze(1)
                    else:
                        all_gt_for_loss = all_gt
                    pred_for_loss = pred
                else:
                    # Original: upsample pred to GT resolution
                    if pred.shape[-2:] != all_gt.shape[-2:]:
                        pred = F.interpolate(pred.unsqueeze(1), size=all_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
                    all_gt_for_loss = all_gt
                    pred_for_loss = pred

                # Mask smoothing (avg pool, matches eval-time LangSplat protocol)
                pred_for_loss = smooth_mask_logits(pred_for_loss, args.mask_smooth_kernel)

                n_items = all_gt_for_loss.shape[0]  # B*N for single-obj, B*N*K for SAM3-MO

                if args.use_point_sampling:
                    # SAM3-style: compute loss on sampled uncertain points
                    # Only on valid views
                    valid_pred = pred_for_loss[valid_mask[:n_items]]
                    valid_gt = all_gt_for_loss[valid_mask[:n_items]]
                    if valid_pred.shape[0] > 0:
                        view_loss = point_sampled_loss(
                            valid_pred, valid_gt,
                            focal_fn=focal_loss, dice_fn=dice_loss,
                            focal_weight=args.focal_weight, dice_weight=args.dice_weight,
                            focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                            num_points=args.num_sample_points,
                        )
                        if args.lovasz_weight > 0:
                            view_loss = view_loss + args.lovasz_weight * lovasz_loss(valid_pred, valid_gt)
                        loss = loss + view_loss

                    # Metrics (use original resolution pred for accurate IoU)
                    for i in range(n_items):
                        if valid_mask[i]:
                            vp = pred_for_loss[i:i+1]
                            vg = all_gt_for_loss[i:i+1]
                            batch_iou_tensor = batch_iou_tensor + compute_iou(vp.unsqueeze(1), vg.unsqueeze(1), return_tensor=True)
                            batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(vp.unsqueeze(1), vg.unsqueeze(1), return_tensor=True)
                            batch_recall_tensor = batch_recall_tensor + compute_recall(vp.unsqueeze(1), vg.unsqueeze(1), return_tensor=True)
                            n_valid += 1
                            if sam3_mo and 'sam3_mo_prompts' in dir():
                                category = sam3_mo_prompts[i] if i < len(sam3_mo_prompts) else "unknown"
                            else:
                                prompt_idx = i % B
                                category = prompts[prompt_idx] if prompt_idx < len(prompts) else "unknown"
                            cat_metrics.update(vp, vg, category)
                else:
                    for i in range(n_items):
                        view_pred = pred_for_loss[i:i+1]
                        view_gt_single = all_gt_for_loss[i:i+1]

                        view_loss = (args.focal_weight * focal_loss(view_pred, view_gt_single, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                    args.dice_weight * dice_loss(view_pred.unsqueeze(1), view_gt_single.unsqueeze(1)))

                        # Lovász loss: directly optimizes IoU for sharper boundaries
                        if args.lovasz_weight > 0:
                            view_loss = view_loss + args.lovasz_weight * lovasz_loss(view_pred, view_gt_single)

                        # For invalid views (empty GT), multiply loss by 0 to zero gradients
                        # but keep graph connected so all trainable params are used
                        if not valid_mask[i]:
                            view_loss = view_loss * 0.0

                        loss = loss + view_loss

                    # Only accumulate metrics for valid views (non-empty GT)
                    if valid_mask[i]:
                        batch_iou_tensor = batch_iou_tensor + compute_iou(view_pred.unsqueeze(1), view_gt_single.unsqueeze(1), return_tensor=True)
                        batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(view_pred.unsqueeze(1), view_gt_single.unsqueeze(1), return_tensor=True)
                        batch_recall_tensor = batch_recall_tensor + compute_recall(view_pred.unsqueeze(1), view_gt_single.unsqueeze(1), return_tensor=True)
                        n_valid += 1

                        # Track per-category metrics
                        if sam3_mo and 'sam3_mo_prompts' in dir():
                            category = sam3_mo_prompts[i] if i < len(sam3_mo_prompts) else "unknown"
                        else:
                            prompt_idx = i % B  # Map back to original batch index
                            category = prompts[prompt_idx] if prompt_idx < len(prompts) else "unknown"
                        cat_metrics.update(view_pred, view_gt_single, category)

            # Pre-compute per-view IoU cache (reused by IoU head, contrastive, align losses)
            _iou_cache = {}
            if multi_object_K == 1 and n_valid > 0:
                _need_ious = (
                    (args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs) or
                    (args.contrastive_weight > 0) or
                    (args.align_weight > 0)
                )
                if _need_ious:
                    all_masks = outputs['all_masks']  # [B*N, Q, H, W] or [B*N*K, Q, H, W]
                    for i in range(n_items):
                        if valid_mask[i]:
                            _iou_cache[i] = compute_per_mask_ious(all_masks[i:i+1], all_gt[i:i+1])

            # IoU prediction loss (only for valid views, single-object only)
            # Multi-object IoU loss is handled inside the multi-object block above
            if multi_object_K == 1 and n_valid > 0 and args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs:
                for i in range(n_items):
                    if i in _iou_cache:
                        iou_pred_loss = F.mse_loss(outputs['iou_pred'][i:i+1], _iou_cache[i].detach())
                        loss = loss + args.iou_head_weight * iou_pred_loss / n_valid

            # Contrastive loss (single-object only)
            if multi_object_K == 1 and n_valid > 0 and args.contrastive_weight > 0:
                for i in range(n_items):
                    if i in _iou_cache:
                        best_idx = _iou_cache[i].argmax(dim=1)
                        if args.contrastive_source == 'logits':
                            scores = outputs['pred_logits'][i:i+1]
                        elif args.contrastive_source == 'iou_pred' and 'iou_pred' in outputs:
                            scores = outputs['iou_pred'][i:i+1]
                        else:
                            scores = None
                        if scores is not None:
                            contrast_loss = contrastive_mask_loss(scores, best_idx, margin=args.contrastive_margin)
                            loss = loss + args.contrastive_weight * contrast_loss / n_valid

            # Text scoring loss: REMOVED. pred_logits now comes from DotProductScoring head,
            # so the existing align loss trains text-query matching end-to-end (SAM3-style).
            # The separate cross-entropy text scoring loss was redundant.

            # Align loss (single-object only; multi-object handled above)
            if multi_object_K == 1 and n_valid > 0 and args.align_weight > 0:
                for i in range(n_items):
                    if i in _iou_cache:
                        logits = outputs['pred_logits'][i:i+1]
                        align_l = align_loss(logits, _iou_cache[i],
                                            alpha=args.align_alpha,
                                            gamma=args.align_gamma,
                                            tau=args.align_tau)
                        loss = loss + args.align_weight * align_l / n_valid

                # PER-LAYER AUXILIARY ALIGN LOSS (single-object per-view path)
                if args.per_layer_align and 'aux_queries' in outputs and outputs['aux_queries'] is not None:
                    aux_align_weight = args.per_layer_align_weight if args.per_layer_align_weight is not None else args.align_weight
                    num_aux_layers = len(outputs['aux_queries'])
                    for aux_q in outputs['aux_queries']:
                        aux_text_scores = base_model.gasa_decoder.compute_scores_for_queries(aux_q)
                        if aux_text_scores is None:
                            continue
                        for i in range(n_items):
                            if i in _iou_cache:
                                aux_logits = aux_text_scores[i:i+1]
                                aux_align_l = align_loss(aux_logits, _iou_cache[i],
                                                         alpha=args.align_alpha,
                                                         gamma=args.align_gamma,
                                                         tau=args.align_tau)
                                loss = loss + aux_align_weight * aux_align_l / (n_valid * num_aux_layers)

            # Presence loss: predict 1.0 when object exists, 0.0 when empty
            # This ALWAYS runs (even for empty views) to train presence detection
            if args.presence_weight > 0 and 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                presence_targets = valid_mask.float().unsqueeze(1)  # [B*N, 1]
                if args.presence_focal:
                    presence_loss = focal_loss(outputs['presence_logit'], presence_targets,
                                               alpha=args.presence_alpha, gamma=args.presence_gamma)
                else:
                    presence_loss = F.binary_cross_entropy_with_logits(
                        outputs['presence_logit'], presence_targets
                    )
                loss = loss + args.presence_weight * presence_loss

            # Centroid loss (batched path) - only for valid views
            if n_valid > 0 and args.use_centroid_head and args.centroid_weight > 0 and 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                pointmaps_full = outputs['pointmaps_full']  # [B*N, H_da3, W_da3, 3]
                pm_h, pm_w = pointmaps_full.shape[1:3]
                # Resize GT masks to match pointmaps resolution
                all_gt_resized = F.interpolate(
                    all_gt.unsqueeze(1).float(),
                    size=(pm_h, pm_w),
                    mode='nearest'
                ).squeeze(1)  # [B*N, H_da3, W_da3]

                per_query_cents = outputs['per_query_centroids']  # [B*N, Q, 3]
                best_idx = outputs['best_idx']  # [B*N]

                # Resize pred masks for mask-based or triangulation centroid
                if args.mask_based_centroid or args.use_triangulation:
                    all_pred_resized = F.interpolate(
                        all_pred.unsqueeze(1),
                        size=(pm_h, pm_w),
                        mode='bilinear', align_corners=False
                    ).squeeze(1)  # [B*N, H_da3, W_da3]

                # TRIANGULATION: Multi-view ray intersection for 3D centroid
                if args.use_triangulation and all_da3_extrinsics is not None and all_da3_intrinsics is not None and N_views > 1 and not sam3_mo:
                    # Reshape to [B, N, ...] for per-scene triangulation
                    pred_resized_bv = all_pred_resized.reshape(B, N_views, pm_h, pm_w)
                    gt_resized_bv = all_gt_resized.reshape(B, N_views, pm_h, pm_w)
                    ext_bv = all_da3_extrinsics.reshape(B, N_views, 4, 4)
                    int_bv = all_da3_intrinsics.reshape(B, N_views, 3, 3)
                    pointmaps_bv = pointmaps_full.reshape(B, N_views, pm_h, pm_w, 3)

                    for b_idx in range(B):
                        # Check if this scene has any valid views
                        scene_valid = valid_mask[b_idx * N_views:(b_idx + 1) * N_views]
                        if scene_valid.sum() < 2:
                            continue  # Need at least 2 views for triangulation

                        # Triangulate predicted centroid
                        pred_tri, pred_valid = triangulate_centroid(
                            pred_resized_bv[b_idx], ext_bv[b_idx], int_bv[b_idx]
                        )

                        # Triangulate GT centroid (for supervision target)
                        gt_tri, gt_valid = triangulate_centroid(
                            gt_resized_bv[b_idx], ext_bv[b_idx], int_bv[b_idx]
                        )

                        if pred_valid and gt_valid:
                            cent_loss = centroid_loss(pred_tri.unsqueeze(0), gt_tri.unsqueeze(0))
                            loss = loss + args.centroid_weight * cent_loss / B
                else:
                    # Original per-view centroid computation
                    for i in range(n_items):
                        if valid_mask[i]:
                            gt_cent = compute_gt_centroid(all_gt_resized[i], pointmaps_full[i])
                            if args.mask_based_centroid:
                                # MASK-BASED: Compute centroid from predicted mask + depth
                                selected_cent = compute_gt_centroid(all_pred_resized[i], pointmaps_full[i])
                            else:
                                # ATTENTION-BASED: Use centroid from selected query
                                selected_cent = per_query_cents[i, best_idx[i]]  # [3]
                            cent_loss = centroid_loss(selected_cent.unsqueeze(0), gt_cent.unsqueeze(0))
                            loss = loss + args.centroid_weight * cent_loss / n_valid

            # Keep aux heads + trainable params connected for DDP gradient sync
            # (0-weighted, so no effect on gradients).
            loss = connect_aux_heads_to_graph(loss, outputs)

            loss = connect_trainable_params_to_graph(loss, model, include_query_proj=True)

            # ALWAYS set accumulated_loss (even if all views invalid) to ensure
            # backward() runs and keeps DDP gradient sync working
            if n_valid > 0:
                batch_loss_tensor = batch_loss_tensor + loss.detach()
            accumulated_loss = loss / args.grad_accum
            valid = n_valid

        # Save visualization data
        if last_vis_data is None and ddp.is_main:
            last_vis_data = {'images': all_views[:B].detach().cpu(), 'gt_masks': all_gt[:B].detach().cpu(),
                             'outputs': {k: v[:B].detach().cpu() if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in outputs.items()},
                             'prompts': prompts}

    except Exception as e:
        logger.warning(f"Error in batched forward: {e}")
        traceback.print_exc()

    return (accumulated_loss, valid, batch_loss_tensor, batch_iou_tensor, batch_macc_tensor,
            batch_recall_tensor, batch_sheaf_loss_tensor, last_vis_data)
