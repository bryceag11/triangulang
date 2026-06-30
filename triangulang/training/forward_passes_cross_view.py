"""Cross-view forward pass for TrianguLang training.

Concatenates SAM3 memories from all views so GASA can attend across views.
Split out of ``forward_passes`` for cohesion; ``_forward_cross_view`` is
re-exported from ``forward_passes`` to keep its original import path.
"""
import traceback

import triangulang
import torch

logger = triangulang.get_logger(__name__)
import torch.nn.functional as F
from torch.amp import autocast
from triangulang.losses.segmentation import (
    focal_loss, dice_loss, align_loss, contrastive_mask_loss, centroid_loss,
)
from triangulang.utils.metrics import (
    compute_iou, compute_recall, compute_mean_accuracy,
    compute_per_mask_ious, compute_gt_centroid,
)
from triangulang.training.forward_passes_common import (
    connect_aux_heads_to_graph, connect_trainable_params_to_graph, get_norm_scale,
)


def _per_scene_avg_ious(b, N_views, valid_mask, all_masks_flat, gt_flat):
    """Average per-query IoUs over all valid views of scene ``b`` -> [1, Q], or None."""
    scene_ious = []
    for v in range(N_views):
        idx = b * N_views + v
        if valid_mask[idx]:
            scene_ious.append(compute_per_mask_ious(all_masks_flat[idx:idx+1], gt_flat[idx:idx+1]))
    if not scene_ious:
        return None
    return torch.stack(scene_ious, dim=0).mean(dim=0)


def _forward_cross_view(model, base_model, images, gt_masks, prompts, batch, args, device, ddp,
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
    B, N_views = images.shape[:2]

    # World-frame PE needs extrinsics from either GT or DA3.
    if gt_extrinsics is None and cached_da3_extrinsics is None:
        raise ValueError("--cross-view requires gt_extrinsics or da3_extrinsics for world-frame pointmaps")

    try:
        with autocast('cuda'):
            # Call through the DDP wrapper so gradient-sync hooks fire.
            cached_pi3x = batch.get('cached_pi3x_pointmaps')
            if cached_pi3x is not None:
                cached_pi3x = cached_pi3x.to(device, non_blocking=True)
            outputs = model(
                images, prompts, gt_masks.float(),
                gt_extrinsics=gt_extrinsics,
                gt_intrinsics=gt_intrinsics,
                intrinsics_orig_hw=intrinsics_orig_hw,
                cached_depth=cached_depth,
                da3_extrinsics=cached_da3_extrinsics,
                da3_intrinsics=cached_da3_intrinsics,
                cross_view_mode=True,
                cached_pi3x_pointmaps=cached_pi3x,
            )
            pred = outputs['pred_masks']            # [B, N, H, W]
            all_view_masks = outputs['all_masks']   # [B, N, Q, H, W]

            # Flatten to [B*N, ...] for per-view processing.
            pred_flat = pred.view(B * N_views, *pred.shape[2:])
            gt_flat = gt_masks.view(B * N_views, *gt_masks.shape[2:]).float()
            all_masks_flat = all_view_masks.view(B * N_views, *all_view_masks.shape[2:])

            if gt_flat.shape[-2:] != pred_flat.shape[-2:]:
                gt_flat = F.interpolate(gt_flat.unsqueeze(1), size=pred_flat.shape[-2:],
                                       mode='nearest').squeeze(1)

            valid_mask = gt_flat.sum(dim=(-2, -1)) > 0  # [B*N]

            loss = torch.tensor(0.0, device=device, requires_grad=True)
            n_valid = 0

            for i in range(B * N_views):
                view_pred = pred_flat[i:i+1]  # [1, H, W]
                view_gt = gt_flat[i:i+1]      # [1, H, W]

                view_loss = (args.focal_weight * focal_loss(view_pred, view_gt, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                            args.dice_weight * dice_loss(view_pred.unsqueeze(1), view_gt.unsqueeze(1)))

                if not valid_mask[i]:
                    view_loss = view_loss * 0.0
                loss = loss + view_loss

                if valid_mask[i]:
                    batch_iou_tensor = batch_iou_tensor + compute_iou(view_pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                    batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(view_pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                    batch_recall_tensor = batch_recall_tensor + compute_recall(view_pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                    n_valid += 1

                    prompt_idx = i // N_views  # back to original batch index
                    category = prompts[prompt_idx] if prompt_idx < len(prompts) else "unknown"
                    cat_metrics.update(view_pred, view_gt, category)

            # IoU prediction loss: target is the per-scene average IoU over valid views
            # (iou_pred is [B, Q] per-scene, not per-view).
            if n_valid > 0 and args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                for b in range(B):
                    avg_scene_ious = _per_scene_avg_ious(b, N_views, valid_mask, all_masks_flat, gt_flat)
                    if avg_scene_ious is not None:
                        iou_pred_loss = F.mse_loss(outputs['iou_pred'][b:b+1], avg_scene_ious.detach())
                        loss = loss + args.iou_head_weight * iou_pred_loss / B

            # Contrastive loss: best query per scene from average IoU over valid views.
            if n_valid > 0 and args.contrastive_weight > 0:
                for b in range(B):
                    avg_scene_ious = _per_scene_avg_ious(b, N_views, valid_mask, all_masks_flat, gt_flat)
                    if avg_scene_ious is not None:
                        best_idx = avg_scene_ious.argmax(dim=1)
                        if args.contrastive_source == 'logits':
                            scores = outputs['pred_logits'][b:b+1]
                        elif args.contrastive_source == 'iou_pred' and 'iou_pred' in outputs:
                            scores = outputs['iou_pred'][b:b+1]
                        else:
                            scores = None
                        if scores is not None:
                            contrast_loss = contrastive_mask_loss(scores, best_idx, margin=args.contrastive_margin)
                            loss = loss + args.contrastive_weight * contrast_loss / B

            # Align loss: target is per-scene average IoU over valid views.
            if n_valid > 0 and args.align_weight > 0:
                for b in range(B):
                    avg_scene_ious = _per_scene_avg_ious(b, N_views, valid_mask, all_masks_flat, gt_flat)
                    if avg_scene_ious is not None:
                        logits = outputs['pred_logits'][b:b+1]
                        align_l = align_loss(logits, avg_scene_ious,
                                            alpha=args.align_alpha,
                                            gamma=args.align_gamma,
                                            tau=args.align_tau)
                        loss = loss + args.align_weight * align_l / B

            # Per-layer auxiliary align loss (SAM3-style): same IoU targets, applied to
            # each intermediate decoder layer's scores.
            if args.per_layer_align and args.align_weight > 0 and 'aux_queries' in outputs and outputs['aux_queries'] is not None:
                aux_align_weight = args.per_layer_align_weight if args.per_layer_align_weight is not None else args.align_weight
                num_aux_layers = len(outputs['aux_queries'])
                cached_avg_ious = {}
                for b in range(B):
                    avg_scene_ious = _per_scene_avg_ious(b, N_views, valid_mask, all_masks_flat, gt_flat)
                    if avg_scene_ious is not None:
                        cached_avg_ious[b] = avg_scene_ious
                for layer_idx, aux_q in enumerate(outputs['aux_queries']):
                    aux_text_scores = base_model.gasa_decoder.compute_scores_for_queries(aux_q)
                    if aux_text_scores is None:
                        continue
                    for b, avg_ious in cached_avg_ious.items():
                        aux_logits = aux_text_scores[b:b+1]
                        aux_align_l = align_loss(aux_logits, avg_ious,
                                                 alpha=args.align_alpha,
                                                 gamma=args.align_gamma,
                                                 tau=args.align_tau)
                        loss = loss + aux_align_weight * aux_align_l / (B * num_aux_layers)

            # Presence loss: target is 1 if any view in the scene has valid GT.
            if args.presence_weight > 0 and 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                scene_has_object = torch.zeros(B, 1, device=device)
                for b in range(B):
                    if valid_mask[b * N_views:(b + 1) * N_views].any():
                        scene_has_object[b, 0] = 1.0
                if args.presence_focal:
                    presence_loss = focal_loss(outputs['presence_logit'], scene_has_object,
                                               alpha=args.presence_alpha, gamma=args.presence_gamma)
                else:
                    presence_loss = F.binary_cross_entropy_with_logits(
                        outputs['presence_logit'], scene_has_object
                    )
                loss = loss + args.presence_weight * presence_loss

            # Centroid loss: GT centroid per view, averaged for the scene target.
            if n_valid > 0 and args.use_centroid_head and args.centroid_weight > 0 and 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                pointmaps_full = outputs['pointmaps_full']  # [B, N, H_da3, W_da3, 3]
                pointmaps_full_flat = pointmaps_full.view(B * N_views, *pointmaps_full.shape[2:])
                pm_h, pm_w = pointmaps_full_flat.shape[1:3]

                gt_resized = F.interpolate(
                    gt_flat.unsqueeze(1).float(), size=(pm_h, pm_w), mode='nearest'
                ).squeeze(1)
                pred_resized = F.interpolate(
                    pred_flat.unsqueeze(1), size=(pm_h, pm_w),
                    mode='bilinear', align_corners=False
                ).squeeze(1)

                per_query_cents = outputs['per_query_centroids']  # [B, Q, 3]
                best_idx_flat = outputs['best_idx']               # [B*N]

                for b in range(B):
                    gt_cents = []
                    pred_cents = []
                    for v in range(N_views):
                        idx = b * N_views + v
                        if valid_mask[idx]:
                            gt_cents.append(compute_gt_centroid(gt_resized[idx], pointmaps_full_flat[idx]))
                            if args.mask_based_centroid:
                                pred_cents.append(compute_gt_centroid(pred_resized[idx], pointmaps_full_flat[idx]))

                    if len(gt_cents) > 0:
                        avg_gt_cent = torch.stack(gt_cents, dim=0).mean(dim=0)  # [3]
                        if args.mask_based_centroid and len(pred_cents) > 0:
                            selected_cent = torch.stack(pred_cents, dim=0).mean(dim=0)  # [3]
                        else:
                            # Attention-based: centroid from the best query of the first valid view.
                            first_valid_idx = b * N_views + [v for v in range(N_views) if valid_mask[b * N_views + v]][0]
                            selected_cent = per_query_cents[b, best_idx_flat[first_valid_idx]]  # [3]
                        cent_loss = centroid_loss(selected_cent.unsqueeze(0), avg_gt_cent.unsqueeze(0))
                        loss = loss + args.centroid_weight * cent_loss / B

            # Centroid error tracking for Acc@m metrics.
            if n_valid > 0 and (args.use_centroid_head or args.eval_localization) and 'pointmaps_full' in outputs:
                with torch.no_grad():
                    pointmaps_full = outputs['pointmaps_full']
                    pointmaps_full_flat = pointmaps_full.view(B * N_views, *pointmaps_full.shape[2:])
                    pm_h, pm_w = pointmaps_full_flat.shape[1:3]

                    gt_resized = F.interpolate(
                        gt_flat.unsqueeze(1).float(), size=(pm_h, pm_w), mode='nearest'
                    ).squeeze(1)
                    pred_resized = F.interpolate(
                        pred_flat.unsqueeze(1), size=(pm_h, pm_w),
                        mode='bilinear', align_corners=False
                    ).squeeze(1)

                    scale = get_norm_scale(outputs)
                    for i in range(B * N_views):
                        if valid_mask[i]:
                            pred_cent = compute_gt_centroid(pred_resized[i], pointmaps_full_flat[i])
                            gt_cent = compute_gt_centroid(gt_resized[i], pointmaps_full_flat[i])
                            dist_error = torch.norm(pred_cent - gt_cent).item() * scale
                            epoch_centroid_errors.append(dist_error)

            # Keep aux heads + trainable params connected for DDP gradient sync.
            loss = connect_aux_heads_to_graph(loss, outputs)
            loss = connect_trainable_params_to_graph(loss, model, include_query_proj=True)

            if n_valid > 0:
                batch_loss_tensor = batch_loss_tensor + loss.detach()
            accumulated_loss = loss / args.grad_accum
            valid = n_valid

    except Exception as e:
        logger.warning(f"Error in cross-view forward: {e}")
        traceback.print_exc()

    return (accumulated_loss, valid, batch_loss_tensor, batch_iou_tensor, batch_macc_tensor,
            batch_recall_tensor, batch_sheaf_loss_tensor, last_vis_data)
