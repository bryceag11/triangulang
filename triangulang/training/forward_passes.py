"""Forward pass helpers for TrianguLang training loop."""
import torch
import torch.nn.functional as F
from torch.amp import autocast
from triangulang.losses.segmentation import (
    focal_loss, dice_loss, align_loss, contrastive_mask_loss,
    centroid_loss, boundary_loss,
)
from triangulang.losses.sheaf_losses import AsymmetricRestrictionSheaf
from triangulang.losses.spatial_losses import spatial_ranking_loss, spatial_selection_loss
from triangulang.utils.metrics import (
    compute_iou, compute_recall, compute_mean_accuracy,
    compute_per_mask_ious, compute_gt_centroid,
)
from triangulang.utils.matching import hungarian_match, text_greedy_match
from triangulang.utils.geometry import triangulate_centroid

from triangulang.training.forward_passes_seq import _compute_sheaf_loss, _forward_sequential

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
    # This concatenates memories from all views and lets GASA attend across views
    B, N_views = images.shape[:2]

    # Check if we have extrinsics (required for world-frame PE)
    if gt_extrinsics is None and cached_da3_extrinsics is None:
        raise ValueError("--cross-view requires gt_extrinsics or da3_extrinsics for world-frame pointmaps")

    try:
        with autocast('cuda'):
            # Call through DDP wrapper (cross_view_mode dispatches to forward_multiview)
            # This ensures DDP's gradient sync hooks are properly triggered
            outputs = model(
                images, prompts, gt_masks.float(),
                gt_extrinsics=gt_extrinsics,
                gt_intrinsics=gt_intrinsics,
                intrinsics_orig_hw=intrinsics_orig_hw,
                cached_depth=cached_depth,
                da3_extrinsics=cached_da3_extrinsics,
                da3_intrinsics=cached_da3_intrinsics,
                cross_view_mode=True
            )
            # pred_masks: [B, N, H, W]
            pred = outputs['pred_masks']
            all_view_masks = outputs['all_masks']  # [B, N, Q, H, W]

            # Flatten for processing: [B*N, ...]
            pred_flat = pred.view(B * N_views, *pred.shape[2:])  # [B*N, H, W]
            gt_flat = gt_masks.view(B * N_views, *gt_masks.shape[2:]).float()  # [B*N, H, W]
            all_masks_flat = all_view_masks.view(B * N_views, *all_view_masks.shape[2:])  # [B*N, Q, H, W]

            # Resize GT if needed
            if gt_flat.shape[-2:] != pred_flat.shape[-2:]:
                gt_flat = F.interpolate(gt_flat.unsqueeze(1), size=pred_flat.shape[-2:],
                                       mode='nearest').squeeze(1)

            # Determine valid views
            valid_mask = gt_flat.sum(dim=(-2, -1)) > 0  # [B*N]

            # Compute loss for all views
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            n_valid = 0

            for i in range(B * N_views):
                view_pred = pred_flat[i:i+1]  # [1, H, W]
                view_gt = gt_flat[i:i+1]  # [1, H, W]

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

                    # Track per-category metrics
                    prompt_idx = i // N_views  # Map back to original batch index
                    category = prompts[prompt_idx] if prompt_idx < len(prompts) else "unknown"
                    cat_metrics.update(view_pred, view_gt, category)

            # IoU prediction loss (cross-view path)
            # Note: iou_pred is [B, Q] (per-scene), not [B*N, Q] (per-view)
            # We compute average IoU across valid views for each scene as the target
            if n_valid > 0 and args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                for b in range(B):
                    # Gather IoUs from all valid views of this scene
                    scene_ious = []
                    for v in range(N_views):
                        idx = b * N_views + v
                        if valid_mask[idx]:
                            actual_ious = compute_per_mask_ious(all_masks_flat[idx:idx+1], gt_flat[idx:idx+1])
                            scene_ious.append(actual_ious)
                    if len(scene_ious) > 0:
                        # Average IoUs across valid views
                        avg_scene_ious = torch.stack(scene_ious, dim=0).mean(dim=0)  # [1, Q]
                        iou_pred_loss = F.mse_loss(outputs['iou_pred'][b:b+1], avg_scene_ious.detach())
                        loss = loss + args.iou_head_weight * iou_pred_loss / B

            # Contrastive loss (cross-view path)
            # Note: pred_logits and iou_pred are [B, Q] (per-scene)
            # Use average IoU across valid views to determine best query per scene
            if n_valid > 0 and args.contrastive_weight > 0:
                for b in range(B):
                    # Gather IoUs from all valid views of this scene
                    scene_ious = []
                    for v in range(N_views):
                        idx = b * N_views + v
                        if valid_mask[idx]:
                            actual_ious = compute_per_mask_ious(all_masks_flat[idx:idx+1], gt_flat[idx:idx+1])
                            scene_ious.append(actual_ious)
                    if len(scene_ious) > 0:
                        avg_scene_ious = torch.stack(scene_ious, dim=0).mean(dim=0)  # [1, Q]
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

            # Align loss (cross-view path)
            # Note: pred_logits is [B, Q] (per-scene)
            # Use average IoU across valid views as the target
            if n_valid > 0 and args.align_weight > 0:
                for b in range(B):
                    # Gather IoUs from all valid views of this scene
                    scene_ious = []
                    for v in range(N_views):
                        idx = b * N_views + v
                        if valid_mask[idx]:
                            actual_ious = compute_per_mask_ious(all_masks_flat[idx:idx+1], gt_flat[idx:idx+1])
                            scene_ious.append(actual_ious)
                    if len(scene_ious) > 0:
                        avg_scene_ious = torch.stack(scene_ious, dim=0).mean(dim=0)  # [1, Q]
                        logits = outputs['pred_logits'][b:b+1]
                        align_l = align_loss(logits, avg_scene_ious,
                                            alpha=args.align_alpha,
                                            gamma=args.align_gamma,
                                            tau=args.align_tau)
                        loss = loss + args.align_weight * align_l / B

            # PER-LAYER AUXILIARY ALIGN LOSS (SAM3-style)
            # Compute align loss on intermediate decoder layer outputs to give
            # intermediate layers direct gradient signal for scoring.
            # Uses same IoU targets as final layer (masks only computed from final layer).
            if args.per_layer_align and args.align_weight > 0 and 'aux_queries' in outputs and outputs['aux_queries'] is not None:
                aux_align_weight = args.per_layer_align_weight if args.per_layer_align_weight is not None else args.align_weight
                num_aux_layers = len(outputs['aux_queries'])
                # Pre-compute per-scene avg IoUs (reuse across aux layers)
                cached_avg_ious = {}
                for b in range(B):
                    scene_ious = []
                    for v in range(N_views):
                        idx = b * N_views + v
                        if valid_mask[idx]:
                            actual_ious = compute_per_mask_ious(all_masks_flat[idx:idx+1], gt_flat[idx:idx+1])
                            scene_ious.append(actual_ious)
                    if len(scene_ious) > 0:
                        cached_avg_ious[b] = torch.stack(scene_ious, dim=0).mean(dim=0)
                # Apply align loss to each intermediate layer
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

            # Presence loss (cross-view path)
            # Note: presence_logit is [B, 1] (per-scene)
            # Target is 1 if any view in the scene has valid GT, 0 otherwise
            if args.presence_weight > 0 and 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                # Check if each scene has at least one valid view
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

            # Centroid loss (cross-view path)
            # Note: per_query_centroids is [B, Q, 3] (per-scene, in world coords)
            # We compute GT centroid per view and average for the scene target
            if n_valid > 0 and args.use_centroid_head and args.centroid_weight > 0 and 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                pointmaps_full = outputs['pointmaps_full']  # [B, N, H_da3, W_da3, 3]
                pointmaps_full_flat = pointmaps_full.view(B * N_views, *pointmaps_full.shape[2:])  # [B*N, H, W, 3]
                pm_h, pm_w = pointmaps_full_flat.shape[1:3]

                # Resize GT masks to match pointmaps resolution
                gt_resized = F.interpolate(
                    gt_flat.unsqueeze(1).float(),
                    size=(pm_h, pm_w),
                    mode='nearest'
                ).squeeze(1)  # [B*N, H_da3, W_da3]

                # Resize pred masks for mask-based centroid
                pred_resized = F.interpolate(
                    pred_flat.unsqueeze(1),
                    size=(pm_h, pm_w),
                    mode='bilinear', align_corners=False
                ).squeeze(1)  # [B*N, H, W]

                per_query_cents = outputs['per_query_centroids']  # [B, Q, 3]
                best_idx_flat = outputs['best_idx']  # [B*N]

                for b in range(B):
                    # Gather GT centroids from valid views and average
                    gt_cents = []
                    pred_cents = []
                    for v in range(N_views):
                        idx = b * N_views + v
                        if valid_mask[idx]:
                            gt_cent = compute_gt_centroid(gt_resized[idx], pointmaps_full_flat[idx])
                            gt_cents.append(gt_cent)
                            if args.mask_based_centroid:
                                pred_cent = compute_gt_centroid(pred_resized[idx], pointmaps_full_flat[idx])
                                pred_cents.append(pred_cent)

                    if len(gt_cents) > 0:
                        avg_gt_cent = torch.stack(gt_cents, dim=0).mean(dim=0)  # [3]
                        if args.mask_based_centroid and len(pred_cents) > 0:
                            selected_cent = torch.stack(pred_cents, dim=0).mean(dim=0)  # [3]
                        else:
                            # Attention-based: use centroid from best query (use first valid view's best_idx)
                            first_valid_idx = b * N_views + [v for v in range(N_views) if valid_mask[b * N_views + v]][0]
                            selected_cent = per_query_cents[b, best_idx_flat[first_valid_idx]]  # [3]
                        cent_loss = centroid_loss(selected_cent.unsqueeze(0), avg_gt_cent.unsqueeze(0))
                        loss = loss + args.centroid_weight * cent_loss / B

            # Centroid error tracking for Acc@m metrics (cross-view path)
            if n_valid > 0 and (args.use_centroid_head or args.eval_localization) and 'pointmaps_full' in outputs:
                with torch.no_grad():
                    pointmaps_full = outputs['pointmaps_full']  # [B, N, H_da3, W_da3, 3]
                    pointmaps_full_flat = pointmaps_full.view(B * N_views, *pointmaps_full.shape[2:])
                    pm_h, pm_w = pointmaps_full_flat.shape[1:3]

                    # Resize GT masks to pointmap resolution
                    gt_resized = F.interpolate(
                        gt_flat.unsqueeze(1).float(),
                        size=(pm_h, pm_w),
                        mode='nearest'
                    ).squeeze(1)

                    # Resize pred masks to pointmap resolution
                    pred_resized = F.interpolate(
                        pred_flat.unsqueeze(1),
                        size=(pm_h, pm_w),
                        mode='bilinear', align_corners=False
                    ).squeeze(1)

                    # Get normalization scale to convert back to meters
                    norm_params = outputs.get('norm_params', None)
                    scale = norm_params['scale'].item() if norm_params and 'scale' in norm_params else 1.0

                    for i in range(B * N_views):
                        if valid_mask[i]:
                            pred_cent = compute_gt_centroid(pred_resized[i], pointmaps_full_flat[i])
                            gt_cent = compute_gt_centroid(gt_resized[i], pointmaps_full_flat[i])
                            dist_error = torch.norm(pred_cent - gt_cent).item() * scale
                            epoch_centroid_errors.append(dist_error)

            # Connect ALL auxiliary heads to graph to ensure DDP gradient sync
            if 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                loss = loss + outputs['presence_logit'].sum() * 0.0
            if 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                loss = loss + outputs['iou_pred'].sum() * 0.0
            if 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                loss = loss + outputs['per_query_centroids'].sum() * 0.0
            if 'text_scores' in outputs and outputs['text_scores'] is not None:
                loss = loss + outputs['text_scores'].sum() * 0.0
            if 'joint_scores' in outputs and outputs['joint_scores'] is not None:
                loss = loss + outputs['joint_scores'].sum() * 0.0

            # Nuclear option: connect ALL trainable GASA decoder params
            base_module = model.module if hasattr(model, 'module') else model
            for p in base_module.gasa_decoder.parameters():
                if p.requires_grad:
                    loss = loss + p.sum() * 0.0
            for p in base_module.query_proj.parameters():
                if p.requires_grad:
                    loss = loss + p.sum() * 0.0

            if n_valid > 0:
                batch_loss_tensor = batch_loss_tensor + loss.detach()
            accumulated_loss = loss / args.grad_accum
            valid = n_valid

    except Exception as e:
        ddp.print(f"Error in cross-view forward: {e}")
        import traceback
        traceback.print_exc()

    return (accumulated_loss, valid, batch_loss_tensor, batch_iou_tensor, batch_macc_tensor,
            batch_recall_tensor, batch_sheaf_loss_tensor, last_vis_data)


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
        # but other objects ARE visible → must not zero out their loss
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
            outputs = model(all_views, all_prompts, fwd_gt,
                          gt_extrinsics=all_extrinsics,
                          gt_intrinsics=all_intrinsics,
                          spatial_qualifier_idx=all_spatial_idx,
                          intrinsics_orig_hw=intrinsics_orig_hw,
                          cached_depth=all_cached_depth,
                          da3_extrinsics=all_da3_extrinsics,
                          da3_intrinsics=all_da3_intrinsics,
                          num_texts=multi_object_K)

            # SAM3-MO: outputs are [B*N*K, ...], reshape GT and valid_mask
            if outputs.get('sam3_mo_K') is not None:
                sam3_K = outputs['sam3_mo_K']
                # Reshape GT: [B*N, K, H, W] → [B*N*K, H, W]
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
                        from scipy.optimize import linear_sum_assignment
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

                # Mask smoothing (29x29 avg pool, matches eval-time LangSplat protocol)
                if args.mask_smooth_kernel > 0:
                    sk = args.mask_smooth_kernel
                    sp = sk // 2
                    pred_for_loss = F.avg_pool2d(
                        pred_for_loss.unsqueeze(1), kernel_size=sk, stride=1, padding=sp,
                        count_include_pad=False
                    ).squeeze(1)

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

            # Text scoring loss: REMOVED — pred_logits now comes from DotProductScoring head,
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

            # Connect ALL auxiliary heads to graph to ensure DDP gradient sync
            # This must happen ALWAYS to avoid "unused parameters" error with DDP
            # Multiply by 0 to connect without affecting gradients

            # Connect presence_logit (from presence_token + presence_head)
            # Even if presence_weight > 0, we already added it above, but * 0.0 is safe
            if 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                loss = loss + outputs['presence_logit'].sum() * 0.0

            # Connect iou_pred (from iou_head)
            if 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                loss = loss + outputs['iou_pred'].sum() * 0.0

            # Connect per_query_centroids (from centroid_proj)
            if 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                loss = loss + outputs['per_query_centroids'].sum() * 0.0

            # Connect text_scores (from scoring head)
            if 'text_scores' in outputs and outputs['text_scores'] is not None:
                loss = loss + outputs['text_scores'].sum() * 0.0
            if 'joint_scores' in outputs and outputs['joint_scores'] is not None:
                loss = loss + outputs['joint_scores'].sum() * 0.0

            # Nuclear option: Connect ALL trainable GASA decoder params to loss
            # This ensures DDP gradient sync never fails due to unused params
            # The * 0.0 means no actual gradient contribution
            base_module = model.module if hasattr(model, 'module') else model
            for p in base_module.gasa_decoder.parameters():
                if p.requires_grad:
                    loss = loss + p.sum() * 0.0
            # Also connect query_proj
            for p in base_module.query_proj.parameters():
                if p.requires_grad:
                    loss = loss + p.sum() * 0.0

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
        ddp.print(f"Error in batched forward: {e}")
        import traceback
        traceback.print_exc()

    return (accumulated_loss, valid, batch_loss_tensor, batch_iou_tensor, batch_macc_tensor,
            batch_recall_tensor, batch_sheaf_loss_tensor, last_vis_data)

