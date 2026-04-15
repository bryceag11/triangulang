"""Forward pass helpers for TrianguLang training loop."""
import triangulang
import torch

logger = triangulang.get_logger(__name__)
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


def _compute_sheaf_loss(sheaf_loss_fn, feature_sheaf_loss_fn, sheaf_preds, sheaf_pointmaps,
                        sheaf_embeddings, args, device, ddp):
    accumulated_loss = None
    batch_loss_contribution = torch.tensor(0.0, device=device)
    batch_sheaf_loss_tensor = torch.tensor(0.0, device=device)

    if sheaf_loss_fn is not None and len(sheaf_preds) >= 2:
        try:
            with autocast('cuda'):
                stacked_preds = torch.stack(sheaf_preds, dim=1)  # [B, N_valid, H, W]
                stacked_pts = torch.stack(sheaf_pointmaps, dim=1)  # [B, N_valid, H, W, 3]
                _, _, H_pred, W_pred = stacked_preds.shape
                _, _, H_pts, W_pts, _ = stacked_pts.shape
                if H_pred != H_pts or W_pred != W_pts:
                    stacked_preds = F.interpolate(
                        stacked_preds, size=(H_pts, W_pts), mode='bilinear', align_corners=False
                    )
                sheaf_loss = sheaf_loss_fn(stacked_preds, stacked_pts)
                accumulated_loss = args.sheaf_weight * sheaf_loss / args.grad_accum
                batch_loss_contribution = batch_loss_contribution + args.sheaf_weight * sheaf_loss.detach()
                batch_sheaf_loss_tensor = sheaf_loss.detach()
        except Exception as e:
            sheaf_loss_fn._failure_count += 1
            if ddp.is_main:
                print(f"  [SHEAF WARNING] Loss computation failed ({sheaf_loss_fn._failure_count} total failures): {e}")
                if sheaf_loss_fn._failure_count <= 3:
                    import traceback
                    traceback.print_exc()
                if sheaf_loss_fn._failure_count == 10:
                    print(f"  [SHEAF ERROR] 10 consecutive failures. Sheaf loss may not be working. "
                          f"Check world_pointmaps and correspondence quality.")

    if feature_sheaf_loss_fn is not None and len(sheaf_embeddings) >= 2 and len(sheaf_pointmaps) >= 2:
        try:
            with autocast('cuda'):
                stacked_embs = torch.stack(sheaf_embeddings, dim=1)  # [B, N_valid, H, W, D]
                stacked_pts = torch.stack(sheaf_pointmaps, dim=1)  # [B, N_valid, H, W, 3]
                B_e, V_e, H_e, W_e, D_e = stacked_embs.shape
                _, _, H_p, W_p, _ = stacked_pts.shape
                if H_e != H_p or W_e != W_p:
                    stacked_embs = stacked_embs.reshape(B_e * V_e, H_e, W_e, D_e).permute(0, 3, 1, 2)
                    stacked_embs = F.interpolate(stacked_embs, size=(H_p, W_p), mode='bilinear', align_corners=False)
                    stacked_embs = stacked_embs.permute(0, 2, 3, 1).reshape(B_e, V_e, H_p, W_p, D_e)

                feature_sheaf_total = torch.tensor(0.0, device=device)
                n_pairs = 0
                for vi in range(V_e):
                    for vj in range(vi + 1, V_e):
                        if args.sheaf_max_frame_distance > 0 and (vj - vi) > args.sheaf_max_frame_distance:
                            continue
                        for b in range(B_e):
                            pts_i = stacked_pts[b, vi].reshape(-1, 3)
                            pts_j = stacked_pts[b, vj].reshape(-1, 3)
                            feats_i = stacked_embs[b, vi].reshape(-1, D_e)
                            feats_j = stacked_embs[b, vj].reshape(-1, D_e)
                            n_sub = min(1024, pts_i.shape[0])
                            idx_i = torch.randperm(pts_i.shape[0], device=device)[:n_sub]
                            pts_i_s, feats_i_s = pts_i[idx_i], feats_i[idx_i]
                            idx_j = torch.randperm(pts_j.shape[0], device=device)[:n_sub]
                            pts_j_s, feats_j_s = pts_j[idx_j], feats_j[idx_j]
                            dists = torch.cdist(pts_i_s, pts_j_s)
                            min_dists, nn_idx = dists.min(dim=-1)
                            valid_corresp = min_dists < args.sheaf_threshold
                            if valid_corresp.sum() < 5:
                                continue
                            fi_matched = feats_i_s[valid_corresp]
                            fj_matched = feats_j_s[nn_idx[valid_corresp]]
                            pi_matched = pts_i_s[valid_corresp]
                            pj_matched = pts_j_s[nn_idx[valid_corresp]]
                            di_matched = min_dists[valid_corresp]
                            ctx_i = AsymmetricRestrictionSheaf.compute_context(pi_matched, pj_matched, di_matched)
                            ctx_j = AsymmetricRestrictionSheaf.compute_context(pj_matched, pi_matched, di_matched)
                            pair_loss = feature_sheaf_loss_fn(fi_matched, fj_matched, ctx_i, ctx_j)
                            feature_sheaf_total = feature_sheaf_total + pair_loss
                            n_pairs += 1

                if n_pairs > 0:
                    feature_sheaf_loss = feature_sheaf_total / n_pairs
                    fsf_term = args.sheaf_weight * feature_sheaf_loss / args.grad_accum
                    accumulated_loss = fsf_term if accumulated_loss is None else accumulated_loss + fsf_term
                    batch_loss_contribution = batch_loss_contribution + args.sheaf_weight * feature_sheaf_loss.detach()
                    batch_sheaf_loss_tensor = feature_sheaf_loss.detach()
        except Exception as e:
            if not hasattr(feature_sheaf_loss_fn, '_failure_count'):
                feature_sheaf_loss_fn._failure_count = 0
            feature_sheaf_loss_fn._failure_count += 1
            if ddp.is_main:
                print(f"  [FEATURE-SHEAF WARNING] Loss failed ({feature_sheaf_loss_fn._failure_count}): {e}")
                if feature_sheaf_loss_fn._failure_count <= 3:
                    import traceback
                    traceback.print_exc()

    return accumulated_loss, batch_loss_contribution, batch_sheaf_loss_tensor

def _forward_sequential(model, base_model, images, gt_masks, prompts, batch, args, device, ddp,
                        N_views, B, N, gt_extrinsics, gt_intrinsics, intrinsics_orig_hw,
                        cached_depth, cached_da3_extrinsics, cached_da3_intrinsics,
                        spatial_qualifier_idx, epoch, start_epoch, batch_idx,
                        sheaf_loss_fn, feature_sheaf_loss_fn, cat_metrics, epoch_centroid_errors,
                        gt_aware_spatial, sync_context,
                        batch_iou_tensor, batch_macc_tensor, batch_recall_tensor,
                        batch_sheaf_loss_tensor):
    accumulated_loss = None
    valid = 0
    batch_loss_tensor = torch.tensor(0.0, device=device)
    last_vis_data = None
    sheaf_preds = []
    sheaf_pointmaps = []
    sheaf_embeddings = []
    # Get pre-computed coverage at ORIGINAL resolution if available
    gt_mask_coverage = batch.get('gt_mask_coverage')  # (B, N) or None
    if gt_mask_coverage is not None:
        gt_mask_coverage = gt_mask_coverage.to(device)

    # Detect multi-object mode
    seq_multi_object = gt_masks.dim() == 5  # [B, K, N, H, W]
    seq_multi_K = 1
    seq_multi_prompts = None  # List[List[str]] [B][K]
    seq_num_objects = None  # [B] tensor of actual K per item
    if seq_multi_object:
        seq_num_objects = batch.get('num_objects', None)
        if seq_num_objects is not None:
            seq_num_objects = seq_num_objects.to(device)
            seq_multi_K = int(seq_num_objects.max().item())
        else:
            seq_multi_K = gt_masks.shape[1]
        seq_multi_prompts = batch.get('multi_object_prompts', None)  # [B][K]

        # Apply spatial augmentation to multi-object prompts
        # Each object gets independently augmented (e.g., "chair" -> "nearest chair")
        # Only applies to multi-instance objects (same label appears 2+ times)
        if seq_multi_prompts is not None and gt_aware_spatial is not None:
            spatial_contexts_raw = batch.get('spatial_context', None)
            spatial_contexts = list(spatial_contexts_raw) if isinstance(spatial_contexts_raw, (list, tuple)) else [None] * B
            augmented_multi_prompts = []
            augmented_spatial_indices_mo = []
            for b_idx in range(B):
                ctx = spatial_contexts[b_idx] if b_idx < len(spatial_contexts) else None
                b_prompts = seq_multi_prompts[b_idx]
                aug_b = []
                aug_b_idx = []
                for k, p in enumerate(b_prompts):
                    aug_p, _, s_idx = gt_aware_spatial.augment(p, ctx)
                    aug_b.append(aug_p)
                    aug_b_idx.append(s_idx)
                augmented_multi_prompts.append(aug_b)
                augmented_spatial_indices_mo.append(aug_b_idx)
            seq_multi_prompts = augmented_multi_prompts
            # Build per-object spatial qualifier indices [K*B]
            spatial_qualifier_idx = torch.tensor(
                [augmented_spatial_indices_mo[b][k]
                 for b in range(B) for k in range(seq_multi_K)],
                device=device, dtype=torch.long
            )

        # Build flat prompt list: [b0_t0, b0_t1, ..., b0_tK, b1_t0, ...] = K*B strings
        if seq_multi_prompts is not None:
            flat_prompts = []
            for b_idx in range(B):
                for k in range(seq_multi_K):
                    flat_prompts.append(seq_multi_prompts[b_idx][k])
        else:
            flat_prompts = prompts  # Fallback to single prompts

    with sync_context():
        for v in range(N):
            view_img = images[:, v]
            if seq_multi_object:
                # Multi-object: use ANY object coverage for validity, all K for loss
                view_gt_all = gt_masks[:, :, v].float()  # [B, K, H, W] all objects for Hungarian
                view_gt = view_gt_all.max(dim=1).values  # [B, H, W] union for coverage check
            else:
                view_gt = gt_masks[:, v].float()  # [B, H, W]
                view_gt_all = None
            # Check if view has valid GT (non-empty mask with sufficient coverage)
            # IMPORTANT: Do NOT skip empty views! Process them with loss masked to 0.
            # Skipping causes DDP straggler problem and requires find_unused_parameters=True.
            if gt_mask_coverage is not None:
                # Use pre-computed coverage at ORIGINAL resolution
                mask_coverage = gt_mask_coverage[:, v].mean().item()  # avg across batch
            else:
                # Fallback: compute on resized mask (less accurate)
                mask_coverage = view_gt.sum() / view_gt.numel()
            is_valid_view = (mask_coverage >= args.min_mask_coverage) if args.min_mask_coverage > 0 else (view_gt.sum() > 0)

            # Get view-specific extrinsics/intrinsics/cached_depth
            view_extrinsics = gt_extrinsics[:, v] if gt_extrinsics is not None else None
            view_intrinsics = gt_intrinsics[:, v] if gt_intrinsics is not None else None
            view_cached_depth = cached_depth[:, v] if cached_depth is not None else None

            # Get view-specific cached DA3-NESTED poses (for world-frame GASA)
            view_da3_extrinsics = cached_da3_extrinsics[:, v] if cached_da3_extrinsics is not None else None
            view_da3_intrinsics = cached_da3_intrinsics[:, v] if cached_da3_intrinsics is not None else None

            try:
                with autocast('cuda'):
                    # Use multi-object prompts if available
                    fwd_prompts = flat_prompts if seq_multi_object and seq_multi_prompts is not None else prompts
                    fwd_num_texts = seq_multi_K if seq_multi_object else 1

                    # SAM3-MO: pass per-object GT for oracle mask selection
                    fwd_gt = view_gt_all if (getattr(args, 'sam3_multi_object', False) and view_gt_all is not None) else view_gt

                    outputs = model(view_img, fwd_prompts, fwd_gt,
                                  gt_extrinsics=view_extrinsics,
                                  gt_intrinsics=view_intrinsics,
                                  spatial_qualifier_idx=spatial_qualifier_idx,
                                  intrinsics_orig_hw=intrinsics_orig_hw,
                                  cached_depth=view_cached_depth,
                                  da3_extrinsics=view_da3_extrinsics,
                                  da3_intrinsics=view_da3_intrinsics,
                                  num_texts=fwd_num_texts)

                    # SAM3-MO: model expanded batch by K. Each item = 1 object.
                    # Use simple per-object single-object loss (no matching needed).
                    if outputs.get('sam3_mo_K') is not None:
                        sam3_K = outputs['sam3_mo_K']
                        # GT: [B, K, H, W] -> [B*K, H, W]
                        per_obj_gt = view_gt_all.reshape(-1, *view_gt_all.shape[2:]).float()  # [B*K, H, W]
                        per_obj_valid = (per_obj_gt.sum(dim=(-2, -1)) > 0)

                        # Predictions: [B*K, ...]
                        pred = outputs['pred_masks'][:, 0] if outputs['pred_masks'].dim() == 4 else outputs['pred_masks']
                        if pred.shape[-2:] != per_obj_gt.shape[-2:]:
                            pred = F.interpolate(pred.unsqueeze(1), size=per_obj_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

                        # Mask smoothing (29x29 avg pool, matches eval-time LangSplat protocol)
                        pred_for_loss = pred
                        if args.mask_smooth_kernel > 0:
                            sk = args.mask_smooth_kernel
                            sp = sk // 2
                            pred_for_loss = F.avg_pool2d(
                                pred.unsqueeze(1), kernel_size=sk, stride=1, padding=sp,
                                count_include_pad=False
                            ).squeeze(1)

                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                        n_obj_valid = 0
                        for oi in range(pred.shape[0]):
                            obj_loss = (
                                args.focal_weight * focal_loss(pred_for_loss[oi:oi+1], per_obj_gt[oi:oi+1], alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                args.dice_weight * dice_loss(pred_for_loss[oi:oi+1].unsqueeze(1), per_obj_gt[oi:oi+1].unsqueeze(1))
                            )
                            # Boundary loss: penalize predictions far from GT boundary
                            if getattr(args, 'boundary_weight', 0) > 0 and per_obj_valid[oi]:
                                obj_loss = obj_loss + args.boundary_weight * boundary_loss(
                                    pred_for_loss[oi:oi+1], per_obj_gt[oi:oi+1])
                            if not per_obj_valid[oi]:
                                obj_loss = obj_loss * 0.0
                            loss = loss + obj_loss

                            if per_obj_valid[oi]:
                                n_obj_valid += 1
                                batch_iou_tensor = batch_iou_tensor + compute_iou(pred[oi:oi+1].unsqueeze(1), per_obj_gt[oi:oi+1].unsqueeze(1), return_tensor=True)
                                batch_recall_tensor = batch_recall_tensor + compute_recall(pred[oi:oi+1].unsqueeze(1), per_obj_gt[oi:oi+1].unsqueeze(1), return_tensor=True)
                                # Track per-category
                                b_idx = oi // sam3_K
                                k_idx = oi % sam3_K
                                if seq_multi_prompts and b_idx < len(seq_multi_prompts) and k_idx < len(seq_multi_prompts[b_idx]):
                                    cat_name = seq_multi_prompts[b_idx][k_idx]
                                else:
                                    cat_name = "unknown"
                                cat_metrics.update(pred[oi:oi+1], per_obj_gt[oi:oi+1], cat_name)

                        # Align loss for SAM3-MO
                        if n_obj_valid > 0 and args.align_weight > 0:
                            all_masks_mo = outputs['all_masks']  # [B*K, Q, H, W]
                            for oi in range(pred.shape[0]):
                                if per_obj_valid[oi]:
                                    oi_ious = compute_per_mask_ious(all_masks_mo[oi:oi+1], per_obj_gt[oi:oi+1])
                                    logits = outputs['pred_logits'][oi:oi+1]
                                    al = align_loss(logits, oi_ious, alpha=args.align_alpha, gamma=args.align_gamma, tau=args.align_tau)
                                    loss = loss + args.align_weight * al / n_obj_valid

                        # Presence loss for SAM3-MO
                        if args.presence_weight > 0 and 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                            presence_targets = per_obj_valid.float().unsqueeze(1)  # [B*K, 1]
                            if args.presence_focal:
                                presence_loss = focal_loss(outputs['presence_logit'], presence_targets,
                                                           alpha=args.presence_alpha, gamma=args.presence_gamma)
                            else:
                                presence_loss = F.binary_cross_entropy_with_logits(
                                    outputs['presence_logit'], presence_targets)
                            loss = loss + args.presence_weight * presence_loss

                        # Centroid loss for SAM3-MO
                        if n_obj_valid > 0 and args.use_centroid_head and args.centroid_weight > 0 and 'per_query_centroids' in outputs:
                            pointmaps_full = outputs['pointmaps_full']  # [B*K, H, W, 3]
                            pm_h, pm_w = pointmaps_full.shape[1:3]
                            gt_resized = F.interpolate(per_obj_gt.unsqueeze(1).float(), size=(pm_h, pm_w), mode='nearest').squeeze(1)
                            per_query_cents = outputs['per_query_centroids']
                            best_idx = outputs['best_idx']
                            for oi in range(pred.shape[0]):
                                if per_obj_valid[oi]:
                                    gt_cent = compute_gt_centroid(gt_resized[oi], pointmaps_full[oi])
                                    selected_cent = per_query_cents[oi, best_idx[oi]]
                                    cent_loss = centroid_loss(selected_cent.unsqueeze(0), gt_cent.unsqueeze(0))
                                    loss = loss + args.centroid_weight * cent_loss / n_obj_valid

                        # Spatial ranking loss for SAM3-MO
                        # Teaches model to produce masks whose centroids respect spatial ordering
                        if n_obj_valid > 0 and args.spatial_ranking_weight > 0 and view_cached_depth is not None:
                            for b_idx in range(B):
                                K_b = int(seq_num_objects[b_idx].item()) if seq_num_objects is not None else sam3_K
                                if K_b < 2:
                                    continue
                                # Get this batch item's predictions and depth
                                b_start = b_idx * sam3_K
                                b_pred = pred[b_start:b_start + K_b]  # [K_b, H, W]
                                b_depth = view_cached_depth[b_idx]  # [1, H, W]
                                if b_depth.shape[-2:] != b_pred.shape[-2:]:
                                    b_depth = F.interpolate(b_depth.unsqueeze(0), size=b_pred.shape[-2:],
                                                            mode='bilinear', align_corners=False).squeeze(0)
                                # Get labels for this batch item
                                b_labels = []
                                for k in range(K_b):
                                    if seq_multi_prompts and b_idx < len(seq_multi_prompts) and k < len(seq_multi_prompts[b_idx]):
                                        b_labels.append(seq_multi_prompts[b_idx][k])
                                    else:
                                        b_labels.append(f"obj_{k}")
                                # Only include valid objects
                                b_valid = per_obj_valid[b_start:b_start + K_b]
                                if b_valid.sum() >= 2:
                                    sr_loss = spatial_ranking_loss(
                                        b_pred, b_depth, b_labels,
                                        margin=args.spatial_ranking_margin
                                    )
                                    loss = loss + args.spatial_ranking_weight * sr_loss / B

                                # Spatial selection loss: trains spatial tokens to pick correct instance
                                if spatial_qualifier_idx is not None and (spatial_qualifier_idx > 0).any():
                                    b_sq = spatial_qualifier_idx[b_start:b_start + K_b]
                                    b_gt = per_obj_gt[b_start:b_start + K_b]
                                    ss_loss = spatial_selection_loss(
                                        b_pred, b_gt, b_depth, b_labels, b_sq
                                    )
                                    loss = loss + args.spatial_ranking_weight * ss_loss / B

                        # Connect unused params for DDP
                        base_module = model.module if hasattr(model, 'module') else model
                        for p in base_module.gasa_decoder.parameters():
                            if p.requires_grad:
                                loss = loss + p.sum() * 0.0

                    # Multi-object: Hungarian matching per batch item (non-SAM3-MO path)
                    elif seq_multi_object and seq_multi_K > 1 and 'all_masks' in outputs:
                        all_masks = outputs['all_masks']  # [B, Q, H, W]
                        text_scores_multi = outputs.get('text_scores', None)  # [B, Q, K] or None

                        # Resize to GT resolution if needed
                        if all_masks.shape[-2:] != view_gt_all.shape[-2:]:
                            all_masks = F.interpolate(all_masks, size=view_gt_all.shape[-2:],
                                                      mode='bilinear', align_corners=False)

                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                        n_matched_total = 0
                        for b_idx in range(B):
                            K_b = int(seq_num_objects[b_idx].item()) if seq_num_objects is not None else seq_multi_K
                            view_preds = all_masks[b_idx]  # [Q, H, W]
                            view_gts = view_gt_all[b_idx, :K_b]  # [K_b, H, W]

                            view_text_scores = None
                            if text_scores_multi is not None and text_scores_multi.dim() == 3:
                                view_text_scores = text_scores_multi[b_idx, :, :K_b]

                            if args.match_strategy == 'text_greedy' and view_text_scores is not None:
                                matched_pairs, unmatched_q = text_greedy_match(view_text_scores, K_b)
                            else:
                                matched_pairs, unmatched_q = hungarian_match(view_preds, view_gts, K_b, view_text_scores)

                            for q_idx, k_idx in matched_pairs:
                                if k_idx < K_b and view_gts[k_idx].sum() > 0:
                                    pred_k = all_masks[b_idx, q_idx]  # [H, W]
                                    gt_k = view_gts[k_idx]  # [H, W]
                                    pair_loss = (
                                        args.focal_weight * focal_loss(pred_k, gt_k, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                        args.dice_weight * dice_loss(pred_k.unsqueeze(0).unsqueeze(0), gt_k.unsqueeze(0).unsqueeze(0))
                                    )
                                    loss = loss + pair_loss
                                    n_matched_total += 1

                                    # Track per-category metrics
                                    cat_name = seq_multi_prompts[b_idx][k_idx] if seq_multi_prompts else prompts[b_idx]
                                    cat_metrics.update(pred_k.unsqueeze(0), gt_k.unsqueeze(0), cat_name)

                            # No-object loss for unmatched queries (sequential path)
                            if args.no_object_weight > 0 and len(unmatched_q) > 0:
                                empty_gt = torch.zeros(1, all_masks.shape[-2], all_masks.shape[-1], device=device)
                                no_obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
                                for q_idx in unmatched_q:
                                    pred_q = all_masks[b_idx, q_idx:q_idx+1]
                                    no_obj_loss = no_obj_loss + F.binary_cross_entropy_with_logits(
                                        pred_q, empty_gt, reduction='mean')
                                no_obj_loss = no_obj_loss / len(unmatched_q)
                                loss = loss + args.no_object_weight * no_obj_loss

                        if n_matched_total > 0:
                            loss = loss / n_matched_total

                        # Use best-matched primary object mask for IoU/metrics tracking
                        pred = outputs['pred_masks'][:, 0] if outputs['pred_masks'].dim() == 4 else outputs['pred_masks']
                        if pred.shape[-2:] != view_gt.shape[-2:]:
                            pred = F.interpolate(pred.unsqueeze(1), size=view_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

                    else:
                        # Single-object path (unchanged)
                        pred = outputs['pred_masks'][:, 0] if outputs['pred_masks'].dim() == 4 else outputs['pred_masks']
                        if pred.shape[-2:] != view_gt.shape[-2:]:
                            pred = F.interpolate(pred.unsqueeze(1), size=view_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

                        # Main losses: focal + dice
                        loss = args.focal_weight * focal_loss(pred, view_gt, alpha=args.focal_alpha, gamma=args.focal_gamma) + args.dice_weight * dice_loss(pred.unsqueeze(1), view_gt.unsqueeze(1))

                    # Presence loss: predict 1.0 when object exists, 0.0 when empty
                    # This always runs (even for empty views) to train presence detection
                    # SAM3-MO handles presence inside its own block above
                    if outputs.get('sam3_mo_K') is None and args.presence_weight > 0 and 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                        presence_target = torch.full((view_img.shape[0], 1), float(is_valid_view), device=view_img.device)
                        if args.presence_focal:
                            presence_loss = focal_loss(outputs['presence_logit'], presence_target,
                                                       alpha=args.presence_alpha, gamma=args.presence_gamma)
                        else:
                            presence_loss = F.binary_cross_entropy_with_logits(
                                outputs['presence_logit'], presence_target
                            )
                        loss = loss + args.presence_weight * presence_loss

                    # Centroid loss: predict 3D object centroid (only for valid views)
                    # SAM3-MO handles centroid inside its own block above
                    if outputs.get('sam3_mo_K') is None and is_valid_view and args.use_centroid_head and args.centroid_weight > 0 and 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                        # Compute GT centroid from mask and full-res pointmaps
                        # pointmaps_full is at DA3 resolution (e.g., 518x518)
                        # view_gt is at training resolution (e.g., 1008x1008)
                        pointmaps_full = outputs['pointmaps_full']  # [B, H_da3, W_da3, 3]
                        pm_h, pm_w = pointmaps_full.shape[1:3]


                        # Resize masks to match pointmaps resolution
                        view_gt_resized = F.interpolate(
                            view_gt.unsqueeze(1).float(),  # [B, 1, H, W]
                            size=(pm_h, pm_w),
                            mode='nearest'
                        ).squeeze(1)  # [B, H_da3, W_da3]

                        gt_centroids = []
                        for b_idx in range(view_gt_resized.shape[0]):
                            gt_cent = compute_gt_centroid(view_gt_resized[b_idx], pointmaps_full[b_idx])
                            gt_centroids.append(gt_cent)
                        gt_centroids = torch.stack(gt_centroids, dim=0)  # [B, 3]

                        # Choose centroid computation method
                        if args.mask_based_centroid:
                            # MASK-BASED: Compute centroid directly from predicted mask + depth
                            # This ties centroid quality directly to mask quality
                            pred_mask_resized = F.interpolate(
                                pred.unsqueeze(1),  # [B, 1, H, W]
                                size=(pm_h, pm_w),
                                mode='bilinear', align_corners=False
                            ).squeeze(1)  # [B, H_pm, W_pm]

                            pred_centroids = []
                            for b_idx in range(pred_mask_resized.shape[0]):
                                pred_cent = compute_gt_centroid(pred_mask_resized[b_idx], pointmaps_full[b_idx])
                                pred_centroids.append(pred_cent)
                            selected_centroids = torch.stack(pred_centroids, dim=0)  # [B, 3]
                        else:
                            # ATTENTION-BASED: Use centroid from selected query (attention-weighted + residual)
                            per_query_cents = outputs['per_query_centroids']  # [B, Q, 3]
                            best_idx = outputs['best_idx']  # [B]
                            B_cent = per_query_cents.shape[0]
                            b_indices = torch.arange(B_cent, device=per_query_cents.device)
                            selected_centroids = per_query_cents[b_indices, best_idx]  # [B, 3]

                        cent_loss = centroid_loss(selected_centroids, gt_centroids)
                        loss = loss + args.centroid_weight * cent_loss

                        # Track distance errors for Acc@m metrics (in REAL meters)
                        # Use MASK-BASED centroid (like evaluation) for comparable Acc@m
                        with torch.no_grad():
                            # Compute pred centroid from predicted mask (matching evaluate_gasa.py)
                            pred_mask_resized = F.interpolate(
                                pred.unsqueeze(1),  # [B, 1, H, W]
                                size=(pm_h, pm_w),
                                mode='bilinear', align_corners=False
                            ).squeeze(1)  # [B, H_pm, W_pm]

                            # Get normalization scale to convert back to meters
                            norm_params = outputs.get('norm_params', None)
                            scale = norm_params['scale'].item() if norm_params and 'scale' in norm_params else 1.0

                            for b_idx in range(pred_mask_resized.shape[0]):
                                # Compute centroid from predicted mask
                                pred_cent = compute_gt_centroid(pred_mask_resized[b_idx], pointmaps_full[b_idx])
                                gt_cent = gt_centroids[b_idx]
                                # Error: multiply by scale to convert normalized units to meters
                                dist_error = torch.norm(pred_cent - gt_cent).item() * scale
                                epoch_centroid_errors.append(dist_error)

                    # Eval-only localization metrics: compute Acc@m from mask+depth WITHOUT centroid head
                    # This tracks 3D localization quality during training without adding any loss
                    elif outputs.get('sam3_mo_K') is None and is_valid_view and args.eval_localization and 'pointmaps_full' in outputs:
                        with torch.no_grad():
                            pointmaps_full = outputs['pointmaps_full']  # [B, H_da3, W_da3, 3]
                            pm_h, pm_w = pointmaps_full.shape[1:3]

                            # Resize GT mask to pointmap resolution
                            view_gt_resized = F.interpolate(
                                view_gt.unsqueeze(1).float(),
                                size=(pm_h, pm_w),
                                mode='nearest'
                            ).squeeze(1)

                            # Resize pred mask to pointmap resolution
                            pred_mask_resized = F.interpolate(
                                pred.unsqueeze(1),
                                size=(pm_h, pm_w),
                                mode='bilinear', align_corners=False
                            ).squeeze(1)

                            # Get normalization scale to convert back to meters
                            norm_params = outputs.get('norm_params', None)
                            scale = norm_params['scale'].item() if norm_params and 'scale' in norm_params else 1.0

                            for b_idx in range(pred_mask_resized.shape[0]):
                                # Compute centroids from masks + depth
                                pred_cent = compute_gt_centroid(pred_mask_resized[b_idx], pointmaps_full[b_idx])
                                gt_cent = compute_gt_centroid(view_gt_resized[b_idx], pointmaps_full[b_idx])
                                # Error: multiply by scale to convert normalized units to meters
                                dist_error = torch.norm(pred_cent - gt_cent).item() * scale
                                epoch_centroid_errors.append(dist_error)

                    # IoU prediction loss: teach model to predict mask quality (only for valid views)
                    if outputs.get('sam3_mo_K') is None and is_valid_view and args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                        all_masks = outputs['all_masks']  # [B, Q, H, W]
                        actual_ious = compute_per_mask_ious(all_masks, view_gt)  # [B, Q]
                        iou_pred_loss = F.mse_loss(outputs['iou_pred'], actual_ious.detach())
                        loss = loss + args.iou_head_weight * iou_pred_loss

                    # Contrastive mask loss: push best mask's score above others
                    # Can use pred_logits (no IoU head) or iou_pred based on --contrastive-source
                    if outputs.get('sam3_mo_K') is None and is_valid_view and args.contrastive_weight > 0:
                        all_masks = outputs['all_masks']  # [B, Q, H, W]
                        actual_ious = compute_per_mask_ious(all_masks, view_gt)
                        best_idx = actual_ious.argmax(dim=1)  # [B]

                        # Choose score source based on flag
                        if args.contrastive_source == 'logits':
                            # Use mean mask logits - no IoU head needed!
                            scores = outputs['pred_logits']  # [B, Q]
                        elif args.contrastive_source == 'iou_pred' and 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                            # Use IoU predictions (requires --use-iou-head)
                            scores = outputs['iou_pred']  # [B, Q]
                        else:
                            scores = None

                        if scores is not None:
                            contrast_loss = contrastive_mask_loss(scores, best_idx, margin=args.contrastive_margin)
                            loss = loss + args.contrastive_weight * contrast_loss

                    # Text scoring loss: REMOVED. pred_logits now comes from DotProductScoring head,
                    # so the existing align loss trains text-query matching end-to-end (SAM3-style).

                    # Align loss: IoU-aware focal loss on pred_logits (SAM3/AlignDETR style)
                    # Trains the classification logits to directly predict mask quality
                    if outputs.get('sam3_mo_K') is None and is_valid_view and args.align_weight > 0:
                        all_masks = outputs['all_masks']  # [B, Q, H, W]
                        actual_ious = compute_per_mask_ious(all_masks, view_gt)  # [B, Q]
                        logits = outputs['pred_logits']  # [B, Q] - mean mask logits
                        align_l = align_loss(logits, actual_ious,
                                            alpha=args.align_alpha,
                                            gamma=args.align_gamma,
                                            tau=args.align_tau)
                        loss = loss + args.align_weight * align_l

                    # Collect predictions and pointmaps for sheaf consistency loss
                    # IMPORTANT: Use world_pointmaps for sheaf loss (world-frame consistency)
                    # Fall back to camera-frame pointmaps if world_pointmaps not available
                    # Only collect for valid views (sheaf loss requires non-empty masks)
                    if outputs.get('sam3_mo_K') is None and is_valid_view and (sheaf_loss_fn is not None or feature_sheaf_loss_fn is not None):
                        sheaf_preds.append(pred)  # [B, H, W] - keep gradients!
                        if 'world_pointmaps' in outputs:
                            sheaf_pointmaps.append(outputs['world_pointmaps'].detach())  # [B, H, W, 3] world-frame
                            # Debug: check if world pointmaps are being used
                            if batch_idx == 0 and v == 0 and epoch == start_epoch:
                                logger.debug(f"  [Sheaf] Using WORLD pointmaps for view {v}")
                        elif 'pointmaps' in outputs:
                            sheaf_pointmaps.append(outputs['pointmaps'].detach())  # [B, H, W, 3] camera-frame fallback
                            if batch_idx == 0 and v == 0 and epoch == start_epoch:
                                logger.warning(f"  [Sheaf] Using CAMERA-FRAME pointmaps for view {v} (world_pointmaps not in outputs)")

                        # Collect embeddings for feature sheaf loss
                        if feature_sheaf_loss_fn is not None:
                            if 'encoder_features' in outputs:
                                emb = outputs['encoder_features']  # [B, H, W, D]
                            else:
                                emb = pred.unsqueeze(-1)  # [B, H, W, 1]
                            sheaf_embeddings.append(emb)

                # For invalid views (empty GT), multiply loss by 0 to zero gradients
                # BUT keep the graph connected so all trainable params are used (DDP requirement)
                # SAM3-MO handles invalid objects and DDP param connection internally
                if outputs.get('sam3_mo_K') is None and not is_valid_view:
                    loss = loss * 0.0
                    # Also connect auxiliary heads (iou_head, centroid_head) to the graph
                    # with 0-weighted dummy terms to ensure DDP gradient sync works
                    if args.use_iou_head and 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                        loss = loss + outputs['iou_pred'].sum() * 0.0
                    if args.use_centroid_head and 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                        loss = loss + outputs['per_query_centroids'].sum() * 0.0
                    if 'text_scores' in outputs and outputs['text_scores'] is not None:
                        loss = loss + outputs['text_scores'].sum() * 0.0
                    if 'joint_scores' in outputs and outputs['joint_scores'] is not None:
                        loss = loss + outputs['joint_scores'].sum() * 0.0

                # Accumulate loss instead of backward per view
                # Divide by grad_accum only (not N) to match original gradient magnitude
                if accumulated_loss is None:
                    accumulated_loss = loss / args.grad_accum
                else:
                    accumulated_loss = accumulated_loss + loss / args.grad_accum

                # Only accumulate metrics for valid views (non-empty GT)
                # SAM3-MO: metrics already accumulated in the SAM3-MO block
                if outputs.get('sam3_mo_K') is not None:
                    batch_loss_tensor = batch_loss_tensor + loss.detach()
                    valid += n_obj_valid  # Match IoU accumulation count
                elif is_valid_view:
                    # Use tensor accumulation to avoid GPU-CPU sync per view
                    batch_loss_tensor = batch_loss_tensor + loss.detach()
                    batch_iou_tensor = batch_iou_tensor + compute_iou(pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                    batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                    batch_recall_tensor = batch_recall_tensor + compute_recall(pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                    valid += 1

                    # Track per-category metrics for proper mIoU
                    for b_idx in range(pred.shape[0]):
                        category = prompts[b_idx] if b_idx < len(prompts) else "unknown"
                        cat_metrics.update(pred[b_idx:b_idx+1], view_gt[b_idx:b_idx+1], category)

                    if last_vis_data is None and ddp.is_main:
                        last_vis_data = {'images': view_img.detach().cpu(), 'gt_masks': view_gt.detach().cpu(),
                                         'outputs': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()},
                                         'prompts': prompts}
            except Exception as e:
                logger.warning(f"Error: {e}")
                import traceback
                traceback.print_exc()

    sheaf_acc_loss, sheaf_batch_contrib, batch_sheaf_loss_tensor = _compute_sheaf_loss(
        sheaf_loss_fn, feature_sheaf_loss_fn, sheaf_preds, sheaf_pointmaps,
        sheaf_embeddings, args, device, ddp)
    if sheaf_acc_loss is not None:
        accumulated_loss = sheaf_acc_loss if accumulated_loss is None else accumulated_loss + sheaf_acc_loss
        batch_loss_tensor = batch_loss_tensor + sheaf_batch_contrib

    return (accumulated_loss, valid, batch_loss_tensor, batch_iou_tensor, batch_macc_tensor,
            batch_recall_tensor, batch_sheaf_loss_tensor, sheaf_preds, sheaf_pointmaps,
            sheaf_embeddings, last_vis_data)
