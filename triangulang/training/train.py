"""
TrianguLang Training Script - GASA Decoder for Geometry-Aware Segmentation

Architecture:
    SAM3 Backbone + Encoder (FROZEN) -> encoder_memory with semantics
    DA3 (FROZEN) -> depth -> pointmaps (world coordinates)
    GASA Decoder (TRAINABLE) -> object queries with geometric attention bias
    SAM3 SegHead (frozen) -> masks

Usage:
    torchrun --nproc_per_node=8 triangulang/training/train.py \
        --run-name my_run --epochs 30

"""
import warnings
# Suppress PyTorch scheduler deprecation warning (internal to SequentialLR)
warnings.filterwarnings('ignore', message='.*epoch parameter in.*scheduler.step.*')

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import sys
import gc
import json
import random
import time
from pathlib import Path
from contextlib import nullcontext

import tyro
import torch
from tqdm import tqdm
import psutil

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'sam3'))
sys.path.insert(0, str(PROJECT_ROOT / 'depth_anything_v3' / 'src'))

# DDP support - auto-detects if running via torchrun (must be after sys.path setup)
from triangulang.utils.ddp_utils import DDPManager

from triangulang.utils.spatial_reasoning import (
    parse_spatial_qualifier,
    get_spatial_qualifier_idx,
)
from triangulang.training.config import TrainConfig
from triangulang.utils.metrics import CategoryMetricsTracker
from triangulang.training.forward_passes import (
    _forward_cross_view, _forward_batch_views, _forward_sequential,
)
from triangulang.training.train_helpers import visualize_predictions
from triangulang.training.train_setup import (
    _setup_config, _init_environment, _load_datasets, _build_model,
    _setup_training, _save_checkpoint, _finalize_epoch, _run_validation_and_save,
)

import triangulang
logger = triangulang.get_logger(__name__)


def main():
    # Parse config
    config = tyro.cli(TrainConfig)
    args = config.to_namespace()
    args = _setup_config(args)

    # Initialize DDP (auto-detects if running via torchrun)
    ddp = DDPManager()
    ddp.init()
    triangulang.configure_logging(ddp.rank)
    device = _init_environment(args, ddp)

    dataset, dataloader, val_dataset, val_dataloader = _load_datasets(args, ddp)
    model, base_model, use_box, use_point = _build_model(args, ddp, device)

    (optimizer, scaler, scheduler, lora_manager, sheaf_loss_fn, feature_sheaf_loss_fn,
     spatial_augmentor, gt_aware_spatial, run_dir, checkpoint_dir,
     best_iou, best_val_miou, start_epoch) = _setup_training(model, base_model, args, device, ddp)

    # Only create dirs and save config on main process
    if ddp.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg_save = vars(args).copy()
        cfg_save['world_size'] = ddp.world_size
        cfg_save['resumed_from'] = str(args.resume) if args.resume else None
        cfg_save['use_box_prompts'] = use_box
        cfg_save['use_point_prompts'] = use_point
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(cfg_save, f, indent=2)
        logger.info(f"Config saved to {run_dir / 'config.json'}")
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(cfg_save, f, indent=2)
        logger.debug(f"Config also saved to {checkpoint_dir / 'config.json'}")

    ddp.barrier()  # Sync before training
    logger.info(f"Training for {args.epochs} epochs (starting from epoch {start_epoch + 1})...")

    cat_metrics = CategoryMetricsTracker()
    history = []
    if ddp.is_main:
        history_path = run_dir / 'history.json'
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                logger.info(f"Loaded existing history with {len(history)} epochs")
            except Exception as e:
                logger.info(f"Warning: Could not load history.json: {e}")
                history = []

    num_samples = 0
    avg_loss, avg_iou, avg_macc, avg_recall = 0.0, 0.0, 0.0, 0.0

    for epoch in range(start_epoch, args.epochs):
        # Early stopping check
        if args.stop_at_epoch > 0 and epoch >= args.stop_at_epoch:
            logger.info(f"[Early Stop] Reached --stop-at-epoch {args.stop_at_epoch}, stopping training.")
            break

        ddp.set_epoch(epoch)  # Important for proper shuffling in DDP
        model.train()
        base_model.sam3.eval()
        if base_model.da3 is not None:
            base_model.da3.eval()

        # Curriculum learning: switch or ramp mask selection strategy based on epoch
        epoch_loss, epoch_iou, epoch_macc, epoch_recall, num_samples = 0.0, 0.0, 0.0, 0.0, 0
        epoch_sheaf_loss = 0.0  # Track sheaf loss separately
        epoch_centroid_errors = []  # Track centroid distance errors for Acc@m metrics
        accum_valid = 0
        last_vis_data = None
        cat_metrics.reset()  # Reset per epoch

        pbar = tqdm(dataloader, desc=f"[R{ddp.rank}] Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # With drop_last=True and DistributedSampler, batch should never be None
            # and all ranks get same number of batches. This is just defensive.
            if batch is None:
                # Still need to do dummy backward for DDP sync if other ranks have data
                if ddp.is_distributed:
                    dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    scaler.scale(dummy_loss).backward()
                continue

            images = batch['images'].to(device, non_blocking=True)
            gt_masks = batch['gt_masks'].to(device, non_blocking=True)
            prompts = batch['prompts']
            B, N = images.shape[:2]

            # Get GT camera parameters if available (for world-consistent pointmaps / sheaf loss)
            # If --no-gt-poses is set, suppress GT poses to force using DA3-NESTED estimated poses
            if args.no_gt_poses:
                gt_extrinsics = None  # Force using DA3 poses instead
                gt_intrinsics = None
            else:
                gt_extrinsics = batch.get('extrinsics', None)  # [B, N, 4, 4]
                gt_intrinsics = batch.get('intrinsics', None)  # [B, N, 3, 3]
            intrinsics_orig_hw = batch.get('orig_hw', None)  # (H, W) original resolution for intrinsics
            if gt_extrinsics is not None:
                gt_extrinsics = gt_extrinsics.to(device, non_blocking=True)
            if gt_intrinsics is not None:
                gt_intrinsics = gt_intrinsics.to(device, non_blocking=True)

            # Get cached depth if available (2-4x faster training)
            cached_depth = batch.get('cached_depth', None)  # [B, N, 1, H, W] or None
            if cached_depth is not None:
                cached_depth = cached_depth.to(device, non_blocking=True)

            # Get cached DA3-NESTED poses if available (for world-frame GASA)
            cached_da3_extrinsics = batch.get('cached_da3_extrinsics', None)  # [B, N, 4, 4] or None
            cached_da3_intrinsics = batch.get('cached_da3_intrinsics', None)  # [B, N, 3, 3] or None
            if cached_da3_extrinsics is not None:
                cached_da3_extrinsics = cached_da3_extrinsics.to(device, non_blocking=True)
                # NOTE: Cache already stores c2w (camera-to-world) after Feb 2026 fix.
                # preprocess_da3_nested.py inverts w2c->c2w via extract_c2w_from_extrinsics().
                # DO NOT invert again here - that was causing double-inversion bug!
            if cached_da3_intrinsics is not None:
                cached_da3_intrinsics = cached_da3_intrinsics.to(device, non_blocking=True)

            # Log cached depth status (first batch of each epoch)
            if batch_idx == 0 and ddp.is_main:
                if cached_depth is not None:
                    logger.debug(f"  [Epoch {epoch}] cached_depth: {cached_depth.shape} DA3 BYPASSED")
                else:
                    logger.debug(f"  [Epoch {epoch}] cached_depth: None - DA3 RUNNING LIVE")

            # Log pose configuration (first batch only)
            if batch_idx == 0 and epoch == start_epoch and ddp.is_main:
                if args.no_gt_poses:
                    logger.debug(f"  GT poses suppressed (--no-gt-poses) -> using DA3-NESTED estimated poses")
                if args.use_da3_poses_for_gasa and cached_da3_extrinsics is not None:
                    logger.debug(f"  World PE / GASA: Using DA3-NESTED estimated poses -> world-frame pointmaps")
                else:
                    logger.debug(f"  World PE / GASA: Using camera-frame pointmaps (train/eval consistent)")
                if sheaf_loss_fn is not None:
                    if gt_extrinsics is not None:
                        logger.debug(f"  Sheaf loss: Using GT extrinsics -> world-frame pointmaps")
                    elif args.no_gt_poses and cached_da3_extrinsics is not None:
                        logger.debug(f"  Sheaf loss: Using DA3-NESTED estimated poses (calibration-free mode)")
                    elif model.da3_has_pose_estimation if hasattr(model, 'da3_has_pose_estimation') else False:
                        logger.debug(f"  Sheaf loss: Using DA3-estimated poses -> world-frame pointmaps")
                    else:
                        logger.debug(f"  Sheaf loss: No world-frame poses available -> camera-frame (less effective)")

            # Apply spatial augmentation if enabled (adds "nearest", "leftmost", etc.)
            # GT-aware mode uses actual mask positions; otherwise random qualifiers
            if gt_aware_spatial is not None:
                # GT-AWARE SPATIAL AUGMENTATION
                # Uses spatial_context from dataloader to determine qualifiers
                augmented_prompts = []
                augmented_spatial_indices = []

                # Get spatial contexts (may be list, tuple, or None depending on batching)
                spatial_contexts_raw = batch.get('spatial_context', None)

                # Handle different batching scenarios
                if spatial_contexts_raw is None:
                    spatial_contexts = [None] * len(prompts)
                elif isinstance(spatial_contexts_raw, (list, tuple)):
                    spatial_contexts = list(spatial_contexts_raw)
                    # Pad with None if needed
                    while len(spatial_contexts) < len(prompts):
                        spatial_contexts.append(None)
                else:
                    # Single context (shouldn't happen in batched mode, but handle it)
                    spatial_contexts = [spatial_contexts_raw] + [None] * (len(prompts) - 1)

                for b_idx, p in enumerate(prompts):
                    ctx = spatial_contexts[b_idx] if b_idx < len(spatial_contexts) else None

                    # Augment with GT-aware method
                    aug_prompt, qualifier_type, spatial_idx = gt_aware_spatial.augment(p, ctx)
                    augmented_prompts.append(aug_prompt)
                    augmented_spatial_indices.append(spatial_idx)

                prompts = augmented_prompts

                # Log stats periodically (end of first epoch)
                if batch_idx == 0 and epoch > start_epoch and ddp.is_main:
                    logger.debug(f"  [Spatial] {gt_aware_spatial.get_stats_summary()}")

            elif spatial_augmentor is not None:
                # Random spatial augmentation
                spatial_qualifiers = ['nearest', 'farthest', 'leftmost', 'rightmost', 'topmost', 'bottommost']
                augmented_prompts = []

                # If multi-instance-only, check if this label appears multiple times in batch
                # (approximation: in real scenes, multi-instance objects span multiple frames)
                if args.spatial_multi_instance_only:
                    label_counts = {}
                    for p in prompts:
                        label_counts[p] = label_counts.get(p, 0) + 1

                for p in prompts:
                    should_augment = random.random() < args.spatial_augment_prob

                    # Skip augmentation if multi-instance-only and this is a singleton
                    if args.spatial_multi_instance_only and label_counts.get(p, 1) < 2:
                        should_augment = False

                    if should_augment:
                        qual = random.choice(spatial_qualifiers)
                        augmented_prompts.append(f"{qual} {p}")
                    else:
                        augmented_prompts.append(p)
                prompts = augmented_prompts

            # Parse spatial qualifiers from prompts and create index tensor
            # This is used for spatial token conditioning
            spatial_qualifier_idx = None
            if args.use_spatial_tokens or args.use_spatial_points:
                sq_indices = []
                base_prompts = []
                for p in prompts:
                    qualifier_type, base = parse_spatial_qualifier(p)
                    sq_indices.append(get_spatial_qualifier_idx(qualifier_type))
                    base_prompts.append(base)
                spatial_qualifier_idx = torch.tensor(sq_indices, device=device, dtype=torch.long)
                # Use base prompts for text encoding (without spatial qualifiers)
                # The spatial qualifier is handled via spatial tokens or pseudo-points
                if args.use_spatial_tokens:
                    prompts = base_prompts  # Strip spatial qualifiers for cleaner text embedding

            # Initialize per-batch metric accumulators (passed into forward helpers)
            batch_iou_tensor = torch.tensor(0.0, device=device)
            batch_macc_tensor = torch.tensor(0.0, device=device)
            batch_recall_tensor = torch.tensor(0.0, device=device)
            batch_sheaf_loss_tensor = torch.tensor(0.0, device=device)

            # Use no_sync for view loop to avoid DDP gradient sync per view
            sync_context = model.no_sync if ddp.is_distributed and hasattr(model, 'no_sync') else nullcontext

            if args.cross_view:
                (accumulated_loss, valid, batch_loss_tensor, batch_iou_tensor, batch_macc_tensor,
                 batch_recall_tensor, batch_sheaf_loss_tensor, last_vis_data) = _forward_cross_view(
                    model, base_model, images, gt_masks, prompts, batch, args, device, ddp,
                    N, B, gt_extrinsics, gt_intrinsics, intrinsics_orig_hw,
                    cached_depth, cached_da3_extrinsics, cached_da3_intrinsics,
                    spatial_qualifier_idx, epoch, start_epoch, batch_idx,
                    cat_metrics, epoch_centroid_errors,
                    batch_iou_tensor, batch_macc_tensor, batch_recall_tensor,
                    batch_sheaf_loss_tensor)

            elif args.batch_views:
                (accumulated_loss, valid, batch_loss_tensor, batch_iou_tensor, batch_macc_tensor,
                 batch_recall_tensor, batch_sheaf_loss_tensor, last_vis_data) = _forward_batch_views(
                    model, base_model, images, gt_masks, prompts, batch, args, device, ddp,
                    N, B, gt_extrinsics, gt_intrinsics, intrinsics_orig_hw,
                    cached_depth, cached_da3_extrinsics, cached_da3_intrinsics,
                    spatial_qualifier_idx, epoch, start_epoch, batch_idx,
                    cat_metrics, epoch_centroid_errors,
                    batch_iou_tensor, batch_macc_tensor, batch_recall_tensor,
                    batch_sheaf_loss_tensor)

            else:
                (accumulated_loss, valid, batch_loss_tensor, batch_iou_tensor, batch_macc_tensor,
                 batch_recall_tensor, batch_sheaf_loss_tensor, sheaf_preds, sheaf_pointmaps,
                 sheaf_embeddings, last_vis_data) = _forward_sequential(
                    model, base_model, images, gt_masks, prompts, batch, args, device, ddp,
                    N, B, N, gt_extrinsics, gt_intrinsics, intrinsics_orig_hw,
                    cached_depth, cached_da3_extrinsics, cached_da3_intrinsics,
                    spatial_qualifier_idx, epoch, start_epoch, batch_idx,
                    sheaf_loss_fn, feature_sheaf_loss_fn, cat_metrics, epoch_centroid_errors,
                    gt_aware_spatial, sync_context,
                    batch_iou_tensor, batch_macc_tensor, batch_recall_tensor,
                    batch_sheaf_loss_tensor)

            # Single backward per batch (outside no_sync, so DDP syncs gradients here)
            # All ranks must call backward for DDP gradient sync
            # If some ranks skip backward() while others call it, NCCL will hang waiting for sync.
            if accumulated_loss is None:
                # Create dummy zero loss CONNECTED TO MODEL to ensure gradient sync
                # IMPORTANT: With find_unused_parameters=False, we must flow gradients through
                # model params or DDP will hang waiting for gradient hooks that never fire.
                # Sum all trainable params with 0 multiplier to connect graph without affecting gradients.
                dummy_loss = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
                accumulated_loss = dummy_loss if dummy_loss != 0 else torch.tensor(0.0, device=device, requires_grad=True)
            scaler.scale(accumulated_loss).backward()
            if accumulated_loss.item() > 0:  # Only count valid accumulations
                accum_valid += 1

            if (batch_idx + 1) % args.grad_accum == 0 and accum_valid > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_valid = 0

            if valid > 0:
                # Convert tensors to floats only once per batch (single GPU-CPU sync per batch, not per view)
                batch_loss = batch_loss_tensor.item() / valid
                batch_iou = batch_iou_tensor.item() / valid
                batch_macc = batch_macc_tensor.item() / valid
                batch_recall = batch_recall_tensor.item() / valid
                batch_sheaf_loss = batch_sheaf_loss_tensor.item()

                epoch_loss += batch_loss
                epoch_iou += batch_iou
                epoch_macc += batch_macc
                epoch_recall += batch_recall
                epoch_sheaf_loss += batch_sheaf_loss
                num_samples += 1

                # Batch-level visualization (if enabled, main only)
                if args.vis_every_batches > 0 and (batch_idx + 1) % args.vis_every_batches == 0 and last_vis_data and ddp.is_main:
                    try:
                        visualize_predictions(run_dir, f"e{epoch+1}_b{batch_idx+1}", last_vis_data['images'],
                                              last_vis_data['gt_masks'], last_vis_data['outputs'], last_vis_data['prompts'])
                    except Exception as e:
                        logger.debug(f"  Batch vis failed: {e}")

            if num_samples > 0:
                cur_miou = cat_metrics.get_miou()
                pbar.set_postfix({'loss': f'{epoch_loss/num_samples:.4f}', 'IoU': f'{100*epoch_iou/num_samples:.1f}%', 'mIoU': f'{100*cur_miou:.1f}%', 'mAcc': f'{100*epoch_macc/num_samples:.1f}%'})

        (avg_loss, avg_iou, avg_macc, avg_recall, avg_sheaf_loss, miou, num_cats, num_samples,
         acc_5cm, acc_10cm, acc_50cm, mean_dist_error) = _finalize_epoch(
            epoch, args, ddp, device, optimizer, cat_metrics, epoch_loss, epoch_iou,
            epoch_macc, epoch_recall, epoch_sheaf_loss, epoch_centroid_errors,
            num_samples, start_epoch, base_model, last_vis_data, run_dir)
        current_lr = optimizer.param_groups[0]['lr']

        if num_samples > 0:
            val_metrics, best_val_miou = _run_validation_and_save(
                model, val_dataloader, base_model, optimizer, scheduler, scaler,
                lora_manager, args, device, ddp, epoch, best_iou, best_val_miou, checkpoint_dir)

            # Save best checkpoint based on training IoU (if not using val or val not run this epoch)
            if not args.save_best_val or val_metrics is None:
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    if ddp.is_main:
                        ckpt = _save_checkpoint(base_model, optimizer, scheduler, scaler, lora_manager,
                                                args, epoch, best_iou, best_val_miou, checkpoint_dir)
                        torch.save(ckpt, checkpoint_dir / 'best.pt')
                        logger.info(f"  -> New best train IoU! Saved to {checkpoint_dir / 'best.pt'}")
            else:
                if avg_iou > best_iou:
                    best_iou = avg_iou

            # Always save last.pt for resume (even if not best)
            if ddp.is_main:
                ckpt = _save_checkpoint(base_model, optimizer, scheduler, scaler, lora_manager,
                                        args, epoch, best_iou, best_val_miou, checkpoint_dir,
                                        include_rng=True)
                torch.save(ckpt, checkpoint_dir / 'last.pt')

            # Save summary every epoch (so we don't lose progress if interrupted)
            if ddp.is_main:
                cat_summary = cat_metrics.summary()
                epoch_summary = {
                    'best_iou': best_iou,
                    'best_val_miou': best_val_miou,
                    'current_epoch': epoch + 1,
                    'total_epochs': args.epochs,
                    'current_loss': avg_loss,
                    'current_iou': avg_iou,
                    'current_miou': miou,
                    'current_mAcc': avg_macc,
                    'num_categories': cat_summary['num_categories'],
                    'per_category_iou': cat_summary['per_category_iou'],
                }
                if val_metrics is not None:
                    epoch_summary['val_loss'] = val_metrics['val_loss']
                    epoch_summary['val_iou'] = val_metrics['val_iou']
                    epoch_summary['val_miou'] = val_metrics['val_miou']
                    epoch_summary['val_mAcc'] = val_metrics['val_mAcc']
                    epoch_summary['val_num_categories'] = val_metrics['val_num_categories']
                if (args.use_centroid_head or args.eval_localization) and len(epoch_centroid_errors) > 0:
                    epoch_summary['acc_5cm'] = acc_5cm
                    epoch_summary['acc_10cm'] = acc_10cm
                    epoch_summary['acc_50cm'] = acc_50cm
                    epoch_summary['mean_dist_error_m'] = mean_dist_error
                with open(run_dir / 'summary.json', 'w') as f:
                    json.dump(epoch_summary, f, indent=2)
                with open(checkpoint_dir / 'summary.json', 'w') as f:
                    json.dump(epoch_summary, f, indent=2)

                history_entry = {
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'iou': avg_iou,
                    'miou': miou,
                    'mAcc': avg_macc,
                    'recall': avg_recall,
                    'lr': current_lr,
                    'num_categories': cat_summary['num_categories'],
                }
                if val_metrics is not None:
                    history_entry['val_loss'] = val_metrics['val_loss']
                    history_entry['val_iou'] = val_metrics['val_iou']
                    history_entry['val_miou'] = val_metrics['val_miou']
                    history_entry['val_mAcc'] = val_metrics['val_mAcc']
                    history_entry['val_num_categories'] = val_metrics['val_num_categories']
                if args.use_sheaf_loss:
                    history_entry['sheaf_loss'] = avg_sheaf_loss
                if (args.use_centroid_head or args.eval_localization) and len(epoch_centroid_errors) > 0:
                    history_entry['acc_5cm'] = acc_5cm
                    history_entry['acc_10cm'] = acc_10cm
                    history_entry['acc_50cm'] = acc_50cm
                    history_entry['mean_dist_error_m'] = mean_dist_error

                existing_idx = None
                for i, h in enumerate(history):
                    if h.get('epoch') == epoch + 1:
                        existing_idx = i
                        break
                if existing_idx is not None:
                    history[existing_idx] = history_entry
                else:
                    history.append(history_entry)
                with open(run_dir / 'history.json', 'w') as f:
                    json.dump(history, f, indent=2)

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Free memory between epochs
        gc.collect()
        torch.cuda.empty_cache()

        # RAM check - stop gracefully if available RAM drops below threshold
        if args.min_ram_gb > 0:
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            # In DDP, rank 0 decides and broadcasts to all ranks
            should_stop = torch.tensor([available_ram_gb < args.min_ram_gb], dtype=torch.bool, device=device)
            if ddp.world_size > 1:
                torch.distributed.broadcast(should_stop, src=0)
            if should_stop.item():
                logger.info(f"[Low RAM] Available: {available_ram_gb:.1f}GB < threshold {args.min_ram_gb}GB")
                logger.info(f"[Low RAM] Stopping gracefully. Checkpoint saved at epoch {epoch+1}.")
                logger.info(f"[Low RAM] Resume with: --resume {checkpoint_dir}")
                break

        # IMPORTANT: Add barrier to ensure all ranks complete the epoch before starting the next one
        # This prevents race conditions where some ranks race ahead while others are still doing I/O
        # The barrier was previously removed for "faster epoch transitions" but this caused deadlocks
        ddp.barrier()

    # Get final category metrics
    final_miou = cat_metrics.get_miou()

    # Save final summary (main only) - overwrites per-epoch summary with final stats
    if ddp.is_main:
        cat_summary = cat_metrics.summary()
        summary = {
            'best_iou': best_iou,
            'best_val_miou': best_val_miou,
            'final_loss': avg_loss if num_samples > 0 else 0.0,
            'final_iou': avg_iou if num_samples > 0 else 0.0,
            'final_miou': cat_summary['mIoU'],
            # mAcc = mean class accuracy (FG_acc + BG_acc) / 2 - balanced metric
            'final_mAcc': avg_macc if num_samples > 0 else 0.0,
            # Mean class recall = TP/(TP+FN) per sample
            'final_mean_class_recall': avg_recall if num_samples > 0 else 0.0,
            'num_categories': cat_summary['num_categories'],
            'per_category_iou': cat_summary['per_category_iou'],
            'epochs': args.epochs,
            'world_size': ddp.world_size,
            'status': 'completed',
        }
        with open(run_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        with open(checkpoint_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Print per-category breakdown
        if cat_summary['per_category_iou']:
            sorted_cats = sorted(cat_summary['per_category_iou'].items(), key=lambda x: x[1], reverse=True)
            logger.info("Per-category IoU (top 10):")
            for cat, iou in sorted_cats[:10]:
                logger.info(f"  {cat}: {100*iou:.1f}%")

    val_str = f", Best Val mIoU: {100*best_val_miou:.1f}%" if best_val_miou > 0 else ""
    print(f"Training complete! Best IoU: {100*best_iou:.1f}%{val_str}, Final mIoU: {100*final_miou:.1f}%")
    print(f"Summary saved to {run_dir / 'summary.json'}")
    print(f"Per-epoch history saved to {run_dir / 'history.json'}")
    ddp.cleanup()


if __name__ == '__main__':
    main()
