"""Dataset-specific evaluation: uCO3D, NVOS, PartImageNet."""
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from triangulang.data.dataset_factory import get_dataset, get_dataset_config
from triangulang.evaluation.visualization import (
    plot_category_iou, plot_summary,
    generate_paper_visualizations, generate_single_object_viz,
)


def _evaluate_uco3d(model, args, device, ddp, data_root, output_dir, viz_dir, total_params, trainable_params, gasa_params):
    from triangulang.data.uco3d_dataset import UCO3DMultiViewDataset
    from triangulang.data.uco3d_factory import create_uco3d_eval_dataset

    # Default to per-frame mode for uCO3D (faster, lower memory, no cross-view attention)
    # Use --view-chunk-size N for chunked processing with cross-view attention
    per_frame_default = False
    if not args.per_frame and args.view_chunk_size == 8:  # Default values = use per_frame
        args.per_frame = True
        per_frame_default = True

    ddp.print(f"\nuCO3D Evaluation")
    ddp.print(f"Using UCO3DMultiViewDataset with:")
    ddp.print(f"  - 50 representative sequences")
    ddp.print(f"  - 50 frames per sequence (uniformly sampled)")
    ddp.print(f"  - Category name as text prompt")
    ddp.print(f"  - Prompt normalization: {args.normalize_prompts}")
    num_seq = args.num_sequences if args.num_sequences else 50
    frames_per_seq = args.frames_per_sequence if args.frames_per_sequence else 50
    ddp.print(f"  - Sequences: {num_seq}" + (" (paper default)" if args.num_sequences is None else ""))
    ddp.print(f"  - Frames/sequence: {frames_per_seq}" + (" (paper default)" if args.frames_per_sequence is None else ""))
    if per_frame_default:
        ddp.print(f"  - Per-frame mode (default for uCO3D, use --view-chunk-size N for cross-view)")
    elif args.per_frame:
        ddp.print(f"  - Per-frame mode (explicit)")
    else:
        ddp.print(f"  - Chunked mode with view-chunk-size={args.view_chunk_size}")
    if getattr(args, 'batch_da3', False):
        da3_cs = args.view_chunk_size if args.view_chunk_size > 0 else 16
        ddp.print(f"  - Batched DA3: enabled (chunk-size={da3_cs})")
        ddp.print(f"    → DA3 processes views together for cross-view depth consistency")
        ddp.print(f"    → Segmentation still runs per-frame (faster eval)")

    # Create evaluation dataset with configurable sequences/frames
    # If --fold is specified, use k-fold CV splitting
    kfold_num_folds = args.num_folds if args.fold is not None else None
    kfold_fold = args.fold

    if kfold_fold is not None:
        ddp.print(f"  - K-fold CV: fold {kfold_fold}/{kfold_num_folds} as validation")

    eval_dataset = create_uco3d_eval_dataset(
        data_root=str(data_root),
        num_views=args.num_frames,  # All frames are views for eval
        image_size=(args.image_size, args.image_size),
        mask_size=(args.mask_size, args.mask_size),
        normalize_prompts=args.normalize_prompts,
        num_sequences=args.num_sequences,
        frames_per_sequence=args.frames_per_sequence,
        seed=args.seed,
        num_folds=kfold_num_folds,
        fold=kfold_fold,
    )

    # Paper viz: collect data during eval
    uco3d_paper_viz_pool = [] if (args.paper_viz and ddp.is_main) else None

    # Run evaluation using dataset
    results = evaluate_with_dataset(
        model=model,
        dataset=eval_dataset,
        device=device,
        ddp=ddp,
        args=args,
        output_dir=output_dir,
        viz_dir=viz_dir if args.visualize else None,
        paper_viz_collector=uco3d_paper_viz_pool,
    )

    # Paper-quality grid visualization (uCO3D path)
    if uco3d_paper_viz_pool and ddp.is_main:
        ddp.print("\nGenerating paper-quality grid visualizations for uCO3D...")
        generate_paper_visualizations(uco3d_paper_viz_pool, args, output_dir)

    # Cross-fold analysis (same as ScanNet++ evaluation)
    if ddp.is_main and args.cross_fold:
        per_cat_iou = results.get('per_category_iou', {})
        per_cat_recall = results.get('per_category_recall', {})

        if len(per_cat_iou) >= args.num_folds:
            ddp.print(f"\n{'='*70}")
            ddp.print(f"CROSS-FOLD ANALYSIS ({args.num_folds} folds)")
            ddp.print("="*70)
            ddp.print("Grouping categories into folds for per-group performance analysis\n")

            # Sort categories alphabetically for deterministic fold assignment
            sorted_categories = sorted(per_cat_iou.keys())
            fold_size = len(sorted_categories) // args.num_folds

            fold_results = []
            for fold_idx in range(args.num_folds):
                start_idx = fold_idx * fold_size
                if fold_idx == args.num_folds - 1:
                    end_idx = len(sorted_categories)
                else:
                    end_idx = start_idx + fold_size

                fold_categories = sorted_categories[start_idx:end_idx]
                fold_ious = [per_cat_iou[cat] for cat in fold_categories]
                fold_recalls = [per_cat_recall.get(cat, 0) for cat in fold_categories]

                fold_mean_iou = np.mean(fold_ious)
                fold_mean_recall = np.mean(fold_recalls)

                fold_results.append({
                    'fold_id': fold_idx,
                    'categories': fold_categories,
                    'num_categories': len(fold_categories),
                    'mean_iou': float(fold_mean_iou),
                    'mean_recall': float(fold_mean_recall),
                })

                ddp.print(f"Fold {fold_idx + 1}/{args.num_folds}: {len(fold_categories)} categories")
                ddp.print(f"  Mean IoU:    {100*fold_mean_iou:.2f}%")
                ddp.print(f"  Mean Recall: {100*fold_mean_recall:.2f}%")
                ddp.print(f"  Categories:  {', '.join(fold_categories[:5])}" +
                          (f", ... (+{len(fold_categories)-5} more)" if len(fold_categories) > 5 else ""))
                ddp.print()

            # Find best/worst folds (matching ScanNet++ format)
            best_fold = max(fold_results, key=lambda x: x['mean_iou'])
            worst_fold = min(fold_results, key=lambda x: x['mean_iou'])

            ddp.print(f"Best fold: Fold {best_fold['fold_id'] + 1} (mIoU={100*best_fold['mean_iou']:.2f}%)")
            ddp.print(f"Worst fold: Fold {worst_fold['fold_id'] + 1} (mIoU={100*worst_fold['mean_iou']:.2f}%)")
            ddp.print(f"Performance gap: {100*(best_fold['mean_iou'] - worst_fold['mean_iou']):.2f}%")
            ddp.print("="*70)

            results['fold_results'] = fold_results

    # Save results and config
    if ddp.is_main:
        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        ddp.print(f"\nResults saved to: {results_file}")

        # Save config (matching ScanNet++ format)
        config = {
            'dataset': 'uco3d',
            'checkpoint': args.checkpoint,
            'num_samples': results.get('num_samples', 0),
            'num_frames': args.num_frames,
            'image_size': args.image_size,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
            'prompt_type': args.prompt_type,
            'per_frame': args.per_frame,
            'view_chunk_size': args.view_chunk_size,
            'batch_da3': getattr(args, 'batch_da3', False),
            'num_sequences': args.num_sequences,
            'frames_per_sequence': args.frames_per_sequence,
            'normalize_prompts': args.normalize_prompts,
            'mask_selection': getattr(model, 'mask_selection', None),
            'semantic_union': args.semantic_union,
            'cross_fold': args.cross_fold,
            'num_folds': args.num_folds if args.cross_fold else None,
            'fold': getattr(args, 'fold', None),
            'distributed': ddp.is_distributed,
            'world_size': ddp.world_size,
            'model_params': {
                'total': total_params,
                'trainable': trainable_params,
                'gasa_decoder': gasa_params,
            },
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        ddp.print(f"Config saved to: {output_dir / 'config.json'}")





def _evaluate_nvos(model, args, device, ddp, data_root, output_dir, viz_dir):
    from triangulang.data.nvos_dataset import NVOSDataset, NVOS_PROMPTS

    ddp.print(f"\nNVOS Evaluation")
    if args.baseline_sam3:
        ddp.print(f"  Mode: BASELINE SAM3 (native decoder, no GASA/depth/cross-view)")
    ddp.print(f"  7 scenes from LLFF (orchid excluded)")
    ddp.print(f"  Metric: IoU on target frame (single-frame, text-only)")

    # Resolve image size
    resolution = args.image_size or model.resolution
    mask_res = args.mask_size or resolution

    eval_dataset = NVOSDataset(
        data_root=str(data_root),
        split='all',
        views_per_sample=2,
        image_size=(resolution, resolution),
        mask_size=(mask_res, mask_res),
        use_language=True,
    )

    if len(eval_dataset) == 0:
        ddp.print("ERROR: No NVOS scenes found. Check --data-root.")
        return

    dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=0,
    )

    all_ious = []
    scene_ious = defaultdict(list)

    pbar = tqdm(dataloader, desc="NVOS Eval", disable=not ddp.is_main)

    for batch_idx, batch in enumerate(pbar):
        try:
            images = batch['images'].to(device).squeeze(0)      # [2, 3, H, W]
            gt_masks = batch['gt_masks'].to(device).squeeze(0)  # [2, H, W]
            prompt = batch['prompt'][0] if isinstance(batch['prompt'], list) else batch['prompt']
            scene_id = batch['scene_id'][0] if isinstance(batch['scene_id'], list) else batch['scene_id']

            # Target frame (view 1) has GT mask; reference (view 0) has scribbles only
            target_gt = gt_masks[1]  # [H, W]
            if target_gt.sum() < 1:
                continue

            target_img = images[1:2]  # [1, 3, H, W]

            # Parse spatial qualifier from prompt (e.g. "nearest fern plant")
            sq_type, base_prompt = parse_spatial_qualifier(prompt)
            sq_idx = get_spatial_qualifier_idx(sq_type)
            sq_tensor = torch.tensor([sq_idx], device=device, dtype=torch.long) if sq_idx > 0 else None

            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    outputs = model.forward(
                        images=target_img,
                        text_prompts=[base_prompt],
                        gt_masks=None,
                        gt_intrinsics=None,
                        spatial_qualifier_idx=sq_tensor,
                    )

            # Extract prediction
            pred_mask = outputs.get('pred_masks')
            if pred_mask is None:
                continue
            if pred_mask.dim() == 4:
                pred_mask = pred_mask[:, 0]
            pred_mask = pred_mask.squeeze(0)  # [H, W]

            # Resize to GT mask size if needed
            if pred_mask.shape != target_gt.shape:
                pred_mask = F.interpolate(
                    pred_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=target_gt.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)

            # CRF / morphological post-processing (SO spatial path)
            if use_crf:
                from triangulang.utils.crf_postprocess import morphological_smooth
                mask_binary = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy().astype(np.float32)
                refined = morphological_smooth(mask_binary, kernel_size=7)
                pred_mask = torch.from_numpy(refined * 10.0 - 5.0).to(pred_mask.device)

            # Compute IoU
            gt_binary = (target_gt > 0.5).float()
            relevancy = torch.sigmoid(pred_mask)
            pred_binary = (relevancy > 0.5).float()
            intersection = (pred_binary * gt_binary).sum()
            union = pred_binary.sum() + gt_binary.sum() - intersection
            iou = (intersection / (union + 1e-6)).item()

            all_ious.append(iou)
            scene_ious[scene_id].append(iou)

            pbar.set_postfix({'mIoU': f'{np.mean(all_ious)*100:.1f}%'})

            # Visualization
            if args.visualize and ddp.is_main and batch_idx < args.viz_samples:
                nvos_viz_dir = viz_dir / 'nvos'
                nvos_viz_dir.mkdir(parents=True, exist_ok=True)

                rgb_for_viz = F.interpolate(
                    target_img[:, :3], size=target_gt.shape[-2:],
                    mode='bilinear', align_corners=False
                ).squeeze(0).permute(1, 2, 0).cpu().numpy()
                rgb_for_viz = (rgb_for_viz * 255).clip(0, 255).astype(np.uint8)

                gt_np = gt_binary.cpu().numpy()
                pred_np = pred_binary.cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb_for_viz)
                axes[0].set_title(f'{scene_id}: "{prompt}"')
                axes[1].imshow(gt_np, cmap='gray')
                axes[1].set_title('GT Mask')
                axes[2].imshow(pred_np, cmap='gray')
                axes[2].set_title(f'Pred (IoU={iou*100:.1f}%)')
                for ax in axes:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(nvos_viz_dir / f'{scene_id}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

        except Exception as e:
            if ddp.is_main:
                import traceback
                print(f"NVOS error on {batch.get('scene_id', '?')}: {e}")
                traceback.print_exc()
            continue

    # Aggregate results
    if ddp.is_main:
        mean_iou = np.mean(all_ious) * 100 if all_ious else 0.0
        per_scene = {k: np.mean(v) * 100 for k, v in scene_ious.items()}
        scene_mean_iou = np.mean(list(per_scene.values())) if per_scene else 0.0

        ddp.print(f"\n{'='*50}")
        ddp.print(f"NVOS Results ({len(all_ious)} samples, {len(scene_ious)} scenes)")
        ddp.print(f"{'='*50}")
        ddp.print(f"  Sample mIoU: {mean_iou:.2f}%")
        ddp.print(f"  Scene  mIoU: {scene_mean_iou:.2f}%")
        ddp.print(f"\n  Per-scene IoU:")
        for scene, siou in sorted(per_scene.items()):
            ddp.print(f"    {scene:20s}: {siou:.2f}%")

        results = {
            'dataset': 'nvos',
            'sample_miou': mean_iou,
            'scene_miou': scene_mean_iou,
            'num_samples': len(all_ious),
            'num_scenes': len(scene_ious),
            'per_scene_iou': per_scene,
        }

        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        ddp.print(f"\nResults saved to: {results_file}")

        if args.visualize:
            ddp.print(f"Visualizations saved to: {viz_dir / 'nvos'}")




def _evaluate_partimagenet(model, args, device, ddp, data_root, output_dir, viz_dir):
    from triangulang.data.partimagenet_dataset import PartImageNetDataset, create_partimagenet_eval_dataset

    ddp.print(f"\nPartImageNet Evaluation")
    ddp.print(f"Using PartImageNetDataset with:")
    ddp.print(f"  - Split: {args.split}")
    ddp.print(f"  - Part query mode: all (evaluates all parts)")
    ddp.print(f"  - 11 super-categories with part-level annotations")

    # Create eval dataset
    eval_dataset = create_partimagenet_eval_dataset(
        data_root=str(data_root),
        split=args.split,
        image_size=(args.image_size, args.image_size),
        mask_size=(args.mask_size, args.mask_size),
        max_samples=args.max_scenes,  # Use max_scenes to limit samples
    )

    ddp.print(f"  - Samples: {len(eval_dataset)}")

    # Run evaluation using dataset (same as uCO3D)
    results = evaluate_with_dataset(
        model=model,
        dataset=eval_dataset,
        device=device,
        ddp=ddp,
        args=args,
        output_dir=output_dir,
        viz_dir=viz_dir if args.visualize else None,
    )

    # Save results
    if ddp.is_main:
        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        ddp.print(f"\nResults saved to: {results_file}")


