"""ScanNet++ scene-based evaluation."""
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from triangulang.evaluation.eval_utils import compute_spatial_gt
from triangulang.evaluation.data_loading import (
    load_gt_centroids, load_gt_poses_for_scene,
)
from triangulang.evaluation.visualization import (
    MASK_COLORS, create_comparison_grid, create_multi_object_viz,
    plot_category_iou, plot_scene_metrics, plot_summary,
    generate_paper_visualizations, generate_single_object_viz,
)
from triangulang.utils.scannetpp_loader import SCANNETPP_SKIP_LABELS
from triangulang.utils.spatial_reasoning import parse_spatial_qualifier
from triangulang.evaluation.eval_scene import evaluate_scene
from triangulang.evaluation.data_loading import load_scene_data


def _evaluate_scannetpp(model, args, device, ddp, data_root, output_dir, viz_dir, total_params, trainable_params, gasa_params):
    if 'train' in args.split:
        semantics_root = data_root / 'semantics_2d_train'
    else:
        # Prefer v2 (all frames) over v1 (GT-subset only)
        semantics_root = data_root / 'semantics_2d_val_v2'
        if not semantics_root.exists():
            semantics_root = data_root / 'semantics_2d_val'

    split_file = data_root / 'splits' / f'{args.split}.txt'

    if split_file.exists():
        with open(split_file) as f:
            split_scene_ids = [line.strip() for line in f if line.strip()]
    else:
        split_scene_ids = None

    available_semantics = {d.name for d in semantics_root.iterdir() if d.is_dir()}

    if split_scene_ids:
        scene_ids = [s for s in split_scene_ids if s in available_semantics]
    else:
        scene_ids = sorted(available_semantics)

    ddp.print(f"Found {len(scene_ids)} scenes with semantics_2d data")

    # Single-object-viz mode: focus on specific scene(s)
    # Track if we already distributed scenes (to avoid double-distribution)
    scenes_already_distributed = False

    # Filter to specific scene(s) if --scene is provided (works for ScanNet++ too)
    if args.scene and not args.single_object_viz:
        requested_scenes = args.scene if isinstance(args.scene, list) else [args.scene]
        scene_ids = [s for s in scene_ids if s in requested_scenes]
        if not scene_ids:
            ddp.print(f"ERROR: --scene {args.scene} not found. Available: {sorted(available_semantics)[:20]}")
            return
        ddp.print(f"Filtering to scene(s): {scene_ids}")

    if args.single_object_viz:
        num_viz_scenes = getattr(args, 'viz_num_scenes', 1)

        if args.viz_scene:
            # Specific scene requested
            if args.viz_scene in scene_ids:
                scene_ids = [args.viz_scene]
                ddp.print(f"\n[single-object-viz] Focusing on scene: {args.viz_scene}")
            else:
                ddp.print(f"ERROR: --viz-scene '{args.viz_scene}' not found in available scenes")
                ddp.print(f"Available scenes (first 20): {scene_ids[:20]}")
                return
        elif args.viz_random_scene:
            # Pick random scenes - with DDP, each rank gets different scenes
            random.seed(args.seed)  # Consistent base seed
            shuffled = scene_ids.copy()
            random.shuffle(shuffled)

            if ddp.is_distributed:
                # With DDP: distribute scenes across ranks
                # e.g., 8 scenes on 8 GPUs = 1 scene per GPU
                total_scenes = min(num_viz_scenes, len(shuffled))
                scenes_per_rank = max(1, total_scenes // ddp.world_size)
                start_idx = ddp.rank * scenes_per_rank
                end_idx = start_idx + scenes_per_rank
                scene_ids = shuffled[start_idx:end_idx]
                ddp.print(f"\n[single-object-viz] Rank {ddp.rank}: {len(scene_ids)} random scene(s): {scene_ids}")
                scenes_already_distributed = True  # Don't distribute again below
            else:
                # Single GPU: take first N random scenes
                scene_ids = shuffled[:num_viz_scenes]
                ddp.print(f"\n[single-object-viz] {len(scene_ids)} random scene(s): {scene_ids}")
        else:
            # Default: use first N scenes alphabetically
            scene_ids = scene_ids[:num_viz_scenes]
            ddp.print(f"\n[single-object-viz] Using first {len(scene_ids)} scene(s): {scene_ids}")
            ddp.print(f"    Tip: Use --viz-random-scene for random selection")
            ddp.print(f"    Tip: Use --viz-num-scenes N for multiple scenes")
    elif args.max_scenes:
        scene_ids = scene_ids[:args.max_scenes]

    # Split scenes across ranks for distributed evaluation
    # (Skip if single-object-viz already distributed scenes)
    total_scenes = len(scene_ids)
    if ddp.is_distributed and not scenes_already_distributed:
        # Each rank gets a subset of scenes
        scenes_per_rank = (total_scenes + ddp.world_size - 1) // ddp.world_size
        start_idx = ddp.rank * scenes_per_rank
        end_idx = min(start_idx + scenes_per_rank, total_scenes)
        scene_ids = scene_ids[start_idx:end_idx]
        ddp.print(f"Rank {ddp.rank}: evaluating scenes {start_idx}-{end_idx} ({len(scene_ids)} scenes)")

    ddp.print(f"\nEvaluating on {total_scenes} scenes from {args.split}...")
    if args.single_prompt:
        ddp.print(f"Mode: SINGLE-PROMPT (prompt view {args.prompt_view}, measure other views)")
        ddp.print(f"  This is our key differentiator from MV-SAM!")
        ddp.print(f"  Uses cross-view attention to propagate understanding.")
    else:
        ddp.print(f"Mode: MULTI-PROMPT (prompt all views)")
    ddp.print(f"Protocol: {args.num_frames} frames, {args.objects_per_scene} objects/scene, ≥{args.min_mask_coverage*100:.2g}% coverage")
    ddp.print(f"Frame sampling: {args.eval_sampling}")

    # Category filtering: collect categories across all scenes and filter rare ones
    allowed_categories = None

    # Spatial query parsing: maps original prompt -> (qualifier, base_prompt)
    # e.g., "leftmost towel" -> ("leftmost", "towel")
    spatial_query_map = {}

    # If --custom-prompts is specified for ScanNet++, use those as allowed categories
    # Parse spatial qualifiers and use base prompts for filtering
    if args.custom_prompts:
        base_prompts = set()
        for prompt in args.custom_prompts:
            qualifier, base = parse_spatial_query(prompt)
            spatial_query_map[prompt.lower()] = (qualifier, base)
            base_prompts.add(base.lower())
            if qualifier:
                ddp.print(f"  Spatial query: '{prompt}' -> qualifier='{qualifier}', base='{base}'")
        allowed_categories = base_prompts
        ddp.print(f"Custom prompts filter: {args.custom_prompts}")
        ddp.print(f"  Base categories for matching: {sorted(base_prompts)}")

    if args.min_category_samples > 1:
        ddp.print(f"Collecting category statistics for filtering (min {args.min_category_samples} samples)...")
        from collections import Counter
        category_counts = Counter()
        for scene_id in scene_ids:
            anno_path = data_root / 'data' / scene_id / 'scans' / 'segments_anno.json'
            if anno_path.exists():
                with open(anno_path) as f:
                    anno = json.load(f)
                for group in anno.get('segGroups', []):
                    label = normalize_label(group.get('label', '')).lower()
                    if label:
                        category_counts[label] += 1

        allowed_categories = {cat for cat, count in category_counts.items()
                              if count >= args.min_category_samples}
        rare_categories = {cat for cat, count in category_counts.items()
                          if count < args.min_category_samples}
        ddp.print(f"  Found {len(category_counts)} categories, keeping {len(allowed_categories)} "
                  f"(filtered {len(rare_categories)} with < {args.min_category_samples} samples)")
        if rare_categories:
            examples = sorted(rare_categories)[:5]
            ddp.print(f"  Filtered examples: {examples}")

    ddp.print(f"Prompt type: {args.prompt_type}")
    if args.prompt_type in ['text_point', 'text_box_point', 'all']:
        if args.sparse_prompts:
            ddp.print(f"  MV-SAM SPARSE PROMPTING: {args.num_pos_points} pos + {args.num_neg_points} neg = {args.num_pos_points + args.num_neg_points} points TOTAL")
            ddp.print(f"  Distributed across {args.num_prompted_frames} frames (out of {args.num_frames})")
            ddp.print(f"  Other frames receive text-only prompts (global semantic context)")
        else:
            ddp.print(f"  DENSE PROMPTING: {args.num_pos_points} pos + {args.num_neg_points} neg points PER FRAME")
            ddp.print(f"  WARNING: This is NOT the MV-SAM protocol. Use --sparse-prompts for fair comparison.")
    if args.consistency_metric:
        ddp.print(f"Computing cross-view consistency (3D centroid variance)")
    if args.visualize:
        ddp.print(f"Saving visualizations to: {viz_dir}")

    # Setup DA3 cache for depth and poses (one cache provides both)
    # Auto-select val_allframes cache when evaluating on val split
    da3_cache_dir = None
    if args.da3_nested_cache:
        da3_cache_dir = data_root / args.da3_nested_cache
        # For val splits, prefer the allframes cache (more frames for Procrustes alignment)
        if 'val' in args.split and not da3_cache_dir.name.endswith('_val_allframes'):
            val_allframes_dir = data_root / f"{args.da3_nested_cache}_val_allframes"
            if val_allframes_dir.exists():
                ddp.print(f"\n📦 Auto-selecting val_allframes cache for val split")
                da3_cache_dir = val_allframes_dir
        if not da3_cache_dir.exists():
            ddp.print(f"WARNING: DA3 cache not found at {da3_cache_dir}")
            ddp.print("         Run 'python scripts/preprocess_da3_nested.py' first.")
            ddp.print("         DA3 will run live (slower) and estimated poses unavailable.")
            da3_cache_dir = None
            if args.use_estimated_poses:
                ddp.print("         Falling back to camera-frame evaluation.")
                args.use_estimated_poses = False
        else:
            ddp.print(f"\n📦 DA3 CACHE: {da3_cache_dir}")
            ddp.print(f"   Provides: cached depth + estimated poses")
    else:
        if args.baseline_sam3:
            ddp.print(f"\n⚡ DA3 SKIPPED: Baseline SAM3 mode (no depth needed)")
        else:
            ddp.print(f"\n⚡ DA3 LIVE: No --da3-nested-cache specified")
            ddp.print(f"   DA3 will run live (slower), no estimated poses available")
        if args.use_estimated_poses:
            ddp.print("   Falling back to camera-frame evaluation.")
            args.use_estimated_poses = False

    # Load GT data for Procrustes evaluation
    gt_centroids_cache = {}
    gt_poses_cache = {}
    if args.procrustes:
        ddp.print(f"\n📐 PROCRUSTES EVALUATION: Enabled")
        ddp.print(f"   Scale estimation: {'7-DoF (with scale)' if args.procrustes_with_scale else '6-DoF (no scale)'}")
        gt_centroids_cache = load_gt_centroids(data_root)
        if gt_centroids_cache:
            # Filter to evaluated scenes if scene_ids available
            if scene_ids:
                gt_centroids_cache = {k: v for k, v in gt_centroids_cache.items() if k in set(scene_ids)}
            ddp.print(f"   Loaded GT centroids for {len(gt_centroids_cache)} scenes")
        else:
            ddp.print(f"   WARNING: centroid_cache.json not found - Procrustes disabled")
            args.procrustes = False

    if args.use_estimated_poses:
        ddp.print(f"\n📍 POSE-FREE EVALUATION: Using DA3-NESTED estimated poses")
    elif args.use_world_poses:
        ddp.print(f"\n📍 WORLD-FRAME EVALUATION: Using GT poses from transforms.json")
    else:
        ddp.print(f"\n📍 CAMERA-FRAME EVALUATION: Using identity poses (default)")

    if args.compare_pose_sources:
        ddp.print(f"\n🔬 POSE COMPARISON: Will run both GT and estimated poses")

    # Save config (only on main rank)
    config = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_scenes': total_scenes,
        'num_frames': args.num_frames,
        'objects_per_scene': args.objects_per_scene,
        'image_size': args.image_size,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'single_prompt': args.single_prompt,
        'prompt_view': args.prompt_view if args.single_prompt else None,
        'consistency_metric': args.consistency_metric,
        'prompt_type': args.prompt_type,
        'num_pos_points': args.num_pos_points,
        'num_neg_points': args.num_neg_points,
        'sparse_prompts': args.sparse_prompts,
        'num_prompted_frames': args.num_prompted_frames,
        'semantic_union': args.semantic_union,
        'use_synonyms': args.use_synonyms,
        'synonym_prob': args.synonym_prob if args.use_synonyms else 0.0,
        'cross_fold': args.cross_fold,
        'num_folds': args.num_folds if args.cross_fold else None,
        'distributed': ddp.is_distributed,
        'world_size': ddp.world_size,
        'model_params': {
            'total': total_params,
            'trainable': trainable_params,
            'gasa_decoder': gasa_params,
        },
        # Pose-free evaluation options
        'use_estimated_poses': args.use_estimated_poses,
        'use_world_poses': args.use_world_poses,
        'compare_pose_sources': args.compare_pose_sources,
        'da3_cache_dir': str(da3_cache_dir) if da3_cache_dir else None,
        # Filtering options
        'min_mask_coverage': args.min_mask_coverage,
        'min_category_samples': args.min_category_samples,
        'num_allowed_categories': len(allowed_categories) if allowed_categories else None,
        # Frame sampling
        'eval_sampling': args.eval_sampling,
        # Procrustes evaluation
        'procrustes': args.procrustes,
        'procrustes_with_scale': args.procrustes_with_scale,
    }

    if ddp.is_main:
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # Setup prompt augmentation for synonym robustness testing
    prompt_augmentor = None
    if args.use_synonyms:
        prompt_augmentor = PromptAugmentor(
            use_synonyms=True,
            synonym_prob=args.synonym_prob,
            use_templates=False,  # Keep prompts simple for eval
        )
        ddp.print(f"\n🔀 Synonym augmentation ENABLED (prob={args.synonym_prob})")
        ddp.print("   Testing robustness to prompt variations (e.g., 'tap' → 'faucet')")

    # Evaluate - use different mode based on single_prompt flag
    all_results = []
    all_category_metrics = defaultdict(lambda: {'iou': [], 'oracle_iou': [], 'pixel_acc': [], 'recall': []})

    if args.single_prompt:
        # SINGLE-PROMPT MODE: Multi-view batch evaluation
        print(f"\nSingle-Prompt Propagation Evaluation")
        for scene_id in tqdm(scene_ids, desc="Scenes (single-prompt)"):
            scene_path = data_root / 'data' / scene_id
            semantics_dir = semantics_root / scene_id

            if not scene_path.exists() or not semantics_dir.exists():
                continue

            result = evaluate_scene_single_prompt(
                model, scene_path, semantics_dir, device,
                num_views=4,  # Use 4 views for single-prompt eval
                objects_per_scene=args.objects_per_scene,
                min_pixel_fraction=args.min_mask_coverage,
                image_size=(args.image_size, args.image_size),
                prompt_view=args.prompt_view,
                use_world_poses=args.use_world_poses,
                use_estimated_poses=args.use_estimated_poses,
                da3_nested_cache_dir=da3_cache_dir,
                allowed_categories=allowed_categories,
            )

            if 'error' in result:
                print(f"  {scene_id}: {result['error']}")
            else:
                all_results.append(result)
                print(f"  {scene_id}: Prompted={100*result['mean_prompted_iou']:.1f}%, "
                      f"Unprompted={100*result['mean_unprompted_iou']:.1f}%, "
                      f"Propagation={result['mean_propagation_ratio']:.2f}x")

        if not all_results:
            print("No valid results!")
            return

        # Compute single-prompt specific metrics
        mean_prompted = np.mean([r['mean_prompted_iou'] for r in all_results])
        mean_unprompted = np.mean([r['mean_unprompted_iou'] for r in all_results])
        mean_propagation = np.mean([r['mean_propagation_ratio'] for r in all_results])

        # 3D localization metrics
        mean_acc_5cm = np.mean([r.get('acc_5cm', 0) for r in all_results])
        mean_acc_10cm = np.mean([r.get('acc_10cm', 0) for r in all_results])
        centroid_errors = [r.get('mean_centroid_error_m', float('inf')) for r in all_results if r.get('mean_centroid_error_m', float('inf')) != float('inf')]
        mean_centroid_error = np.mean(centroid_errors) if centroid_errors else float('inf')

        # World-frame metrics if available
        has_world_metrics = any('acc_5cm_world' in r for r in all_results)
        if has_world_metrics:
            mean_acc_5cm_world = np.mean([r.get('acc_5cm_world', 0) for r in all_results if 'acc_5cm_world' in r])
            mean_acc_10cm_world = np.mean([r.get('acc_10cm_world', 0) for r in all_results if 'acc_10cm_world' in r])
            centroid_errors_world = [r.get('mean_centroid_error_world_m', float('inf')) for r in all_results if r.get('mean_centroid_error_world_m', float('inf')) != float('inf')]
            mean_centroid_error_world = np.mean(centroid_errors_world) if centroid_errors_world else float('inf')

        print("\n" + "="*60)
        print("SINGLE-PROMPT EVALUATION RESULTS")
        print("="*60)
        print(f"Scenes evaluated: {len(all_results)}")
        print("-"*60)
        print(f"Mean Prompted IoU:    {100*mean_prompted:.2f}%")
        print(f"Mean Unprompted IoU:  {100*mean_unprompted:.2f}%")
        print(f"Propagation Ratio:    {mean_propagation:.2f}x")
        print("-"*60)
        print(f"3D Localization (prediction frame):")
        print(f"  Acc@5cm:            {100*mean_acc_5cm:.1f}%")
        print(f"  Acc@10cm:           {100*mean_acc_10cm:.1f}%")
        print(f"  Mean Error:         {mean_centroid_error*100:.1f} cm")
        if has_world_metrics:
            print(f"3D Localization (world frame, GT poses reference):")
            print(f"  Acc@5cm (world):    {100*mean_acc_5cm_world:.1f}%")
            print(f"  Acc@10cm (world):   {100*mean_acc_10cm_world:.1f}%")
            print(f"  Mean Error (world): {mean_centroid_error_world*100:.1f} cm")
        print("-"*60)
        print(f"  (Propagation > 0.8 means good cross-view transfer)")
        print("="*60)

        results_dict = {
            'mode': 'single_prompt',
            'checkpoint': args.checkpoint,
            'split': args.split,
            'num_scenes': len(all_results),
            'mean_prompted_iou': float(mean_prompted),
            'mean_unprompted_iou': float(mean_unprompted),
            'propagation_ratio': float(mean_propagation),
            # 3D localization metrics
            'acc_5cm': float(mean_acc_5cm),
            'acc_10cm': float(mean_acc_10cm),
            'mean_centroid_error_m': float(mean_centroid_error) if mean_centroid_error != float('inf') else None,
            # Pose options
            'use_world_poses': args.use_world_poses,
            'use_estimated_poses': args.use_estimated_poses,
            'per_scene_results': all_results,
        }

        # Add world-frame metrics if available
        if has_world_metrics:
            results_dict['acc_5cm_world'] = float(mean_acc_5cm_world)
            results_dict['acc_10cm_world'] = float(mean_acc_10cm_world)
            results_dict['mean_centroid_error_world_m'] = float(mean_centroid_error_world) if mean_centroid_error_world != float('inf') else None

        output_path = output_dir / 'results_single_prompt.json'
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {output_path}")
        return

    # MULTI-PROMPT MODE: Standard per-frame evaluation (MV-SAM protocol)
    ddp.print(f"\nStandard Evaluation (prompt_type={args.prompt_type})")
    if args.prompt_type != 'text_only':
        ddp.print(f"    Using {args.num_pos_points} positive + {args.num_neg_points} negative points per frame")

    # Paper viz: ALL ranks collect viz data (will be gathered to rank 0 later)
    collect_viz = (args.paper_viz or args.single_object_viz)
    paper_viz_pool = [] if collect_viz else None

    # Use tqdm only on main rank to avoid duplicate progress bars
    scene_iterator = tqdm(scene_ids, desc=f"Scenes (rank {ddp.rank})", disable=not ddp.is_main)
    for scene_id in scene_iterator:
        scene_path = data_root / 'data' / scene_id
        semantics_dir = semantics_root / scene_id

        if not scene_path.exists():
            continue

        if not semantics_dir.exists():
            continue

        result = evaluate_scene(
            model, scene_path, semantics_dir, device,
            num_frames=args.num_frames,
            objects_per_scene=args.objects_per_scene,
            min_pixel_fraction=args.min_mask_coverage,
            image_size=(args.image_size, args.image_size),
            save_viz=args.visualize and ddp.is_main,  # Only save viz on main rank
            viz_dir=viz_dir,
            viz_samples=args.viz_samples,
            prompt_type=args.prompt_type,
            num_pos_points=args.num_pos_points,
            num_neg_points=args.num_neg_points,
            sparse_prompts=args.sparse_prompts,
            num_prompted_frames=args.num_prompted_frames,
            output_localization=args.output_localization,
            output_depth=args.output_depth,
            prompt_augmentor=prompt_augmentor,
            semantic_union=args.semantic_union,
            da3_cache_dir=da3_cache_dir,
            # Procrustes evaluation
            procrustes=args.procrustes,
            procrustes_with_scale=args.procrustes_with_scale,
            gt_centroids_cache=gt_centroids_cache,
            data_root=data_root,
            # Category filtering
            allowed_categories=allowed_categories,
            # Spatial query filtering
            spatial_query_map=spatial_query_map,
            spatial_eval=args.spatial_eval,
            # Paper visualization collector
            paper_viz_collector=paper_viz_pool,
            # Frame selection
            frame_names=args.frame_names,
            eval_sampling=args.eval_sampling,
            multi_object_eval=getattr(args, 'multi_object_eval', False),
            temporal_smooth_alpha=getattr(args, 'temporal_smooth_alpha', 0.0),
            use_crf=getattr(args, 'use_crf', False),
        )

        if 'error' in result:
            print(f"  [rank {ddp.rank}] {scene_id}: {result['error']}")
        else:
            all_results.append(result)

            for cat, iou in result.get('per_category_iou', {}).items():
                all_category_metrics[cat]['iou'].append(iou)
            for cat, oracle_iou in result.get('per_category_oracle_iou', {}).items():
                all_category_metrics[cat]['oracle_iou'].append(oracle_iou)
            for cat, pixel_acc in result.get('per_category_pixel_acc', {}).items():
                all_category_metrics[cat]['pixel_acc'].append(pixel_acc)
            for cat, recall in result.get('per_category_recall', {}).items():
                all_category_metrics[cat]['recall'].append(recall)

            # Save per-rank partial results after each scene (never lose work)
            partial_dir = output_dir / 'partial'
            partial_dir.mkdir(parents=True, exist_ok=True)
            partial_path = partial_dir / f'rank{ddp.rank}_results.json'
            try:
                partial_ious = [r['mean_iou'] for r in all_results if 'mean_iou' in r]
                partial_summary = {
                    'rank': ddp.rank,
                    'scenes_done': len(all_results),
                    'scene_ids': [r.get('scene_id', '?') for r in all_results],
                    'running_miou': float(np.mean(partial_ious)) if partial_ious else 0.0,
                    'running_iou': float(np.mean([r['mean_iou'] for r in all_results])) if all_results else 0.0,
                    'last_updated': datetime.now().isoformat(),
                    'results': all_results,
                    'category_metrics': {cat: {k: v for k, v in metrics.items()} for cat, metrics in all_category_metrics.items()},
                }
                with open(partial_path, 'w') as f:
                    json.dump(partial_summary, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
            except Exception as e:
                print(f"  [rank {ddp.rank}] Warning: failed to save partial results: {e}")

    # Gather results from all ranks
    ddp.barrier()  # Sync before gathering

    if ddp.is_distributed:
        import pickle
        # Serialize results for gathering
        local_results_bytes = pickle.dumps(all_results)
        local_metrics_bytes = pickle.dumps(dict(all_category_metrics))

        # Gather all results to rank 0
        if ddp.is_main:
            gathered_results = [None] * ddp.world_size
            gathered_metrics = [None] * ddp.world_size
        else:
            gathered_results = None
            gathered_metrics = None

        # Use dist.gather_object for Python objects
        dist.gather_object(all_results, gathered_results, dst=0)
        dist.gather_object(dict(all_category_metrics), gathered_metrics, dst=0)

        # Gather viz data from all ranks to rank 0 for unified visualization
        if paper_viz_pool is not None:
            gathered_viz = [None] * ddp.world_size if ddp.is_main else None
            dist.gather_object(paper_viz_pool, gathered_viz, dst=0)
            if ddp.is_main:
                paper_viz_pool = []
                for rank_pool in gathered_viz:
                    if rank_pool:
                        paper_viz_pool.extend(rank_pool)
                print(f"[Viz] Gathered {len(paper_viz_pool)} viz samples from {ddp.world_size} ranks")

        if ddp.is_main:
            # Merge results from all ranks
            all_results = []
            for rank_results in gathered_results:
                all_results.extend(rank_results)

            # Merge category metrics
            merged_category_metrics = defaultdict(lambda: {'iou': [], 'oracle_iou': [], 'pixel_acc': [], 'recall': []})
            for rank_metrics in gathered_metrics:
                for cat, metrics in rank_metrics.items():
                    for key, values in metrics.items():
                        merged_category_metrics[cat][key].extend(values)
            all_category_metrics = merged_category_metrics

    if not all_results:
        ddp.print("No valid results!")
        ddp.cleanup()
        return

    # Only compute and print final metrics on main rank
    if not ddp.is_main:
        ddp.cleanup()
        return

    # Compute final metrics
    sample_iou = np.mean([r['sample_iou'] for r in all_results])
    sample_oracle_iou = np.mean([r.get('oracle_iou', r['sample_iou']) for r in all_results])
    scene_miou = np.mean([r['miou'] for r in all_results])
    scene_oracle_miou = np.mean([r.get('oracle_miou', r['miou']) for r in all_results])
    sample_pixel_acc = np.mean([r['pixel_acc'] for r in all_results])
    sample_recall = np.mean([r['recall'] for r in all_results])
    sample_precision = np.mean([r['precision'] for r in all_results])
    sample_f1 = np.mean([r['f1'] for r in all_results])

    # Global pixel accuracy from raw counts (more accurate)
    total_tp = sum(r['total_tp'] for r in all_results)
    total_fp = sum(r['total_fp'] for r in all_results)
    total_fn = sum(r['total_fn'] for r in all_results)
    total_tn = sum(r['total_tn'] for r in all_results)
    all_pixels = total_tp + total_fp + total_fn + total_tn
    global_pixel_acc = (total_tp + total_tn) / all_pixels if all_pixels > 0 else 0.0

    global_per_cat_iou = {cat: np.mean(m['iou']) for cat, m in all_category_metrics.items() if m['iou']}
    global_per_cat_oracle_iou = {cat: np.mean(m['oracle_iou']) for cat, m in all_category_metrics.items() if m.get('oracle_iou')}
    global_per_cat_pixel_acc = {cat: np.mean(m['pixel_acc']) for cat, m in all_category_metrics.items() if m['pixel_acc']}
    global_per_cat_recall = {cat: np.mean(m['recall']) for cat, m in all_category_metrics.items() if m['recall']}
    global_miou = np.mean(list(global_per_cat_iou.values())) if global_per_cat_iou else 0.0
    global_oracle_miou = np.mean(list(global_per_cat_oracle_iou.values())) if global_per_cat_oracle_iou else 0.0
    global_mean_class_recall = np.mean(list(global_per_cat_recall.values())) if global_per_cat_recall else 0.0

    # For spatial eval reporting
    spatial_qualifiers_set = {'nearest', 'farthest', 'leftmost', 'rightmost', 'topmost', 'bottommost',
                              'closest', 'left', 'right', 'top', 'bottom'}

    avg_preprocess_ms = np.mean([r['avg_preprocess_ms'] for r in all_results if 'avg_preprocess_ms' in r])
    avg_inference_ms = np.mean([r['avg_inference_ms'] for r in all_results if 'avg_inference_ms' in r])
    total_samples = sum(r['num_samples'] for r in all_results)

    # Aggregate Acc@m metrics for 3D localization (weighted by num_centroid_samples)
    total_centroid_samples = sum(r.get('num_centroid_samples', 0) for r in all_results)
    if total_centroid_samples > 0:
        global_acc_5cm = sum(r.get('acc_5cm', 0) * r.get('num_centroid_samples', 0) for r in all_results) / total_centroid_samples
        global_acc_10cm = sum(r.get('acc_10cm', 0) * r.get('num_centroid_samples', 0) for r in all_results) / total_centroid_samples
        global_acc_50cm = sum(r.get('acc_50cm', 0) * r.get('num_centroid_samples', 0) for r in all_results) / total_centroid_samples
        # Mean centroid error - filter out inf values
        valid_errors = [r.get('mean_centroid_error_m', float('inf')) for r in all_results if r.get('mean_centroid_error_m', float('inf')) != float('inf')]
        global_mean_centroid_error = np.mean(valid_errors) if valid_errors else float('inf')
    else:
        global_acc_5cm, global_acc_10cm, global_acc_50cm, global_mean_centroid_error = 0, 0, 0, float('inf')

    # Aggregate Procrustes-aligned localization metrics
    total_procrustes_samples = sum(r.get('procrustes_num_samples', 0) for r in all_results)
    if total_procrustes_samples > 0:
        global_procrustes_acc_5cm = sum(
            (r.get('procrustes_acc_5cm') or 0) * r.get('procrustes_num_samples', 0)
            for r in all_results
        ) / total_procrustes_samples
        global_procrustes_acc_10cm = sum(
            (r.get('procrustes_acc_10cm') or 0) * r.get('procrustes_num_samples', 0)
            for r in all_results
        ) / total_procrustes_samples
        valid_procrustes_errors = [
            r.get('procrustes_mean_error_m') for r in all_results
            if r.get('procrustes_mean_error_m') is not None
        ]
        global_procrustes_mean_error = np.mean(valid_procrustes_errors) if valid_procrustes_errors else None
        procrustes_scales = [r.get('procrustes_scale') for r in all_results if r.get('procrustes_scale') is not None]
        avg_procrustes_scale = np.mean(procrustes_scales) if procrustes_scales else None
    else:
        global_procrustes_acc_5cm = global_procrustes_acc_10cm = global_procrustes_mean_error = avg_procrustes_scale = None

    # Aggregate cross-view consistency
    valid_consistency = [r.get('consistency_iou') for r in all_results if r.get('consistency_iou') is not None]
    total_consistency_objects = sum(r.get('num_consistency_objects', 0) for r in all_results)
    global_consistency_iou = np.mean(valid_consistency) if valid_consistency else None

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Scenes evaluated: {len(all_results)}")
    print(f"Total samples: {total_samples}")
    print(f"Categories: {len(global_per_cat_iou)}")
    print("-"*70)
    print(f"{'Metric':<25} {'Selected':<15} {'Oracle':<15} {'Gap':<10}")
    print("-"*70)
    print(f"{'Sample-avg IoU:':<25} {100*sample_iou:>13.2f}%  {100*sample_oracle_iou:>13.2f}%  {100*(sample_oracle_iou-sample_iou):>+8.2f}%")
    print(f"{'Scene-avg mIoU:':<25} {100*scene_miou:>13.2f}%  {100*scene_oracle_miou:>13.2f}%  {100*(scene_oracle_miou-scene_miou):>+8.2f}%")
    print(f"{'Global mIoU:':<25} {100*global_miou:>13.2f}%  {100*global_oracle_miou:>13.2f}%  {100*(global_oracle_miou-global_miou):>+8.2f}%")
    print("-"*70)
    print(f"mAcc (Pixel Acc): {100*global_pixel_acc:.2f}%  (sample-avg: {100*sample_pixel_acc:.2f}%)")
    print(f"Mean Class Recall:{100*global_mean_class_recall:.2f}%  (sample-avg: {100*sample_recall:.2f}%)")
    print(f"Precision:        {100*sample_precision:.2f}%")
    print(f"F1 Score:         {100*sample_f1:.2f}%")
    print("-"*70)
    if total_centroid_samples > 0:
        print(f"3D Localization (IoU-based, same pointmap):")
        print(f"  Acc@5cm:        {100*global_acc_5cm:.2f}%")
        print(f"  Acc@10cm:       {100*global_acc_10cm:.2f}%")
        print(f"  Acc@50cm:       {100*global_acc_50cm:.2f}%")
        if global_mean_centroid_error != float('inf'):
            print(f"  Mean Error:     {global_mean_centroid_error*100:.1f} cm")
        print(f"  Samples:        {total_centroid_samples}")
        print("-"*60)

    # Procrustes-aligned localization 
    if total_procrustes_samples > 0:
        print(f"📐 Procrustes-aligned Localization (vs GT mesh centroids):")
        print(f"  Acc@5cm:        {100*global_procrustes_acc_5cm:.2f}%")
        print(f"  Acc@10cm:       {100*global_procrustes_acc_10cm:.2f}%")
        if global_procrustes_mean_error is not None:
            print(f"  Mean Error:     {global_procrustes_mean_error*100:.1f} cm")
        if avg_procrustes_scale is not None:
            print(f"  Avg Scale:      {avg_procrustes_scale:.3f}")
        print(f"  Samples:        {total_procrustes_samples}")
        print("-"*60)
    # Spatial eval metrics
    if args.spatial_eval:
        spatial_results = [r for r in all_results if r.get('spatial_miou') is not None]
        spatial_per_cat = {cat: np.mean(m['iou']) for cat, m in all_category_metrics.items()
                          if m['iou'] and cat.split()[0].lower() in spatial_qualifiers_set}
        spatial_global_miou = np.mean(list(spatial_per_cat.values())) if spatial_per_cat else 0.0
        total_spatial_queries = sum(r.get('spatial_num_queries', 0) for r in all_results)
        print(f"Spatial Language Evaluation:")
        print(f"  Spatial mIoU:    {100*spatial_global_miou:.2f}%")
        print(f"  Spatial queries: {total_spatial_queries} ({len(spatial_per_cat)} unique)")
        if spatial_per_cat:
            # Break down by qualifier type
            for qual in ['nearest', 'farthest', 'leftmost', 'rightmost']:
                qual_cats = {c: v for c, v in spatial_per_cat.items() if c.startswith(qual)}
                if qual_cats:
                    print(f"    {qual}: {100*np.mean(list(qual_cats.values())):.1f}% ({len(qual_cats)} cats)")
        print("-"*60)
    # Cross-view consistency metric
    if global_consistency_iou is not None:
        print(f"Cross-View Consistency:")
        print(f"  Consistency IoU: {100*global_consistency_iou:.2f}%")
        print(f"  Objects:         {total_consistency_objects}")
        print("-"*60)
    print(f"Avg preprocess:   {avg_preprocess_ms:.1f} ms")
    print(f"Avg inference:    {avg_inference_ms:.1f} ms")
    print(f"Total per sample: {avg_preprocess_ms + avg_inference_ms:.1f} ms")
    print("-"*60)

    sorted_cats = sorted(global_per_cat_iou.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 categories:")
    for cat, iou in sorted_cats[:5]:
        recall = global_per_cat_recall.get(cat, 0)
        print(f"  {cat}: IoU={100*iou:.1f}%, Recall={100*recall:.1f}%")
    print("\nBottom 5 categories:")
    for cat, iou in sorted_cats[-5:]:
        recall = global_per_cat_recall.get(cat, 0)
        print(f"  {cat}: IoU={100*iou:.1f}%, Recall={100*recall:.1f}%")

    # Cross-fold analysis (stratified category grouping)
    fold_results = None
    if args.cross_fold and len(global_per_cat_iou) >= args.num_folds:
        print(f"\n{'='*70}")
        print(f"CROSS-FOLD ANALYSIS ({args.num_folds} folds)")
        print("="*70)
        print("Grouping categories into folds for per-group performance analysis\n")

        # Sort categories alphabetically for deterministic fold assignment
        sorted_categories = sorted(global_per_cat_iou.keys())
        fold_size = len(sorted_categories) // args.num_folds

        fold_results = []
        for fold_idx in range(args.num_folds):
            start_idx = fold_idx * fold_size
            if fold_idx == args.num_folds - 1:
                # Last fold gets remaining categories
                end_idx = len(sorted_categories)
            else:
                end_idx = (fold_idx + 1) * fold_size

            fold_categories = sorted_categories[start_idx:end_idx]
            fold_ious = [global_per_cat_iou[cat] for cat in fold_categories]
            fold_recalls = [global_per_cat_recall.get(cat, 0) for cat in fold_categories]

            fold_mean_iou = np.mean(fold_ious)
            fold_mean_recall = np.mean(fold_recalls)

            fold_results.append({
                'fold_id': fold_idx,
                'categories': fold_categories,
                'num_categories': len(fold_categories),
                'mean_iou': float(fold_mean_iou),
                'mean_recall': float(fold_mean_recall),
            })

            print(f"Fold {fold_idx + 1}/{args.num_folds}: {len(fold_categories)} categories")
            print(f"  Mean IoU:    {100*fold_mean_iou:.2f}%")
            print(f"  Mean Recall: {100*fold_mean_recall:.2f}%")
            print(f"  Categories:  {', '.join(fold_categories[:5])}" +
                  (f", ... (+{len(fold_categories)-5} more)" if len(fold_categories) > 5 else ""))
            print()

        # Find best/worst folds
        best_fold = max(fold_results, key=lambda x: x['mean_iou'])
        worst_fold = min(fold_results, key=lambda x: x['mean_iou'])

        print(f"Best fold: Fold {best_fold['fold_id'] + 1} (mIoU={100*best_fold['mean_iou']:.2f}%)")
        print(f"Worst fold: Fold {worst_fold['fold_id'] + 1} (mIoU={100*worst_fold['mean_iou']:.2f}%)")
        print(f"Performance gap: {100*(best_fold['mean_iou'] - worst_fold['mean_iou']):.2f}%")
        print("="*70)

    # Build results dict
    results_dict = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_scenes': len(all_results),
        'total_samples': total_samples,
        # Selected mask metrics
        'sample_iou': float(sample_iou),
        'scene_miou': float(scene_miou),
        'global_miou': float(global_miou),
        # Oracle metrics (best possible mask selection)
        'oracle_sample_iou': float(sample_oracle_iou),
        'oracle_scene_miou': float(scene_oracle_miou),
        'oracle_global_miou': float(global_oracle_miou),
        # Gap between oracle and selected (room for improvement)
        'iou_gap': float(sample_oracle_iou - sample_iou),
        'miou_gap': float(global_oracle_miou - global_miou),
        # mAcc = global pixel accuracy (TP+TN)/total - standard in most papers
        'mAcc': float(global_pixel_acc),
        'sample_pixel_acc': float(sample_pixel_acc),
        # Mean class recall = TP/(TP+FN) per category, averaged
        'mean_class_recall': float(global_mean_class_recall),
        'sample_recall': float(sample_recall),
        'precision': float(sample_precision),
        'f1': float(sample_f1),
        'avg_preprocess_ms': float(avg_preprocess_ms),
        'avg_inference_ms': float(avg_inference_ms),
        # 3D Localization Accuracy (Acc@m) - IoU-based (same pointmap)
        'acc_5cm': float(global_acc_5cm),
        'acc_10cm': float(global_acc_10cm),
        'acc_50cm': float(global_acc_50cm),
        'mean_centroid_error_m': float(global_mean_centroid_error) if global_mean_centroid_error != float('inf') else None,
        'num_centroid_samples': total_centroid_samples,
        # Procrustes-aligned localization 
        'procrustes_acc_5cm': float(global_procrustes_acc_5cm) if global_procrustes_acc_5cm is not None else None,
        'procrustes_acc_10cm': float(global_procrustes_acc_10cm) if global_procrustes_acc_10cm is not None else None,
        'procrustes_mean_error_m': float(global_procrustes_mean_error) if global_procrustes_mean_error is not None else None,
        'procrustes_avg_scale': float(avg_procrustes_scale) if avg_procrustes_scale is not None else None,
        'procrustes_num_samples': total_procrustes_samples,
        # Cross-view consistency (do corresponding 3D points get same prediction?)
        'consistency_iou': float(global_consistency_iou) if global_consistency_iou is not None else None,
        'num_consistency_objects': total_consistency_objects,
        'per_category_iou': {k: float(v) for k, v in global_per_cat_iou.items()},
        'per_category_oracle_iou': {k: float(v) for k, v in global_per_cat_oracle_iou.items()},
        'per_category_pixel_acc': {k: float(v) for k, v in global_per_cat_pixel_acc.items()},
        'per_category_recall': {k: float(v) for k, v in global_per_cat_recall.items()},
        'fold_results': fold_results if fold_results else None,
        'per_scene_results': [
            {
                'scene_id': r['scene_id'],
                'miou': float(r['miou']),
                'oracle_miou': float(r.get('oracle_miou', r['miou'])),
                'pixel_acc': float(r['pixel_acc']),
                'global_pixel_acc': float(r['global_pixel_acc']),
                'recall': float(r['recall']),
                'precision': float(r['precision']),
                'f1': float(r['f1']),
                'num_samples': r['num_samples'],
                'consistency_iou': float(r['consistency_iou']) if r.get('consistency_iou') is not None else None,
            }
            for r in all_results
        ],
        'model_params': config['model_params'],
    }

    # Save results JSON
    output_path = output_dir / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate plots
    print("\nGenerating plots...")

    if global_per_cat_iou:
        plot_category_iou(global_per_cat_iou, output_dir / 'category_iou.png')

    if len(all_results) > 1:
        plot_scene_metrics(all_results, output_dir / 'scene_metrics.png')

    plot_summary(results_dict, output_dir / 'summary.png')

    # Paper-quality grid visualization (ScanNet++ path)
    # Viz data already gathered from all ranks above (before non-main ranks exit)
    if paper_viz_pool and args.paper_viz:
        print("\nGenerating paper-quality grid visualizations...")
        generate_paper_visualizations(paper_viz_pool, args, output_dir)

    # Single-object focused visualization
    if paper_viz_pool and args.single_object_viz:
        print("\nGenerating single-object focused visualization...")
        generate_single_object_viz(paper_viz_pool, args, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")

    # Cleanup DDP
    ddp.cleanup()


