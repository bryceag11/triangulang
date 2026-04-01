"""Dataset-based evaluation (uCO3D, SpinNeRF, etc.)."""
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
from triangulang.utils.ddp_utils import DDPManager



from triangulang.data.dataset_factory import get_dataset, get_dataset_config


def evaluate_with_dataset(
    model: TrianguLangModel,
    dataset: Dataset,
    device: str,
    ddp: DDPManager,
    args,
    output_dir: Path,
    viz_dir: Optional[Path] = None,
    paper_viz_collector: Optional[List] = None,
) -> Dict:
    """
    Evaluate model on a dataset object (used for uCO3D and other benchmarks).

    Args:
        model: TrianguLangModel instance
        dataset: Dataset returning samples with 'images', 'gt_masks', 'prompt', etc.
        device: Device for computation
        ddp: DDP manager for distributed evaluation
        args: Command line arguments
        paper_viz_collector: If not None, append paper viz data dicts here.
        output_dir: Directory for results
        viz_dir: Directory for visualizations (None to skip)

    Returns:
        Dictionary with evaluation metrics
    """
    from torch.utils.data import DataLoader

    model.eval()

    # Create dataloader
    # For distributed: each rank gets a subset
    if ddp.is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Full metrics tracking (matching ScanNet++ evaluation)
    all_ious = []
    all_oracle_ious = []
    all_categories = []
    category_ious = defaultdict(list)
    category_oracle_ious = defaultdict(list)
    category_recalls = defaultdict(list)
    category_precisions = defaultdict(list)
    category_f1s = defaultdict(list)
    all_recalls = []
    all_precisions = []
    all_f1s = []
    centroid_errors = []
    centroid_errors_world = []

    pbar = tqdm(dataloader, desc="Evaluating", disable=not ddp.is_main)

    for batch_idx, batch in enumerate(pbar):
        try:
            images = batch['images'].to(device)  # [1, N, 3, H, W]
            gt_masks = batch['gt_masks'].to(device)  # [1, N, H, W]
            intrinsics = batch.get('intrinsics', None)
            extrinsics = batch.get('extrinsics', None)
            if intrinsics is not None:
                intrinsics = intrinsics.to(device)
            if extrinsics is not None:
                extrinsics = extrinsics.to(device)
            prompt = batch['prompt'][0] if isinstance(batch['prompt'], list) else batch['prompt']
            category = batch.get('category', [prompt])[0] if 'category' in batch else prompt
            scene_id = batch.get('scene_id', ['unknown'])[0]

            B, N, C, H, W = images.shape
            images = images.squeeze(0)  # [N, 3, H, W]
            gt_masks = gt_masks.squeeze(0)  # [N, H, W]
            if intrinsics is not None:
                intrinsics = intrinsics.squeeze(0)  # [N, 3, 3]
            if extrinsics is not None:
                extrinsics = extrinsics.squeeze(0)  # [N, 4, 4]

            # Generate point/box prompts from GT masks if needed
            prompt_type = getattr(args, 'prompt_type', 'text_only')
            num_pos = getattr(args, 'num_pos_points', 10)
            num_neg = getattr(args, 'num_neg_points', 2)

            # For multi-view, we need prompts per view
            all_point_prompts = []
            all_point_labels = []
            all_box_prompts = []
            all_box_labels = []

            # Determine if text should be used based on prompt type
            no_text_types = ('point_only', 'box_only', 'box_point_only')
            use_text = prompt_type not in no_text_types
            text_for_model = prompt if use_text else ''

            if prompt_type != 'text_only':
                for view_idx in range(N):
                    view_gt = gt_masks[view_idx]  # [H, W]
                    prompts = create_prompts_from_gt(
                        view_gt, prompt_type, num_pos, num_neg, device
                    )
                    if prompts['point_prompts'] is not None:
                        all_point_prompts.append(prompts['point_prompts'].squeeze(0))  # [N_pts, 2]
                        all_point_labels.append(prompts['point_labels'].squeeze(0))  # [N_pts]
                    if prompts['box_prompts'] is not None:
                        all_box_prompts.append(prompts['box_prompts'].squeeze(0))  # [1, 4]
                        all_box_labels.append(prompts['box_labels'].squeeze(0))  # [1]

                # Stack into batched tensors
                if all_point_prompts:
                    all_point_prompts = torch.stack(all_point_prompts, dim=0)  # [N, N_pts, 2]
                    all_point_labels = torch.stack(all_point_labels, dim=0)  # [N, N_pts]
                else:
                    all_point_prompts = None
                    all_point_labels = None

                if all_box_prompts:
                    all_box_prompts = torch.stack(all_box_prompts, dim=0)  # [N, 1, 4]
                    all_box_labels = torch.stack(all_box_labels, dim=0)  # [N, 1]
                else:
                    all_box_prompts = None
                    all_box_labels = None
            else:
                all_point_prompts = None
                all_point_labels = None
                all_box_prompts = None
                all_box_labels = None

            # Run inference
            all_frame_masks = []
            all_frame_all_masks = []  # For oracle IoU: all candidate masks per frame
            first_frame_outputs = None  # Store first frame outputs for paper_viz

            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    if args.per_frame:
                        # Per-frame processing using model.forward()
                        # Option 1: --batch-da3 - batch DA3 for cross-view depth, then per-frame segmentation
                        # Option 2: default - run DA3 independently per frame (no cross-view depth consistency)

                        # Pre-compute batched DA3 depth if requested
                        batched_depths = None
                        batched_da3_intrinsics = None
                        if getattr(args, 'batch_da3', False):
                            # Batch all views for DA3 (in chunks to avoid OOM)
                            da3_chunk_size = args.view_chunk_size if args.view_chunk_size > 0 else 16
                            all_depths = []
                            all_da3_intrinsics = []

                            for chunk_start in range(0, N, da3_chunk_size):
                                chunk_end = min(chunk_start + da3_chunk_size, N)
                                chunk_images = images[chunk_start:chunk_end]  # [chunk, 3, H, W]

                                # Run DA3 on chunk - model expects [B, N, C, H, W] for multi-view
                                # Reshape to [1, chunk, C, H, W] so DA3 sees all views together
                                chunk_batch = chunk_images.unsqueeze(0)  # [1, chunk, 3, H, W]

                                # Get depth using DA3's batch processing
                                # Note: This calls DA3 with proper multi-view batching
                                da3_res = model.da3_resolution
                                da3_H = da3_W = (da3_res // 14) * 14
                                chunk_resized = F.interpolate(chunk_images, size=(da3_H, da3_W), mode='bilinear', align_corners=False)

                                # DA3 forward with multi-view batch (proper cross-view attention)
                                da3_output = model.da3.model.forward(
                                    chunk_resized.unsqueeze(0),  # [1, chunk, 3, H, W]
                                    extrinsics=None, intrinsics=None,
                                    export_feat_layers=[], infer_gs=False
                                )

                                chunk_depth = da3_output.depth  # [1, chunk, H, W] or [1, chunk, 1, H, W]
                                if chunk_depth.dim() == 5:
                                    chunk_depth = chunk_depth.squeeze(2)  # [1, chunk, H, W]
                                chunk_depth = chunk_depth.squeeze(0)  # [chunk, H, W]

                                # Resize depth to model resolution
                                if chunk_depth.shape[-2:] != (model.resolution, model.resolution):
                                    chunk_depth = F.interpolate(
                                        chunk_depth.unsqueeze(1),
                                        size=(model.resolution, model.resolution),
                                        mode='bilinear', align_corners=False
                                    ).squeeze(1)

                                all_depths.append(chunk_depth)

                                # Get intrinsics from DA3 if available
                                if hasattr(da3_output, 'intrinsics') and da3_output.intrinsics is not None:
                                    chunk_intr = da3_output.intrinsics
                                    if chunk_intr.dim() == 4:  # [1, chunk, 3, 3]
                                        chunk_intr = chunk_intr.squeeze(0)  # [chunk, 3, 3]
                                    elif chunk_intr.dim() == 2:  # [3, 3]
                                        chunk_intr = chunk_intr.unsqueeze(0).expand(chunk_end - chunk_start, -1, -1)
                                    all_da3_intrinsics.append(chunk_intr)

                            batched_depths = torch.cat(all_depths, dim=0)  # [N, H, W]
                            if all_da3_intrinsics:
                                batched_da3_intrinsics = torch.cat(all_da3_intrinsics, dim=0)  # [N, 3, 3]

                        # Now run per-frame segmentation
                        for frame_idx in range(N):
                            frame_img = images[frame_idx:frame_idx+1]  # [1, 3, H, W]
                            frame_intrinsics = intrinsics[frame_idx:frame_idx+1] if intrinsics is not None else None
                            frame_extrinsics = extrinsics[frame_idx:frame_idx+1] if extrinsics is not None else None

                            # Get prompts for this frame
                            frame_point_prompts = all_point_prompts[frame_idx:frame_idx+1] if all_point_prompts is not None else None
                            frame_point_labels = all_point_labels[frame_idx:frame_idx+1] if all_point_labels is not None else None
                            frame_box_prompts = all_box_prompts[frame_idx:frame_idx+1] if all_box_prompts is not None else None
                            frame_box_labels = all_box_labels[frame_idx:frame_idx+1] if all_box_labels is not None else None

                            # Get pre-computed depth if using batched DA3
                            frame_cached_depth = None
                            frame_da3_intrinsics = None
                            if batched_depths is not None:
                                frame_cached_depth = batched_depths[frame_idx:frame_idx+1].unsqueeze(1)  # [1, 1, H, W]
                                if batched_da3_intrinsics is not None:
                                    frame_da3_intrinsics = batched_da3_intrinsics[frame_idx:frame_idx+1]

                            # Use model.forward() for single-frame inference
                            # Pass cached_depth if we have batched DA3 depth
                            outputs = model.forward(
                                images=frame_img,  # [1, 3, H, W]
                                text_prompts=[text_for_model],
                                gt_masks=None,
                                gt_intrinsics=frame_intrinsics,
                                gt_extrinsics=frame_extrinsics,
                                da3_extrinsics=frame_extrinsics,
                                da3_intrinsics=frame_da3_intrinsics if frame_da3_intrinsics is not None else frame_intrinsics,
                                cached_depth=frame_cached_depth,
                                point_prompts=frame_point_prompts,
                                point_labels=frame_point_labels,
                                box_prompts=frame_box_prompts,
                                box_labels=frame_box_labels,
                            )

                            # Save first frame outputs for paper_viz (has depth from DA3)
                            if frame_idx == 0:
                                first_frame_outputs = outputs

                            frame_mask = outputs.get('pred_masks')
                            if frame_mask is not None:
                                # pred_masks from forward() is [B, H, W] or [B, 1, H, W]
                                if frame_mask.dim() == 4:
                                    frame_mask = frame_mask[:, 0]  # [B, H, W]
                                all_frame_masks.append(frame_mask)
                            # Collect all candidate masks for oracle IoU
                            frame_all_masks = outputs.get('all_masks')
                            if frame_all_masks is not None:
                                all_frame_all_masks.append(frame_all_masks)
                    else:
                        # Chunked processing with cross-view attention within chunks
                        chunk_size = args.view_chunk_size if args.view_chunk_size > 0 else N
                        for chunk_start in range(0, N, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, N)
                            chunk_images = images[chunk_start:chunk_end]  # [chunk, 3, H, W]
                            chunk_intrinsics = intrinsics[chunk_start:chunk_end] if intrinsics is not None else None
                            chunk_extrinsics = extrinsics[chunk_start:chunk_end] if extrinsics is not None else None

                            # Get prompts for this chunk
                            chunk_point_prompts = all_point_prompts[chunk_start:chunk_end] if all_point_prompts is not None else None
                            chunk_point_labels = all_point_labels[chunk_start:chunk_end] if all_point_labels is not None else None
                            chunk_box_prompts = all_box_prompts[chunk_start:chunk_end] if all_box_prompts is not None else None
                            chunk_box_labels = all_box_labels[chunk_start:chunk_end] if all_box_labels is not None else None

                            # Pass extrinsics as both gt and da3 (model uses da3 when use_da3_poses_for_gasa=True)
                            outputs = model.forward_multiview(
                                images=chunk_images.unsqueeze(0),  # [1, chunk, 3, H, W]
                                text_prompts=[text_for_model],
                                gt_masks=None,
                                gt_intrinsics=chunk_intrinsics.unsqueeze(0) if chunk_intrinsics is not None else None,
                                gt_extrinsics=chunk_extrinsics.unsqueeze(0) if chunk_extrinsics is not None else None,
                                da3_extrinsics=chunk_extrinsics.unsqueeze(0) if chunk_extrinsics is not None else None,
                                da3_intrinsics=chunk_intrinsics.unsqueeze(0) if chunk_intrinsics is not None else None,
                                point_prompts=chunk_point_prompts.unsqueeze(0) if chunk_point_prompts is not None else None,
                                point_labels=chunk_point_labels.unsqueeze(0) if chunk_point_labels is not None else None,
                                box_prompts=chunk_box_prompts.unsqueeze(0) if chunk_box_prompts is not None else None,
                                box_labels=chunk_box_labels.unsqueeze(0) if chunk_box_labels is not None else None,
                            )

                            # Save first chunk outputs for paper_viz (has depth from DA3)
                            if chunk_start == 0:
                                first_frame_outputs = outputs

                            chunk_masks = outputs.get('masks') or outputs.get('pred_masks')
                            if chunk_masks is not None:
                                # Normalize to [chunk, H, W]
                                if chunk_masks.dim() == 6:
                                    chunk_masks = chunk_masks[0, :, 0, 0]  # [chunk, H, W]
                                elif chunk_masks.dim() == 5:
                                    chunk_masks = chunk_masks[0, :, 0]  # [chunk, H, W]
                                elif chunk_masks.dim() == 4:
                                    chunk_masks = chunk_masks[0]  # [chunk, H, W]
                                all_frame_masks.append(chunk_masks)
                            # Collect all candidate masks for oracle IoU
                            chunk_all_masks = outputs.get('all_masks')
                            if chunk_all_masks is not None:
                                all_frame_all_masks.append(chunk_all_masks)

            # Concatenate all frames/chunks
            if not all_frame_masks:
                continue
            pred_masks = torch.cat(all_frame_masks, dim=0)  # [N, H, W]

            # Resize pred masks to match GT
            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                pred_masks = F.interpolate(
                    pred_masks.unsqueeze(1).float(),
                    size=gt_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            # Compute full metrics per view (IoU, recall, precision, F1)
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
            gt_binary = (gt_masks > 0.5).float()

            # Per-view metrics
            intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))
            pred_sum = pred_binary.sum(dim=(-2, -1))
            gt_sum = gt_binary.sum(dim=(-2, -1))
            union = pred_sum + gt_sum - intersection

            # IoU = TP / (TP + FP + FN)
            iou_per_view = (intersection / (union + 1e-6)).mean().item()

            # Recall = TP / (TP + FN) = intersection / gt_sum
            recall_per_view = (intersection / (gt_sum + 1e-6)).mean().item()

            # Precision = TP / (TP + FP) = intersection / pred_sum
            precision_per_view = (intersection / (pred_sum + 1e-6)).mean().item()

            # F1 = 2 * precision * recall / (precision + recall)
            f1_per_view = 2 * precision_per_view * recall_per_view / (precision_per_view + recall_per_view + 1e-6)

            # Oracle IoU: compute best possible mask from all candidates
            oracle_iou_per_view = iou_per_view  # Fallback to selected
            if all_frame_all_masks:
                try:
                    oracle_ious_views = []
                    for vi in range(gt_masks.shape[0]):
                        gt_v = gt_masks[vi]  # [H, W]
                        # Find the all_masks tensor for this view
                        # Per-frame: each entry is [1, Q, H, W] for one frame
                        # Chunked: each entry is [1, N_chunk, Q, H, W] or [chunk, Q, H, W]
                        if args.per_frame and vi < len(all_frame_all_masks):
                            masks_v = all_frame_all_masks[vi]
                            # Shape: [1, Q, H, W] from model.forward()
                            if masks_v.dim() == 4:
                                masks_v = masks_v[0]  # [Q, H, W]
                            elif masks_v.dim() == 5:
                                masks_v = masks_v[0, 0]  # [Q, H, W]
                        else:
                            break  # Can't match views in chunked mode easily
                        # Resize if needed
                        if masks_v.shape[-2:] != gt_v.shape[-2:]:
                            masks_v = F.interpolate(
                                masks_v.unsqueeze(0).float(), size=gt_v.shape[-2:],
                                mode='bilinear', align_corners=False
                            ).squeeze(0)
                        oracle_result = compute_oracle_iou(masks_v.unsqueeze(0), gt_v)
                        oracle_ious_views.append(oracle_result['oracle_iou'])
                    if oracle_ious_views:
                        oracle_iou_per_view = np.mean(oracle_ious_views)
                except Exception:
                    pass  # Fallback to selected IoU

            all_ious.append(iou_per_view)
            all_oracle_ious.append(oracle_iou_per_view)
            all_recalls.append(recall_per_view)
            all_precisions.append(precision_per_view)
            all_f1s.append(f1_per_view)
            all_categories.append(category)
            category_ious[category].append(iou_per_view)
            category_oracle_ious[category].append(oracle_iou_per_view)
            category_recalls[category].append(recall_per_view)
            category_precisions[category].append(precision_per_view)
            category_f1s[category].append(f1_per_view)

            # Compute 3D centroid error if depth/pointmaps available
            # Get pointmaps from model outputs (if model ran DA3 internally)
            try:
                pointmaps = outputs.get('pointmaps')
                gt_centroid = batch.get('gt_centroid')

                if pointmaps is not None and gt_centroid is not None:
                    # pointmaps: [N, H, W, 3] or [1, N, H, W, 3]
                    if pointmaps.dim() == 5:
                        pointmaps = pointmaps.squeeze(0)  # [N, H, W, 3]

                    # Resize pointmaps to match mask size if needed
                    pts_h, pts_w = pointmaps.shape[1:3]
                    mask_h, mask_w = pred_binary.shape[-2:]
                    if pts_h != mask_h or pts_w != mask_w:
                        pointmaps = F.interpolate(
                            pointmaps.permute(0, 3, 1, 2),  # [N, 3, H, W]
                            size=(mask_h, mask_w),
                            mode='bilinear',
                            align_corners=False
                        ).permute(0, 2, 3, 1)  # [N, H, W, 3]

                    # Compute predicted centroid from mask + pointmaps (mean over all views)
                    pred_centroid_list = []
                    for v in range(pred_binary.shape[0]):
                        mask_v = pred_binary[v]  # [H, W]
                        pts_v = pointmaps[v]  # [H, W, 3]
                        if mask_v.sum() > 0:
                            # Weighted mean of 3D points
                            pts_flat = pts_v.view(-1, 3)  # [HW, 3]
                            mask_flat = mask_v.view(-1)  # [HW]
                            centroid_v = (pts_flat * mask_flat.unsqueeze(-1)).sum(0) / mask_flat.sum()
                            pred_centroid_list.append(centroid_v)

                    if pred_centroid_list:
                        pred_centroid = torch.stack(pred_centroid_list).mean(0)  # [3]
                        gt_centroid_tensor = gt_centroid.to(device).squeeze()  # [3]
                        error = torch.norm(pred_centroid - gt_centroid_tensor).item()
                        centroid_errors.append(error)
            except Exception as e:
                pass  # Skip centroid error if data not available

            # Update progress bar
            pbar.set_postfix({'mIoU': f'{np.mean(all_ious)*100:.1f}%', 'Recall': f'{np.mean(all_recalls)*100:.1f}%'})

            # Visualization
            if viz_dir and batch_idx < args.viz_samples:
                # Save first few samples
                save_visualization(
                    images=images.cpu(),
                    gt_masks=gt_binary.cpu(),
                    pred_masks=pred_binary.cpu(),
                    prompt=prompt,
                    iou=iou_per_view,
                    output_path=viz_dir / f'{scene_id.replace("/", "_")}_{batch_idx}.png',
                )

            # Collect paper viz data (one frame per sequence, first view)
            if paper_viz_collector is not None:
                pv_img = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pv_gt = gt_binary[0].cpu().numpy().astype(np.float32)
                pv_pred = pred_binary[0].cpu().numpy().astype(np.float32)
                pv_depth = None
                # Use first_frame_outputs if available (per-frame mode), else fall back to outputs
                viz_outputs = first_frame_outputs if first_frame_outputs is not None else outputs
                if viz_outputs is not None and 'depth' in viz_outputs:
                    depth_tensor = viz_outputs['depth']
                    # Handle various depth tensor shapes
                    if depth_tensor.dim() == 4:  # [B, 1, H, W] or [B, N, H, W]
                        pv_depth = depth_tensor[0, 0].cpu().float().numpy()
                    elif depth_tensor.dim() == 3:  # [B, H, W] or [1, H, W]
                        pv_depth = depth_tensor[0].cpu().float().numpy()
                    elif depth_tensor.dim() == 2:  # [H, W]
                        pv_depth = depth_tensor.cpu().float().numpy()
                # Get frame name from batch if available
                frame_names = batch.get('frame_names', batch.get('frame_ids', None))
                if frame_names is not None:
                    pv_frame_name = frame_names[0] if isinstance(frame_names, (list, tuple)) else str(frame_names)
                else:
                    pv_frame_name = f"frame_{batch_idx}"

                paper_viz_collector.append({
                    'image': pv_img,
                    'gt_mask': pv_gt,
                    'pred_mask': pv_pred,
                    'depth': pv_depth,
                    'label': prompt if isinstance(prompt, str) else str(prompt),
                    'scene_id': scene_id,
                    'frame_name': pv_frame_name,
                    'depth_source': 'live (DA3 per-frame)',  # uCO3D always runs DA3 live
                    'iou': iou_per_view,
                })

        except Exception as e:
            if ddp.is_main:
                print(f"Error processing batch {batch_idx}: {e}")
            continue

    # Gather results across ranks
    if ddp.is_distributed:
        # Gather sample-level metrics (tensors)
        all_ious_tensor = torch.tensor(all_ious, device=device)
        gathered_ious = [torch.zeros_like(all_ious_tensor) for _ in range(ddp.world_size)]
        dist.all_gather(gathered_ious, all_ious_tensor)
        all_ious = torch.cat(gathered_ious).cpu().numpy().tolist()

        all_oracle_ious_tensor = torch.tensor(all_oracle_ious, device=device)
        gathered_oracle_ious = [torch.zeros_like(all_oracle_ious_tensor) for _ in range(ddp.world_size)]
        dist.all_gather(gathered_oracle_ious, all_oracle_ious_tensor)
        all_oracle_ious = torch.cat(gathered_oracle_ious).cpu().numpy().tolist()

        # Gather per-category metrics (Python dicts) using all_gather_object
        all_category_ious_list = [None] * ddp.world_size
        all_category_recalls_list = [None] * ddp.world_size
        all_category_oracle_ious_list = [None] * ddp.world_size
        dist.all_gather_object(all_category_ious_list, dict(category_ious))
        dist.all_gather_object(all_category_recalls_list, dict(category_recalls))
        dist.all_gather_object(all_category_oracle_ious_list, dict(category_oracle_ious))

        # Merge category metrics from all ranks
        merged_category_ious = defaultdict(list)
        merged_category_recalls = defaultdict(list)
        merged_category_oracle_ious = defaultdict(list)
        for rank_cat_ious in all_category_ious_list:
            for cat, ious in rank_cat_ious.items():
                merged_category_ious[cat].extend(ious)
        for rank_cat_recalls in all_category_recalls_list:
            for cat, recalls in rank_cat_recalls.items():
                merged_category_recalls[cat].extend(recalls)
        for rank_cat_oracle_ious in all_category_oracle_ious_list:
            for cat, ious in rank_cat_oracle_ious.items():
                merged_category_oracle_ious[cat].extend(ious)

        category_ious = merged_category_ious
        category_recalls = merged_category_recalls
        category_oracle_ious = merged_category_oracle_ious

    # Compute metrics
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_oracle_iou = np.mean(all_oracle_ious) if all_oracle_ious else 0.0
    mean_recall = np.mean(all_recalls) if all_recalls else 0.0
    mean_precision = np.mean(all_precisions) if all_precisions else 0.0
    mean_f1 = np.mean(all_f1s) if all_f1s else 0.0

    # Per-category metrics (now from all ranks in DDP mode)
    category_miou = {}
    category_oracle_miou = {}
    category_mrecall = {}
    for cat, ious in category_ious.items():
        category_miou[cat] = np.mean(ious)
    for cat, ious in category_oracle_ious.items():
        category_oracle_miou[cat] = np.mean(ious)
    for cat, recalls in category_recalls.items():
        category_mrecall[cat] = np.mean(recalls)

    # Global mIoU (mean of per-category mIoU)
    global_miou = np.mean(list(category_miou.values())) if category_miou else 0.0
    global_oracle_miou = np.mean(list(category_oracle_miou.values())) if category_oracle_miou else 0.0
    global_mrecall = np.mean(list(category_mrecall.values())) if category_mrecall else 0.0

    # Compute Acc@m metrics if centroid errors are available
    acc_5cm = sum(1 for e in centroid_errors if e < 0.05) / max(len(centroid_errors), 1) if centroid_errors else None
    acc_10cm = sum(1 for e in centroid_errors if e < 0.10) / max(len(centroid_errors), 1) if centroid_errors else None
    acc_50cm = sum(1 for e in centroid_errors if e < 0.50) / max(len(centroid_errors), 1) if centroid_errors else None
    mean_centroid_error = np.mean(centroid_errors) if centroid_errors else None

    results = {
        'dataset': 'uco3d',
        'num_samples': len(all_ious),
        # Sample-averaged metrics
        'sample_iou': mean_iou,
        'oracle_iou': mean_oracle_iou,
        'sample_recall': mean_recall,
        'sample_precision': mean_precision,
        'sample_f1': mean_f1,
        # Category-averaged metrics (mIoU, mRecall)
        'scene_miou': global_miou,  # Named scene_miou for compatibility with ScanNet++ format
        'global_miou': global_miou,
        'oracle_miou': global_oracle_miou,
        'mean_class_recall': global_mrecall,
        # Per-category breakdowns
        'per_category_iou': category_miou,
        'per_category_oracle_iou': category_oracle_miou,
        'per_category_recall': category_mrecall,
        'num_categories': len(category_miou),
        # Oracle gap
        'iou_gap': mean_oracle_iou - mean_iou,
        'miou_gap': global_oracle_miou - global_miou,
    }

    # Add Acc@m metrics if available
    if acc_5cm is not None:
        results['acc_5cm'] = acc_5cm
        results['acc_10cm'] = acc_10cm
        results['acc_50cm'] = acc_50cm
        results['mean_centroid_error_m'] = float(mean_centroid_error) if mean_centroid_error is not None else None
        results['num_centroid_samples'] = len(centroid_errors)

    if ddp.is_main:
        ddp.print("\n" + "="*70)
        ddp.print("EVALUATION RESULTS (uCO3D)")
        ddp.print("="*70)
        ddp.print(f"Samples evaluated: {results['num_samples']}")
        ddp.print(f"Categories: {results['num_categories']}")
        ddp.print("-"*70)
        ddp.print(f"{'Metric':<25} {'Selected':<15} {'Oracle':<15} {'Gap':<10}")
        ddp.print("-"*70)
        ddp.print(f"{'Sample-avg IoU:':<25} {100*mean_iou:>13.2f}%  {100*mean_oracle_iou:>13.2f}%  {100*(mean_oracle_iou-mean_iou):>+8.2f}%")
        ddp.print(f"{'Global mIoU:':<25} {100*global_miou:>13.2f}%  {100*global_oracle_miou:>13.2f}%  {100*(global_oracle_miou-global_miou):>+8.2f}%")
        ddp.print("-"*70)
        ddp.print(f"Mean Class Recall:{100*global_mrecall:.2f}%  (sample-avg: {100*mean_recall:.2f}%)")
        ddp.print(f"Precision:        {100*mean_precision:.2f}%")
        ddp.print(f"F1 Score:         {100*mean_f1:.2f}%")
        ddp.print("-"*70)

        if acc_5cm is not None:
            ddp.print(f"3D Localization (IoU-based, same pointmap):")
            ddp.print(f"  Acc@5cm:        {100*acc_5cm:.2f}%")
            ddp.print(f"  Acc@10cm:       {100*acc_10cm:.2f}%")
            ddp.print(f"  Acc@50cm:       {100*acc_50cm:.2f}%")
            if mean_centroid_error is not None:
                ddp.print(f"  Mean Error:     {mean_centroid_error*100:.1f} cm")
            ddp.print(f"  Samples:        {len(centroid_errors)}")
            ddp.print("-"*60)

        # Top/bottom categories
        sorted_cats = sorted(category_miou.items(), key=lambda x: x[1], reverse=True)
        ddp.print(f"\nTop 5 categories:")
        for cat, iou in sorted_cats[:5]:
            recall = category_mrecall.get(cat, 0)
            ddp.print(f"  {cat}: IoU={100*iou:.1f}%, Recall={100*recall:.1f}%")
        ddp.print(f"\nBottom 5 categories:")
        for cat, iou in sorted_cats[-5:]:
            recall = category_mrecall.get(cat, 0)
            ddp.print(f"  {cat}: IoU={100*iou:.1f}%, Recall={100*recall:.1f}%")

    # Tests whether spatial prefixes ("nearest X", "leftmost X") degrade performance.
    # For single-instance datasets like uCO3D, spatial queries should be no-ops.
    GENERIC_SPATIAL_QUALIFIERS = ['nearest', 'farthest', 'leftmost', 'rightmost']
    spatial_eval_enabled = getattr(args, 'spatial_eval', False)

    if spatial_eval_enabled and ddp.is_main:
        ddp.print(f"\nSpatial Eval Pass (single-instance robustness)")

    spatial_ious_generic = []
    spatial_details_generic = defaultdict(list)

    if spatial_eval_enabled:
        # Distribute samples across DDP ranks
        sample_indices = list(range(len(dataset)))
        if ddp.is_distributed:
            rank_indices = [i for i in sample_indices if i % ddp.world_size == ddp.rank]
        else:
            rank_indices = sample_indices

        spatial_pbar = tqdm(rank_indices, desc="Spatial Eval", disable=not ddp.is_main)
        for sample_idx in spatial_pbar:
          for qualifier in GENERIC_SPATIAL_QUALIFIERS:
            try:
                batch_s = dataset[sample_idx]
                images_s = batch_s['images'].to(device)  # [N, 3, H, W]
                gt_masks_s = batch_s['gt_masks'].to(device)  # [N, H, W]
                prompt_s = batch_s['prompt'] if isinstance(batch_s['prompt'], str) else batch_s['prompt'][0]
                intrinsics_s = batch_s.get('intrinsics')
                if intrinsics_s is not None:
                    intrinsics_s = intrinsics_s.to(device)
                extrinsics_s = batch_s.get('extrinsics')
                if extrinsics_s is not None:
                    extrinsics_s = extrinsics_s.to(device)

                spatial_prompt = f"{qualifier} {prompt_s}"
                sq_type_g, _ = parse_spatial_qualifier(spatial_prompt)
                sq_idx_g = get_spatial_qualifier_idx(sq_type_g)
                sq_tensor_g = torch.tensor([sq_idx_g], device=device, dtype=torch.long) if sq_idx_g > 0 else None

                N_s = images_s.shape[0]
                frame_ious = []

                with torch.no_grad():
                    with autocast('cuda', dtype=torch.float16):
                        for fi in range(N_s):
                            gt_fi = gt_masks_s[fi]
                            if gt_fi.sum() < 1:
                                continue
                            frame_img = images_s[fi:fi+1]
                            frame_intr = intrinsics_s[fi:fi+1] if intrinsics_s is not None else None
                            frame_ext = extrinsics_s[fi:fi+1] if extrinsics_s is not None else None
                            outputs_g = model.forward(
                                images=frame_img,
                                text_prompts=[spatial_prompt],
                                gt_masks=None,
                                gt_intrinsics=frame_intr,
                                gt_extrinsics=frame_ext,
                                spatial_qualifier_idx=sq_tensor_g,
                            )
                            pred_g = outputs_g.get('pred_masks')
                            if pred_g is None:
                                continue
                            if pred_g.dim() == 4:
                                pred_g = pred_g[:, 0]
                            pred_g = pred_g.squeeze(0)
                            if pred_g.shape != gt_fi.shape:
                                pred_g = F.interpolate(
                                    pred_g.unsqueeze(0).unsqueeze(0).float(),
                                    size=gt_fi.shape[-2:], mode='bilinear', align_corners=False
                                ).squeeze(0).squeeze(0)
                            gt_bin_g = (gt_fi > 0.5).float()
                            pred_bin_g = (torch.sigmoid(pred_g) > 0.5).float()
                            inter_g = (pred_bin_g * gt_bin_g).sum()
                            union_g = pred_bin_g.sum() + gt_bin_g.sum() - inter_g
                            frame_ious.append((inter_g / (union_g + 1e-6)).item())

                if frame_ious:
                    sample_iou = np.mean(frame_ious)
                    spatial_ious_generic.append(sample_iou)
                    spatial_details_generic[qualifier].append(sample_iou)

            except Exception:
                continue

        # DDP gather spatial results
        if ddp.is_distributed:
            spatial_local = {
                'spatial_ious': spatial_ious_generic,
                'spatial_details': {k: list(v) for k, v in spatial_details_generic.items()},
            }
            spatial_gathered = [None] * ddp.world_size if ddp.is_main else None
            dist.gather_object(spatial_local, spatial_gathered, dst=0)
            if ddp.is_main:
                spatial_ious_generic = []
                spatial_details_generic = defaultdict(list)
                for rd in spatial_gathered:
                    spatial_ious_generic.extend(rd['spatial_ious'])
                    for q, v in rd['spatial_details'].items():
                        spatial_details_generic[q].extend(v)

        if spatial_ious_generic:
            spatial_mean = np.mean(spatial_ious_generic)
            spatial_per_q = {q: np.mean(v) for q, v in spatial_details_generic.items()}
            results['spatial_eval'] = {
                'num_samples': len(spatial_ious_generic),
                'spatial_miou': float(spatial_mean),
                'baseline_miou': float(mean_iou),
                'delta': float(spatial_mean - mean_iou),
                'per_qualifier_iou': {q: float(v) for q, v in spatial_per_q.items()},
            }
            if ddp.is_main:
                ddp.print(f"\nSpatial Robustness")
                ddp.print(f"  Baseline mIoU:  {100*mean_iou:.2f}%")
                ddp.print(f"  Spatial mIoU:   {100*spatial_mean:.2f}%  (delta={100*(spatial_mean-mean_iou):+.2f}%)")
                for q in GENERIC_SPATIAL_QUALIFIERS:
                    if q in spatial_per_q:
                        ddp.print(f"    {q:<12} IoU={100*spatial_per_q[q]:5.1f}%  (n={len(spatial_details_generic[q])})")
                ddp.print("="*70)

    return results




