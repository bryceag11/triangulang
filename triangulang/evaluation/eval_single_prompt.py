"""Single-prompt evaluation: multiview and scene-level."""
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



def evaluate_multiview_single_prompt(
    model: TrianguLangModel,
    images: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    label: str,
    prompt_view: int = 0,
    device: str = 'cuda',
    world_extrinsics: Optional[torch.Tensor] = None,
    world_intrinsics: Optional[torch.Tensor] = None,
    gt_extrinsics: Optional[torch.Tensor] = None,
    gt_intrinsics: Optional[torch.Tensor] = None,
    use_world_poses: bool = False,
) -> Dict:
    """
    Multi-view single-prompt evaluation - Our key differentiator from MV-SAM.

    Process N views together, prompt only view 0, measure IoU on views 1-N.
    This tests whether GASA can propagate object understanding across views
    using geometric attention.

    Args:
        model: TrianguLangModel
        images: List of [C, H, W] tensors for each view
        gt_masks: List of [H, W] tensors for each view
        label: Text prompt (applied only to prompt_view)
        prompt_view: Which view to prompt (default 0)
        device: cuda/cpu
        world_extrinsics: [N, 4, 4] camera-to-world transforms for world PE (estimated or GT)
        world_intrinsics: [N, 3, 3] or [3, 3] intrinsics for world PE
        gt_extrinsics: [N, 4, 4] GT camera-to-world transforms (for GT 3D centroid computation)
        gt_intrinsics: [N, 3, 3] or [3, 3] GT intrinsics (for GT 3D centroid computation)
        use_world_poses: If True, use world-frame pointmaps for GASA and localization

    Returns:
        dict with prompted_iou, unprompted_iou, per_view_ious, and 3D metrics
    """
    N = len(images)
    if N < 2:
        return {'error': 'Need at least 2 views for single-prompt eval'}

    resolution = model.resolution

    # 1. Preprocess all images
    img_tensors = []
    for img in images:
        if img.shape[-2:] != (resolution, resolution):
            img = F.interpolate(img.unsqueeze(0), size=(resolution, resolution),
                               mode='bilinear', align_corners=False).squeeze(0)
        img_tensors.append(img)

    # Stack into batch [N, C, H, W]
    img_batch = torch.stack(img_tensors, dim=0).to(device)

    # 2. Get depth and pointmaps for all views
    with autocast('cuda', dtype=torch.float16):
        depth, pose, intrinsics = model.get_depth_and_pose(img_batch)

    # Compute pointmaps - either in camera frame or world frame
    if use_world_poses and world_extrinsics is not None:
        # Use provided world-frame poses for cross-view consistency
        world_extrinsics = world_extrinsics.to(device=device, dtype=depth.dtype)
        if world_intrinsics is not None:
            world_intrinsics = world_intrinsics.to(device=device, dtype=depth.dtype)
            if world_intrinsics.dim() == 2:
                world_intrinsics = world_intrinsics.unsqueeze(0).expand(N, -1, -1)
        else:
            world_intrinsics = intrinsics

        # world_intrinsics should already be scaled to model resolution by caller
        # (e.g. cached intrinsics scaled from 336x504 -> 504x504 at load time)
        pointmaps, _ = model.pointmap_computer(depth, world_extrinsics, world_intrinsics, normalize=True)
    else:
        # Default: camera-frame pointmaps (identity pose)
        pointmaps, _ = model.pointmap_computer(depth, pose, intrinsics, normalize=True)

    pointmaps = pointmaps.squeeze(1)

    # Downsample pointmaps
    pts = pointmaps.permute(0, 3, 1, 2)
    pts = F.adaptive_avg_pool2d(pts, (model.attn_map_size, model.attn_map_size))
    pointmaps_small = pts.permute(0, 2, 3, 1)  # [N, H', W', 3]

    # 3. Run SAM3 backbone + encoder for all views (batched, matching forward_multiview)
    with autocast('cuda', dtype=torch.float16):
        backbone_out = {"img_batch_all_stages": img_batch}
        backbone_out.update(model.sam3.backbone.forward_image(img_batch))  # [N, ...] batched

        # Text encoding - repeat for each view (encoder expects 1:1 img:text)
        text_prompts_expanded = [label] * N
        text_out = model.sam3.backbone.forward_text(text_prompts_expanded, device=device)
        backbone_out.update(text_out)

        # Run encoder batched for all N views at once
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, N, 4, device=device),
            box_mask=torch.zeros(N, 0, device=device, dtype=torch.bool),
        )
        find_input = FindStage(
            img_ids=torch.arange(N, device=device, dtype=torch.long),
            text_ids=torch.arange(N, device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )

        prompt, prompt_mask, backbone_out = model.sam3._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )
        backbone_out, encoder_out, _ = model.sam3._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

    # Extract encoder memories: [N, L, D]
    encoder_memory = encoder_out["encoder_hidden_states"].transpose(0, 1)  # [N, L, D]
    L_per_view = encoder_memory.shape[1]

    # Extract per-view FPN features for seghead (needs per-view spatial maps)
    fpn_features_list = []
    for i in range(N):
        fpn_features_list.append([f[i:i+1] for f in backbone_out['backbone_fpn']])

    # 4. Concatenate all encoder memories for cross-view attention
    # This is the key: GASA attends to features from ALL views
    all_memory = encoder_memory.unsqueeze(0).view(1, N * L_per_view, -1)  # [1, N*L, D]

    # Concatenate pointmaps (flatten spatial dims)
    H_pts, W_pts = pointmaps_small.shape[1:3]

    # Reshape pointmaps to match encoder memory spatial size
    all_pointmaps_list = []
    for i in range(N):
        pts_flat = pointmaps_small[i].reshape(-1, 3)  # [H'*W', 3]
        # Resize to match L_per_view
        if pts_flat.shape[0] != L_per_view:
            pts_2d = pointmaps_small[i].permute(2, 0, 1).unsqueeze(0)  # [1, 3, H', W']
            target_size = int(math.sqrt(L_per_view))
            pts_2d = F.adaptive_avg_pool2d(pts_2d, (target_size, target_size))
            pts_flat = pts_2d.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            if pts_flat.shape[0] > L_per_view:
                pts_flat = pts_flat[:L_per_view]
            elif pts_flat.shape[0] < L_per_view:
                pad = torch.zeros(L_per_view - pts_flat.shape[0], 3, device=device)
                pts_flat = torch.cat([pts_flat, pad], dim=0)
        all_pointmaps_list.append(pts_flat)

    all_pointmaps = torch.cat(all_pointmaps_list, dim=0).unsqueeze(0)  # [1, N*L, 3]

    # 5. Get text embedding (all N copies are identical, take first)
    text_embedding = backbone_out.get('language_features', None)
    if text_embedding is not None:
        text_embedding = text_embedding.transpose(0, 1)  # [N, T, D]
        text_embedding = text_embedding[:1]  # [1, T, D] - all copies identical

    # 6. Run GASA decoder on concatenated memory (cross-view attention!)
    with autocast('cuda', dtype=torch.float16):
        queries, presence_logit, centroid_pred, iou_pred, per_query_centroids, text_scores, joint_scores, aux_outputs = model.gasa_decoder(
            all_memory,
            pointmaps_small[0],  # Use view 0's pointmap structure
            text_embedding,
            box_prompts=None,
            box_labels=None
        )
        queries = model.query_proj(queries)  # [1, Q, D]

    # 7. Compute GT pointmaps if GT poses provided (for accurate GT centroid measurement)
    gt_pointmaps = None
    if gt_extrinsics is not None:
        gt_extrinsics_dev = gt_extrinsics.to(device=device, dtype=depth.dtype)
        if gt_intrinsics is not None:
            gt_intrinsics_dev = gt_intrinsics.to(device=device, dtype=depth.dtype)
            if gt_intrinsics_dev.dim() == 2:
                gt_intrinsics_dev = gt_intrinsics_dev.unsqueeze(0).expand(N, -1, -1)
        else:
            gt_intrinsics_dev = intrinsics
        gt_pointmaps, _ = model.pointmap_computer(depth, gt_extrinsics_dev, gt_intrinsics_dev, normalize=True)
        gt_pointmaps = gt_pointmaps.squeeze(1)

    # 8. Generate masks for each view using shared queries
    per_view_results = []
    all_pred_masks = []  # Collect for cross-view consistency
    centroid_errors = []  # Collect for Acc@Xcm metrics
    centroid_errors_world = []  # Centroid errors in world frame (if GT poses available)

    for i in range(N):
        # Get pixel embeddings for this view
        fpn_features = fpn_features_list[i]
        with autocast('cuda', dtype=torch.float16):
            pixel_embed = model.sam3.segmentation_head.pixel_decoder(fpn_features)
            instance_embeds = model.sam3.segmentation_head.instance_seg_head(pixel_embed)

            # Predict mask using shared queries
            mask_preds = model.sam3.segmentation_head.mask_predictor(queries, instance_embeds)  # [1, Q, H, W]

        # Select best mask by confidence
        pred_logits = mask_preds.mean(dim=(-2, -1))
        best_idx = pred_logits.argmax(dim=1)
        pred_mask = mask_preds[0, best_idx[0]]  # [H, W]

        # Compute IoU with GT
        gt = gt_masks[i]
        if gt.shape != pred_mask.shape:
            gt = F.interpolate(gt.unsqueeze(0).unsqueeze(0).float(),
                              size=pred_mask.shape, mode='nearest').squeeze()

        pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
        gt_binary = (gt > 0.5).float()

        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = (intersection / union.clamp(min=1.0)).item()

        per_view_results.append({
            'view_idx': i,
            'iou': iou,
            'is_prompted': i == prompt_view,
        })

        # Collect pred mask for consistency metric
        all_pred_masks.append(pred_mask.detach())

        # Compute 3D centroid error using the pointmaps used for prediction
        pred_centroid = compute_3d_centroid(pred_mask.detach(), pointmaps[i])
        gt_centroid = compute_3d_centroid(gt.to(device), pointmaps[i])
        if pred_centroid is not None and gt_centroid is not None:
            error = compute_centroid_error(pred_centroid, gt_centroid)
            centroid_errors.append(error)

        # Also compute world-frame centroid error if GT pointmaps available
        # This is the "true" localization error using GT coordinate system
        if gt_pointmaps is not None:
            pred_centroid_world = compute_3d_centroid(pred_mask.detach(), gt_pointmaps[i])
            gt_centroid_world = compute_3d_centroid(gt.to(device), gt_pointmaps[i])
            if pred_centroid_world is not None and gt_centroid_world is not None:
                error_world = compute_centroid_error(pred_centroid_world, gt_centroid_world)
                centroid_errors_world.append(error_world)

    # 9. Compute aggregate metrics
    prompted_ious = [r['iou'] for r in per_view_results if r['is_prompted']]
    unprompted_ious = [r['iou'] for r in per_view_results if not r['is_prompted']]

    # 10. Compute cross-view consistency
    consistency_result = compute_cross_view_consistency(all_pred_masks, pointmaps)

    # 11. Compute centroid accuracy metrics
    acc_5cm = sum(1 for e in centroid_errors if e < 0.05) / max(len(centroid_errors), 1)
    acc_10cm = sum(1 for e in centroid_errors if e < 0.10) / max(len(centroid_errors), 1)
    mean_centroid_error = np.mean(centroid_errors) if centroid_errors else float('inf')

    # World-frame accuracy (using GT poses for reference)
    acc_5cm_world = None
    acc_10cm_world = None
    mean_centroid_error_world = None
    if centroid_errors_world:
        acc_5cm_world = sum(1 for e in centroid_errors_world if e < 0.05) / max(len(centroid_errors_world), 1)
        acc_10cm_world = sum(1 for e in centroid_errors_world if e < 0.10) / max(len(centroid_errors_world), 1)
        mean_centroid_error_world = np.mean(centroid_errors_world)

    result = {
        'prompted_iou': np.mean(prompted_ious) if prompted_ious else 0.0,
        'unprompted_iou': np.mean(unprompted_ious) if unprompted_ious else 0.0,
        'all_views_iou': np.mean([r['iou'] for r in per_view_results]),
        'per_view_results': per_view_results,
        'propagation_ratio': np.mean(unprompted_ious) / max(np.mean(prompted_ious), 0.01) if prompted_ious else 0.0,
        # 3D metrics (camera-frame or world-frame depending on input)
        'cross_view_consistency': consistency_result['consistency'],
        'num_correspondences': consistency_result['num_correspondences'],
        'acc_5cm': acc_5cm,
        'acc_10cm': acc_10cm,
        'mean_centroid_error_m': mean_centroid_error,
    }

    # Add world-frame metrics if GT poses were provided
    if acc_5cm_world is not None:
        result['acc_5cm_world'] = acc_5cm_world
        result['acc_10cm_world'] = acc_10cm_world
        result['mean_centroid_error_world_m'] = mean_centroid_error_world

    return result


@torch.no_grad()
def evaluate_scene_single_prompt(
    model: TrianguLangModel,
    scene_path: Path,
    semantics_dir: Path,
    device: str = 'cuda',
    num_views: int = 4,
    objects_per_scene: int = 5,
    min_pixel_fraction: float = 0.001,
    image_size: Tuple[int, int] = (1008, 1008),
    prompt_view: int = 0,
    use_world_poses: bool = False,
    use_estimated_poses: bool = False,
    da3_nested_cache_dir: Optional[Path] = None,
    allowed_categories: Optional[set] = None,
) -> Dict:
    """
    Evaluate single-prompt propagation on a scene.

    For each object, find N views where it's visible, prompt view 0,
    measure how well the model propagates to views 1-N.

    Args:
        use_world_poses: If True, use world-frame poses for pointmaps
        use_estimated_poses: If True, use DA3-NESTED estimated poses (else GT)
        da3_nested_cache_dir: Path to DA3-NESTED cache directory
    """
    from triangulang.evaluation.evaluate_gasa import load_scene_data, load_gt_masks

    images, objects, available_frames = load_scene_data(scene_path, semantics_dir)
    available_frames_set = set(available_frames)
    images = [img for img in images if img.name in available_frames_set]

    if len(images) < num_views:
        return {'error': f'Need at least {num_views} images, found {len(images)}'}

    # Match training skip_labels + structural elements
    skip_labels = SCANNETPP_SKIP_LABELS | {'shoes', 'book', 'shoe'}
    valid_objects = [(obj_id, obj) for obj_id, obj in objects.items()
                     if obj['label'] and obj['label'].lower() not in skip_labels]

    # Filter by allowed categories if specified (supports substring matching)
    if allowed_categories is not None:
        def matches_allowed(label):
            label_lower = label.lower()
            for allowed in allowed_categories:
                # Exact match or substring match (e.g., "towel" matches "kitchen towel")
                if allowed == label_lower or allowed in label_lower or label_lower in allowed:
                    return True
            return False
        valid_objects = [(obj_id, obj) for obj_id, obj in valid_objects
                         if matches_allowed(obj['label'])]

    if len(valid_objects) == 0:
        return {'error': 'No valid objects'}

    random.shuffle(valid_objects)

    results = {
        'prompted_ious': [],
        'unprompted_ious': [],
        'propagation_ratios': [],
        'objects_evaluated': 0,
        # 3D metrics
        'cross_view_consistencies': [],
        'acc_5cm_list': [],
        'acc_10cm_list': [],
        'centroid_errors': [],
    }

    # Load GT poses if using world poses
    gt_transforms = None
    gt_intrinsics_shared = None
    if use_world_poses or use_estimated_poses:
        gt_transforms, gt_intrinsics_shared = load_gt_poses(scene_path)

    # Load estimated poses cache if using estimated poses
    scene_id = scene_path.name
    estimated_cache = None
    if use_estimated_poses and da3_nested_cache_dir:
        # Will load per-object frame selection below
        pass

    for obj_id, obj_data in valid_objects[:objects_per_scene]:
        label = obj_data['label']

        # Find images where this object is visible with sufficient coverage
        obj_images = []
        obj_gt_masks = []
        obj_frame_names = []  # Track frame names for pose lookup

        for img_path in images:
            frame_name = img_path.name
            gt_masks_dict = load_gt_masks(semantics_dir, frame_name)

            if obj_id not in gt_masks_dict:
                continue

            gt_mask = gt_masks_dict[obj_id]
            pixel_fraction = gt_mask.sum() / gt_mask.size

            if pixel_fraction >= min_pixel_fraction:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size, Image.BILINEAR)
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

                # Resize GT mask
                gt_tensor = torch.from_numpy(gt_mask).float()
                gt_tensor = F.interpolate(gt_tensor.unsqueeze(0).unsqueeze(0),
                                         size=image_size, mode='nearest').squeeze()

                obj_images.append(img_tensor)
                obj_gt_masks.append(gt_tensor)
                obj_frame_names.append(frame_name)

                if len(obj_images) >= num_views:
                    break

        if len(obj_images) < num_views:
            continue

        # Prepare pose tensors for this object's views
        world_extrinsics = None
        world_intrinsics = None
        gt_extrinsics = None
        gt_intrinsics = None

        if use_world_poses or use_estimated_poses:
            selected_frame_names = obj_frame_names[:num_views]

            # Always load GT extrinsics for ground truth centroid comparison
            if gt_transforms is not None:
                gt_ext_list = []
                for fname in selected_frame_names:
                    ext = get_frame_extrinsics(gt_transforms, fname)
                    if ext is not None:
                        gt_ext_list.append(ext)
                    else:
                        gt_ext_list.append(torch.eye(4))
                if gt_ext_list:
                    gt_extrinsics = torch.stack(gt_ext_list)
                    gt_intrinsics = gt_intrinsics_shared

            # Load world extrinsics (either GT or estimated)
            if use_estimated_poses and da3_nested_cache_dir:
                # Load from DA3-NESTED cache
                estimated_cache = load_cached_da3_nested(
                    da3_nested_cache_dir, scene_id, selected_frame_names
                )
                if estimated_cache is not None:
                    world_extrinsics = estimated_cache['extrinsics']
                    world_intrinsics = estimated_cache['intrinsics']
                    # Scale intrinsics from cache resolution to model resolution
                    # Cache stores intrinsics at processing resolution (e.g. 336x504)
                    # but model computes pointmaps at model resolution (e.g. 504x504)
                    cache_h, cache_w = estimated_cache['depths'].shape[-2:]
                    model_res = model.resolution
                    if cache_h != model_res or cache_w != model_res:
                        scale_x = model_res / cache_w
                        scale_y = model_res / cache_h
                        world_intrinsics = world_intrinsics.clone()
                        world_intrinsics[:, 0, 0] *= scale_x  # fx
                        world_intrinsics[:, 1, 1] *= scale_y  # fy
                        world_intrinsics[:, 0, 2] *= scale_x  # cx
                        world_intrinsics[:, 1, 2] *= scale_y  # cy
                else:
                    # Fall back to GT poses if cache not available
                    world_extrinsics = gt_extrinsics
                    world_intrinsics = gt_intrinsics
            else:
                # Use GT poses for world coordinates
                world_extrinsics = gt_extrinsics
                world_intrinsics = gt_intrinsics

        # Scale GT intrinsics from original resolution to model resolution if needed
        # (cached DA3 intrinsics are already scaled above; this handles GT fallback paths)
        if world_intrinsics is not None and gt_transforms is not None:
            orig_w = gt_transforms.get('w', None)
            orig_h = gt_transforms.get('h', None)
            model_res = model.resolution
            # Only scale if these are GT intrinsics (at original resolution),
            # not already-scaled cache intrinsics
            if orig_w and orig_h and world_intrinsics is gt_intrinsics:
                scale_x = model_res / orig_w
                scale_y = model_res / orig_h
                world_intrinsics = world_intrinsics.clone()
                if world_intrinsics.dim() == 2:
                    world_intrinsics[0, 0] *= scale_x  # fx
                    world_intrinsics[1, 1] *= scale_y  # fy
                    world_intrinsics[0, 2] *= scale_x  # cx
                    world_intrinsics[1, 2] *= scale_y  # cy
                else:
                    world_intrinsics[:, 0, 0] *= scale_x
                    world_intrinsics[:, 1, 1] *= scale_y
                    world_intrinsics[:, 0, 2] *= scale_x
                    world_intrinsics[:, 1, 2] *= scale_y

        # Scale GT intrinsics to model resolution for accurate GT centroid computation
        gt_intrinsics_scaled = gt_intrinsics
        if gt_intrinsics is not None and gt_transforms is not None:
            orig_w = gt_transforms.get('w', None)
            orig_h = gt_transforms.get('h', None)
            model_res = model.resolution
            if orig_w and orig_h and (orig_w != model_res or orig_h != model_res):
                scale_x = model_res / orig_w
                scale_y = model_res / orig_h
                gt_intrinsics_scaled = gt_intrinsics.clone()
                if gt_intrinsics_scaled.dim() == 2:
                    gt_intrinsics_scaled[0, 0] *= scale_x
                    gt_intrinsics_scaled[1, 1] *= scale_y
                    gt_intrinsics_scaled[0, 2] *= scale_x
                    gt_intrinsics_scaled[1, 2] *= scale_y
                else:
                    gt_intrinsics_scaled[:, 0, 0] *= scale_x
                    gt_intrinsics_scaled[:, 1, 1] *= scale_y
                    gt_intrinsics_scaled[:, 0, 2] *= scale_x
                    gt_intrinsics_scaled[:, 1, 2] *= scale_y

        # Run multi-view single-prompt evaluation
        mv_result = evaluate_multiview_single_prompt(
            model,
            obj_images[:num_views],
            obj_gt_masks[:num_views],
            label,
            prompt_view=prompt_view,
            device=device,
            world_extrinsics=world_extrinsics,
            world_intrinsics=world_intrinsics,
            gt_extrinsics=gt_extrinsics,
            gt_intrinsics=gt_intrinsics_scaled,
            use_world_poses=use_world_poses or use_estimated_poses,
        )

        if 'error' not in mv_result:
            results['prompted_ious'].append(mv_result['prompted_iou'])
            results['unprompted_ious'].append(mv_result['unprompted_iou'])
            results['propagation_ratios'].append(mv_result['propagation_ratio'])
            results['objects_evaluated'] += 1
            # 3D metrics
            results['cross_view_consistencies'].append(mv_result['cross_view_consistency'])
            results['acc_5cm_list'].append(mv_result['acc_5cm'])
            results['acc_10cm_list'].append(mv_result['acc_10cm'])
            if mv_result['mean_centroid_error_m'] != float('inf'):
                results['centroid_errors'].append(mv_result['mean_centroid_error_m'])

            # World-frame metrics (if GT poses were provided)
            if 'acc_5cm_world' in mv_result:
                if 'acc_5cm_world_list' not in results:
                    results['acc_5cm_world_list'] = []
                    results['acc_10cm_world_list'] = []
                    results['centroid_errors_world'] = []
                results['acc_5cm_world_list'].append(mv_result['acc_5cm_world'])
                results['acc_10cm_world_list'].append(mv_result['acc_10cm_world'])
                if mv_result.get('mean_centroid_error_world_m') and mv_result['mean_centroid_error_world_m'] != float('inf'):
                    results['centroid_errors_world'].append(mv_result['mean_centroid_error_world_m'])

    if results['objects_evaluated'] == 0:
        return {'error': 'No objects with sufficient views'}

    scene_result = {
        'scene_id': scene_path.name,
        'objects_evaluated': results['objects_evaluated'],
        'mean_prompted_iou': np.mean(results['prompted_ious']),
        'mean_unprompted_iou': np.mean(results['unprompted_ious']),
        'mean_propagation_ratio': np.mean(results['propagation_ratios']),
        'std_propagation_ratio': np.std(results['propagation_ratios']),
        # 3D metrics
        'cross_view_consistency': np.mean(results['cross_view_consistencies']) if results['cross_view_consistencies'] else 0.0,
        'acc_5cm': np.mean(results['acc_5cm_list']) if results['acc_5cm_list'] else 0.0,
        'acc_10cm': np.mean(results['acc_10cm_list']) if results['acc_10cm_list'] else 0.0,
        'mean_centroid_error_m': np.mean(results['centroid_errors']) if results['centroid_errors'] else float('inf'),
    }

    # Add world-frame metrics if available
    if 'acc_5cm_world_list' in results:
        scene_result['acc_5cm_world'] = np.mean(results['acc_5cm_world_list']) if results['acc_5cm_world_list'] else 0.0
        scene_result['acc_10cm_world'] = np.mean(results['acc_10cm_world_list']) if results['acc_10cm_world_list'] else 0.0
        scene_result['mean_centroid_error_world_m'] = np.mean(results['centroid_errors_world']) if results['centroid_errors_world'] else float('inf')

    return scene_result

