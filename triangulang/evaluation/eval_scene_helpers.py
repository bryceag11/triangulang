"""Helper functions extracted from eval_scene.evaluate_scene().

Split out to keep eval_scene.py under 1000 lines.  All helpers are
pure functions (no module-level side effects) and depend only on
standard library, numpy, torch, PIL, and the triangulang utilities
already imported in eval_scene.py.
"""

import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast

from triangulang.utils.spatial_reasoning import (
    parse_spatial_qualifier,
    get_spatial_qualifier_idx,
)
from triangulang.evaluation.eval_utils import (
    create_prompts_from_gt,
    compute_metrics,
    compute_oracle_iou,
    compute_spatial_gt,
    compute_cross_view_consistency,
)
from triangulang.evaluation.data_loading import load_gt_masks
from triangulang.models.sheaf_embeddings import compute_3d_localization, format_localization_text
from triangulang.utils.metrics import compute_gt_centroid


# Helper: spatial qualifier filtering

def _prepare_spatial_queries(
    valid_objects,
    spatial_query_map,
    da3_cache_dir,
    scene_path,
    semantics_dir,
    images,
):
    """Filter valid_objects down to the spatially-qualified instance.

    When *spatial_query_map* provides a qualifier (e.g. "leftmost") for a
    label, and depth information is available, this selects the single
    instance that best satisfies the qualifier.  All other instances of that
    label are discarded so the subsequent evaluation targets the correct
    object.

    Returns the (possibly filtered) list of (obj_id, obj_data) pairs.
    """
    if not (spatial_query_map and da3_cache_dir is not None):
        return valid_objects

    label_to_objects = defaultdict(list)
    for obj_id, obj_data in valid_objects:
        label_to_objects[obj_data["label"].lower()].append((obj_id, obj_data))

    filtered = []
    for label_lower, obj_list in label_to_objects.items():
        spatial_qualifier = None
        for _orig_prompt, (qualifier, base) in spatial_query_map.items():
            if qualifier and (base.lower() == label_lower or
                              base.lower() in label_lower or
                              label_lower in base.lower()):
                spatial_qualifier = qualifier
                break

        if spatial_qualifier and len(obj_list) > 1:
            reference_frame = images[0]
            ref_gt_masks = load_gt_masks(semantics_dir, reference_frame.name)

            candidate_masks = []
            candidate_objs = []
            for oid, obj_data in obj_list:
                if oid in ref_gt_masks and ref_gt_masks[oid].sum() > 0:
                    candidate_masks.append(ref_gt_masks[oid])
                    candidate_objs.append((oid, obj_data))

            if len(candidate_masks) > 1:
                ref_depth = _load_depth_from_cache(da3_cache_dir, scene_path, reference_frame)

                if ref_depth is not None:
                    mask_h, mask_w = candidate_masks[0].shape
                    if ref_depth.shape != (mask_h, mask_w):
                        ref_depth = ref_depth.astype(np.float32)
                        ref_depth = np.array(Image.fromarray(ref_depth).resize(
                            (mask_w, mask_h), Image.BILINEAR))

                    spatial_gt_map = compute_spatial_gt(candidate_masks, ref_depth)

                    if spatial_qualifier in spatial_gt_map:
                        selected_idx = spatial_gt_map[spatial_qualifier]
                        selected_obj = candidate_objs[selected_idx]
                        filtered.append(selected_obj)
                        print(f"    Spatial filter '{spatial_qualifier}' for '{label_lower}': "
                              f"selected obj {selected_obj[0]} "
                              f"(idx {selected_idx}/{len(candidate_masks)})")
                        continue

            # Fallback: filtering failed -- keep all
            filtered.extend(obj_list)
        else:
            filtered.extend(obj_list)

    return filtered


# Helper: multi-instance spatial items

def _prepare_multi_instance_eval(
    valid_objects,
    spatial_flag,
    da3_cache_dir,
    scene_path,
    semantics_dir,
    images,
):
    """Build auto-generated spatial items for multi-instance labels.

    For each label that appears at least twice in *valid_objects* we generate
    up to four items -- one per spatial qualifier in
    ['nearest', 'farthest', 'leftmost', 'rightmost'] -- each paired with
    the ground-truth object id that satisfies that qualifier.

    Returns a list of (label, [obj_id], spatial_prompt) triples (may be
    empty if *spatial_flag* is False or no depth cache is available).
    """
    MAX_SPATIAL_ITEMS_PER_SCENE = 8
    spatial_items = []

    if not (spatial_flag and da3_cache_dir is not None):
        return spatial_items

    QUALIFIERS = ["nearest", "farthest", "leftmost", "rightmost"]

    label_to_all_objs = defaultdict(list)
    for obj_id, obj_data in valid_objects:
        label_to_all_objs[obj_data["label"]].append(obj_id)

    multi_instance_labels = {lab: oids for lab, oids in label_to_all_objs.items()
                             if len(oids) >= 2}

    if not multi_instance_labels:
        return spatial_items

    reference_frame = images[0]
    ref_gt_masks = load_gt_masks(semantics_dir, reference_frame.name)
    ref_depth = _load_depth_from_cache(da3_cache_dir, scene_path, reference_frame)

    if ref_depth is None:
        return spatial_items

    for label, obj_id_list in multi_instance_labels.items():
        if len(spatial_items) >= MAX_SPATIAL_ITEMS_PER_SCENE:
            break

        candidate_masks = []
        candidate_oids = []
        for oid in obj_id_list:
            if oid in ref_gt_masks and ref_gt_masks[oid].sum() > 0:
                candidate_masks.append(ref_gt_masks[oid])
                candidate_oids.append(oid)

        if len(candidate_masks) < 2:
            continue

        mask_h, mask_w = candidate_masks[0].shape
        depth_for_spatial = ref_depth
        if ref_depth.shape != (mask_h, mask_w):
            depth_for_spatial = ref_depth.astype(np.float32)
            depth_for_spatial = np.array(Image.fromarray(depth_for_spatial).resize(
                (mask_w, mask_h), Image.BILINEAR))

        spatial_gt_map = compute_spatial_gt(candidate_masks, depth_for_spatial)

        for qualifier in QUALIFIERS:
            if len(spatial_items) >= MAX_SPATIAL_ITEMS_PER_SCENE:
                break
            if qualifier in spatial_gt_map:
                selected_oid = candidate_oids[spatial_gt_map[qualifier]]
                spatial_prompt = f"{qualifier} {label}"
                spatial_items.append((label, [selected_oid], spatial_prompt))

    return spatial_items


# Helper: per-object multi-frame forward pass

def _evaluate_single_object(
    model,
    label,
    obj_ids,
    images,
    semantics_dir,
    image_size,
    device,
    spatial_prompt_override,
    spatial_query_map,
    prompt_type,
    num_pos_points,
    num_neg_points,
    use_sparse,
    prompted_indices,
    points_per_prompted_frame,
    da3_cache_dir,
    scene_path,
    precomputed_backbone,
    output_localization,
    save_viz,
    viz_samples,
    paper_viz_collector,
    min_pixel_fraction,
    procrustes,
    scene_da3_extrinsics,
    prompt_augmentor,
    current_viz_count,
):
    """Run the model on every frame for a single (label, obj_ids) item.

    Returns a dict with keys:
      metrics_list       -- list of per-frame metric dicts
      category_label     -- string used for per-category aggregation
      viz_entries        -- list of dicts for the visualisation grid
      consistency_entry  -- float or None (cross-view consistency)
      raw_centroids      -- dict {obj_id: [world_cent, ...]}
      preprocess_times   -- list of floats (seconds)
      inference_times    -- list of floats (seconds)
      localization_entries -- list of localization dicts
    """
    category_label = spatial_prompt_override if spatial_prompt_override else label

    # Resolve spatial prompt from map if not already set
    if not spatial_prompt_override and spatial_query_map:
        for _orig, (qualifier, base) in spatial_query_map.items():
            if qualifier and (base.lower() == label.lower() or
                              base.lower() in label.lower() or
                              label.lower() in base.lower()):
                spatial_prompt_override = f"{qualifier} {label}"
                break

    aug_label = label
    if prompt_augmentor is not None:
        aug_label = prompt_augmentor.augment_language(label)

    obj_pred_masks = []
    obj_pointmaps = []

    metrics_list = []
    viz_entries = []
    localization_entries = []
    raw_centroids = defaultdict(list)
    preprocess_times_out = []
    inference_times_out = []

    for frame_idx, img_path in enumerate(images):
        gt_masks_dict = load_gt_masks(semantics_dir, img_path.name)

        gt_mask = None
        for oid in obj_ids:
            if oid in gt_masks_dict:
                if gt_mask is None:
                    gt_mask = gt_masks_dict[oid].copy()
                else:
                    gt_mask = np.maximum(gt_mask, gt_masks_dict[oid])

        if gt_mask is None:
            continue
        if gt_mask.sum() / gt_mask.size < min_pixel_fraction:
            continue

        t_pre0 = time.perf_counter()

        img = Image.open(img_path).convert("RGB")
        img = img.resize(image_size, Image.BILINEAR)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)
        gt_tensor = F.interpolate(gt_tensor, size=image_size, mode="nearest").squeeze()
        gt_tensor = gt_tensor.to(device)

        preprocess_times_out.append(time.perf_counter() - t_pre0)

        try:
            t_inf0 = time.perf_counter()

            if use_sparse:
                if frame_idx in prompted_indices:
                    frame_pos, frame_neg = points_per_prompted_frame[frame_idx]
                    prompts = create_prompts_from_gt(
                        gt_tensor, prompt_type, frame_pos, frame_neg, device)
                else:
                    prompts = create_prompts_from_gt(
                        gt_tensor, "text_only", 0, 0, device)
            else:
                prompts = create_prompts_from_gt(
                    gt_tensor, prompt_type, num_pos_points, num_neg_points, device)

            cached_depth, da3_intrinsics, da3_extrinsics, depth_source =                 _load_cached_depth(da3_cache_dir, scene_path, img_path, device)

            model_prompt = spatial_prompt_override if spatial_prompt_override else aug_label
            text_input = [model_prompt] if prompts.get("use_text", True) else [""]

            sq_type, _ = parse_spatial_qualifier(model_prompt)
            sq_idx = get_spatial_qualifier_idx(sq_type)
            sq_tensor = (torch.tensor([sq_idx], device=device, dtype=torch.long)
                         if sq_idx > 0 else None)

            frame_backbone = precomputed_backbone.get(img_path.stem)

            with autocast("cuda", dtype=torch.float16):
                outputs = model(
                    img_tensor, text_input, gt_tensor.unsqueeze(0),
                    box_prompts=prompts["box_prompts"],
                    box_labels=prompts["box_labels"],
                    point_prompts=prompts["point_prompts"],
                    point_labels=prompts["point_labels"],
                    cached_depth=cached_depth,
                    da3_intrinsics=da3_intrinsics,
                    da3_extrinsics=da3_extrinsics,
                    precomputed_sam3=frame_backbone,
                    spatial_qualifier_idx=sq_tensor,
                )
            pred = outputs["pred_masks"][:, 0]

            inference_times_out.append(time.perf_counter() - t_inf0)

            if pred.shape[-2:] != gt_tensor.shape[-2:]:
                pred = F.interpolate(pred.unsqueeze(1), size=gt_tensor.shape[-2:],
                                     mode="bilinear", align_corners=False).squeeze(1)

            metrics = compute_metrics(pred, gt_tensor.unsqueeze(0))

            if "all_masks" in outputs:
                oracle_iou = compute_oracle_iou(outputs["all_masks"], gt_tensor)["oracle_iou"]
            else:
                oracle_iou = metrics["iou"]

            frame_record = {
                "iou": metrics["iou"],
                "oracle_iou": oracle_iou,
                "pixel_acc": metrics["pixel_acc"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "category_label": category_label,
            }

            # 3D centroid error via pointmaps
            if "pointmaps_full" in outputs:
                try:
                    pointmaps_full = outputs["pointmaps_full"]
                    pm_H, pm_W = pointmaps_full.shape[1:3]
                    pred_rsz = (F.interpolate(pred.unsqueeze(1), size=(pm_H, pm_W),
                                              mode="bilinear", align_corners=False).squeeze(1)
                                if pred.shape[-2:] != (pm_H, pm_W) else pred)
                    gt_rsz = (F.interpolate(gt_tensor.unsqueeze(0).unsqueeze(0).float(),
                                           size=(pm_H, pm_W),
                                           mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                              if gt_tensor.shape[-2:] != (pm_H, pm_W) else gt_tensor.float())
                    pred_cent = compute_gt_centroid(pred_rsz[0], pointmaps_full[0])
                    gt_cent = compute_gt_centroid(gt_rsz, pointmaps_full[0])
                    frame_record["centroid_error"] = torch.norm(pred_cent - gt_cent).item()

                    if procrustes and pred_cent is not None:
                        frame_stem = img_path.stem
                        if frame_stem in scene_da3_extrinsics:
                            norm_params = outputs.get("norm_params", None)
                            if (norm_params is not None
                                    and "scale" in norm_params
                                    and "centroid" in norm_params):
                                cent_offset = norm_params["centroid"]
                                if cent_offset.dim() > 1:
                                    cent_offset = cent_offset[0]
                                denorm_cent = (pred_cent * norm_params["scale"]
                                               + cent_offset).cpu().numpy()
                            else:
                                denorm_cent = pred_cent.cpu().numpy()

                            if outputs.get("pointmaps_in_world_frame", False):
                                world_cent = denorm_cent
                            else:
                                c2w = scene_da3_extrinsics[frame_stem]
                                world_cent = c2w[:3, :3] @ denorm_cent + c2w[:3, 3]
                            for obj_id in obj_ids:
                                raw_centroids[obj_id].append(world_cent)
                except Exception:
                    pass

            # 3D centroid error via depth+intrinsics fallback
            elif "depth" in outputs and "intrinsics" in outputs:
                intrinsics = outputs["intrinsics"]
                if intrinsics.dim() == 3 and intrinsics.shape[-2:] == (3, 3):
                    try:
                        pred_loc = compute_3d_localization(
                            pred_masks=pred, depth=outputs["depth"],
                            intrinsics=intrinsics, threshold=0.5)
                        gt_loc = compute_3d_localization(
                            pred_masks=gt_tensor.unsqueeze(0).float(),
                            depth=outputs["depth"],
                            intrinsics=intrinsics, threshold=0.5)
                        if (pred_loc["centroid_3d"] is not None
                                and gt_loc["centroid_3d"] is not None):
                            frame_record["centroid_error"] = torch.norm(
                                pred_loc["centroid_3d"] - gt_loc["centroid_3d"]).item()
                    except Exception:
                        pass

            metrics_list.append(frame_record)

            # 3D localization output
            localization_data = None
            if output_localization and "depth" in outputs and "intrinsics" in outputs:
                try:
                    loc_result = compute_3d_localization(
                        pred_masks=pred, depth=outputs["depth"],
                        intrinsics=outputs["intrinsics"], threshold=0.5)
                    loc_text = format_localization_text(loc_result["centroid_3d"])[0]
                    localization_data = {
                        "centroid_3d": loc_result["centroid_3d"][0].cpu().tolist(),
                        "centroid_2d": loc_result["centroid_2d"][0].cpu().tolist(),
                        "mean_depth": loc_result["mean_depth"][0].item(),
                        "description": loc_text,
                    }
                    localization_entries.append({
                        "label": aug_label,
                        "frame": img_path.name,
                        **localization_data,
                    })
                except Exception as exc:
                    print(f"    Localization error: {exc}")

            # Visualisation entries
            if save_viz and current_viz_count + len(viz_entries) < viz_samples:
                pred_np = (torch.sigmoid(pred[0]).cpu().numpy() > 0.5).astype(np.float32)
                viz_entries.append({
                    "image": img_np,
                    "gt_mask": gt_tensor.cpu().numpy(),
                    "pred_mask": pred_np,
                    "label": aug_label,
                    "iou": metrics["iou"],
                    "localization": localization_data,
                })

            if paper_viz_collector is not None:
                pv_depth = (outputs["depth"][0].cpu().float().numpy()
                            if "depth" in outputs else None)
                paper_viz_collector.append({
                    "image": img_np.copy(),
                    "gt_mask": gt_tensor.cpu().numpy(),
                    "pred_mask": (torch.sigmoid(pred[0]).cpu().numpy() > 0.5).astype(np.float32),
                    "depth": pv_depth,
                    "label": aug_label,
                    "scene_id": scene_path.name,
                    "frame_name": img_path.stem,
                    "depth_source": depth_source,
                    "iou": metrics["iou"],
                })

            if "pointmaps_full" in outputs:
                obj_pred_masks.append(pred[0].detach())
                obj_pointmaps.append(outputs["pointmaps_full"][0].detach())

        except Exception as exc:
            print(f"Error processing {img_path.name}: {exc}")
            continue

    # Cross-view consistency
    consistency_entry = None
    if len(obj_pred_masks) >= 2:
        try:
            ptmaps_stack = torch.stack(obj_pointmaps)
            cr = compute_cross_view_consistency(
                obj_pred_masks, ptmaps_stack, threshold=0.05, subsample=1024)
            if cr["num_correspondences"] > 0:
                consistency_entry = cr["consistency"]
        except Exception:
            pass

    return {
        "metrics_list": metrics_list,
        "category_label": category_label,
        "viz_entries": viz_entries,
        "consistency_entry": consistency_entry,
        "raw_centroids": dict(raw_centroids),
        "preprocess_times": preprocess_times_out,
        "inference_times": inference_times_out,
        "localization_entries": localization_entries,
    }


# Helper: aggregate per-scene results into a summary dict

def _aggregate_scene_results(
    results,
    category_metrics,
    preprocess_times,
    inference_times,
    consistency_ious,
    procrustes,
    procrustes_alignment,
    gt_centroids_cache,
    scene_id,
    raw_centroids_per_object,
    eval_items,
    save_viz,
    viz_data,
    viz_dir,
    scene_path,
):
    """Compute and return the final per-scene metrics dict.

    If no predictions were collected, returns {'error': 'No valid predictions'}.
    """
    if not results["iou"]:
        return {"error": "No valid predictions"}

    # Procrustes-aligned 3D localisation
    procrustes_errors = []
    procrustes_scale = None

    if (procrustes and procrustes_alignment is not None
            and gt_centroids_cache and scene_id in gt_centroids_cache):
        R, t, s = procrustes_alignment
        procrustes_scale = s

        for _label, obj_ids, *_ in eval_items:
            for obj_id in obj_ids:
                obj_id_str = str(obj_id)
                if (obj_id_str in gt_centroids_cache[scene_id]
                        and obj_id in raw_centroids_per_object):
                    gt_cent_mesh = np.array(gt_centroids_cache[scene_id][obj_id_str])
                    raw_cents = raw_centroids_per_object[obj_id]
                    if raw_cents:
                        pred_cent_da3 = np.mean(raw_cents, axis=0)
                        pred_cent_aligned = s * R @ pred_cent_da3 + t
                        procrustes_errors.append(
                            np.linalg.norm(pred_cent_aligned - gt_cent_mesh))

    procrustes_acc_5cm = None
    procrustes_acc_10cm = None
    procrustes_mean_error = None
    if procrustes_errors:
        procrustes_acc_5cm = sum(1 for e in procrustes_errors if e < 0.05) / len(procrustes_errors)
        procrustes_acc_10cm = sum(1 for e in procrustes_errors if e < 0.10) / len(procrustes_errors)
        procrustes_mean_error = np.mean(procrustes_errors)

    # Visualisation save
    if save_viz and viz_data and viz_dir is not None:
        _save_scene_viz(viz_data, viz_dir, scene_path)

    # Per-category aggregation
    per_cat_iou = {cat: np.mean(m["iou"])
                   for cat, m in category_metrics.items() if m["iou"]}
    per_cat_oracle_iou = {cat: np.mean(m["oracle_iou"])
                          for cat, m in category_metrics.items() if m.get("oracle_iou")}
    per_cat_pixel_acc = {cat: np.mean(m["pixel_acc"])
                         for cat, m in category_metrics.items() if m["pixel_acc"]}
    per_cat_recall = {cat: np.mean(m["recall"])
                      for cat, m in category_metrics.items() if m["recall"]}
    miou = np.mean(list(per_cat_iou.values())) if per_cat_iou else 0.0
    oracle_miou = np.mean(list(per_cat_oracle_iou.values())) if per_cat_oracle_iou else 0.0
    mrecall = np.mean(list(per_cat_recall.values())) if per_cat_recall else 0.0

    spatial_qualifiers_set = {"nearest", "farthest", "leftmost", "rightmost",
                              "topmost", "bottommost", "closest",
                              "left", "right", "top", "bottom"}
    spatial_cat_iou = {cat: v for cat, v in per_cat_iou.items()
                       if cat.split()[0].lower() in spatial_qualifiers_set}
    spatial_miou = np.mean(list(spatial_cat_iou.values())) if spatial_cat_iou else None

    total_tp = sum(results["tp"])
    total_fp = sum(results["fp"])
    total_fn = sum(results["fn"])
    total_tn = sum(results["tn"])
    total_pixels = total_tp + total_fp + total_fn + total_tn
    global_pixel_acc = (total_tp + total_tn) / total_pixels if total_pixels > 0 else 0.0

    avg_preprocess_ms = np.mean(preprocess_times) * 1000 if preprocess_times else 0
    avg_inference_ms = np.mean(inference_times) * 1000 if inference_times else 0

    centroid_errors = results.get("centroid_errors", [])
    acc_5cm = sum(1 for e in centroid_errors if e < 0.05) / max(len(centroid_errors), 1)
    acc_10cm = sum(1 for e in centroid_errors if e < 0.10) / max(len(centroid_errors), 1)
    acc_50cm = sum(1 for e in centroid_errors if e < 0.50) / max(len(centroid_errors), 1)
    mean_centroid_error = np.mean(centroid_errors) if centroid_errors else float("inf")

    return {
        "scene_id": scene_path.name,
        "sample_iou": np.mean(results["iou"]),
        "oracle_iou": np.mean(results.get("oracle_iou", results["iou"])),
        "miou": miou,
        "oracle_miou": oracle_miou,
        "pixel_acc": np.mean(results["pixel_acc"]),
        "global_pixel_acc": global_pixel_acc,
        "recall": np.mean(results["recall"]),
        "mrecall": mrecall,
        "precision": np.mean(results["precision"]),
        "f1": np.mean(results["f1"]),
        "num_samples": len(results["iou"]),
        "num_categories": len(per_cat_iou),
        "per_category_iou": per_cat_iou,
        "per_category_oracle_iou": per_cat_oracle_iou,
        "per_category_pixel_acc": per_cat_pixel_acc,
        "per_category_recall": per_cat_recall,
        "avg_preprocess_ms": avg_preprocess_ms,
        "avg_inference_ms": avg_inference_ms,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
        "acc_5cm": acc_5cm,
        "acc_10cm": acc_10cm,
        "acc_50cm": acc_50cm,
        "mean_centroid_error_m": mean_centroid_error,
        "num_centroid_samples": len(centroid_errors),
        "spatial_miou": spatial_miou,
        "spatial_num_queries": len(spatial_cat_iou),
        "spatial_per_category_iou": spatial_cat_iou,
        "procrustes_acc_5cm": procrustes_acc_5cm,
        "procrustes_acc_10cm": procrustes_acc_10cm,
        "procrustes_mean_error_m": procrustes_mean_error,
        "procrustes_scale": procrustes_scale,
        "procrustes_num_samples": len(procrustes_errors) if procrustes_errors else 0,
        "consistency_iou": np.mean(consistency_ious) if consistency_ious else None,
        "num_consistency_objects": len(consistency_ious),
    }


# Private utilities

def _load_depth_from_cache(da3_cache_dir, scene_path, frame_path):
    """Load a depth array from the DA3 nested cache; returns None on failure."""
    cache_path = da3_cache_dir / scene_path.name / f"{frame_path.stem}.pt"
    if not cache_path.exists():
        return None
    try:
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
        depth = cache_data["depth"].numpy()
        if depth.ndim == 4:
            depth = depth.squeeze()
        elif depth.ndim == 3:
            depth = depth.squeeze(0)
        return depth
    except Exception:
        return None


def _load_cached_depth(da3_cache_dir, scene_path, img_path, device):
    """Load depth + camera matrices from the DA3 nested cache for one frame.

    Returns (cached_depth, da3_intrinsics, da3_extrinsics, depth_source)
    where tensors are on *device* (or None if not available).
    """
    if da3_cache_dir is None:
        return None, None, None, "live (no --da3-nested-cache)"

    cache_path = da3_cache_dir / scene_path.name / f"{img_path.stem}.pt"
    if not cache_path.exists():
        return None, None, None, f"live (cache not found: {cache_path})"

    try:
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
        depth = cache_data["depth"].float()
        if depth.dim() == 4:
            depth = depth.squeeze(0)
        elif depth.dim() == 2:
            depth = depth.unsqueeze(0)
        cached_depth = depth.unsqueeze(0).to(device)
        depth_source = f"cache:{img_path.stem}"

        da3_intrinsics = None
        da3_extrinsics = None
        if "intrinsics" in cache_data:
            da3_intrinsics = cache_data["intrinsics"].float().unsqueeze(0).to(device)
        if "extrinsics" in cache_data:
            da3_extrinsics = cache_data["extrinsics"].float().unsqueeze(0).to(device)

        return cached_depth, da3_intrinsics, da3_extrinsics, depth_source
    except Exception as exc:
        return None, None, None, f"live (cache error: {exc})"


def _save_scene_viz(viz_data, viz_dir, scene_path):
    """Save a comparison grid PNG for a scene.

    Imports matplotlib locally to avoid a hard dependency at module import time.
    """
    try:
        import matplotlib.pyplot as plt
        from triangulang.evaluation.visualization import create_comparison_grid
        viz_dir.mkdir(parents=True, exist_ok=True)
        fig = create_comparison_grid(
            [v["image"] for v in viz_data],
            [v["gt_mask"] for v in viz_data],
            [v["pred_mask"] for v in viz_data],
            [v["label"] for v in viz_data],
            [v["iou"] for v in viz_data],
            scene_path.name,
        )
        viz_path = viz_dir / f"{scene_path.name}_comparison.png"
        fig.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved viz: {viz_path.name}")
    except Exception as exc:
        print(f"    Viz save failed: {exc}")
