import torch
import torch.nn.functional as F
from collections import defaultdict


def spatial_ranking_loss(pred_masks, depth, labels, margin=0.5):
    """Learned spatial ranking loss for multi-instance objects.

    For each group of same-label objects, creates pairwise ranking targets
    based on depth (nearest/farthest) and position (left/right/top/bottom).
    Uses margin ranking loss so the model learns to produce masks whose
    centroids match the correct spatial ordering.

    Args:
        pred_masks: [K, H, W] predicted mask logits for K objects
        depth: [1, H, W] or [H, W] depth map
        labels: list of K label strings
        margin: ranking margin (default 0.5)

    Returns:
        scalar loss (0 if no multi-instance groups)
    """
    K, H, W = pred_masks.shape
    device = pred_masks.device

    if depth.dim() == 3:
        depth = depth.squeeze(0)  # [H, W]

    # Group objects by label
    label_groups = defaultdict(list)
    for k, label in enumerate(labels):
        # Strip spatial qualifiers to get base label
        base = label
        for prefix in ['nearest ', 'farthest ', 'leftmost ', 'rightmost ',
                        'topmost ', 'bottommost ', 'closest ', 'center ']:
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        label_groups[base].append(k)

    # Only process groups with 2+ instances
    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    # Compute soft centroids for all objects
    mask_probs = torch.sigmoid(pred_masks)  # [K, H, W]
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)

    mask_sum = mask_probs.sum(dim=(-2, -1)).clamp(min=1e-6)  # [K]
    # Weighted centroid coordinates
    centroid_y = (mask_probs * y_coords.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum  # [K]
    centroid_x = (mask_probs * x_coords.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum  # [K]

    # Depth at centroid via bilinear sampling
    cy_norm = (centroid_y / (H - 1)) * 2 - 1  # [-1, 1]
    cx_norm = (centroid_x / (W - 1)) * 2 - 1
    grid = torch.stack([cx_norm, cy_norm], dim=-1).view(1, K, 1, 2)  # [1, K, 1, 2]
    depth_at_cent = F.grid_sample(
        depth.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        grid, mode='bilinear', padding_mode='border', align_corners=True
    ).view(K)  # [K]

    ranking_loss_fn = torch.nn.MarginRankingLoss(margin=margin, reduction='mean')

    for base_label, indices in label_groups.items():
        if len(indices) < 2:
            continue

        # Create pairwise ranking targets for depth (smaller = nearer)
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                ki, kj = indices[i_idx], indices[j_idx]

                # Depth ranking: if depth[ki] < depth[kj], ki is nearer
                # MarginRankingLoss: loss(x1, x2, y) where y=1 means x1 should be ranked higher
                di = depth_at_cent[ki].unsqueeze(0)
                dj = depth_at_cent[kj].unsqueeze(0)
                # Target: -1 if di < dj (ki nearer), +1 if dj < di (kj nearer)
                with torch.no_grad():
                    depth_target = torch.sign(dj - di).clamp(-1, 1)
                    if depth_target.item() == 0:
                        depth_target = torch.tensor([1.0], device=device)
                total_loss = total_loss + ranking_loss_fn(di, dj, depth_target)
                n_pairs += 1

                # X-position ranking (left-right)
                xi = centroid_x[ki].unsqueeze(0)
                xj = centroid_x[kj].unsqueeze(0)
                with torch.no_grad():
                    x_target = torch.sign(xj - xi).clamp(-1, 1)
                    if x_target.item() == 0:
                        x_target = torch.tensor([1.0], device=device)
                total_loss = total_loss + ranking_loss_fn(xi, xj, x_target)
                n_pairs += 1

    if n_pairs > 0:
        return total_loss / n_pairs
    return total_loss


def spatial_selection_loss(pred_masks, gt_masks, depth, labels, spatial_indices):
    """Spatial selection loss: trains spatial tokens to select the correct instance.

    When a spatial qualifier is active (e.g., "nearest chair" with spatial_idx=1),
    this loss checks whether the predicted mask for that object actually corresponds
    to the spatially correct GT instance (the nearest one by depth).

    Uses cross-entropy: for each spatially-qualified object, creates a target
    distribution over GT instances based on the qualifier, and penalizes the
    model's mask if it matches the wrong instance.

    Args:
        pred_masks: [K, H, W] predicted mask logits (one per object)
        gt_masks: [K, H, W] GT masks (one per object)
        depth: [1, H, W] or [H, W] depth map
        labels: list of K label strings (may include spatial prefixes)
        spatial_indices: [K] tensor of spatial qualifier indices (0=none, 1=nearest, etc.)

    Returns:
        scalar loss (0 if no spatial qualifiers active)
    """
    K, H, W = pred_masks.shape
    device = pred_masks.device

    if depth.dim() == 3:
        depth = depth.squeeze(0)

    # Strip spatial prefixes to get base labels
    base_labels = []
    for label in labels:
        base = label
        for prefix in ['nearest ', 'farthest ', 'leftmost ', 'rightmost ',
                        'topmost ', 'bottommost ', 'closest ', 'center ']:
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        base_labels.append(base)

    # Group objects by base label
    label_groups = defaultdict(list)
    for k, base in enumerate(base_labels):
        label_groups[base].append(k)

    total_loss = torch.tensor(0.0, device=device)
    n_spatial = 0

    # Compute GT mask centroids and depths (detached — these are targets)
    with torch.no_grad():
        gt_binary = (gt_masks > 0.5).float()
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
        x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)

        gt_sum = gt_binary.sum(dim=(-2, -1)).clamp(min=1e-6)
        gt_cy = (gt_binary * y_coords.unsqueeze(0)).sum(dim=(-2, -1)) / gt_sum
        gt_cx = (gt_binary * x_coords.unsqueeze(0)).sum(dim=(-2, -1)) / gt_sum

        # Depth at GT centroids
        cy_norm = (gt_cy / (H - 1)) * 2 - 1
        cx_norm = (gt_cx / (W - 1)) * 2 - 1
        grid = torch.stack([cx_norm, cy_norm], dim=-1).view(1, K, 1, 2)
        gt_depth = F.grid_sample(
            depth.unsqueeze(0).unsqueeze(0), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        ).view(K)

    # For each spatially-qualified object, compute IoU of its prediction against
    # ALL same-label GT masks, then penalize if the best-matching GT isn't the
    # spatially correct one
    for base, indices in label_groups.items():
        if len(indices) < 2:
            continue

        # Find which indices have spatial qualifiers
        for k in indices:
            sq = spatial_indices[k].item() if isinstance(spatial_indices, torch.Tensor) else spatial_indices[k]
            if sq == 0:
                continue  # No spatial qualifier

            # Determine which GT instance is the "correct" one for this qualifier
            group_depths = gt_depth[indices]
            group_cx = gt_cx[indices]
            group_cy = gt_cy[indices]
            group_valid = gt_sum[indices] > 1  # Has visible GT

            if group_valid.sum() < 2:
                continue

            # Find target index within group
            # Table-driven spatial target selection
            spatial_dispatch = {
                1: (group_depths, True, 1e6),     # nearest
                2: (group_depths, False, -1e6),    # farthest
                3: (group_cx, True, 1e6),          # leftmost
                4: (group_cx, False, -1e6),         # rightmost
                5: (group_cy, True, 1e6),          # topmost
                6: (group_cy, False, -1e6),         # bottommost
            }
            if sq not in spatial_dispatch:
                continue
            values, find_min, sentinel = spatial_dispatch[sq]
            masked = values.clone()
            masked[~group_valid] = sentinel
            target_in_group = (masked.argmin() if find_min else masked.argmax()).item()

            target_k = indices[target_in_group]

            # Compute IoU of this prediction against each same-label GT
            pred_prob = torch.sigmoid(pred_masks[k])  # [H, W]
            ious = []
            for j in indices:
                if not group_valid[indices.index(j)]:
                    ious.append(torch.tensor(0.0, device=device))
                    continue
                gt_j = gt_binary[j]
                inter = (pred_prob * gt_j).sum()
                union = pred_prob.sum() + gt_j.sum() - inter
                ious.append(inter / union.clamp(min=1e-6))
            ious = torch.stack(ious)  # [num_group]

            # Cross-entropy: prediction should best match the spatially-correct GT
            # Treat IoUs as logits, target is the correct instance index
            target_idx = torch.tensor(target_in_group, device=device, dtype=torch.long)
            # Scale ious to make softmax sharper
            selection_loss = F.cross_entropy(ious.unsqueeze(0) * 10.0, target_idx.unsqueeze(0))
            total_loss = total_loss + selection_loss
            n_spatial += 1

    if n_spatial > 0:
        return total_loss / n_spatial
    return total_loss
