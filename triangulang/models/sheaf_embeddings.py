"""
Sheaf Embedding Modules for Multi-View Consistency

Implements explicit sheaf structure for cross-view feature consistency:
- Option 2: RestrictionMapNetworks - Learned F_ij transformations
- Option 4: ConsistentEmbeddingLoss - Simple auxiliary consistency loss

See docs/sheaf_embeddings.md for full documentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConsistentEmbeddingLoss(nn.Module):
    """
    Simple auxiliary loss for sheaf-consistent embeddings (Option 4).

    For corresponding points (p_i, p_j) across views:
        ||T_ij(e_i) - e_j||^2 should be small

    Where T_ij is a learned transformation (or identity for aligned systems).
    This is the simplest way to add explicit sheaf consistency.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        use_transform: bool = True,
        correspondence_threshold: float = 0.05,  # 5cm
        max_correspondences: int = 1000,  # Limit for memory
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.correspondence_threshold = correspondence_threshold
        self.max_correspondences = max_correspondences

        if use_transform:
            # Learn embedding transformation based on relative geometry
            self.transform = nn.Sequential(
                nn.Linear(embed_dim + 6, embed_dim),  # +6 for rel geometry
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.transform = None

    def find_correspondences(
        self,
        pts_i: torch.Tensor,  # [B, N, 3]
        pts_j: torch.Tensor,  # [B, N, 3]
        valid_i: torch.Tensor,  # [B, N]
        valid_j: torch.Tensor,  # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find corresponding points via 3D nearest neighbor."""
        B, N, _ = pts_i.shape
        device = pts_i.device

        # Compute distances
        dists = torch.cdist(pts_i, pts_j)  # [B, N, N]

        # Get mutual nearest neighbors
        nn_i_to_j = dists.argmin(dim=-1)  # [B, N]
        nn_j_to_i = dists.argmin(dim=-2)  # [B, N]

        # Check mutual consistency
        idx_i = torch.arange(N, device=device).expand(B, -1)
        mutual = (nn_j_to_i.gather(1, nn_i_to_j) == idx_i)

        # Check distance threshold
        min_dists = dists.gather(-1, nn_i_to_j.unsqueeze(-1)).squeeze(-1)
        close_enough = min_dists < self.correspondence_threshold

        # Combine validity
        valid_j_corr = valid_j.gather(1, nn_i_to_j)
        valid_corr = mutual & close_enough & valid_i & valid_j_corr

        return nn_i_to_j, valid_corr, min_dists

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, V, H, W, D] or [B, V, N, D]
        pointmaps: torch.Tensor,   # [B, V, H, W, 3] or [B, V, N, 3]
        valid_masks: Optional[torch.Tensor] = None,  # [B, V, H, W] or [B, V, N]
    ) -> torch.Tensor:
        """
        Compute consistency loss across view pairs.

        Args:
            embeddings: Feature embeddings per view
            pointmaps: 3D point positions per view
            valid_masks: Which points are valid (have depth)

        Returns:
            Scalar consistency loss
        """
        # Handle both [B, V, H, W, D] and [B, V, N, D] formats
        if embeddings.dim() == 5:
            B, V, H, W, D = embeddings.shape
            emb_flat = embeddings.reshape(B, V, -1, D)  # [B, V, HW, D]
            pts_flat = pointmaps.reshape(B, V, -1, 3)   # [B, V, HW, 3]
            if valid_masks is not None:
                valid_flat = valid_masks.reshape(B, V, -1)
            else:
                valid_flat = (pointmaps.norm(dim=-1) > 1e-6).reshape(B, V, -1)
        else:
            B, V, N, D = embeddings.shape
            emb_flat = embeddings
            pts_flat = pointmaps
            if valid_masks is not None:
                valid_flat = valid_masks
            else:
                valid_flat = pointmaps.norm(dim=-1) > 1e-6

        total_loss = 0.0
        n_pairs = 0

        for i in range(V):
            for j in range(i + 1, V):
                # Find correspondences via 3D nearest neighbor
                nn_idx, valid_corr, _ = self.find_correspondences(
                    pts_flat[:, i], pts_flat[:, j],
                    valid_flat[:, i], valid_flat[:, j]
                )

                # Skip if no valid correspondences
                if valid_corr.sum() == 0:
                    continue

                # Get corresponding embeddings
                emb_i = emb_flat[:, i]  # [B, N, D]
                emb_j = emb_flat[:, j].gather(
                    1, nn_idx.unsqueeze(-1).expand(-1, -1, D)
                )  # [B, N, D]

                # Apply transform if using one
                if self.transform is not None:
                    pts_j_corr = pts_flat[:, j].gather(
                        1, nn_idx.unsqueeze(-1).expand(-1, -1, 3)
                    )
                    rel_pos = pts_j_corr - pts_flat[:, i]
                    rel_dir = F.normalize(rel_pos, dim=-1, eps=1e-6)
                    rel_geom = torch.cat([rel_pos, rel_dir], dim=-1)
                    emb_i_transformed = self.transform(
                        torch.cat([emb_i, rel_geom], dim=-1)
                    )
                else:
                    emb_i_transformed = emb_i

                # Consistency loss: transformed embeddings should match
                diff = (emb_i_transformed - emb_j) ** 2
                pair_loss = (diff * valid_corr.unsqueeze(-1)).sum()
                pair_loss = pair_loss / (valid_corr.sum() * D + 1e-6)

                total_loss = total_loss + pair_loss
                n_pairs += 1

        if n_pairs == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return total_loss / n_pairs


class RestrictionMapNetwork(nn.Module):
    """
    Learn explicit restriction maps F_ij between view pairs (Option 2).

    For sheaf consistency: F_ij(f_i) should equal f_j for corresponding points.
    This provides a more principled sheaf structure than simple consistency loss.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 256,
        use_pair_adaptation: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # Restriction map: transforms features based on relative geometry
        # Input: feature + relative 3D offset + relative direction
        self.restriction_mlp = nn.Sequential(
            nn.Linear(feat_dim + 6, hidden_dim),  # +3 rel_pos + 3 rel_dir
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim)
        )

        # Optional: view-pair specific gating
        if use_pair_adaptation:
            self.pair_gate = nn.Sequential(
                nn.Linear(6, 64),  # From mean relative geometry
                nn.GELU(),
                nn.Linear(64, feat_dim),
                nn.Sigmoid()
            )
        else:
            self.pair_gate = None

    def forward(
        self,
        feat_i: torch.Tensor,      # [B, N, D]
        feat_j: torch.Tensor,      # [B, N, D]
        pts_i: torch.Tensor,       # [B, N, 3]
        pts_j: torch.Tensor,       # [B, N, 3]
        corr_idx: torch.Tensor,    # [B, N] indices into j
        valid_corr: torch.Tensor,  # [B, N] valid correspondences
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply restriction map and compute consistency loss.

        Returns:
            restricted_feat: F_ij(f_i) [B, N, D]
            consistency_loss: ||F_ij(f_i) - f_j||^2 for valid correspondences
        """
        B, N, D = feat_i.shape

        # Get corresponding points from j
        pts_j_corr = pts_j.gather(1, corr_idx.unsqueeze(-1).expand(-1, -1, 3))

        # Compute relative geometry
        rel_pos = pts_j_corr - pts_i  # [B, N, 3]
        rel_dir = F.normalize(rel_pos, dim=-1, eps=1e-6)  # [B, N, 3]
        rel_geom = torch.cat([rel_pos, rel_dir], dim=-1)  # [B, N, 6]

        # Apply restriction map
        restriction_input = torch.cat([feat_i, rel_geom], dim=-1)
        restricted_feat = self.restriction_mlp(restriction_input)  # [B, N, D]

        # Optional: apply pair-specific gating
        if self.pair_gate is not None:
            # Use mean geometry as pair context
            mean_geom = (rel_geom * valid_corr.unsqueeze(-1)).sum(dim=1) / (valid_corr.sum(dim=1, keepdim=True) + 1e-6)
            gate = self.pair_gate(mean_geom).unsqueeze(1)  # [B, 1, D]
            restricted_feat = restricted_feat * gate

        # Get corresponding features from j
        feat_j_corr = feat_j.gather(1, corr_idx.unsqueeze(-1).expand(-1, -1, D))

        # Consistency loss
        diff = (restricted_feat - feat_j_corr) ** 2
        consistency_loss = (diff * valid_corr.unsqueeze(-1)).sum() / (valid_corr.sum() * D + 1e-6)

        return restricted_feat, consistency_loss


class SheafEmbeddingModule(nn.Module):
    """
    Combined sheaf embedding module supporting multiple modes.

    Modes:
        - 'consistency': Simple embedding consistency loss (Option 4)
        - 'restriction': Restriction map networks (Option 2)
        - 'both': Both consistency and restriction losses
    """

    def __init__(
        self,
        embed_dim: int = 256,
        mode: str = 'consistency',
        use_transform: bool = True,
        correspondence_threshold: float = 0.05,
    ):
        super().__init__()
        self.mode = mode
        self.correspondence_threshold = correspondence_threshold

        if mode in ['consistency', 'both']:
            self.consistency_loss = ConsistentEmbeddingLoss(
                embed_dim=embed_dim,
                use_transform=use_transform,
                correspondence_threshold=correspondence_threshold,
            )
        else:
            self.consistency_loss = None

        if mode in ['restriction', 'both']:
            self.restriction_net = RestrictionMapNetwork(
                feat_dim=embed_dim,
                hidden_dim=embed_dim,
            )
        else:
            self.restriction_net = None

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, V, H, W, D]
        pointmaps: torch.Tensor,   # [B, V, H, W, 3]
        valid_masks: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute sheaf embedding losses.

        Returns:
            dict with 'consistency_loss' and/or 'restriction_loss'
        """
        outputs = {}

        if self.consistency_loss is not None:
            outputs['consistency_loss'] = self.consistency_loss(
                embeddings, pointmaps, valid_masks
            )

        if self.restriction_net is not None:
            # For restriction, we compute pairwise losses
            B, V, H, W, D = embeddings.shape
            emb_flat = embeddings.reshape(B, V, -1, D)
            pts_flat = pointmaps.reshape(B, V, -1, 3)

            if valid_masks is not None:
                valid_flat = valid_masks.reshape(B, V, -1)
            else:
                valid_flat = (pointmaps.norm(dim=-1) > 1e-6).reshape(B, V, -1)

            total_restriction_loss = 0.0
            n_pairs = 0

            for i in range(V):
                for j in range(i + 1, V):
                    # Find correspondences
                    corr_idx, valid_corr, _ = self.consistency_loss.find_correspondences(
                        pts_flat[:, i], pts_flat[:, j],
                        valid_flat[:, i], valid_flat[:, j]
                    ) if self.consistency_loss else self._find_correspondences(
                        pts_flat[:, i], pts_flat[:, j],
                        valid_flat[:, i], valid_flat[:, j]
                    )

                    if valid_corr.sum() == 0:
                        continue

                    _, loss = self.restriction_net(
                        emb_flat[:, i], emb_flat[:, j],
                        pts_flat[:, i], pts_flat[:, j],
                        corr_idx, valid_corr
                    )
                    total_restriction_loss = total_restriction_loss + loss
                    n_pairs += 1

            outputs['restriction_loss'] = total_restriction_loss / max(n_pairs, 1)

        return outputs

    def _find_correspondences(self, pts_i, pts_j, valid_i, valid_j):
        """Fallback correspondence finder if consistency_loss not available."""
        B, N, _ = pts_i.shape
        device = pts_i.device

        dists = torch.cdist(pts_i, pts_j)
        nn_i_to_j = dists.argmin(dim=-1)
        nn_j_to_i = dists.argmin(dim=-2)

        idx_i = torch.arange(N, device=device).expand(B, -1)
        mutual = (nn_j_to_i.gather(1, nn_i_to_j) == idx_i)
        min_dists = dists.gather(-1, nn_i_to_j.unsqueeze(-1)).squeeze(-1)
        close_enough = min_dists < self.correspondence_threshold

        valid_j_corr = valid_j.gather(1, nn_i_to_j)
        valid_corr = mutual & close_enough & valid_i & valid_j_corr

        return nn_i_to_j, valid_corr, min_dists


def compute_3d_localization(
    pred_masks: torch.Tensor,    # [B, 1, H, W] or [B, H, W]
    depth: torch.Tensor,         # [B, 1, H, W] or [B, H, W]
    intrinsics: torch.Tensor,    # [B, 3, 3]
    threshold: float = 0.5,
) -> dict:
    """
    Compute 3D localization from mask and depth (camera-relative).

    This is the key insight: you DON'T need camera pose estimation
    for object localization! With metric depth + mask:
        mask centroid (u, v) + depth(u, v) + intrinsics
        → back-project → (X, Y, Z) in camera frame
        → "object is 1.2m ahead, 0.3m left"

    Args:
        pred_masks: Predicted segmentation masks
        depth: Metric depth maps (in meters)
        intrinsics: Camera intrinsic matrices [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        threshold: Mask threshold for binarization

    Returns:
        dict with:
            - centroid_3d: [B, 3] camera-relative 3D centroid (X, Y, Z)
            - point_cloud: [B, N, 3] 3D points of masked region
            - centroid_2d: [B, 2] image-space centroid (u, v)
            - mean_depth: [B] average depth of masked region
    """
    # Ensure correct shapes
    if pred_masks.dim() == 4:
        pred_masks = pred_masks.squeeze(1)  # [B, H, W]
    if depth.dim() == 4:
        depth = depth.squeeze(1)  # [B, H, W]

    B, H, W = pred_masks.shape
    device = pred_masks.device

    # Binarize masks
    binary_masks = (pred_masks > threshold).float()

    # Create pixel coordinate grids
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

    # Get intrinsics
    fx = intrinsics[:, 0, 0]  # [B]
    fy = intrinsics[:, 1, 1]  # [B]
    cx = intrinsics[:, 0, 2]  # [B]
    cy = intrinsics[:, 1, 2]  # [B]

    # Back-project to 3D (camera frame)
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    Z = depth  # [B, H, W]
    X = (u_coords - cx.view(B, 1, 1)) * Z / fx.view(B, 1, 1)
    Y = (v_coords - cy.view(B, 1, 1)) * Z / fy.view(B, 1, 1)

    # Stack to 3D points
    points_3d = torch.stack([X, Y, Z], dim=-1)  # [B, H, W, 3]

    # Compute masked centroid
    mask_sum = binary_masks.sum(dim=(1, 2), keepdim=True).clamp(min=1)  # [B, 1, 1]

    # 2D centroid
    centroid_u = (u_coords * binary_masks).sum(dim=(1, 2)) / mask_sum.squeeze()
    centroid_v = (v_coords * binary_masks).sum(dim=(1, 2)) / mask_sum.squeeze()
    centroid_2d = torch.stack([centroid_u, centroid_v], dim=-1)  # [B, 2]

    # Mean depth
    mean_depth = (depth * binary_masks).sum(dim=(1, 2)) / mask_sum.squeeze()  # [B]

    # 3D centroid (weighted by mask)
    centroid_3d = (points_3d * binary_masks.unsqueeze(-1)).sum(dim=(1, 2)) / mask_sum  # [B, 3]

    # Extract point cloud for masked region
    points_flat = points_3d.reshape(B, -1, 3)  # [B, HW, 3]
    mask_flat = binary_masks.reshape(B, -1)  # [B, HW]

    # Get valid mask points (variable per batch, so use list)
    point_clouds = []
    for b in range(B):
        valid_idx = mask_flat[b] > 0.5
        pc = points_flat[b, valid_idx]  # [N_valid, 3]
        point_clouds.append(pc)

    return {
        'centroid_3d': centroid_3d,
        'centroid_2d': centroid_2d,
        'mean_depth': mean_depth,
        'point_clouds': point_clouds,  # List of [N_i, 3] tensors
        'points_3d_full': points_3d,   # [B, H, W, 3] for further processing
    }


def format_localization_text(
    centroid_3d: torch.Tensor,  # [B, 3] or [3]
    include_direction: bool = True,
) -> list:
    """
    Format 3D localization as human-readable text.

    Args:
        centroid_3d: Camera-relative 3D position (X=right, Y=down, Z=forward)
        include_direction: Include directional descriptions

    Returns:
        List of formatted strings per batch item
    """
    if centroid_3d.dim() == 1:
        centroid_3d = centroid_3d.unsqueeze(0)

    results = []
    for b in range(centroid_3d.shape[0]):
        x, y, z = centroid_3d[b].tolist()

        # Distance
        dist = (x**2 + y**2 + z**2) ** 0.5

        if include_direction:
            # Direction descriptions
            lr = "right" if x > 0 else "left"
            ud = "below" if y > 0 else "above"

            text = f"{z:.2f}m ahead"
            if abs(x) > 0.1:
                text += f", {abs(x):.2f}m {lr}"
            if abs(y) > 0.1:
                text += f", {abs(y):.2f}m {ud}"
            text += f" (total: {dist:.2f}m)"
        else:
            text = f"X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m (dist={dist:.2f}m)"

        results.append(text)

    return results
