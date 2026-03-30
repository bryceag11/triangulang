"""Sheaf-inspired losses for multi-view consistency.

Enforces consistent predictions across views for the same 3D point,
using the discrete analog of the sheaf global section condition.
Provides SheafConsistencyLoss and GeometricContrastiveLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LearnedRestrictionMap(nn.Module):
    """
    Learned restriction maps for non-trivial sheaf structure.

    In sheaf theory, restriction maps R_{v->e}: F(v) -> F(e) transform
    per-vertex data (view predictions) to the edge stalk (comparison space).
    Identity maps yield a constant sheaf; learned maps enable view-dependent
    transformations that account for occlusion, viewing angle, and depth
    uncertainty.

    Design: affine map conditioned on geometric context
        R(s, ctx) = s * sigmoid(alpha_net(ctx)) + 0.1 * beta_net(ctx)

    where ctx = [depth_normalized, correspondence_distance, displacement_norm]

    Initialized to approximate identity (alpha ≈ 1, beta ≈ 0), so training
    begins equivalent to the constant sheaf and learns deviations.
    """

    def __init__(self, context_dim: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.context_dim = context_dim

        # alpha_net: confidence weighting (sigmoid output -> (0, 1))
        # High alpha = trust this prediction; low alpha = uncertain (e.g., depth edge)
        self.alpha_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # beta_net: view-dependent bias (small, scaled by 0.1)
        self.beta_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize to approximate identity: sigmoid(2.0) ≈ 0.88, beta ≈ 0
        nn.init.zeros_(self.alpha_net[-1].weight)
        nn.init.constant_(self.alpha_net[-1].bias, 2.0)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, pred: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply learned restriction map.

        Args:
            pred: [N] scalar predictions (sigmoid probabilities in [0, 1])
            context: [N, context_dim] geometric context features

        Returns:
            transformed: [N] transformed predictions
        """
        alpha = torch.sigmoid(self.alpha_net(context).squeeze(-1))  # [N], in (0, 1)
        beta = self.beta_net(context).squeeze(-1) * 0.1  # [N], small bias
        return pred * alpha + beta

    @staticmethod
    def compute_context(
        pts: torch.Tensor,       # [N, 3] 3D points in this view
        pts_nn: torch.Tensor,    # [N, 3] nearest neighbor points in other view
        dists_nn: torch.Tensor,  # [N] distances to nearest neighbors
    ) -> torch.Tensor:
        """
        Compute geometric context features for restriction maps.

        Context features:
        1. Depth (z-coordinate, normalized): farther points have less reliable depth
        2. Correspondence distance (normalized by 10cm): quality of the 3D match
        3. Displacement norm (normalized by 10cm): geometric uncertainty proxy

        Returns:
            context: [N, 3] geometric context features
        """
        # Depth: z-coordinate, sign-preserving normalization
        depth = pts[:, 2:3]  # [N, 1]
        depth_scale = depth.abs().max().clamp(min=1e-6)
        depth_normalized = depth / depth_scale

        # Correspondence distance: how far is the nearest match (10cm = 1.0)
        corr_dist = (dists_nn / 0.1).unsqueeze(-1)  # [N, 1]

        # Displacement magnitude between matched points (geometric uncertainty)
        displacement = (pts - pts_nn).norm(dim=-1, keepdim=True) / 0.1  # [N, 1]

        return torch.cat([depth_normalized, corr_dist, displacement], dim=-1)  # [N, 3]


class SheafConsistencyLoss(nn.Module):
    """
    Penalize inconsistent predictions at geometrically corresponding points.

    This is the discrete analog of sheaf cohomology's consistency energy:
        E_cons(a) = ||δ⁰ a||² = Σ_{(i,j)} ||R^e_j a_j - B_e R^e_i a_i||²

    With identity restriction maps (default), this is a constant sheaf where
    the same 3D point should receive the same prediction. With learned
    restriction maps (restriction_map != None), the maps transform predictions
    based on geometric context before comparison, enabling view-dependent
    weighting that accounts for occlusion, depth noise, and viewing angle.

    Args:
        threshold: Distance threshold in meters for "same 3D point" (hard mode)
                   Default 10cm. After w2c->c2w fix, median cross-view error is ~7cm.
                   - <5cm: ~35% correspondences
                   - <10cm: ~65% correspondences (recommended)
                   - <30cm: ~97% correspondences
        use_soft_correspondences: Use Gaussian-weighted matching instead of hard NN
        sigma: Gaussian kernel bandwidth in meters (soft mode). Controls falloff:
               - sigma=0.05 (5cm): tight, for very accurate depth
               - sigma=0.10 (10cm): recommended (matches ~7cm median error)
               - sigma=0.20 (20cm): loose, more tolerant
        detach_target: If True, detach the target view predictions (no gradient).
                      This prevents the "fighting" problem where both views get
                      gradients to match each other, causing sheaf loss to increase.
                      Only the source view learns to match the (frozen) target.
                      DEFAULT: True (fixed 2026-01-26 — False caused loss to increase).
        symmetric_detach: If True (default), randomly swap source/target each pair
                         to avoid always detaching the same view direction.
        mutual_nn: If True (default), require mutual nearest neighbors for correspondences.
                  A pair (p_i, p_j) is valid only if p_j is NN of p_i AND p_i is NN of p_j.
                  This filters noisy correspondences from depth estimation errors.
    """

    def __init__(
        self,
        threshold: float = 0.10,  # 10cm default - with w2c->c2w fix, median error is ~7cm
        use_soft_correspondences: bool = False,
        sigma: float = 0.10,  # 10cm Gaussian bandwidth - matches corrected cross-view error
        subsample: int = 1024,  # Subsample points for memory efficiency
        detach_target: bool = True,  # Default True: prevents fighting gradients
        max_frame_distance: int = 0,  # 0 = all pairs, 1 = adjacent only, 2 = ±2 frames, etc.
        symmetric_detach: bool = True,  # Randomly swap source/target direction
        mutual_nn: bool = True,  # Require mutual nearest neighbors
        restriction_map: Optional[nn.Module] = None,  # Learned restriction maps (non-trivial sheaf)
    ):
        super().__init__()
        self.threshold = threshold
        self.use_soft_correspondences = use_soft_correspondences
        self.sigma = sigma
        self.subsample = subsample
        self.detach_target = detach_target
        self.max_frame_distance = max_frame_distance
        self.symmetric_detach = symmetric_detach
        self.mutual_nn = mutual_nn
        self.restriction_map = restriction_map
        self._failure_count = 0
        self._total_calls = 0
    
    def forward(
        self, 
        masks: torch.Tensor,      # [B, N_views, H, W] or [B, N_views, 1, H, W]
        pointmaps: torch.Tensor,  # [B, N_views, H, W, 3] world coordinates
        valid_masks: Optional[torch.Tensor] = None,  # [B, N_views, H, W] valid depth
    ) -> torch.Tensor:
        """
        Compute sheaf consistency loss across view pairs.
        
        Returns:
            Scalar loss value
        """
        # Handle different mask shapes
        if masks.dim() == 5:
            masks = masks.squeeze(2)  # [B, N, H, W]
        
        B, N, H, W = masks.shape
        device = masks.device
        
        # Flatten spatial dimensions
        masks_flat = masks.reshape(B, N, -1)  # [B, N, H*W]
        pts_flat = pointmaps.reshape(B, N, -1, 3)  # [B, N, H*W, 3]
        
        if valid_masks is not None:
            valid_flat = valid_masks.reshape(B, N, -1)  # [B, N, H*W]
        else:
            # Use norm > threshold to detect valid points
            # NOTE: pointmaps may be normalized/centered, so z > 0 check doesn't work!
            # After centering, ~50% of valid points have z <= 0
            valid_flat = (pointmaps.norm(dim=-1) > 1e-6).reshape(B, N, -1)
        
        self._total_calls += 1
        total_loss = 0.0
        n_pairs = 0

        # Iterate over view pairs (optionally restricted to nearby frames)
        for i in range(N):
            for j in range(i + 1, N):
                # Skip pairs that are too far apart in sequence
                if self.max_frame_distance > 0 and (j - i) > self.max_frame_distance:
                    continue

                pair_loss = self._compute_pair_loss(
                    masks_flat[:, i], masks_flat[:, j],
                    pts_flat[:, i], pts_flat[:, j],
                    valid_flat[:, i], valid_flat[:, j],
                )
                total_loss = total_loss + pair_loss
                n_pairs += 1

        if n_pairs == 0:
            # Return zero with gradient attached to input to keep DDP happy
            return (masks * 0.0).sum()

        return total_loss / n_pairs
    
    def _compute_pair_loss(
        self,
        mask_i: torch.Tensor,   # [B, H*W]
        mask_j: torch.Tensor,   # [B, H*W]
        pts_i: torch.Tensor,    # [B, H*W, 3]
        pts_j: torch.Tensor,    # [B, H*W, 3]
        valid_i: torch.Tensor,  # [B, H*W]
        valid_j: torch.Tensor,  # [B, H*W]
    ) -> torch.Tensor:
        """Compute consistency loss for a single view pair."""
        B = mask_i.shape[0]
        device = mask_i.device

        losses = []

        for b in range(B):
            # Get valid points
            vi = valid_i[b]
            vj = valid_j[b]

            pts_i_valid = pts_i[b, vi]  # [N_i, 3]
            pts_j_valid = pts_j[b, vj]  # [N_j, 3]
            mask_i_valid = mask_i[b, vi]  # [N_i]
            mask_j_valid = mask_j[b, vj]  # [N_j]

            if pts_i_valid.shape[0] < 10 or pts_j_valid.shape[0] < 10:
                continue

            # Subsample for memory efficiency
            if pts_i_valid.shape[0] > self.subsample:
                idx = torch.randperm(pts_i_valid.shape[0], device=device)[:self.subsample]
                pts_i_valid = pts_i_valid[idx]
                mask_i_valid = mask_i_valid[idx]

            if pts_j_valid.shape[0] > self.subsample:
                idx = torch.randperm(pts_j_valid.shape[0], device=device)[:self.subsample]
                pts_j_valid = pts_j_valid[idx]
                mask_j_valid = mask_j_valid[idx]

            # Compute pairwise distances
            dists = torch.cdist(pts_i_valid, pts_j_valid)  # [N_i, N_j]

            # Sigmoid guard: check if inputs are logits or already probabilities
            # Logits typically have values outside [0, 1]; probabilities are in [0, 1]
            with torch.no_grad():
                i_min, i_max = mask_i_valid.min().item(), mask_i_valid.max().item()
                looks_like_probs = (i_min >= -0.01 and i_max <= 1.01)
            if looks_like_probs and (i_max - i_min) < 0.5:
                # Already probabilities (or very compressed range) — don't double-sigmoid
                pred_i = mask_i_valid.clamp(0, 1)
                pred_j = mask_j_valid.clamp(0, 1)
                if not hasattr(self, '_sigmoid_warned'):
                    self._sigmoid_warned = True
                    print(f"[Sheaf] WARNING: Inputs appear to be probabilities (range [{i_min:.2f}, {i_max:.2f}]), "
                          f"skipping sigmoid to avoid double-sigmoid. If these are logits, this is a bug.")
            else:
                pred_i = torch.sigmoid(mask_i_valid)  # [N_i]
                pred_j = torch.sigmoid(mask_j_valid)  # [N_j]

            # Detach target to prevent "fighting" — only source view gets gradients.
            # With symmetric_detach, randomly pick which view is the target each pair
            # to avoid always learning in one direction.
            if self.detach_target:
                if self.symmetric_detach and torch.rand(1).item() > 0.5:
                    pred_i = pred_i.detach()
                else:
                    pred_j = pred_j.detach()

            if self.use_soft_correspondences:
                # Gaussian kernel weighting: w_ij = exp(-||p_i - p_j||^2 / (2*sigma^2))
                gaussian_weights = torch.exp(-dists.pow(2) / (2 * self.sigma ** 2))  # [N_i, N_j]

                # Hard cutoff at 3*sigma to prevent distant points from contaminating
                # the weighted average through Gaussian tail mass
                cutoff_mask = (dists <= 3.0 * self.sigma)
                gaussian_weights = gaussian_weights * cutoff_mask.float()

                # Check if any point has valid correspondences
                has_valid = (gaussian_weights.sum(dim=-1) > 1e-8)
                if has_valid.sum() < 5:
                    continue

                # Weighted average of predictions in view j for each point in view i
                weight_sums = gaussian_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [N_i, 1]
                normalized_weights = gaussian_weights / weight_sums  # [N_i, N_j]

                pred_j_weighted = torch.matmul(normalized_weights, pred_j.unsqueeze(-1)).squeeze(-1)  # [N_i]

                # Squared difference weighted by best match quality (confidence)
                diff = (pred_i - pred_j_weighted) ** 2  # [N_i]
                correspondence_confidence = gaussian_weights.max(dim=-1).values  # [N_i]

                # Only include points with valid correspondences
                diff = diff[has_valid]
                correspondence_confidence = correspondence_confidence[has_valid]

                # Debug: log soft correspondence quality periodically
                if not hasattr(self, '_soft_log_counter'):
                    self._soft_log_counter = 0
                self._soft_log_counter += 1
                if self._soft_log_counter % 5000 == 1:
                    min_dists = dists.min(dim=-1).values
                    median_dist = min_dists.median().item()
                    mean_conf = correspondence_confidence.mean().item()
                    high_conf = (correspondence_confidence > 0.5).float().mean().item() * 100
                    print(f"[Sheaf-Soft] median_dist: {median_dist:.3f}m, mean_weight: {mean_conf:.2f}, "
                          f"high_conf(>0.5): {high_conf:.0f}%, valid: {has_valid.sum().item()}/{len(has_valid)}")

                # Loss: weighted by correspondence confidence
                loss = (diff * correspondence_confidence).sum() / (correspondence_confidence.sum() + 1e-6)
            else:
                # Hard nearest neighbor matching
                min_dists_ij, min_indices_ij = dists.min(dim=-1)  # i→j: [N_i]
                pred_j_nn = pred_j[min_indices_ij]  # [N_i]
                valid_corresp = (min_dists_ij < self.threshold)

                # Mutual nearest neighbor filtering: require that j→i also maps back
                # This filters noisy correspondences from depth estimation errors
                if self.mutual_nn:
                    min_dists_ji, min_indices_ji = dists.min(dim=0)  # j→i: [N_j]
                    # For each i, check if j's NN maps back to i
                    i_indices = torch.arange(len(min_indices_ij), device=device)
                    j_matched = min_indices_ij  # Which j does each i match to?
                    i_back = min_indices_ji[j_matched]  # Which i does that j match back to?
                    is_mutual = (i_back == i_indices)
                    valid_corresp = valid_corresp & is_mutual

                valid_corresp_f = valid_corresp.float()

                # Debug: log correspondence stats periodically
                if not hasattr(self, '_log_counter'):
                    self._log_counter = 0
                self._log_counter += 1
                if self._log_counter % 5000 == 1:
                    n_valid = valid_corresp_f.sum().item()
                    n_total = len(valid_corresp_f)
                    median_dist = min_dists_ij.median().item()
                    mutual_str = " (mutual NN)" if self.mutual_nn else ""
                    print(f"[Sheaf] corresp: {n_valid:.0f}/{n_total} ({100*n_valid/n_total:.1f}%){mutual_str}, "
                          f"median_dist: {median_dist:.3f}m")

                if valid_corresp_f.sum() < 5:
                    continue

                # Apply learned restriction maps if available
                if self.restriction_map is not None:
                    # Compute geometric context for both views
                    ctx_i = LearnedRestrictionMap.compute_context(
                        pts_i_valid, pts_j_valid[min_indices_ij], min_dists_ij
                    )
                    ctx_j = LearnedRestrictionMap.compute_context(
                        pts_j_valid[min_indices_ij], pts_i_valid, min_dists_ij
                    )
                    # Transform predictions through restriction maps
                    pred_i_mapped = self.restriction_map(pred_i, ctx_i)
                    pred_j_mapped = self.restriction_map(pred_j_nn, ctx_j)
                    diff = (pred_i_mapped - pred_j_mapped) ** 2
                else:
                    # Identity restriction maps (constant sheaf)
                    diff = (pred_i - pred_j_nn) ** 2

                # Weighted by valid correspondences
                loss = (diff * valid_corresp_f).sum() / (valid_corresp_f.sum() + 1e-6)

            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()


class AsymmetricRestrictionSheaf(nn.Module):
    """
    Non-constant sheaf with asymmetric, direction-dependent restriction maps.

    Unlike LearnedRestrictionMap (which uses a single map for both directions),
    this module uses separate networks for source→edge and target→edge
    transformations: R_{i→e}(s_i, ctx_i) via alpha_net_source, and
    R_{j→e}(s_j, ctx_j) via alpha_net_target.

    Context is expanded beyond LearnedRestrictionMap with viewing angle proxy
    and depth edge indicator (5D total).

    This is a genuinely non-constant sheaf because:
    - Different views get different transformations (asymmetric maps)
    - The maps depend on per-point geometric context (not global)
    - Source and target directions have independent learned parameters

    Initialized near-identity so training starts from the constant sheaf.
    """

    def __init__(self, context_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.context_dim = context_dim

        # Source → edge restriction map
        self.alpha_net_source = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.beta_net_source = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Target → edge restriction map (separate parameters)
        self.alpha_net_target = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.beta_net_target = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize near-identity: sigmoid(2.0) ≈ 0.88, beta ≈ 0
        for net in [self.alpha_net_source, self.alpha_net_target]:
            nn.init.zeros_(net[-1].weight)
            nn.init.constant_(net[-1].bias, 2.0)
        for net in [self.beta_net_source, self.beta_net_target]:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward_source(self, pred: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply source → edge restriction map. pred: [N], context: [N, 5]."""
        alpha = torch.sigmoid(self.alpha_net_source(context).squeeze(-1))
        beta = self.beta_net_source(context).squeeze(-1) * 0.1
        return pred * alpha + beta

    def forward_target(self, pred: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply target → edge restriction map. pred: [N], context: [N, 5]."""
        alpha = torch.sigmoid(self.alpha_net_target(context).squeeze(-1))
        beta = self.beta_net_target(context).squeeze(-1) * 0.1
        return pred * alpha + beta

    @staticmethod
    def compute_context(
        pts: torch.Tensor,       # [N, 3]
        pts_nn: torch.Tensor,    # [N, 3]
        dists_nn: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """
        Compute 5D geometric context: depth, corr_dist, displacement,
        viewing_angle_proxy, depth_edge_indicator.
        """
        # 1. Depth (normalized)
        depth = pts[:, 2:3]
        depth_scale = depth.abs().max().clamp(min=1e-6)
        depth_normalized = depth / depth_scale

        # 2. Correspondence distance (10cm = 1.0)
        corr_dist = (dists_nn / 0.1).unsqueeze(-1)

        # 3. Displacement magnitude
        displacement = (pts - pts_nn).norm(dim=-1, keepdim=True) / 0.1

        # 4. Viewing angle proxy: ratio of lateral to depth displacement
        lateral_disp = (pts[:, :2] - pts_nn[:, :2]).norm(dim=-1, keepdim=True)
        depth_disp = (pts[:, 2:3] - pts_nn[:, 2:3]).abs().clamp(min=1e-6)
        viewing_angle = torch.atan2(lateral_disp, depth_disp) / (3.14159 / 2)  # normalized to [0, 1]

        # 5. Depth edge indicator: large depth gradient suggests occlusion boundary
        depth_edge = (depth_disp / depth_scale).clamp(max=1.0)

        return torch.cat([depth_normalized, corr_dist, displacement,
                          viewing_angle, depth_edge], dim=-1)  # [N, 5]


class FeatureSheafLoss(nn.Module):
    """
    Non-constant sheaf on feature vectors with learned linear restriction maps.

    This is the most mathematically genuine sheaf formulation:
    - Stalks: F(v_i) = R^d_stalk (feature vectors at corresponding pixels)
    - Edge space: F(e_ij) = R^d_edge (shared comparison space)
    - Restriction maps: R_{v→e}(f, ctx) = Linear(f) * sigmoid(GateNet(ctx))

    The gate varies per-point based on geometry, making this a non-constant
    sheaf: different views and different spatial locations get different
    effective projections into the comparison space.

    Loss: ||R_i(f_i, ctx_i) - R_j(f_j, ctx_j)||^2 in the edge space.

    Args:
        d_stalk: Dimension of input feature vectors (SAM3 features)
        d_edge: Dimension of the edge comparison space
        context_dim: Dimension of geometric context
        gate_hidden: Hidden dim for the gating network
    """

    def __init__(
        self,
        d_stalk: int = 256,
        d_edge: int = 32,
        context_dim: int = 5,
        gate_hidden: int = 16,
    ):
        super().__init__()
        self.d_stalk = d_stalk
        self.d_edge = d_edge

        # Shared linear projection: stalk → edge space
        self.projection = nn.Linear(d_stalk, d_edge, bias=True)

        # Per-point gate conditioned on geometric context
        # Output: d_edge-dimensional gate vector
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, d_edge),
        )

        # Initialize projection to preserve scale
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Initialize gate near 1 (sigmoid(2) ≈ 0.88) so starts near linear projection
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, 2.0)

    def restriction_map(
        self,
        features: torch.Tensor,   # [N, d_stalk]
        context: torch.Tensor,    # [N, context_dim]
    ) -> torch.Tensor:
        """
        Apply restriction map: R(f, ctx) = Linear(f) * sigmoid(Gate(ctx)).
        Returns: [N, d_edge]
        """
        projected = self.projection(features)  # [N, d_edge]
        gate = torch.sigmoid(self.gate_net(context))  # [N, d_edge]
        return projected * gate

    def forward(
        self,
        features_i: torch.Tensor,  # [N, d_stalk]
        features_j: torch.Tensor,  # [N, d_stalk]
        context_i: torch.Tensor,   # [N, context_dim]
        context_j: torch.Tensor,   # [N, context_dim]
    ) -> torch.Tensor:
        """
        Compute sheaf consistency loss in feature space.

        Both inputs should be at corresponding 3D points (already matched).
        Returns scalar loss.
        """
        mapped_i = self.restriction_map(features_i, context_i)  # [N, d_edge]
        mapped_j = self.restriction_map(features_j, context_j)  # [N, d_edge]
        return (mapped_i - mapped_j).pow(2).mean()


class ExplicitSheafLaplacian(nn.Module):
    """
    Builds the coboundary operator δ and sheaf Laplacian L_F explicitly.

    Given a view graph with V views and E edges (view pairs with overlap),
    constructs:
    - δ: C^0(F) → C^1(F), the coboundary operator
    - L_F = δ^T δ, the sheaf Laplacian
    - Energy = s^T L_F s = ||δ(s)||^2

    Can compute eigenvalues to analyze sheaf cohomology:
    - H^0 = ker(L_F) = global sections (consistent predictions)
    - dim(H^0) > 0 means some predictions are globally consistent

    This directly connects to the sheaf theory literature and provides
    verifiable spectral properties.

    Args:
        n_views: Number of views
        restriction_type: 'scalar' uses AsymmetricRestrictionSheaf,
                          'feature' uses FeatureSheafLoss restriction maps
        d_stalk: Feature dimension (only for restriction_type='feature')
        d_edge: Edge space dimension (only for restriction_type='feature')
        context_dim: Geometric context dimension
    """

    def __init__(
        self,
        n_views: int = 4,
        restriction_type: str = 'scalar',
        d_stalk: int = 256,
        d_edge: int = 32,
        context_dim: int = 5,
    ):
        super().__init__()
        self.n_views = n_views
        self.restriction_type = restriction_type
        self.d_stalk = d_stalk
        self.d_edge = d_edge

        if restriction_type == 'scalar':
            self.restriction = AsymmetricRestrictionSheaf(
                context_dim=context_dim,
            )
        elif restriction_type == 'feature':
            self.restriction = FeatureSheafLoss(
                d_stalk=d_stalk,
                d_edge=d_edge,
                context_dim=context_dim,
            )
        else:
            raise ValueError(f"Unknown restriction_type: {restriction_type}")

    def build_coboundary_energy(
        self,
        data: list,
        contexts: list,
    ) -> torch.Tensor:
        """
        Compute sheaf Laplacian energy ||δ(s)||^2 = s^T L_F s.

        For scalar sheaf:
            data: list of [N_points] tensors (one per view) — scalar predictions
            contexts: list of (ctx_source, ctx_target) pairs per edge

        For feature sheaf:
            data: list of [N_points, d_stalk] tensors (one per view)
            contexts: list of (ctx_source, ctx_target) pairs per edge

        Returns scalar energy.
        """
        n_views = len(data)
        energy = torch.tensor(0.0, device=data[0].device)
        n_edges = 0

        edge_idx = 0
        for i in range(n_views):
            for j in range(i + 1, n_views):
                if edge_idx >= len(contexts):
                    break
                ctx_i, ctx_j = contexts[edge_idx]
                edge_idx += 1

                if self.restriction_type == 'scalar':
                    mapped_i = self.restriction.forward_source(data[i], ctx_i)
                    mapped_j = self.restriction.forward_target(data[j], ctx_j)
                    edge_energy = (mapped_i - mapped_j).pow(2).mean()
                else:
                    mapped_i = self.restriction.restriction_map(data[i], ctx_i)
                    mapped_j = self.restriction.restriction_map(data[j], ctx_j)
                    edge_energy = (mapped_i - mapped_j).pow(2).mean()

                energy = energy + edge_energy
                n_edges += 1

        if n_edges > 0:
            energy = energy / n_edges
        return energy

    def build_laplacian_matrix(
        self,
        n_points: int,
        contexts: list,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Build the explicit sheaf Laplacian matrix L_F = δ^T δ for spectral analysis.

        Only works for scalar sheaf (restriction_type='scalar').
        Returns L_F as a dense matrix of shape [n_views * n_points, n_views * n_points].

        This is intended for small examples (spectral analysis / verification),
        not for training on large data.
        """
        if self.restriction_type != 'scalar':
            raise NotImplementedError(
                "Explicit Laplacian matrix only implemented for scalar sheaf. "
                "For feature sheaf, use build_coboundary_energy() directly."
            )

        n_views = self.n_views
        N = n_views * n_points
        if device is None:
            device = next(self.parameters()).device

        L = torch.zeros(N, N, device=device)

        edge_idx = 0
        for i in range(n_views):
            for j in range(i + 1, n_views):
                if edge_idx >= len(contexts):
                    break
                ctx_i, ctx_j = contexts[edge_idx]
                edge_idx += 1

                for p in range(n_points):
                    # Get scalar restriction map values at this point
                    # For point p in view i: R_source(1, ctx_i[p])
                    dummy_one = torch.ones(1, device=device)
                    ctx_ip = ctx_i[p:p+1]
                    ctx_jp = ctx_j[p:p+1]

                    r_i = self.restriction.forward_source(dummy_one, ctx_ip).item()
                    r_j = self.restriction.forward_target(dummy_one, ctx_jp).item()

                    # Coboundary: δ(s)_e = r_i * s_i - r_j * s_j
                    # L_F contribution: (r_i * s_i - r_j * s_j)^2
                    # = r_i^2 * s_i^2 - 2 * r_i * r_j * s_i * s_j + r_j^2 * s_j^2
                    idx_i = i * n_points + p
                    idx_j = j * n_points + p

                    L[idx_i, idx_i] += r_i ** 2
                    L[idx_j, idx_j] += r_j ** 2
                    L[idx_i, idx_j] -= r_i * r_j
                    L[idx_j, idx_i] -= r_i * r_j

        return L

    def compute_spectrum(
        self,
        n_points: int,
        contexts: list,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Compute eigenvalues of the sheaf Laplacian for spectral analysis.

        Returns sorted eigenvalues. Properties:
        - All eigenvalues >= 0 (L_F is positive semidefinite)
        - Number of zero eigenvalues = dim(H^0) = number of global sections
        - For constant sheaf on connected graph: dim(H^0) = 1
        - For non-constant sheaf: dim(H^0) can vary
        """
        L = self.build_laplacian_matrix(n_points, contexts, device)
        eigenvalues = torch.linalg.eigvalsh(L)
        return eigenvalues


class GeometricContrastiveLoss(nn.Module):
    """
    Contrastive learning with geometric correspondences as positive pairs.
    
    Unlike standard contrastive learning (SimCLR, MoCo) which uses augmentation,
    we use 3D geometry to define positives: features at the same 3D location
    across different views should be similar.
    
    This learns restriction maps that minimize sheaf consistency energy
    while maximizing discriminability.
    
    Args:
        temperature: InfoNCE temperature
        n_anchors: Number of anchor points to sample per batch
        threshold: Distance threshold for positive pairs
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        n_anchors: int = 256,
        threshold: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.n_anchors = n_anchors
        self.threshold = threshold
    
    def forward(
        self,
        features: torch.Tensor,   # [B, N_views, H, W, D] or [B, N_views, D, H, W]
        pointmaps: torch.Tensor,  # [B, N_views, H, W, 3]
        valid_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute geometric contrastive loss.
        
        Returns:
            Scalar loss value
        """
        # Handle channel-first format
        if features.shape[-1] != features.shape[-2]:  # Not square, likely [B, N, D, H, W]
            features = features.permute(0, 1, 3, 4, 2)  # -> [B, N, H, W, D]
        
        B, N, H, W, D = features.shape
        device = features.device
        
        # Flatten spatial dimensions
        feats_flat = features.reshape(B, N, H*W, D)  # [B, N, HW, D]
        pts_flat = pointmaps.reshape(B, N, H*W, 3)   # [B, N, HW, 3]
        
        if valid_masks is not None:
            valid_flat = valid_masks.reshape(B, N, -1)
        else:
            # Use norm > threshold (pointmaps may be normalized/centered)
            valid_flat = (pointmaps.norm(dim=-1) > 1e-6).reshape(B, N, -1)
        
        total_loss = 0.0
        n_pairs = 0
        
        # Use view 0 as anchor, compute loss against other views
        for v in range(1, N):
            pair_loss = self._compute_pair_contrastive(
                feats_flat[:, 0], feats_flat[:, v],
                pts_flat[:, 0], pts_flat[:, v],
                valid_flat[:, 0], valid_flat[:, v],
            )
            total_loss = total_loss + pair_loss
            n_pairs += 1
        
        return total_loss / max(n_pairs, 1)
    
    def _compute_pair_contrastive(
        self,
        feats_i: torch.Tensor,  # [B, HW, D]
        feats_j: torch.Tensor,  # [B, HW, D]
        pts_i: torch.Tensor,    # [B, HW, 3]
        pts_j: torch.Tensor,    # [B, HW, 3]
        valid_i: torch.Tensor,  # [B, HW]
        valid_j: torch.Tensor,  # [B, HW]
    ) -> torch.Tensor:
        """Compute contrastive loss for a single view pair."""
        B, HW, D = feats_i.shape
        device = feats_i.device
        
        losses = []
        
        for b in range(B):
            vi = valid_i[b]
            vj = valid_j[b]
            
            pts_i_valid = pts_i[b, vi]
            pts_j_valid = pts_j[b, vj]
            feats_i_valid = feats_i[b, vi]
            feats_j_valid = feats_j[b, vj]
            
            N_i, N_j = pts_i_valid.shape[0], pts_j_valid.shape[0]
            
            if N_i < 10 or N_j < 10:
                continue
            
            # Sample anchor points
            n_anchors = min(self.n_anchors, N_i)
            anchor_idx = torch.randperm(N_i, device=device)[:n_anchors]
            
            anchor_pts = pts_i_valid[anchor_idx]    # [n_anchors, 3]
            anchor_feats = feats_i_valid[anchor_idx]  # [n_anchors, D]
            
            # Find positive: nearest 3D point in view j
            dists_3d = torch.cdist(anchor_pts, pts_j_valid)  # [n_anchors, N_j]
            min_dists, pos_idx = dists_3d.min(dim=-1)  # [n_anchors]
            
            # Only use anchors with valid correspondences
            valid_anchors = min_dists < self.threshold
            if valid_anchors.sum() < 5:
                continue
            
            anchor_feats = anchor_feats[valid_anchors]
            pos_feats = feats_j_valid[pos_idx[valid_anchors]]
            
            # Normalize features
            anchor_norm = F.normalize(anchor_feats, dim=-1)
            pos_norm = F.normalize(pos_feats, dim=-1)
            all_norm = F.normalize(feats_j_valid, dim=-1)
            
            # Positive similarity
            pos_sim = (anchor_norm * pos_norm).sum(-1) / self.temperature  # [n_valid]
            
            # All similarities (for denominator)
            all_sim = torch.mm(anchor_norm, all_norm.t()) / self.temperature  # [n_valid, N_j]
            
            # InfoNCE loss
            loss = -pos_sim + torch.logsumexp(all_sim, dim=-1)
            losses.append(loss.mean())
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(losses).mean()


class CycleConsistencyLoss(nn.Module):
    """
    Penalize violations of the cocycle condition in view triplets.
    
    For a true sheaf, composing restriction maps around a cycle should
    return to the starting point: R_{ki} R_{jk} R_{ij} = I
    
    This loss measures deviation from this condition.
    
    In practice: if we warp features i -> j -> k -> i, we should return
    to the original features.
    
    Args:
        threshold: Distance threshold for correspondences
    """
    
    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold
    
    def forward(
        self,
        features: torch.Tensor,   # [B, N_views, H, W, D]
        pointmaps: torch.Tensor,  # [B, N_views, H, W, 3]
    ) -> torch.Tensor:
        """
        Compute cycle consistency loss over view triplets.
        """
        B, N, H, W, D = features.shape
        
        if N < 3:
            return torch.tensor(0.0, device=features.device)
        
        # Sample triplets
        total_loss = 0.0
        n_triplets = 0
        
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    triplet_loss = self._compute_triplet_loss(
                        features[:, i], features[:, j], features[:, k],
                        pointmaps[:, i], pointmaps[:, j], pointmaps[:, k],
                    )
                    total_loss = total_loss + triplet_loss
                    n_triplets += 1
        
        return total_loss / max(n_triplets, 1)
    
    def _compute_triplet_loss(
        self,
        feat_i, feat_j, feat_k,
        pts_i, pts_j, pts_k,
    ) -> torch.Tensor:
        """Compute cycle loss for a single triplet."""
        # This is computationally expensive, so we use a simplified version:
        # Check if features at corresponding 3D points form a consistent triangle
        
        B, H, W, D = feat_i.shape
        device = feat_i.device
        
        # Flatten
        feat_i = feat_i.reshape(B, -1, D)
        feat_j = feat_j.reshape(B, -1, D)
        feat_k = feat_k.reshape(B, -1, D)
        pts_i = pts_i.reshape(B, -1, 3)
        pts_j = pts_j.reshape(B, -1, 3)
        pts_k = pts_k.reshape(B, -1, 3)
        
        losses = []
        
        for b in range(B):
            # Sample anchor points from view i
            n_anchors = min(128, pts_i.shape[1])
            anchor_idx = torch.randperm(pts_i.shape[1], device=device)[:n_anchors]
            
            anchor_pts = pts_i[b, anchor_idx]
            anchor_feats = feat_i[b, anchor_idx]
            
            # Find correspondences: i -> j
            dists_ij = torch.cdist(anchor_pts, pts_j[b])
            min_dists_ij, idx_j = dists_ij.min(dim=-1)
            
            # Find correspondences: j -> k
            pts_j_corresp = pts_j[b, idx_j]
            dists_jk = torch.cdist(pts_j_corresp, pts_k[b])
            min_dists_jk, idx_k = dists_jk.min(dim=-1)
            
            # Find correspondences: k -> i (should return to anchor)
            pts_k_corresp = pts_k[b, idx_k]
            dists_ki = torch.cdist(pts_k_corresp, anchor_pts)
            min_dists_ki, idx_i_return = dists_ki.min(dim=-1)
            
            # Valid cycles: all correspondences within threshold
            valid = ((min_dists_ij < self.threshold) & 
                     (min_dists_jk < self.threshold) & 
                     (min_dists_ki < self.threshold))
            
            if valid.sum() < 5:
                continue
            
            # Cycle consistency: features at i should match features at i after cycle
            feat_i_return = feat_i[b, idx_i_return[valid]]
            feat_i_orig = anchor_feats[valid]
            
            # Features should be similar if they're the same 3D point
            loss = F.mse_loss(F.normalize(feat_i_return, dim=-1),
                              F.normalize(feat_i_orig, dim=-1))
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(losses).mean()


# Combined Loss for TrianguLang

class TrianguLangSheafLoss(nn.Module):
    """
    Combined loss function for TrianguLang with sheaf-inspired consistency.
    
    Total loss = λ_seg * L_seg + λ_sheaf * L_sheaf + λ_contrast * L_contrast
    
    Args:
        lambda_sheaf: Weight for sheaf consistency loss
        lambda_contrast: Weight for contrastive loss  
        lambda_cycle: Weight for cycle consistency loss (optional, expensive)
        threshold: Distance threshold for correspondences (meters)
    """
    
    def __init__(
        self,
        lambda_sheaf: float = 0.1,
        lambda_contrast: float = 0.05,
        lambda_cycle: float = 0.0,  # Disabled by default (expensive)
        threshold: float = 0.05,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        self.lambda_sheaf = lambda_sheaf
        self.lambda_contrast = lambda_contrast
        self.lambda_cycle = lambda_cycle
        
        self.sheaf_loss = SheafConsistencyLoss(threshold=threshold)
        self.contrast_loss = GeometricContrastiveLoss(temperature=temperature, threshold=threshold)
        
        if lambda_cycle > 0:
            self.cycle_loss = CycleConsistencyLoss(threshold=threshold)
        else:
            self.cycle_loss = None
    
    def forward(
        self,
        pred_masks: torch.Tensor,   # [B, N, H, W]
        pointmaps: torch.Tensor,    # [B, N, H, W, 3]
        features: Optional[torch.Tensor] = None,  # [B, N, H, W, D] for contrastive
        valid_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute all sheaf-inspired losses.
        
        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary with individual loss values for logging
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Sheaf consistency loss (always computed)
        if self.lambda_sheaf > 0:
            l_sheaf = self.sheaf_loss(pred_masks, pointmaps, valid_masks)
            loss_dict['loss_sheaf'] = l_sheaf.item()
            total_loss = total_loss + self.lambda_sheaf * l_sheaf
        
        # Contrastive loss (requires features)
        if self.lambda_contrast > 0 and features is not None:
            l_contrast = self.contrast_loss(features, pointmaps, valid_masks)
            loss_dict['loss_contrast'] = l_contrast.item()
            total_loss = total_loss + self.lambda_contrast * l_contrast
        
        # Cycle consistency loss (optional, expensive)
        if self.lambda_cycle > 0 and features is not None and self.cycle_loss is not None:
            l_cycle = self.cycle_loss(features, pointmaps)
            loss_dict['loss_cycle'] = l_cycle.item()
            total_loss = total_loss + self.lambda_cycle * l_cycle
        
        return total_loss, loss_dict


# Example usage in training loop

if __name__ == "__main__":
    # Example dimensions
    B, N, H, W, D = 2, 4, 64, 64, 256
    
    # Simulated data
    pred_masks = torch.randn(B, N, H, W)
    pointmaps = torch.randn(B, N, H, W, 3)
    features = torch.randn(B, N, H, W, D)
    
    # Initialize loss
    sheaf_losses = TrianguLangSheafLoss(
        lambda_sheaf=0.1,
        lambda_contrast=0.05,
        lambda_cycle=0.0,  # Disabled by default
        threshold=0.05,
    )
    
    # Compute
    total_loss, loss_dict = sheaf_losses(pred_masks, pointmaps, features)
    
    print(f"Total sheaf loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")