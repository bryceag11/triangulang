"""Extended sheaf losses: ExplicitSheafLaplacian, GeometricContrastiveLoss, CycleConsistencyLoss, TrianguLangSheafLoss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from triangulang.losses.sheaf_losses import SheafConsistencyLoss, FeatureSheafLoss, AsymmetricRestrictionSheaf

class ExplicitSheafLaplacian(nn.Module):
    """
    Builds the coboundary operator δ and sheaf Laplacian L_F explicitly.

    Given a view graph with V views and E edges (view pairs with overlap),
    constructs:
    - δ: C^0(F) -> C^1(F), the coboundary operator
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
            data: list of [N_points] tensors (one per view), scalar predictions
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