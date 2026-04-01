"""Geometry-Aware Semantic Attention (GASA) module.

Attention mechanism that biases cross-view feature matching using 3D geometric
distance, preventing semantically similar but spatially distant matches.
Includes PointmapComputer, WorldSpacePositionalEncoding, and the GASA layer.

Positional encoding classes (PluckerEmbedding, RayRoPE3D,
WorldSpacePositionalEncoding, CameraRelativePositionalEncoding) live in
triangulang.models.positional_encodings and are re-exported here for
backward compatibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Backward-compatible re-exports from positional_encodings
from triangulang.models.positional_encodings import (
    PluckerEmbedding,
    RayRoPE3D,
    WorldSpacePositionalEncoding,
    CameraRelativePositionalEncoding,
)

__all__ = [
    'PointmapComputer',
    'GeometryAwareSemanticAttention',
    'GASABlock',
    'GASAEncoder',
    'SymmetricCentroidHead',
    # re-exports
    'PluckerEmbedding',
    'RayRoPE3D',
    'WorldSpacePositionalEncoding',
    'CameraRelativePositionalEncoding',
]


class PointmapComputer(nn.Module):
    """
    Compute world-coordinate pointmaps from depth, poses, and intrinsics.

    Converts per-pixel depth values into 3D world coordinates using camera parameters.
    This enables spatial reasoning across views - pixels at the same 3D location
    will have similar coordinates regardless of which view they appear in.

    Math:
        P_cam = d * K^{-1} * [u, v, 1]^T    # Unproject to camera coords
        P_world = R * P_cam + t              # Transform to world coords
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        depths: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute pointmaps from depth, poses, and intrinsics.

        Args:
            depths: [B, N, H, W] or [B*N, 1, H, W] depth maps
            poses: [B, N, 4, 4] or [B*N, 4, 4] camera-to-world transforms
            intrinsics: [B, N, 3, 3] or [B*N, 3, 3] camera intrinsics
            normalize: If True, center and scale pointmaps for numerical stability

        Returns:
            pointmaps: [B, N, H, W, 3] world coordinates for each pixel
        """
        # Handle different input shapes
        if depths.dim() == 4 and depths.shape[1] == 1:
            # [B*N, 1, H, W] -> need to infer B and N
            BN, _, H, W = depths.shape
            depths = depths.squeeze(1)  # [B*N, H, W]

            if poses.dim() == 3:  # [B*N, 4, 4]
                N = 1  # Assume single view per batch element
                B = BN
            else:  # [B, N, 4, 4]
                B, N = poses.shape[:2]
                poses = poses.view(B * N, 4, 4)
                intrinsics = intrinsics.view(B * N, 3, 3)
        else:
            # [B, N, H, W]
            B, N, H, W = depths.shape
            depths = depths.view(B * N, H, W)
            poses = poses.view(B * N, 4, 4)
            intrinsics = intrinsics.view(B * N, 3, 3)

        device = depths.device
        BN = B * N

        # Create pixel grid [H, W, 2]
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Homogeneous pixel coordinates [H, W, 3]
        pixels = torch.stack([u_coords, v_coords, torch.ones_like(u_coords)], dim=-1)
        pixels = pixels.unsqueeze(0).expand(BN, -1, -1, -1)  # [BN, H, W, 3]

        # Unproject to camera coordinates
        # rays_cam = K^{-1} @ pixels
        K_inv = torch.inverse(intrinsics)  # [BN, 3, 3]
        pixels_flat = pixels.view(BN, H * W, 3)  # [BN, H*W, 3]
        rays_cam = torch.bmm(pixels_flat, K_inv.transpose(1, 2))  # [BN, H*W, 3]

        # Scale by depth to get 3D points in camera coordinates
        depths_flat = depths.view(BN, H * W, 1)  # [BN, H*W, 1]
        points_cam = rays_cam * depths_flat  # [BN, H*W, 3]

        # Transform to world coordinates: P_world = R @ P_cam + t
        R = poses[:, :3, :3]  # [BN, 3, 3]
        t = poses[:, :3, 3:4]  # [BN, 3, 1]

        points_world = torch.bmm(points_cam, R.transpose(1, 2)) + t.transpose(1, 2)  # [BN, H*W, 3]

        # Reshape to [B, N, H, W, 3]
        pointmaps = points_world.view(B, N, H, W, 3)

        norm_params = None
        if normalize:
            # Center and scale for numerical stability
            # This helps with attention computations
            centroid = pointmaps.mean(dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1, 3]
            pointmaps_centered = pointmaps - centroid
            scale = pointmaps_centered.abs().mean() + 1e-6
            pointmaps = pointmaps_centered / scale
            # Save params to convert back to meters: real = normalized * scale + centroid
            norm_params = {'centroid': centroid.squeeze(), 'scale': scale}

        return pointmaps, norm_params


class GeometryAwareSemanticAttention(nn.Module):
    """
    Geometry-Aware Semantic Attention (GASA) - Our core contribution.

    Standard cross-attention matches features based on semantic similarity alone,
    which leads to "hallucinations" (matching a mug handle to a door handle in
    the background). GASA adds a geometric bias that penalizes semantically
    similar but spatially distant matches.

    Formula:
        Attn(Q, K) = Softmax(QK^T / sqrt(d) + beta * phi(||P_Q - P_K||))

    where:
        - Q, K: Semantic features (from SAM3)
        - P_Q, P_K: 3D world coordinates (from DA3)
        - phi: Learnable kernel that penalizes distance (negative log or MLP)
        - beta: Learnable strength of geometric veto

    Reference:
        - GTA (ICLR 2024): Uses epipolar constraints (requires known poses)
        - Our contribution: Uses estimated metric geometry (pose-free)

    Ablation modes:
        - cross_view=True (default): Full cross-view attention across all N views
        - cross_view=False: Single-view attention only, world-PE provides cross-view signal
          (Tests MV-SAM's finding that single-view attn + 3D PE ≈ cross-view attn)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.1,
        use_learned_kernel: bool = True,
        kernel_type: str = 'learned',  # 'learned', 'rbf', or 'fixed'
        max_distance: float = 2.0,  # Max distance for attention (in normalized coords)
        cross_view: bool = True,  # If False, attention is per-view only (ablation)
        bidirectional: bool = False  # If True, boost nearby + suppress distant; else suppress-only
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Temperature for distance kernel (smaller = sharper)
            use_learned_kernel: If True, use MLP for distance kernel; else use fixed
            kernel_type: Distance kernel type - 'learned' (MLP), 'rbf' (Gaussian), 'fixed' (-d)
            max_distance: Distances beyond this get minimal attention
            cross_view: If True, attention across all views; if False, per-view only
            bidirectional: If True, boost nearby + suppress distant; else suppress-only (clamp max=0)
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.temperature = temperature
        self.max_distance = max_distance
        self.cross_view = cross_view
        self.kernel_type = kernel_type

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable strength of geometric bias
        # Initialized to 1.0, will be learned during training
        self.beta = nn.Parameter(torch.tensor(1.0))

        # Distance kernel
        if kernel_type == 'rbf':
            self.distance_kernel = None
            self.rbf_log_sigma = nn.Parameter(torch.tensor(0.0))  # σ = 1.0 meter
        elif use_learned_kernel and kernel_type != 'fixed':
            # Learned MLP to convert distance to attention bias
            self.distance_kernel = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            nn.init.xavier_uniform_(self.distance_kernel[0].weight, gain=0.1)
            nn.init.zeros_(self.distance_kernel[0].bias)
            nn.init.xavier_uniform_(self.distance_kernel[2].weight, gain=0.1)
            if bidirectional:
                nn.init.constant_(self.distance_kernel[2].bias, 2.0)  # Start positive (boost nearby)
            else:
                nn.init.constant_(self.distance_kernel[2].bias, -1.0)  # Start negative (suppress-only)
        else:
            self.distance_kernel = None

        self.dropout = nn.Dropout(dropout)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def compute_geometric_bias(
        self,
        pointmaps: torch.Tensor,
        spatial_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute geometric attention bias from 3D coordinates.

        Points close in 3D space get positive bias (attend more).
        Points far apart get negative bias (attend less).

        Args:
            pointmaps: [B, N, H, W, 3] world coordinates
            spatial_shape: (H', W') attention spatial resolution

        Returns:
            bias: [B, num_heads, N*H'*W', N*H'*W'] geometric attention bias
        """
        B, N, H, W, _ = pointmaps.shape
        H_attn, W_attn = spatial_shape

        # Downsample pointmaps to attention resolution if needed
        if H != H_attn or W != W_attn:
            # Reshape for interpolation: [B*N, 3, H, W]
            pts = pointmaps.permute(0, 1, 4, 2, 3).reshape(B * N, 3, H, W)
            pts = F.interpolate(pts, size=(H_attn, W_attn), mode='bilinear', align_corners=False)
            pts = pts.reshape(B, N, 3, H_attn, W_attn).permute(0, 1, 3, 4, 2)
        else:
            pts = pointmaps

        # Flatten to [B, N*H'*W', 3]
        pts_flat = pts.reshape(B, N * H_attn * W_attn, 3)

        # Compute pairwise distances [B, N*H'*W', N*H'*W']
        distances = torch.cdist(pts_flat, pts_flat)  # Euclidean distance

        # Apply distance kernel to get attention bias
        if self.kernel_type == 'rbf' and hasattr(self, 'rbf_log_sigma'):
            sigma = torch.exp(self.rbf_log_sigma)
            if self.bidirectional:
                max_boost = 5.0
                rbf = torch.exp(-(distances ** 2) / (2 * sigma ** 2 + 1e-6))
                bias = max_boost * rbf - 2.5  # Range: [-2.5, +2.5]
            else:
                bias = -torch.exp(-(distances ** 2) / (2 * sigma ** 2 + 1e-6))
        elif self.distance_kernel is not None:
            # Learned kernel
            dist_flat = distances.reshape(-1, 1)
            bias_flat = self.distance_kernel(dist_flat)
            bias = bias_flat.reshape(B, N * H_attn * W_attn, N * H_attn * W_attn)
        else:
            # Fixed kernel
            if self.bidirectional:
                bias = 5.0 - torch.log(1 + distances / self.temperature)
            else:
                bias = -distances

        # Clamp to prevent extreme values
        clamp_max = 10.0 if self.bidirectional else 0.0
        bias = torch.clamp(bias, min=-10, max=clamp_max)

        # Expand for multi-head: [B, num_heads, L, L]
        bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        return bias

    def forward(
        self,
        features: torch.Tensor,
        pointmaps: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        attn_size: int = 16  # Downsample to this resolution for memory efficiency
    ) -> torch.Tensor:
        """
        Apply geometry-aware semantic attention.

        Args:
            features: [B, N, H, W, D] semantic features from SAM3
            pointmaps: [B, N, H, W, 3] world coordinates from DA3
            pe: [B, N, H, W, D] optional positional embeddings to add
            attn_size: Resolution for attention computation (default 16 = 16x16)

        Returns:
            output: [B, N, H, W, D] attended features
        """
        B, N, H, W, D = features.shape

        # Memory optimization: downsample for attention, then upsample
        if H > attn_size or W > attn_size:
            # Downsample features for attention
            features_down = features.permute(0, 1, 4, 2, 3).reshape(B * N, D, H, W)
            features_down = F.adaptive_avg_pool2d(features_down, (attn_size, attn_size))
            features_down = features_down.reshape(B, N, D, attn_size, attn_size).permute(0, 1, 3, 4, 2)

            # Downsample pointmaps
            pts_down = pointmaps.permute(0, 1, 4, 2, 3).reshape(B * N, 3, H, W)
            pts_down = F.adaptive_avg_pool2d(pts_down, (attn_size, attn_size))
            pts_down = pts_down.reshape(B, N, 3, attn_size, attn_size).permute(0, 1, 3, 4, 2)

            # Downsample PE if provided
            if pe is not None:
                pe_down = pe.permute(0, 1, 4, 2, 3).reshape(B * N, D, H, W)
                pe_down = F.adaptive_avg_pool2d(pe_down, (attn_size, attn_size))
                pe_down = pe_down.reshape(B, N, D, attn_size, attn_size).permute(0, 1, 3, 4, 2)
            else:
                pe_down = None

            # Run attention at lower resolution
            out_down = self._forward_attention(features_down, pts_down, pe_down, attn_size, attn_size)

            # Upsample back to original resolution
            out_up = out_down.permute(0, 1, 4, 2, 3).reshape(B * N, D, attn_size, attn_size)
            out_up = F.interpolate(out_up, size=(H, W), mode='bilinear', align_corners=False)
            output = out_up.reshape(B, N, D, H, W).permute(0, 1, 3, 4, 2)

            # Residual from original features (preserve high-freq details)
            output = output + features * 0.1
            return output
        else:
            return self._forward_attention(features, pointmaps, pe, H, W)

    def _forward_attention(
        self,
        features: torch.Tensor,
        pointmaps: torch.Tensor,
        pe: Optional[torch.Tensor],
        H: int,
        W: int
    ) -> torch.Tensor:
        """Core attention computation at given resolution."""
        B, N, _, _, D = features.shape

        # Add positional embeddings if provided
        if pe is not None:
            features = features + pe

        if self.cross_view:
            # Cross-view attention: all N views attend to each other
            return self._forward_cross_view(features, pointmaps, H, W)
        else:
            # Single-view attention: each view attends only within itself
            # World-PE (already added above) provides cross-view consistency signal
            return self._forward_single_view(features, pointmaps, H, W)

    def _forward_cross_view(
        self,
        features: torch.Tensor,
        pointmaps: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """Cross-view attention: tokens from all N views attend to each other."""
        B, N, _, _, D = features.shape
        L = N * H * W  # Total sequence length across all views

        # Flatten all views together: [B, N*H*W, D]
        x = features.reshape(B, L, D)

        # Q, K, V projections
        Q = self.q_proj(x)  # [B, L, D]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention: [B, num_heads, L, head_dim]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute semantic attention scores: [B, num_heads, L, L]
        attn_semantic = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Compute geometric bias (skip if pointmaps is None - semantic-only fallback)
        if pointmaps is not None:
            # Use lower resolution for efficiency
            attn_H, attn_W = min(H, 16), min(W, 16)
            geo_bias = self.compute_geometric_bias(pointmaps, (attn_H, attn_W))

            # If attention resolution differs, interpolate bias
            if attn_H != H or attn_W != W:
                # Reshape bias for interpolation
                geo_bias = geo_bias.view(B * self.num_heads, N * attn_H * attn_W, N * attn_H * attn_W)
                # This is approximate - full interpolation would be expensive
                # Instead, we'll upsample in a simpler way
                geo_bias = F.interpolate(
                    geo_bias.unsqueeze(1),
                    size=(L, L),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                geo_bias = geo_bias.view(B, self.num_heads, L, L)

            # Combine semantic and geometric attention
            attn_scores = attn_semantic + self.beta * geo_bias
        else:
            # No pointmaps - semantic-only attention
            attn_scores = attn_semantic

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, V)  # [B, num_heads, L, head_dim]

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        out = self.out_proj(out)

        # Residual connection and layer norm
        out = self.norm(x + out)

        # Reshape back to spatial: [B, N, H, W, D]
        return out.reshape(B, N, H, W, D)

    def _forward_single_view(
        self,
        features: torch.Tensor,
        pointmaps: Optional[torch.Tensor],
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Single-view attention: each view attends only within itself.

        This ablation tests the MV-SAM finding that single-view attention + 3D PE
        can match full cross-view attention. World-PE is already added to features
        before this is called, providing the cross-view consistency signal.
        """
        B, N, _, _, D = features.shape
        L_per_view = H * W

        # Reshape to process all views in parallel: [B*N, H*W, D]
        x = features.reshape(B * N, L_per_view, D)
        pointmaps_flat = pointmaps.reshape(B * N, H, W, 3) if pointmaps is not None else None

        # Q, K, V projections
        Q = self.q_proj(x)  # [B*N, L, D]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention: [B*N, num_heads, L, head_dim]
        Q = Q.view(B * N, L_per_view, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * N, L_per_view, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * N, L_per_view, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute semantic attention scores: [B*N, num_heads, L, L]
        attn_semantic = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Compute geometric bias per view (skip if no pointmaps - semantic-only fallback)
        if pointmaps_flat is not None:
            attn_H, attn_W = min(H, 16), min(W, 16)

            # Compute per-view geometric bias (no cross-view distances)
            if H != attn_H or W != attn_W:
                pts = F.interpolate(
                    pointmaps_flat.permute(0, 3, 1, 2),  # [B*N, 3, H, W]
                    size=(attn_H, attn_W),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)  # [B*N, attn_H, attn_W, 3]
            else:
                pts = pointmaps_flat

            # Flatten to [B*N, H'*W', 3]
            pts_flat = pts.reshape(B * N, attn_H * attn_W, 3)

            # Compute pairwise distances within each view [B*N, L', L']
            distances = torch.cdist(pts_flat, pts_flat)

            # Apply distance kernel
            if self.kernel_type == 'rbf' and hasattr(self, 'rbf_log_sigma'):
                sigma = torch.exp(self.rbf_log_sigma)
                if self.bidirectional:
                    max_boost = 5.0
                    rbf = torch.exp(-(distances ** 2) / (2 * sigma ** 2 + 1e-6))
                    geo_bias = max_boost * rbf - 2.5
                else:
                    geo_bias = -torch.exp(-(distances ** 2) / (2 * sigma ** 2 + 1e-6))
            elif self.distance_kernel is not None:
                dist_flat = distances.reshape(-1, 1)
                bias_flat = self.distance_kernel(dist_flat)
                geo_bias = bias_flat.reshape(B * N, attn_H * attn_W, attn_H * attn_W)
            else:
                if self.bidirectional:
                    geo_bias = 5.0 - torch.log(1 + distances / self.temperature)
                else:
                    geo_bias = -distances

            clamp_max = 10.0 if self.bidirectional else 0.0
            geo_bias = torch.clamp(geo_bias, min=-10, max=clamp_max)

            # Expand for multi-head: [B*N, num_heads, L', L']
            geo_bias = geo_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # Interpolate to match attention resolution if needed
            if attn_H != H or attn_W != W:
                geo_bias = geo_bias.reshape(B * N * self.num_heads, attn_H * attn_W, attn_H * attn_W)
                geo_bias = F.interpolate(
                    geo_bias.unsqueeze(1),
                    size=(L_per_view, L_per_view),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                geo_bias = geo_bias.view(B * N, self.num_heads, L_per_view, L_per_view)

            # Combine semantic and geometric attention
            attn_scores = attn_semantic + self.beta * geo_bias
        else:
            # No pointmaps - semantic-only attention
            attn_scores = attn_semantic

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, V)  # [B*N, num_heads, L, head_dim]

        # Reshape and project
        out = out.transpose(1, 2).reshape(B * N, L_per_view, D)  # [B*N, L, D]
        out = self.out_proj(out)

        # Residual connection and layer norm
        out = self.norm(x + out)

        # Reshape back to [B, N, H, W, D]
        return out.reshape(B, N, H, W, D)


class GASABlock(nn.Module):
    """
    Full GASA block with attention + FFN.

    Follows standard transformer block structure:
    1. GASA attention (geometry-aware)
    2. FFN (feed-forward network)
    3. Residual connections + layer norm
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        temperature: float = 0.1,
        cross_view: bool = True  # If False, per-view attention only (ablation)
    ):
        super().__init__()

        # GASA attention
        self.gasa = GeometryAwareSemanticAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            temperature=temperature,
            cross_view=cross_view
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        features: torch.Tensor,
        pointmaps: torch.Tensor,
        pe: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, H, W, D]
            pointmaps: [B, N, H, W, 3]
            pe: [B, N, H, W, D] optional positional embeddings

        Returns:
            output: [B, N, H, W, D]
        """
        # GASA attention (includes residual)
        x = self.gasa(features, pointmaps, pe)

        # FFN with residual
        B, N, H, W, D = x.shape
        x_flat = x.reshape(B * N * H * W, D)
        x_flat = self.norm(x_flat + self.ffn(x_flat))

        return x_flat.reshape(B, N, H, W, D)


class GASAEncoder(nn.Module):
    """
    Multi-layer GASA encoder for cross-view feature fusion.

    Takes SAM3 features + DA3 geometry and produces 3D-consistent features
    by allowing cross-view attention guided by geometry.

    Ablation modes:
        - cross_view=True (default): Full cross-view attention
        - cross_view=False: Single-view attention + world-PE only
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        temperature: float = 0.1,
        use_world_pe: bool = True,
        cross_view: bool = True  # If False, per-view attention only (ablation)
    ):
        """
        Args:
            d_model: Feature dimension
            num_layers: Number of GASA blocks
            num_heads: Attention heads per block
            ffn_dim: FFN hidden dimension
            dropout: Dropout rate
            temperature: Distance kernel temperature
            use_world_pe: If True, add world-space positional embeddings
            cross_view: If True, cross-view attention; if False, per-view only
        """
        super().__init__()
        self.use_world_pe = use_world_pe
        self.cross_view = cross_view

        # World-space positional encoding
        if use_world_pe:
            self.world_pe = WorldSpacePositionalEncoding(d_model=d_model)

        # GASA layers
        self.layers = nn.ModuleList([
            GASABlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                temperature=temperature,
                cross_view=cross_view
            )
            for _ in range(num_layers)
        ])

        # Final projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        features: torch.Tensor,
        pointmaps: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode features with geometry-aware cross-view attention.

        Args:
            features: [B, N, H, W, D] SAM3 features
            pointmaps: [B, N, H, W, 3] world coordinates (can be None for semantic-only)

        Returns:
            output: [B, N, H, W, D] geometry-aware features
        """
        # Compute world-space positional embeddings
        pe = None
        if self.use_world_pe and pointmaps is not None:
            pe = self.world_pe(pointmaps)  # [B, N, H, W, D]

        # Apply GASA layers
        x = features
        for layer in self.layers:
            x = layer(x, pointmaps, pe)

        # Final projection
        B, N, H, W, D = x.shape
        x = x.reshape(B * N * H * W, D)
        x = self.out_proj(x)

        return x.reshape(B, N, H, W, D)


class SymmetricCentroidHead(nn.Module):
    """
    Symmetric output head for 3D centroid prediction.

    Ensures permutation invariance despite DA3 not being permutation equivariant.
    Predicts per-view confidence weights and computes weighted average of
    per-view centroids.

    Formula:
        C_final = sum(w_i * C_i) / sum(w_i)

    where w_i is the learned confidence for view i.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()

        # Per-view feature aggregation
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Confidence prediction (per view)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Optional: predict offset from mask centroid
        self.offset_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)
        )

    def forward(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        pointmaps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict 3D centroid from multi-view features.

        Args:
            features: [B, N, H, W, D] or [B, N, D, H, W]
            masks: [B, N, 1, H, W] predicted masks
            pointmaps: [B, N, H, W, 3] world coordinates

        Returns:
            centroid: [B, 3] predicted 3D centroid
            confidences: [B, N] per-view confidences
        """
        # Ensure features are [B, N, H, W, D]
        # Check if features are in channel-first format [B, N, D, H, W]
        # by comparing shape[2] (should be H if spatial) vs shape[4] (should be D)
        if features.dim() == 5 and features.shape[2] > features.shape[3]:
            # [B, N, D, H, W] -> [B, N, H, W, D]
            features = features.permute(0, 1, 3, 4, 2)

        B, N, H, W, D = features.shape

        # Compute per-view mask-weighted centroids
        masks_squeezed = masks.squeeze(2)  # [B, N, H, W]

        # Normalize masks for weighted averaging
        mask_sums = masks_squeezed.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        mask_weights = masks_squeezed / mask_sums  # [B, N, H, W]

        # Compute per-view 3D centroids: weighted average of pointmap
        # [B, N, H, W, 3] * [B, N, H, W, 1] -> sum over H, W
        centroids_per_view = (pointmaps * mask_weights.unsqueeze(-1)).sum(dim=[2, 3])  # [B, N, 3]

        # Pool features per view for confidence prediction
        # [B, N, H, W, D] -> [B, N, D]
        features_pooled = (features * mask_weights.unsqueeze(-1)).sum(dim=[2, 3])

        # Predict confidence per view
        confidences = self.confidence_head(features_pooled).squeeze(-1)  # [B, N]

        # Predict offset per view
        offsets = self.offset_head(features_pooled)  # [B, N, 3]

        # Apply offsets to centroids
        centroids_per_view = centroids_per_view + offsets

        # Weighted average (symmetric operation)
        conf_normalized = F.softmax(confidences, dim=1)  # [B, N]
        centroid = (centroids_per_view * conf_normalized.unsqueeze(-1)).sum(dim=1)  # [B, 3]

        return centroid, confidences
