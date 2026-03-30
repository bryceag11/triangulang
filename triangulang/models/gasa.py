"""Geometry-Aware Semantic Attention (GASA) module.

Attention mechanism that biases cross-view feature matching using 3D geometric
distance, preventing semantically similar but spatially distant matches.
Includes PointmapComputer, WorldSpacePositionalEncoding, and the GASA layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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


class PluckerEmbedding(nn.Module):
    """
    Plücker ray embeddings for camera pose conditioning.

    From MV-Foundation (arXiv 2512.15708): For each pixel, construct a raymap
    with ray origin (o) and direction (d), computed relative to camera 0.

    Formula:
        p(u,v) = [o(u,v), d(u,v)]  # 6D Plücker coordinates

    This provides explicit camera pose encoding without requiring world coordinates.
    Alternative to world-space PE for ablation studies.

    Reference: MV-Foundation (Segre et al., 2025)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_frequencies: int = 6,
        include_input: bool = True
    ):
        """
        Args:
            d_model: Output embedding dimension
            num_frequencies: Number of frequency bands for sinusoidal encoding
            include_input: If True, concatenate raw Plücker coords to encoding
        """
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Plücker coords are 6D (origin xyz + direction xyz)
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freq_bands', freq_bands)

        # Input dimension: 6 (plucker) * 2 (sin/cos) * num_frequencies + optional 6 (raw)
        input_dim = 6 * 2 * num_frequencies
        if include_input:
            input_dim += 6

        # Project to model dimension
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Initialize with small weights
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_plucker_coords(
        self,
        depths: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Plücker coordinates for each pixel.

        Args:
            depths: [B, N, H, W] depth maps
            poses: [B, N, 4, 4] camera-to-world transforms
            intrinsics: [B, N, 3, 3] camera intrinsics

        Returns:
            plucker: [B, N, H, W, 6] Plücker coordinates (origin + direction)
        """
        B, N, H, W = depths.shape
        device = depths.device

        # Create pixel grid [H, W, 2]
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Homogeneous pixel coordinates [H, W, 3]
        pixels = torch.stack([u_coords, v_coords, torch.ones_like(u_coords)], dim=-1)
        pixels = pixels.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1, -1)  # [B, N, H, W, 3]

        # Get reference camera (camera 0) pose
        ref_pose = poses[:, 0:1, :, :]  # [B, 1, 4, 4]

        # For each view, compute rays relative to camera 0
        plucker_list = []
        for i in range(N):
            # Camera intrinsics for this view
            K_inv = torch.inverse(intrinsics[:, i])  # [B, 3, 3]

            # Ray directions in camera space
            pixels_flat = pixels[:, i].reshape(B, H * W, 3)  # [B, H*W, 3]
            rays_cam = torch.bmm(pixels_flat, K_inv.transpose(1, 2))  # [B, H*W, 3]
            rays_cam = F.normalize(rays_cam, dim=-1)

            # Camera pose for this view
            R = poses[:, i, :3, :3]  # [B, 3, 3]
            t = poses[:, i, :3, 3]    # [B, 3]

            # Transform rays to world space
            rays_world = torch.bmm(rays_cam, R.transpose(1, 2))  # [B, H*W, 3]

            # Camera origin in world space
            origin = t.unsqueeze(1).expand(-1, H * W, -1)  # [B, H*W, 3]

            # Transform to reference frame (camera 0)
            ref_R = ref_pose[:, 0, :3, :3]  # [B, 3, 3]
            ref_t = ref_pose[:, 0, :3, 3]    # [B, 3]
            ref_R_inv = ref_R.transpose(1, 2)

            # Origin relative to camera 0
            origin_rel = torch.bmm(origin - ref_t.unsqueeze(1), ref_R_inv.transpose(1, 2))
            # Direction relative to camera 0
            rays_rel = torch.bmm(rays_world, ref_R_inv.transpose(1, 2))

            # Plücker coords: [origin, direction]
            plucker = torch.cat([origin_rel, rays_rel], dim=-1)  # [B, H*W, 6]
            plucker_list.append(plucker.reshape(B, H, W, 6))

        return torch.stack(plucker_list, dim=1)  # [B, N, H, W, 6]

    def forward(
        self,
        depths: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Plücker embeddings.

        Args:
            depths: [B, N, H, W] depth maps
            poses: [B, N, 4, 4] camera-to-world transforms
            intrinsics: [B, N, 3, 3] camera intrinsics

        Returns:
            pe: [B, N, H, W, D] Plücker positional embeddings
        """
        # Compute Plücker coordinates
        plucker = self.compute_plucker_coords(depths, poses, intrinsics)  # [B, N, H, W, 6]

        original_shape = plucker.shape[:-1]  # [B, N, H, W]
        plucker_flat = plucker.reshape(-1, 6)  # [*, 6]

        # Apply frequency encoding
        plucker_scaled = plucker_flat.unsqueeze(1) * self.freq_bands.view(1, -1, 1) * math.pi
        sin_enc = torch.sin(plucker_scaled)
        cos_enc = torch.cos(plucker_scaled)

        # Interleave sin/cos and flatten
        encoding = torch.stack([sin_enc, cos_enc], dim=-1)  # [*, num_freq, 6, 2]
        encoding = encoding.reshape(plucker_flat.shape[0], -1)  # [*, num_freq * 6 * 2]

        if self.include_input:
            encoding = torch.cat([plucker_flat, encoding], dim=-1)

        # Project to model dimension
        pe = self.proj(encoding)

        return pe.reshape(*original_shape, self.d_model)


class RayRoPE3D(nn.Module):
    """
    3D Rotary Positional Encoding inspired by RayRoPE (Wu et al., arXiv 2601.15275).

    Key ideas adapted for our GASA decoder:
    1. SE(3) invariance: project all positions into query-camera frame (relative coords)
    2. Multiplicative RoPE: rotate Q and K (not additive, not in V)
    3. Depth uncertainty: analytically compute E[RoPE] under depth confidence
    4. Works with head_dim=32: 4 coords (3D direction + depth) × 4 frequencies × 2 (sin/cos)

    Unlike additive world PE:
    - Position info doesn't leak into Values (cleaner separation of what vs where)
    - Per-frequency distance sensitivity (high freq = precise, low freq = coarse)
    - Attention depends on RELATIVE geometry (not absolute world coords)

    Unlike original RayRoPE:
    - Uses known depth from DA3 (not predicted from tokens) — we already have good metric depth
    - Optional sigma from DA3 depth_conf (not learned from scratch)
    - Simpler: 4 coordinates instead of 12 (works with our head_dim=32)

    Usage:
        rayrope = RayRoPE3D(head_dim=32, num_freqs=4)
        # In attention:
        Q_rot, K_rot = rayrope(Q_proj, K_proj, memory_pos_3d, query_pos_3d, poses, intrinsics)
        attn = Q_rot @ K_rot.T / sqrt(d)
    """

    def __init__(
        self,
        head_dim: int = 32,
        num_freqs: int = 4,
        coord_dim: int = 4,  # 3D direction + depth
        max_period: float = 4.0,
        freq_base: float = 3.0,
        use_uncertainty: bool = True,
    ):
        """
        Args:
            head_dim: Dimension per attention head (must equal coord_dim * num_freqs * 2)
            num_freqs: Number of RoPE frequency bands per coordinate
            coord_dim: Number of position coordinates (4 = direction_xyz + depth)
            max_period: Maximum period for lowest frequency
            freq_base: Geometric ratio between adjacent frequencies
            use_uncertainty: If True, compute expected RoPE under depth uncertainty
        """
        super().__init__()
        self.head_dim = head_dim
        self.num_freqs = num_freqs
        self.coord_dim = coord_dim
        self.use_uncertainty = use_uncertainty

        assert head_dim == coord_dim * num_freqs * 2, (
            f"head_dim={head_dim} must equal coord_dim * num_freqs * 2 = "
            f"{coord_dim} * {num_freqs} * 2 = {coord_dim * num_freqs * 2}"
        )

        # Compute log-spaced frequencies (same as RayRoPE)
        # freq[i] = 2*pi / period[i], period geometrically spaced
        min_period = max_period / (freq_base ** (num_freqs - 1))
        log_freqs = torch.linspace(
            math.log(2 * math.pi / max_period),
            math.log(2 * math.pi / min_period),
            num_freqs,
        )
        freqs = torch.exp(log_freqs)  # [num_freqs]
        self.register_buffer('freqs', freqs)

    def compute_projective_coords(
        self,
        positions_3d: torch.Tensor,
        w2c: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project 3D world positions into a camera's projective frame.

        Following RayRoPE reference: projects through normalized K to get
        intrinsics-aware direction + disparity. This makes the encoding sensitive
        to focal length (how pixels map to rays), not just 3D direction.

        When intrinsics are None, falls back to camera-frame direction + log-depth.

        Args:
            positions_3d: [B, L, 3] world-space 3D positions
            w2c: [B, 4, 4] world-to-camera transform
            intrinsics: [B, 3, 3] camera intrinsics (optional but recommended)

        Returns:
            coords: [B, L, 4] projective coordinates (dir_x, dir_y, dir_z, disparity)
        """
        B, L, _ = positions_3d.shape

        # Transform to camera frame: P_cam = R @ P_world + t
        R = w2c[:, :3, :3]  # [B, 3, 3]
        t = w2c[:, :3, 3:]  # [B, 3, 1]
        P_cam = torch.bmm(positions_3d, R.transpose(1, 2)) + t.transpose(1, 2)  # [B, L, 3]

        # Depth in camera frame (z-axis)
        z = P_cam[..., 2:3].clamp(min=0.01)  # [B, L, 1]

        if intrinsics is not None:
            # Project through normalized intrinsics (matches RayRoPE reference normalize_K).
            # K_norm normalizes to ~[-0.5, 0.5] image-plane coords:
            #   fx_n = fx/W, fy_n = fy/H, cx_n = cx/W - 0.5, cy_n = cy/H - 0.5
            # We approximate W, H from principal point (cx ≈ W/2, cy ≈ H/2).
            fx = intrinsics[:, 0, 0]  # [B]
            fy = intrinsics[:, 1, 1]  # [B]
            cx = intrinsics[:, 0, 2]  # [B]
            cy = intrinsics[:, 1, 2]  # [B]
            # Approximate image size from principal point
            W = (2.0 * cx).clamp(min=1.0)  # [B]
            H = (2.0 * cy).clamp(min=1.0)  # [B]

            # Build normalized K: [B, 3, 3]
            K_norm = torch.zeros_like(intrinsics)
            K_norm[:, 0, 0] = fx / W
            K_norm[:, 1, 1] = fy / H
            K_norm[:, 0, 2] = cx / W - 0.5
            K_norm[:, 1, 2] = cy / H - 0.5
            K_norm[:, 2, 2] = 1.0

            # Project camera-frame points through K_norm: [B, L, 3]
            P_img = torch.bmm(P_cam, K_norm.transpose(1, 2))  # [B, L, 3]
            direction = F.normalize(P_img, dim=-1)  # [B, L, 3]
        else:
            # Fallback: raw camera-frame direction (no intrinsics)
            direction = F.normalize(P_cam, dim=-1)  # [B, L, 3]

        # Disparity encoding (1/z), matching RayRoPE reference 'inv_d' type
        disparity = (1.0 / z).clamp(max=10.0)  # [B, L, 1], clamp for stability

        coords = torch.cat([direction, disparity], dim=-1)  # [B, L, 4]
        return coords

    def compute_rope_angles(
        self,
        coords: torch.Tensor,
        depth_conf: Optional[torch.Tensor] = None,
    ):
        """Convert position coordinates to RoPE rotation angles.

        Args:
            coords: [B, L, 4] projective coordinates
            depth_conf: [B, L] optional depth confidence (higher = more certain)

        Returns:
            cos_angles: [B, L, head_dim//2] cosine components
            sin_angles: [B, L, head_dim//2] sine components
        """
        B, L, C = coords.shape  # C = coord_dim = 4

        # Compute angles: freq[f] * coord[c] for all f,c pairs
        # coords: [B, L, C], freqs: [F]
        # Result: [B, L, F, C] -> reshape to [B, L, F*C]
        angles = torch.einsum('blc,f->blfc', coords, self.freqs)  # [B, L, F, C]
        angles = angles.reshape(B, L, -1)  # [B, L, F*C = num_freqs * coord_dim]

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # Depth uncertainty: compute E[cos(ω·x)] and E[sin(ω·x)] analytically
        # When depth is uncertain, high-frequency components blur (graceful degradation)
        if self.use_uncertainty and depth_conf is not None:
            # depth_conf is DA3's confidence: higher = more certain, range ~[1, 5+]
            # Convert to sigma (uncertainty): sigma = 1 / conf
            sigma = 1.0 / (depth_conf.clamp(min=1.0) + 1e-6)  # [B, L]

            # Only the depth coordinate (index 3) has uncertainty
            # depth_freq_indices: positions in the angle vector that correspond to depth
            depth_freq_angles_low = coords[..., 3:4] - sigma.unsqueeze(-1)  # [B, L, 1]
            depth_freq_angles_high = coords[..., 3:4] + sigma.unsqueeze(-1)  # [B, L, 1]

            # For each frequency applied to depth: E[cos(ω·d)] over [d-σ, d+σ]
            for f_idx in range(self.num_freqs):
                angle_idx = f_idx * self.coord_dim + 3  # depth is coordinate index 3
                freq = self.freqs[f_idx]

                omega_low = freq * depth_freq_angles_low.squeeze(-1)  # [B, L]
                omega_high = freq * depth_freq_angles_high.squeeze(-1)  # [B, L]
                delta = omega_high - omega_low  # [B, L]

                # Avoid division by zero when sigma is very small
                is_certain = (delta.abs() < 1e-4)

                safe_delta = torch.where(is_certain, torch.ones_like(delta), delta)
                E_cos = (torch.sin(omega_high) - torch.sin(omega_low)) / safe_delta
                E_sin = (torch.cos(omega_low) - torch.cos(omega_high)) / safe_delta

                # Use expected values where uncertain, point values where certain
                cos_angles[:, :, angle_idx] = torch.where(
                    is_certain, cos_angles[:, :, angle_idx], E_cos
                )
                sin_angles[:, :, angle_idx] = torch.where(
                    is_certain, sin_angles[:, :, angle_idx], E_sin
                )

        return cos_angles, sin_angles

    def apply_rope(
        self,
        x: torch.Tensor,
        cos_angles: torch.Tensor,
        sin_angles: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position encoding to input tensor.

        Splits each head into pairs and applies 2D rotation:
            [x1', x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]

        Args:
            x: [B, H, L, D] input (Q or K), H=num_heads, D=head_dim
            cos_angles: [B, L, D//2] cosine rotation angles
            sin_angles: [B, L, D//2] sine rotation angles

        Returns:
            rotated: [B, H, L, D] rotated output
        """
        B, H, L, D = x.shape
        half_d = D // 2

        # cos/sin: [B, L, D//2] -> [B, 1, L, D//2] for broadcasting over heads
        cos_a = cos_angles.unsqueeze(1)  # [B, 1, L, half_d]
        sin_a = sin_angles.unsqueeze(1)  # [B, 1, L, half_d]

        # Split into even/odd pairs
        x1 = x[..., :half_d]  # [B, H, L, half_d]
        x2 = x[..., half_d:]  # [B, H, L, half_d]

        # Apply 2D rotation per pair
        out1 = x1 * cos_a - x2 * sin_a
        out2 = x1 * sin_a + x2 * cos_a

        return torch.cat([out1, out2], dim=-1)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        memory_pos: torch.Tensor,
        query_pos: torch.Tensor,
        w2c: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        depth_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RayRoPE to Q and K tensors.

        When w2c is provided: full SE(3)-invariant projective RoPE.
        When w2c is None: falls back to normalized world-space RoPE (still useful).

        Args:
            Q: [B, H, Q_len, D] query tensor (after projection, before attention)
            K: [B, H, L, D] key tensor
            memory_pos: [B, L, 3] 3D positions of memory/key tokens
            query_pos: [B, Q_len, 3] 3D positions of query tokens
            w2c: [B, 4, 4] world-to-camera (optional, enables SE(3) invariance)
            intrinsics: [B, 3, 3] camera intrinsics (optional)
            depth_conf: [B, L] depth confidence for memory tokens (optional)

        Returns:
            Q_rot: [B, H, Q_len, D] rotated queries
            K_rot: [B, H, L, D] rotated keys
        """
        if w2c is not None:
            # Full projective RoPE: transform to camera frame
            key_coords = self.compute_projective_coords(memory_pos, w2c, intrinsics)
            query_coords = self.compute_projective_coords(query_pos, w2c, intrinsics)
        else:
            # Fallback: use normalized world coords + dummy depth
            # Center and scale for numerical stability
            all_pos = torch.cat([memory_pos, query_pos], dim=1)
            centroid = all_pos.mean(dim=1, keepdim=True)
            scale = (all_pos - centroid).abs().mean().clamp(min=1e-6)
            mem_norm = (memory_pos - centroid) / scale
            q_norm = (query_pos - centroid) / scale

            # Use L2 norm as proxy for depth
            mem_depth = torch.norm(mem_norm, dim=-1, keepdim=True)
            q_depth = torch.norm(q_norm, dim=-1, keepdim=True)

            # Use disparity (1/depth) to match projective path encoding
            mem_disp = (1.0 / mem_depth.clamp(min=0.01)).clamp(max=10.0)
            q_disp = (1.0 / q_depth.clamp(min=0.01)).clamp(max=10.0)
            key_coords = torch.cat([F.normalize(mem_norm, dim=-1), mem_disp], dim=-1)
            query_coords = torch.cat([F.normalize(q_norm, dim=-1), q_disp], dim=-1)

        # Compute RoPE angles
        key_cos, key_sin = self.compute_rope_angles(key_coords, depth_conf)
        query_cos, query_sin = self.compute_rope_angles(query_coords, depth_conf=None)

        # Apply rotation to Q and K (NOT V — keeps position out of values)
        Q_rot = self.apply_rope(Q, query_cos, query_sin)
        K_rot = self.apply_rope(K, key_cos, key_sin)

        return Q_rot, K_rot

    def forward_multiview(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        memory_pos: torch.Tensor,
        query_pos: torch.Tensor,
        w2c_per_view: torch.Tensor,
        intrinsics_per_view: torch.Tensor,
        num_cameras: int,
        scale: float,
        attn_bias: Optional[torch.Tensor] = None,
        depth_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply per-view RayRoPE for cross-view attention (correct multi-camera handling).

        In cross-view mode, memory tokens come from N different cameras. For RoPE to be
        geometrically consistent, we must project all positions into a common frame before
        computing rotation angles. We do this N times (once per camera) and average the
        attention outputs, providing SE(3)-invariant multi-view attention.

        For object queries (not tied to specific cameras): all queries are projected into
        each camera's frame along with all keys. The per-camera attention outputs are
        averaged to produce a frame-invariant result.

        Args:
            Q: [B, H, Q_len, D] query tensor (object queries, shared across views)
            K: [B, H, L, D] key tensor (L = N * tokens_per_cam, concatenated from N views)
            V: [B, H, L, D] value tensor
            memory_pos: [B, L, 3] 3D world positions of ALL memory tokens
            query_pos: [B, Q_len, 3] 3D world positions of query tokens
            w2c_per_view: [B, N, 4, 4] world-to-camera for each view
            intrinsics_per_view: [B, N, 3, 3] intrinsics for each view
            num_cameras: int, number of cameras N
            scale: float, attention scale factor (1/sqrt(d))
            attn_bias: [B, 1, Q_len, L] optional GASA geometric bias (frame-invariant)
            depth_conf: [B, L] optional depth confidence

        Returns:
            out: [B, H, Q_len, D] attention output (averaged across camera frames)
            attn_probs: [B, H, Q_len, L] attention probabilities (averaged)
        """
        B, H, Q_len, D = Q.shape
        L = K.shape[2]

        out_sum = torch.zeros(B, H, Q_len, D, device=Q.device, dtype=Q.dtype)
        attn_sum = torch.zeros(B, H, Q_len, L, device=Q.device, dtype=Q.dtype)

        for c in range(num_cameras):
            w2c_c = w2c_per_view[:, c]  # [B, 4, 4]
            intr_c = intrinsics_per_view[:, c]  # [B, 3, 3]

            # Project ALL positions into camera c's frame
            key_coords_c = self.compute_projective_coords(memory_pos, w2c_c, intr_c)
            query_coords_c = self.compute_projective_coords(query_pos, w2c_c, intr_c)

            # Compute per-camera RoPE angles
            key_cos_c, key_sin_c = self.compute_rope_angles(key_coords_c, depth_conf)
            query_cos_c, query_sin_c = self.compute_rope_angles(query_coords_c, depth_conf=None)

            # Rotate Q and K into camera c's frame
            Q_rot_c = self.apply_rope(Q, query_cos_c, query_sin_c)
            K_rot_c = self.apply_rope(K, key_cos_c, key_sin_c)

            # Compute attention in camera c's frame
            attn_c = torch.matmul(Q_rot_c, K_rot_c.transpose(-2, -1)) * scale

            # Add GASA geometric bias if available (frame-invariant, same for all cameras)
            if attn_bias is not None:
                attn_c = attn_c + attn_bias

            attn_probs_c = F.softmax(attn_c, dim=-1)
            out_c = torch.matmul(attn_probs_c, V)

            out_sum = out_sum + out_c
            attn_sum = attn_sum + attn_probs_c

        # Average across camera frames for SE(3)-invariant output
        out = out_sum / num_cameras
        attn_probs = attn_sum / num_cameras

        return out, attn_probs


class WorldSpacePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for 3D world coordinates.

    Key insight from MV-SAM: If two pixels map to the same 3D point (across views),
    they should have IDENTICAL positional embeddings. This provides strong inductive
    bias for cross-view consistency.

    Unlike 2D PE which gives different embeddings to the same 3D point in different
    views, world-space PE is view-invariant by construction.

    Formula:
        PE(x) = [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^{L-1} * pi * x), cos(2^{L-1} * pi * x)]

    Reference: NeRF (Mildenhall et al., 2020), MV-SAM (2026)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_frequencies: int = 10,
        max_frequency: float = 512.0,
        include_input: bool = True
    ):
        """
        Args:
            d_model: Output embedding dimension
            num_frequencies: Number of frequency bands (L in the formula)
            max_frequency: Maximum frequency for the encoding
            include_input: If True, concatenate raw XYZ to encoding
        """
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Compute frequency bands: 2^0, 2^1, ..., 2^{L-1} scaled to max_frequency
        freq_bands = 2.0 ** torch.linspace(0, math.log2(max_frequency), num_frequencies)
        self.register_buffer('freq_bands', freq_bands)

        # Input dimension: 3 (xyz) * 2 (sin/cos) * num_frequencies + optional 3 (raw xyz)
        input_dim = 3 * 2 * num_frequencies
        if include_input:
            input_dim += 3

        # Project to model dimension
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Initialize with small weights for stable training
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D coordinates to positional embeddings.

        Args:
            xyz: [B, N, H, W, 3] or [B, L, 3] world coordinates

        Returns:
            pe: [B, N, H, W, D] or [B, L, D] positional embeddings
        """
        original_shape = xyz.shape[:-1]  # Everything except last dim (3)
        xyz_flat = xyz.reshape(-1, 3)  # [*, 3]

        # Apply frequency encoding
        # xyz_scaled: [*, num_freq, 3]
        xyz_scaled = xyz_flat.unsqueeze(1) * self.freq_bands.view(1, -1, 1) * math.pi

        # Sin and cos: [*, num_freq, 3] each
        sin_enc = torch.sin(xyz_scaled)
        cos_enc = torch.cos(xyz_scaled)

        # Interleave sin/cos and flatten: [*, num_freq * 3 * 2]
        encoding = torch.stack([sin_enc, cos_enc], dim=-1)  # [*, num_freq, 3, 2]
        encoding = encoding.reshape(xyz_flat.shape[0], -1)  # [*, num_freq * 3 * 2]

        # Optionally include raw input
        if self.include_input:
            encoding = torch.cat([xyz_flat, encoding], dim=-1)

        # Project to model dimension
        pe = self.proj(encoding)

        # Reshape to original spatial dimensions
        return pe.reshape(*original_shape, self.d_model)


class CameraRelativePositionalEncoding(WorldSpacePositionalEncoding):
    """World PE but always in camera frame (SE(3)-invariant).

    When poses available: projects world-frame pointmaps to camera frame via w2c.
    When poses unavailable: pointmaps are already camera-frame (identity pose).

    This decouples the PE frame from the GASA bias frame:
    - GASA bias can use world-frame pointmaps (for cross-view consistency)
    - PE always uses camera-frame pointmaps (for SE(3) invariance)
    """

    def forward(self, pointmaps, w2c=None):
        if w2c is not None:
            # Project to camera frame: P_cam = R @ P_world + t
            R = w2c[:, :3, :3]  # [B, 3, 3]
            t = w2c[:, :3, 3:]  # [B, 3, 1]
            # pointmaps: [B, L, 3] or [B, H, W, 3] or [B, N, H, W, 3]
            shape = pointmaps.shape
            pts = pointmaps.reshape(shape[0], -1, 3)
            pts_cam = torch.bmm(pts, R.transpose(1, 2)) + t.transpose(1, 2)
            pointmaps = pts_cam.reshape(shape)
        return super().forward(pointmaps)


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
