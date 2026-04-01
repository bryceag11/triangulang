"""Positional encoding modules for the GASA architecture.

Classes extracted from gasa.py:
- PluckerEmbedding: Plücker ray embeddings for camera pose conditioning
- RayRoPE3D: 3D Rotary Positional Encoding
- WorldSpacePositionalEncoding: Sinusoidal PE for 3D world coordinates
- CameraRelativePositionalEncoding: SE(3)-invariant camera-frame PE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
