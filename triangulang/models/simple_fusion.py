"""
Simplified DA3-SAM3 fusion head that can actually learn
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from triangulang import BPE_PATH as _BPE_PATH

# Import GASA components for geometry-aware attention
from triangulang.models.gasa import (
    GASAEncoder,
    PointmapComputer,
    WorldSpacePositionalEncoding
)


class SimpleFusionHead(nn.Module):
    """
    Simple fusion head that actually trains
    - No complex attention mechanisms
    - Direct feature concatenation
    - Strong initialization
    - Single output path
    """

    def __init__(self, sam_channels=256, depth_channels=1, hidden_dim=256, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Depth feature extraction with more capacity
        self.depth_conv = nn.Sequential(
            nn.Conv2d(depth_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Fusion layers with more depth
        total_channels = sam_channels + hidden_dim
        fusion_layers = [
            nn.Conv2d(total_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ]

        # Add intermediate layers
        for _ in range(num_layers - 1):
            fusion_layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])

        # Final output layer
        fusion_layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=1))
        self.fusion = nn.Sequential(*fusion_layers)

        # Strong initialization to escape 0.5 trap
        self._init_weights()

    def _init_weights(self):
        """Initialize with stronger values to escape sigmoid(0)=0.5"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Larger initialization for better gradient flow
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Initialize final layer bias negative to start with low predictions
                    if m.out_channels == 1:
                        nn.init.constant_(m.bias, -2.0)
                    else:
                        nn.init.constant_(m.bias, 0.01)

    def forward(self, sam_features, depth_map):
        """
        Args:
            sam_features: [B, C, H, W] features from SAM3
            depth_map: [B, 1, H', W'] depth from DA3

        Returns:
            dict with 'masks' key containing [B, 1, H, W] mask logits
        """
        B, C, H, W = sam_features.shape

        # Resize depth to match SAM features
        if depth_map.shape[2:] != (H, W):
            depth_map = F.interpolate(depth_map, size=(H, W), mode='bilinear', align_corners=False)

        # Extract depth features
        depth_features = self.depth_conv(depth_map)

        # Concatenate and fuse
        combined = torch.cat([sam_features, depth_features], dim=1)
        mask_logits = self.fusion(combined)

        # Return in expected format
        return {
            'masks': mask_logits,
            'presence_3d': torch.ones(B, 1).to(sam_features.device),  # Dummy
            'depth_confidence': torch.ones(B, 1, H, W).to(sam_features.device)  # Dummy
        }


class CrossAttentionFusionHead(nn.Module):
    """
    Bidirectional cross-attention fusion: depth ↔ SAM3 features

    Key insight: Both modalities must be preserved in the output.
    - Depth→SAM attention: depth learns what semantic regions are important
    - SAM→Depth attention: semantics learn where in 3D space to focus
    - Final output combines both attended features + originals
    """

    def __init__(self, sam_channels=256, depth_channels=1, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_size = 16  # 16x16 = 256 tokens for stable attention

        # Project depth to hidden dim
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Project SAM features to hidden dim
        self.sam_proj = nn.Sequential(
            nn.Conv2d(sam_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Bidirectional cross-attention (depth→SAM and SAM→depth)
        self.depth_to_sam_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.sam_to_depth_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )

        # Layer norms for stability
        self.depth_norm = nn.LayerNorm(hidden_dim)
        self.sam_norm = nn.LayerNorm(hidden_dim)
        self.post_attn_depth_norm = nn.LayerNorm(hidden_dim)
        self.post_attn_sam_norm = nn.LayerNorm(hidden_dim)

        # Combine the attended features (depth_attended + sam_attended + originals)
        # Input: 4 * hidden_dim (depth_orig, sam_orig, depth_attended, sam_attended)
        self.combine = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    if m.out_channels == 1:
                        # Small init for output layer - allows gradients to flow
                        nn.init.normal_(m.weight, mean=0, std=0.01)
                        nn.init.constant_(m.bias, -2.0)  # Start with low predictions
                    else:
                        nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, sam_features, depth_map):
        """
        Args:
            sam_features: [B, C, H, W] from SAM3
            depth_map: [B, 1, H', W'] from DA3
        """
        B, C, H, W = sam_features.shape

        # Resize depth to match SAM features
        if depth_map.shape[2:] != (H, W):
            depth_map = F.interpolate(depth_map, size=(H, W), mode='bilinear', align_corners=False)

        # Project to hidden dim at full resolution
        depth_feat = self.depth_proj(depth_map)   # [B, hidden_dim, H, W]
        sam_feat = self.sam_proj(sam_features)     # [B, hidden_dim, H, W]

        # Downsample for attention (reduces memory and improves stability)
        depth_small = F.adaptive_avg_pool2d(depth_feat, (self.attn_size, self.attn_size))
        sam_small = F.adaptive_avg_pool2d(sam_feat, (self.attn_size, self.attn_size))

        # Reshape for attention: [B, attn_size^2, hidden_dim]
        depth_seq = depth_small.flatten(2).permute(0, 2, 1)  # [B, 256, D]
        sam_seq = sam_small.flatten(2).permute(0, 2, 1)      # [B, 256, D]

        # Bidirectional cross-attention with pre-norm
        # Use float32 for attention stability
        depth_normed = self.depth_norm(depth_seq.float())
        sam_normed = self.sam_norm(sam_seq.float())

        depth_attended, _ = self.depth_to_sam_attn(
            depth_normed, sam_normed, sam_normed
        )
        depth_attended = depth_seq.float() + depth_attended  # Residual

        # SAM attends to depth: "where spatially should I focus?"
        sam_attended, _ = self.sam_to_depth_attn(
            sam_normed, depth_normed, depth_normed
        )
        sam_attended = sam_seq.float() + sam_attended  # Residual

        # Final normalization with clamping for stability
        depth_attended = self.post_attn_depth_norm(torch.clamp(depth_attended, -100, 100))
        sam_attended = self.post_attn_sam_norm(torch.clamp(sam_attended, -100, 100))

        # Reshape back to spatial
        depth_attended = depth_attended.permute(0, 2, 1).reshape(
            B, self.hidden_dim, self.attn_size, self.attn_size
        )
        sam_attended = sam_attended.permute(0, 2, 1).reshape(
            B, self.hidden_dim, self.attn_size, self.attn_size
        )

        # Upsample attended features to original resolution
        depth_attended = F.interpolate(depth_attended, size=(H, W), mode='bilinear', align_corners=False)
        sam_attended = F.interpolate(sam_attended, size=(H, W), mode='bilinear', align_corners=False)

        # Combine ALL features: original + attended for both modalities
        # This ensures we don't lose information from either modality
        combined = torch.cat([
            depth_feat,      # Original depth features
            sam_feat,        # Original SAM features (required)
            depth_attended,  # Depth refined by attending to SAM
            sam_attended,    # SAM refined by attending to depth
        ], dim=1)

        # Fuse and output
        fused = self.combine(combined)
        mask_logits = self.output(fused)

        return {
            'masks': mask_logits,
            'presence_3d': torch.ones(B, 1).to(sam_features.device),
            'depth_confidence': torch.ones(B, 1, H, W).to(sam_features.device)
        }


class GatedFusionHead(nn.Module):
    """
    Gated fusion: learn per-pixel gates to blend depth and SAM3 features
    Gate learns which modality to trust at each spatial location
    """

    def __init__(self, sam_channels=256, depth_channels=1, hidden_dim=256, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Depth feature extraction
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # SAM feature projection
        self.sam_proj = nn.Conv2d(sam_channels, hidden_dim, kernel_size=1) if sam_channels != hidden_dim else nn.Identity()

        # Gate network: takes both features, outputs per-pixel gate [0, 1]
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()  # Gate in [0, 1]
        )

        # Fusion layers after gating
        fusion_layers = []
        for i in range(num_layers):
            fusion_layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        fusion_layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=1))
        self.fusion = nn.Sequential(*fusion_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    if m.out_channels == 1 and not isinstance(m, nn.Sequential):
                        nn.init.constant_(m.bias, -2.0)
                    else:
                        nn.init.constant_(m.bias, 0.01)

    def forward(self, sam_features, depth_map):
        """
        Args:
            sam_features: [B, C, H, W] from SAM3
            depth_map: [B, 1, H', W'] from DA3

        Gate formula: fused = gate * depth_feat + (1 - gate) * sam_feat
        """
        B, C, H, W = sam_features.shape

        # Resize depth
        if depth_map.shape[2:] != (H, W):
            depth_map = F.interpolate(depth_map, size=(H, W), mode='bilinear', align_corners=False)

        # Extract features
        depth_feat = self.depth_encoder(depth_map)  # [B, hidden_dim, H, W]
        sam_feat = self.sam_proj(sam_features)       # [B, hidden_dim, H, W]

        # Compute gate
        combined_for_gate = torch.cat([sam_feat, depth_feat], dim=1)
        gate = self.gate_net(combined_for_gate)  # [B, 1, H, W] in [0, 1]

        # Gated fusion: blend features based on learned gate
        # gate=1 means trust depth, gate=0 means trust SAM3
        fused = gate * depth_feat + (1 - gate) * sam_feat

        # Generate mask
        mask_logits = self.fusion(fused)

        return {
            'masks': mask_logits,
            'gate': gate,  # Return gate for visualization/analysis
            'presence_3d': torch.ones(B, 1).to(sam_features.device),
            'depth_confidence': torch.ones(B, 1, H, W).to(sam_features.device)
        }


class GASADecoderHead(nn.Module):
    """
    GASA as Decoder: Simple mask head after GASA-processed features.

    Key insight: If GASA already does cross-view fusion with geometric bias,
    we might not need a complex fusion head. Just a simple conv to predict masks.

    This tests the hypothesis: "GASA IS the decoder, fusion head is redundant"
    """

    def __init__(self, feature_dim=256, hidden_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        # Simple mask head - just 2 convs
        self.mask_head = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

        # Initialize final layer with negative bias (start predicting background)
        nn.init.kaiming_normal_(self.mask_head[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.mask_head[0].bias, 0.01)
        nn.init.kaiming_normal_(self.mask_head[3].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.mask_head[3].bias, -2.0)

    def forward(self, sam_features, depth_map):
        """
        Args:
            sam_features: [B, C, H, W] GASA-processed features
            depth_map: [B, 1, H', W'] depth (NOT USED - GASA already fused geometry)

        Returns:
            dict with 'masks' key containing [B, 1, H, W] mask logits
        """
        B, C, H, W = sam_features.shape

        # Simple forward - depth already incorporated via GASA's geometric bias
        mask_logits = self.mask_head(sam_features)

        return {
            'masks': mask_logits,
            'presence_3d': torch.ones(B, 1).to(sam_features.device),
            'depth_confidence': torch.ones(B, 1, H, W).to(sam_features.device)
        }


class WorldPEOnlyHead(nn.Module):
    """
    MV-SAM Style: No explicit cross-view attention, just World-Space PE.

    Key insight from MV-SAM: Adding 3D positional embeddings to features
    provides enough inductive bias for cross-view consistency WITHOUT
    explicit cross-view attention.

    This is the simplest baseline - tests if GASA attention is even needed.
    """

    def __init__(self, feature_dim=256, hidden_dim=256, num_layers=3):
        super().__init__()
        self.feature_dim = feature_dim

        # World-Space PE (MV-SAM style)
        self.world_pe = WorldSpacePE(d_model=feature_dim, num_frequencies=10)

        # Per-view decoder (no cross-view communication)
        layers = []
        in_dim = feature_dim  # Features already have PE added
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = out_dim
        layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

        # Initialize
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    if m.out_channels == 1:
                        nn.init.constant_(m.bias, -2.0)
                    else:
                        nn.init.constant_(m.bias, 0.01)

    def forward(self, sam_features, depth_map, intrinsics=None, extrinsics=None):
        """
        Args:
            sam_features: [B, C, H, W] or [B*N, C, H, W] SAM features
            depth_map: [B, 1, H', W'] or [B*N, 1, H', W'] depth maps
            intrinsics: [B, N, 3, 3] or [B*N, 3, 3] camera intrinsics
            extrinsics: [B, N, 4, 4] or [B*N, 4, 4] camera extrinsics

        Returns:
            dict with 'masks' key
        """
        B, C, H, W = sam_features.shape

        # If we have camera params, add World-Space PE
        if intrinsics is not None and extrinsics is not None:
            # Compute world coordinates from depth
            world_coords = depth_to_world_coords(
                depth_map, intrinsics, extrinsics,
                image_size=(H, W) if depth_map.shape[-2:] != (H, W) else None
            )

            # Resize world coords to feature resolution if needed
            if world_coords.shape[1:3] != (H, W):
                # world_coords: [B, H', W', 3] -> [B, 3, H', W'] -> interpolate -> [B, H, W, 3]
                wc = world_coords.permute(0, 3, 1, 2)  # [B, 3, H', W']
                wc = F.interpolate(wc, size=(H, W), mode='bilinear', align_corners=False)
                world_coords = wc.permute(0, 2, 3, 1)  # [B, H, W, 3]

            # Get PE: [B, H, W, C]
            pe = self.world_pe(world_coords)

            # Add to features: [B, C, H, W] + [B, H, W, C].permute
            sam_features = sam_features + pe.permute(0, 3, 1, 2)

        # Per-view decoder (independent, no cross-view attention)
        mask_logits = self.decoder(sam_features)

        return {
            'masks': mask_logits,
            'presence_3d': torch.ones(B, 1).to(sam_features.device),
            'depth_confidence': torch.ones(B, 1, H, W).to(sam_features.device)
        }


class LightweightCrossViewHead(nn.Module):
    """
    Lightweight Cross-View Fusion (~100K params, ~10x faster than GASA)

    Key insight: Maybe we don't need full attention. Instead:
    1. Global pool each view's features -> [B, N, D]
    2. Concatenate all views -> [B, N*D]
    3. MLP predicts per-view importance weights -> [B, N]
    4. Weighted sum of spatial features -> [B, D, H, W]
    5. Simple decoder -> mask

    This tests: "Is expensive cross-view attention actually needed?"
    """

    def __init__(self, feature_dim=256, hidden_dim=128, max_views=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_views = max_views

        # View importance predictor: takes pooled features from all views
        # Input: [B, max_views * feature_dim], Output: [B, max_views]
        self.view_importance = nn.Sequential(
            nn.Linear(max_views * feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, max_views),
        )

        # Simple mask decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

        # Initialize
        nn.init.constant_(self.decoder[-1].bias, -2.0)

    def forward(self, sam_features, depth_map, num_views=None):
        """
        Args:
            sam_features: [B*N, C, H, W] features (flattened batch)
            depth_map: [B*N, 1, H', W'] depth (not used heavily here)
            num_views: N (must be provided for multi-view)
        """
        BN, C, H, W = sam_features.shape

        if num_views is None:
            # Single view mode - just decode directly
            mask_logits = self.decoder(sam_features)
            return {
                'masks': mask_logits,
                'presence_3d': torch.ones(BN, 1).to(sam_features.device),
                'depth_confidence': torch.ones(BN, 1, H, W).to(sam_features.device)
            }

        B = BN // num_views
        N = num_views

        # Reshape to [B, N, C, H, W]
        features = sam_features.view(B, N, C, H, W)

        # Global pool per view: [B, N, C]
        pooled = features.mean(dim=[3, 4])  # [B, N, C]

        # Pad to max_views if needed
        if N < self.max_views:
            pad = torch.zeros(B, self.max_views - N, C, device=pooled.device)
            pooled_padded = torch.cat([pooled, pad], dim=1)
        else:
            pooled_padded = pooled[:, :self.max_views]

        # Predict view importance: [B, max_views]
        pooled_flat = pooled_padded.view(B, -1)  # [B, max_views * C]
        importance = self.view_importance(pooled_flat)  # [B, max_views]
        importance = importance[:, :N]  # [B, N] - only use actual views
        importance = F.softmax(importance, dim=1)  # Normalize

        # Weighted sum of features: [B, C, H, W]
        # importance: [B, N] -> [B, N, 1, 1, 1]
        weights = importance.view(B, N, 1, 1, 1)
        fused = (features * weights).sum(dim=1)  # [B, C, H, W]

        # Decode to mask
        mask_logits = self.decoder(fused)  # [B, 1, H, W]

        # Expand back to all views (same mask for all views)
        mask_logits = mask_logits.unsqueeze(1).expand(-1, N, -1, -1, -1)
        mask_logits = mask_logits.reshape(BN, 1, H, W)

        return {
            'masks': mask_logits,
            'view_weights': importance,  # For analysis
            'presence_3d': torch.ones(BN, 1).to(sam_features.device),
            'depth_confidence': torch.ones(BN, 1, H, W).to(sam_features.device)
        }


class DeformableCrossViewHead(nn.Module):
    """
    Deformable Cross-View Attention (~500K params)

    Instead of attending to ALL tokens, learn to attend to K=8 sparse points
    per query. Much faster than full attention, often works as well.

    Inspired by Deformable DETR and Deformable Attention.
    """

    def __init__(self, feature_dim=256, hidden_dim=128, num_points=8, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_points = num_points
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # Predict sampling offsets: [B, N, H, W] -> [B, N, H, W, num_points, 2]
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_points * 2, kernel_size=1),
        )

        # Predict attention weights for each point
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_points, kernel_size=1),
        )

        # Value projection
        self.value_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        # Mask decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

        nn.init.constant_(self.decoder[-1].bias, -2.0)
        # Initialize offsets to small values
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)

    def forward(self, sam_features, depth_map, num_views=None):
        """
        Args:
            sam_features: [B*N, C, H, W]
            depth_map: [B*N, 1, H', W']
            num_views: N
        """
        BN, C, H, W = sam_features.shape

        if num_views is None or num_views == 1:
            # Single view - just decode
            mask_logits = self.decoder(sam_features)
            return {
                'masks': mask_logits,
                'presence_3d': torch.ones(BN, 1).to(sam_features.device),
                'depth_confidence': torch.ones(BN, 1, H, W).to(sam_features.device)
            }

        B = BN // num_views
        N = num_views

        # Predict offsets and weights
        offsets = self.offset_predictor(sam_features)  # [BN, num_points*2, H, W]
        offsets = offsets.view(BN, self.num_points, 2, H, W).permute(0, 3, 4, 1, 2)  # [BN, H, W, K, 2]
        offsets = torch.tanh(offsets) * 0.5  # Limit offset range to [-0.5, 0.5] of image

        weights = self.weight_predictor(sam_features)  # [BN, num_points, H, W]
        weights = weights.permute(0, 2, 3, 1)  # [BN, H, W, K]
        weights = F.softmax(weights, dim=-1)

        # Values
        values = self.value_proj(sam_features)  # [BN, C, H, W]

        # Create sampling grid
        # Base grid: [H, W, 2] normalized to [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=sam_features.device),
            torch.linspace(-1, 1, W, device=sam_features.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).unsqueeze(3)  # [1, H, W, 1, 2]

        # Add offsets to get sampling locations
        sample_grid = base_grid + offsets  # [BN, H, W, K, 2]

        # Reshape for grid_sample: [BN, H*W*K, 1, 2]
        sample_grid_flat = sample_grid.view(BN, H * W * self.num_points, 1, 2)

        # Sample values at offset locations
        # values: [BN, C, H, W] -> sample at [BN, H*W*K] locations
        sampled = F.grid_sample(values, sample_grid_flat, mode='bilinear', align_corners=True)
        # sampled: [BN, C, H*W*K, 1] -> [BN, C, H, W, K]
        sampled = sampled.view(BN, C, H, W, self.num_points)

        # Weighted sum over K points
        # weights: [BN, H, W, K] -> [BN, 1, H, W, K]
        weights = weights.unsqueeze(1)
        attended = (sampled * weights).sum(dim=-1)  # [BN, C, H, W]

        # Output projection + residual
        out = self.out_proj(attended) + sam_features

        # Decode to mask
        mask_logits = self.decoder(out)

        return {
            'masks': mask_logits,
            'presence_3d': torch.ones(BN, 1).to(sam_features.device),
            'depth_confidence': torch.ones(BN, 1, H, W).to(sam_features.device)
        }


class CostVolumeHead(nn.Module):
    """
    Cost Volume Style Fusion (~300K params)

    Builds correlation volume between views (like stereo matching):
    1. For each pixel in view i, compute dot product with all pixels in view j
    2. This gives a 4D cost volume
    3. Use 3D convs to aggregate and predict mask

    No learned attention weights - purely geometric correlation.
    """

    def __init__(self, feature_dim=256, hidden_dim=64, corr_radius=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.corr_radius = corr_radius  # Local correlation window
        self.corr_levels = 4  # Pyramid levels

        # Feature projection (reduce dim for correlation)
        self.feat_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)

        # Correlation volume processor
        # Input: correlation features, Output: refined features
        corr_channels = (2 * corr_radius + 1) ** 2 * self.corr_levels
        self.corr_processor = nn.Sequential(
            nn.Conv2d(corr_channels + feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Mask decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

        nn.init.constant_(self.decoder[-1].bias, -2.0)

    def compute_correlation(self, feat1, feat2):
        """
        Compute local correlation between feat1 and feat2.

        Args:
            feat1: [B, C, H, W] query features
            feat2: [B, C, H, W] reference features

        Returns:
            corr: [B, (2r+1)^2, H, W] correlation volume
        """
        B, C, H, W = feat1.shape
        r = self.corr_radius

        # Normalize features
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)

        # Pad feat2 for local correlation
        feat2_pad = F.pad(feat2, [r, r, r, r], mode='replicate')

        # Compute local correlation
        corr_list = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                feat2_shift = feat2_pad[:, :, r+dy:r+dy+H, r+dx:r+dx+W]
                corr = (feat1 * feat2_shift).sum(dim=1, keepdim=True)  # [B, 1, H, W]
                corr_list.append(corr)

        corr = torch.cat(corr_list, dim=1)  # [B, (2r+1)^2, H, W]
        return corr

    def forward(self, sam_features, depth_map, num_views=None):
        """
        Args:
            sam_features: [B*N, C, H, W]
            depth_map: [B*N, 1, H', W']
            num_views: N
        """
        BN, C, H, W = sam_features.shape

        if num_views is None or num_views == 1:
            # Single view - no correlation possible
            feat = self.feat_proj(sam_features)
            # Just use zeros for correlation
            corr = torch.zeros(BN, (2*self.corr_radius+1)**2 * self.corr_levels, H, W, device=sam_features.device)
            combined = torch.cat([corr, sam_features], dim=1)
            processed = self.corr_processor(combined)
            mask_logits = self.decoder(processed)
            return {
                'masks': mask_logits,
                'presence_3d': torch.ones(BN, 1).to(sam_features.device),
                'depth_confidence': torch.ones(BN, 1, H, W).to(sam_features.device)
            }

        B = BN // num_views
        N = num_views

        # Project features
        feat = self.feat_proj(sam_features)  # [BN, hidden, H, W]
        feat = feat.view(B, N, -1, H, W)  # [B, N, hidden, H, W]

        # Compute correlation pyramid for each view pair
        all_corrs = []
        for v in range(N):
            view_corrs = []
            for scale in range(self.corr_levels):
                scale_factor = 2 ** scale
                if scale > 0:
                    # Reshape to 4D for avg_pool2d: [B*N, C, H, W]
                    feat_4d = feat.view(B * N, -1, H, W)
                    feat_scaled_4d = F.avg_pool2d(feat_4d, scale_factor)
                    H_s, W_s = feat_scaled_4d.shape[-2:]
                    feat_scaled = feat_scaled_4d.view(B, N, -1, H_s, W_s)
                else:
                    feat_scaled = feat

                # Correlate with all other views and average
                corrs_with_others = []
                for v2 in range(N):
                    if v2 != v:
                        corr = self.compute_correlation(feat_scaled[:, v], feat_scaled[:, v2])
                        corrs_with_others.append(corr)

                # Average correlation across other views
                avg_corr = torch.stack(corrs_with_others, dim=0).mean(dim=0)  # [B, (2r+1)^2, H', W']

                # Upsample back to original resolution
                if scale > 0:
                    avg_corr = F.interpolate(avg_corr, size=(H, W), mode='bilinear', align_corners=False)

                view_corrs.append(avg_corr)

            # Concatenate all scales
            view_corr = torch.cat(view_corrs, dim=1)  # [B, (2r+1)^2 * levels, H, W]
            all_corrs.append(view_corr)

        # Stack all views
        corr_volume = torch.stack(all_corrs, dim=1)  # [B, N, corr_channels, H, W]
        corr_volume = corr_volume.view(BN, -1, H, W)  # [BN, corr_channels, H, W]

        # Combine with original features
        combined = torch.cat([corr_volume, sam_features], dim=1)
        processed = self.corr_processor(combined)
        mask_logits = self.decoder(processed)

        return {
            'masks': mask_logits,
            'presence_3d': torch.ones(BN, 1).to(sam_features.device),
            'depth_confidence': torch.ones(BN, 1, H, W).to(sam_features.device)
        }


class CrossViewAttention(nn.Module):
    """
    Cross-View Attention module for multi-view consistency

    Allows views to communicate with each other before fusion:
    - Each view attends to all other views
    - Learns cross-view correspondences (same object across views)
    - Improves consistency without explicit 3D geometry
    """

    def __init__(self, feature_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attn_size = 16  # Downsample to 16x16 for efficiency

        # Cross-view attention layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm for stability
            )
            for _ in range(num_layers)
        ])

        # Positional encoding for views (learnable)
        # Max 32 views should be enough for MVImgNet
        self.view_pos_embed = nn.Parameter(torch.randn(1, 32, feature_dim) * 0.02)

        # Spatial position encoding (2D sinusoidal)
        self.register_buffer('spatial_pos', self._make_spatial_pos(self.attn_size))

        # Learnable residual scale (starts at 0.1 for stability with many views)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _make_spatial_pos(self, size):
        """Create 2D sinusoidal positional encoding"""
        h = w = size
        pos_embed = torch.zeros(1, h * w, self.feature_dim)

        d_model = self.feature_dim
        pe_h = torch.zeros(h, d_model // 2)
        pe_w = torch.zeros(w, d_model // 2)

        position_h = torch.arange(0, h).unsqueeze(1)
        position_w = torch.arange(0, w).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model // 2, 2) * -(torch.log(torch.tensor(10000.0)) / (d_model // 2)))

        pe_h[:, 0::2] = torch.sin(position_h * div_term[:d_model // 4])
        pe_h[:, 1::2] = torch.cos(position_h * div_term[:d_model // 4])
        pe_w[:, 0::2] = torch.sin(position_w * div_term[:d_model // 4])
        pe_w[:, 1::2] = torch.cos(position_w * div_term[:d_model // 4])

        # Combine height and width encodings
        for i in range(h):
            for j in range(w):
                pos_embed[0, i * w + j, :d_model // 2] = pe_h[i]
                pos_embed[0, i * w + j, d_model // 2:] = pe_w[j]

        return pos_embed

    def forward(self, features, num_views):
        """
        Apply cross-view attention

        Args:
            features: [B*N, C, H, W] features from all views (batched)
            num_views: N (number of views per sample)

        Returns:
            [B*N, C, H, W] features with cross-view attention applied
        """
        BN, C, H, W = features.shape
        B = BN // num_views
        N = num_views

        # Downsample for attention efficiency
        features_small = F.adaptive_avg_pool2d(features, (self.attn_size, self.attn_size))

        # Reshape to [B, N, C, h, w]
        features_small = features_small.view(B, N, C, self.attn_size, self.attn_size)

        # Flatten spatial dims: [B, N, h*w, C]
        features_flat = features_small.permute(0, 1, 3, 4, 2).reshape(B, N, -1, C)

        # Add spatial position encoding
        spatial_pos = self.spatial_pos[:, :self.attn_size * self.attn_size, :]
        features_flat = features_flat + spatial_pos.unsqueeze(1)  # [B, N, h*w, C]

        # Add view position encoding
        view_pos = self.view_pos_embed[:, :N, :].unsqueeze(2)  # [1, N, 1, C]
        features_flat = features_flat + view_pos  # [B, N, h*w, C]

        # Reshape for cross-view attention: [B, N*h*w, C]
        # Each position in each view can attend to all positions in all views
        seq_len = N * self.attn_size * self.attn_size
        features_seq = features_flat.reshape(B, seq_len, C)

        # Apply transformer layers with float32 for numerical stability
        features_seq = features_seq.float()
        for layer in self.layers:
            features_seq = layer(features_seq)

        # Clamp to prevent extreme values
        features_seq = torch.clamp(features_seq, min=-100, max=100)

        # Reshape back: [B, N, h, w, C] -> [B*N, C, h, w]
        features_small = features_seq.reshape(B, N, self.attn_size, self.attn_size, C)
        features_small = features_small.permute(0, 1, 4, 2, 3).reshape(BN, C, self.attn_size, self.attn_size)

        # Upsample back to original resolution
        features_attended = F.interpolate(features_small, size=(H, W), mode='bilinear', align_corners=False)

        # Scaled residual connection (prevents gradient explosion with many views)
        # residual_scale starts at 0.1 and is learned during training
        return features + self.residual_scale * features_attended.to(features.dtype)


def depth_to_world_coords(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: tuple = None
) -> torch.Tensor:
    """
    Convert depth maps to 3D world coordinates.

    This is the key function for World-Space PE per RESEARCH_STRATEGY.md:
    P_world = R^T @ (d * K^{-1} @ [u, v, 1]^T - t)

    Args:
        depth: [B, N, 1, H, W] or [B*N, 1, H, W] depth maps
        intrinsics: [B, N, 3, 3] or [B*N, 3, 3] camera intrinsic matrices K
        extrinsics: [B, N, 4, 4] or [B*N, 4, 4] camera extrinsic matrices [R|t]
        image_size: Optional (H_img, W_img) if depth is at different resolution

    Returns:
        world_coords: [B, N, H, W, 3] or [B*N, H, W, 3] XYZ world coordinates
    """
    # Handle batching
    if depth.dim() == 5:
        B, N, _, H, W = depth.shape
        depth = depth.view(B * N, 1, H, W)
        intrinsics = intrinsics.view(B * N, 3, 3)
        extrinsics = extrinsics.view(B * N, 4, 4)
        batch_mode = 'BN'
    else:
        BN, _, H, W = depth.shape
        B = BN // (intrinsics.shape[0] if intrinsics.dim() == 3 else 1)
        N = BN // B
        batch_mode = 'flat'

    device = depth.device
    dtype = depth.dtype

    # Create pixel coordinate grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )

    # If depth resolution differs from image resolution, scale pixel coords
    if image_size is not None:
        H_img, W_img = image_size
        u = u * (W_img / W)
        v = v * (H_img / H)

    # Homogeneous pixel coords: [3, H*W]
    ones = torch.ones_like(u)
    pixels = torch.stack([u, v, ones], dim=0).reshape(3, -1)  # [3, H*W]

    # Inverse intrinsics: [BN, 3, 3]
    K_inv = torch.inverse(intrinsics.float())

    # Extract R and t from extrinsics [R|t] = [3x3 | 3x1]
    R = extrinsics[:, :3, :3].float()  # [BN, 3, 3]
    t = extrinsics[:, :3, 3:4].float()  # [BN, 3, 1]

    # Unproject to camera coordinates: P_cam = d * K^{-1} @ [u,v,1]
    # pixels: [3, H*W] -> [1, 3, H*W] -> [BN, 3, H*W]
    pixels_batch = pixels.unsqueeze(0).expand(BN, -1, -1)  # [BN, 3, H*W]

    # K_inv @ pixels: [BN, 3, 3] @ [BN, 3, H*W] -> [BN, 3, H*W]
    rays = torch.bmm(K_inv, pixels_batch)  # [BN, 3, H*W]

    # Depth: [BN, 1, H, W] -> [BN, 1, H*W]
    depth_flat = depth.view(BN, 1, -1)  # [BN, 1, H*W]

    # Camera coordinates: d * rays
    P_cam = depth_flat * rays  # [BN, 3, H*W]

    # Transform to world coordinates: P_world = R^T @ (P_cam - t)
    # Note: extrinsics typically encode world-to-camera, so we invert with R^T
    P_cam_centered = P_cam - t  # [BN, 3, H*W]
    R_T = R.transpose(1, 2)  # [BN, 3, 3]
    P_world = torch.bmm(R_T, P_cam_centered)  # [BN, 3, H*W]

    # Reshape to spatial: [BN, 3, H, W] -> [BN, H, W, 3]
    world_coords = P_world.view(BN, 3, H, W).permute(0, 2, 3, 1)

    if batch_mode == 'BN':
        world_coords = world_coords.view(B, N, H, W, 3)

    return world_coords


class WorldSpacePE(nn.Module):
    """
    World-Space Positional Encoding per RESEARCH_STRATEGY.md.

    Key insight: If two pixels in different views map to the same 3D world point,
    they should have IDENTICAL positional embeddings. This provides strong
    inductive bias for cross-view consistency.

    Formula (NeRF-style):
        PE(P_world) = [sin(2^0 π P), cos(2^0 π P), ..., sin(2^{L-1} π P), cos(2^{L-1} π P)]

    Unlike the current depth_pe which just encodes per-view depth values,
    this encodes actual 3D world coordinates.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_frequencies: int = 10,
        max_coord: float = 10.0,  # Expected max world coordinate magnitude
        include_input: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        self.max_coord = max_coord
        self.include_input = include_input

        # Frequency bands: 2^0, 2^1, ..., 2^{L-1}
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
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

        # Learnable scale (starts small)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, world_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D world coordinates to positional embeddings.

        Args:
            world_coords: [..., 3] world coordinates (X, Y, Z)

        Returns:
            pe: [..., d_model] positional embeddings
        """
        original_shape = world_coords.shape[:-1]
        xyz = world_coords.reshape(-1, 3)  # [*, 3]

        # Normalize coordinates for stable encoding
        xyz_norm = xyz / self.max_coord

        # Apply frequency encoding
        # xyz_scaled: [*, L, 3] where L = num_frequencies
        xyz_scaled = xyz_norm.unsqueeze(1) * self.freq_bands.view(1, -1, 1) * torch.pi

        # Sin and cos
        sin_enc = torch.sin(xyz_scaled)  # [*, L, 3]
        cos_enc = torch.cos(xyz_scaled)  # [*, L, 3]

        # Interleave and flatten: [*, L*3*2]
        encoding = torch.stack([sin_enc, cos_enc], dim=-1)  # [*, L, 3, 2]
        encoding = encoding.reshape(xyz.shape[0], -1)  # [*, L*3*2]

        # Optionally include raw normalized input
        if self.include_input:
            encoding = torch.cat([xyz_norm, encoding], dim=-1)

        # Project to model dimension
        pe = self.proj(encoding)

        # Reshape to original spatial dimensions
        pe = pe.reshape(*original_shape, self.d_model)

        return self.scale * pe


class WorldSpaceCrossViewAttention(nn.Module):
    """
    Cross-View Attention with proper World-Space Positional Encoding.

    Per RESEARCH_STRATEGY.md: Same 3D point → identical positional embedding,
    regardless of which view it appears in. This is the key insight from MV-SAM.

    Requires camera intrinsics and extrinsics to compute world coordinates.
    Falls back to depth-only PE if camera params not provided.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_frequencies: int = 10,
        max_coord: float = 10.0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attn_size = 16  # Downsample for efficiency

        # World-Space Positional Encoding
        self.world_pe = WorldSpacePE(
            d_model=feature_dim,
            num_frequencies=num_frequencies,
            max_coord=max_coord
        )

        # Fallback depth encoder (when camera params not available)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
        )
        for m in self.depth_encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.depth_pe_scale = nn.Parameter(torch.tensor(0.1))

        # Cross-view attention layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        # View positional encoding (learnable)
        self.view_pos_embed = nn.Parameter(torch.randn(1, 32, feature_dim) * 0.02)

        # 2D spatial positional encoding
        self.register_buffer('spatial_pos', self._make_spatial_pos(self.attn_size))

        # Learnable residual scale
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _make_spatial_pos(self, size):
        """Create 2D sinusoidal positional encoding"""
        h = w = size
        d_model = self.feature_dim

        pos_embed = torch.zeros(1, h * w, d_model)
        pe_h = torch.zeros(h, d_model // 2)
        pe_w = torch.zeros(w, d_model // 2)

        position_h = torch.arange(0, h).unsqueeze(1).float()
        position_w = torch.arange(0, w).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() *
                            -(torch.log(torch.tensor(10000.0)) / (d_model // 2)))

        pe_h[:, 0::2] = torch.sin(position_h * div_term[:d_model // 4])
        pe_h[:, 1::2] = torch.cos(position_h * div_term[:d_model // 4])
        pe_w[:, 0::2] = torch.sin(position_w * div_term[:d_model // 4])
        pe_w[:, 1::2] = torch.cos(position_w * div_term[:d_model // 4])

        for i in range(h):
            for j in range(w):
                pos_embed[0, i * w + j, :d_model // 2] = pe_h[i]
                pos_embed[0, i * w + j, d_model // 2:] = pe_w[j]

        return pos_embed

    def forward(
        self,
        features: torch.Tensor,
        num_views: int,
        depth: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        extrinsics: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply world-space-aware cross-view attention.

        Args:
            features: [B*N, C, H, W] features from all views
            num_views: N
            depth: [B*N, 1, H_d, W_d] depth maps
            intrinsics: [B*N, 3, 3] or [B, N, 3, 3] camera intrinsics K
            extrinsics: [B*N, 4, 4] or [B, N, 4, 4] camera extrinsics [R|t]

        Returns:
            [B*N, C, H, W] features with cross-view attention applied
        """
        BN, C, H, W = features.shape
        B = BN // num_views
        N = num_views

        # Downsample features for attention efficiency
        features_small = F.adaptive_avg_pool2d(features, (self.attn_size, self.attn_size))

        # Compute positional embeddings
        pe = None
        if depth is not None:
            depth_small = F.adaptive_avg_pool2d(depth, (self.attn_size, self.attn_size))

            # Use world-space PE if camera params available
            if intrinsics is not None and extrinsics is not None:
                # Reshape camera params if needed
                if intrinsics.dim() == 4:  # [B, N, 3, 3]
                    intrinsics = intrinsics.view(BN, 3, 3)
                if extrinsics.dim() == 4:  # [B, N, 4, 4]
                    extrinsics = extrinsics.view(BN, 4, 4)

                # Compute world coordinates: [BN, h, w, 3]
                world_coords = depth_to_world_coords(
                    depth_small, intrinsics, extrinsics
                )

                # Apply World-Space PE: [BN, h, w, C]
                pe = self.world_pe(world_coords)
                # Reshape to [BN, C, h, w]
                pe = pe.permute(0, 3, 1, 2)
            else:
                # Fallback to depth-only PE
                depth_min = depth_small.view(BN, -1).min(dim=1, keepdim=True)[0].view(BN, 1, 1, 1)
                depth_max = depth_small.view(BN, -1).max(dim=1, keepdim=True)[0].view(BN, 1, 1, 1)
                depth_norm = (depth_small - depth_min) / (depth_max - depth_min + 1e-6)
                pe = self.depth_encoder(depth_norm) * self.depth_pe_scale

        # Reshape to [B, N, C, h, w]
        features_small = features_small.view(B, N, C, self.attn_size, self.attn_size)

        # Flatten spatial dims: [B, N, h*w, C]
        features_flat = features_small.permute(0, 1, 3, 4, 2).reshape(B, N, -1, C)

        # Add 2D spatial position encoding
        spatial_pos = self.spatial_pos[:, :self.attn_size * self.attn_size, :]
        features_flat = features_flat + spatial_pos.unsqueeze(1)

        # Add view position encoding
        view_pos = self.view_pos_embed[:, :N, :].unsqueeze(2)
        features_flat = features_flat + view_pos

        # Add world-space or depth PE
        if pe is not None:
            pe = pe.view(B, N, C, self.attn_size, self.attn_size)
            pe_flat = pe.permute(0, 1, 3, 4, 2).reshape(B, N, -1, C)
            features_flat = features_flat + pe_flat

        # Reshape for cross-view attention: [B, N*h*w, C]
        seq_len = N * self.attn_size * self.attn_size
        features_seq = features_flat.reshape(B, seq_len, C)

        # Apply transformer layers
        features_seq = features_seq.float()
        for layer in self.layers:
            features_seq = layer(features_seq)

        features_seq = torch.clamp(features_seq, min=-100, max=100)

        # Reshape back: [B, N, h, w, C] -> [B*N, C, h, w]
        features_small = features_seq.reshape(B, N, self.attn_size, self.attn_size, C)
        features_small = features_small.permute(0, 1, 4, 2, 3).reshape(
            BN, C, self.attn_size, self.attn_size
        )

        # Upsample back to original resolution
        features_attended = F.interpolate(
            features_small, size=(H, W), mode='bilinear', align_corners=False
        )

        # Scaled residual connection
        return features + self.residual_scale * features_attended.to(features.dtype)


class CrossViewAttention3D(nn.Module):
    """
    3D-Aware Cross-View Attention module for multi-view consistency.

    Key improvement over CrossViewAttention:
    - Uses depth maps to create 3D positional embeddings
    - Points at similar depths (likely same 3D location) attend more strongly
    - Follows MV-SAM's insight: 3D PE enables implicit cross-view consistency

    This is our core contribution for multi-view consistency.
    """

    def __init__(self, feature_dim=256, num_heads=4, num_layers=2, dropout=0.1,
                 depth_embed_dim=64, use_depth_pe=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attn_size = 16  # Downsample to 16x16 for efficiency
        self.use_depth_pe = use_depth_pe

        # Depth positional encoding (our key addition)
        if use_depth_pe:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, depth_embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(depth_embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(depth_embed_dim, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
            )
            # Initialize to near-zero so depth PE starts small
            for m in self.depth_encoder.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Cross-view attention layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm for stability
            )
            for _ in range(num_layers)
        ])

        # Positional encoding for views (learnable)
        self.view_pos_embed = nn.Parameter(torch.randn(1, 32, feature_dim) * 0.02)

        # Spatial position encoding (2D sinusoidal)
        self.register_buffer('spatial_pos', self._make_spatial_pos(self.attn_size))

        # Learnable residual scale
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Depth PE scale (learnable, starts small)
        if use_depth_pe:
            self.depth_pe_scale = nn.Parameter(torch.tensor(0.1))

    def _make_spatial_pos(self, size):
        """Create 2D sinusoidal positional encoding"""
        h = w = size
        pos_embed = torch.zeros(1, h * w, self.feature_dim)

        d_model = self.feature_dim
        pe_h = torch.zeros(h, d_model // 2)
        pe_w = torch.zeros(w, d_model // 2)

        position_h = torch.arange(0, h).unsqueeze(1)
        position_w = torch.arange(0, w).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model // 2, 2) * -(torch.log(torch.tensor(10000.0)) / (d_model // 2)))

        pe_h[:, 0::2] = torch.sin(position_h * div_term[:d_model // 4])
        pe_h[:, 1::2] = torch.cos(position_h * div_term[:d_model // 4])
        pe_w[:, 0::2] = torch.sin(position_w * div_term[:d_model // 4])
        pe_w[:, 1::2] = torch.cos(position_w * div_term[:d_model // 4])

        for i in range(h):
            for j in range(w):
                pos_embed[0, i * w + j, :d_model // 2] = pe_h[i]
                pos_embed[0, i * w + j, d_model // 2:] = pe_w[j]

        return pos_embed

    def forward(self, features, num_views, depth=None):
        """
        Apply 3D-aware cross-view attention.

        Args:
            features: [B*N, C, H, W] features from all views (batched)
            num_views: N (number of views per sample)
            depth: [B*N, 1, H_d, W_d] depth maps (optional, enables 3D PE)

        Returns:
            [B*N, C, H, W] features with cross-view attention applied
        """
        BN, C, H, W = features.shape
        B = BN // num_views
        N = num_views

        # Downsample for attention efficiency
        features_small = F.adaptive_avg_pool2d(features, (self.attn_size, self.attn_size))

        # Compute depth positional embeddings if depth provided
        depth_pe = None
        if self.use_depth_pe and depth is not None:
            # Resize depth to attention size
            depth_small = F.adaptive_avg_pool2d(depth, (self.attn_size, self.attn_size))
            # Normalize depth to [0, 1] range for stability
            depth_min = depth_small.view(BN, -1).min(dim=1, keepdim=True)[0].view(BN, 1, 1, 1)
            depth_max = depth_small.view(BN, -1).max(dim=1, keepdim=True)[0].view(BN, 1, 1, 1)
            depth_norm = (depth_small - depth_min) / (depth_max - depth_min + 1e-6)
            # Encode depth to positional embeddings
            depth_pe = self.depth_encoder(depth_norm)  # [B*N, C, h, w]

        # Reshape to [B, N, C, h, w]
        features_small = features_small.view(B, N, C, self.attn_size, self.attn_size)

        # Flatten spatial dims: [B, N, h*w, C]
        features_flat = features_small.permute(0, 1, 3, 4, 2).reshape(B, N, -1, C)

        # Add spatial position encoding (2D)
        spatial_pos = self.spatial_pos[:, :self.attn_size * self.attn_size, :]
        features_flat = features_flat + spatial_pos.unsqueeze(1)  # [B, N, h*w, C]

        # Add view position encoding
        view_pos = self.view_pos_embed[:, :N, :].unsqueeze(2)  # [1, N, 1, C]
        features_flat = features_flat + view_pos  # [B, N, h*w, C]

        # Add depth positional encoding (3D - our key addition!)
        if depth_pe is not None:
            depth_pe = depth_pe.view(B, N, C, self.attn_size, self.attn_size)
            depth_pe_flat = depth_pe.permute(0, 1, 3, 4, 2).reshape(B, N, -1, C)
            features_flat = features_flat + self.depth_pe_scale * depth_pe_flat

        # Reshape for cross-view attention: [B, N*h*w, C]
        seq_len = N * self.attn_size * self.attn_size
        features_seq = features_flat.reshape(B, seq_len, C)

        # Apply transformer layers with float32 for numerical stability
        features_seq = features_seq.float()
        for layer in self.layers:
            features_seq = layer(features_seq)

        # Clamp to prevent extreme values
        features_seq = torch.clamp(features_seq, min=-100, max=100)

        # Reshape back: [B, N, h, w, C] -> [B*N, C, h, w]
        features_small = features_seq.reshape(B, N, self.attn_size, self.attn_size, C)
        features_small = features_small.permute(0, 1, 4, 2, 3).reshape(BN, C, self.attn_size, self.attn_size)

        # Upsample back to original resolution
        features_attended = F.interpolate(features_small, size=(H, W), mode='bilinear', align_corners=False)

        # Scaled residual connection
        return features + self.residual_scale * features_attended.to(features.dtype)


class DepthAdapter(nn.Module):
    """
    Lightweight adapter to inject depth information into SAM3 features.
    Uses FiLM-style conditioning (scale and shift) to modulate features.
    Does NOT require unfreezing SAM3.
    """

    def __init__(self, feature_dim=256, depth_hidden=64):
        super().__init__()

        # Depth encoder - small network to process depth
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, depth_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(depth_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_hidden, depth_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(depth_hidden),
            nn.ReLU(inplace=True),
        )

        # FiLM parameters: scale (gamma) and shift (beta)
        self.film_scale = nn.Conv2d(depth_hidden, feature_dim, kernel_size=1)
        self.film_shift = nn.Conv2d(depth_hidden, feature_dim, kernel_size=1)

        # Initialize scale to 1 and shift to 0 (identity at start)
        nn.init.ones_(self.film_scale.weight.data * 0 + 1)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

    def forward(self, sam_features, depth):
        """
        Apply FiLM conditioning: out = gamma * sam_features + beta

        Args:
            sam_features: [B, C, H, W] from SAM3
            depth: [B, 1, H', W'] from DA3

        Returns:
            [B, C, H, W] depth-conditioned features
        """
        B, C, H, W = sam_features.shape

        # Resize depth to match features
        if depth.shape[2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)

        # Encode depth
        depth_feat = self.depth_encoder(depth)

        # Compute FiLM parameters
        gamma = self.film_scale(depth_feat)  # [B, C, H, W]
        beta = self.film_shift(depth_feat)   # [B, C, H, W]

        # Apply FiLM: scale + shift
        return gamma * sam_features + beta


class DepthCrossAttentionAdapter(nn.Module):
    """
    More powerful adapter using cross-attention between depth and SAM3 features.
    Depth features query SAM3 features to find depth-relevant regions.
    """

    def __init__(self, feature_dim=256, num_heads=4, attn_size=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.attn_size = attn_size

        # Depth projection
        self.depth_proj = nn.Sequential(
            nn.Conv2d(1, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
        )

        # Cross-attention: depth attends to SAM features
        self.cross_attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=0.1, batch_first=True
        )

        # Layer norms
        self.norm_depth = nn.LayerNorm(feature_dim)
        self.norm_sam = nn.LayerNorm(feature_dim)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
        )

    def forward(self, sam_features, depth):
        """
        Args:
            sam_features: [B, C, H, W]
            depth: [B, 1, H', W']
        """
        B, C, H, W = sam_features.shape

        # Resize depth
        if depth.shape[2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)

        # Project depth
        depth_feat = self.depth_proj(depth)  # [B, C, H, W]

        # Downsample for attention
        depth_small = F.adaptive_avg_pool2d(depth_feat, (self.attn_size, self.attn_size))
        sam_small = F.adaptive_avg_pool2d(sam_features, (self.attn_size, self.attn_size))

        # Flatten for attention: [B, h*w, C]
        depth_seq = depth_small.flatten(2).permute(0, 2, 1)
        sam_seq = sam_small.flatten(2).permute(0, 2, 1)

        # Cross attention with pre-norm
        depth_normed = self.norm_depth(depth_seq)
        sam_normed = self.norm_sam(sam_seq)

        attn_out, _ = self.cross_attn(depth_normed, sam_normed, sam_normed)

        # Residual
        depth_seq = depth_seq + attn_out

        # Reshape and upsample
        depth_out = depth_seq.permute(0, 2, 1).reshape(B, C, self.attn_size, self.attn_size)
        depth_out = F.interpolate(depth_out, size=(H, W), mode='bilinear', align_corners=False)

        # Project and add as residual to SAM features
        depth_out = self.out_proj(depth_out)

        return sam_features + depth_out


class PromptCrossAttention(nn.Module):
    """
    Cross-attention between prompts (text + optional bbox) and visual features.

    Much stronger than FiLM! Uses actual attention to focus on relevant regions.

    Supports:
    - Text prompts: global conditioning
    - BBox prompts: spatial attention bias
    - Point prompts: sparse spatial attention
    """

    def __init__(
        self,
        feature_dim=256,
        text_dim=1024,
        hidden_dim=256,
        num_heads=8,
        use_bbox=True,
        use_points=False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_bbox = use_bbox
        self.use_points = use_points

        # Text projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # BBox embedding (normalized coords -> embedding)
        if use_bbox:
            self.bbox_embed = nn.Sequential(
                nn.Linear(4, hidden_dim),  # [x1, y1, x2, y2] normalized
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Point embedding
        if use_points:
            self.point_embed = nn.Sequential(
                nn.Linear(2, hidden_dim),  # [x, y] normalized
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Cross-attention: visual features attend to prompt embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        # Project features to/from attention space
        self.feat_proj_in = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.feat_proj_out = nn.Conv2d(hidden_dim, feature_dim, 1)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Initialize output projection near identity
        nn.init.zeros_(self.feat_proj_out.weight)
        nn.init.zeros_(self.feat_proj_out.bias)

    def forward(
        self,
        features,
        text_embedding,
        bbox=None,
        points=None,
        num_views=None,
    ):
        """
        Args:
            features: [B*N, C, H, W] visual features
            text_embedding: [B, D] text embedding
            bbox: [B*N, 4] normalized bbox [x1, y1, x2, y2] or None
            points: [B*N, P, 2] normalized points or None
            num_views: N for expanding text to match features

        Returns:
            [B*N, C, H, W] prompt-conditioned features
        """
        BN, C, H, W = features.shape
        B_text = text_embedding.shape[0]

        # Expand text embedding to match views
        if BN != B_text:
            if num_views is not None and BN == B_text * num_views:
                text_embedding = text_embedding.repeat_interleave(num_views, dim=0)
            elif B_text < BN and BN % B_text == 0:
                repeat_factor = BN // B_text
                text_embedding = text_embedding.repeat_interleave(repeat_factor, dim=0)

        # Build prompt tokens
        prompt_tokens = []

        # Text token
        text_token = self.text_proj(text_embedding)  # [BN, hidden]
        prompt_tokens.append(text_token.unsqueeze(1))  # [BN, 1, hidden]

        # BBox token (if provided)
        if self.use_bbox and bbox is not None:
            bbox_token = self.bbox_embed(bbox)  # [BN, hidden]
            prompt_tokens.append(bbox_token.unsqueeze(1))  # [BN, 1, hidden]

        # Point tokens (if provided)
        if self.use_points and points is not None:
            # points: [BN, P, 2]
            point_tokens = self.point_embed(points)  # [BN, P, hidden]
            prompt_tokens.append(point_tokens)

        # Concatenate all prompt tokens: [BN, num_tokens, hidden]
        prompt_tokens = torch.cat(prompt_tokens, dim=1)

        # Project features to attention space
        feat_hidden = self.feat_proj_in(features)  # [BN, hidden, H, W]
        feat_flat = feat_hidden.flatten(2).permute(0, 2, 1)  # [BN, H*W, hidden]

        # Cross-attention: features (query) attend to prompts (key/value)
        attn_out, _ = self.cross_attn(
            query=feat_flat,
            key=prompt_tokens,
            value=prompt_tokens,
        )  # [BN, H*W, hidden]

        # Residual + norm
        attn_out = self.norm(feat_flat + attn_out)

        # Reshape back to spatial
        attn_out = attn_out.permute(0, 2, 1).view(BN, self.hidden_dim, H, W)

        # Project back to feature space (residual)
        out = features + self.feat_proj_out(attn_out)

        return out


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning"""

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # x: [B, in_features] or [B, C, H, W]
        original_shape = x.shape

        if len(original_shape) == 4:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            out = (x_flat @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        else:
            return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class SimpleDA3SAM3(nn.Module):
    """
    Simplified DA3-SAM3 model for testing
    Supports multiple fusion strategies: 'concat', 'cross_attention', 'gated'

    Architecture options:
    - Late fusion (default): SAM3 features + DA3 depth -> fusion head
    - Early fusion: Depth conditions SAM3 features BEFORE fusion head
    - Cross-view attention: Views communicate in multiview setting
    - Text conditioning: Text prompts condition SAM3 features for object-specific segmentation

    Use cases:
    - Single-view: forward() - late or early fusion
    - Multi-view: forward_multiview() - adds cross-view attention option
    """

    FUSION_TYPES = ['concat', 'cross_attention', 'gated', 'gasa_decoder', 'world_pe_only',
                    'lightweight', 'deformable', 'cost_volume',
                    'gasa_lightweight', 'gasa_deformable', 'gasa_cost_volume']
    EARLY_FUSION_TYPES = [None, 'cross_attention']  # FiLM removed - too weak

    def __init__(
        self,
        da3_model_name="depth-anything/DA3METRIC-LARGE",
        freeze_encoders=True,
        unfreeze_decoder=False,  # Unfreeze SAM3 mask decoder
        fusion_type='concat',  # 'concat', 'cross_attention', or 'gated'
        fusion_hidden_dim=256,
        fusion_num_layers=4,
        fusion_num_heads=4,  # For cross-attention fusion head
        # Early fusion options
        early_fusion=None,  # None or 'cross_attention' (FiLM removed - too weak)
        early_fusion_hidden=64,  # Hidden dim for early fusion adapter
        # Cross-view attention options
        use_cross_view_attention=False,  # Enable cross-view attention for multiview
        cross_view_layers=2,  # Number of cross-view transformer layers
        cross_view_heads=4,  # Number of attention heads
        use_depth_pe=False,  # Use per-view depth positional embeddings 
        use_world_pe=False,  # Use proper World-Space PE (requires camera params)
        world_pe_max_coord=10.0,  # Max coordinate for world-space PE normalization
        # GASA options (per RESEARCH_STRATEGY.md - our core contribution)
        use_gasa=False,  # Use Geometry-Aware Semantic Attention
        gasa_layers=2,  # Number of GASA layers
        gasa_heads=8,  # Number of GASA attention heads
        gasa_temperature=0.1,  # Distance kernel temperature (smaller = sharper geometric filtering)
        gasa_attn_size=16,  # Attention resolution (16x16 default for memory efficiency)
        # View chunking for memory efficiency
        encoder_chunk_size=None,  # Process views through encoders in chunks (None = all at once)
        view_chunk_size=None,  # Process views in GASA attention in chunks (None = all at once)
        # Text conditioning options
        # IMPORTANT: SAM3's native text encoder handles text→image conditioning internally.
        # These options are for ADDITIONAL conditioning on top of SAM3's backbone features.
        # For best results, use SAM3's full forward_grounding() pipeline instead.
        text_conditioning='none',  # 'none' or 'cross_attention' (FiLM removed)
        text_embed_dim=1024,  # SAM3 text embedding dimension (1024 from CLIP-style encoder)
        # LoRA options
        use_lora=False,
        lora_sam3=True,   # Apply LoRA to SAM3
        lora_da3=False,   # Apply LoRA to DA3
        lora_rank=8,
        lora_alpha=16,
        # Memory optimization options
        sam3_backbone_only=False,  # Only load SAM3 backbone (saves ~32M params)
        # DA3 multi-view options
        da3_multiview=False,  # Use DA3's multi-view mode for consistent depth across views
    ):
        super().__init__()
        self.use_lora = use_lora
        self.fusion_type = fusion_type
        self.early_fusion = early_fusion
        self.use_cross_view_attention = use_cross_view_attention
        self.use_depth_pe = use_depth_pe
        self.use_world_pe = use_world_pe
        self.use_gasa = use_gasa
        self.gasa_attn_size = gasa_attn_size
        self.encoder_chunk_size = encoder_chunk_size
        self.view_chunk_size = view_chunk_size
        self.text_conditioning_type = text_conditioning  # 'none' or 'cross_attention'

        if fusion_type not in self.FUSION_TYPES:
            raise ValueError(f"fusion_type must be one of {self.FUSION_TYPES}, got {fusion_type}")
        if early_fusion not in self.EARLY_FUSION_TYPES:
            raise ValueError(f"early_fusion must be one of {self.EARLY_FUSION_TYPES}, got {early_fusion}")

        # Import here to avoid circular imports
        from depth_anything_3.api import DepthAnything3
        from sam3 import build_sam3_image_model

        # Store options
        self.da3_multiview = da3_multiview
        self.sam3_backbone_only = sam3_backbone_only

        # Load pretrained models
        print(f"Loading DA3 ({da3_model_name})...")
        self.da3 = DepthAnything3.from_pretrained(da3_model_name)

        # Check if DA3 supports multi-view
        if da3_multiview:
            alt_start = self.da3.config.get("net", {}).get("alt_start", -1)
            if alt_start == -1:
                print(f"  WARNING: {da3_model_name} has alt_start=-1, NO multi-view support!")
                print(f"  For multi-view, use: DA3-LARGE, DA3-GIANT, or DA3NESTED-GIANT-LARGE")
                print(f"  Falling back to per-view depth inference.")
                self.da3_multiview = False
            else:
                print(f"  Multi-view enabled (global attention starts at block {alt_start})")

        print("Loading SAM3...")
        if sam3_backbone_only:
            # Load full model then extract backbone (saves ~32M params in forward pass)
            full_sam3 = build_sam3_image_model(bpe_path=_BPE_PATH)
            self.sam3_backbone = full_sam3.backbone
            self.sam3 = None  # Don't keep the full model
            print(f"  Backbone-only mode: {sum(p.numel() for p in self.sam3_backbone.parameters())/1e6:.1f}M params")
        else:
            self.sam3 = build_sam3_image_model(bpe_path=_BPE_PATH)
            self.sam3_backbone = self.sam3.backbone  # Alias for compatibility

        # Select fusion head based on type
        print(f"Using fusion type: {fusion_type}")
        if fusion_type == 'concat':
            self.fusion_head = SimpleFusionHead(
                sam_channels=256,
                hidden_dim=fusion_hidden_dim,
                num_layers=fusion_num_layers
            )
        elif fusion_type == 'cross_attention':
            self.fusion_head = CrossAttentionFusionHead(
                sam_channels=256,
                hidden_dim=fusion_hidden_dim,
                num_heads=fusion_num_heads,
                num_layers=fusion_num_layers
            )
        elif fusion_type == 'gated':
            self.fusion_head = GatedFusionHead(
                sam_channels=256,
                hidden_dim=fusion_hidden_dim,
                num_layers=fusion_num_layers
            )
        elif fusion_type == 'gasa_decoder':
            # GASA as decoder: simple mask head, GASA does the heavy lifting
            print("  -> GASA as decoder mode: simple mask head after GASA")
            self.fusion_head = GASADecoderHead(
                feature_dim=256,
                hidden_dim=fusion_hidden_dim
            )
            # Force GASA on when using gasa_decoder
            if not use_gasa:
                print("  -> Auto-enabling GASA for gasa_decoder fusion type")
                use_gasa = True
                self.use_gasa = True
        elif fusion_type == 'world_pe_only':
            # MV-SAM style: World-Space PE, no cross-view attention
            print("  -> MV-SAM style: World-Space PE only, no explicit cross-view attention")
            self.fusion_head = WorldPEOnlyHead(
                feature_dim=256,
                hidden_dim=fusion_hidden_dim,
                num_layers=fusion_num_layers
            )
            # Disable GASA and cross-view attention for this mode
            use_gasa = False
            self.use_gasa = False
            use_cross_view_attention = False
            self.use_cross_view_attention = False
        elif fusion_type == 'lightweight':
            # Lightweight cross-view: ~100K params, pool -> MLP -> weighted sum
            print("  -> Lightweight cross-view fusion (~100K params)")
            self.fusion_head = LightweightCrossViewHead(
                feature_dim=256,
                hidden_dim=fusion_hidden_dim,
                max_views=16
            )
            # Disable GASA - lightweight handles cross-view itself
            use_gasa = False
            self.use_gasa = False
        elif fusion_type == 'deformable':
            # Deformable attention: ~500K params, sparse attention to K points
            print("  -> Deformable cross-view attention (~500K params)")
            self.fusion_head = DeformableCrossViewHead(
                feature_dim=256,
                hidden_dim=fusion_hidden_dim,
                num_points=8,
                num_heads=4
            )
            # Disable GASA - deformable handles cross-view itself
            use_gasa = False
            self.use_gasa = False
        elif fusion_type == 'cost_volume':
            # Cost volume: ~300K params, correlation-based matching
            print("  -> Cost volume fusion (~300K params)")
            self.fusion_head = CostVolumeHead(
                feature_dim=256,
                hidden_dim=64,
                corr_radius=4
            )
            # Disable GASA - cost volume handles cross-view itself
            use_gasa = False
            self.use_gasa = False
        # GASA + Lightweight combos: GASA for cross-view, lightweight head for mask
        elif fusion_type == 'gasa_lightweight':
            print("  -> GASA + Lightweight: GASA cross-view + lightweight mask head")
            self.fusion_head = LightweightCrossViewHead(
                feature_dim=256,
                hidden_dim=fusion_hidden_dim,
                max_views=16
            )
            # KEEP GASA enabled - this combo uses GASA for cross-view
            if not use_gasa:
                print("  -> Auto-enabling GASA for gasa_lightweight fusion type")
                use_gasa = True
                self.use_gasa = True
        elif fusion_type == 'gasa_deformable':
            print("  -> GASA + Deformable: GASA cross-view + deformable mask head")
            self.fusion_head = DeformableCrossViewHead(
                feature_dim=256,
                hidden_dim=fusion_hidden_dim,
                num_points=8,
                num_heads=4
            )
            # KEEP GASA enabled
            if not use_gasa:
                print("  -> Auto-enabling GASA for gasa_deformable fusion type")
                use_gasa = True
                self.use_gasa = True
        elif fusion_type == 'gasa_cost_volume':
            print("  -> GASA + Cost Volume: GASA cross-view + cost volume mask head")
            self.fusion_head = CostVolumeHead(
                feature_dim=256,
                hidden_dim=64,
                corr_radius=4
            )
            # KEEP GASA enabled
            if not use_gasa:
                print("  -> Auto-enabling GASA for gasa_cost_volume fusion type")
                use_gasa = True
                self.use_gasa = True

        # Initialize early fusion adapter (depth conditions SAM features before fusion)
        self.depth_adapter = None
        if early_fusion == 'cross_attention':
            print("Using early fusion: Cross-attention adapter")
            self.depth_adapter = DepthCrossAttentionAdapter(
                feature_dim=256,
                num_heads=fusion_num_heads
            )

        # Initialize cross-view attention for multiview processing
        self.cross_view_attn = None
        self.gasa_encoder = None
        self.pointmap_computer = None

        if use_gasa:
            # GASA: Our core contribution - geometry-aware semantic attention
            print(f"Using GASA (Geometry-Aware Semantic Attention): {gasa_layers} layers, {gasa_heads} heads")
            print(f"  Temperature: {gasa_temperature} (geometric distance kernel)")
            print(f"  (Requires camera intrinsics/extrinsics at forward time)")
            self.gasa_encoder = GASAEncoder(
                d_model=256,
                num_layers=gasa_layers,
                num_heads=gasa_heads,
                ffn_dim=1024,
                dropout=0.1,
                temperature=gasa_temperature,
                use_world_pe=True  # Always use World-Space PE with GASA
            )
            self.pointmap_computer = PointmapComputer()
        elif use_cross_view_attention:
            if use_world_pe:
                # World-Space PE: same 3D point -> same PE across views (per RESEARCH_STRATEGY.md)
                print(f"Using World-Space PE cross-view attention: {cross_view_layers} layers, {cross_view_heads} heads")
                print(f"  (Requires camera intrinsics/extrinsics at forward time)")
                self.cross_view_attn = WorldSpaceCrossViewAttention(
                    feature_dim=256,
                    num_heads=cross_view_heads,
                    num_layers=cross_view_layers,
                    max_coord=world_pe_max_coord
                )
            elif use_depth_pe:
                print(f"Using depth PE cross-view attention: {cross_view_layers} layers, {cross_view_heads} heads")
                self.cross_view_attn = CrossViewAttention3D(
                    feature_dim=256,
                    num_heads=cross_view_heads,
                    num_layers=cross_view_layers,
                    use_depth_pe=True
                )
            else:
                print(f"Using cross-view attention: {cross_view_layers} layers, {cross_view_heads} heads")
                self.cross_view_attn = CrossViewAttention(
                    feature_dim=256,
                    num_heads=cross_view_heads,
                    num_layers=cross_view_layers
                )

        # Initialize text conditioning module
        # NOTE: SAM3's native text encoder handles text→image conditioning internally.
        # FiLM was REMOVED (too weak). Use SAM3's forward_grounding() for best results.
        self.text_conditioning = None
        if text_conditioning == 'cross_attention':
            print("Using PromptCrossAttention for additional text conditioning")
            self.text_conditioning = PromptCrossAttention(
                feature_dim=256,
                text_dim=text_embed_dim,
                hidden_dim=256,
                num_heads=8,
                use_bbox=False,  # Can be enabled if bbox prompts are provided
                use_points=False,
            )
        elif text_conditioning == 'none':
            print("No additional text conditioning (SAM3 handles text internally)")
            self.text_conditioning = None
        else:
            raise ValueError(f"text_conditioning must be 'none' or 'cross_attention', got {text_conditioning}")

        # Freeze encoders if requested
        if freeze_encoders:
            for param in self.da3.parameters():
                param.requires_grad = False
            # Freeze SAM3 (backbone or full model)
            if sam3_backbone_only:
                for param in self.sam3_backbone.parameters():
                    param.requires_grad = False
            else:
                for param in self.sam3.parameters():
                    param.requires_grad = False

        # Optionally unfreeze SAM3 mask decoder (only if not backbone-only)
        if unfreeze_decoder and not sam3_backbone_only:
            decoder_params = 0
            for name, param in self.sam3.named_parameters():
                if 'mask_decoder' in name or 'decoder' in name:
                    param.requires_grad = True
                    decoder_params += param.numel()
            print(f"  Unfroze SAM3 decoder: {decoder_params/1e6:.2f}M params")

        # Add LoRA adapters
        self.lora_adapters = nn.ModuleDict()
        if use_lora:
            print(f"Adding LoRA (rank={lora_rank}, alpha={lora_alpha})...")
            if lora_sam3:
                self._add_lora_to_sam3(lora_rank, lora_alpha)
            if lora_da3:
                self._add_lora_to_da3(lora_rank, lora_alpha)

        self.training = False

    def _add_lora_to_sam3(self, rank, alpha):
        """Add LoRA adapters to SAM3 attention layers"""
        # Use backbone for backbone-only mode
        sam3_module = self.sam3_backbone if self.sam3_backbone_only else self.sam3

        # Add LoRA to the last few transformer blocks
        lora_count = 0
        for name, module in sam3_module.named_modules():
            if 'attn' in name and hasattr(module, 'qkv'):
                # Get dimensions from the qkv projection
                if hasattr(module.qkv, 'in_features'):
                    in_dim = module.qkv.in_features
                    out_dim = module.qkv.out_features
                    lora_name = name.replace('.', '_')
                    self.lora_adapters[lora_name] = LoRALayer(in_dim, out_dim, rank, alpha)
                    lora_count += 1

        print(f"  SAM3: Added {lora_count} LoRA adapters")

    def _add_lora_to_da3(self, rank, alpha):
        """Add LoRA adapters to DA3 attention layers"""
        lora_count = 0
        for name, module in self.da3.named_modules():
            # DA3/DINOv2 uses 'attn' modules with qkv projections
            if 'attn' in name and hasattr(module, 'qkv'):
                if hasattr(module.qkv, 'in_features'):
                    in_dim = module.qkv.in_features
                    out_dim = module.qkv.out_features
                    lora_name = f"da3_{name.replace('.', '_')}"
                    self.lora_adapters[lora_name] = LoRALayer(in_dim, out_dim, rank, alpha)
                    lora_count += 1
        print(f"  DA3: Added {lora_count} LoRA adapters")

    def extract_da3_depth(self, images):
        """Extract depth maps from DA3"""
        with torch.no_grad():
            # DA3 needs proper dimensions
            B, C, H, W = images.shape
            patch_size = 14
            new_H = (H // patch_size) * patch_size
            new_W = (W // patch_size) * patch_size

            if new_H != H or new_W != W:
                da3_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
            else:
                da3_images = images

            # DA3 inference
            da3_input = da3_images.unsqueeze(1)  # [B, 1, 3, H, W]
            da3_output = self.da3.forward(
                da3_input,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],  # Empty list instead of None
                infer_gs=False
            )

            if hasattr(da3_output, 'depth'):
                depth = da3_output.depth
            elif isinstance(da3_output, dict) and 'depth' in da3_output:
                depth = da3_output['depth']
            else:
                depth = da3_output

            # Ensure shape [B, 1, H, W]
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)

            return depth

    def extract_da3_depth_multiview(self, images):
        """
        Extract depth maps from DA3 using multi-view mode.

        Multi-view mode processes all views jointly for consistent depth.
        Also returns estimated camera poses and intrinsics.

        Args:
            images: [B, N, 3, H, W] multi-view images

        Returns:
            depth: [B, N, 1, H, W] depth maps
            da3_extrinsics: [B, N, 4, 4] estimated camera poses (if available)
            da3_intrinsics: [B, N, 3, 3] estimated intrinsics (if available)
        """
        B, N, C, H, W = images.shape

        with torch.no_grad():
            # DA3 multi-view expects list of images or stacked tensor
            # Reshape to list format for inference API
            images_list = [images[b] for b in range(B)]  # List of [N, 3, H, W]

            all_depths = []
            all_extrinsics = []
            all_intrinsics = []

            for batch_images in images_list:
                # Convert to list of [3, H, W] tensors
                view_list = [batch_images[v] for v in range(N)]

                # DA3 inference with multi-view
                prediction = self.da3.inference(
                    image=view_list,
                    process_res=504,  # Standard DA3 resolution
                )

                # Extract outputs
                depth = prediction.depth  # [N, H', W']
                if depth.dim() == 3:
                    depth = depth.unsqueeze(1)  # [N, 1, H', W']

                all_depths.append(depth)

                # Get camera parameters if available
                if hasattr(prediction, 'extrinsics') and prediction.extrinsics is not None:
                    all_extrinsics.append(prediction.extrinsics)  # [N, 4, 4]
                if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None:
                    all_intrinsics.append(prediction.intrinsics)  # [N, 3, 3]

            # Stack batch dimension
            depth = torch.stack(all_depths, dim=0)  # [B, N, 1, H', W']

            # Handle camera params
            da3_extrinsics = None
            da3_intrinsics = None
            if all_extrinsics:
                da3_extrinsics = torch.stack(all_extrinsics, dim=0)  # [B, N, 4, 4]
            if all_intrinsics:
                da3_intrinsics = torch.stack(all_intrinsics, dim=0)  # [B, N, 3, 3]

            return depth, da3_extrinsics, da3_intrinsics

    def extract_sam3_features(self, images, text_prompts=None):
        """
        Extract features from SAM3, optionally conditioned on text prompts.

        Args:
            images: [B, 3, H, W] input images
            text_prompts: Optional list of text prompts (one per image)
                         When provided, returns text-conditioned features

        Returns:
            sam3_features: [B, 256, H', W'] features from SAM3
            text_embeddings: [B, D] text embeddings (if text_prompts provided)
        """
        B, C, H, W = images.shape

        # SAM3 expects 1008x1008
        sam3_size = 1008
        if H != sam3_size or W != sam3_size:
            sam3_images = F.interpolate(images, size=(sam3_size, sam3_size), mode='bilinear', align_corners=False)
        else:
            sam3_images = images

        # Extract features (use sam3_backbone for backbone-only mode compatibility)
        with torch.no_grad():
            backbone_output = self.sam3_backbone.forward_image(sam3_images)
            fpn_features = backbone_output['backbone_fpn']
            sam3_features = fpn_features[-1]  # Use highest level features [B, 256, H', W']

            # Get text-conditioned features if prompts provided
            text_embeddings = None
            if text_prompts is not None and len(text_prompts) > 0:
                # SAM3's text encoder
                text_outputs = self.sam3_backbone.forward_text(
                    text_prompts, device=images.device
                )
                # Extract text embeddings for conditioning
                # SAM3 returns language_embeds with shape [tokens, num_prompts, D]
                # e.g., [32, 2, 1024] for 2 prompts
                if 'language_embeds' in text_outputs:
                    text_embeddings = text_outputs['language_embeds']  # [tokens, B, D]
                    # Transpose to [B, tokens, D] then pool over tokens to get [B, D]
                    if text_embeddings.dim() == 3:
                        # Shape is [tokens, num_prompts, D] -> [num_prompts, tokens, D]
                        text_embeddings = text_embeddings.permute(1, 0, 2)
                        # Pool over tokens: [num_prompts, D]
                        text_embeddings = text_embeddings.mean(dim=1)

        return sam3_features, text_embeddings

    def generate_sam3_masks(self, images, text_prompts):
        """
        Generate masks directly from SAM3 using text prompts.
        Used for pseudo-label generation when GT masks are not available.

        Args:
            images: [B, 3, H, W] input images
            text_prompts: List of text prompts

        Returns:
            masks: [B, 1, H, W] binary masks
        """
        B, C, H, W = images.shape

        # SAM3 expects 1008x1008
        sam3_size = 1008
        if H != sam3_size or W != sam3_size:
            sam3_images = F.interpolate(images, size=(sam3_size, sam3_size), mode='bilinear', align_corners=False)
        else:
            sam3_images = images

        with torch.no_grad():
            # Use SAM3's text-prompted segmentation
            # This calls the full SAM3 pipeline with text prompts
            try:
                masks = self.sam3.predict_with_text(sam3_images, text_prompts)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
            except (AttributeError, NotImplementedError):
                # Fallback: Use backbone features + simple projection
                # This is a rough approximation when full SAM3 API isn't available
                backbone_output = self.sam3.backbone.forward_image(sam3_images)
                fpn_features = backbone_output['backbone_fpn']
                features = fpn_features[-1]  # [B, 256, H', W']

                # Simple global pooling + threshold to get rough mask
                # This is a placeholder - ideally use full SAM3 API
                feature_mag = features.norm(dim=1, keepdim=True)
                feature_mag = F.interpolate(feature_mag, size=(H, W), mode='bilinear', align_corners=False)

                # Normalize and threshold
                masks = (feature_mag - feature_mag.min()) / (feature_mag.max() - feature_mag.min() + 1e-6)

        # Resize to original image size
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)

        return masks

    def forward(self, images, text_prompts=None, use_parallel=True, return_features=False):
        """
        Forward pass with optional parallel DA3/SAM3 processing

        Args:
            images: [B, 3, H, W] input images
            text_prompts: List of text prompts for object-specific segmentation
                         Text prompts required for object selection
            use_parallel: If True, run DA3 and SAM3 in parallel using CUDA streams
            return_features: If True, also return intermediate features (for multiview)

        Returns:
            dict with 'masks' and 'depth' keys
        """
        if use_parallel and images.is_cuda:
            # Parallel extraction using CUDA streams
            depth, sam3_features, text_embeddings = self._extract_parallel(images, text_prompts)
        else:
            # Sequential extraction (fallback for CPU or debugging)
            depth = self.extract_da3_depth(images)
            sam3_features, text_embeddings = self.extract_sam3_features(images, text_prompts)

        # Apply text conditioning if enabled
        # NOTE: By default (text_conditioning='none'), SAM3's backbone features are used directly.
        if self.text_conditioning is not None and text_embeddings is not None:
            sam3_features = self.text_conditioning(sam3_features, text_embeddings)

        # Apply early fusion if enabled (depth conditions SAM features)
        if self.depth_adapter is not None:
            sam3_features = self.depth_adapter(sam3_features, depth)

        # Apply fusion head
        outputs = self.fusion_head(sam3_features, depth)

        # Add depth to outputs
        outputs['depth'] = depth

        # Optionally return features for multiview processing
        if return_features:
            outputs['sam3_features'] = sam3_features

        return outputs

    def _extract_parallel(self, images, text_prompts=None):
        """Extract DA3 depth and SAM3 features in parallel using CUDA streams"""
        # Create streams
        stream_da3 = torch.cuda.Stream()
        stream_sam3 = torch.cuda.Stream()

        # Launch DA3 on its stream
        with torch.cuda.stream(stream_da3):
            depth = self.extract_da3_depth(images)

        # Launch SAM3 on its stream
        with torch.cuda.stream(stream_sam3):
            sam3_features, text_embeddings = self.extract_sam3_features(images, text_prompts)

        # Synchronize both streams
        torch.cuda.synchronize()

        return depth, sam3_features, text_embeddings

    def forward_multiview(self, images, text_prompts=None, intrinsics=None, extrinsics=None):
        """
        Forward pass for multiple views - batches all views together for efficiency

        Pipeline:
        1. Extract features from all views (DA3 + SAM3)
        2. Apply text conditioning (for prompted segmentation)
        3. Apply early fusion if enabled (depth adapter)
        4. Apply cross-view attention if enabled (views communicate)
        5. Apply fusion head to get masks

        Args:
            images: [B, N, 3, H, W] input images (B=batch, N=num_views)
                    or List of [B, 3, H, W] tensors
            text_prompts: List of text prompts (one per batch item)
                         Text prompts required for object selection
            intrinsics: [B, N, 3, 3] camera intrinsic matrices (for World-Space PE)
            extrinsics: [B, N, 4, 4] camera extrinsic matrices (for World-Space PE)

        Returns:
            dict with:
            - 'masks': [B, N, 1, H', W'] masks for each view
            - 'depth': [B, N, 1, H', W'] depth for each view
        """
        # Handle list input
        if isinstance(images, list):
            images = torch.stack(images, dim=1)

        B, N, C, H, W = images.shape

        # Reshape to batch all views together: [B*N, C, H, W]
        images_flat = images.view(B * N, C, H, W)

        # If no cross-view attention or GASA, just use standard forward
        if self.cross_view_attn is None and self.gasa_encoder is None:
            outputs = self.forward(images_flat, text_prompts, use_parallel=True)
        else:
            # Extract features first (parallel DA3/SAM3)
            # Pass original text_prompts (not expanded) - we'll handle view expansion in text_conditioning

            # Use encoder chunking if specified (for memory efficiency with many views)
            enc_chunk = self.encoder_chunk_size
            if enc_chunk is not None and N > enc_chunk:
                # Process views through encoders in chunks to reduce peak memory
                # This allows scaling to 20+ views on a single GPU
                depth_chunks = []
                sam3_chunks = []
                text_embeddings = None

                for start_idx in range(0, N, enc_chunk):
                    end_idx = min(start_idx + enc_chunk, N)
                    # Get chunk of images: [B, chunk_size, C, H, W] -> [B*chunk, C, H, W]
                    chunk_images = images[:, start_idx:end_idx].reshape(-1, C, H, W)

                    # Only get text embeddings on first chunk (they're shared across views)
                    chunk_prompts = text_prompts if start_idx == 0 else None

                    if chunk_images.is_cuda:
                        chunk_depth, chunk_sam3, chunk_text_emb = self._extract_parallel(chunk_images, chunk_prompts)
                    else:
                        chunk_depth = self.extract_da3_depth(chunk_images)
                        chunk_sam3, chunk_text_emb = self.extract_sam3_features(chunk_images, chunk_prompts)

                    # Move to CPU temporarily to free GPU memory, then back for concat
                    depth_chunks.append(chunk_depth)
                    sam3_chunks.append(chunk_sam3)
                    if chunk_text_emb is not None:
                        text_embeddings = chunk_text_emb

                    # Clear CUDA cache between chunks to release intermediate encoder activations
                    if chunk_images.is_cuda:
                        torch.cuda.empty_cache()

                # Concatenate all chunks: [B*N, ...]
                depth = torch.cat(depth_chunks, dim=0)
                sam3_features = torch.cat(sam3_chunks, dim=0)
            else:
                # Process all views at once (original behavior)
                if images_flat.is_cuda:
                    depth, sam3_features, text_embeddings = self._extract_parallel(images_flat, text_prompts)
                else:
                    depth = self.extract_da3_depth(images_flat)
                    sam3_features, text_embeddings = self.extract_sam3_features(images_flat, text_prompts)

            # Apply text conditioning if enabled
            # NOTE: By default (text_conditioning='none'), SAM3's backbone features are used directly.
            # SAM3's full forward_grounding() pipeline handles text conditioning internally.
            # These options are for ADDITIONAL conditioning on backbone features.
            # text_embeddings has [B] samples, sam3_features has [B*N] samples
            if self.text_conditioning is not None and text_embeddings is not None:
                # Both TextConditioningModule and PromptCrossAttention support num_views
                sam3_features = self.text_conditioning(sam3_features, text_embeddings, num_views=N)

            # Apply early fusion if enabled
            if self.depth_adapter is not None:
                sam3_features = self.depth_adapter(sam3_features, depth)

            # Apply GASA or cross-view attention
            if self.use_gasa and self.gasa_encoder is not None:
                # GASA: Geometry-Aware Semantic Attention
                # Requires camera intrinsics and extrinsics
                if intrinsics is None or extrinsics is None:
                    raise ValueError("GASA requires camera intrinsics and extrinsics. "
                                   "Pass intrinsics=[B,N,3,3] and extrinsics=[B,N,4,4] to forward_multiview.")

                # sam3_features: [B*N, D, H, W] -> [B, N, H, W, D]
                _, D, H_feat, W_feat = sam3_features.shape
                features_spatial = sam3_features.view(B, N, D, H_feat, W_feat)
                features_spatial = features_spatial.permute(0, 1, 3, 4, 2)  # [B, N, H, W, D]

                # Compute pointmaps from depth + camera params
                depth_reshaped = depth.view(B, N, 1, depth.shape[-2], depth.shape[-1])

                # Resize depth to feature resolution if needed
                if depth.shape[-2:] != (H_feat, W_feat):
                    depth_for_points = F.interpolate(
                        depth.view(B * N, 1, depth.shape[-2], depth.shape[-1]),
                        size=(H_feat, W_feat),
                        mode='bilinear',
                        align_corners=False
                    )
                    depth_for_points = depth_for_points.view(B, N, H_feat, W_feat)
                else:
                    depth_for_points = depth_reshaped.squeeze(2)

                # Compute pointmaps: [B, N, H, W, 3]
                pointmaps, _ = self.pointmap_computer(
                    depth_for_points,
                    extrinsics,  # [B, N, 4, 4] camera-to-world
                    intrinsics,  # [B, N, 3, 3]
                    normalize=True
                )

                # Apply GASA encoder with optional view chunking for memory efficiency
                chunk_size = self.view_chunk_size
                if chunk_size is not None and N > chunk_size:
                    # Process views in chunks to reduce memory
                    features_chunks = []
                    for start_idx in range(0, N, chunk_size):
                        end_idx = min(start_idx + chunk_size, N)
                        feat_chunk = features_spatial[:, start_idx:end_idx]  # [B, chunk, H, W, D]
                        pts_chunk = pointmaps[:, start_idx:end_idx]  # [B, chunk, H, W, 3]
                        out_chunk = self.gasa_encoder(feat_chunk, pts_chunk)
                        features_chunks.append(out_chunk)
                    features_gasa = torch.cat(features_chunks, dim=1)  # [B, N, H, W, D]
                else:
                    # Process all views together
                    features_gasa = self.gasa_encoder(features_spatial, pointmaps)  # [B, N, H, W, D]

                # Reshape back to [B*N, D, H, W] for fusion head
                sam3_features = features_gasa.permute(0, 1, 4, 2, 3).reshape(B * N, D, H_feat, W_feat)

            elif self.cross_view_attn is not None:
                # Standard cross-view attention
                if self.use_world_pe:
                    # World-Space PE: pass camera params for proper 3D encoding
                    sam3_features = self.cross_view_attn(
                        sam3_features, num_views=N, depth=depth,
                        intrinsics=intrinsics, extrinsics=extrinsics
                    )
                elif self.use_depth_pe:
                    # Depth PE: pass depth for per-view depth encoding
                    sam3_features = self.cross_view_attn(sam3_features, num_views=N, depth=depth)
                else:
                    sam3_features = self.cross_view_attn(sam3_features, num_views=N)

            # Apply fusion head
            # Different fusion types need different arguments
            if self.fusion_type == 'world_pe_only' and intrinsics is not None and extrinsics is not None:
                # WorldPEOnlyHead needs camera params for World-Space PE
                intrinsics_flat = intrinsics.view(B * N, 3, 3)
                extrinsics_flat = extrinsics.view(B * N, 4, 4)
                outputs = self.fusion_head(sam3_features, depth, intrinsics_flat, extrinsics_flat)
            elif self.fusion_type in ['lightweight', 'deformable', 'cost_volume',
                                       'gasa_lightweight', 'gasa_deformable', 'gasa_cost_volume']:
                # These fusion heads need num_views for cross-view processing
                outputs = self.fusion_head(sam3_features, depth, num_views=N)
            else:
                outputs = self.fusion_head(sam3_features, depth)
            outputs['depth'] = depth

        # Reshape outputs back to [B, N, ...]
        masks = outputs['masks']  # [B*N, 1, H', W']
        depth = outputs['depth']  # [B*N, 1, H', W']

        _, _, H_out, W_out = masks.shape
        masks = masks.view(B, N, 1, H_out, W_out)
        depth = depth.view(B, N, 1, depth.shape[-2], depth.shape[-1])

        return {
            'masks': masks,
            'depth': depth,
            'presence_3d': outputs.get('presence_3d'),
            'depth_confidence': outputs.get('depth_confidence')
        }


class TrianguLangGASAModel(nn.Module):
    """
    TrianguLang: Geometry-Aware Semantic Consensus for Pose-Free 3D Localization

    This is our main model that implements:
    1. Frozen SAM3 for semantic features
    2. Frozen DA3 for depth + poses + intrinsics
    3. PointmapComputer: depth + geometry -> world coordinates
    4. WorldSpacePositionalEncoding: 3D coords -> high-dim embeddings
    5. GASA (Geometry-Aware Semantic Attention): cross-view fusion with geometric bias
    6. SymmetricCentroidHead: 3D localization output

    Key insight: Standard cross-attention hallucinates correspondences between
    semantically similar but spatially distant features. GASA uses DA3's predicted
    geometry to veto these false matches.
    """

    def __init__(
        self,
        da3_model_name: str = "depth-anything/DA3METRIC-LARGE",
        freeze_encoders: bool = True,
        # GASA parameters
        gasa_layers: int = 2,
        gasa_heads: int = 8,
        gasa_dim: int = 256,
        gasa_temperature: float = 0.1,
        use_world_pe: bool = True,
        # Fusion head parameters
        fusion_type: str = 'gated',
        fusion_hidden_dim: int = 256,
        fusion_num_layers: int = 3,
        # Output options
        predict_centroid: bool = True
    ):
        super().__init__()
        self.predict_centroid = predict_centroid
        self.gasa_dim = gasa_dim

        # Import dependencies
        from depth_anything_3.api import DepthAnything3
        from sam3 import build_sam3_image_model
        from .gasa import (
            PointmapComputer,
            WorldSpacePositionalEncoding,
            GASAEncoder,
            SymmetricCentroidHead
        )

        # Load frozen foundation models
        print("Loading DA3...")
        self.da3 = DepthAnything3.from_pretrained(da3_model_name)

        print("Loading SAM3...")
        self.sam3 = build_sam3_image_model(bpe_path=_BPE_PATH)

        # Freeze encoders
        if freeze_encoders:
            for param in self.da3.parameters():
                param.requires_grad = False
            for param in self.sam3.parameters():
                param.requires_grad = False

        # Geometry modules (our contribution)
        self.pointmap_computer = PointmapComputer()

        # Optional world-space PE (can also be part of GASA)
        self.use_world_pe = use_world_pe
        if use_world_pe:
            self.world_pe = WorldSpacePositionalEncoding(d_model=gasa_dim)

        # GASA encoder for cross-view fusion
        self.gasa_encoder = GASAEncoder(
            d_model=gasa_dim,
            num_layers=gasa_layers,
            num_heads=gasa_heads,
            ffn_dim=gasa_dim * 4,
            dropout=0.1,
            temperature=gasa_temperature,
            use_world_pe=False  # We handle PE separately for flexibility
        )

        # Feature projection (SAM3 -> GASA dim)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(256, gasa_dim, kernel_size=1),
            nn.BatchNorm2d(gasa_dim),
            nn.ReLU(inplace=True)
        )

        # Fusion head for mask prediction
        if fusion_type == 'gated':
            self.fusion_head = GatedFusionHead(
                sam_channels=gasa_dim,
                hidden_dim=fusion_hidden_dim,
                num_layers=fusion_num_layers
            )
        elif fusion_type == 'cross_attention':
            self.fusion_head = CrossAttentionFusionHead(
                sam_channels=gasa_dim,
                hidden_dim=fusion_hidden_dim,
                num_heads=4,
                num_layers=fusion_num_layers
            )
        else:
            self.fusion_head = SimpleFusionHead(
                sam_channels=gasa_dim,
                hidden_dim=fusion_hidden_dim,
                num_layers=fusion_num_layers
            )

        # Centroid prediction head
        if predict_centroid:
            self.centroid_head = SymmetricCentroidHead(d_model=gasa_dim)

        print(f"TrianguLang initialized with GASA ({gasa_layers} layers, {gasa_heads} heads)")

    def extract_da3_full(self, images: torch.Tensor):
        """
        Extract depth, poses, and intrinsics from DA3.

        Args:
            images: [B, N, 3, H, W] multi-view images

        Returns:
            depth: [B, N, H', W']
            poses: [B, N, 4, 4] camera-to-world transforms
            intrinsics: [B, N, 3, 3]
        """
        B, N, C, H, W = images.shape

        with torch.no_grad():
            # DA3 expects [B, N, 3, H, W] directly for multi-view
            patch_size = 14
            new_H = (H // patch_size) * patch_size
            new_W = (W // patch_size) * patch_size

            if new_H != H or new_W != W:
                images_resized = F.interpolate(
                    images.view(B * N, C, H, W),
                    size=(new_H, new_W),
                    mode='bilinear',
                    align_corners=False
                ).view(B, N, C, new_H, new_W)
            else:
                images_resized = images

            # DA3 multi-view inference
            da3_output = self.da3.forward(
                images_resized,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False
            )

            # Extract outputs
            if hasattr(da3_output, 'depth'):
                depth = da3_output.depth  # [B*N, H, W] or similar
            elif isinstance(da3_output, dict):
                depth = da3_output['depth']
            else:
                depth = da3_output

            # Get poses and intrinsics
            if hasattr(da3_output, 'extrinsics') and da3_output.extrinsics is not None:
                poses = torch.from_numpy(da3_output.extrinsics).to(images.device)
            else:
                # Default identity poses if not available
                poses = torch.eye(4, device=images.device).unsqueeze(0).unsqueeze(0)
                poses = poses.expand(B, N, -1, -1).clone()

            if hasattr(da3_output, 'intrinsics') and da3_output.intrinsics is not None:
                intrinsics = torch.from_numpy(da3_output.intrinsics).to(images.device)
            else:
                # Default intrinsics (approximate)
                fx = fy = new_W * 0.8
                cx, cy = new_W / 2, new_H / 2
                K = torch.tensor([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=torch.float32, device=images.device)
                intrinsics = K.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

            # Ensure proper shapes
            if depth.dim() == 3:
                depth = depth.view(B, N, depth.shape[-2], depth.shape[-1])
            elif depth.dim() == 4 and depth.shape[1] == 1:
                depth = depth.squeeze(1).view(B, N, depth.shape[-2], depth.shape[-1])

            if poses.dim() == 3:
                poses = poses.view(B, N, 4, 4)
            if intrinsics.dim() == 3:
                intrinsics = intrinsics.view(B, N, 3, 3)

        return depth, poses, intrinsics

    def extract_sam3_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract semantic features from SAM3."""
        B, N, C, H, W = images.shape

        # SAM3 expects 1008x1008
        sam3_size = 1008
        images_flat = images.view(B * N, C, H, W)

        if H != sam3_size or W != sam3_size:
            sam3_images = F.interpolate(
                images_flat,
                size=(sam3_size, sam3_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            sam3_images = images_flat

        with torch.no_grad():
            backbone_output = self.sam3.backbone.forward_image(sam3_images)
            fpn_features = backbone_output['backbone_fpn']
            features = fpn_features[-1]  # [B*N, 256, H', W']

        # Project to GASA dimension
        features = self.feature_proj(features)

        return features  # [B*N, gasa_dim, H', W']

    def forward(self, images: torch.Tensor, text_prompts=None):
        """
        Forward pass for multi-view 3D localization.

        Args:
            images: [B, N, 3, H, W] multi-view images
            text_prompts: Optional list of text prompts (for future use)

        Returns:
            dict with:
                - 'masks': [B, N, 1, H', W'] segmentation masks
                - 'depth': [B, N, H', W'] depth maps
                - 'pointmaps': [B, N, H', W', 3] world coordinates
                - 'centroid': [B, 3] predicted 3D centroid (if predict_centroid=True)
                - 'confidences': [B, N] per-view confidences
        """
        B, N, C, H, W = images.shape

        # 1. Extract DA3 outputs: depth, poses, intrinsics
        depth, poses, intrinsics = self.extract_da3_full(images)

        # 2. Compute world-space pointmaps
        _, _, H_d, W_d = depth.shape[0], depth.shape[1], depth.shape[2], depth.shape[3]
        pointmaps, _ = self.pointmap_computer(
            depth, poses, intrinsics, normalize=True
        )  # [B, N, H_d, W_d, 3]

        # 3. Extract SAM3 features
        sam3_features = self.extract_sam3_features(images)  # [B*N, D, H', W']

        # Reshape to [B, N, H', W', D]
        _, D, H_f, W_f = sam3_features.shape
        sam3_features = sam3_features.view(B, N, D, H_f, W_f)
        sam3_features = sam3_features.permute(0, 1, 3, 4, 2)  # [B, N, H', W', D]

        # Resize pointmaps to match feature resolution
        if (H_d, W_d) != (H_f, W_f):
            pointmaps_resized = F.interpolate(
                pointmaps.permute(0, 1, 4, 2, 3).reshape(B * N, 3, H_d, W_d),
                size=(H_f, W_f),
                mode='bilinear',
                align_corners=False
            ).view(B, N, 3, H_f, W_f).permute(0, 1, 3, 4, 2)
        else:
            pointmaps_resized = pointmaps

        # 4. Add world-space positional embeddings
        if self.use_world_pe:
            pe = self.world_pe(pointmaps_resized)  # [B, N, H', W', D]
            sam3_features = sam3_features + pe

        # 5. Apply GASA encoder (cross-view attention with geometric bias)
        fused_features = self.gasa_encoder(sam3_features, pointmaps_resized)  # [B, N, H', W', D]

        # 6. Apply fusion head for mask prediction
        # Reshape for fusion head: [B*N, D, H', W']
        fused_flat = fused_features.permute(0, 1, 4, 2, 3).reshape(B * N, D, H_f, W_f)
        depth_flat = depth.view(B * N, 1, H_d, W_d)

        fusion_output = self.fusion_head(fused_flat, depth_flat)
        masks = fusion_output['masks']  # [B*N, 1, H', W']

        # Reshape masks
        masks = masks.view(B, N, 1, masks.shape[-2], masks.shape[-1])

        # 7. Predict 3D centroid (if enabled)
        centroid = None
        confidences = None
        if self.predict_centroid:
            centroid, confidences = self.centroid_head(
                fused_features, masks, pointmaps_resized
            )

        return {
            'masks': masks,
            'depth': depth,
            'pointmaps': pointmaps,
            'centroid': centroid,
            'confidences': confidences,
            'poses': poses,
            'intrinsics': intrinsics
        }