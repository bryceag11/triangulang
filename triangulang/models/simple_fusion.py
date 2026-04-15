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
    Bidirectional cross-attention fusion: depth <-> SAM3 features

    Key insight: Both modalities must be preserved in the output.
    - Depth->SAM attention: depth learns what semantic regions are important
    - SAM->Depth attention: semantics learn where in 3D space to focus
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

        # Bidirectional cross-attention (depth->SAM and SAM->depth)
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


