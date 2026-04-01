"""Decoder layer components for the GASA Decoder.

Classes extracted from gasa_decoder.py:
- GASADecoderLayer: Single decoder layer with geometric attention bias
- MaskRefiner: Lightweight mask refinement for sharper boundaries
- SpatialAttentionBias: Depth-conditioned spatial attention bias
- TextConditionedSpatialBias: ViL3DRel-inspired text-conditioned spatial bias
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from triangulang.models.positional_encodings import (
    WorldSpacePositionalEncoding,
    CameraRelativePositionalEncoding,
    PluckerEmbedding,
    RayRoPE3D,
)
from triangulang.models.gasa import PointmapComputer  # noqa: F401


class GASADecoderLayer(nn.Module):
    """
    GASA Decoder Layer - Our core contribution

    Key innovation: Geometry-Aware cross-attention
        Attn(Q, K) = Softmax(QK^T / sqrt(d) + β * φ(||P_Q - P_K||))

    The geometric bias β * φ(distance) penalizes attending to
    semantically similar but spatially distant regions.

    With use_gasa=False, this becomes standard cross-attention (for ablation).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1, use_gasa: bool = True, gasa_beta_init: float = 1.0,
                 gasa_kernel_dim: int = 32, gasa_fixed_kernel: bool = False,
                 gasa_kernel_type: str = 'learned',
                 use_depth_crossattn: bool = False, per_layer_text: bool = False,
                 gasa_bidirectional: bool = False, post_norm: bool = True,
                 ffn_fp32: bool = True,
                 use_image_to_token: bool = False,
                 use_pos_refine: bool = False,
                 use_box_rpb: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_gasa = use_gasa  # Ablation: disable geometric bias
        self.gasa_fixed_kernel = gasa_fixed_kernel  # Ablation: use φ(d) = -d instead of MLP
        self.gasa_kernel_type = gasa_kernel_type  # 'learned', 'rbf', or 'fixed'
        self.gasa_bidirectional = gasa_bidirectional  # Bidirectional: boost nearby + suppress distant
        self.use_depth_crossattn = use_depth_crossattn  # Deep depth fusion: cross-attention to depth features
        self.per_layer_text = per_layer_text  # Per-layer text cross-attention (like SAM3)
        self.use_image_to_token = use_image_to_token  # SAM3-style: pixels attend back to queries
        self.use_pos_refine = use_pos_refine  # SAM3-style: predict position delta each layer
        self.use_box_rpb = use_box_rpb  # SAM3-style: box-relative position bias
        self.post_norm = post_norm  # Post-norm (SAM3-style) vs pre-norm ordering
        self.ffn_fp32 = ffn_fp32  # Run FFN in FP32 (SAM3 disables autocast in FFN)

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Per-layer text cross-attention (SAM3-style - text conditioning at EVERY layer)
        if per_layer_text:
            self.text_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.text_norm = nn.LayerNorm(d_model)

        # Cross-attention to encoder memory (with geometric bias if use_gasa=True)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Image-to-token cross-attention (SAM3 TwoWayTransformer step 4)
        # Pixels attend back to queries, making pixel features mask-aware
        if use_image_to_token:
            self.image_to_token_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True)
            self.image_to_token_norm = nn.LayerNorm(d_model)

        # 3D position refinement (like SAM3's bbox_embed but in 3D)
        # Each layer predicts a 3D position delta from query features
        if use_pos_refine:
            self.pos_refine_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 3),
            )
            # Init near-zero so positions start stable
            nn.init.zeros_(self.pos_refine_mlp[-1].weight)
            nn.init.zeros_(self.pos_refine_mlp[-1].bias)

        # 2D box-relative position bias (SAM3's boxRPB)
        # Each layer predicts a 2D box, computes per-pixel position relative to box edges,
        # maps through learned MLP → per-head attention bias
        if use_box_rpb:
            # Box prediction: query → (cx, cy, w, h) delta
            self.bbox_embed = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 4),
            )
            nn.init.zeros_(self.bbox_embed[-1].weight)
            nn.init.zeros_(self.bbox_embed[-1].bias)
            # Box-relative position → per-head attention bias (like SAM3's boxRPB_embed_x/y)
            self.box_rpb_embed_x = nn.Sequential(
                nn.Linear(2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, n_heads),
            )
            self.box_rpb_embed_y = nn.Sequential(
                nn.Linear(2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, n_heads),
            )

        # Learnable geometric bias strength (only used if use_gasa=True)
        # gasa_beta_init controls initial value for ablation experiments
        if use_gasa:
            self.beta = nn.Parameter(torch.tensor(gasa_beta_init))

            # Distance kernel: learned MLP, RBF, or fixed φ(d) = -d
            if gasa_kernel_type == 'rbf':
                # RBF kernel: φ(d) = -exp(d² / 2σ²) with learnable σ
                # Standard choice in spatial attention; serves as ablation baseline
                self.distance_kernel = None  # RBF computed analytically
                self.rbf_log_sigma = nn.Parameter(torch.tensor(0.0))  # σ = exp(0) = 1.0 meter
            elif gasa_fixed_kernel or gasa_kernel_type == 'fixed':
                # Fixed kernel: φ(d) = -d (no learnable parameters)
                self.distance_kernel = None
            else:
                # Learnable distance kernel with configurable hidden dim
                self.distance_kernel = nn.Sequential(
                    nn.Linear(1, gasa_kernel_dim),
                    nn.ReLU(),
                    nn.Linear(gasa_kernel_dim, 1),
                )
                nn.init.xavier_uniform_(self.distance_kernel[0].weight, gain=0.1)
                nn.init.zeros_(self.distance_kernel[0].bias)
                nn.init.xavier_uniform_(self.distance_kernel[2].weight, gain=0.1)
                if gasa_bidirectional:
                    nn.init.constant_(self.distance_kernel[2].bias, 2.0)  # Start positive (boost nearby)
                else:
                    nn.init.constant_(self.distance_kernel[2].bias, -1.0)  # Start negative (suppress-only)

        # FFN (ReLU to match SAM3's decoder; GELU was used in pre-norm mode)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Depth cross-attention: queries explicitly attend to depth/3D position features
        # This is DIFFERENT from GASA bias - it treats depth as a separate modality
        if use_depth_crossattn:
            self.depth_proj = nn.Linear(3, d_model)  # Project [x,y,z] to d_model
            self.depth_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.depth_norm = nn.LayerNorm(d_model)
            # Initialize depth_proj to preserve scale
            nn.init.xavier_uniform_(self.depth_proj.weight, gain=0.1)
            nn.init.zeros_(self.depth_proj.bias)

    def compute_geometric_bias(self, query_pos, key_pos):
        """Compute geometric attention bias based on 3D distances.

        If gasa_bidirectional: positive bias for nearby (boost), negative for distant (suppress).
        Otherwise: suppress-only (clamp max=0).
        """
        # Pairwise distances: [B, Q, L]
        distances = torch.cdist(query_pos, key_pos)

        if self.gasa_kernel_type == 'rbf' and hasattr(self, 'rbf_log_sigma'):
            sigma = torch.exp(self.rbf_log_sigma)  # Ensure σ > 0
            if self.gasa_bidirectional:
                max_boost = 5.0
                rbf = torch.exp(-(distances ** 2) / (2 * sigma ** 2 + 1e-6))
                bias = max_boost * rbf - 2.5  # Range: [-2.5, +2.5]
            else:
                bias = -torch.exp(-(distances ** 2) / (2 * sigma ** 2 + 1e-6))
        elif self.gasa_fixed_kernel or self.distance_kernel is None:
            if self.gasa_bidirectional:
                bias = 5.0 - distances
            else:
                bias = -distances
        else:
            # Learned kernel
            dist_flat = distances.reshape(-1, 1)
            bias_flat = self.distance_kernel(dist_flat)
            bias = bias_flat.reshape(distances.shape)

        # Clamp to prevent extreme values
        clamp_max = 10.0 if self.gasa_bidirectional else 0.0
        return torch.clamp(bias, min=-10, max=clamp_max)

    def forward(self, queries, memory, memory_pos, query_pos=None, return_query_pos=False,
                text_embedding=None, rayrope_ctx=None, query_pe=None, memory_v=None,
                memory_pe=None, text_attn_mask=None, spatial_attn_bias=None,
                reference_boxes=None, spatial_hw=None):
        """Forward pass with geometry-aware attention.

        Args:
            queries: [B, Q, D] query features
            memory: [B, L, D] encoder memory (may include PE if not using additive_pe)
            memory_pos: [B, L, 3] 3D positions of memory tokens
            query_pos: [B, Q, 3] 3D positions of queries (optional, uses scene centroid if None)
            reference_boxes: [B, Q, 4] predicted boxes (cx, cy, w, h) normalized [0,1] for boxRPB
            spatial_hw: (H, W) tuple — spatial dims of memory for boxRPB coordinate grid
            return_query_pos: If True, also return attention-weighted query positions for next layer
            text_embedding: [B, T, D] text embeddings for per-layer conditioning (optional)
            rayrope_ctx: dict with 'rayrope', 'w2c', 'intrinsics', 'depth_conf' (optional)
                         When provided, applies rotary PE to Q/K instead of additive PE.
            query_pe: [B, Q, D] learned positional embedding for queries (optional, SAM3-style)
                      Added to Q and K in self-attention, and Q in cross-attention.
            memory_v: [B, L, D] raw memory WITHOUT PE for V projection (optional, --clean-v).
                      SAM3 uses V=memory (no PE). When None, V uses same memory as K.
            memory_pe: [B, L, D] world PE to add to K at attention time (--additive-pe).
                       SAM3-style: K = memory + pe, V = memory (pe-free). When None, falls
                       PE baked into memory.

        Returns:
            queries: [B, Q, D] updated query features
            query_pos_out: [B, Q, 3] attention-weighted positions (only if return_query_pos=True)
        """
        B, Q, D = queries.shape
        L = memory.shape[1]

        # 1. Self-attention among queries
        # SAM3 adds query PE to both Q and K: q = k = with_pos_embed(tgt, query_pos)
        if self.post_norm:
            # SAM3-style: attend first, then norm after residual
            sa_q = queries + query_pe if query_pe is not None else queries
            sa_k = sa_q  # Same as SAM3: q = k = with_pos_embed(tgt, tgt_query_pos)
            q2, _ = self.self_attn(sa_q, sa_k, queries)  # V = queries (no PE on V)
            queries = self.norm1(queries + self.dropout(q2))
        else:
            # Pre-norm: norm first, then attend
            q = self.norm1(queries)
            sa_q = q + query_pe if query_pe is not None else q
            sa_k = sa_q
            q2, _ = self.self_attn(sa_q, sa_k, q)  # V = q (no PE on V)
            queries = queries + self.dropout(q2)

        # 1.5 Per-layer text cross-attention (SAM3-style - queries attend to text at EVERY layer)
        # SAM3 adds query PE to Q here too: query=with_pos_embed(tgt, tgt_query_pos)
        # text_attn_mask: optional [Q, K*T] bool mask for grouped text attention
        if self.per_layer_text and text_embedding is not None:
            text_q = queries + query_pe if query_pe is not None else queries
            text_cond, _ = self.text_cross_attn(text_q, text_embedding, text_embedding,
                                                 attn_mask=text_attn_mask)
            queries = self.text_norm(queries + text_cond)

        # 2. Cross-attention to memory with geometric bias
        # SAM3 adds query PE to Q: query=with_pos_embed(tgt, tgt_query_pos)
        if self.post_norm:
            q = queries + query_pe if query_pe is not None else queries
        else:
            q = self.norm2(queries)
            q = q + query_pe if query_pe is not None else q

        Q_proj = self.q_proj(q).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        # --additive-pe (SAM3-style): K = k_proj(memory) + k_proj(pe), V = v_proj(memory)
        # Legacy: memory already has PE baked in, so K and V both see PE unless --clean-v
        if memory_pe is not None:
            # SAM3-style: add PE to K at attention time, V stays PE-free
            K_proj = self.k_proj(memory + memory_pe).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            V_proj = self.v_proj(memory).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            K_proj = self.k_proj(memory).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            # Legacy: --clean-v passes raw memory separately for V
            v_input = memory_v if memory_v is not None else memory
            V_proj = self.v_proj(v_input).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Check for cross-view RayRoPE mode (per-camera attention loop)
        use_multiview_rayrope = (rayrope_ctx is not None and rayrope_ctx.get('num_cameras', 1) > 1)

        if use_multiview_rayrope:
            # Cross-view RayRoPE: per-camera attention with averaged outputs
            # This handles Q/K rotation, attention, GASA bias, softmax, and V multiplication
            rayrope = rayrope_ctx['rayrope']
            _qpos = query_pos if query_pos is not None else memory_pos.mean(dim=1, keepdim=True).expand(-1, Q, -1)

            # Compute GASA geometric bias (frame-invariant, shared across camera frames)
            geo_bias = None
            if self.use_gasa:
                if query_pos is not None:
                    geo_bias_raw = self.compute_geometric_bias(query_pos, memory_pos)
                else:
                    query_pos_proxy = memory_pos.mean(dim=1, keepdim=True).expand(-1, Q, -1)
                    geo_bias_raw = self.compute_geometric_bias(query_pos_proxy, memory_pos)
                geo_bias = self.beta * geo_bias_raw.unsqueeze(1)  # [B, 1, Q, L]

            out, attn_probs = rayrope.forward_multiview(
                Q_proj, K_proj, V_proj,
                memory_pos=memory_pos, query_pos=_qpos,
                w2c_per_view=rayrope_ctx['w2c_per_view'],
                intrinsics_per_view=rayrope_ctx['intrinsics_per_view'],
                num_cameras=rayrope_ctx['num_cameras'],
                scale=self.scale, attn_bias=geo_bias,
                depth_conf=rayrope_ctx.get('depth_conf'),
            )
            out = out.transpose(1, 2).reshape(B, Q, D)
            out = self.out_proj(out)
            queries = queries + self.dropout(out)
        else:
            # Standard path: single-view RayRoPE or no RayRoPE
            if rayrope_ctx is not None:
                rayrope = rayrope_ctx['rayrope']
                _qpos = query_pos if query_pos is not None else memory_pos.mean(dim=1, keepdim=True).expand(-1, Q, -1)
                Q_proj, K_proj = rayrope(
                    Q_proj, K_proj,
                    memory_pos=memory_pos,
                    query_pos=_qpos,
                    w2c=rayrope_ctx.get('w2c'),
                    intrinsics=rayrope_ctx.get('intrinsics'),
                    depth_conf=rayrope_ctx.get('depth_conf'),
                )

            # Semantic attention
            attn_semantic = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) * self.scale

            # Geometric bias (only if use_gasa=True - our core contribution)
            if self.use_gasa:
                if query_pos is not None:
                    geo_bias = self.compute_geometric_bias(query_pos, memory_pos)
                else:
                    query_pos_proxy = memory_pos.mean(dim=1, keepdim=True).expand(-1, Q, -1)
                    geo_bias = self.compute_geometric_bias(query_pos_proxy, memory_pos)

                # Combined attention: semantic + geometric
                geo_bias = geo_bias.unsqueeze(1)  # [B, 1, Q, L]
                attn_scores = attn_semantic + self.beta * geo_bias
            else:
                # Standard attention (no geometric bias) - for ablation
                attn_scores = attn_semantic

            # Add spatial attention bias (depth-conditioned, per-pixel weighting)
            if spatial_attn_bias is not None:
                attn_scores = attn_scores + spatial_attn_bias  # [B, n_heads, Q, L]

            # 2D box-relative position bias (SAM3-style boxRPB)
            if self.use_box_rpb and reference_boxes is not None and spatial_hw is not None:
                H_s, W_s = spatial_hw
                # Compute box-relative pixel positions
                # reference_boxes: [B, Q, 4] as (cx, cy, w, h) in [0, 1]
                boxes_xyxy = torch.zeros_like(reference_boxes)
                boxes_xyxy[..., 0] = reference_boxes[..., 0] - reference_boxes[..., 2] / 2  # x1
                boxes_xyxy[..., 1] = reference_boxes[..., 1] - reference_boxes[..., 3] / 2  # y1
                boxes_xyxy[..., 2] = reference_boxes[..., 0] + reference_boxes[..., 2] / 2  # x2
                boxes_xyxy[..., 3] = reference_boxes[..., 1] + reference_boxes[..., 3] / 2  # y2

                # Pixel grid [0, 1]
                coords_y = torch.linspace(0, 1, H_s, device=queries.device)
                coords_x = torch.linspace(0, 1, W_s, device=queries.device)

                # Relative positions: pixel coord minus box edges → [B, Q, H or W, 2]
                # delta_x[:,:,:,0] = pixel_x - x1, delta_x[:,:,:,1] = pixel_x - x2
                delta_x = coords_x.view(1, 1, -1, 1) - boxes_xyxy[:, :, None, 0::2]  # [B, Q, W, 2]
                delta_y = coords_y.view(1, 1, -1, 1) - boxes_xyxy[:, :, None, 1::2]  # [B, Q, H, 2]

                # Log-scale (SAM3 style) for better gradient flow
                delta_x_log = torch.sign(delta_x) * torch.log2(torch.abs(delta_x * 8) + 1.0) / 3.0
                delta_y_log = torch.sign(delta_y) * torch.log2(torch.abs(delta_y * 8) + 1.0) / 3.0

                # MLP → per-head bias
                rpb_x = self.box_rpb_embed_x(delta_x_log)  # [B, Q, W, n_heads]
                rpb_y = self.box_rpb_embed_y(delta_y_log)  # [B, Q, H, n_heads]

                # Outer sum: [B, Q, H, W, n_heads]
                rpb = rpb_y.unsqueeze(3) + rpb_x.unsqueeze(2)
                rpb = rpb.reshape(B, Q, H_s * W_s, self.n_heads)  # [B, Q, L, n_heads]
                rpb = rpb.permute(0, 3, 1, 2)  # [B, n_heads, Q, L]

                attn_scores = attn_scores + rpb

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs_dropped = self.dropout(attn_probs)

            out = torch.matmul(attn_probs_dropped, V_proj)
            out = out.transpose(1, 2).reshape(B, Q, D)
            out = self.out_proj(out)
            queries = queries + self.dropout(out)

        # Apply post-norm after cross-attention (SAM3-style)
        if self.post_norm:
            queries = self.norm2(queries)

        # 2.5 Depth cross-attention: queries attend to 3D position features
        # This is DIFFERENT from GASA - treats depth as separate modality for explicit fusion
        if self.use_depth_crossattn:
            q_depth = self.depth_norm(queries)
            depth_tokens = self.depth_proj(memory_pos)  # [B, L, D] - project 3D coords to d_model
            depth_out, _ = self.depth_cross_attn(q_depth, depth_tokens, depth_tokens)
            queries = queries + self.dropout(depth_out)

        # 2.5 Image-to-token cross-attention (SAM3 TwoWayTransformer step 4)
        # Pixels attend back to queries → pixel features become mask-aware
        # This updates memory (pixel features), making subsequent dot product sharper
        #
        # SAM3's TwoWayTransformer operates on ~4K tokens (64×64), not raw encoder memory.
        # We pool memory to 64×64 before attention, then bilinear upsample the update back.
        updated_memory = None
        if self.use_image_to_token:
            B_mem, L_mem, D_mem = memory.shape
            # Pool to ~64×64 (4096 tokens) to match SAM3's scale
            H_mem = int(L_mem ** 0.5)
            W_mem = H_mem
            target_size = 64
            if H_mem > target_size:
                # Reshape to spatial, pool, flatten
                mem_spatial = memory.permute(0, 2, 1).reshape(B_mem, D_mem, H_mem, W_mem)
                mem_pooled = F.adaptive_avg_pool2d(mem_spatial, (target_size, target_size))
                mem_flat = mem_pooled.flatten(2).permute(0, 2, 1)  # [B, 4096, D]
                # Also pool PE if available
                if memory_pe is not None:
                    pe_spatial = memory_pe.permute(0, 2, 1).reshape(B_mem, D_mem, H_mem, W_mem)
                    pe_pooled = F.adaptive_avg_pool2d(pe_spatial, (target_size, target_size))
                    pe_flat = pe_pooled.flatten(2).permute(0, 2, 1)
                    img_q = mem_flat + pe_flat
                else:
                    img_q = mem_flat
            else:
                mem_flat = memory
                img_q = memory + memory_pe if memory_pe is not None else memory

            tok_k = queries + query_pe if query_pe is not None else queries
            img2tok_out, _ = self.image_to_token_attn(img_q, tok_k, queries)  # [B, 4096, D]
            mem_updated_flat = mem_flat + self.dropout(img2tok_out)

            # Upsample back to original resolution if pooled
            if H_mem > target_size:
                mem_up = mem_updated_flat.permute(0, 2, 1).reshape(B_mem, D_mem, target_size, target_size)
                mem_up = F.interpolate(mem_up, size=(H_mem, W_mem), mode='bilinear', align_corners=False)
                updated_memory = self.image_to_token_norm(mem_up.flatten(2).permute(0, 2, 1))
            else:
                updated_memory = self.image_to_token_norm(mem_updated_flat)

        # 2.7 Position and box refinement (SAM3-style iterative sharpening)
        refined_query_pos = query_pos
        refined_boxes = reference_boxes
        if self.use_pos_refine and query_pos is not None:
            pos_delta = self.pos_refine_mlp(queries)  # [B, Q, 3]
            refined_query_pos = query_pos + pos_delta
        if self.use_box_rpb and reference_boxes is not None:
            box_delta = self.bbox_embed(queries)  # [B, Q, 4]
            # Inverse sigmoid + delta + sigmoid (SAM3's refinement approach)
            ref_inv = torch.log(reference_boxes / (1 - reference_boxes + 1e-6) + 1e-6)
            refined_boxes = (ref_inv + box_delta).sigmoid()

        # 3. FFN (SAM3 runs FFN in FP32 by disabling autocast)
        if self.post_norm:
            if self.ffn_fp32:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    ffn_out = self.ffn(queries)
            else:
                ffn_out = self.ffn(queries)
            queries = self.norm3(queries + ffn_out)
        else:
            # Pre-norm
            q = self.norm3(queries)
            if self.ffn_fp32:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    ffn_out = self.ffn(q)
            else:
                ffn_out = self.ffn(q)
            queries = queries + ffn_out

        if return_query_pos:
            # Use refined 3D position if available, else attention-weighted
            if self.use_pos_refine and refined_query_pos is not None:
                query_pos_out = refined_query_pos.detach()  # Detach for next layer (like SAM3)
            else:
                # Compute attention-weighted position for next layer
                attn_probs_mean = attn_probs.mean(dim=1)
                query_pos_out = torch.bmm(attn_probs_mean, memory_pos)
            if updated_memory is not None:
                return queries, query_pos_out, updated_memory, refined_boxes
            return queries, query_pos_out, refined_boxes

        if updated_memory is not None:
            return queries, updated_memory, refined_boxes
        return queries, refined_boxes


class MaskRefiner(nn.Module):
    """Lightweight mask refinement for sharper boundaries.

    Takes coarse mask logits (288×288) and refines at same resolution
    using image-guided convolutions. No upsampling (avoids OOM).
    The key insight: sharpen boundaries at coarse res BEFORE bilinear upsample.

    Architecture:
        coarse_mask [B, Q, 288, 288] + image [B, 3, 288, 288]
        → 3 conv layers with image features
        → residual connection (starts as identity)
        → refined_mask [B, Q, 288, 288]

    ~50K parameters.
    """

    def __init__(self, in_channels=1, img_channels=3, hidden_dim=16):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels + img_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        # Initialize last conv to zero → starts as identity (no refinement)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, coarse_mask, image):
        """Refine coarse masks using image guidance.

        Args:
            coarse_mask: [B, Q, H_c, W_c] mask logits
            image: [B, 3, H, W] input image

        Returns:
            refined_mask: [B, Q, H_c, W_c] refined mask logits (same resolution)
        """
        B, Q, H_c, W_c = coarse_mask.shape

        # Downsample image to mask resolution
        img_small = F.interpolate(image, size=(H_c, W_c), mode='bilinear', align_corners=False)

        # Process one query at a time to save memory
        refined_list = []
        for q in range(Q):
            mask_q = coarse_mask[:, q:q+1, :, :]  # [B, 1, H_c, W_c]
            x = torch.cat([mask_q, img_small], dim=1)  # [B, 4, H_c, W_c]
            residual = self.refine(x)  # [B, 1, H_c, W_c]
            refined_list.append(mask_q + residual)

        return torch.cat(refined_list, dim=1)  # [B, Q, H_c, W_c]


class SpatialAttentionBias(nn.Module):
    """Depth-conditioned spatial attention bias for cross-attention.

    Instead of a static spatial embedding on queries, this creates a per-pixel
    attention bias based on the spatial qualifier and depth/position information.
    The bias is added to cross-attention scores, softly guiding attention toward
    the spatially correct region (e.g., nearest → low-depth pixels).

    This gives the spatial token actual geometric information rather than
    relying on a single learned embedding.
    """

    def __init__(self, n_heads: int = 8):
        super().__init__()
        # Learned temperature per qualifier per head
        self.spatial_temp = nn.Parameter(torch.ones(8, n_heads) * 2.0)
        self.n_heads = n_heads

    def forward(self, spatial_qualifier_idx, depth, H_attn, W_attn):
        """Create spatial attention bias map.

        Args:
            spatial_qualifier_idx: [B] qualifier indices (0=none, 1=nearest, ...)
            depth: [B, 1, H, W] depth map
            H_attn, W_attn: spatial dims of the attention map (L = H_attn * W_attn)

        Returns:
            bias: [B, n_heads, 1, L] attention bias (broadcast over Q dimension)
                  or None if no spatial qualifiers active
        """
        B = spatial_qualifier_idx.shape[0]
        device = spatial_qualifier_idx.device

        if (spatial_qualifier_idx == 0).all():
            return None

        # Resize depth to attention map size
        depth_resized = F.interpolate(depth, size=(H_attn, W_attn), mode='bilinear',
                                       align_corners=False)  # [B, 1, H, W]
        depth_flat = depth_resized.view(B, -1)  # [B, L]

        # Create normalized coordinate grids
        y_coords = torch.linspace(0, 1, H_attn, device=device).view(H_attn, 1).expand(H_attn, W_attn)
        x_coords = torch.linspace(0, 1, W_attn, device=device).view(1, W_attn).expand(H_attn, W_attn)
        y_flat = y_coords.reshape(-1).unsqueeze(0).expand(B, -1)  # [B, L]
        x_flat = x_coords.reshape(-1).unsqueeze(0).expand(B, -1)  # [B, L]

        # Normalize depth to [0, 1] per sample
        d_min = depth_flat.min(dim=-1, keepdim=True).values
        d_max = depth_flat.max(dim=-1, keepdim=True).values
        depth_norm = (depth_flat - d_min) / (d_max - d_min + 1e-6)  # [B, L]

        # Build spatial prior for each sample based on qualifier
        # prior: [B, L] — higher = more attention
        bias = torch.zeros(B, depth_flat.shape[-1], device=device)
        for b in range(B):
            sq = spatial_qualifier_idx[b].item()
            if sq == 0:  # none
                continue
            elif sq == 1:  # nearest (low depth)
                bias[b] = -depth_norm[b]
            elif sq == 2:  # farthest (high depth)
                bias[b] = depth_norm[b]
            elif sq == 3:  # leftmost (low x)
                bias[b] = -x_flat[b]
            elif sq == 4:  # rightmost (high x)
                bias[b] = x_flat[b]
            elif sq == 5:  # topmost (low y)
                bias[b] = -y_flat[b]
            elif sq == 6:  # bottommost (high y)
                bias[b] = y_flat[b]
            elif sq == 7:  # center
                cx, cy = 0.5, 0.5
                dist = ((x_flat[b] - cx) ** 2 + (y_flat[b] - cy) ** 2).sqrt()
                bias[b] = -dist

        # Apply per-head learned temperature: [B, L] -> [B, n_heads, 1, L]
        # Gather temperature for each sample's qualifier
        temps = self.spatial_temp[spatial_qualifier_idx]  # [B, n_heads]
        bias = bias.unsqueeze(1).unsqueeze(2) * temps.unsqueeze(-1).unsqueeze(-1)  # [B, n_heads, 1, L]

        return bias


class TextConditionedSpatialBias(nn.Module):
    """ViL3DRel-inspired text-conditioned spatial attention bias.

    Instead of using hardcoded qualifier indices, this module uses the text
    embedding to learn which spatial dimensions (depth, x, y) are relevant
    for the current query. The text "nearest chair" should learn to activate
    the depth dimension, while "leftmost chair" activates x.

    Architecture (from ViL3DRel):
        g = W_spatial * text_cls      → [B, 6] gating vector
        f = [depth, x, y, 1/depth, 1-x, 1-y]  → [B, L, 6] spatial features per pixel
        bias = sigmoid(g) * f          → [B, L] per-pixel spatial score
    """

    def __init__(self, text_dim: int = 256, spatial_dim: int = 6, n_heads: int = 8):
        super().__init__()
        # Text → spatial gate projection
        self.text_to_gate = nn.Sequential(
            nn.Linear(text_dim, spatial_dim * n_heads),
        )
        self.spatial_dim = spatial_dim
        self.n_heads = n_heads
        # Initialize near-zero so bias starts neutral
        nn.init.zeros_(self.text_to_gate[0].weight)
        nn.init.zeros_(self.text_to_gate[0].bias)

    def forward(self, text_cls, depth, H_attn, W_attn):
        """Create text-conditioned spatial attention bias.

        Args:
            text_cls: [B, D] text CLS embedding (256-dim)
            depth: [B, 1, H, W] depth map
            H_attn, W_attn: spatial dims of attention map

        Returns:
            bias: [B, n_heads, 1, L] attention bias or None
        """
        B = text_cls.shape[0]
        device = text_cls.device

        # Compute spatial gate from text: which dimensions matter?
        gate = self.text_to_gate(text_cls)  # [B, spatial_dim * n_heads]
        gate = gate.view(B, self.n_heads, self.spatial_dim)  # [B, n_heads, spatial_dim]
        gate = torch.sigmoid(gate)  # [B, n_heads, spatial_dim]

        # Build spatial features per pixel
        depth_resized = F.interpolate(depth, size=(H_attn, W_attn), mode='bilinear',
                                       align_corners=False)
        depth_flat = depth_resized.view(B, -1)  # [B, L]

        # Normalize depth per sample
        d_min = depth_flat.min(dim=-1, keepdim=True).values
        d_max = depth_flat.max(dim=-1, keepdim=True).values
        depth_norm = (depth_flat - d_min) / (d_max - d_min + 1e-6)  # [B, L]

        # Coordinate grids
        y_coords = torch.linspace(0, 1, H_attn, device=device).view(H_attn, 1).expand(H_attn, W_attn)
        x_coords = torch.linspace(0, 1, W_attn, device=device).view(1, W_attn).expand(H_attn, W_attn)
        y_flat = y_coords.reshape(-1).unsqueeze(0).expand(B, -1)
        x_flat = x_coords.reshape(-1).unsqueeze(0).expand(B, -1)

        # Stack spatial features: [B, L, 6]
        # [depth_norm, 1-depth_norm, x, 1-x, y, 1-y]
        # This way "nearest" (low depth) can be captured by gating depth_norm dimension negatively
        # or by gating the 1-depth_norm dimension positively
        spatial_features = torch.stack([
            -depth_norm,       # nearest: high value at low depth
            depth_norm,        # farthest: high value at high depth
            -x_flat,           # leftmost: high value at low x
            x_flat,            # rightmost: high value at high x
            -y_flat,           # topmost: high value at low y
            y_flat,            # bottommost: high value at high y
        ], dim=-1)  # [B, L, 6]

        # Apply text-conditioned gate: [B, n_heads, 6] @ [B, L, 6].T → [B, n_heads, L]
        # Einstein notation: gate[b,h,d] * features[b,l,d] → bias[b,h,l]
        bias = torch.einsum('bhd,bld->bhl', gate, spatial_features)  # [B, n_heads, L]

        # Reshape for attention: [B, n_heads, 1, L] (broadcast over Q)
        bias = bias.unsqueeze(2)

        return bias
