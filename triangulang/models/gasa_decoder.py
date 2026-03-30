"""
GASA Decoder - Geometry-Aware Self-Attention Decoder

Replaces SAM3's decoder with geometry-aware cross-attention that fuses
SAM3 features with geometric encoding for cross-view consistency.

Key classes:
- GASADecoderLayer: Single decoder layer with geometric attention bias
- MaskRefiner: Lightweight mask refinement for sharper boundaries
- SpatialAttentionBias: Depth-conditioned spatial attention bias
- TextConditionedSpatialBias: ViL3DRel-inspired text-conditioned spatial bias
- GASADecoder: Full decoder with text conditioning, presence tokens, etc.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from triangulang.models.gasa import (
    PointmapComputer,
    WorldSpacePositionalEncoding,
    CameraRelativePositionalEncoding,
    PluckerEmbedding,
    RayRoPE3D,
)

from sam3.sam.prompt_encoder import PositionEmbeddingRandom
from sam3.model.model_misc import MLP as SAM3MLP


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
        L = H_attn * W_attn
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


class GASADecoder(nn.Module):
    """
    GASA Decoder - Replaces SAM3's decoder

    Key differences from SAM3's decoder:
    1. World-Space PE (from DA3 pointmaps) instead of 2D PE
    2. Geometric attention bias (GASA)
    3. Text-conditioned queries
    4. Presence token for "is object in image?" prediction
    5. Box prompt encoding (geometry encoder)

    With use_gasa=False, becomes standard decoder (for ablation experiments).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, num_layers: int = 6, num_queries: int = 100,
                 use_presence_token: bool = True, use_box_prompts: bool = True, use_point_prompts: bool = True,
                 use_world_pe: bool = True, use_gasa: bool = True, use_centroid_head: bool = False,
                 use_iterative_pos: bool = False, cross_view: bool = True, pe_type: str = 'world',
                 use_iou_head: bool = False, use_spatial_tokens: bool = False, gasa_beta_init: float = 1.0,
                 gasa_kernel_dim: int = 32, gasa_fixed_kernel: bool = False, gasa_kernel_type: str = 'learned',
                 use_depth_crossattn: bool = False,
                 per_layer_text: bool = False, gasa_bidirectional: bool = False,
                 dim_feedforward: int = 2048, post_norm: bool = True,
                 use_query_pe: bool = False, ffn_fp32: bool = True,
                 no_initial_text: bool = False, no_text_proj: bool = False,
                 clean_v: bool = False,
                 additive_pe: bool = False,
                 grouped_text_attn: bool = False,
                 use_spatial_attn_bias: bool = False,
                 use_text_spatial_bias: bool = False,
                 use_image_to_token: bool = False,
                 use_pos_refine: bool = False,
                 use_box_rpb: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.no_initial_text = no_initial_text
        self.no_text_proj = no_text_proj
        self.clean_v = clean_v
        self.additive_pe = additive_pe
        self.grouped_text_attn = grouped_text_attn
        self.use_presence_token = use_presence_token
        self.use_box_prompts = use_box_prompts
        self.use_point_prompts = use_point_prompts  # MV-SAM style click prompts
        self.use_world_pe = use_world_pe
        self.use_gasa = use_gasa  # Ablation: disable geometric bias
        self.use_centroid_head = use_centroid_head  # 3D localization output
        self.use_iterative_pos = use_iterative_pos  # Iterative query positions (Option B)
        self.cross_view = cross_view  # Ablation: cross-view vs single-view attention
        self.pe_type = pe_type  # Ablation: 'world', 'camera_relative', 'plucker', 'rayrope', or 'none'
        self.use_iou_head = use_iou_head  # IoU prediction for zero-shot mask selection
        self.use_spatial_tokens = use_spatial_tokens  # Spatial qualifier embeddings
        self.use_spatial_attn_bias = use_spatial_attn_bias  # Depth-conditioned spatial attention bias
        self.use_depth_crossattn = use_depth_crossattn  # Deep depth fusion via cross-attention
        self.per_layer_text = per_layer_text  # Per-layer text cross-attention (SAM3-style)

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # SPATIAL TOKEN EMBEDDINGS: Condition queries on spatial qualifiers
        # Indices: 0=none, 1=nearest, 2=farthest, 3=left, 4=right, 5=top, 6=bottom, 7=center
        if use_spatial_tokens:
            self.spatial_embeddings = nn.Embedding(8, d_model)
            # Initialize with small values so spatial tokens start neutral
            nn.init.normal_(self.spatial_embeddings.weight, std=0.02)

        # Depth-conditioned spatial attention bias
        if use_spatial_attn_bias:
            self.spatial_attn_bias = SpatialAttentionBias(n_heads=n_heads)

        # Text-conditioned spatial bias (ViL3DRel-inspired)
        self.use_text_spatial_bias = use_text_spatial_bias
        if use_text_spatial_bias:
            self.text_spatial_bias = TextConditionedSpatialBias(
                text_dim=d_model, spatial_dim=6, n_heads=n_heads
            )

        # Text conditioning - queries cross-attend to text
        # SAM3's language_features are always 256-dim, project to our d_model
        # Note: text_proj is ONLY for cross-attention conditioning.
        # Scoring uses raw text_embedding directly (matching SAM3's DotProductScoring).
        self.text_proj = nn.Linear(256, d_model)
        self.text_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.text_norm = nn.LayerNorm(d_model)

        # Positional Encoding (our key contribution - replaces 2D PE)
        # pe_type: 'world' (additive sinusoidal), 'camera_relative' (additive sinusoidal in camera frame),
        #          'plucker' (additive ray), 'rayrope' (rotary), 'none'
        self.rayrope = None
        self.plucker_pe = None
        if pe_type == 'world':
            self.world_pe = WorldSpacePositionalEncoding(d_model=d_model, num_frequencies=10, max_frequency=512.0)
        elif pe_type == 'camera_relative':
            self.world_pe = CameraRelativePositionalEncoding(d_model=d_model, num_frequencies=10, max_frequency=512.0)
        elif pe_type == 'plucker':
            self.world_pe = None
            self.plucker_pe = PluckerEmbedding(d_model=d_model, num_frequencies=6)
        elif pe_type == 'rayrope':
            self.world_pe = None
            self.plucker_pe = None
            head_dim = d_model // n_heads  # 256 // 8 = 32
            # 4 coords (3D direction + depth) × 4 frequencies × 2 (sin/cos pair) = 32
            self.rayrope = RayRoPE3D(
                head_dim=head_dim,
                num_freqs=4,
                coord_dim=4,
                max_period=4.0,
                freq_base=3.0,
                use_uncertainty=True,
            )
        else:  # 'none'
            self.world_pe = None
            self.plucker_pe = None

        # PRESENCE TOKEN: Predicts p(object exists in image)
        # This helps filter false positives when object isn't in the image
        if use_presence_token:
            self.presence_token = nn.Embedding(1, d_model)
            self.presence_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
            self.presence_norm = nn.LayerNorm(d_model)

        # CENTROID HEAD: Predicts residual offset to attention-weighted 3D positions
        # This enables Acc@5cm, Acc@10cm metrics for 3D localization
        # Initialize to output ~zeros so we start from pure geometric positions
        if use_centroid_head:
            self.centroid_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 3),  # [x, y, z] residual offset
            )
            self.centroid_norm = nn.LayerNorm(d_model)
            # Initialize last layer to output zeros - start from pure attention-weighted positions
            nn.init.zeros_(self.centroid_head[-1].weight)
            nn.init.zeros_(self.centroid_head[-1].bias)

        # IOU HEAD: Predicts IoU with GT for each query mask
        # Enables zero-shot mask selection without GT at inference time
        if use_iou_head:
            self.iou_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()  # Output ∈ [0, 1] for IoU
            )

        # TEXT SCORING HEAD: SAM3-style DotProductScoring for text-aware mask selection
        # Replicates SAM3's _create_dot_product_scoring() architecture:
        #   prompt_mlp: 2-layer MLP (256→2048→256) with residual + LayerNorm
        #   prompt_proj: Linear(256→256) for projected text
        #   hs_proj: Linear(256→256) for projected queries
        # Order: MLP → mean_pool → project → dot product (matching SAM3)
        self.scoring_prompt_mlp = SAM3MLP(
            input_dim=d_model, hidden_dim=2048, output_dim=d_model,
            num_layers=2, dropout=0.1, residual=True,
            out_norm=nn.LayerNorm(d_model),
        )
        self.scoring_prompt_proj = nn.Linear(d_model, d_model)  # was scoring_text_proj
        self.scoring_hs_proj = nn.Linear(d_model, d_model)      # was scoring_query_proj
        self.scoring_scale = float(1.0 / (d_model ** 0.5))

        # SHARED PROMPT ENCODER: SAM3-style PositionEmbeddingRandom for both points and boxes
        # This is shared because SAM3 treats box corners as special point types (labels 2, 3)
        if use_box_prompts or use_point_prompts:
            # Positional encoding: SAM3-style with random Gaussian matrix + sin/cos
            # Outputs d_model dimensions (num_pos_feats*2 from sin+cos)
            self.prompt_pe = PositionEmbeddingRandom(d_model // 2)
            # Label embeddings: 4 types like SAM3:
            #   0 = negative point (background)
            #   1 = positive point (foreground)
            #   2 = box corner 1 (top-left)
            #   3 = box corner 2 (bottom-right)
            self.num_point_embeddings = 4
            self.point_embeddings = nn.ModuleList([
                nn.Embedding(1, d_model) for _ in range(self.num_point_embeddings)
            ])
            # Not-a-point embedding for padding (label=-1)
            self.not_a_point_embed = nn.Embedding(1, d_model)
            # Store image size for coordinate normalization
            self.prompt_input_image_size = (1024, 1024)  # Default SAM3 size

        # BOX PROMPT ENCODER: Cross-attention layer for box prompts
        # Boxes are encoded as 2 corner points using shared PE (SAM3 style)
        if use_box_prompts:
            self.box_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
            self.box_norm = nn.LayerNorm(d_model)

        # POINT/CLICK PROMPT ENCODER: Cross-attention layer for point prompts
        if use_point_prompts:
            self.point_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
            self.point_norm = nn.LayerNorm(d_model)

        # GASA decoder layers (use_gasa controls geometric bias)
        # gasa_beta_init controls initial strength (for ablation experiments)
        # gasa_kernel_dim and gasa_fixed_kernel control kernel architecture
        # use_depth_crossattn adds explicit cross-attention to depth features
        # per_layer_text adds SAM3-style per-layer text cross-attention
        self.layers = nn.ModuleList([
            GASADecoderLayer(d_model=d_model, n_heads=n_heads, dim_feedforward=dim_feedforward,
                             use_gasa=use_gasa,
                             gasa_beta_init=gasa_beta_init, gasa_kernel_dim=gasa_kernel_dim,
                             gasa_fixed_kernel=gasa_fixed_kernel, gasa_kernel_type=gasa_kernel_type,
                             use_depth_crossattn=use_depth_crossattn,
                             per_layer_text=per_layer_text,
                             gasa_bidirectional=gasa_bidirectional,
                             post_norm=post_norm,
                             ffn_fp32=ffn_fp32,
                             use_image_to_token=use_image_to_token,
                             use_pos_refine=use_pos_refine,
                             use_box_rpb=use_box_rpb)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # QUERY PE: Learned positional embedding for queries (SAM3-style)
        # SAM3 uses conditional PE from bbox reference points, we use learned embeddings.
        # Added to Q,K in self-attention and Q in cross-attention at every layer.
        self.use_query_pe = use_query_pe
        if use_query_pe:
            # +1 for presence token if used (prepended to queries)
            total_queries = num_queries + (1 if use_presence_token else 0)
            self.query_pe_embed = nn.Embedding(total_queries, d_model)
            nn.init.normal_(self.query_pe_embed.weight.data, std=0.02)

    def encode_boxes(self, boxes, labels=None):
        """Encode bounding boxes as prompt tokens using SAM3-style corner encoding.

        SAM3 treats boxes as 2 special point types (corners), using the same
        PositionEmbeddingRandom as points. This unified approach preserves
        spatial structure better than linear projection.

        Args:
            boxes: [B, N_boxes, 4] normalized boxes in cxcywh format (cx, cy, w, h) ∈ [0,1]
            labels: [B, N_boxes] box labels (unused, kept for API compatibility)
                   In SAM3, corners always use embeddings[2] and [3]

        Returns:
            box_tokens: [B, N_boxes*2, D] encoded corner tokens (2 per box)
        """
        B, N_boxes = boxes.shape[:2]

        # Convert cxcywh to corner coordinates (x1, y1, x2, y2)
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        x1 = cx - w / 2  # top-left x
        y1 = cy - h / 2  # top-left y
        x2 = cx + w / 2  # bottom-right x
        y2 = cy + h / 2  # bottom-right y

        # Stack as corner pairs: [B, N_boxes, 2, 2] where last dims are (x, y) for each corner
        corners = torch.stack([
            torch.stack([x1, y1], dim=-1),  # top-left corner
            torch.stack([x2, y2], dim=-1),  # bottom-right corner
        ], dim=2)  # [B, N_boxes, 2, 2]

        # Reshape to [B, N_boxes*2, 2] for encoding
        corners_flat = corners.view(B, N_boxes * 2, 2)

        # Apply SAM3-style positional encoding
        # Points are normalized [0,1], use _pe_encoding directly
        corner_embedding = self.prompt_pe._pe_encoding(corners_flat.to(torch.float))

        # Reshape back to [B, N_boxes, 2, D] to add corner-specific embeddings
        corner_embedding = corner_embedding.view(B, N_boxes, 2, -1)

        # Add corner type embeddings (SAM3 style)
        # point_embeddings[2] = box corner 1 (top-left)
        # point_embeddings[3] = box corner 2 (bottom-right)
        corner_embedding[:, :, 0, :] = corner_embedding[:, :, 0, :] + self.point_embeddings[2].weight
        corner_embedding[:, :, 1, :] = corner_embedding[:, :, 1, :] + self.point_embeddings[3].weight

        # Flatten back to [B, N_boxes*2, D]
        return corner_embedding.view(B, N_boxes * 2, -1)

    def encode_points(self, points, labels, image_size=None):
        """Encode click points as prompt tokens using SAM3-style positional encoding.

        Uses PositionEmbeddingRandom (sine/cosine PE through Gaussian matrix) instead
        of linear projection. This preserves spatial structure critical for point prompts.

        Args:
            points: [B, N_points, 2] points (x, y) - can be normalized [0,1] or pixel coords
            labels: [B, N_points] point labels:
                    -1 = not a point (padding)
                     0 = negative point (background)
                     1 = positive point (foreground)
                     2 = box corner 1 (top-left)
                     3 = box corner 2 (bottom-right)
            image_size: Optional (H, W) for coordinate normalization. If None, assumes
                       points are already normalized to [0, 1].

        Returns:
            point_tokens: [B, N_points, D] encoded point tokens with positional + label info
        """
        # Get positional encoding via SAM3's method (using shared prompt_pe)
        if image_size is not None:
            # Points are in pixel coordinates
            # Add +0.5 to shift to center of pixel (SAM3 convention)
            pixel_points = points + 0.5
            point_embedding = self.prompt_pe.forward_with_coords(pixel_points, image_size)
        else:
            # Points are already normalized to [0, 1]
            # Use _pe_encoding directly (expects normalized coords)
            # _pe_encoding does: coords = 2*coords - 1, then Gaussian matrix, then sin/cos
            point_embedding = self.prompt_pe._pe_encoding(points.to(torch.float))

        # Add label-specific embeddings (SAM3 style with torch.where for batched ops)
        # Not-a-point (padding, label=-1)
        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        # Negative point (label=0)
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1),
            point_embedding + self.point_embeddings[0].weight,
            point_embedding,
        )
        # Positive point (label=1)
        point_embedding = torch.where(
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight,
            point_embedding,
        )
        # Box corner 1 (label=2) - can be used for box prompts encoded as points
        point_embedding = torch.where(
            (labels == 2).unsqueeze(-1),
            point_embedding + self.point_embeddings[2].weight,
            point_embedding,
        )
        # Box corner 2 (label=3)
        point_embedding = torch.where(
            (labels == 3).unsqueeze(-1),
            point_embedding + self.point_embeddings[3].weight,
            point_embedding,
        )

        return point_embedding

    def load_state_dict_compat(self, state_dict, strict=True):
        """Load state dict with backward compatibility for old checkpoint formats.

        Handles parameter name changes from the SAM3-style prompt encoding update:
        - Old: point_proj (Linear), point_label_embed (Embedding), box_proj, box_label_embed
        - New: prompt_pe (PositionEmbeddingRandom), point_embeddings (ModuleList)

        Args:
            state_dict: Checkpoint state dict
            strict: If True, raise error on missing/unexpected keys (after compat mapping)

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        # Create a copy to avoid modifying the original
        compat_state_dict = {}
        skipped_keys = []

        for key, value in state_dict.items():
            # Skip old linear projection layers (incompatible architecture)
            if key in ['point_proj.weight', 'point_proj.bias',
                       'box_proj.weight', 'box_proj.bias',
                       'box_label_embed.weight']:
                skipped_keys.append(key)
                continue

            # Map old point_pe to new prompt_pe
            if key.startswith('point_pe.'):
                new_key = key.replace('point_pe.', 'prompt_pe.')
                compat_state_dict[new_key] = value
                continue

            # Map old point_label_embed to first 2 point_embeddings (if shape matches)
            if key == 'point_label_embed.weight':
                # Old: [2, d_model] embedding for neg/pos labels
                # New: ModuleList of 4 [1, d_model] embeddings
                # We can initialize the first 2 from old weights
                if value.shape[0] >= 2:
                    # This is a heuristic - won't be exact but better than random
                    skipped_keys.append(key + ' (mapped to point_embeddings init)')
                continue

            # Map old point_input_image_size to prompt_input_image_size
            if key == 'point_input_image_size':
                compat_state_dict['prompt_input_image_size'] = value
                continue

            # Map old scoring projections to new SAM3-style names
            if key.startswith('scoring_text_proj.'):
                new_key = key.replace('scoring_text_proj.', 'scoring_prompt_proj.')
                compat_state_dict[new_key] = value
                continue
            if key.startswith('scoring_query_proj.'):
                new_key = key.replace('scoring_query_proj.', 'scoring_hs_proj.')
                compat_state_dict[new_key] = value
                continue

            # Skip old text_proj MLP params (now always simple Linear)
            # Old MLP had keys like text_proj.0.weight, text_proj.2.weight, etc.
            if key.startswith('text_proj.') and key.split('.')[1].isdigit():
                skipped_keys.append(key)
                continue
            # Skip old text_proj_residual_norm (removed)
            if key.startswith('text_proj_residual_norm.'):
                skipped_keys.append(key)
                continue

            # Keep all other keys unchanged
            compat_state_dict[key] = value

        if skipped_keys:
            print(f"  [Compat] Skipped old prompt encoding params: {skipped_keys}")

        # Handle size mismatches (e.g., num_queries changed between train and eval)
        current_sd = self.state_dict()
        resized_keys = []
        for key in list(compat_state_dict.keys()):
            if key in current_sd and compat_state_dict[key].shape != current_sd[key].shape:
                resized_keys.append(f"{key}: ckpt={list(compat_state_dict[key].shape)} vs model={list(current_sd[key].shape)}")
                del compat_state_dict[key]
        if resized_keys:
            print(f"  [Compat] Dropped size-mismatched keys (will use model init): {resized_keys}")

        return super().load_state_dict(compat_state_dict, strict=strict)

    def forward(self, memory, pointmaps, text_embedding=None, box_prompts=None, box_labels=None,
                point_prompts=None, point_labels=None, depths=None, poses=None, intrinsics=None,
                spatial_qualifier_idx=None,
                poses_per_view=None, intrinsics_per_view=None, num_cameras=1,
                num_texts=1, tokens_per_text=None):
        """
        Args:
            memory: [B, L, D] encoder memory from SAM3
            pointmaps: [B, H, W, 3] world coordinates from DA3
            text_embedding: [B, T, D] or [B, K*T, D] text embeddings for conditioning
            box_prompts: [B, N_boxes, 4] optional box prompts (cxcywh, normalized)
            box_labels: [B, N_boxes] optional box labels (1=positive, 0=negative)
            point_prompts: [B, N_points, 2] optional click prompts (x, y normalized)
            point_labels: [B, N_points] optional point labels (1=positive, 0=negative)
            depths: [B, 1, H, W] depths for Plücker PE (optional, needed if pe_type='plucker')
            poses: [B, 4, 4] camera poses for single-view (optional)
            intrinsics: [B, 3, 3] camera intrinsics for single-view (optional)
            spatial_qualifier_idx: [B] optional spatial qualifier indices (0=none, 1=nearest, etc.)
            poses_per_view: [B, N, 4, 4] per-view c2w poses for cross-view rayrope (optional)
            intrinsics_per_view: [B, N, 3, 3] per-view intrinsics for cross-view rayrope (optional)
            num_cameras: int, number of cameras (1=single-view, >1=cross-view)
            num_texts: int, number of text prompts per batch item (K for multi-object, 1 for single)
            tokens_per_text: int, number of tokens per text prompt (for reshaping K*T -> K,T)

        Returns:
            queries: [B, Q, D] decoded object queries
            presence_logit: [B, 1] presence prediction (if use_presence_token)
            centroid_pred: [B, 3] 3D centroid prediction (if use_centroid_head)
        """
        B = memory.shape[0]
        L = memory.shape[1]
        device = memory.device

        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1).clone()

        # SPATIAL TOKEN CONDITIONING: Add spatial embedding to queries
        # This teaches the model to prefer objects matching spatial qualifiers
        if self.use_spatial_tokens and spatial_qualifier_idx is not None:
            spatial_embed = self.spatial_embeddings(spatial_qualifier_idx)  # [B, D]
            queries = queries + spatial_embed.unsqueeze(1)  # Broadcast to all queries

        # PRESENCE TOKEN: Add to queries, will be separated later
        presence_out = None
        if self.use_presence_token:
            presence_token = self.presence_token.weight.unsqueeze(0).expand(B, -1, -1)
            # Concatenate presence token with queries for joint processing
            queries_with_presence = torch.cat([presence_token, queries], dim=1)
        else:
            queries_with_presence = queries

        # TEXT CONDITIONING: Queries attend to text to know what to find
        # IMPORTANT: Always run text modules to ensure DDP gradient sync (avoid unused params error)
        # Store text for per-layer text conditioning (if per_layer_text=True)
        # When no_initial_text=True, skip initial cross-attention (SAM3 does 6x per-layer only, not 7x)
        # When no_text_proj=True, bypass text_proj entirely (SAM3 has no text projection before ca_text)
        #
        # GROUPED TEXT ATTENTION (--grouped-text-attn):
        # When enabled with multi-object (num_texts > 1), creates an attention mask so each query
        # group only attends to its assigned text tokens. Matches SAM3's text routing where each
        # query sees exactly ONE text, but in a single forward pass.
        # Q queries split into K groups of Q//K. Group g attends to text tokens [g*T : (g+1)*T].
        # Presence token (if used) attends to ALL texts for global context.
        text_proj_for_layers = None
        text_attn_mask = None  # [Q_total, K*T] mask for grouped text attention

        # Build grouped text attention mask if needed
        if self.grouped_text_attn and num_texts > 1 and tokens_per_text is not None:
            Q_total = queries_with_presence.shape[1]  # Q + 1 (presence) or Q
            K = num_texts
            T_per = tokens_per_text
            KT = K * T_per
            Q_obj = self.num_queries  # Object queries (excluding presence token)
            queries_per_group = Q_obj // K  # Queries per text group
            remainder = Q_obj % K

            # Build mask: True = BLOCK attention (PyTorch convention for bool masks)
            text_attn_mask = torch.ones(Q_total, KT, dtype=torch.bool, device=queries.device)

            # Presence token (index 0 if used) attends to ALL texts
            offset = 1 if self.use_presence_token else 0
            if self.use_presence_token:
                text_attn_mask[0, :] = False  # Presence sees everything

            # Assign query groups to text groups
            q_start = offset
            for g in range(K):
                # Extra query for first `remainder` groups to handle Q not divisible by K
                n_q = queries_per_group + (1 if g < remainder else 0)
                q_end = q_start + n_q
                t_start = g * T_per
                t_end = (g + 1) * T_per
                text_attn_mask[q_start:q_end, t_start:t_end] = False  # Allow attention
                q_start = q_end

        if text_embedding is not None:
            if self.no_text_proj:
                text_for_ca = text_embedding  # SAM3-style: raw text directly to cross-attn
                # DDP: still connect text_proj params with zero contribution
                dummy_proj = self.text_proj(text_embedding[:, :1])
                text_for_ca = text_for_ca + dummy_proj.sum() * 0.0
            else:
                text_for_ca = self.text_proj(text_embedding)
            if self.no_initial_text:
                # SAM3-style: skip initial text conditioning, only use per-layer
                # Still run modules with zero contribution for DDP gradient sync
                dummy_cond, _ = self.text_cross_attn(queries_with_presence[:, :1], text_for_ca[:, :1], text_for_ca[:, :1])
                queries_with_presence = queries_with_presence + self.text_norm(dummy_cond).sum() * 0.0
            else:
                text_cond, _ = self.text_cross_attn(queries_with_presence, text_for_ca, text_for_ca,
                                                     attn_mask=text_attn_mask)
                queries_with_presence = self.text_norm(queries_with_presence + text_cond)
            # Store for per-layer conditioning
            text_proj_for_layers = text_for_ca
        else:
            # Connect text params to graph with zero contribution (for DDP)
            dummy_text = torch.zeros(B, 1, 256, device=queries.device, dtype=queries.dtype)
            dummy_proj = self.text_proj(dummy_text)
            dummy_cond, _ = self.text_cross_attn(queries_with_presence[:, :1], dummy_proj, dummy_proj)
            queries_with_presence = queries_with_presence + self.text_norm(dummy_cond).sum() * 0.0

        # BOX PROMPT CONDITIONING: Queries attend to box prompts
        if self.use_box_prompts and box_prompts is not None and box_labels is not None:
            box_tokens = self.encode_boxes(box_prompts, box_labels)
            box_cond, _ = self.box_cross_attn(queries_with_presence, box_tokens, box_tokens)
            queries_with_presence = self.box_norm(queries_with_presence + box_cond)

        # POINT/CLICK PROMPT CONDITIONING: Queries attend to click prompts (MV-SAM style)
        if self.use_point_prompts and point_prompts is not None and point_labels is not None:
            point_tokens = self.encode_points(point_prompts, point_labels)
            point_cond, _ = self.point_cross_attn(queries_with_presence, point_tokens, point_tokens)
            queries_with_presence = self.point_norm(queries_with_presence + point_cond)

        # Flatten pointmaps - handle both [B, H, W, 3] and [B, L, 3] formats
        if pointmaps.dim() == 4:
            # Spatial format: [B, H, W, 3]
            H, W = pointmaps.shape[1:3]
            memory_pos = pointmaps.view(B, H * W, 3)
        else:
            # Already flattened: [B, L, 3] (e.g., from cross-view concatenation)
            memory_pos = pointmaps
            H = W = int(math.sqrt(memory_pos.shape[1]))  # Approximate for size matching

        # Handle size mismatch
        if L != memory_pos.shape[1]:
            if pointmaps.dim() == 4:
                # Spatial format - can use adaptive pooling
                pts = pointmaps.permute(0, 3, 1, 2)
                target_size = int(math.sqrt(L))
                pts = F.adaptive_avg_pool2d(pts, (target_size, target_size))
                memory_pos = pts.permute(0, 2, 3, 1).view(B, -1, 3)
            else:
                # Already flattened - use interpolation
                # Reshape to pseudo-spatial, interpolate, flatten
                L_pts = memory_pos.shape[1]
                H_pts = int(math.sqrt(L_pts))
                if H_pts * H_pts != L_pts:
                    # Not a perfect square, just truncate/pad
                    pass
                else:
                    pts = memory_pos.view(B, H_pts, H_pts, 3).permute(0, 3, 1, 2)
                    target_size = int(math.sqrt(L))
                    pts = F.adaptive_avg_pool2d(pts, (target_size, target_size))
                    memory_pos = pts.permute(0, 2, 3, 1).view(B, -1, 3)

            # Ensure exact size match
            if memory_pos.shape[1] > L:
                memory_pos = memory_pos[:, :L]
            elif memory_pos.shape[1] < L:
                pad = torch.zeros(B, L - memory_pos.shape[1], 3, device=device)
                memory_pos = torch.cat([memory_pos, pad], dim=1)

        # Add Positional Encoding to memory (our key contribution)
        # pe_type: 'world' (additive sinusoidal), 'camera_relative' (additive sinusoidal in camera frame),
        #          'plucker' (additive ray), 'rayrope' (rotary), 'none'
        #
        # For rayrope: NO additive PE. RoPE is applied at Q/K level inside each decoder layer.
        # We compute w2c here and pass rayrope module + camera params through to layers.
        rayrope_ctx = None  # Will be set for rayrope PE type
        if self.pe_type == 'rayrope' and self.rayrope is not None:
            memory_with_pe = memory  # No additive PE — RoPE handles it in attention
            if num_cameras > 1 and poses_per_view is not None and intrinsics_per_view is not None:
                # Cross-view mode: per-camera RayRoPE attention
                w2c_per_view = torch.linalg.inv(poses_per_view)  # [B, N, 4, 4]
                rayrope_ctx = {
                    'rayrope': self.rayrope,
                    'w2c_per_view': w2c_per_view,
                    'intrinsics_per_view': intrinsics_per_view,
                    'num_cameras': num_cameras,
                    'depth_conf': None,
                }
            else:
                # Single-view mode: one w2c for all tokens
                if poses is not None:
                    w2c = torch.linalg.inv(poses)  # [B, 4, 4]
                else:
                    w2c = None
                rayrope_ctx = {
                    'rayrope': self.rayrope,
                    'w2c': w2c,
                    'intrinsics': intrinsics,
                    'depth_conf': None,
                }
        elif self.pe_type == 'camera_relative' and self.world_pe is not None:
            if poses is not None:
                w2c = torch.linalg.inv(poses)  # [B, 4, 4]
            else:
                w2c = None
            memory_pe = self.world_pe(memory_pos, w2c=w2c)
            memory_with_pe = memory + memory_pe
        elif self.pe_type == 'world' and self.world_pe is not None:
            memory_pe = self.world_pe(memory_pos)
            memory_with_pe = memory + memory_pe
        elif self.pe_type == 'plucker' and self.plucker_pe is not None:
            if depths is not None and poses is not None and intrinsics is not None:
                if depths.dim() == 4:
                    depths_for_plucker = depths.squeeze(1).unsqueeze(1)
                else:
                    depths_for_plucker = depths.unsqueeze(1)
                poses_for_plucker = poses.unsqueeze(1)
                intrinsics_for_plucker = intrinsics.unsqueeze(1)
                H_pm, W_pm = pointmaps.shape[1:3]
                if depths_for_plucker.shape[2] != H_pm or depths_for_plucker.shape[3] != W_pm:
                    depths_for_plucker = F.interpolate(depths_for_plucker, size=(H_pm, W_pm), mode='bilinear', align_corners=False)
                plucker_pe = self.plucker_pe(depths_for_plucker.squeeze(1).unsqueeze(1), poses_for_plucker, intrinsics_for_plucker)
                plucker_pe_flat = plucker_pe.squeeze(1).view(B, -1, self.d_model)
                if plucker_pe_flat.shape[1] != L:
                    plucker_pe_flat = F.interpolate(
                        plucker_pe_flat.permute(0, 2, 1).unsqueeze(-1),
                        size=(L, 1), mode='bilinear', align_corners=False
                    ).squeeze(-1).permute(0, 2, 1)
                memory_pe = plucker_pe_flat
                memory_with_pe = memory + memory_pe
            else:
                memory_pe = None
                memory_with_pe = memory
        else:
            memory_pe = None
            memory_with_pe = memory

        # Run GASA decoder layers
        # --additive-pe (SAM3-style): pass PE separately, layers add to K only (V stays clean)
        # Legacy (no --additive-pe): PE baked into memory_with_pe, optionally --clean-v for V
        if self.additive_pe and memory_pe is not None:
            # SAM3-style: memory stays clean, PE added fresh to K each layer
            layer_memory = memory       # Clean visual features
            layer_memory_pe = memory_pe  # PE added to K at attention time
            memory_v = None              # Not needed — V uses clean memory automatically
        else:
            # Legacy: PE baked into memory
            layer_memory = memory_with_pe
            layer_memory_pe = None
            memory_v = memory if self.clean_v else None
        Q = queries_with_presence.shape[1]
        query_pos = memory_pos.mean(dim=1, keepdim=True).expand(-1, Q, -1).clone()

        # Query PE: learned positional embedding for queries (SAM3-style)
        # Same embedding added at every layer — gives queries persistent identity
        query_pe = None
        if self.use_query_pe:
            query_pe = self.query_pe_embed.weight[:Q].unsqueeze(0).expand(B, -1, -1)

        # Compute depth-conditioned spatial attention bias (if enabled)
        _spatial_attn_bias = None
        if self.use_spatial_attn_bias and spatial_qualifier_idx is not None and depths is not None:
            if (spatial_qualifier_idx > 0).any():
                H_attn = int(L ** 0.5)
                W_attn = H_attn  # Assuming square feature map
                _spatial_attn_bias = self.spatial_attn_bias(
                    spatial_qualifier_idx, depths, H_attn, W_attn
                )

        # Text-conditioned spatial bias (ViL3DRel-inspired)
        if self.use_text_spatial_bias and text_embedding is not None and depths is not None:
            H_attn = int(L ** 0.5)
            W_attn = H_attn
            # Use mean of text tokens as CLS representation
            text_cls = text_embedding.mean(dim=1)  # [B, D]
            text_spatial = self.text_spatial_bias(text_cls, depths, H_attn, W_attn)
            if _spatial_attn_bias is not None:
                _spatial_attn_bias = _spatial_attn_bias + text_spatial
            else:
                _spatial_attn_bias = text_spatial

        # Collect per-layer auxiliary outputs for intermediate supervision (SAM3-style)
        aux_layer_queries = []

        # Check if any layer uses image-to-token attention (returns updated memory)
        _has_img2tok = any(getattr(layer, 'use_image_to_token', False) for layer in self.layers)
        _has_box_rpb = any(getattr(layer, 'use_box_rpb', False) for layer in self.layers)

        # Initialize reference boxes for boxRPB (center of image, half-size)
        reference_boxes = None
        spatial_hw = None
        if _has_box_rpb:
            H_mem = int(L ** 0.5)
            spatial_hw = (H_mem, H_mem)
            # Initial box: centered, covering most of image
            reference_boxes = torch.tensor([0.5, 0.5, 0.8, 0.8], device=memory.device)
            reference_boxes = reference_boxes.unsqueeze(0).unsqueeze(0).expand(B, Q, -1).clone()

        # Unified layer call with all optional features
        _layer_kwargs = dict(
            text_embedding=text_proj_for_layers,
            rayrope_ctx=rayrope_ctx,
            query_pe=query_pe,
            memory_v=memory_v,
            memory_pe=layer_memory_pe,
            text_attn_mask=text_attn_mask,
            spatial_attn_bias=_spatial_attn_bias,
            reference_boxes=reference_boxes,
            spatial_hw=spatial_hw,
        )

        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            use_return_pos = self.use_iterative_pos or is_last

            result = layer(
                queries_with_presence, layer_memory, memory_pos,
                query_pos=query_pos, return_query_pos=use_return_pos,
                **_layer_kwargs,
            )

            # Unpack: result is always a tuple. refined_boxes is always last.
            if not isinstance(result, tuple):
                queries_with_presence = result
            else:
                items = list(result)
                # Pop refined_boxes (last element, may be None)
                refined_b = items.pop()
                if refined_b is not None and _has_box_rpb:
                    reference_boxes = refined_b.detach()
                    _layer_kwargs['reference_boxes'] = reference_boxes

                if use_return_pos:
                    if _has_img2tok and len(items) >= 3:
                        queries_with_presence, query_pos, layer_memory = items[0], items[1], items[2]
                    else:
                        queries_with_presence, query_pos = items[0], items[1]
                else:
                    if _has_img2tok and len(items) >= 2:
                        queries_with_presence, layer_memory = items[0], items[1]
                    else:
                        queries_with_presence = items[0]

            aux_layer_queries.append(queries_with_presence)

        queries_with_presence = self.norm(queries_with_presence)

        # Separate presence token from queries
        if self.use_presence_token:
            presence_out = queries_with_presence[:, 0:1, :]  # [B, 1, D]
            queries = queries_with_presence[:, 1:, :]  # [B, Q, D]
            # Predict presence logit
            presence_logit = self.presence_head(self.presence_norm(presence_out)).squeeze(-1)  # [B, 1]
            presence_logit = presence_logit.clamp(-10.0, 10.0)  # SAM3-style clamping
        else:
            queries = queries_with_presence
            presence_logit = None

        # CENTROID HEAD: Predict 3D centroid using attention-weighted 3D positions
        # Each query's centroid = weighted average of 3D points it attends to
        centroid_pred = None
        per_query_centroids = None
        if self.use_centroid_head:
            # query_pos contains attention-weighted 3D positions for each query [B, Q+1, 3] or [B, Q, 3]
            if self.use_presence_token:
                # Skip presence token, get query centroids
                per_query_centroids = query_pos[:, 1:, :]  # [B, Q, 3]
            else:
                per_query_centroids = query_pos  # [B, Q, 3]

            # Optional: refine centroids with learnable head (residual correction)
            # This allows learning to adjust for systematic biases
            query_normed = self.centroid_norm(queries)
            centroid_offset = self.centroid_head(query_normed)  # [B, Q, 3]
            per_query_centroids = per_query_centroids + centroid_offset

            # For backward compatibility, also output mean centroid
            centroid_pred = per_query_centroids.mean(dim=1)  # [B, 3]

        # IOU HEAD: Predict IoU with GT for each query mask
        # Enables zero-shot mask selection at inference time
        iou_pred = None
        if self.use_iou_head:
            iou_pred = self.iou_head(queries).squeeze(-1)  # [B, Q]

        # TEXT SCORING: SAM3-style dot-product scoring for text-aware mask selection
        # Order matches SAM3: MLP → mean_pool → project → dot product with projected queries
        # IMPORTANT: Use raw text_embedding (256-dim from VETextEncoder), NOT text_proj_for_layers.
        # SAM3's DotProductScoring.prompt_mlp receives raw text tokens directly.
        # text_proj is only for cross-attention conditioning, not for scoring.
        text_scores = None
        if text_embedding is not None:
            # Apply 2-layer MLP to RAW text tokens BEFORE pooling (SAM3-style)
            text_processed = self.scoring_prompt_mlp(text_embedding)  # [B, T, D] or [B, K*T, D]
            proj_queries = self.scoring_hs_proj(queries)  # [B, Q, D]

            if num_texts > 1 and tokens_per_text is not None:
                # MULTI-OBJECT: Per-text pooling → [B, Q, K] scores
                K = num_texts
                T_per = tokens_per_text
                D = text_processed.shape[-1]
                text_per_obj = text_processed.view(B, K, T_per, D)  # [B, K, T, D]
                pooled_per_text = text_per_obj.mean(dim=2)  # [B, K, D]
                proj_per_text = self.scoring_prompt_proj(pooled_per_text)  # [B, K, D]
                # Score each query against each text: [B, Q, D] @ [B, D, K] -> [B, Q, K]
                text_scores = torch.bmm(proj_queries, proj_per_text.transpose(1, 2))  # [B, Q, K]
                text_scores = text_scores * self.scoring_scale
                text_scores = text_scores.clamp(-12.0, 12.0)
            else:
                # SINGLE-OBJECT: Global pool → [B, Q] scores (original behavior)
                pooled_text = text_processed.mean(dim=1)  # [B, D]
                proj_text = self.scoring_prompt_proj(pooled_text)  # [B, D]
                text_scores = torch.matmul(proj_queries, proj_text.unsqueeze(-1)).squeeze(-1)  # [B, Q]
                text_scores = text_scores * self.scoring_scale
                text_scores = text_scores.clamp(-12.0, 12.0)
        else:
            # Still connect scoring params to graph for DDP (zero contribution)
            dummy = self.scoring_prompt_proj.weight.sum() * 0.0 + self.scoring_hs_proj.weight.sum() * 0.0
            for p in self.scoring_prompt_mlp.parameters():
                dummy = dummy + p.sum() * 0.0
            text_scores = torch.zeros(B, self.num_queries, device=device) + dummy

        # JOINT SCORING: Combine text scores with presence probability (SAM3-style)
        # This suppresses queries when the object isn't present in the image
        if presence_logit is not None and text_scores is not None:
            presence_prob = presence_logit.sigmoid()  # [B, 1]
            if text_scores.dim() == 3:
                # Multi-object: text_scores [B, Q, K], presence_prob [B, 1] -> [B, Q, K]
                joint_scores = text_scores.sigmoid() * presence_prob.unsqueeze(-1)
            else:
                # Single-object: text_scores [B, Q], presence_prob [B, 1] -> [B, Q]
                joint_scores = text_scores.sigmoid() * presence_prob
            # Convert back to logit space for compatibility (SAM3-style: eps=1e-3, clamp [-10, 10])
            joint_scores = torch.log(joint_scores.clamp(min=1e-3) / (1 - joint_scores.clamp(max=1-1e-3)))
            joint_scores = joint_scores.clamp(-10.0, 10.0)
        else:
            joint_scores = text_scores

        # DDP GRADIENT SYNC FIX: Ensure ALL conditional parameters receive gradients
        # This prevents "unused parameters" errors when using batched views
        # We add a zero-contribution term that touches params in conditional blocks
        ddp_fix = torch.zeros(1, device=queries.device, dtype=queries.dtype)

        # Text conditioning params (might be skipped if text_embedding is None)
        for p in self.text_proj.parameters():
            ddp_fix = ddp_fix + p.sum() * 0.0
        ddp_fix = ddp_fix + self.text_norm.weight.sum() * 0.0
        # text_cross_attn params
        ddp_fix = ddp_fix + self.text_cross_attn.in_proj_weight.sum() * 0.0

        # World PE params (used when pe_type='world')
        if hasattr(self, 'world_pe') and self.world_pe is not None:
            for p in self.world_pe.parameters():
                ddp_fix = ddp_fix + p.sum() * 0.0

        # Plucker PE params (used when pe_type='plucker')
        if hasattr(self, 'plucker_pe') and self.plucker_pe is not None:
            for p in self.plucker_pe.parameters():
                ddp_fix = ddp_fix + p.sum() * 0.0

        # RayRoPE params (used when pe_type='rayrope')
        # RayRoPE has no learnable params (only buffers), but include for future extensions
        if hasattr(self, 'rayrope') and self.rayrope is not None:
            for p in self.rayrope.parameters():
                ddp_fix = ddp_fix + p.sum() * 0.0

        # Box prompt params (might be skipped if box_prompts is None)
        if self.use_box_prompts and hasattr(self, 'box_cross_attn'):
            ddp_fix = ddp_fix + self.box_cross_attn.in_proj_weight.sum() * 0.0
            ddp_fix = ddp_fix + self.box_norm.weight.sum() * 0.0

        # Point prompt params (might be skipped if point_prompts is None)
        if self.use_point_prompts and hasattr(self, 'point_cross_attn'):
            ddp_fix = ddp_fix + self.point_cross_attn.in_proj_weight.sum() * 0.0
            ddp_fix = ddp_fix + self.point_norm.weight.sum() * 0.0

        # Text scoring params (always used but connect anyway for safety)
        ddp_fix = ddp_fix + self.scoring_prompt_proj.weight.sum() * 0.0
        ddp_fix = ddp_fix + self.scoring_hs_proj.weight.sum() * 0.0
        for p in self.scoring_prompt_mlp.parameters():
            ddp_fix = ddp_fix + p.sum() * 0.0

        # Add to queries to connect to output (zero contribution)
        queries = queries + ddp_fix.view(1, 1, 1).expand_as(queries) * 0.0

        # Cache text-side projections for per-layer auxiliary scoring
        # (text side is same for all layers, only query side changes)
        if text_embedding is not None:
            self._cached_text_processed = text_processed
            self._cached_num_texts = num_texts
            self._cached_tokens_per_text = tokens_per_text
        else:
            self._cached_text_processed = None

        # Extract auxiliary per-layer queries (for per-layer align loss)
        # aux_layer_queries contains pre-norm outputs from each layer
        # We need to norm + separate presence token for each
        aux_outputs = None
        if len(aux_layer_queries) > 1:  # Only if >1 layer (skip last = final)
            aux_outputs = []
            for layer_q in aux_layer_queries[:-1]:  # Exclude last (= final output)
                normed = self.norm(layer_q)
                if self.use_presence_token:
                    layer_queries = normed[:, 1:, :]  # Skip presence token
                else:
                    layer_queries = normed
                aux_outputs.append(layer_queries)

        return queries, presence_logit, centroid_pred, iou_pred, per_query_centroids, text_scores, joint_scores, aux_outputs

    def compute_scores_for_queries(self, queries):
        """Compute text scores for arbitrary queries using cached text projections.

        Used for per-layer auxiliary align loss: reuse the text-side projection
        (computed once in forward) but project queries from each intermediate layer.

        Args:
            queries: [B, Q, D] query embeddings from any decoder layer

        Returns:
            text_scores: [B, Q] or [B, Q, K] scores, or None if no text cached
        """
        if self._cached_text_processed is None:
            return None

        text_processed = self._cached_text_processed
        num_texts = self._cached_num_texts
        tokens_per_text = self._cached_tokens_per_text

        proj_queries = self.scoring_hs_proj(queries)  # [B, Q, D]

        if num_texts > 1 and tokens_per_text is not None:
            B = queries.shape[0]
            K = num_texts
            T_per = tokens_per_text
            D = text_processed.shape[-1]
            text_per_obj = text_processed.view(B, K, T_per, D)
            pooled_per_text = text_per_obj.mean(dim=2)
            proj_per_text = self.scoring_prompt_proj(pooled_per_text)
            text_scores = torch.bmm(proj_queries, proj_per_text.transpose(1, 2))
            text_scores = text_scores * self.scoring_scale
            text_scores = text_scores.clamp(-12.0, 12.0)
        else:
            pooled_text = text_processed.mean(dim=1)
            proj_text = self.scoring_prompt_proj(pooled_text)
            text_scores = torch.matmul(proj_queries, proj_text.unsqueeze(-1)).squeeze(-1)
            text_scores = text_scores * self.scoring_scale
            text_scores = text_scores.clamp(-12.0, 12.0)

        return text_scores


