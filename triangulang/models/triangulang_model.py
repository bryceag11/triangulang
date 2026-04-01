"""
TrianguLangModel - Main model for geometry-aware multi-view segmentation.

Architecture:
    1. SAM3 backbone + encoder (FROZEN) -> encoder_memory
    2. DA3 (FROZEN) -> depth -> pointmaps
    3. GASA Decoder (TRAINABLE) -> object queries
    4. SAM3 seghead -> masks
"""

import math
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np

from sam3 import build_sam3_image_model
from sam3.model.geometry_encoders import Prompt
from sam3.model.data_misc import FindStage

from depth_anything_3.api import DepthAnything3

from triangulang.models.gasa import (
    PointmapComputer,
    WorldSpacePositionalEncoding,
    CameraRelativePositionalEncoding,
    PluckerEmbedding,
    RayRoPE3D,
)
from triangulang.models.gasa_decoder import GASADecoder, MaskRefiner
from triangulang import BPE_PATH as _BPE_PATH

from triangulang.utils.spatial_reasoning import (
    spatial_to_pseudo_point_tensor,
)

import triangulang.models.model_utils as _mu


class TrianguLangModel(nn.Module):
    def __init__(self, sam3_model, da3_model, d_model: int = 256, n_heads: int = 8,
                 num_decoder_layers: int = 6, num_queries: int = 100, train_seghead: bool = False,
                 attn_map_size: int = 64, use_presence_token: bool = True, use_box_prompts: bool = True,
                 use_point_prompts: bool = True, mask_selection: str = 'iou_match',
                 use_world_pe: bool = True, use_gasa: bool = True, use_centroid_head: bool = False,
                 box_prompt_dropout: float = 0.0, point_prompt_dropout: float = 0.0,
                 num_pos_points: int = 10, num_neg_points: int = 2, use_iterative_pos: bool = False,
                 cross_view: bool = True, pe_type: str = 'world', da3_model_name: str = 'DA3METRIC-LARGE',
                 use_iou_head: bool = False, use_spatial_tokens: bool = False,
                 use_spatial_points: bool = False, use_object_aware_spatial: bool = False,
                 da3_resolution: int = 504,
                 pointmap_normalize: bool = True, resolution: int = 1008, gasa_beta_init: float = 1.0,
                 use_da3_poses_for_gasa: bool = False, use_gt_poses_for_gasa: bool = False,
                 sheaf_use_gt_poses: bool = False,
                 gasa_kernel_dim: int = 32,
                 gasa_fixed_kernel: bool = False, gasa_kernel_type: str = 'learned',
                 use_depth_crossattn: bool = False,
                 per_layer_text: bool = False, pred_logits_source: str = 'mask_mean',
                 gasa_bidirectional: bool = False,
                 query_proj_mlp: bool = False, no_query_proj: bool = False, train_mask_embed: bool = False,
                 use_mask_refiner: bool = False,
                 dim_feedforward: int = 2048, post_norm: bool = True,
                 use_query_pe: bool = False, ffn_fp32: bool = True,
                 no_initial_text: bool = False, no_text_proj: bool = False,
                 clean_v: bool = False, additive_pe: bool = False,
                 grouped_text_attn: bool = False,
                 per_text_decode: bool = False,
                 use_spatial_attn_bias: bool = False,
                 use_text_spatial_bias: bool = False,
                 use_image_to_token: bool = False,
                 use_pos_refine: bool = False,
                 use_box_rpb: bool = False):
        super().__init__()
        self.per_text_decode = per_text_decode
        self.sam3 = sam3_model
        self.da3 = da3_model
        self.d_model = d_model
        self.use_da3_poses_for_gasa = use_da3_poses_for_gasa  # Use DA3-NESTED poses for world-frame GASA
        self.use_gt_poses_for_gasa = use_gt_poses_for_gasa  # Use GT COLMAP poses for world-frame GASA
        self.sheaf_use_gt_poses = sheaf_use_gt_poses  # Force GT poses for sheaf world pointmaps
        self.resolution = resolution  # Must match img_size passed to build_sam3_image_model
        self.da3_resolution = da3_resolution
        self.attn_map_size = attn_map_size
        self.use_presence_token = use_presence_token
        self.use_box_prompts = use_box_prompts
        self.use_point_prompts = use_point_prompts  # MV-SAM style click prompts
        self.use_world_pe = use_world_pe
        self.use_gasa = use_gasa  # Ablation: disable geometric bias
        self.use_centroid_head = use_centroid_head  # 3D localization output
        self.use_iou_head = use_iou_head  # IoU prediction for zero-shot mask selection
        self.use_spatial_tokens = use_spatial_tokens  # Spatial qualifier embeddings
        self.use_spatial_points = use_spatial_points  # Convert spatial to pseudo-points
        self.use_object_aware_spatial = use_object_aware_spatial  # Object-aware spatial selection
        self.pointmap_normalize = pointmap_normalize  # Whether to normalize pointmaps (disable for accurate Acc@m)
        self.mask_selection = mask_selection
        self.pred_logits_source = pred_logits_source  # 'mask_mean' or 'text_scoring'
        self.topk_masks = 5  # Legacy default for majority_vote
        self.box_prompt_dropout = box_prompt_dropout  # Randomly drop box prompts during training
        self.point_prompt_dropout = point_prompt_dropout  # Randomly drop point prompts during training
        self.num_pos_points = num_pos_points  # Number of positive points per image (default 10)
        self.num_neg_points = num_neg_points  # Number of negative points per image (default 2)
        self.prompt_type = 'all'  # Can be set to 'random' for per-batch random prompt selection
        self.cross_view = cross_view  # Ablation: cross-view vs single-view attention
        self.pe_type = pe_type  # Ablation: 'world', 'camera_relative', 'plucker', 'rayrope', or 'none'
        self.profile = False  # Enable profiling with set_profile(True)
        self._profile_times = {}  # Accumulated times per component

        # DA3 model capabilities - determines pose estimation behavior
        # DA3METRIC-LARGE: monocular only, no pose estimation, metric depth
        # DA3-LARGE: multi-view, pose estimation, relative depth
        # DA3NESTED-GIANT-LARGE: multi-view, pose estimation, metric depth
        self.da3_model_name = da3_model_name.upper()
        self.da3_has_pose_estimation = 'METRIC' not in self.da3_model_name and 'MONO' not in self.da3_model_name

        # Freeze DA3
        if self.da3 is not None:
            for p in self.da3.parameters():
                p.requires_grad_(False)
        if self.da3 is not None:
            self.da3.eval()

        # Freeze SAM3 backbone + encoder + all other components
        for p in self.sam3.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sam3.transformer.encoder.parameters():
            p.requires_grad_(False)
        for p in self.sam3.geometry_encoder.parameters():
            p.requires_grad_(False)
        for p in self.sam3.transformer.decoder.parameters():
            p.requires_grad_(False)
        # Also freeze dot_prod_scoring
        # This module has ~1.1M params that were being fine-tuned unintentionally
        if hasattr(self.sam3, 'dot_prod_scoring'):
            for p in self.sam3.dot_prod_scoring.parameters():
                p.requires_grad_(False)

        # Seghead: optionally trainable
        for p in self.sam3.segmentation_head.parameters():
            p.requires_grad_(train_seghead)

        # Pointmap computer
        self.pointmap_computer = PointmapComputer()

        # GASA Decoder (TRAINABLE - replaces SAM3's decoder)
        self.gasa_decoder = GASADecoder(
            d_model=d_model, n_heads=n_heads,
            num_layers=num_decoder_layers, num_queries=num_queries,
            use_presence_token=use_presence_token, use_box_prompts=use_box_prompts,
            use_point_prompts=use_point_prompts, use_world_pe=use_world_pe,
            use_gasa=use_gasa, use_centroid_head=use_centroid_head,
            use_iterative_pos=use_iterative_pos, cross_view=cross_view,
            pe_type=pe_type, use_iou_head=use_iou_head,
            use_spatial_tokens=use_spatial_tokens, gasa_beta_init=gasa_beta_init,
            gasa_kernel_dim=gasa_kernel_dim, gasa_fixed_kernel=gasa_fixed_kernel,
            gasa_kernel_type=gasa_kernel_type,
            use_depth_crossattn=use_depth_crossattn, per_layer_text=per_layer_text,
            gasa_bidirectional=gasa_bidirectional,
            dim_feedforward=dim_feedforward, post_norm=post_norm,
            use_query_pe=use_query_pe, ffn_fp32=ffn_fp32,
            no_initial_text=no_initial_text,
            no_text_proj=no_text_proj,
            clean_v=clean_v,
            additive_pe=additive_pe,
            grouped_text_attn=grouped_text_attn,
            use_spatial_attn_bias=use_spatial_attn_bias,
            use_text_spatial_bias=use_text_spatial_bias,
            use_image_to_token=use_image_to_token,
            use_pos_refine=use_pos_refine,
            use_box_rpb=use_box_rpb,
        )

        # Project to SAM3's expected dimension
        self.query_proj_mlp = query_proj_mlp
        self.no_query_proj = no_query_proj
        if no_query_proj:
            # No projection — decoder output goes directly to mask_embed (like SAM3)
            self.query_proj = nn.Identity()
        elif query_proj_mlp:
            # 3-layer MLP matching mask_embed's structure for better distribution bridging
            self.query_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 256),
            )
        else:
            self.query_proj = nn.Linear(d_model, 256)

        # Optionally unfreeze mask_embed to adapt to GASA's query distribution
        self.train_mask_embed = train_mask_embed
        if train_mask_embed:
            for p in self.sam3.segmentation_head.mask_predictor.mask_embed.parameters():
                p.requires_grad_(True)

        # Mask refinement head for sharper boundaries
        self.use_mask_refiner = use_mask_refiner
        if use_mask_refiner:
            self.mask_refiner = MaskRefiner(in_channels=1, img_channels=3, hidden_dim=32)


    def set_profile(self, e): _mu.set_profile(self, e)
    def _profile_start(self): return _mu._profile_start(self)
    def _profile_end(self, n, t): _mu._profile_end(self, n, t)
    def get_profile_summary(self): return _mu.get_profile_summary(self)
    @staticmethod
    def mask_to_box(mask, jitter_ratio=0.05, expand_ratio=0.1): return _mu.mask_to_box(mask, jitter_ratio, expand_ratio)
    @staticmethod
    def mask_to_box_batched(masks, jitter_ratio=0.05, expand_ratio=0.1): return _mu.mask_to_box_batched(masks, jitter_ratio, expand_ratio)
    @staticmethod
    def sample_points_from_mask_batched(masks, num_positive=10, num_negative=2, jitter_ratio=0.02): return _mu.sample_points_from_mask_batched(masks, num_positive, num_negative, jitter_ratio)
    @staticmethod
    def sample_points_from_mask(mask, num_positive=10, num_negative=2, jitter_ratio=0.02): return _mu.sample_points_from_mask(mask, num_positive, num_negative, jitter_ratio)
    def select_mask_by_confidence(self, outputs, logits=None, presence_logit=None): return _mu.select_mask_by_confidence(self, outputs, logits, presence_logit)
    def select_mask_by_iou(self, outputs, gt_mask): return _mu.select_mask_by_iou(self, outputs, gt_mask)
    def select_mask_by_majority_vote(self, outputs, k=5): return _mu.select_mask_by_majority_vote(self, outputs, k)
    def select_mask_by_predicted_iou(self, outputs, pred_ious): return _mu.select_mask_by_predicted_iou(self, outputs, pred_ious)
    def select_mask_by_spatial(self, outputs, depth, spatial_idx, gt_mask=None, fallback="iou"): return _mu.select_mask_by_spatial(self, outputs, depth, spatial_idx, gt_mask, fallback)
    @torch.no_grad()
    def get_depth_and_pose(self, images): return _mu.get_depth_and_pose(self, images)
    @torch.no_grad()
    def precompute_sam3_features(self, images, text_prompt): return _mu.precompute_sam3_features(self, images, text_prompt)


    def forward(self, images, text_prompts, gt_masks=None,
                box_prompts=None, box_labels=None, point_prompts=None, point_labels=None,
                gt_extrinsics=None, gt_intrinsics=None, spatial_qualifier_idx=None,
                intrinsics_orig_hw=None, cached_depth=None,
                da3_extrinsics=None, da3_intrinsics=None,
                cross_view_mode=False, num_texts=1,
                precomputed_sam3=None):
        # Cross-view mode: dispatch to forward_multiview() for true multi-view attention.
        # This path goes through DDP's __call__ hooks, ensuring proper gradient sync.
        if cross_view_mode:
            return self.forward_multiview(
                images, text_prompts, gt_masks,
                gt_extrinsics=gt_extrinsics, gt_intrinsics=gt_intrinsics,
                intrinsics_orig_hw=intrinsics_orig_hw,
                cached_depth=cached_depth,
                point_prompts=point_prompts, point_labels=point_labels,
                box_prompts=box_prompts, box_labels=box_labels,
                da3_extrinsics=da3_extrinsics, da3_intrinsics=da3_intrinsics
            )

        B = images.shape[0]
        device = images.device

        # Resize to SAM3 resolution
        if images.shape[-2:] != (self.resolution, self.resolution):
            sam3_images = F.interpolate(images, size=(self.resolution, self.resolution),
                                        mode='bilinear', align_corners=False)
        else:
            sam3_images = images

        # 1. Get depth and camera parameters from DA3 (or use cached depth)
        t0 = self._profile_start()
        if self.da3 is None:
            # No DA3 model loaded (no GASA, no PE, no centroid) — skip depth entirely
            depth = None
            da3_pose = None
            da3_intrinsics = None
            self._profile_end("1_DA3_skipped", t0)
        elif cached_depth is not None:
            # Use pre-computed depth (bypasses DA3 for 2-4x speedup)
            depth = cached_depth.to(device=device, dtype=sam3_images.dtype)
            # Upsample cached depth to SAM3 resolution if needed
            if depth.shape[-2:] != (self.resolution, self.resolution):
                depth = F.interpolate(depth, size=(self.resolution, self.resolution),
                                      mode='bilinear', align_corners=False)
            da3_pose = None
            # Keep da3_intrinsics if passed (from cache), otherwise will use fallback
            # Note: da3_intrinsics parameter is used if provided, else falls back to default
            self._profile_end("1_DA3_cached", t0)
        else:
            # Run DA3 live
            depth, da3_pose, da3_intrinsics = self.get_depth_and_pose(sam3_images)
            self._profile_end("1_DA3", t0)

        if depth is not None:
            # Use GT intrinsics if provided, otherwise use DA3 estimates
            # When using cached depth without GT intrinsics, create default intrinsics
            if gt_intrinsics is not None:
                intrinsics = gt_intrinsics.to(device=device, dtype=depth.dtype)
            elif da3_intrinsics is not None:
                intrinsics = da3_intrinsics
            else:
                # Fallback: create default intrinsics (assumes square image, centered principal point)
                # This path is taken when using cached_depth without GT intrinsics
                focal = self.resolution * 0.8  # Reasonable default focal length
                cx, cy = self.resolution / 2, self.resolution / 2
                intrinsics = torch.tensor([
                    [focal, 0, cx],
                    [0, focal, cy],
                    [0, 0, 1]
                ], device=device, dtype=depth.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    
            # Apply focal/300 scaling for DA3METRIC-LARGE to get metric depth in meters
            # DA3METRIC outputs "canonical depth" calibrated for focal=300 pixels
            # Formula: depth_meters = raw_depth * (focal_at_da3_res / 300)
            # See: https://github.com/ByteDance-Seed/Depth-Anything-3 FAQ
            # Note: DA3NESTED-GIANT-LARGE outputs meters directly, no scaling needed
            if 'METRIC' in self.da3_model_name and 'NESTED' not in self.da3_model_name:
                # Get focal length from intrinsics (GT or estimated)
                focal = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2  # [B]
    
                # Determine what resolution the intrinsics are at
                da3_res = (self.da3_resolution // 14) * 14  # Actual DA3 resolution (patch-aligned)
    
                if gt_intrinsics is not None and intrinsics_orig_hw is not None:
                    # GT intrinsics are at original resolution, scale to DA3 resolution
                    orig_h, orig_w = intrinsics_orig_hw
                    # Use height for scaling (consistent with test_da3_depth.py)
                    focal_at_da3 = focal * (da3_res / orig_h)
                else:
                    # Estimated intrinsics are at SAM3 resolution, scale to DA3 resolution
                    focal_at_da3 = focal * (da3_res / self.resolution)
    
                # Apply depth scaling: depth_meters = raw_depth * (focal / 300)
                depth_scale = (focal_at_da3 / 300.0).view(B, 1, 1, 1)  # [B, 1, 1, 1]
                depth = depth * depth_scale
    
            # Scale intrinsics from original resolution to depth resolution for pointmap computation
            # NOTE: depth is upsampled to SAM3 resolution (self.resolution) in run_da3()
            # So intrinsics must also be scaled to SAM3 resolution, not DA3 resolution
            depth_h, depth_w = depth.shape[-2:]
            if gt_intrinsics is not None and intrinsics_orig_hw is not None:
                orig_h, orig_w = intrinsics_orig_hw
                scale_x = depth_w / orig_w
                scale_y = depth_h / orig_h
                # Scale intrinsics: fx, cx scale by x; fy, cy scale by y
                intrinsics_scaled = intrinsics.clone()
                intrinsics_scaled[:, 0, 0] = intrinsics[:, 0, 0] * scale_x  # fx scales with width
                intrinsics_scaled[:, 1, 1] = intrinsics[:, 1, 1] * scale_y  # fy scales with height
                intrinsics_scaled[:, 0, 2] = intrinsics[:, 0, 2] * scale_x  # cx scales with width
                intrinsics_scaled[:, 1, 2] = intrinsics[:, 1, 2] * scale_y  # cy scales with height
            elif da3_intrinsics is not None and cached_depth is not None:
                # DA3 cached intrinsics: dataloader already scales intrinsics when resizing depth
                # (see scannetpp_loader.py). This block handles any additional resize from
                # cached_depth resolution to self.resolution (e.g., if cache is 504 but model wants 1008).
                # With proper dataloader fix, scale should be 1.0 (no additional scaling needed).
                cache_h = cached_depth.shape[-2]
                cache_w = cached_depth.shape[-1]
                scale_x = depth_w / cache_w
                scale_y = depth_h / cache_h
                intrinsics_scaled = intrinsics.clone()
                intrinsics_scaled[:, 0, 0] = intrinsics[:, 0, 0] * scale_x  # fx
                intrinsics_scaled[:, 1, 1] = intrinsics[:, 1, 1] * scale_y  # fy
                intrinsics_scaled[:, 0, 2] = intrinsics[:, 0, 2] * scale_x  # cx
                intrinsics_scaled[:, 1, 2] = intrinsics[:, 1, 2] * scale_y  # cy
                # Debug: log the scaling (should be 1.00 if dataloader handled it)
                if not hasattr(self, '_intrinsics_scale_logged'):
                    self._intrinsics_scale_logged = True
                    import torch.distributed as _dist
                    if not _dist.is_initialized() or _dist.get_rank() == 0:
                        print(f"[Intrinsics] Scaled from depth input ({cache_w}×{cache_h}) to model res ({depth_w}×{depth_h}): scale_x={scale_x:.2f}, scale_y={scale_y:.2f}")
            else:
                # DA3 estimated intrinsics from live inference are already at SAM3 res
                intrinsics_scaled = intrinsics.clone()
    
            # Compute pointmaps for GASA
            # Option 1 (default): Camera-frame (identity pose) - train/eval consistent
            # Option 2 (--use-da3-poses-for-gasa): World-frame using DA3-NESTED estimated poses
            # Always define identity_pose (needed for Plucker PE in GASA decoder)
            identity_pose = torch.eye(4, device=device, dtype=depth.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    
            if self.use_gt_poses_for_gasa and gt_extrinsics is not None:
                # Use GT COLMAP poses for world-frame GASA
                # Globally consistent across ALL views (no chunk boundary issues)
                gasa_pose = gt_extrinsics.to(device=device, dtype=depth.dtype)
                pointmaps, norm_params = self.pointmap_computer(depth, gasa_pose, intrinsics_scaled, normalize=self.pointmap_normalize)
                pointmaps = pointmaps.squeeze(1)
                self._using_world_frame_gasa = True
            elif self.use_da3_poses_for_gasa and da3_extrinsics is not None:
                # Use DA3-NESTED estimated poses for world-frame GASA
                # This enables cross-view consistency with train/eval consistent estimated poses
                gasa_pose = da3_extrinsics.to(device=device, dtype=depth.dtype)
                pointmaps, norm_params = self.pointmap_computer(depth, gasa_pose, intrinsics_scaled, normalize=self.pointmap_normalize)
                pointmaps = pointmaps.squeeze(1)
                self._using_world_frame_gasa = True
            else:
                # Default: Camera-frame (identity pose) - ensures train/eval consistency
                pointmaps, norm_params = self.pointmap_computer(depth, identity_pose, intrinsics_scaled, normalize=self.pointmap_normalize)
                pointmaps = pointmaps.squeeze(1)
                self._using_world_frame_gasa = False
    
            # For sheaf loss (training only), compute world-frame pointmaps separately
            # Depth and poses must be from compatible coordinate systems:
            # - DA3-NESTED depth + DA3-NESTED poses = consistent (same processing chunk)
            # - DA3METRIC depth + GT poses = consistent (metric depth is in camera frame, GT pose = c2w)
            # - Any cached depth + GT poses = valid if GT poses are camera-to-world
            # Priority (default): 1) cached DA3 poses, 2) DA3 live poses, 3) GT poses (metric only)
            # With --sheaf-use-gt-poses: GT poses first (needed for stratified sampling across chunks)
            # IMPORTANT: normalize=False for sheaf loss! Each view must stay in shared world frame.
            world_pointmaps = None
            if not hasattr(self, '_world_pm_debug_logged'):
                import torch.distributed as _dist
                # Only log on rank 0 to avoid DDP spam
                self._world_pm_debug_logged = _dist.is_initialized() and _dist.get_rank() != 0
            if self.sheaf_use_gt_poses and gt_extrinsics is not None:
                # --sheaf-use-gt-poses: Force GT poses for sheaf world pointmaps.
                # Works with any depth source (cached DA3-NESTED or DA3METRIC).
                # GT poses from COLMAP are consistent across ALL views (no chunk issues).
                if not self._world_pm_debug_logged:
                    print(f"[World PM] Using GT poses (--sheaf-use-gt-poses) → world-frame pointmaps")
                    self._world_pm_debug_logged = True
                gt_pose = gt_extrinsics.to(device=device, dtype=depth.dtype)
                world_pointmaps, _ = self.pointmap_computer(depth, gt_pose, intrinsics_scaled, normalize=False)
                world_pointmaps = world_pointmaps.squeeze(1)
            elif da3_extrinsics is not None and cached_depth is not None:
                # Cached DA3-NESTED depth + cached DA3-NESTED poses - consistent coordinate system
                if not self._world_pm_debug_logged:
                    print(f"[World PM] Using cached DA3 poses (matches cached DA3 depth)")
                    self._world_pm_debug_logged = True
                da3_pose_cached = da3_extrinsics.to(device=device, dtype=depth.dtype)
                world_pointmaps, _ = self.pointmap_computer(depth, da3_pose_cached, intrinsics_scaled, normalize=False)
                world_pointmaps = world_pointmaps.squeeze(1)
            elif self.da3_has_pose_estimation and da3_pose is not None:
                if not self._world_pm_debug_logged:
                    print(f"[World PM] Using DA3 live poses")
                    self._world_pm_debug_logged = True
                world_pointmaps, _ = self.pointmap_computer(depth, da3_pose, intrinsics_scaled, normalize=False)
                world_pointmaps = world_pointmaps.squeeze(1)
            elif da3_extrinsics is not None:
                if not self._world_pm_debug_logged:
                    print(f"[World PM] WARNING: Using cached DA3-NESTED extrinsics - may have chunk alignment issues!")
                    self._world_pm_debug_logged = True
                da3_pose_cached = da3_extrinsics.to(device=device, dtype=depth.dtype)
                world_pointmaps, _ = self.pointmap_computer(depth, da3_pose_cached, intrinsics_scaled, normalize=False)
                world_pointmaps = world_pointmaps.squeeze(1)
            elif gt_extrinsics is not None and self.da3_model_name in ('DA3METRIC-LARGE', 'depth-anything/DA3METRIC-LARGE'):
                # DA3METRIC gives metric depth in camera frame. GT poses give camera-to-world.
                # Together they produce valid world-frame pointmaps.
                if not self._world_pm_debug_logged:
                    print(f"[World PM] Using GT poses + DA3METRIC metric depth → world-frame pointmaps")
                    self._world_pm_debug_logged = True
                gt_pose = gt_extrinsics.to(device=device, dtype=depth.dtype)
                world_pointmaps, _ = self.pointmap_computer(depth, gt_pose, intrinsics_scaled, normalize=False)
                world_pointmaps = world_pointmaps.squeeze(1)
            else:
                if not self._world_pm_debug_logged:
                    print(f"[World PM] NO POSES AVAILABLE - world_pointmaps will be None!")
                    self._world_pm_debug_logged = True
    
            # Downsample pointmaps for decoder (configurable via attn_map_size)
            pts = pointmaps.permute(0, 3, 1, 2)
            pts = F.adaptive_avg_pool2d(pts, (self.attn_map_size, self.attn_map_size))
            pointmaps_small = pts.permute(0, 2, 3, 1)
    
            # Also downsample world_pointmaps if computed (for sheaf loss)
            world_pointmaps_small = None
            if world_pointmaps is not None:
                wpts = world_pointmaps.permute(0, 3, 1, 2)
                wpts = F.adaptive_avg_pool2d(wpts, (self.attn_map_size, self.attn_map_size))
                world_pointmaps_small = wpts.permute(0, 2, 3, 1)

        else:
            # No depth (DA3 skipped) — set zero pointmaps for downstream code
            pointmaps_small = torch.zeros(B, self.attn_map_size, self.attn_map_size, 3, device=device)
            pointmaps = torch.zeros(B, self.resolution, self.resolution, 3, device=device)
            norm_params = None
            world_pointmaps = None
            world_pointmaps_small = None
            identity_pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
            intrinsics_scaled = None
            intrinsics = None

        # 2. Run SAM3 backbone and encoder (or use precomputed backbone)
        tokens_per_text = None  # Track for multi-object per-text scoring
        with torch.no_grad():
            if precomputed_sam3 is not None and 'backbone_fpn' in precomputed_sam3:
                # Use pre-computed backbone features (skip ViT, the expensive part)
                # Still run text encoder + SAM3 encoder (text-conditioned)
                backbone_out = {"img_batch_all_stages": sam3_images}
                backbone_out['backbone_fpn'] = [f.to(device) for f in precomputed_sam3['backbone_fpn']]
                backbone_out['vision_features'] = precomputed_sam3['vision_features'].to(device)
                backbone_out['vision_pos_enc'] = [p.to(device) for p in precomputed_sam3['vision_pos_enc']]
                sam2_out = precomputed_sam3.get('sam2_backbone_out')
                if sam2_out is not None:
                    backbone_out['sam2_backbone_out'] = {
                        'vision_features': sam2_out['vision_features'].to(device),
                        'vision_pos_enc': [p.to(device) for p in sam2_out['vision_pos_enc']],
                        'backbone_fpn': [f.to(device) for f in sam2_out['backbone_fpn']],
                    }
                else:
                    backbone_out['sam2_backbone_out'] = None
            else:
                t0 = self._profile_start()
                backbone_out = {"img_batch_all_stages": sam3_images}
                backbone_out.update(self.sam3.backbone.forward_image(sam3_images))
                self._profile_end("2_SAM3_backbone", t0)

            t0 = self._profile_start()
            # Deduplicate text prompts (SAM3-style): encode unique texts once, index back
            unique_texts = list(dict.fromkeys(text_prompts))  # preserves order, removes dups
            text_idx_map = [unique_texts.index(t) for t in text_prompts]  # map back to unique

            if num_texts > 1:
                # MULTI-OBJECT: Encode unique texts, then index to get K*B features
                multi_text_out = self.sam3.backbone.forward_text(unique_texts, device=device)
                unique_features = multi_text_out['language_features']  # [T, U, D] where U = len(unique_texts)
                T_tokens = unique_features.shape[0]
                tokens_per_text = T_tokens

                # Expand unique features back to K*B via index map
                idx_tensor = torch.tensor(text_idx_map, device=device, dtype=torch.long)
                multi_text_features = unique_features[:, idx_tensor, :]  # [T, K*B, D]

                # Extract primary text (first of K) for each batch item → [T, B, D]
                # Primary text indices: [0, K, 2K, ...] (stride by K through flat list)
                K = num_texts
                primary_indices = list(range(0, K * B, K))
                backbone_out['language_features'] = multi_text_features[:, primary_indices, :]  # [T, B, D]
                backbone_out['language_mask'] = multi_text_out['language_mask'][torch.tensor([text_idx_map[i] for i in primary_indices], device=device), :]  # [B, T]
                if 'text_embeds' in multi_text_out:
                    backbone_out['text_embeds'] = multi_text_out['text_embeds'][torch.tensor([text_idx_map[i] for i in primary_indices], device=device)]  # [B, D]
            else:
                text_out = self.sam3.backbone.forward_text(unique_texts, device=device)
                # Index back to B (handles duplicate prompts across batch items)
                idx_tensor = torch.tensor(text_idx_map, device=device, dtype=torch.long)
                backbone_out['language_features'] = text_out['language_features'][:, idx_tensor, :]
                backbone_out['language_mask'] = text_out['language_mask'][idx_tensor, :]
                if 'text_embeds' in text_out:
                    backbone_out['text_embeds'] = text_out['text_embeds'][idx_tensor]
            self._profile_end("3_SAM3_text", t0)

            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, B, 4, device=device),
                box_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
            )
            find_input = FindStage(
                img_ids=torch.arange(B, device=device, dtype=torch.long),
                text_ids=torch.arange(B, device=device, dtype=torch.long),
                input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                input_points=None, input_points_mask=None,
            )

            t0 = self._profile_start()
            prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
            backbone_out, encoder_out, _ = self.sam3._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
            self._profile_end("4_SAM3_encoder", t0)

        # 3. Get encoder memory
        encoder_memory = encoder_out["encoder_hidden_states"].transpose(0, 1)  # [B, L, D]

        # Get text embeddings

        # SAM3-STYLE MULTI-OBJECT: Expand batch by K after encoder
        # Backbone/encoder run once on [B, ...]. Then we expand:
        #   encoder_memory: [B, L, D] → [B*K, L, D]
        #   text_embedding: [K*B, T, D] → [B*K, T, D] (1 text per expanded item)
        #   All other tensors: repeat_interleave K times
        # GASA decoder then runs on [B*K, ...] with num_texts=1 per item.
        sam3_mo_mode = getattr(self, 'sam3_multi_object', False) and num_texts > 1

        if num_texts > 1 and sam3_mo_mode:
            # SAM3-style: expand batch by K, each item sees 1 text
            K = num_texts
            text_embedding = multi_text_features.transpose(0, 1)  # [K*B, T, D]
            # Already in (B, K) order: [v0_k0, v0_k1, ..., v0_kK, v1_k0, ...] → [B*K, T, D]
            # Each group of K entries corresponds to one view's K texts
            T_per = text_embedding.shape[1]
            D_text = text_embedding.shape[2]
            # text_embedding is already [B*K, T, D] in correct order (B groups of K texts)
            # Expand encoder memory: [B, L, D] → [B*K, L, D]
            encoder_memory = encoder_memory.repeat_interleave(K, dim=0)
            # Expand depth, pointmaps, extrinsics etc.
            if depth is not None:
                depth = depth.repeat_interleave(K, dim=0)
            if da3_extrinsics is not None:
                da3_extrinsics = da3_extrinsics.repeat_interleave(K, dim=0)
            if da3_intrinsics is not None:
                da3_intrinsics = da3_intrinsics.repeat_interleave(K, dim=0)
            if intrinsics_scaled is not None:
                intrinsics_scaled = intrinsics_scaled.repeat_interleave(K, dim=0)
            if gt_extrinsics is not None:
                gt_extrinsics = gt_extrinsics.repeat_interleave(K, dim=0)
            if gt_masks is not None:
                if gt_masks.dim() == 4 and gt_masks.shape[1] == K:
                    # Per-object GT [B, K, H, W] → [B*K, H, W]
                    gt_masks = gt_masks.reshape(-1, *gt_masks.shape[2:])
                else:
                    gt_masks = gt_masks.repeat_interleave(K, dim=0)
            if spatial_qualifier_idx is not None:
                # If already [B*K] (per-object from MO spatial augmentation), don't expand
                if spatial_qualifier_idx.shape[0] != B * K:
                    spatial_qualifier_idx = spatial_qualifier_idx.repeat_interleave(K, dim=0)
            # Expand pointmaps (computed before encoder)
            if pointmaps_small is not None:
                pointmaps_small = pointmaps_small.repeat_interleave(K, dim=0)
            if world_pointmaps_small is not None:
                world_pointmaps_small = world_pointmaps_small.repeat_interleave(K, dim=0)
            # NOTE: Do NOT expand backbone_fpn here — it's too large at high res.
            # Instead, we compute pixel_embed/instance_embeds from original FPN
            # and expand instance_embeds AFTER (much smaller). See segmentation head section below.
            # Expand identity_pose (fallback for decoder pose)
            identity_pose = identity_pose.repeat_interleave(K, dim=0)
            # Expand full-res pointmaps for centroid computation
            if pointmaps is not None:
                pointmaps = pointmaps.repeat_interleave(K, dim=0)
            # Store original B for training loop reshaping
            self._sam3_mo_B_orig = B
            self._sam3_mo_K = K
            B = B * K  # Update batch size for rest of forward
            num_texts = 1  # Each expanded item has 1 text
        elif num_texts > 1:
            # Original multi-object: reshape K*B text features → [B, K*T, D]
            text_embedding = multi_text_features.transpose(0, 1)  # [K*B, T, D]
            K = num_texts
            T_per = text_embedding.shape[1]
            D_text = text_embedding.shape[2]
            text_embedding = text_embedding.view(B, K, T_per, D_text).reshape(B, K * T_per, D_text)
        else:
            text_embedding = backbone_out.get('language_features', None)
            if text_embedding is not None:
                text_embedding = text_embedding.transpose(0, 1)  # [B, T, D]

        # 4. Handle box/point prompts
        # Priority: 1) Pre-computed prompts (for evaluation), 2) Extract from GT (training)
        final_box_prompts = box_prompts
        final_box_labels = box_labels
        final_point_prompts = point_prompts
        final_point_labels = point_labels

        # Only extract from GT if pre-computed prompts not provided
        if final_box_prompts is None and self.use_box_prompts and gt_masks is not None:
            use_box_this_forward = True

            if self.training and self.prompt_type == 'random':
                choice = random.choice(['text_only', 'text_box', 'text_point', 'all'])
                if choice in ['text_only', 'text_point']:
                    use_box_this_forward = False

            if use_box_this_forward and self.training and self.box_prompt_dropout > 0:
                if random.random() < self.box_prompt_dropout:  # Use Python random, no GPU sync
                    use_box_this_forward = False

            if use_box_this_forward:
                # Use batched version - no Python loops
                masks_2d = gt_masks[:, 0] if gt_masks.dim() == 4 else gt_masks  # [B, H, W]
                boxes = self.mask_to_box_batched(masks_2d, jitter_ratio=0.05, expand_ratio=0.1)  # [B, 4]
                final_box_prompts = boxes.unsqueeze(1)  # [B, 1, 4]
                final_box_labels = torch.ones(B, 1, device=device, dtype=torch.long)  # [B, 1]

        # 4b. Extract point/click prompts from GT masks (MV-SAM style)
        if final_point_prompts is None and self.use_point_prompts and gt_masks is not None:
            use_point_this_forward = True

            if self.training and self.prompt_type == 'random':
                choice = random.choice(['text_only', 'text_box', 'text_point', 'all'])
                if choice in ['text_only', 'text_box']:
                    use_point_this_forward = False

            if use_point_this_forward and self.training and self.point_prompt_dropout > 0:
                if random.random() < self.point_prompt_dropout:  # Use Python random, no GPU sync
                    use_point_this_forward = False

            if use_point_this_forward:
                # Use batched version - no Python loops
                masks_2d = gt_masks[:, 0] if gt_masks.dim() == 4 else gt_masks  # [B, H, W]
                final_point_prompts, final_point_labels = self.sample_points_from_mask_batched(
                    masks_2d, num_positive=self.num_pos_points, num_negative=self.num_neg_points
                )  # [B, N_points, 2], [B, N_points]

        # 4c. Spatial-as-pseudo-points: Convert spatial qualifiers to additional point prompts
        # This leverages existing point training without new architecture
        if self.use_spatial_points and spatial_qualifier_idx is not None:
            for i in range(B):
                sq_idx = spatial_qualifier_idx[i].item() if isinstance(spatial_qualifier_idx, torch.Tensor) else spatial_qualifier_idx[i]
                if sq_idx > 0:  # 0 = no spatial qualifier
                    # Map index to qualifier type
                    idx_to_type = {1: 'depth_min', 2: 'depth_max', 3: 'x_min', 4: 'x_max', 5: 'y_min', 6: 'y_max', 7: 'center'}
                    qualifier_type = idx_to_type.get(sq_idx)
                    if qualifier_type:
                        spatial_pts, spatial_lbls = spatial_to_pseudo_point_tensor(
                            qualifier_type, depth[i] if qualifier_type.startswith('depth') else None, device=device
                        )
                        if spatial_pts is not None:
                            # Append to existing point prompts or create new
                            if final_point_prompts is not None:
                                # Expand spatial point to match batch dim and concatenate
                                final_point_prompts = torch.cat([
                                    final_point_prompts,
                                    spatial_pts.unsqueeze(0).expand(B, -1, -1)
                                ], dim=1)
                                final_point_labels = torch.cat([
                                    final_point_labels,
                                    spatial_lbls.unsqueeze(0).expand(B, -1)
                                ], dim=1)
                            else:
                                final_point_prompts = spatial_pts.unsqueeze(0).expand(B, -1, -1)
                                final_point_labels = spatial_lbls.unsqueeze(0).expand(B, -1)
                            break  # Only add once (same for all batch items with same qualifier)

        # 5. Run GASA decoder (TRAINABLE - replaces SAM3's decoder)
        # Pass the actual pose used for pointmaps (not identity) so rayrope/camera_relative PE
        # can properly project to camera frame for SE(3)-invariant encoding.
        # For camera-frame pointmaps (identity pose), this is identity (correct).
        # For world-frame pointmaps (DA3/GT pose), this is the actual c2w pose.
        gasa_decoder_pose = identity_pose
        if self.use_gt_poses_for_gasa and gt_extrinsics is not None and depth is not None:
            gasa_decoder_pose = gt_extrinsics.to(device=device, dtype=depth.dtype)
        elif self.use_da3_poses_for_gasa and da3_extrinsics is not None and depth is not None:
            gasa_decoder_pose = da3_extrinsics.to(device=device, dtype=depth.dtype)

        # 5. Run GASA decoder + SAM3 segmentation head
        t0 = self._profile_start()

        # PER-TEXT DECODE: Process each text independently through the decoder
        # This eliminates cross-text query competition — each text gets ALL Q queries.
        # Equivalent to SAM3's per-text batch approach but batched for efficiency.
        if getattr(self, 'per_text_decode', False) and num_texts > 1 and tokens_per_text is not None:
            K = num_texts
            T_per = tokens_per_text
            D_text = text_embedding.shape[-1]

            # Reshape text: [B, K*T, D] → [B, K, T, D]
            text_per_obj = text_embedding.view(B, K, T_per, D_text)

            # Pre-compute instance embeds once (frozen SAM3 params, no grad accumulates)
            # NOTE: Do NOT use torch.no_grad() here — it breaks AMP GradScaler's inf checks
            fpn_features = backbone_out['backbone_fpn']
            pixel_embed = self.sam3.segmentation_head.pixel_decoder(fpn_features)
            instance_embeds = self.sam3.segmentation_head.instance_seg_head(pixel_embed)

            # Loop over texts one at a time using gradient checkpointing to save memory
            per_text_masks = []
            per_text_logits = []
            first_mask_preds = None
            grad_text_indices = set(range(K))  # All texts get gradients

            for k_idx in range(K):
                text_k = text_per_obj[:, k_idx]  # [B, T, D]

                sq_k = spatial_qualifier_idx if self.use_spatial_tokens and spatial_qualifier_idx is not None else None

                # Use gradient checkpointing to trade compute for memory
                def _decode_text(mem, pm, txt, dep, pos, intr, sq):
                    q, pres, cent, iou_p, pqc, ts, js, _ = self.gasa_decoder(
                        mem, pm, txt,
                        box_prompts=None, box_labels=None,
                        point_prompts=None, point_labels=None,
                        depths=dep, poses=pos, intrinsics=intr,
                        spatial_qualifier_idx=sq,
                        num_texts=1, tokens_per_text=None,
                    )
                    q = self.query_proj(q)
                    # Return auxiliary outputs as dummy sum for DDP graph connectivity
                    # (all unused head params must participate in backward pass)
                    aux_dummy = torch.tensor(0.0, device=q.device)
                    for aux in [pres, cent, iou_p, pqc, ts, js]:
                        if aux is not None:
                            aux_dummy = aux_dummy + aux.sum() * 0.0
                    return q, aux_dummy

                if self.training and K > 2:
                    queries_k, aux_dummy_k = torch.utils.checkpoint.checkpoint(
                        _decode_text,
                        encoder_memory, pointmaps_small, text_k,
                        depth, gasa_decoder_pose, intrinsics_scaled, sq_k,
                        use_reentrant=False,
                    )
                else:
                    queries_k, aux_dummy_k = _decode_text(
                        encoder_memory, pointmaps_small, text_k,
                        depth, gasa_decoder_pose, intrinsics_scaled, sq_k,
                    )

                masks_k = self.sam3.segmentation_head.mask_predictor(queries_k, instance_embeds)  # [B, Q, H, W]
                # Add dummy to first mask to keep aux heads in computation graph
                if k_idx == 0:
                    masks_k = masks_k + aux_dummy_k
                if k_idx == 0:
                    first_mask_preds = masks_k

                logits_k = masks_k.mean(dim=(-2, -1))  # [B, Q]
                if gt_masks is not None and gt_masks.dim() == 4 and gt_masks.shape[1] > k_idx:
                    gt_k = gt_masks[:, k_idx]  # [B, H, W]
                    if masks_k.shape[-2:] != gt_k.shape[-2:]:
                        masks_k_resized = F.interpolate(masks_k, size=gt_k.shape[-2:], mode='bilinear', align_corners=False)
                    else:
                        masks_k_resized = masks_k
                    best_mask_k, best_idx_k = self.select_mask_by_iou(masks_k_resized, gt_k)
                else:
                    best_mask_k, best_idx_k = self.select_mask_by_confidence(masks_k)
                per_text_masks.append(best_mask_k)  # [B, 1, H, W]
                per_text_logits.append(logits_k[torch.arange(B), best_idx_k])  # [B]

            # Stack: per_text_masks [B, K, H, W]
            pred_masks_stacked = torch.cat(per_text_masks, dim=1)  # [B, K, H, W]
            pred_logits = torch.stack(per_text_logits, dim=1)  # [B, K]

            pred_masks = per_text_masks[0]  # [B, 1, H, W]
            best_idx = torch.zeros(B, dtype=torch.long, device=device)
            mask_preds = first_mask_preds  # [B, Q, H, W]

            presence_logit = None
            centroid_pred = None
            iou_pred = None
            per_query_centroids = None
            text_scores = None
            joint_scores = None
            aux_queries_proj = None

            self._profile_end("5_GASA_decoder_per_text", t0)

            if pred_masks_stacked.shape[-2:] != images.shape[-2:]:
                pred_masks_stacked = F.interpolate(pred_masks_stacked, size=images.shape[-2:], mode='bilinear', align_corners=False)
            if pred_masks.shape[-2:] != images.shape[-2:]:
                pred_masks = F.interpolate(pred_masks, size=images.shape[-2:], mode='bilinear', align_corners=False)

            # Store which texts have gradients for loss computation
            grad_text_indices_list = sorted(grad_text_indices)

            outputs = {
                'pred_masks': pred_masks,
                'pred_logits': pred_logits,
                'all_masks': mask_preds,
                'per_text_masks': pred_masks_stacked,  # [B, K, H, W] — one best mask per text
                'grad_text_indices': grad_text_indices_list,  # Which texts have gradients for loss
                'best_idx': best_idx,
                'depth': depth,
                'pointmaps': pointmaps_small,
                'pointmaps_full': pointmaps,
                'pointmaps_in_world_frame': getattr(self, '_using_world_frame_gasa', False),
                'intrinsics': intrinsics,
                'norm_params': norm_params,
                'num_texts': num_texts,
                'tokens_per_text': tokens_per_text,
            }
            if world_pointmaps_small is not None:
                outputs['world_pointmaps'] = world_pointmaps_small
            if presence_logit is not None:
                outputs['presence_logit'] = presence_logit
            if centroid_pred is not None:
                outputs['centroid_pred'] = centroid_pred
            if iou_pred is not None:
                outputs['iou_pred'] = iou_pred
            if text_scores is not None:
                outputs['text_scores'] = text_scores

            return outputs

        # STANDARD PATH: Single decoder pass for all texts
        queries, presence_logit, centroid_pred, iou_pred, per_query_centroids, text_scores, joint_scores, aux_outputs = self.gasa_decoder(
            encoder_memory, pointmaps_small, text_embedding,
            box_prompts=final_box_prompts, box_labels=final_box_labels,
            point_prompts=final_point_prompts, point_labels=final_point_labels,
            depths=depth, poses=gasa_decoder_pose, intrinsics=intrinsics_scaled,
            spatial_qualifier_idx=spatial_qualifier_idx if self.use_spatial_tokens else None,
            num_texts=num_texts, tokens_per_text=tokens_per_text,
        )
        queries = self.query_proj(queries)
        # Project auxiliary layer queries too (for per-layer align loss)
        aux_queries_proj = None
        if aux_outputs is not None:
            aux_queries_proj = [self.query_proj(aq) for aq in aux_outputs]
        self._profile_end("5_GASA_decoder", t0)

        # 6. Run SAM3's segmentation head
        t0 = self._profile_start()
        with torch.no_grad():
            fpn_features = backbone_out['backbone_fpn']
            pixel_embed = self.sam3.segmentation_head.pixel_decoder(fpn_features)
            instance_embeds = self.sam3.segmentation_head.instance_seg_head(pixel_embed)

        # SAM3-MO: instance_embeds is [B_orig, D, H, W] — expand to [B_orig*K, D, H, W]
        # to match queries which are [B_orig*K, Q, D]. FPN was NOT expanded (too large).
        if sam3_mo_mode and hasattr(self, '_sam3_mo_K'):
            instance_embeds = instance_embeds.repeat_interleave(self._sam3_mo_K, dim=0)

        mask_preds = self.sam3.segmentation_head.mask_predictor(queries, instance_embeds)

        # Mask refinement: upsample with learned convolutions + image guidance
        if self.use_mask_refiner:
            mask_preds = self.mask_refiner(mask_preds, images)
        self._profile_end("6_SAM3_seghead", t0)

        # 7. Select best mask based on strategy
        # For multi-object (num_texts > 1): skip single-mask selection — training loop
        # handles Hungarian matching per-view. Return all masks and scores.
        if num_texts > 1:
            # Multi-object: pred_logits uses mask_mean (text_scores is [B,Q,K], not compatible with [B,Q])
            pred_logits = mask_preds.mean(dim=(-2, -1))  # [B, Q]
            # Return first query's mask as pred_masks for backward compat (not used in multi-object loss)
            pred_masks = mask_preds[:, 0:1]  # [B, 1, H, W]
            best_idx = torch.zeros(B, dtype=torch.long, device=device)
        else:
            # pred_logits source controlled by --pred-logits-source:
            #   mask_mean: pred_logits = mask_preds.mean() — old text-agnostic behavior (proven, default)
            #   text_scoring: pred_logits = joint_scores — text-aware via DotProductScoring
            if self.pred_logits_source == 'text_scoring' and joint_scores is not None:
                pred_logits = joint_scores  # [B, Q] - text-query × presence scores
            else:
                pred_logits = mask_preds.mean(dim=(-2, -1))  # [B, Q] - text-agnostic (default)

            # Text scoring for mask selection: use text-aware logits for selection
            # when pred_logits_source='text_scoring', both at train AND eval.
            # Previously this was eval-only (not self.training guard), causing
            # train/eval mismatch: train used mask_mean, eval used text_scoring.
            use_text_scoring_selection = (
                self.pred_logits_source == 'text_scoring' and joint_scores is not None
            )

            # Object-aware spatial selection: use mask centroids + depth for "nearest chair" etc.
            # This takes priority when spatial qualifier is present
            # Spatial reasoning happens AFTER text scoring — filters among text-matched queries
            # At eval, use presence to weight mask selection (suppress absent objects)
            _eval_presence = presence_logit if not self.training else None

            if self.use_object_aware_spatial and spatial_qualifier_idx is not None and (spatial_qualifier_idx > 0).any():
                pred_masks, best_idx = self.select_mask_by_spatial(
                    mask_preds, depth, spatial_qualifier_idx, gt_masks=gt_masks,
                    fallback='iou' if gt_masks is not None else 'confidence'
                )
            elif use_text_scoring_selection:
                # Text-aware mask selection via DotProductScoring logits
                pred_masks, best_idx = self.select_mask_by_confidence(mask_preds, logits=pred_logits, presence_logit=_eval_presence)
            elif self.mask_selection == 'iou_match' and gt_masks is not None:
                # SAM3's approach: select mask with best IoU to GT during training
                pred_masks, best_idx = self.select_mask_by_iou(mask_preds, gt_masks)
            elif self.mask_selection == 'predicted_iou' and iou_pred is not None:
                # Zero-shot: use predicted IoU to select mask (requires trained IoU head)
                pred_masks, best_idx = self.select_mask_by_predicted_iou(mask_preds, iou_pred)
            elif self.mask_selection == 'majority_vote':
                # Aggregate top-k masks via soft voting
                pred_masks, best_idx = self.select_mask_by_majority_vote(mask_preds, self.topk_masks)
            else:
                pred_masks, best_idx = self.select_mask_by_confidence(mask_preds, presence_logit=_eval_presence)  # mask mean

        if pred_masks.shape[-2:] != images.shape[-2:]:
            pred_masks = F.interpolate(pred_masks, size=images.shape[-2:], mode='bilinear', align_corners=False)

        outputs = {
            'pred_masks': pred_masks,
            'pred_logits': pred_logits,
            'all_masks': mask_preds,  # Return all masks for analysis
            'best_idx': best_idx,
            'depth': depth,
            'pointmaps': pointmaps_small,  # Camera-frame, downsampled for decoder
            'pointmaps_full': pointmaps,  # Full DA3 resolution for centroid computation (camera or world frame)
            'pointmaps_in_world_frame': getattr(self, '_using_world_frame_gasa', False),  # True if pointmaps are already in DA3 world frame
            'intrinsics': intrinsics,  # For 3D localization
            'norm_params': norm_params,  # To convert normalized centroids back to meters
            'num_texts': num_texts,  # For multi-object training loop
            'tokens_per_text': tokens_per_text,  # For multi-object per-text scoring
            'sam3_mo_K': getattr(self, '_sam3_mo_K', None) if sam3_mo_mode else None,  # SAM3-MO: K objects expanded in batch
        }
        # World-frame pointmaps for sheaf loss (only when available)
        if world_pointmaps_small is not None:
            outputs['world_pointmaps'] = world_pointmaps_small
        if presence_logit is not None:
            outputs['presence_logit'] = presence_logit
        if centroid_pred is not None:
            outputs['centroid_pred'] = centroid_pred
        if per_query_centroids is not None:
            outputs['per_query_centroids'] = per_query_centroids
        if iou_pred is not None:
            outputs['iou_pred'] = iou_pred
        if text_scores is not None:
            outputs['text_scores'] = text_scores
        if joint_scores is not None:
            outputs['joint_scores'] = joint_scores
        if aux_queries_proj is not None:
            outputs['aux_queries'] = aux_queries_proj  # List of [B, Q, 256] per layer (excl. final)

        return outputs


    def forward_multiview(self, images, text_prompts, gt_masks=None,
                          gt_extrinsics=None, gt_intrinsics=None,
                          intrinsics_orig_hw=None, cached_depth=None,
                          point_prompts=None, point_labels=None,
                          box_prompts=None, box_labels=None,
                          da3_extrinsics=None, da3_intrinsics=None):
        return _mu.forward_multiview(
            self, images, text_prompts, gt_masks,
            gt_extrinsics=gt_extrinsics, gt_intrinsics=gt_intrinsics,
            intrinsics_orig_hw=intrinsics_orig_hw, cached_depth=cached_depth,
            point_prompts=point_prompts, point_labels=point_labels,
            box_prompts=box_prompts, box_labels=box_labels,
            da3_extrinsics=da3_extrinsics, da3_intrinsics=da3_intrinsics,
        )

