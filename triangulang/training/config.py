"""
Tyro-compatible dataclass configuration for TrianguLang training.

Replaces the argparse block in train.py. Use:
    config = tyro.cli(TrainConfig)
    args = config.to_namespace()  # flat namespace matching old argparse output
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated, Literal, Optional, Sequence

import tyro

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Nested config groups

@dataclass
class PresetConfig:
    """Preset flags that override multiple defaults at once."""

    sam3_defaults: bool = False
    # Apply SAM3-matching defaults (post-norm, 2048 FFN, ReLU, FP32 FFN, text_scoring, confidence selection, align loss)


@dataclass
class DA3Config:
    """Depth Anything v3 model configuration."""

    da3_model: Literal[
        'depth-anything/DA3METRIC-LARGE',
        'depth-anything/DA3-LARGE',
        'depth-anything/DA3NESTED-GIANT-LARGE',
    ] = 'depth-anything/DA3NESTED-GIANT-LARGE'
    # DA3 variant: METRIC-LARGE (fast, per-frame), DA3-LARGE (chunk, relative), NESTED-GIANT (chunk, metric+poses)

    da3_resolution: Optional[int] = None
    # Resolution for depth estimation (504=fast, 756=quality, 1008=full)

    use_cached_depth: bool = False
    # Use pre-computed DA3 depth from cache

    da3_cache_name: str = 'da3_nested_cache'

    da3_chunk_size: int = 8
    # DA3-NESTED chunk size for chunk_aware sampling

    use_cached_pi3x: bool = False
    # Use pre-computed world-frame pointmaps from MapAnything/PI3X (bypasses DA3 for pointmaps)

    pi3x_cache_name: str = 'ma_cache_train'
    # PI3X cache directory name under data root


@dataclass
class ModelConfig:
    """GASA decoder architecture parameters."""

    d_model: int = 256
    n_heads: int = 8
    num_decoder_layers: int = 6
    num_queries: int = 10
    dim_feedforward: int = 2048

    post_norm: bool = True
    # Use post-norm (SAM3-style) instead of pre-norm

    ffn_fp32: bool = True
    # Run decoder FFN in FP32, matching SAM3

    use_query_pe: bool = False
    # Add learned PE to queries at every decoder layer (SAM3-style)

    use_depth_crossattn: bool = False
    # Add cross-attention to depth features in each GASA layer

    per_layer_text: bool = True
    # Text cross-attention at every decoder layer (like SAM3)

    no_initial_text: bool = False
    # Skip the initial text cross-attention before decoder layers

    no_text_proj: bool = False
    # Bypass text_proj Linear before cross-attention

    clean_v: bool = False
    # Keep V PE-free by passing raw memory separately

    additive_pe: bool = False
    # SAM3-style PE: keep memory and world PE separate (not summed)

    use_image_to_token: bool = False
    # Image-to-token cross-attention in each GASA layer (SAM3 TwoWayTransformer style)

    use_pos_refine: bool = False
    # Iterative 3D position refinement (like SAM3 box_refine but in 3D)

    use_box_rpb: bool = False
    # 2D box-relative position bias (SAM3 boxRPB)

    query_proj_mlp: bool = False
    # Use 3-layer MLP for query_proj instead of single Linear

    no_query_proj: bool = False
    # Remove query_proj entirely (matching SAM3 architecture)

    train_mask_embed: bool = False
    # Unfreeze mask_embed MLP (3 layers in SAM3 MaskPredictor)

    use_mask_refiner: bool = False

    init_scoring_from_sam3: bool = False
    # Initialize scoring heads from SAM3 pretrained weights

    init_text_crossattn_from_sam3: bool = False
    # Initialize text cross-attention from SAM3 pretrained decoder

    init_decoder_from_sam3: bool = False
    # Initialize GASA self-attn/FFN/norms from SAM3 pretrained decoder

    train_seghead: bool = False


@dataclass
class GASAConfig:
    """Geometry-Aware Self-Attention (GASA) parameters."""

    use_gasa: bool = True
    # Geometric bias in cross-attention based on 3D distances

    gasa_beta_init: float = 1.0
    # Initial geometric bias strength

    gasa_kernel_dim: int = 32
    # Hidden dimension of distance kernel MLP

    gasa_fixed_kernel: bool = False
    # Use fixed kernel phi(d)=-d instead of learned MLP

    gasa_kernel_type: Literal['learned', 'rbf', 'fixed'] = 'learned'

    gasa_bidirectional: bool = False
    # Boost nearby tokens + suppress distant (not just boost)

    attn_map_size: int = 64
    # Spatial size of geometric bias map


@dataclass
class PEConfig:
    """Positional encoding configuration."""

    use_world_pe: bool = True

    pe_type: Literal['world', 'camera_relative', 'plucker', 'rayrope', 'none'] = 'world'
    # PE type: world (3D coords), camera_relative, plucker (ray), rayrope, none

    no_pointmap_normalize: bool = False
    # Disable pointmap normalization (keeps raw meter-scale coords)

    use_iterative_pos: bool = False
    # Update query positions each layer via attention-weighted centroids


@dataclass
class CrossViewConfig:
    """Cross-view attention and pose configuration."""

    cross_view: bool = False
    # Concatenate SAM3 memories from all N views for multi-view fusion

    use_da3_poses_for_gasa: bool = False
    # Use cached DA3-NESTED poses for world-frame pointmaps

    use_gt_poses_for_gasa: bool = False
    # Use GT COLMAP poses for GASA pointmaps

    no_gt_poses: bool = False

    sheaf_use_gt_poses: bool = False
    # Force sheaf loss to use GT COLMAP poses


@dataclass
class MultiObjectConfig:
    """Multi-object training parameters."""

    multi_object: bool = False

    match_strategy: Literal['hungarian', 'text_greedy'] = 'hungarian'

    per_text_decode: bool = False
    # Each text gets its own decoder pass (like SAM3)

    grouped_text_attn: bool = False
    # Restrict each query group to attend only to its assigned text

    sam3_multi_object: bool = False
    # SAM3-style: expand batch by K objects per sample

    num_objects: int = 1
    # Fixed objects per sample (0 = all visible)

    max_objects: int = 0
    # Cap K to limit memory usage


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""

    dataset: Literal['scannetpp', 'nvos', 'spinnerf', 'mvimgnet', 'uco3d', 'partimagenet'] = 'scannetpp'

    data_root: Optional[str] = None
    # Dataset root directory (default: dataset-specific)

    split: str = 'train_balanced_v4'
    views: int = 8

    sampling_strategy: Literal[
        'random', 'stratified', 'sequential', 'chunk_aware', 'overlap_30', 'overlap_50'
    ] = 'stratified'
    # View sampling: random, stratified (even spacing), sequential, chunk_aware (DA3 chunks), overlap_N

    max_scenes: int = 100

    resolution: int = 504
    # SAM3 input resolution (must be divisible by 14)

    min_mask_coverage: float = 0.000
    # Minimum mask coverage fraction at original resolution

    samples_per_scene: int = 1
    min_category_samples: int = 1

    exclude_categories: Optional[Sequence[str]] = None
    include_categories: Optional[Sequence[str]] = None

    # uCO3D-specific
    frames_per_sequence: int = 50
    # [uCO3D] Frames to sample per sequence

    samples_per_sequence: int = 1
    # [uCO3D] Training samples per sequence

    # PartImageNet-specific
    part_query_mode: Literal[
        'random', 'all', 'head', 'body', 'foot', 'tail', 'wing', 'hand', 'fin'
    ] = 'random'
    # [PartImageNet] Which part queries to use

    augment: bool = False
    # [PartImageNet] Enable data augmentation

    # Class-balanced sampling
    class_balanced: bool = True
    # Inverse frequency weighting for class-balanced sampling

    class_balance_power: float = 0.7
    # Power for class weights: 1.0=inverse freq, 0.5=sqrt, 0=uniform


@dataclass
class PromptConfig:
    """Prompt type and prompt generation configuration."""

    prompt_type: Literal[
        'all', 'text_only', 'box_only', 'point_only', 'text_box', 'text_point', 'random'
    ] = 'all'

    use_presence_token: bool = True
    # Predict whether object exists in each image

    use_box_prompts: bool = True
    box_prompt_dropout: float = 0.2
    use_point_prompts: bool = True
    point_prompt_dropout: float = 0.2
    num_pos_points: int = 10
    num_neg_points: int = 2


@dataclass
class TrainingConfig:
    """Core training hyperparameters."""

    epochs: int = 30

    stop_at_epoch: int = 0
    # Stop early at this epoch (0=disabled)

    batch_size: int = 8
    num_workers: int = 8

    prefetch_factor: int = 2
    # Batches to prefetch per worker

    no_persistent_workers: bool = False
    # Disable persistent workers to prevent memory accumulation

    lr: float = 1e-4

    lr_scheduler: Literal['none', 'cosine', 'step'] = 'cosine'

    lr_warmup_epochs: int = 2
    lr_min: float = 1e-6
    lr_step_size: int = 10
    lr_gamma: float = 0.5

    grad_accum: int = 1
    grad_clip: float = 1.0
    seed: int = 42


@dataclass
class CheckpointConfig:
    """Checkpoint and resume configuration."""

    run_name: Optional[str] = None
    checkpoint_dir: str = str(PROJECT_ROOT / 'checkpoints')
    resume: Optional[str] = None

    load_weights: Optional[str] = None
    # Load model weights only (no optimizer/scheduler, starts from epoch 0)


@dataclass
class ValidationConfig:
    """Validation configuration."""

    val_every: int = 0
    # Run validation every N epochs (0=disabled)

    val_split: str = 'val_cache_good'
    val_max_samples: Optional[int] = None

    save_best_val: bool = True
    # Save best checkpoint based on validation mIoU instead of training IoU


@dataclass
class LossConfig:
    """Loss function weights and parameters."""

    focal_weight: float = 2.0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    dice_weight: float = 0.5

    boundary_weight: float = 0.0
    # Distance-transform boundary loss (recommended: 0.3-1.0)

    lovasz_weight: float = 0.0
    # Lovasz-Hinge loss (recommended: 1.0-2.0)

    use_point_sampling: bool = False
    # SAM3-style uncertain point sampling for loss computation

    num_sample_points: int = 4096
    # Points to sample when point_sampling is enabled

    loss_at_native_res: bool = False
    # Compute loss at 288x288 instead of upsampling to GT resolution

    presence_weight: float = 2.0
    # Weight for presence loss (0=disabled)

    presence_focal: bool = True
    # Use focal loss for presence (SAM3-style)

    presence_alpha: float = 0.5
    presence_gamma: float = 0.0

    no_object_weight: float = 1.0
    # Weight for no-object loss on unmatched queries

    align_weight: float = 1.0
    # IoU-aware focal on pred_logits (SAM3/AlignDETR-style, 0=disabled)

    align_alpha: float = 0.5
    align_gamma: float = 2.0

    align_tau: float = 2.0
    # Temperature for rank-based weighting

    per_layer_align: bool = False
    # Compute align loss on intermediate decoder layer outputs

    per_layer_align_weight: Optional[float] = None

    contrastive_weight: float = 0.0
    contrastive_margin: float = 0.5

    contrastive_source: Literal['logits', 'iou_pred'] = 'logits'
    # Score source for contrastive loss

    use_iou_head: bool = False
    iou_head_weight: float = 1.0

    pred_logits_source: Literal['mask_mean', 'text_scoring'] = 'mask_mean'
    # Source for pred_logits (used by align loss and mask selection)

    mask_smooth_kernel: int = 0
    # Avg pool smoothing on mask logits before loss (0=disabled, 29=LangSplat)


@dataclass
class SheafConfig:
    """Sheaf consistency loss configuration."""

    use_sheaf_loss: bool = False
    # Cross-view geometric consistency loss

    sheaf_weight: float = 0.1

    sheaf_type: Literal['constant', 'feature'] = 'constant'
    sheaf_d_edge: int = 32

    sheaf_threshold: float = 0.10
    # Distance threshold in meters for correspondences

    sheaf_soft_correspondences: bool = False
    # Gaussian-weighted correspondences instead of hard NN

    sheaf_sigma: float = 0.10
    # Gaussian bandwidth in meters for soft correspondences

    sheaf_detach_target: bool = True
    # Detach target view predictions to prevent fighting

    sheaf_max_frame_distance: int = 0
    # Max frame distance for sheaf pairs (0=all)

    sheaf_symmetric_detach: bool = True
    # Randomly swap which view is detached

    sheaf_mutual_nn: bool = True
    # Require mutual nearest neighbors for correspondences


@dataclass
class CentroidConfig:
    """3D localization / centroid head configuration."""

    use_centroid_head: bool = False
    centroid_weight: float = 0.5

    mask_based_centroid: bool = False
    # Compute centroid from mask+depth instead of attention-weighted position

    use_triangulation: bool = False
    # Multi-view ray triangulation for 3D localization

    eval_localization: bool = False
    # Track 3D metrics from mask+depth without training centroid head


@dataclass
class OutputConfig:
    """Inference output configuration."""

    output_localization: bool = False
    output_depth: bool = False
    output_pointcloud: bool = False


@dataclass
class SemanticConfig:
    """Semantic union and mask selection configuration."""

    semantic_union: bool = False
    # Merge all instances of same class into one mask

    mask_selection: Literal[
        'confidence', 'iou_match', 'predicted_iou', 'majority_vote', 'text_score'
    ] = 'confidence'


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    vis_every_epochs: int = 0
    # Visualize every N epochs (0=disabled)

    vis_every_batches: int = 0
    # Visualize every N batches (0=disabled)


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""

    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_sam3: bool = False
    lora_da3: bool = False
    lora_mask_embed: bool = False


@dataclass
class PerformanceConfig:
    """Performance optimization options."""

    torch_compile: bool = False
    compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = 'reduce-overhead'

    batch_views: bool = False
    # Batch all views in forward pass (causes OOM, not recommended)

    profile: bool = False

    min_ram_gb: float = 0
    # Minimum available RAM in GB (0=disabled)


@dataclass
class SpatialConfig:
    """Spatial reasoning configuration."""

    use_spatial_tokens: bool = False
    # Learnable spatial token embeddings for spatial qualifiers

    use_spatial_attn_bias: bool = False
    # Depth-conditioned spatial attention bias

    use_text_spatial_bias: bool = False
    # ViL3DRel-inspired text-conditioned spatial bias

    use_spatial_points: bool = False
    # Convert spatial qualifiers to pseudo-point prompts

    use_object_aware_spatial: bool = False

    spatial_augment_prob: float = 0.0
    # Probability of augmenting labels with spatial qualifiers (0=disabled)

    spatial_multi_instance_only: bool = True
    # Only augment objects appearing 2+ times in scene

    spatial_relational_prob: float = 0.0
    # Probability of generating relational queries (0=disabled)

    spatial_gt_aware: bool = False
    # Use GT-aware spatial augmentation

    spatial_ranking_weight: float = 0.0
    spatial_ranking_margin: float = 0.5


# Top-level config that nests all groups

@dataclass
class TrainConfig:
    """Top-level training configuration for TrianguLang GASA decoder."""

    preset: Annotated[PresetConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=PresetConfig)
    da3: Annotated[DA3Config, tyro.conf.OmitArgPrefixes] = field(default_factory=DA3Config)
    model: Annotated[ModelConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=ModelConfig)
    gasa: Annotated[GASAConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=GASAConfig)
    pe: Annotated[PEConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=PEConfig)
    cross_view: Annotated[CrossViewConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=CrossViewConfig)
    multi_obj: Annotated[MultiObjectConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=MultiObjectConfig)
    data: Annotated[DataConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=DataConfig)
    prompt: Annotated[PromptConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=PromptConfig)
    training: Annotated[TrainingConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=TrainingConfig)
    checkpoint: Annotated[CheckpointConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=CheckpointConfig)
    validation: Annotated[ValidationConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=ValidationConfig)
    loss: Annotated[LossConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=LossConfig)
    sheaf: Annotated[SheafConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=SheafConfig)
    centroid: Annotated[CentroidConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=CentroidConfig)
    output: Annotated[OutputConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=OutputConfig)
    semantic: Annotated[SemanticConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=SemanticConfig)
    visualization: Annotated[VisualizationConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=VisualizationConfig)
    lora: Annotated[LoRAConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=LoRAConfig)
    performance: Annotated[PerformanceConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=PerformanceConfig)
    spatial: Annotated[SpatialConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=SpatialConfig)

    def to_namespace(self) -> argparse.Namespace:
        """Flatten all nested dataclass fields into a single argparse.Namespace.

        This produces an object with the same attribute names (e.g., args.lr,
        args.use_gasa) that the rest of train.py expects from the old argparse
        code.
        """
        flat: dict = {}
        for group_field in fields(self):
            group = getattr(self, group_field.name)
            for f in fields(group):
                value = getattr(group, f.name)
                if f.name in flat:
                    raise ValueError(
                        f"Duplicate field name '{f.name}' found in group "
                        f"'{group_field.name}': all field names must be unique across groups."
                    )
                flat[f.name] = value
        return argparse.Namespace(**flat)

    @staticmethod
    def get_parser_defaults() -> dict:
        """Return a dict of {field_name: default_value} mirroring the old argparse defaults.

        Used by the resume-config and --sam3-defaults logic to detect which
        values are still at their defaults vs explicitly overridden.
        """
        defaults: dict = {}
        for group_field in fields(TrainConfig):
            group_cls = group_field.type
            # Resolve string annotations
            if isinstance(group_cls, str):
                group_cls = eval(group_cls)
            # Handle default_factory
            if group_field.default_factory is not type:
                group_default = group_field.default_factory()
            else:
                group_default = group_cls()
            for f in fields(group_default):
                defaults[f.name] = getattr(group_default, f.name)
        return defaults
