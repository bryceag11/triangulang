"""
Tyro-compatible dataclass configuration for benchmark evaluation.

Usage:
    args = tyro.cli(BenchmarkConfig)
    ns = args.to_namespace()  # Flat argparse-like namespace
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from typing import Annotated, Literal, Optional, Tuple, List

import tyro


@dataclass
class ModelConfig:
    """Model loading and architecture options."""

    checkpoint: Optional[str] = None
    # Path to GASA checkpoint. Not required when using --baseline-sam3.

    train_config: Optional[str] = None
    # Path to training config.json (default: auto-detect from checkpoint path)

    baseline_sam3: bool = False
    # Use native SAM3 as baseline (no GASA decoder)

    skip_trained_seghead: bool = False
    # Keep default SAM3 seghead instead of loading trained one

    num_queries: Optional[int] = None
    # Override num_queries from checkpoint config

    mask_selection: Optional[Literal["iou_match", "confidence", "predicted_iou", "text_score"]] = None

    da3_resolution: Optional[int] = None
    # Override DA3 depth resolution (default: from config, fallback 504)


@dataclass
class DataConfig:
    """Dataset and data loading options."""

    dataset: Literal["scannetpp", "nvos", "spinnerf", "uco3d", "partimagenet", "lerf_ovs", "lerf_loc"] = "scannetpp"

    data_root: Optional[str] = None
    split: str = "nvs_sem_val_v2"
    max_scenes: int = 50
    scene: Optional[List[str]] = None
    num_frames: int = 100
    objects_per_scene: int = 5

    eval_sampling: Literal["stratified", "sequential", "random", "overlap"] = "sequential"
    # Frame sampling strategy for evaluation

    min_mask_coverage: float = 0.01
    # Minimum mask coverage fraction for object to be evaluated

    min_category_samples: int = 1

    frame_names: Optional[List[str]] = None
    # Specific frame filenames to evaluate (overrides --num-frames)

    semantic_union: bool = False
    # Merge all instances of same class into one GT mask

    seed: int = 42


@dataclass
class ResolutionConfig:
    """Image and mask resolution options."""

    image_size: Optional[int] = None
    # Model input size (default: from training config)

    image_height: Optional[int] = None
    # Enables rectangular input (if different from image_size)

    image_width: Optional[int] = None
    # Enables rectangular input (if different from image_size)

    native_resolution: bool = False
    # Use native dataset resolution (padded to nearest multiple of 14)

    mask_size: Optional[int] = None
    # GT/prediction comparison size (default: (image_size // 14) * 4)


@dataclass
class InferenceConfig:
    """Inference and processing options."""

    view_chunk_size: int = 8
    # Process views in chunks to avoid OOM (0=all at once)

    per_frame: bool = False
    # Process each frame independently (no cross-view attention)

    batch_da3: bool = False
    # In per-frame mode, batch all views for DA3 before per-frame segmentation

    temporal_smooth_alpha: float = 0.0
    # EMA smoothing on mask logits across frames (0=disabled, 0.6-0.8=typical)

    use_crf: bool = False
    # CRF post-processing for sharper mask boundaries


@dataclass
class OutputConfig:
    """Output paths and saving options."""

    output: Optional[str] = None
    run_name: Optional[str] = None
    output_depth: bool = False
    output_pointcloud: bool = False
    output_localization: bool = False


@dataclass
class VisualizationConfig:
    """Standard visualization options."""

    visualize: bool = False
    viz_samples: int = 5


@dataclass
class PaperVizConfig:
    """Paper-quality grid visualization (no text, no axes, high DPI)."""

    paper_viz: bool = False
    paper_viz_rows: int = 4
    paper_viz_sets: int = 10
    paper_viz_dpi: int = 600

    paper_viz_mode: Literal["single_object", "single_scene", "multi_scene"] = "multi_scene"

    paper_viz_scenes: Optional[List[str]] = None
    paper_viz_objects: Optional[List[str]] = None

    paper_viz_mask_color: Literal["white", "blue", "green", "colored"] = "white"
    # Color for standalone mask columns

    paper_viz_overlay_alpha: float = 0.5


@dataclass
class SingleObjectVizConfig:
    """Single-object focused visualization mode."""

    single_object_viz: bool = False
    # Evaluate specific objects in a scene, show top-k IoU frames

    viz_scene: Optional[str] = None
    viz_objects: Optional[List[str]] = None
    viz_topk: int = 4
    viz_num_objects: int = 4
    viz_separate: bool = False
    viz_random_scene: bool = False
    viz_num_scenes: int = 1
    viz_min_iou: float = 0.0


@dataclass
class PromptConfig:
    """Prompt type and prompting strategy options."""

    prompt_type: Literal[
        "text_only", "text_box", "text_point", "text_box_point", "all",
        "point_only", "box_only", "box_point_only"
    ] = "text_only"

    use_point_prompts: bool = False
    use_box_prompts: bool = False
    num_pos_points: int = 10
    num_neg_points: int = 2

    sparse_prompts: bool = True
    # Distribute total points across a few frames (MV-SAM style)

    num_prompted_frames: int = 3
    # Frames to receive point prompts in sparse mode

    single_prompt: bool = False
    # Prompt view 0 only, measure views 1-N

    prompt_view: int = 0
    # Which view to prompt in single-prompt mode

    custom_prompts: Optional[List[str]] = None
    # Override dataset prompts with custom text queries

    prompt_aliases: bool = True
    # Normalize GT categories via LERF_PROMPT_ALIASES dict

    use_synonyms: bool = False
    # Test robustness to prompt variations

    synonym_prob: float = 0.5

    normalize_prompts: bool = True
    # [uCO3D] Normalize LVIS category names to simpler prompts


@dataclass
class EvalModeConfig:
    """Special evaluation modes and metrics."""

    consistency_metric: bool = False
    # Compute cross-view consistency (3D centroid variance)

    spatial_eval: bool = False
    # Evaluate spatial queries like "nearest chair", "leftmost table"

    no_spatial_eval: bool = False
    # Force disable spatial eval even if model was trained with spatial tokens

    multi_object_eval: bool = False
    # Batch all objects per frame in a single forward pass

    cross_fold: bool = False
    num_folds: int = 5

    fold: Optional[int] = None
    # Which fold to use as val set (0 to num_folds-1)


@dataclass
class LERFConfig:
    """LERF-OVS / LERF-Loc specific options."""

    lerf_multiview: bool = False
    # Use multi-view forward for LERF eval instead of single-view

    lerf_mv_identity_poses: bool = False
    # Force identity poses in LERF multi-view eval

    langsplat_protocol: bool = False
    # LangSplat IoU protocol: min-max normalization + threshold

    langsplat_thresh: float = 0.4

    no_loc_smoothing: bool = False
    # Disable 29x29 avg-pool smoothing before localization argmax


@dataclass
class PoseConfig:
    """Pose estimation and Procrustes alignment options."""

    procrustes: bool = False
    # Enable Procrustes-aligned localization evaluation

    procrustes_with_scale: bool = True
    # 7-DoF Procrustes (with scale). False=6-DoF.

    use_estimated_poses: bool = False
    # Use DA3-NESTED estimated poses instead of GT

    use_world_poses: bool = False
    # Use world-frame poses for pointmaps instead of camera-frame

    da3_nested_cache: str = "da3_nested_cache_1008"

    compare_pose_sources: bool = False
    # Run evaluation with both GT and estimated poses


@dataclass
class UCO3DConfig:
    """uCO3D dataset specific options."""

    num_sequences: Optional[int] = None
    # [uCO3D] Number of sequences to evaluate

    frames_per_sequence: Optional[int] = None
    # [uCO3D] Frames per sequence


@dataclass
class BenchmarkConfig:
    """Evaluate GASA Decoder on Multi-View Benchmarks."""

    model: Annotated[ModelConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=ModelConfig)
    data: Annotated[DataConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=DataConfig)
    resolution: Annotated[ResolutionConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=ResolutionConfig)
    inference: Annotated[InferenceConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=InferenceConfig)
    output_cfg: Annotated[OutputConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=OutputConfig)
    viz: Annotated[VisualizationConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=VisualizationConfig)
    paper_viz_cfg: Annotated[PaperVizConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=PaperVizConfig)
    single_obj_viz: Annotated[SingleObjectVizConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=SingleObjectVizConfig)
    prompt: Annotated[PromptConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=PromptConfig)
    eval_mode: Annotated[EvalModeConfig, tyro.conf.OmitArgPrefixes, tyro.conf.FlagConversionOff] = field(default_factory=EvalModeConfig)
    lerf: Annotated[LERFConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=LERFConfig)
    pose: Annotated[PoseConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=PoseConfig)
    uco3d: Annotated[UCO3DConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=UCO3DConfig)

    def to_namespace(self) -> argparse.Namespace:
        """Flatten nested config into a flat argparse.Namespace.

        This allows the rest of the code to use args.xxx access unchanged.
        All nested fields are merged into a single flat namespace.
        Conflicts are avoided because field names are unique across sub-configs.
        """
        flat = {}
        for f in fields(self):
            sub = getattr(self, f.name)
            if hasattr(sub, '__dataclass_fields__'):
                for sf in fields(sub):
                    flat[sf.name] = getattr(sub, sf.name)
            else:
                flat[f.name] = sub
        return argparse.Namespace(**flat)
