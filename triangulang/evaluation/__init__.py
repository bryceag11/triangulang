"""TrianguLang evaluation package."""
from .eval_utils import (
    compute_metrics, compute_oracle_iou, compute_3d_centroid,
    compute_centroid_error, umeyama_alignment, compute_cross_view_consistency,
    compute_spatial_gt, create_prompts_from_gt,
)
from .data_loading import (
    load_model, load_scene_data, load_gt_masks, load_gt_poses,
    get_frame_extrinsics, load_cached_da3_nested,
    load_gt_centroids, load_gt_poses_for_scene,
    BaselineSAM3Wrapper, count_parameters,
)
from .visualization import MASK_COLORS
