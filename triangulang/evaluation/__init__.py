"""TrianguLang evaluation package."""
from .eval_utils import (
    compute_metrics, compute_oracle_iou, compute_3d_centroid,  # noqa: F401
    compute_centroid_error, umeyama_alignment, compute_cross_view_consistency,  # noqa: F401
    compute_spatial_gt, create_prompts_from_gt,  # noqa: F401
)
from .data_loading import (
    load_model, load_scene_data, load_gt_masks, load_gt_poses,  # noqa: F401
    get_frame_extrinsics, load_cached_da3_nested,  # noqa: F401
    load_gt_centroids, load_gt_poses_for_scene,  # noqa: F401
    BaselineSAM3Wrapper, count_parameters,  # noqa: F401
)
from .visualization import MASK_COLORS  # noqa: F401
