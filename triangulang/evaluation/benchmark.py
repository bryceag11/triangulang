"""
TrianguLang Benchmark

Entry point for all modes. Dispatches to dataset-specific
functions based on --dataset flag.

Usage:
    torchrun --nproc_per_node=8 triangulang/evaluation/benchmark.py \
        --checkpoint checkpoints/my_run/best.pt \
        --run-name my_run
"""
import sys
import json
import random
import math
import tyro
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'sam3'))
sys.path.insert(0, str(PROJECT_ROOT / 'depth_anything_v3' / 'src'))

from sam3 import build_sam3_image_model
from triangulang.utils.ddp_utils import DDPManager
from triangulang.evaluation.config import BenchmarkConfig
from triangulang.evaluation.data_loading import (
    BaselineSAM3Wrapper, count_parameters, load_model,
)
from triangulang.evaluation.eval_lerf import _evaluate_lerf
from triangulang.evaluation.eval_scannetpp import _evaluate_scannetpp
from triangulang.evaluation.eval_datasets import _evaluate_uco3d, _evaluate_nvos, _evaluate_partimagenet
from triangulang import BPE_PATH as _BPE_PATH
import triangulang

logger = triangulang.get_logger(__name__)


def main():
    config = tyro.cli(BenchmarkConfig)
    args = config.to_namespace()

    if not args.baseline_sam3 and not args.checkpoint:
        raise SystemExit("Error: --checkpoint is required unless --baseline-sam3 is used")

    if args.mask_size is None and args.image_size is not None:
        args.mask_size = (args.image_size // 14) * 4

    # Initialize DDP
    ddp = DDPManager()
    ddp.init(timeout_minutes=120)

    triangulang.configure_logging(ddp.rank)

    random.seed(args.seed + ddp.rank)
    np.random.seed(args.seed + ddp.rank)
    torch.manual_seed(args.seed + ddp.rank)

    device = ddp.device if ddp.is_distributed else ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    logger.info(f"Device: {device}" + (f" (DDP: {ddp.world_size} GPUs)" if ddp.is_distributed else ""))

    # Output directory
    if args.run_name:
        run_name = args.run_name
    elif args.baseline_sam3:
        run_name = f"baseline_sam3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        ckpt_name = Path(args.checkpoint).parent.name
        run_name = f"eval_{ckpt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.dataset == 'lerf_ovs':
        output_dir = PROJECT_ROOT / 'runs' / 'lerf' / run_name
    else:
        output_dir = PROJECT_ROOT / 'runs' / 'final' / run_name
    if ddp.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / 'visualizations'
    if args.visualize and ddp.is_main:
        viz_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load model
    if args.baseline_sam3:
        if args.image_size is None:
            args.image_size = 1008
        if args.mask_size is None:
            args.mask_size = (args.image_size // 14) * 4
        sam3_res = math.ceil(args.image_size / 14) * 14
        logger.info(f"Loading baseline SAM3 (native decoder, no GASA/depth/cross-view)...")
        logger.info(f"  SAM3 img_size={sam3_res} (from --image-size {args.image_size})")
        sam3_model = build_sam3_image_model(bpe_path=_BPE_PATH, img_size=sam3_res).to(device)
        model = BaselineSAM3Wrapper(sam3_model, resolution=sam3_res)
        model.to(device)
        total_params, trainable_params = count_parameters(model)
        gasa_params = 0
        logger.info(f"Baseline SAM3 Parameters: {total_params/1e6:.2f}M")
    else:
        logger.info(f"Loading model from {args.checkpoint}...")
        model = load_model(args.checkpoint, device, da3_resolution=args.da3_resolution,
                          num_queries=args.num_queries, skip_trained_seghead=args.skip_trained_seghead,
                          train_config_path=args.train_config, resolution=args.image_size)

        if args.image_size is None:
            args.image_size = model.resolution
        if args.mask_size is None:
            args.mask_size = (args.image_size // 14) * 4
        logger.info(f"  image_size={args.image_size}, mask_size={args.mask_size}")

        # Auto-enable spatial tokens if trained with them
        ckpt_dir = Path(args.checkpoint).parent
        train_config_path = args.train_config or str(ckpt_dir / 'config.json')
        if Path(train_config_path).exists():
            with open(train_config_path) as f:
                train_config = json.load(f)
            if train_config.get('use_spatial_tokens', False) and not args.spatial_eval and not args.no_spatial_eval:
                args.spatial_eval = True
                logger.info(f"  Auto-enabled --spatial-eval")
            if args.no_spatial_eval:
                args.spatial_eval = False

        if model.pred_logits_source == 'text_scoring' and not args.mask_selection:
            model.mask_selection = 'confidence'

        if args.mask_selection:
            model.mask_selection = args.mask_selection

        total_params, trainable_params = count_parameters(model)
        gasa_params = sum(p.numel() for p in model.gasa_decoder.parameters())
        logger.info(f"Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable, {gasa_params/1e6:.2f}M GASA")

    # Resolve data root
    if args.data_root is None:
        dataset_paths = {
            'scannetpp': 'data/scannetpp', 'uco3d': 'data/uco3d',
            'partimagenet': 'data/partimagenet/PartImageNet',
            'lerf_ovs': 'data/lerf_ovs',
            'nvos': 'data/nvos',
        }
        if args.dataset not in dataset_paths:
            raise ValueError(f"--data-root required for dataset {args.dataset}")
        args.data_root = str(PROJECT_ROOT / dataset_paths[args.dataset])
    data_root = Path(args.data_root)

    # Dispatch to dataset-specific handler
    if args.dataset == 'uco3d':
        _evaluate_uco3d(model, args, device, ddp, data_root, output_dir, viz_dir,
                        total_params, trainable_params, gasa_params)
    elif args.dataset == 'lerf_ovs':
        _evaluate_lerf(model, args, device, ddp, data_root, output_dir, viz_dir)
    elif args.dataset == 'nvos':
        _evaluate_nvos(model, args, device, ddp, data_root, output_dir, viz_dir)
    elif args.dataset == 'partimagenet':
        _evaluate_partimagenet(model, args, device, ddp, data_root, output_dir, viz_dir)
    else:
        _evaluate_scannetpp(model, args, device, ddp, data_root, output_dir, viz_dir,
                            total_params, trainable_params, gasa_params)


if __name__ == '__main__':
    main()
