"""Factory functions for creating UCO3D dataset instances."""
from typing import Dict, List, Tuple, Optional, Union
from triangulang.data.uco3d_dataset import UCO3DMultiViewDataset
from triangulang.data.uco3d_utils import normalize_prompt



def get_uco3d_eval_config() -> Dict:
    """
    Get evaluation configuration matching the paper description:
    - 50 representative sequences
    - 50 frames per sequence (uniformly sampled)
    """
    return {
        'num_sequences': 50,
        'frames_per_sequence': 50,
        'num_views': 8,  # Views per evaluation sample
        'split': 'val',
    }


def get_uco3d_kfold_datasets(
    num_folds: int = 5,
    fold: int = 0,
    num_sequences: Optional[int] = None,
    **kwargs,
) -> Tuple['UCO3DMultiViewDataset', 'UCO3DMultiViewDataset']:
    """
    Create train and val datasets for k-fold cross validation.

    Args:
        num_folds: Number of folds (e.g., 5 for 5-fold CV)
        fold: Which fold to use as validation (0 to num_folds-1)
        num_sequences: Optional limit on sequences per split (after fold split)
        **kwargs: Additional arguments passed to UCO3DMultiViewDataset

    Returns:
        (train_dataset, val_dataset) tuple

    Example:
        # 5-fold cross validation, using fold 0 as validation
        train_ds, val_ds = get_uco3d_kfold_datasets(num_folds=5, fold=0)

        # Run all 5 folds
        for fold in range(5):
            train_ds, val_ds = get_uco3d_kfold_datasets(num_folds=5, fold=fold)
            # Train and evaluate...
    """
    train_dataset = UCO3DMultiViewDataset(
        split='train',
        num_folds=num_folds,
        fold=fold,
        num_sequences=num_sequences,
        **kwargs,
    )
    val_dataset = UCO3DMultiViewDataset(
        split='val',
        num_folds=num_folds,
        fold=fold,
        num_sequences=num_sequences,
        **kwargs,
    )
    return train_dataset, val_dataset


def get_uco3d_train_eval_split(
    data_root: str = None,
    num_train_sequences: int = 500,
    num_eval_sequences: int = 50,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Create train/eval split for uCO3D with non-overlapping sequences.

    Paper protocol (uco3d.md): 50 sequences for eval, 50 frames each.
    For training, use separate sequences.

    Args:
        data_root: Path to uCO3D dataset
        num_train_sequences: Number of sequences for training
        num_eval_sequences: Number of sequences for evaluation (default: 50 per paper)
        seed: Random seed for reproducible splits

    Returns:
        (train_sequences, eval_sequences) - lists of sequence paths
    """
    import random

    # Create temporary dataset to get all sequences
    temp_dataset = UCO3DMultiViewDataset(
        data_root=data_root,
        num_sequences=None,  # Get all
        frames_per_sequence=1,  # Minimal loading
        use_depth=False,
    )

    all_sequences = temp_dataset.sequences.copy()
    random.seed(seed)
    random.shuffle(all_sequences)

    # Split: first N for eval (held out), rest for train
    eval_sequences = all_sequences[:num_eval_sequences]
    train_sequences = all_sequences[num_eval_sequences:num_eval_sequences + num_train_sequences]

    print(f"[uCO3D Split] Train: {len(train_sequences)} sequences, Eval: {len(eval_sequences)} sequences")

    return train_sequences, eval_sequences


def create_uco3d_eval_dataset(
    data_root: str = None,
    num_views: int = 8,
    image_size: Tuple[int, int] = (504, 504),
    mask_size: Tuple[int, int] = (128, 128),
    normalize_prompts: bool = True,
    num_sequences: int = None,
    frames_per_sequence: int = None,
    seed: int = 42,
    # K-fold cross validation
    num_folds: Optional[int] = None,
    fold: Optional[int] = None,
) -> UCO3DMultiViewDataset:
    """
    Create uCO3D evaluation dataset with paper-specified sampling.

    From uco3d.md: "We sample 50 representative sequences...uniformly sample
    50 frames from each sequence"

    Args:
        normalize_prompts: If True, simplify LVIS category names for better text encoder understanding
        num_sequences: Number of sequences to sample (default: 50 from paper protocol)
        frames_per_sequence: Frames per sequence (default: 50 from paper protocol)
        seed: Random seed for reproducible sequence sampling
        num_folds: For k-fold CV, number of folds (e.g., 5). If set, ignores original train/val split.
        fold: Which fold to use as validation (0 to num_folds-1). Required if num_folds is set.
    """
    config = get_uco3d_eval_config()

    return UCO3DMultiViewDataset(
        data_root=data_root,
        split=config['split'],
        num_views=num_views,
        image_size=image_size,
        mask_size=mask_size,
        num_sequences=num_sequences if num_sequences is not None else config['num_sequences'],
        frames_per_sequence=frames_per_sequence if frames_per_sequence is not None else config['frames_per_sequence'],
        samples_per_sequence=1,  # One sample per sequence for eval
        normalize_prompts=normalize_prompts,
        seed=seed,
        num_folds=num_folds,
        fold=fold,
    )


