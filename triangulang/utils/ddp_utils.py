"""Distributed Data Parallel (DDP) utilities for multi-GPU training.

Provides DDPManager for process group init, model wrapping, distributed
sampling, and rank-aware logging.
"""

import os
import socket
from datetime import timedelta
from typing import Optional, Any
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np


class DistributedWeightedSampler:
    """
    Weighted random sampler for distributed training.

    Each rank samples indices based on weights, with different random seeds
    to ensure different samples across GPUs. Supports set_epoch for proper shuffling.
    """

    def __init__(
        self,
        weights: list,
        num_samples: int,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        seed: int = 0,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_samples = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        # Number of samples per replica
        self.num_samples_per_replica = (num_samples + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples_per_replica * num_replicas

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across ranks."""
        self.epoch = epoch

    def __iter__(self):
        # Create generator with epoch + rank for reproducibility across ranks
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * self.num_replicas + self.rank)

        # Sample indices based on weights
        indices = torch.multinomial(
            self.weights,
            self.num_samples_per_replica,
            replacement=self.replacement,
            generator=g
        ).tolist()

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica


class DDPManager:
    """
    Manager class for Distributed Data Parallel training.

    Handles initialization, model wrapping, and dataloader setup for
    single-node multi-GPU and multi-node training.

    Environment variables (set by torchrun or manually):
        MASTER_ADDR: IP address of rank 0 node
        MASTER_PORT: Port for communication
        WORLD_SIZE: Total number of processes
        RANK: Global rank of this process
        LOCAL_RANK: Local rank on this node (GPU index)
    """

    def __init__(self):
        self.initialized = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = None
        self.sampler = None
        self._backend = None

    @property
    def is_main(self) -> bool:
        """True if this is the main process (rank 0)."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """True if running in distributed mode."""
        return self.world_size > 1

    def init(self, backend: str = "nccl", timeout_minutes: int = 30) -> "DDPManager":
        """
        Initialize distributed training.

        Args:
            backend: "nccl" for GPU, "gloo" for CPU or as fallback
            timeout_minutes: Timeout for initialization

        Returns:
            self for chaining
        """
        # Check if we're in distributed mode
        if "WORLD_SIZE" not in os.environ:
            print("[DDP] Running in single-GPU mode (WORLD_SIZE not set)")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return self

        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if self.world_size <= 1:
            print("[DDP] Running in single-GPU mode (WORLD_SIZE=1)")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return self

        # Set device before init
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
            backend = "gloo"  # NCCL requires CUDA

        # Initialize process group
        timeout = timedelta(minutes=timeout_minutes)

        print(f"[DDP] Initializing rank {self.rank}/{self.world_size} "
              f"(local_rank={self.local_rank}) on {socket.gethostname()}")
        print(f"[DDP] MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
              f"MASTER_PORT={os.environ.get('MASTER_PORT')}")

        dist.init_process_group(
            backend=backend,
            timeout=timeout,
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device if torch.cuda.is_available() else None,
        )

        self._backend = backend
        self.initialized = True

        # Synchronize before continuing
        dist.barrier()

        if self.is_main:
            print(f"[DDP] Initialized {self.world_size} processes")

        return self

    def wrap_model(
        self,
        model: torch.nn.Module,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
    ) -> torch.nn.Module:
        """
        Wrap model with DistributedDataParallel.

        Args:
            model: PyTorch model (should already be on correct device)
            find_unused_parameters: Set True if some params don't receive gradients
            gradient_as_bucket_view: Memory optimization (usually True)

        Returns:
            DDP-wrapped model (or original if not distributed)
        """
        if not self.is_distributed:
            return model.to(self.device)

        model = model.to(self.device)

        return DDP(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

    def wrap_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn: Optional[Any] = None,
        sample_weights: Optional[list] = None,
        **kwargs,
    ) -> DataLoader:
        """
        Create a DataLoader with DistributedSampler.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size PER GPU (total = batch_size * world_size)
            shuffle: Whether to shuffle (handled by sampler in distributed mode)
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop incomplete batches (recommended for DDP)
            collate_fn: Custom collate function
            sample_weights: Optional per-sample weights for class-balanced sampling
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader with appropriate sampler
        """
        if sample_weights is not None:
            # Use weighted sampling
            if self.is_distributed:
                self.sampler = DistributedWeightedSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    num_replicas=self.world_size,
                    rank=self.rank,
                    replacement=True,  # Required for weighted sampling
                    seed=42,
                )
                print(f"[DDP R{self.rank}] Using DistributedWeightedSampler for class-balanced training")
            else:
                # Single GPU: use standard WeightedRandomSampler
                self.sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    replacement=True,
                )
                print("[DDP] Using WeightedRandomSampler for class-balanced training")
            shuffle = False  # Sampler handles sampling
        elif self.is_distributed:
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            shuffle = False  # Sampler handles shuffling
        else:
            self.sampler = None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if self.sampler is None else False,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs,
        )

    def set_epoch(self, epoch: int):
        """
        Set epoch for DistributedSampler (required for proper shuffling).

        Call this at the beginning of each epoch.
        """
        if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """
        All-reduce a tensor across all processes.

        Args:
            tensor: Input tensor
            op: "mean", "sum", "min", "max"

        Returns:
            Reduced tensor
        """
        if not self.is_distributed:
            return tensor

        ops = {
            "mean": dist.ReduceOp.SUM,
            "sum": dist.ReduceOp.SUM,
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
        }

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=ops[op])

        if op == "mean":
            tensor /= self.world_size

        return tensor

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all processes.

        Args:
            tensor: Local tensor [...]

        Returns:
            Gathered tensor [world_size, ...]
        """
        if not self.is_distributed:
            return tensor.unsqueeze(0)

        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.stack(gathered)

    def print(self, *args, **kwargs):
        """Print only on main process."""
        if self.is_main:
            print(*args, **kwargs)

    @contextmanager
    def main_only(self):
        """Context manager that only executes on main process."""
        if self.is_main:
            yield
        else:
            yield None

    def cleanup(self):
        """Clean up distributed training."""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False

    def get_model(self, ddp_model: torch.nn.Module) -> torch.nn.Module:
        """Get the underlying model from DDP wrapper."""
        if isinstance(ddp_model, DDP):
            return ddp_model.module
        return ddp_model

    def save_checkpoint(
        self,
        state_dict: dict,
        path: str,
        only_main: bool = True,
    ):
        """
        Save checkpoint (only on main process by default).

        Args:
            state_dict: State dict to save
            path: Save path
            only_main: If True, only save on rank 0
        """
        if only_main and not self.is_main:
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(state_dict, path)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[str] = None,
    ) -> dict:
        """
        Load checkpoint with proper device mapping.

        Args:
            path: Checkpoint path
            map_location: Device mapping (defaults to self.device)

        Returns:
            Loaded state dict
        """
        if map_location is None:
            map_location = self.device

        return torch.load(path, map_location=map_location)


def setup_ddp_env(
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
):
    """
    Manually set DDP environment variables.

    Usually not needed when using torchrun, but useful for debugging.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)


# Convenience function for quick setup
def init_ddp(backend: str = "nccl") -> DDPManager:
    """Quick DDP initialization."""
    return DDPManager().init(backend=backend)
