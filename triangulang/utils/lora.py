"""LoRA (Low-Rank Adaptation) for efficient fine-tuning of frozen backbones."""

import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning.

    Computes: output = original_output + (x @ A.T @ B.T) * scaling
    Where A and B are low-rank matrices that adapt the frozen weights.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices: A projects down, B projects up
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros (LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta to add to original output."""
        # x: [..., in_features] -> [..., out_features]
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAManager:
    """Manages LoRA adapters for frozen models using forward hooks.

    This properly applies LoRA by registering forward hooks on target modules,
    ensuring gradients flow through LoRA params while base model stays frozen.
    """

    def __init__(self, rank: int = 8, alpha: float = 16.0):
        self.rank = rank
        self.alpha = alpha
        self.adapters = nn.ModuleDict()
        self.hooks = []
        self._adapter_count = 0

    def add_lora_to_model(self, model: nn.Module, model_name: str = "model",
                          target_modules: list = None) -> int:
        """Add LoRA adapters to attention QKV projections in a model.

        Args:
            model: The model to add LoRA to
            model_name: Prefix for adapter names (e.g., "sam3", "da3")
            target_modules: List of module name patterns to target (default: ['qkv', 'q_proj', 'k_proj', 'v_proj'])

        Returns:
            Number of LoRA adapters added
        """
        if target_modules is None:
            target_modules = ['qkv', 'q_proj', 'k_proj', 'v_proj', 'in_proj']

        count = 0
        for name, module in model.named_modules():
            # Check if this is a target module (attention projection)
            is_target = any(target in name for target in target_modules)
            if not is_target:
                continue

            # Check if it's a Linear layer with weight
            if isinstance(module, nn.Linear):
                in_dim = module.in_features
                out_dim = module.out_features

                # Create unique adapter name
                adapter_name = f"{model_name}_{name.replace('.', '_')}"

                # Create LoRA adapter
                adapter = LoRALayer(in_dim, out_dim, self.rank, self.alpha)
                self.adapters[adapter_name] = adapter

                # Register forward hook to apply LoRA
                hook = self._create_hook(adapter)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)

                count += 1

        self._adapter_count += count
        return count

    def _create_hook(self, adapter: LoRALayer):
        """Create a forward hook that adds LoRA delta to module output."""
        def hook(module, input, output):
            # input is a tuple, get the actual input tensor
            x = input[0]
            # Add LoRA delta to output
            lora_delta = adapter(x)
            return output + lora_delta
        return hook

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def parameters(self):
        """Return all LoRA parameters for optimizer."""
        return self.adapters.parameters()

    def state_dict(self):
        """Return state dict for saving."""
        return self.adapters.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict for resuming."""
        self.adapters.load_state_dict(state_dict)

    def to(self, device):
        """Move adapters to device."""
        self.adapters.to(device)
        return self

    @property
    def num_adapters(self) -> int:
        return self._adapter_count

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.adapters.parameters())
