"""
Direction calculation module
Computes reasoning direction from activation differences between RL and distilled models
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.nn.functional import normalize


class DirectionCalculator:
    """Calculate and normalize reasoning directions from model activations"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize direction calculator

        Args:
            device: Device for tensor computations
        """
        self.device = device
        self.directions = {}

    def capture_activations(
        self,
        model,
        tokenizer,
        prompts: List[str],
        layers: List[int],
        position: str = "residual"
    ) -> Dict[int, torch.Tensor]:
        """
        Capture activations at specified layers

        Args:
            model: Language model
            tokenizer: Model tokenizer
            prompts: List of input prompts
            layers: Layer indices to capture activations from
            position: Activation position (residual, pre_attn, post_attn, mid_attn)

        Returns:
            Dictionary mapping layer indices to activation tensors
        """
        activations = {layer: [] for layer in layers}

        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # Hook to capture activations
                def hook_fn(module, input, output, layer_idx):
                    if isinstance(output, tuple):
                        activation = output[0]  # Usually hidden states are first
                    else:
                        activation = output

                    # Take mean over sequence dimension
                    activations[layer_idx].append(activation.mean(dim=1).cpu())

                # Register hooks
                hooks = []
                for layer_idx in layers:
                    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                        layer_module = model.model.layers[layer_idx]
                    else:
                        # Fallback for different model architectures
                        continue

                    hook = layer_module.register_forward_hook(
                        lambda m, i, o, idx=layer_idx: hook_fn(m, i, o, idx)
                    )
                    hooks.append(hook)

                # Forward pass
                _ = model(**inputs)

                # Remove hooks
                for hook in hooks:
                    hook.remove()

        # Average activations across prompts
        mean_activations = {}
        for layer_idx, acts in activations.items():
            if acts:
                mean_activations[layer_idx] = torch.stack(acts).mean(dim=0)

        return mean_activations

    def calculate_direction(
        self,
        rl_activations: Dict[int, torch.Tensor],
        distilled_activations: Dict[int, torch.Tensor],
        normalize_output: bool = True
    ) -> Dict[int, torch.Tensor]:
        """
        Calculate reasoning direction as difference between RL and distilled activations

        Args:
            rl_activations: Activations from RL-trained model
            distilled_activations: Activations from distilled model
            normalize_output: Whether to normalize direction vectors

        Returns:
            Dictionary mapping layer indices to direction vectors
        """
        directions = {}

        for layer_idx in rl_activations.keys():
            if layer_idx in distilled_activations:
                # Calculate difference
                direction = rl_activations[layer_idx] - distilled_activations[layer_idx]

                # Normalize if requested
                if normalize_output:
                    direction = normalize(direction, p=2, dim=-1)

                directions[layer_idx] = direction.to(self.device)

        self.directions = directions
        return directions

    def get_direction(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get direction vector for a specific layer"""
        return self.directions.get(layer_idx)

    def save_directions(self, filepath: str):
        """Save calculated directions to file"""
        torch.save(self.directions, filepath)
        print(f"Directions saved to {filepath}")

    def load_directions(self, filepath: str):
        """Load directions from file"""
        self.directions = torch.load(filepath, map_location=self.device)
        print(f"Directions loaded from {filepath}")

    def compute_direction_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Compute statistics about direction vectors

        Returns:
            Dictionary with norm, mean, and std for each layer
        """
        stats = {}

        for layer_idx, direction in self.directions.items():
            stats[layer_idx] = {
                'norm': direction.norm().item(),
                'mean': direction.mean().item(),
                'std': direction.std().item(),
                'max': direction.max().item(),
                'min': direction.min().item()
            }

        return stats
