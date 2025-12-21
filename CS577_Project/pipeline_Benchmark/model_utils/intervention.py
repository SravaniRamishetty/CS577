"""
Intervention utilities for adding/subtracting reasoning directions during generation.
"""

import torch
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class InterventionHandler:
    """
    Handle interventions on model activations during forward pass.
    """

    def __init__(
        self,
        model,
        directions: Dict[int, torch.Tensor],
        layer_indices: Optional[List[int]] = None,
        intervention_strength: float = 1.0
    ):
        """
        Initialize intervention handler.

        Args:
            model: The model to intervene on
            directions: Dictionary mapping layer indices to direction vectors
            layer_indices: Specific layers to intervene on (None = all layers with directions)
            intervention_strength: Strength of intervention (positive = add, negative = subtract)
        """
        self.model = model
        self.directions = directions
        self.intervention_strength = intervention_strength
        self.hooks = []

        # Determine which layers to intervene on
        if layer_indices is None:
            self.layer_indices = list(directions.keys())
        else:
            self.layer_indices = [idx for idx in layer_indices if idx in directions]

        logger.info(f"InterventionHandler initialized for {len(self.layer_indices)} layers "
                   f"with strength {intervention_strength}")

    def _get_intervention_hook(self, layer_idx: int) -> Callable:
        """
        Create an intervention hook for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Hook function
        """
        direction = self.directions[layer_idx]

        def hook_fn(module, input, output):
            # Handle tuple outputs (e.g., from attention layers)
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Add the intervention
            # direction shape: (hidden_size,)
            # hidden_states shape: (batch_size, seq_len, hidden_size)
            intervention = self.intervention_strength * direction.to(hidden_states.device)
            intervened = hidden_states + intervention

            # Return in the same format as input
            if rest is not None:
                return (intervened,) + rest
            else:
                return intervened

        return hook_fn

    def register_hooks(self):
        """Register intervention hooks on specified layers."""
        self.remove_hooks()

        for layer_idx in self.layer_indices:
            layer_name = f"model.layers.{layer_idx}"

            try:
                module = self.model.get_submodule(layer_name)
                hook = module.register_forward_hook(self._get_intervention_hook(layer_idx))
                self.hooks.append(hook)
                logger.debug(f"Registered intervention hook on layer {layer_idx}")
            except AttributeError:
                logger.warning(f"Could not find module {layer_name}, skipping")

    def remove_hooks(self):
        """Remove all intervention hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("Removed all intervention hooks")

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


def apply_direction_intervention(
    model,
    tokenizer,
    prompt: str,
    directions: Dict[int, torch.Tensor],
    layer_indices: Optional[List[int]] = None,
    intervention_strength: float = 1.0,
    max_new_tokens: int = 512,
    **generation_kwargs
) -> str:
    """
    Generate text with direction intervention.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        directions: Direction vectors for each layer
        layer_indices: Layers to intervene on
        intervention_strength: Intervention strength
        max_new_tokens: Maximum tokens to generate
        **generation_kwargs: Additional generation parameters

    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Apply intervention
    with InterventionHandler(
        model=model,
        directions=directions,
        layer_indices=layer_indices,
        intervention_strength=intervention_strength
    ):
        # Generate with intervention active
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs
            )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text


class LayerAblator:
    """
    Ablate (zero out or add noise to) specific layers.
    """

    def __init__(
        self,
        model,
        layer_indices: List[int],
        ablation_method: str = "zero",
        noise_std: float = 0.1
    ):
        """
        Initialize layer ablator.

        Args:
            model: The model
            layer_indices: Layers to ablate
            ablation_method: "zero", "noise", or "mean"
            noise_std: Standard deviation for noise ablation
        """
        self.model = model
        self.layer_indices = layer_indices
        self.ablation_method = ablation_method
        self.noise_std = noise_std
        self.hooks = []
        self.mean_activations = {}

        logger.info(f"LayerAblator initialized for layers {layer_indices} "
                   f"with method '{ablation_method}'")

    def _get_ablation_hook(self, layer_idx: int) -> Callable:
        """
        Create an ablation hook for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            # Handle tuple outputs
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Apply ablation
            if self.ablation_method == "zero":
                ablated = torch.zeros_like(hidden_states)
            elif self.ablation_method == "noise":
                noise = torch.randn_like(hidden_states) * self.noise_std
                ablated = noise
            elif self.ablation_method == "mean":
                if layer_idx in self.mean_activations:
                    mean_act = self.mean_activations[layer_idx]
                    ablated = mean_act.expand_as(hidden_states)
                else:
                    logger.warning(f"No mean activation for layer {layer_idx}, using zeros")
                    ablated = torch.zeros_like(hidden_states)
            else:
                raise ValueError(f"Unknown ablation method: {self.ablation_method}")

            # Return in same format
            if rest is not None:
                return (ablated,) + rest
            else:
                return ablated

        return hook_fn

    def set_mean_activations(self, mean_activations: Dict[int, torch.Tensor]):
        """Set mean activations for mean ablation method."""
        self.mean_activations = mean_activations

    def register_hooks(self):
        """Register ablation hooks."""
        self.remove_hooks()

        for layer_idx in self.layer_indices:
            layer_name = f"model.layers.{layer_idx}"

            try:
                module = self.model.get_submodule(layer_name)
                hook = module.register_forward_hook(self._get_ablation_hook(layer_idx))
                self.hooks.append(hook)
                logger.debug(f"Registered ablation hook on layer {layer_idx}")
            except AttributeError:
                logger.warning(f"Could not find module {layer_name}, skipping")

    def remove_hooks(self):
        """Remove all ablation hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        """Context manager entry."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove_hooks()

    def __del__(self):
        """Cleanup."""
        self.remove_hooks()


class PathPatcher:
    """
    Patch activations by swapping them between two runs.
    """

    def __init__(
        self,
        model,
        source_activations: Dict[int, torch.Tensor],
        layer_indices: List[int]
    ):
        """
        Initialize path patcher.

        Args:
            model: The model
            source_activations: Activations to patch in
            layer_indices: Layers to patch
        """
        self.model = model
        self.source_activations = source_activations
        self.layer_indices = layer_indices
        self.hooks = []

        logger.info(f"PathPatcher initialized for {len(layer_indices)} layers")

    def _get_patch_hook(self, layer_idx: int) -> Callable:
        """Create a patching hook."""
        source_act = self.source_activations[layer_idx]

        def hook_fn(module, input, output):
            # Handle tuple outputs
            if isinstance(output, tuple):
                rest = output[1:]
            else:
                rest = None

            # Use source activation
            patched = source_act.to(output[0].device if isinstance(output, tuple) else output.device)

            # Return in same format
            if rest is not None:
                return (patched,) + rest
            else:
                return patched

        return hook_fn

    def register_hooks(self):
        """Register patching hooks."""
        self.remove_hooks()

        for layer_idx in self.layer_indices:
            if layer_idx not in self.source_activations:
                logger.warning(f"No source activation for layer {layer_idx}, skipping")
                continue

            layer_name = f"model.layers.{layer_idx}"

            try:
                module = self.model.get_submodule(layer_name)
                hook = module.register_forward_hook(self._get_patch_hook(layer_idx))
                self.hooks.append(hook)
                logger.debug(f"Registered patch hook on layer {layer_idx}")
            except AttributeError:
                logger.warning(f"Could not find module {layer_name}, skipping")

    def remove_hooks(self):
        """Remove all patching hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        """Context manager entry."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove_hooks()

    def __del__(self):
        """Cleanup."""
        self.remove_hooks()
