"""
Activation collection utilities for extracting layer-wise representations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActivationCache:
    """Container for cached activations."""
    activations: Dict[str, torch.Tensor]
    layer_names: List[str]
    num_tokens: int

    def get_layer_activation(self, layer_idx: int) -> torch.Tensor:
        """Get activation for a specific layer index."""
        layer_name = self.layer_names[layer_idx]
        return self.activations[layer_name]

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert all activations to numpy arrays."""
        return {k: v.cpu().numpy() for k, v in self.activations.items()}


class ActivationCollector:
    """
    Collect activations from specific layers during forward pass.
    """

    def __init__(
        self,
        model,
        layer_indices: Optional[List[int]] = None,
        activation_type: str = "residual"
    ):
        """
        Initialize activation collector.

        Args:
            model: The model to collect activations from
            layer_indices: Specific layer indices to collect (None = all layers)
            activation_type: Type of activation to collect ("residual", "mlp", "attention")
        """
        self.model = model
        self.activation_type = activation_type
        self.activations = {}
        self.hooks = []

        # Determine which layers to hook
        if layer_indices is None:
            self.layer_indices = list(range(model.config.num_hidden_layers))
        else:
            self.layer_indices = layer_indices

        logger.info(f"ActivationCollector initialized for {len(self.layer_indices)} layers")

    def _get_hook_fn(self, name: str) -> Callable:
        """
        Create a hook function for a specific layer.

        Args:
            name: Name identifier for the layer

        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            # Store the activation
            if isinstance(output, tuple):
                # For attention layers, output is typically (hidden_states, ...)
                activation = output[0]
            else:
                activation = output

            # Detach and clone to avoid gradient issues
            self.activations[name] = activation.detach().clone()

        return hook_fn

    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        # Clear existing hooks
        self.remove_hooks()

        # Determine layer pattern based on model type
        # This needs to be adapted for specific model architectures
        for layer_idx in self.layer_indices:
            if self.activation_type == "residual":
                # Hook the output of each transformer layer
                layer_name = f"model.layers.{layer_idx}"
            elif self.activation_type == "mlp":
                layer_name = f"model.layers.{layer_idx}.mlp"
            elif self.activation_type == "attention":
                layer_name = f"model.layers.{layer_idx}.self_attn"
            else:
                raise ValueError(f"Unknown activation type: {self.activation_type}")

            # Get the actual module
            try:
                module = self.model.get_submodule(layer_name)
                hook = module.register_forward_hook(self._get_hook_fn(layer_name))
                self.hooks.append(hook)
                logger.debug(f"Registered hook on {layer_name}")
            except AttributeError:
                logger.warning(f"Could not find module {layer_name}, skipping")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("Removed all hooks")

    def collect(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> ActivationCache:
        """
        Collect activations for a given input.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            ActivationCache with collected activations
        """
        # Register hooks
        self._register_hooks()

        # Clear previous activations
        self.activations = {}

        # Forward pass
        with torch.no_grad():
            if attention_mask is not None:
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                _ = self.model(input_ids=input_ids)

        # Create activation cache
        layer_names = sorted(self.activations.keys())
        num_tokens = input_ids.shape[1]

        cache = ActivationCache(
            activations=self.activations.copy(),
            layer_names=layer_names,
            num_tokens=num_tokens
        )

        # Remove hooks after collection
        self.remove_hooks()

        return cache

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


def collect_activations(
    model,
    tokenizer,
    texts: List[str],
    layer_indices: Optional[List[int]] = None,
    activation_type: str = "residual",
    batch_size: int = 1,
    device: str = "cuda",
    position: str = "last"  # "last", "mean", "all"
) -> Dict[int, torch.Tensor]:
    """
    Collect activations for a list of texts.

    Args:
        model: The model
        tokenizer: The tokenizer
        texts: List of input texts
        layer_indices: Specific layers to collect
        activation_type: Type of activation
        batch_size: Batch size for processing
        device: Device to use
        position: Which token position to extract ("last", "mean", "all")

    Returns:
        Dictionary mapping layer indices to activation tensors
    """
    collector = ActivationCollector(
        model=model,
        layer_indices=layer_indices,
        activation_type=activation_type
    )

    all_activations = {idx: [] for idx in (layer_indices or range(model.config.num_hidden_layers))}

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Collect activations
        cache = collector.collect(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        # Extract activations based on position
        for layer_idx in all_activations.keys():
            activation = cache.get_layer_activation(layer_idx)

            if position == "last":
                # Get last token position for each sequence
                seq_lengths = inputs["attention_mask"].sum(dim=1)
                batch_acts = []
                for b in range(activation.shape[0]):
                    last_pos = seq_lengths[b] - 1
                    batch_acts.append(activation[b, last_pos])
                act = torch.stack(batch_acts)
            elif position == "mean":
                # Mean pool over sequence length
                act = activation.mean(dim=1)
            elif position == "all":
                # Keep all positions
                act = activation
            else:
                raise ValueError(f"Unknown position: {position}")

            all_activations[layer_idx].append(act)

    # Concatenate all batches
    result = {}
    for layer_idx, acts in all_activations.items():
        if position == "all":
            result[layer_idx] = torch.cat(acts, dim=0)
        else:
            result[layer_idx] = torch.cat(acts, dim=0)

    logger.info(f"Collected activations for {len(texts)} texts across {len(result)} layers")

    return result


def compute_contrastive_directions(
    reasoning_activations: Dict[int, torch.Tensor],
    control_activations: Dict[int, torch.Tensor],
    normalize: bool = True
) -> Dict[int, torch.Tensor]:
    """
    Compute contrastive directions by taking mean differences.

    Args:
        reasoning_activations: Activations on reasoning tasks
        control_activations: Activations on control tasks
        normalize: Whether to normalize the directions

    Returns:
        Dictionary mapping layer indices to direction vectors
    """
    directions = {}

    for layer_idx in reasoning_activations.keys():
        if layer_idx not in control_activations:
            logger.warning(f"Layer {layer_idx} not in control activations, skipping")
            continue

        # Compute mean activations
        reasoning_mean = reasoning_activations[layer_idx].mean(dim=0)
        control_mean = control_activations[layer_idx].mean(dim=0)

        # Compute direction
        direction = reasoning_mean - control_mean

        # Normalize if requested
        if normalize:
            direction = direction / (direction.norm() + 1e-8)

        directions[layer_idx] = direction

    logger.info(f"Computed contrastive directions for {len(directions)} layers")

    return directions
