"""
Activation patching and intervention module
Implements hooks to modify model activations during generation
"""

import torch
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class ActivationPatcher:
    """Patch model activations to test reasoning direction influence"""

    def __init__(
        self,
        model,
        directions: Dict[int, torch.Tensor],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize activation patcher

        Args:
            model: Language model to patch
            directions: Dictionary mapping layer indices to direction vectors
            device: Device for computations
        """
        self.model = model
        self.directions = directions
        self.device = device
        self.hooks = []
        self.active_layers = []

    def create_intervention_hook(
        self,
        layer_idx: int,
        strength: float = 0.1,
        direction: Optional[torch.Tensor] = None
    ) -> Callable:
        """
        Create hook function for activation patching

        Args:
            layer_idx: Layer index to patch
            strength: Intervention strength (can be negative)
            direction: Direction vector (uses stored direction if None)

        Returns:
            Hook function
        """
        if direction is None:
            direction = self.directions.get(layer_idx)

        if direction is None:
            raise ValueError(f"No direction available for layer {layer_idx}")

        direction = direction.to(self.device)

        def hook_fn(module, input, output):
            """
            Hook function to modify activations
            New_Activation = Original_Activation + (Strength * Direction_Vector)
            """
            if isinstance(output, tuple):
                # Extract hidden states
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add direction with specified strength
            modified = hidden_states + (strength * direction.unsqueeze(0))

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            else:
                return modified

        return hook_fn

    def apply_intervention(
        self,
        layers: List[int],
        strength: float = 0.1,
        directions: Optional[Dict[int, torch.Tensor]] = None
    ):
        """
        Apply intervention to specified layers

        Args:
            layers: List of layer indices to patch
            strength: Intervention strength
            directions: Optional custom directions (uses stored if None)
        """
        self.remove_hooks()  # Remove any existing hooks

        if directions is None:
            directions = self.directions

        for layer_idx in layers:
            if layer_idx not in directions:
                print(f"Warning: No direction for layer {layer_idx}, skipping")
                continue

            # Get layer module
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layer_module = self.model.model.layers[layer_idx]
            else:
                print(f"Warning: Could not find layer {layer_idx}")
                continue

            # Create and register hook
            hook_fn = self.create_intervention_hook(layer_idx, strength, directions[layer_idx])
            hook = layer_module.register_forward_hook(hook_fn)

            self.hooks.append(hook)
            self.active_layers.append(layer_idx)

        print(f"Applied interventions to layers: {self.active_layers}")

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active_layers = []

    def generate_with_intervention(
        self,
        tokenizer,
        prompt: str,
        layers: List[int],
        strength: float = 0.1,
        max_new_tokens: int = 512,
        **generation_kwargs
    ) -> str:
        """
        Generate text with intervention applied

        Args:
            tokenizer: Model tokenizer
            prompt: Input prompt
            layers: Layers to intervene on
            strength: Intervention strength
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Apply intervention
        self.apply_intervention(layers, strength)

        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove hooks
        self.remove_hooks()

        return generated_text

    def sweep_layers_and_strengths(
        self,
        tokenizer,
        prompt: str,
        layer_range: Tuple[int, int],
        strength_range: Tuple[float, float],
        num_strengths: int = 10,
        max_new_tokens: int = 512,
        evaluator=None
    ) -> List[Dict]:
        """
        Sweep across layers and intervention strengths

        Args:
            tokenizer: Model tokenizer
            prompt: Input prompt
            layer_range: (min_layer, max_layer) tuple
            strength_range: (min_strength, max_strength) tuple
            num_strengths: Number of strength values to test
            max_new_tokens: Maximum tokens to generate
            evaluator: Optional ReasoningEvaluator for quality metrics

        Returns:
            List of results dictionaries
        """
        results = []
        strengths = np.linspace(strength_range[0], strength_range[1], num_strengths)

        for layer in range(layer_range[0], layer_range[1] + 1):
            if layer not in self.directions:
                continue

            for strength in strengths:
                output = self.generate_with_intervention(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    layers=[layer],
                    strength=float(strength),
                    max_new_tokens=max_new_tokens
                )

                result = {
                    'layer': layer,
                    'strength': float(strength),
                    'output': output,
                    'token_count': len(tokenizer.encode(output))
                }

                # Add quality metrics if evaluator provided
                if evaluator is not None:
                    tokens = evaluator.count_tokens(output, tokenizer, split_by_tags=True)
                    quality = evaluator.analyze_reasoning_quality(output)
                    result['tokens'] = tokens
                    result['quality'] = quality

                results.append(result)

        return results
