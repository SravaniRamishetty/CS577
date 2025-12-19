"""
Logit lens utilities for decoding intermediate layer states.
"""

import torch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LogitLens:
    """
    Apply logit lens to decode intermediate layer representations.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize logit lens.

        Args:
            model: The language model
            tokenizer: The tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

        # Get the output embedding/unembedding layer
        # This varies by model architecture
        if hasattr(model, 'lm_head'):
            self.unembed = model.lm_head
        elif hasattr(model, 'embed_out'):
            self.unembed = model.embed_out
        else:
            # Try to find it automatically
            for name, module in model.named_modules():
                if 'lm_head' in name.lower() or 'embed_out' in name.lower():
                    self.unembed = module
                    break
            else:
                raise ValueError("Could not find unembedding layer")

        # Get layer norm if it exists
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            self.final_norm = model.model.norm
        elif hasattr(model, 'norm'):
            self.final_norm = model.norm
        else:
            self.final_norm = None

        logger.info("LogitLens initialized")

    def decode_activation(
        self,
        activation: torch.Tensor,
        top_k: int = 10,
        apply_norm: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Decode an activation vector to top-k tokens.

        Args:
            activation: Activation vector (hidden_size,)
            top_k: Number of top tokens to return
            apply_norm: Whether to apply final layer norm

        Returns:
            List of (token, probability) tuples
        """
        with torch.no_grad():
            # Apply final layer norm if requested and available
            if apply_norm and self.final_norm is not None:
                activation = self.final_norm(activation)

            # Project through unembedding
            logits = self.unembed(activation)

            # Apply softmax
            probs = torch.softmax(logits, dim=-1)

            # Get top-k
            top_probs, top_indices = torch.topk(probs, k=top_k)

            # Decode tokens
            results = []
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.decode([idx.item()])
                results.append((token, prob.item()))

        return results

    def decode_layer_activations(
        self,
        layer_activations: Dict[int, torch.Tensor],
        position: int = -1,
        top_k: int = 10,
        apply_norm: bool = True
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Decode activations from multiple layers.

        Args:
            layer_activations: Dictionary mapping layer indices to activations
            position: Token position to decode (-1 = last)
            top_k: Number of top tokens per layer
            apply_norm: Whether to apply layer norm

        Returns:
            Dictionary mapping layer indices to decoded tokens
        """
        results = {}

        for layer_idx, activation in layer_activations.items():
            # Handle batched activations
            if activation.dim() == 3:
                # Shape: (batch, seq_len, hidden_size)
                act = activation[0, position, :]
            elif activation.dim() == 2:
                # Shape: (seq_len, hidden_size)
                act = activation[position, :]
            else:
                # Shape: (hidden_size,)
                act = activation

            # Decode
            decoded = self.decode_activation(act, top_k=top_k, apply_norm=apply_norm)
            results[layer_idx] = decoded

        return results


def decode_layer_states(
    model,
    tokenizer,
    layer_activations: Dict[int, torch.Tensor],
    position: int = -1,
    top_k: int = 5
) -> Dict[int, List[str]]:
    """
    Convenience function to decode layer states.

    Args:
        model: The model
        tokenizer: The tokenizer
        layer_activations: Layer activations
        position: Token position
        top_k: Number of top predictions

    Returns:
        Dictionary mapping layer indices to top token predictions
    """
    lens = LogitLens(model, tokenizer)
    decoded = lens.decode_layer_activations(
        layer_activations,
        position=position,
        top_k=top_k
    )

    # Simplify to just tokens
    results = {}
    for layer_idx, token_probs in decoded.items():
        results[layer_idx] = [token for token, _ in token_probs]

    return results


def analyze_reasoning_emergence(
    model,
    tokenizer,
    reasoning_activations: Dict[int, torch.Tensor],
    target_tokens: List[str],
    position: int = -1,
    top_k: int = 20
) -> Dict[int, Dict]:
    """
    Analyze when reasoning-related tokens emerge across layers.

    Args:
        model: The model
        tokenizer: The tokenizer
        reasoning_activations: Activations from reasoning task
        target_tokens: Tokens to track (e.g., numbers, operations)
        position: Token position
        top_k: Number of top predictions to check

    Returns:
        Dictionary with analysis results per layer
    """
    lens = LogitLens(model, tokenizer)
    decoded = lens.decode_layer_activations(
        reasoning_activations,
        position=position,
        top_k=top_k
    )

    results = {}

    for layer_idx, token_probs in decoded.items():
        # Find target tokens in predictions
        target_ranks = {}
        target_probs = {}

        for rank, (token, prob) in enumerate(token_probs):
            token_clean = token.strip().lower()
            for target in target_tokens:
                if target.lower() in token_clean:
                    if target not in target_ranks:
                        target_ranks[target] = rank
                        target_probs[target] = prob

        results[layer_idx] = {
            "top_tokens": [t for t, _ in token_probs[:5]],
            "target_ranks": target_ranks,
            "target_probs": target_probs,
            "has_targets": len(target_ranks) > 0
        }

    return results


def compare_decoded_states(
    model1, tokenizer1, activations1: Dict[int, torch.Tensor],
    model2, tokenizer2, activations2: Dict[int, torch.Tensor],
    position: int = -1,
    top_k: int = 10
) -> Dict[int, Dict]:
    """
    Compare decoded states between two models.

    Args:
        model1: First model
        tokenizer1: First tokenizer
        activations1: Activations from first model
        model2: Second model
        tokenizer2: Second tokenizer
        activations2: Activations from second model
        position: Token position
        top_k: Number of top predictions

    Returns:
        Comparison results per layer
    """
    lens1 = LogitLens(model1, tokenizer1)
    lens2 = LogitLens(model2, tokenizer2)

    decoded1 = lens1.decode_layer_activations(activations1, position=position, top_k=top_k)
    decoded2 = lens2.decode_layer_activations(activations2, position=position, top_k=top_k)

    common_layers = set(decoded1.keys()) & set(decoded2.keys())

    results = {}

    for layer_idx in sorted(common_layers):
        tokens1 = [t for t, _ in decoded1[layer_idx]]
        tokens2 = [t for t, _ in decoded2[layer_idx]]

        # Compute overlap
        set1 = set(tokens1)
        set2 = set(tokens2)
        overlap = len(set1 & set2)
        jaccard = overlap / len(set1 | set2) if len(set1 | set2) > 0 else 0.0

        # Check top-1 agreement
        top1_match = tokens1[0] == tokens2[0] if tokens1 and tokens2 else False

        results[layer_idx] = {
            "model1_top": tokens1[:5],
            "model2_top": tokens2[:5],
            "overlap_count": overlap,
            "jaccard_similarity": jaccard,
            "top1_match": top1_match
        }

    logger.info(f"Compared decoded states across {len(results)} layers")

    return results
