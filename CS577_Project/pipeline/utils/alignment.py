"""
Alignment analysis utilities for comparing representations across models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


def centering_matrix(n: int) -> torch.Tensor:
    """Create centering matrix for CKA."""
    return torch.eye(n) - torch.ones(n, n) / n


def linear_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute linear kernel matrix."""
    return X @ X.T


def rbf_kernel(X: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute RBF (Gaussian) kernel matrix.

    Args:
        X: Input matrix (n, d)
        sigma: Kernel width

    Returns:
        Kernel matrix (n, n)
    """
    # Compute pairwise squared distances
    XX = (X * X).sum(dim=1, keepdim=True)
    distances = XX + XX.T - 2 * X @ X.T

    # Apply Gaussian kernel
    K = torch.exp(-distances / (2 * sigma ** 2))

    return K


def compute_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel: str = "linear"
) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two representations.

    Args:
        X: First representation (n, d1)
        Y: Second representation (n, d2)
        kernel: Kernel type ("linear" or "rbf")

    Returns:
        CKA similarity score (0 to 1)
    """
    n = X.shape[0]
    assert Y.shape[0] == n, "X and Y must have same number of samples"

    # Compute kernel matrices
    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    elif kernel == "rbf":
        K = rbf_kernel(X)
        L = rbf_kernel(Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Center the kernel matrices
    H = centering_matrix(n).to(X.device)
    K_c = H @ K @ H
    L_c = H @ L @ H

    # Compute CKA
    numerator = torch.trace(K_c @ L_c)
    denominator = torch.sqrt(torch.trace(K_c @ K_c) * torch.trace(L_c @ L_c))

    if denominator == 0:
        return 0.0

    cka_score = (numerator / denominator).item()

    return cka_score


def compute_layer_cka(
    activations1: Dict[int, torch.Tensor],
    activations2: Dict[int, torch.Tensor],
    kernel: str = "linear"
) -> Dict[Tuple[int, int], float]:
    """
    Compute CKA between all pairs of layers from two models.

    Args:
        activations1: Activations from first model
        activations2: Activations from second model
        kernel: Kernel type

    Returns:
        Dictionary mapping (layer1, layer2) to CKA score
    """
    cka_matrix = {}

    for layer1, acts1 in activations1.items():
        for layer2, acts2 in activations2.items():
            # Flatten if needed
            if acts1.dim() > 2:
                acts1_flat = acts1.reshape(acts1.shape[0], -1)
            else:
                acts1_flat = acts1

            if acts2.dim() > 2:
                acts2_flat = acts2.reshape(acts2.shape[0], -1)
            else:
                acts2_flat = acts2

            # Compute CKA
            cka_score = compute_cka(acts1_flat, acts2_flat, kernel=kernel)
            cka_matrix[(layer1, layer2)] = cka_score

    logger.info(f"Computed CKA for {len(cka_matrix)} layer pairs")

    return cka_matrix


def compute_cosine_similarity(
    vec1: torch.Tensor,
    vec2: torch.Tensor
) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    # Normalize
    vec1_norm = vec1 / (vec1.norm() + 1e-8)
    vec2_norm = vec2 / (vec2.norm() + 1e-8)

    # Compute dot product
    similarity = (vec1_norm * vec2_norm).sum().item()

    return similarity


def compute_direction_alignment(
    directions1: Dict[int, torch.Tensor],
    directions2: Dict[int, torch.Tensor]
) -> Dict[int, float]:
    """
    Compute cosine similarity between direction vectors for each layer.

    Args:
        directions1: Direction vectors from first model
        directions2: Direction vectors from second model

    Returns:
        Dictionary mapping layer indices to cosine similarity
    """
    similarities = {}

    common_layers = set(directions1.keys()) & set(directions2.keys())

    for layer in sorted(common_layers):
        sim = compute_cosine_similarity(directions1[layer], directions2[layer])
        similarities[layer] = sim

    logger.info(f"Computed direction alignment for {len(similarities)} layers")

    return similarities


def analyze_cross_model_alignment(
    model1_activations: Dict[int, torch.Tensor],
    model2_activations: Dict[int, torch.Tensor],
    model1_directions: Optional[Dict[int, torch.Tensor]] = None,
    model2_directions: Optional[Dict[int, torch.Tensor]] = None,
    kernel: str = "linear"
) -> Dict[str, any]:
    """
    Comprehensive cross-model alignment analysis.

    Args:
        model1_activations: Activations from model 1
        model2_activations: Activations from model 2
        model1_directions: Direction vectors from model 1
        model2_directions: Direction vectors from model 2
        kernel: CKA kernel type

    Returns:
        Dictionary with alignment analysis results
    """
    results = {}

    # 1. Compute full CKA matrix
    cka_matrix = compute_layer_cka(
        model1_activations,
        model2_activations,
        kernel=kernel
    )
    results["cka_matrix"] = cka_matrix

    # 2. Find best aligned layers (max CKA for each layer)
    layer1_indices = sorted(set(l1 for l1, _ in cka_matrix.keys()))
    layer2_indices = sorted(set(l2 for _, l2 in cka_matrix.keys()))

    best_alignments = {}
    for l1 in layer1_indices:
        scores = [(l2, cka_matrix[(l1, l2)]) for l2 in layer2_indices]
        best_l2, best_score = max(scores, key=lambda x: x[1])
        best_alignments[l1] = {"aligned_layer": best_l2, "cka_score": best_score}

    results["best_layer_alignments"] = best_alignments

    # 3. Compute diagonal CKA (corresponding layers)
    common_layers = set(layer1_indices) & set(layer2_indices)
    diagonal_cka = {l: cka_matrix[(l, l)] for l in sorted(common_layers)}
    results["diagonal_cka"] = diagonal_cka
    results["mean_diagonal_cka"] = np.mean(list(diagonal_cka.values()))

    # 4. Direction alignment if provided
    if model1_directions is not None and model2_directions is not None:
        direction_similarities = compute_direction_alignment(
            model1_directions,
            model2_directions
        )
        results["direction_similarities"] = direction_similarities
        results["mean_direction_similarity"] = np.mean(list(direction_similarities.values()))

    # 5. Compute statistics
    all_cka_scores = list(cka_matrix.values())
    results["cka_statistics"] = {
        "mean": np.mean(all_cka_scores),
        "std": np.std(all_cka_scores),
        "min": np.min(all_cka_scores),
        "max": np.max(all_cka_scores),
        "median": np.median(all_cka_scores)
    }

    logger.info(f"Cross-model alignment analysis complete. Mean CKA: {results['cka_statistics']['mean']:.4f}")

    return results


def compute_representational_drift(
    baseline_activations: Dict[int, torch.Tensor],
    current_activations: Dict[int, torch.Tensor]
) -> Dict[int, float]:
    """
    Compute representational drift between baseline and current activations.

    Args:
        baseline_activations: Baseline activations (e.g., from RL model)
        current_activations: Current activations (e.g., from distilled model)

    Returns:
        Dictionary mapping layer indices to drift scores (0 = no drift, higher = more drift)
    """
    drift_scores = {}

    common_layers = set(baseline_activations.keys()) & set(current_activations.keys())

    for layer in sorted(common_layers):
        baseline = baseline_activations[layer]
        current = current_activations[layer]

        # Flatten if needed
        if baseline.dim() > 2:
            baseline = baseline.reshape(baseline.shape[0], -1)
        if current.dim() > 2:
            current = current.reshape(current.shape[0], -1)

        # Compute mean representations
        baseline_mean = baseline.mean(dim=0)
        current_mean = current.mean(dim=0)

        # Compute drift as normalized distance
        distance = (baseline_mean - current_mean).norm()
        baseline_norm = baseline_mean.norm()

        drift = (distance / (baseline_norm + 1e-8)).item()
        drift_scores[layer] = drift

    logger.info(f"Computed representational drift for {len(drift_scores)} layers")

    return drift_scores


def find_corresponding_layers(
    cka_matrix: Dict[Tuple[int, int], float],
    threshold: float = 0.5
) -> Dict[int, List[int]]:
    """
    Find corresponding layers between two models based on CKA scores.

    Args:
        cka_matrix: CKA scores for all layer pairs
        threshold: Minimum CKA score to consider alignment

    Returns:
        Dictionary mapping model1 layers to list of aligned model2 layers
    """
    layer1_indices = sorted(set(l1 for l1, _ in cka_matrix.keys()))
    layer2_indices = sorted(set(l2 for _, l2 in cka_matrix.keys()))

    correspondences = {}

    for l1 in layer1_indices:
        aligned = []
        for l2 in layer2_indices:
            if cka_matrix[(l1, l2)] >= threshold:
                aligned.append((l2, cka_matrix[(l1, l2)]))

        # Sort by CKA score
        aligned.sort(key=lambda x: x[1], reverse=True)
        correspondences[l1] = [l2 for l2, _ in aligned]

    return correspondences
