"""
Visualization utilities for reasoning direction analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_intervention_effects(
    intervention_strengths: List[float],
    metrics: Dict[float, float],
    metric_name: str = "Reasoning Score",
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Plot the effect of intervention strength on a metric.

    Args:
        intervention_strengths: List of intervention strengths
        metrics: Dictionary mapping strength to metric value
        metric_name: Name of the metric
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    strengths = sorted(intervention_strengths)
    values = [metrics.get(s, 0) for s in strengths]

    plt.plot(strengths, values, marker='o', linewidth=2, markersize=8)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')

    plt.xlabel('Intervention Strength', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(title or f'{metric_name} vs Intervention Strength', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_layer_alignment(
    cka_matrix: Dict[Tuple[int, int], float],
    model1_name: str = "RL Model",
    model2_name: str = "Distilled Model",
    save_path: Optional[str] = None
):
    """
    Plot CKA alignment matrix between two models.

    Args:
        cka_matrix: Dictionary mapping (layer1, layer2) to CKA score
        model1_name: Name of first model
        model2_name: Name of second model
        save_path: Path to save figure
    """
    # Extract layer indices
    layer1_indices = sorted(set(l1 for l1, _ in cka_matrix.keys()))
    layer2_indices = sorted(set(l2 for _, l2 in cka_matrix.keys()))

    # Create matrix
    matrix = np.zeros((len(layer1_indices), len(layer2_indices)))
    for i, l1 in enumerate(layer1_indices):
        for j, l2 in enumerate(layer2_indices):
            matrix[i, j] = cka_matrix.get((l1, l2), 0)

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=layer2_indices,
        yticklabels=layer1_indices,
        cmap='viridis',
        cbar_kws={'label': 'CKA Score'},
        vmin=0,
        vmax=1,
        annot=False
    )

    plt.xlabel(f'{model2_name} Layers', fontsize=12)
    plt.ylabel(f'{model1_name} Layers', fontsize=12)
    plt.title(f'Layer Alignment: {model1_name} vs {model2_name}', fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_probe_accuracy(
    layer_accuracies: Dict[int, float],
    model_name: str = "Model",
    save_path: Optional[str] = None,
    comparison_accuracies: Optional[Dict[int, float]] = None,
    comparison_name: Optional[str] = None
):
    """
    Plot probe accuracy across layers.

    Args:
        layer_accuracies: Dictionary mapping layer indices to accuracy
        model_name: Name of the model
        save_path: Path to save figure
        comparison_accuracies: Optional second model accuracies for comparison
        comparison_name: Name of comparison model
    """
    plt.figure(figsize=(12, 6))

    layers = sorted(layer_accuracies.keys())
    accuracies = [layer_accuracies[l] for l in layers]

    plt.plot(layers, accuracies, marker='o', linewidth=2, markersize=8, label=model_name)

    if comparison_accuracies is not None:
        comp_layers = sorted(comparison_accuracies.keys())
        comp_accs = [comparison_accuracies[l] for l in comp_layers]
        plt.plot(comp_layers, comp_accs, marker='s', linewidth=2, markersize=8,
                label=comparison_name or "Comparison Model")

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Probe Accuracy', fontsize=12)
    plt.title('Linear Probe Accuracy Across Layers', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_direction_similarity(
    direction_similarities: Dict[int, float],
    save_path: Optional[str] = None,
    title: str = "Reasoning Direction Similarity Across Layers"
):
    """
    Plot direction vector similarities across layers.

    Args:
        direction_similarities: Dictionary mapping layer to cosine similarity
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    layers = sorted(direction_similarities.keys())
    similarities = [direction_similarities[l] for l in layers]

    plt.plot(layers, similarities, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([-1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_ablation_results(
    ablation_results: Dict[int, Dict[str, float]],
    metric_name: str = "Performance",
    save_path: Optional[str] = None
):
    """
    Plot results of layer ablation experiments.

    Args:
        ablation_results: Dictionary mapping layer to metrics dict
        metric_name: Name of the metric being plotted
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))

    layers = sorted(ablation_results.keys())

    # Collect different ablation methods
    methods = set()
    for result in ablation_results.values():
        methods.update(result.keys())

    # Plot each method
    for method in sorted(methods):
        values = [ablation_results[l].get(method, 0) for l in layers]
        plt.plot(layers, values, marker='o', linewidth=2, markersize=6, label=method)

    plt.xlabel('Layer Ablated', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Layer Ablation Effects on {metric_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_multiple_models_comparison(
    model_results: Dict[str, Dict[int, float]],
    metric_name: str = "Metric",
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Plot comparison of multiple models across layers.

    Args:
        model_results: Dictionary mapping model name to layer results
        metric_name: Name of the metric
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(14, 6))

    for model_name, layer_values in model_results.items():
        layers = sorted(layer_values.keys())
        values = [layer_values[l] for l in layers]
        plt.plot(layers, values, marker='o', linewidth=2, markersize=6, label=model_name)

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(title or f'{metric_name} Comparison Across Models', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def create_summary_figure(
    probe_results: Dict[str, Dict[int, float]],
    direction_similarities: Dict[int, float],
    cka_diagonal: Dict[int, float],
    save_path: Optional[str] = None
):
    """
    Create a comprehensive summary figure with multiple subplots.

    Args:
        probe_results: Probe accuracies for different models
        direction_similarities: Direction similarities across layers
        cka_diagonal: Diagonal CKA scores
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Probe accuracies
    ax = axes[0, 0]
    for model_name, layer_accs in probe_results.items():
        layers = sorted(layer_accs.keys())
        accs = [layer_accs[l] for l in layers]
        ax.plot(layers, accs, marker='o', linewidth=2, label=model_name)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Probe Accuracy')
    ax.set_title('Linear Probe Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Direction similarities
    ax = axes[0, 1]
    layers = sorted(direction_similarities.keys())
    sims = [direction_similarities[l] for l in layers]
    ax.plot(layers, sims, marker='o', linewidth=2, color='green')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Direction Vector Alignment')
    ax.grid(True, alpha=0.3)

    # 3. CKA diagonal
    ax = axes[1, 0]
    layers = sorted(cka_diagonal.keys())
    cka_scores = [cka_diagonal[l] for l in layers]
    ax.plot(layers, cka_scores, marker='o', linewidth=2, color='orange')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('CKA Score')
    ax.set_title('Layer-wise Representation Alignment (CKA)')
    ax.grid(True, alpha=0.3)

    # 4. Summary statistics (text)
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "Summary Statistics\n\n"
    summary_text += f"Mean Direction Similarity: {np.mean(sims):.3f}\n"
    summary_text += f"Mean CKA Score: {np.mean(cka_scores):.3f}\n\n"

    for model_name, layer_accs in probe_results.items():
        mean_acc = np.mean(list(layer_accs.values()))
        summary_text += f"{model_name} Mean Probe Acc: {mean_acc:.3f}\n"

    ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
           family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved summary figure to {save_path}")

    plt.show()
