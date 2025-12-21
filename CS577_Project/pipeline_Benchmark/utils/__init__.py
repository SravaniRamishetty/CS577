"""
Utility functions for the reasoning direction pipeline.
"""

from .data_utils import load_dataset, create_control_dataset, prepare_prompts
from .probe import LinearProbe, train_probe, evaluate_probe
from .logit_lens import LogitLens, decode_layer_states
from .alignment import compute_cka, compute_cosine_similarity, analyze_cross_model_alignment
from .visualization import plot_intervention_effects, plot_layer_alignment, plot_probe_accuracy

__all__ = [
    # Data utilities
    "load_dataset",
    "create_control_dataset",
    "prepare_prompts",
    # Probing
    "LinearProbe",
    "train_probe",
    "evaluate_probe",
    # Logit lens
    "LogitLens",
    "decode_layer_states",
    # Alignment
    "compute_cka",
    "compute_cosine_similarity",
    "analyze_cross_model_alignment",
    # Visualization
    "plot_intervention_effects",
    "plot_layer_alignment",
    "plot_probe_accuracy",
]
