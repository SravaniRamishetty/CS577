"""
Linear probing utilities for analyzing intermediate representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """
    Simple linear probe for classification tasks.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None
    ):
        """
        Initialize linear probe.

        Args:
            input_dim: Input dimension (hidden size)
            num_classes: Number of output classes
            hidden_dims: Optional hidden layer dimensions (None = single linear layer)
        """
        super().__init__()

        layers = []

        if hidden_dims is None or len(hidden_dims) == 0:
            # Simple linear probe
            layers.append(nn.Linear(input_dim, num_classes))
        else:
            # MLP probe
            current_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                current_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(current_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


def train_probe(
    probe: LinearProbe,
    train_activations: torch.Tensor,
    train_labels: torch.Tensor,
    val_activations: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 32,
    device: str = "cuda",
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train a linear probe.

    Args:
        probe: The probe to train
        train_activations: Training activations
        train_labels: Training labels
        val_activations: Validation activations (optional)
        val_labels: Validation labels (optional)
        learning_rate: Learning rate
        epochs: Number of epochs
        batch_size: Batch size
        device: Device to train on
        verbose: Whether to print progress

    Returns:
        Dictionary with training history
    """
    probe = probe.to(device)
    train_activations = train_activations.to(device)
    train_labels = train_labels.to(device)

    # Create data loader
    train_dataset = TensorDataset(train_activations, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer and loss
    optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Training loop
    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_acts, batch_labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            logits = probe(batch_acts)
            loss = criterion(logits, batch_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        # Compute training metrics
        avg_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        # Validation
        if val_activations is not None and val_labels is not None:
            val_loss, val_acc = evaluate_probe(
                probe, val_activations, val_labels, device=device
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

    return history


def evaluate_probe(
    probe: LinearProbe,
    activations: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 32
) -> Tuple[float, float]:
    """
    Evaluate a linear probe.

    Args:
        probe: The probe
        activations: Test activations
        labels: Test labels
        device: Device
        batch_size: Batch size

    Returns:
        Tuple of (loss, accuracy)
    """
    probe.eval()
    activations = activations.to(device)
    labels = labels.to(device)

    dataset = TensorDataset(activations, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_acts, batch_labels in loader:
            logits = probe(batch_acts)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def train_layer_probes(
    layer_activations: Dict[int, torch.Tensor],
    labels: torch.Tensor,
    num_classes: int,
    hidden_dims: Optional[List[int]] = None,
    val_activations: Optional[Dict[int, torch.Tensor]] = None,
    val_labels: Optional[torch.Tensor] = None,
    **train_kwargs
) -> Dict[int, Tuple[LinearProbe, Dict]]:
    """
    Train probes for multiple layers.

    Args:
        layer_activations: Dictionary mapping layer indices to activations
        labels: Training labels
        num_classes: Number of classes
        hidden_dims: Hidden dimensions for probes
        val_activations: Validation activations per layer
        val_labels: Validation labels
        **train_kwargs: Arguments passed to train_probe

    Returns:
        Dictionary mapping layer indices to (probe, history) tuples
    """
    results = {}

    for layer_idx, acts in layer_activations.items():
        logger.info(f"Training probe for layer {layer_idx}")

        # Create probe
        probe = LinearProbe(
            input_dim=acts.shape[-1],
            num_classes=num_classes,
            hidden_dims=hidden_dims
        )

        # Get validation data for this layer if available
        val_acts = val_activations.get(layer_idx) if val_activations else None

        # Train probe
        history = train_probe(
            probe=probe,
            train_activations=acts,
            train_labels=labels,
            val_activations=val_acts,
            val_labels=val_labels,
            **train_kwargs
        )

        results[layer_idx] = (probe, history)

    logger.info(f"Trained probes for {len(results)} layers")

    return results


def compute_probe_accuracy_curve(
    probe_results: Dict[int, Tuple[LinearProbe, Dict]],
    test_activations: Dict[int, torch.Tensor],
    test_labels: torch.Tensor,
    device: str = "cuda"
) -> Dict[int, float]:
    """
    Compute accuracy for each layer's probe.

    Args:
        probe_results: Results from train_layer_probes
        test_activations: Test activations per layer
        test_labels: Test labels
        device: Device

    Returns:
        Dictionary mapping layer indices to test accuracy
    """
    accuracies = {}

    for layer_idx, (probe, _) in probe_results.items():
        if layer_idx not in test_activations:
            logger.warning(f"No test activations for layer {layer_idx}")
            continue

        _, accuracy = evaluate_probe(
            probe=probe,
            activations=test_activations[layer_idx],
            labels=test_labels,
            device=device
        )

        accuracies[layer_idx] = accuracy

    return accuracies


def compare_probe_accuracies(
    model1_accuracies: Dict[int, float],
    model2_accuracies: Dict[int, float],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, any]:
    """
    Compare probe accuracies between two models.

    Args:
        model1_accuracies: Accuracies for model 1
        model2_accuracies: Accuracies for model 2
        model1_name: Name of model 1
        model2_name: Name of model 2

    Returns:
        Comparison statistics
    """
    common_layers = set(model1_accuracies.keys()) & set(model2_accuracies.keys())

    if not common_layers:
        logger.warning("No common layers between models")
        return {}

    differences = {}
    for layer_idx in sorted(common_layers):
        diff = model1_accuracies[layer_idx] - model2_accuracies[layer_idx]
        differences[layer_idx] = diff

    avg_diff = np.mean(list(differences.values()))
    max_diff = max(differences.values())
    min_diff = min(differences.values())

    results = {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "common_layers": sorted(common_layers),
        "differences": differences,
        "avg_difference": avg_diff,
        "max_difference": max_diff,
        "min_difference": min_diff,
        "model1_avg": np.mean([model1_accuracies[l] for l in common_layers]),
        "model2_avg": np.mean([model2_accuracies[l] for l in common_layers]),
    }

    logger.info(f"Probe accuracy comparison: {model1_name} avg = {results['model1_avg']:.4f}, "
               f"{model2_name} avg = {results['model2_avg']:.4f}, diff = {avg_diff:.4f}")

    return results
