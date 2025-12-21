"""
Main pipeline script for reasoning direction analysis.

This script orchestrates the complete pipeline:
1. Load datasets and models
2. Extract reasoning directions via contrastive analysis
3. Perform interventions and evaluate effects
4. Conduct linear probing experiments
5. Apply logit lens analysis
6. Compute cross-model alignment metrics
7. Generate visualizations and reports
"""

import argparse
import logging
import torch
import json
from pathlib import Path
from typing import Dict, List

from .config import PipelineConfig, load_preset_config, RESULTS_DIR
from .model_utils import (
    load_model_and_tokenizer,
    collect_activations,
    compute_contrastive_directions,
    apply_direction_intervention,
    LayerAblator
)
from .utils import (
    load_dataset,
    create_control_dataset,
    prepare_prompts,
    LinearProbe,
    train_layer_probes,
    compute_probe_accuracy_curve,
    compare_probe_accuracies,
    LogitLens,
    decode_layer_states,
    compare_decoded_states,
    analyze_cross_model_alignment,
    plot_intervention_effects,
    plot_layer_alignment,
    plot_probe_accuracy,
    plot_direction_similarity,
    create_summary_figure
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_run_directory(config: PipelineConfig) -> Path:
    """Create run directory for outputs."""
    if config.run_name:
        run_dir = Path("pipeline/runs") / config.run_name
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("pipeline/runs") / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = run_dir / "config.json"
    # Note: This is simplified - you may want to implement proper config serialization
    logger.info(f"Run directory: {run_dir}")

    return run_dir


def stage1_extract_reasoning_directions(
    config: PipelineConfig,
    model_wrapper,
    run_dir: Path
) -> Dict[int, torch.Tensor]:
    """
    Stage 1: Extract reasoning directions via contrastive activation analysis.
    """
    logger.info("=" * 80)
    logger.info("Stage 1: Extracting Reasoning Directions")
    logger.info("=" * 80)

    # Load datasets
    logger.info("Loading reasoning datasets...")
    gsm8k_data = load_dataset(
        config.dataset.gsm8k_path,
        split=config.dataset.gsm8k_split,
        sample_size=config.dataset.gsm8k_sample_size,
        seed=config.dataset.seed
    )

    math_data = load_dataset(
        config.dataset.math_path,
        split=config.dataset.math_split,
        sample_size=config.dataset.math_sample_size,
        seed=config.dataset.seed
    )

    # Create control dataset
    logger.info("Creating control dataset...")
    control_data = create_control_dataset(
        size=max(len(gsm8k_data), len(math_data)),
        task_type=config.dataset.control_task_type,
        seed=config.dataset.seed
    )

    # Prepare prompts
    reasoning_prompts = prepare_prompts(gsm8k_data, dataset_type="gsm8k", include_cot_prompt=True)
    control_prompts = prepare_prompts(control_data, dataset_type="control", include_cot_prompt=False)

    # Collect activations
    logger.info("Collecting activations on reasoning tasks...")
    reasoning_activations = collect_activations(
        model=model_wrapper.model,
        tokenizer=model_wrapper.tokenizer,
        texts=reasoning_prompts,
        layer_indices=None,  # All layers
        activation_type=config.experiment.activation_type,
        device=model_wrapper.device
    )

    logger.info("Collecting activations on control tasks...")
    control_activations = collect_activations(
        model=model_wrapper.model,
        tokenizer=model_wrapper.tokenizer,
        texts=control_prompts,
        layer_indices=None,
        activation_type=config.experiment.activation_type,
        device=model_wrapper.device
    )

    # Compute contrastive directions
    logger.info("Computing contrastive reasoning directions...")
    directions = compute_contrastive_directions(
        reasoning_activations=reasoning_activations,
        control_activations=control_activations,
        normalize=config.experiment.normalize_activations
    )

    # Save directions
    directions_path = run_dir / "reasoning_directions.pt"
    torch.save(directions, directions_path)
    logger.info(f"Saved reasoning directions to {directions_path}")

    return directions


def stage2_intervention_analysis(
    config: PipelineConfig,
    model_wrapper,
    directions: Dict[int, torch.Tensor],
    run_dir: Path
):
    """
    Stage 2: Evaluate intervention effects.
    """
    logger.info("=" * 80)
    logger.info("Stage 2: Intervention Analysis")
    logger.info("=" * 80)

    # Load test prompts
    test_data = load_dataset(
        config.dataset.gsm8k_path,
        split="test",
        sample_size=20,  # Small sample for testing
        seed=config.dataset.seed
    )
    test_prompts = prepare_prompts(test_data, dataset_type="gsm8k")

    results = {}

    for strength in config.experiment.intervention_strengths:
        logger.info(f"Testing intervention strength: {strength}")

        generations = []
        for prompt in test_prompts[:5]:  # Test on subset
            generated = apply_direction_intervention(
                model=model_wrapper.model,
                tokenizer=model_wrapper.tokenizer,
                prompt=prompt,
                directions=directions,
                intervention_strength=strength,
                max_new_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature
            )
            generations.append(generated)

        results[strength] = generations

    # Save results
    results_path = run_dir / "intervention_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved intervention results to {results_path}")

    return results


def stage3_linear_probing(
    config: PipelineConfig,
    model_wrapper,
    run_dir: Path
) -> Dict:
    """
    Stage 3: Linear probing experiments.
    """
    logger.info("=" * 80)
    logger.info("Stage 3: Linear Probing")
    logger.info("=" * 80)

    # This is a simplified version - you would need to define specific probing tasks
    logger.info("Linear probing stage - implement specific tasks as needed")

    # Placeholder for demonstration
    return {}


def run_full_pipeline(config: PipelineConfig):
    """
    Run the complete reasoning direction analysis pipeline.
    """
    logger.info("Starting Reasoning Direction Analysis Pipeline")
    logger.info(f"Configuration: {config}")

    # Setup run directory
    run_dir = setup_run_directory(config)

    # Load model
    logger.info(f"Loading model: {config.model.rl_model_name}")
    model_wrapper = load_model_and_tokenizer(
        model_name=config.model.rl_model_name,
        device_map=config.model.device_map,
        torch_dtype=config.model.torch_dtype,
        load_in_8bit=config.model.load_in_8bit,
        load_in_4bit=config.model.load_in_4bit,
        use_flash_attention=config.model.use_flash_attention
    )

    # Stage 1: Extract reasoning directions
    directions = stage1_extract_reasoning_directions(config, model_wrapper, run_dir)

    # Stage 2: Intervention analysis
    intervention_results = stage2_intervention_analysis(config, model_wrapper, directions, run_dir)

    # Stage 3: Linear probing (implement as needed)
    probing_results = stage3_linear_probing(config, model_wrapper, run_dir)

    logger.info("=" * 80)
    logger.info("Pipeline Complete!")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run reasoning direction analysis pipeline")

    parser.add_argument(
        "--preset",
        type=str,
        default="quick_test",
        choices=["quick_test", "full_analysis", "layer_specific"],
        help="Configuration preset to use"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run (default: timestamp)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_preset_config(args.preset)

    # Apply overrides
    if args.run_name:
        config.run_name = args.run_name

    if args.model:
        config.model.rl_model_name = args.model

    # Run pipeline
    run_full_pipeline(config)


if __name__ == "__main__":
    main()
