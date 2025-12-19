"""
Main pipeline runner
Orchestrates the complete reasoning direction analysis workflow
"""

import argparse
import torch
import json
from pathlib import Path

from model_loader import ModelLoader
from data_processor import DataProcessor
from direction_calculator import DirectionCalculator
from intervention import ActivationPatcher
from evaluator import ReasoningEvaluator


def run_pipeline(
    rl_model_name: str = "Qwen/QwQ-32B-Preview",
    distilled_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    dataset_name: str = "HuggingFaceH4/MATH-500",
    num_samples: int = 10,
    layers_to_test: list = None,
    strength_range: tuple = (-0.1, 0.1),
    num_strengths: int = 5,
    output_dir: str = "./results",
    save_directions: bool = True
):
    """
    Run complete reasoning direction analysis pipeline

    Args:
        rl_model_name: RL-trained model name
        distilled_model_name: Distilled model name
        dataset_name: Dataset to use
        num_samples: Number of samples to process
        layers_to_test: List of layer indices (None = test all)
        strength_range: (min, max) intervention strength
        num_strengths: Number of strength values to test
        output_dir: Directory to save results
        save_directions: Whether to save computed directions
    """
    print("="*60)
    print("REASONING DIRECTION ANALYSIS PIPELINE")
    print("="*60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load models
    print("\n[Step 1/6] Loading models...")
    loader = ModelLoader(rl_model_name, distilled_model_name)
    models = loader.load_models(torch_dtype=torch.float16)

    rl_model = models['rl_model']
    rl_tokenizer = models['rl_tokenizer']
    distilled_model = models['distilled_model']
    distilled_tokenizer = models['distilled_tokenizer']

    model_info = loader.get_model_info()
    print(f"RL Model: {model_info['rl_model']['num_layers']} layers")
    print(f"Distilled Model: {model_info['distilled_model']['num_layers']} layers")

    # Step 2: Load and prepare data
    print("\n[Step 2/6] Loading dataset...")
    processor = DataProcessor(dataset_name, include_toy_tasks=True)
    dataset = processor.load_dataset(max_samples=num_samples)
    toy_tasks = processor.get_toy_tasks()

    # Combine dataset and toy tasks
    all_examples = dataset + toy_tasks
    print(f"Loaded {len(all_examples)} examples")

    # Step 3: Calculate reasoning directions
    print("\n[Step 3/6] Calculating reasoning directions...")
    calculator = DirectionCalculator()

    # Prepare prompts
    prompts = processor.prepare_batch(all_examples[:num_samples], rl_tokenizer)

    # Determine layers to analyze
    num_layers = model_info['rl_model']['num_layers']
    if layers_to_test is None:
        # Sample every 5th layer
        layers_to_test = list(range(0, num_layers, 5))

    print(f"Capturing activations for {len(layers_to_test)} layers...")

    # Capture activations
    rl_activations = calculator.capture_activations(
        rl_model, rl_tokenizer, prompts[:5], layers_to_test
    )
    distilled_activations = calculator.capture_activations(
        distilled_model, distilled_tokenizer, prompts[:5], layers_to_test
    )

    # Calculate directions
    directions = calculator.calculate_direction(rl_activations, distilled_activations)
    print(f"Computed directions for {len(directions)} layers")

    # Save directions
    if save_directions:
        directions_path = output_path / "reasoning_directions.pt"
        calculator.save_directions(str(directions_path))

    # Step 4: Apply interventions
    print("\n[Step 4/6] Running interventions...")
    patcher = ActivationPatcher(rl_model, directions)

    # Test on a sample prompt
    test_prompt = prompts[0]

    # Baseline generation
    print("Generating baseline...")
    inputs = rl_tokenizer(test_prompt, return_tensors="pt").to(rl_model.device)
    with torch.no_grad():
        baseline_output = rl_model.generate(**inputs, max_new_tokens=256)
    baseline_text = rl_tokenizer.decode(baseline_output[0], skip_special_tokens=True)

    # Sweep interventions
    print("Sweeping layers and strengths...")
    intervention_results = patcher.sweep_layers_and_strengths(
        tokenizer=rl_tokenizer,
        prompt=test_prompt,
        layer_range=(min(layers_to_test), max(layers_to_test)),
        strength_range=strength_range,
        num_strengths=num_strengths,
        max_new_tokens=256
    )

    print(f"Completed {len(intervention_results)} intervention experiments")

    # Step 5: Evaluate results
    print("\n[Step 5/6] Evaluating results...")
    evaluator = ReasoningEvaluator()

    # Analyze layer sensitivity
    sensitivity = evaluator.analyze_layer_sensitivity(intervention_results)
    critical_layers = evaluator.identify_critical_layers(sensitivity)

    print(f"Critical layers identified: {critical_layers}")

    # Step 6: Generate report
    print("\n[Step 6/6] Generating report...")
    report = evaluator.generate_report(
        intervention_results,
        output_file=str(output_path / "evaluation_report.txt")
    )

    # Save all results
    results_data = {
        'model_info': model_info,
        'directions_stats': calculator.compute_direction_stats(),
        'critical_layers': critical_layers,
        'layer_sensitivity': sensitivity,
        'intervention_results': intervention_results,
        'baseline_output': baseline_text
    }

    with open(output_path / "results.json", 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print("\n" + "="*60)
    print("Pipeline completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reasoning direction analysis pipeline")
    parser.add_argument("--rl-model", type=str, default="Qwen/QwQ-32B-Preview")
    parser.add_argument("--distilled-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./results")

    args = parser.parse_args()

    run_pipeline(
        rl_model_name=args.rl_model,
        distilled_model_name=args.distilled_model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
