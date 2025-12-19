"""
Configuration file for reasoning direction analysis pipeline.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = PROJECT_ROOT / "pipeline" / "runs"

# Model configurations
@dataclass
class ModelConfig:
    """Configuration for models to be analyzed."""

    # RL-trained model
    rl_model_name: str = "Qwen/QwQ-32B-Preview"
    rl_model_short_name: str = "qwq-32b"

    # Distilled model
    distilled_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    distilled_model_short_name: str = "deepseek-r1-distill-qwen-32b"

    # Model loading parameters
    device_map: str = "auto"
    torch_dtype: str = "float16"  # or "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class DatasetConfig:
    """Configuration for datasets."""

    # GSM8K
    gsm8k_path: str = "openai/gsm8k"
    gsm8k_split: str = "test"
    gsm8k_sample_size: int = 100  # Number of samples to use

    # MATH
    math_path: str = "hendrycks/competition_math"
    math_split: str = "test"
    math_sample_size: int = 100

    # Control tasks (for contrastive analysis)
    control_task_type: str = "simple_qa"  # Type of control task

    # Random seed for reproducibility
    seed: int = 42

    # Data processing
    max_length: int = 512
    include_solution: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for reasoning direction experiments."""

    # Contrastive activation analysis
    num_layers: Optional[int] = None  # Will be set based on model
    use_all_layers: bool = True
    layer_range: Optional[List[int]] = None  # e.g., [10, 20] for specific range

    # Activation collection
    activation_type: str = "residual"  # "residual", "mlp", "attention"
    normalize_activations: bool = True

    # Direction extraction
    aggregation_method: str = "mean"  # "mean", "pca", "ica"
    direction_selection_metric: str = "intervention_success"  # or "cosine_similarity"

    # Intervention parameters
    intervention_strengths: List[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])
    intervention_layers: Optional[List[int]] = None  # Layers to intervene on

    # Linear probing
    probe_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    probe_learning_rate: float = 0.001
    probe_epochs: int = 50
    probe_batch_size: int = 32

    # Logit lens
    use_logit_lens: bool = True
    logit_lens_layers: Optional[List[int]] = None  # Specific layers for logit lens

    # Cross-model alignment
    use_cka: bool = True
    cka_kernel: str = "linear"  # "linear" or "rbf"
    compute_cosine_similarity: bool = True

    # Ablation experiments
    ablation_methods: List[str] = field(default_factory=lambda: ["zero", "noise", "mean"])
    noise_std: float = 0.1

    # Path patching
    use_path_patching: bool = True
    patch_positions: str = "all"  # "all", "early", "middle", "late"


@dataclass
class ComputeConfig:
    """Configuration for compute resources."""

    # GPU settings
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_gpus: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # Memory optimization
    use_gradient_checkpointing: bool = False
    use_cpu_offload: bool = False
    max_memory_per_gpu: Optional[str] = "80GB"  # For A100

    # Parallel processing
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "reasoning-direction-analysis"
    wandb_entity: Optional[str] = None

    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = str(RESULTS_DIR / "tensorboard")

    # Logging
    log_level: str = "INFO"
    log_interval: int = 10

    # Checkpointing
    save_intermediate_results: bool = True
    checkpoint_interval: int = 100


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Output directories
    output_dir: str = str(RESULTS_DIR)
    run_name: Optional[str] = None

    def __post_init__(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR / "activations", exist_ok=True)
        os.makedirs(RESULTS_DIR / "directions", exist_ok=True)
        os.makedirs(RESULTS_DIR / "interventions", exist_ok=True)
        os.makedirs(RESULTS_DIR / "probes", exist_ok=True)
        os.makedirs(RESULTS_DIR / "visualizations", exist_ok=True)

        if self.run_name:
            run_dir = RUNS_DIR / self.run_name
            os.makedirs(run_dir, exist_ok=True)


# Default configuration
def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()


# Configuration presets for different experiments
EXPERIMENT_PRESETS = {
    "quick_test": {
        "dataset": {
            "gsm8k_sample_size": 10,
            "math_sample_size": 10,
        },
        "experiment": {
            "intervention_strengths": [-1.0, 0.0, 1.0],
            "probe_epochs": 10,
        }
    },
    "full_analysis": {
        "dataset": {
            "gsm8k_sample_size": 500,
            "math_sample_size": 500,
        },
        "experiment": {
            "intervention_strengths": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "probe_epochs": 100,
        }
    },
    "layer_specific": {
        "experiment": {
            "use_all_layers": False,
            "layer_range": [15, 25],  # Middle layers for 32-layer model
        }
    }
}


def load_preset_config(preset_name: str) -> PipelineConfig:
    """Load a preset configuration."""
    if preset_name not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(EXPERIMENT_PRESETS.keys())}")

    config = get_default_config()
    preset = EXPERIMENT_PRESETS[preset_name]

    # Update configuration with preset values
    for section, values in preset.items():
        if hasattr(config, section):
            section_config = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)

    return config
