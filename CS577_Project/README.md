# Reasoning Direction Analysis in RL-Trained vs Distilled Language Models

This repository contains code for analyzing how reasoning emerges in RL-trained versus distilled language models using mechanistic interpretability techniques.

## Project Overview

We investigate reasoning circuits in transformer models by comparing:
- **RL-Trained Model**: QwQ-32B
- **Distilled Model**: DeepSeek-R1-Distill-Qwen-32B

Our analysis uses three core methodologies:
1. **Reasoning Direction Localization**: Identify layer-specific reasoning directions via contrastive analysis
2. **Layer-Wise Reasoning Decoding**: Probe intermediate reasoning states using linear probes and logit lens
3. **Causal Intervention**: Establish causality through ablation and path patching

## Repository Structure

```
CS577_Project/
├── pipeline/
│   ├── config.py                  # Configuration management
│   ├── run_pipeline.py           # Main pipeline script
│   ├── model_utils/              # Model loading and intervention utilities
│   │   ├── model_loader.py       # Model/tokenizer loading
│   │   ├── activation_collection.py  # Activation extraction
│   │   └── intervention.py       # Direction interventions and ablation
│   ├── utils/                    # Analysis utilities
│   │   ├── data_utils.py        # Dataset loading and processing
│   │   ├── probe.py             # Linear probing
│   │   ├── logit_lens.py        # Logit lens analysis
│   │   ├── alignment.py         # CKA and similarity metrics
│   │   └── visualization.py     # Plotting utilities
│   └── runs/                    # Experiment outputs
├── data/                        # Datasets (GSM8K, MATH)
├── results/                     # Analysis results
│   ├── activations/
│   ├── directions/
│   ├── interventions/
│   ├── probes/
│   └── visualizations/
├── notebooks/                   # Jupyter notebooks for analysis
├── requirements.txt            # Python dependencies
├── setup.sh                   # Setup script
└── README.md                 # This file
```

## Setup

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended: A100 with 80GB VRAM)
- Hugging Face account (for model access)

### Installation

1. **Clone the repository**:
```bash
cd CS577_Project
```

2. **Run the setup script**:
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Check Python version
- Create a virtual environment
- Install all dependencies
- Set up Hugging Face authentication
- Create necessary directories

3. **Manual setup (alternative)**:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with Hugging Face
huggingface-cli login
```

## Usage

### Quick Start

Run a quick test with a small sample:

```bash
python -m pipeline.run_pipeline --preset quick_test
```

### Full Analysis

Run the complete analysis pipeline:

```bash
python -m pipeline.run_pipeline --preset full_analysis --run_name my_experiment
```

### Command-line Options

```bash
python -m pipeline.run_pipeline --help
```

Available options:
- `--preset`: Configuration preset (`quick_test`, `full_analysis`, `layer_specific`)
- `--run_name`: Name for the experiment run
- `--model`: Override the default model

### Configuration

Edit `pipeline/config.py` to customize:
- Model parameters
- Dataset sizes
- Intervention strengths
- Probe architecture
- Compute resources

Example configuration presets:

```python
# Quick test (10 samples)
python -m pipeline.run_pipeline --preset quick_test

# Full analysis (500 samples)
python -m pipeline.run_pipeline --preset full_analysis

# Layer-specific analysis (layers 15-25)
python -m pipeline.run_pipeline --preset layer_specific
```

## Datasets

The project uses two main reasoning benchmarks:

1. **GSM8K**: 8,500 grade-school math word problems
   - Multi-step arithmetic reasoning
   - Natural language questions with step-by-step solutions

2. **MATH**: 12,500 competition-level math problems
   - Algebra, geometry, number theory, probability
   - LaTeX-formatted reasoning traces

Datasets are automatically downloaded from Hugging Face on first run.

## Experiments

### 1. Reasoning Direction Extraction

Extract reasoning-specific directions via contrastive activation analysis:

```python
from pipeline.model_utils import collect_activations, compute_contrastive_directions

# Collect activations on reasoning vs control tasks
reasoning_acts = collect_activations(model, tokenizer, reasoning_prompts)
control_acts = collect_activations(model, tokenizer, control_prompts)

# Compute contrastive directions
directions = compute_contrastive_directions(reasoning_acts, control_acts)
```

### 2. Intervention Analysis

Test the effect of adding/subtracting reasoning directions:

```python
from pipeline.model_utils import apply_direction_intervention

# Apply intervention during generation
output = apply_direction_intervention(
    model, tokenizer, prompt,
    directions=directions,
    intervention_strength=1.0  # Positive = enhance, negative = suppress
)
```

### 3. Linear Probing

Train probes to detect intermediate reasoning states:

```python
from pipeline.utils import LinearProbe, train_layer_probes

# Train probes for each layer
probe_results = train_layer_probes(
    layer_activations=activations,
    labels=labels,
    num_classes=num_classes
)
```

### 4. Logit Lens Analysis

Decode intermediate layer representations:

```python
from pipeline.utils import LogitLens

lens = LogitLens(model, tokenizer)
decoded = lens.decode_layer_activations(layer_activations)
```

### 5. Cross-Model Alignment

Compare representations between RL and distilled models:

```python
from pipeline.utils import analyze_cross_model_alignment

alignment = analyze_cross_model_alignment(
    model1_activations,
    model2_activations,
    model1_directions,
    model2_directions
)
```

## Results

Results are saved to `pipeline/runs/<run_name>/`:

- `reasoning_directions.pt`: Extracted direction vectors
- `intervention_results.json`: Intervention generation samples
- `probe_accuracies.json`: Linear probe performance
- `alignment_results.json`: Cross-model alignment metrics
- `visualizations/`: Generated plots

### Visualization

Generate summary visualizations:

```python
from pipeline.utils import create_summary_figure

create_summary_figure(
    probe_results=probe_accuracies,
    direction_similarities=dir_similarities,
    cka_diagonal=cka_scores,
    save_path="results/summary.png"
)
```

## Advanced Usage

### Custom Interventions

Implement custom intervention strategies:

```python
from pipeline.model_utils import InterventionHandler

with InterventionHandler(model, directions, intervention_strength=1.5) as handler:
    # Interventions active during this context
    output = model.generate(**inputs)
```

### Layer Ablation

Ablate specific layers to test their necessity:

```python
from pipeline.model_utils import LayerAblator

with LayerAblator(model, layer_indices=[15, 16, 17], ablation_method="zero"):
    output = model.generate(**inputs)
```

### Path Patching

Swap activations between reasoning and control runs:

```python
from pipeline.model_utils import PathPatcher

with PathPatcher(model, source_activations, layer_indices=[10, 15, 20]):
    output = model.generate(**inputs)
```

## Development

### Adding New Experiments

1. Create a new stage function in `pipeline/run_pipeline.py`
2. Add configuration options in `pipeline/config.py`
3. Implement analysis in appropriate utility module

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black pipeline/

# Type checking
mypy pipeline/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{reasoning-direction-analysis,
  title={Reasoning Direction Analysis in RL-Trained vs Distilled Language Models},
  author={Your Name},
  year={2025},
  howpublished={\\url{https://github.com/yourusername/CS577_Project}}
}
```

## References

- QwQ-32B: [Qwen Team](https://huggingface.co/Qwen/QwQ-32B-Preview)
- DeepSeek-R1-Distill: [DeepSeek AI](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- GSM8K: [Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)
- MATH: [Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874)
- Mechanistic Interpretability: [Olah et al., 2020](https://distill.pub/2020/circuits/)

## License

This project is licensed under the Apache License 2.0 - see LICENSE file for details.

## Acknowledgments

- Inspired by the [refusal_direction](https://github.com/andyrdt/refusal_direction) repository
- Built on HuggingFace Transformers and PyTorch
- Uses mechanistic interpretability tools: nnsight, TransformerLens, baukit

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

## Roadmap

- [ ] Implement cross-model direction transplantation
- [ ] Add support for more model architectures
- [ ] Expand to more reasoning benchmarks
- [ ] Create interactive visualization dashboard
- [ ] Add comprehensive test suite
- [ ] Optimize for multi-GPU training
