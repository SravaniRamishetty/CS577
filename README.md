# CS577

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
