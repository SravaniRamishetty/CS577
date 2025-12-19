"""
Model loader for QwQ-32B and DeepSeek-R1-Distill-Qwen-32B
Loads models using TransformerLens (HookedTransformer) for activation patching
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple


class ModelLoader:
    """Loads and manages RL-trained and distilled models"""

    def __init__(
        self,
        rl_model_name: str = "Qwen/QwQ-32B-Preview",
        distilled_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize model loader

        Args:
            rl_model_name: HuggingFace model ID for RL-trained model
            distilled_model_name: HuggingFace model ID for distilled model
            device: Device to load models on
        """
        self.rl_model_name = rl_model_name
        self.distilled_model_name = distilled_model_name
        self.device = device

        self.rl_model = None
        self.distilled_model = None
        self.rl_tokenizer = None
        self.distilled_tokenizer = None

    def load_models(
        self,
        load_rl: bool = True,
        load_distilled: bool = True,
        torch_dtype: torch.dtype = torch.float16
    ) -> Dict[str, any]:
        """
        Load specified models

        Args:
            load_rl: Whether to load RL-trained model
            load_distilled: Whether to load distilled model
            torch_dtype: Data type for model weights

        Returns:
            Dictionary containing loaded models and tokenizers
        """
        models = {}

        if load_rl:
            print(f"Loading RL-trained model: {self.rl_model_name}")
            self.rl_tokenizer = AutoTokenizer.from_pretrained(self.rl_model_name)
            self.rl_model = AutoModelForCausalLM.from_pretrained(
                self.rl_model_name,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
            models['rl_model'] = self.rl_model
            models['rl_tokenizer'] = self.rl_tokenizer

        if load_distilled:
            print(f"Loading distilled model: {self.distilled_model_name}")
            self.distilled_tokenizer = AutoTokenizer.from_pretrained(self.distilled_model_name)
            self.distilled_model = AutoModelForCausalLM.from_pretrained(
                self.distilled_model_name,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
            models['distilled_model'] = self.distilled_model
            models['distilled_tokenizer'] = self.distilled_tokenizer

        return models

    def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded models"""
        info = {}

        if self.rl_model is not None:
            info['rl_model'] = {
                'name': self.rl_model_name,
                'num_layers': self.rl_model.config.num_hidden_layers,
                'hidden_size': self.rl_model.config.hidden_size,
                'vocab_size': self.rl_model.config.vocab_size
            }

        if self.distilled_model is not None:
            info['distilled_model'] = {
                'name': self.distilled_model_name,
                'num_layers': self.distilled_model.config.num_hidden_layers,
                'hidden_size': self.distilled_model.config.hidden_size,
                'vocab_size': self.distilled_model.config.vocab_size
            }

        return info
