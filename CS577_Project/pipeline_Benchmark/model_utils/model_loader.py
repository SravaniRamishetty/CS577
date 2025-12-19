"""
Model loading utilities for reasoning direction analysis.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Wrapper class for language models with utility methods."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_name: str,
        device: str = "cuda"
    ):
        """
        Initialize model wrapper.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            model_name: Name/identifier of the model
            device: Device to run model on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device

        # Get model architecture info
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size

        logger.info(f"Loaded {model_name}: {self.num_layers} layers, hidden size {self.hidden_size}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def get_layer_names(self) -> list:
        """Get list of layer names in the model."""
        layer_names = []
        for name, _ in self.model.named_modules():
            if "layer" in name.lower() or "block" in name.lower():
                layer_names.append(name)
        return layer_names

    def __repr__(self) -> str:
        return f"ModelWrapper({self.model_name}, layers={self.num_layers}, hidden_size={self.hidden_size})"


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    **kwargs
) -> AutoTokenizer:
    """
    Load tokenizer for a model.

    Args:
        model_name: HuggingFace model name or path
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional tokenizer arguments

    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer for {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        **kwargs
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: str = "float16",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_flash_attention: bool = True,
    trust_remote_code: bool = True,
    **kwargs
) -> AutoModelForCausalLM:
    """
    Load a causal language model.

    Args:
        model_name: HuggingFace model name or path
        device_map: Device mapping strategy
        torch_dtype: Data type for model weights
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        use_flash_attention: Whether to use Flash Attention 2
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional model arguments

    Returns:
        Loaded model
    """
    logger.info(f"Loading model {model_name}")

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    # Prepare model loading arguments
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        **kwargs
    }

    # Add quantization if requested
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    # Add Flash Attention if requested
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        logger.info(f"Successfully loaded {model_name}")
        return model

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        # Fallback: try without flash attention
        if use_flash_attention:
            logger.info("Retrying without Flash Attention...")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            logger.info(f"Successfully loaded {model_name} (without Flash Attention)")
            return model
        else:
            raise


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    **kwargs
) -> ModelWrapper:
    """
    Load both model and tokenizer and return a ModelWrapper.

    Args:
        model_name: HuggingFace model name or path
        device: Device to run model on
        **kwargs: Additional arguments passed to load_model

    Returns:
        ModelWrapper instance
    """
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, **kwargs)

    return ModelWrapper(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=device
    )


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model without loading it fully.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Dictionary with model information
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    return {
        "model_name": model_name,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": getattr(config, "intermediate_size", None),
    }
