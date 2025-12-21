"""
Data processing module for MATH-500 benchmark and toy tasks
Handles prompt formatting with chat templates
"""

from datasets import load_dataset
from typing import List, Dict, Optional
import json


class DataProcessor:
    """Process and format prompts for reasoning tasks"""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceH4/MATH-500",
        include_toy_tasks: bool = True
    ):
        """
        Initialize data processor

        Args:
            dataset_name: HuggingFace dataset name
            include_toy_tasks: Whether to include simple toy problems
        """
        self.dataset_name = dataset_name
        self.include_toy_tasks = include_toy_tasks
        self.dataset = None
        self.toy_tasks = []

        if include_toy_tasks:
            self.toy_tasks = [
                {"problem": "What is 2+2?", "answer": "4"},
                {"problem": "What is 10-5?", "answer": "5"},
                {"problem": "Is 7 a prime number?", "answer": "Yes"},
                {"problem": "What is the square root of 16?", "answer": "4"}
            ]

    def load_dataset(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load MATH-500 benchmark dataset

        Args:
            split: Dataset split to load
            max_samples: Maximum number of samples to load

        Returns:
            List of dataset examples
        """
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name, split=split)

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        return list(self.dataset)

    def format_prompt(
        self,
        problem: str,
        tokenizer,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format prompt using model's chat template

        Args:
            problem: The math problem or reasoning task
            tokenizer: Model tokenizer with chat template
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": problem})

        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple formatting
            formatted = f"User: {problem}\nAssistant:"

        return formatted

    def get_toy_tasks(self) -> List[Dict]:
        """Get simple toy tasks for testing"""
        return self.toy_tasks

    def prepare_batch(
        self,
        examples: List[Dict],
        tokenizer,
        problem_key: str = "problem",
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Prepare a batch of prompts

        Args:
            examples: List of problem dictionaries
            tokenizer: Model tokenizer
            problem_key: Key for problem text in example dict
            system_prompt: Optional system prompt

        Returns:
            List of formatted prompts
        """
        prompts = []
        for example in examples:
            problem = example.get(problem_key, example.get('question', str(example)))
            formatted = self.format_prompt(problem, tokenizer, system_prompt)
            prompts.append(formatted)

        return prompts
