"""
Data loading and processing utilities.
"""

import random
from datasets import load_dataset as hf_load_dataset
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str,
    split: str = "test",
    sample_size: Optional[int] = None,
    seed: int = 42
) -> List[Dict]:
    """
    Load a dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset
        split: Dataset split to load
        sample_size: Number of samples to use (None = all)
        seed: Random seed for sampling

    Returns:
        List of dataset examples
    """
    logger.info(f"Loading dataset {dataset_name} (split: {split})")

    # Load from HuggingFace
    if dataset_name == "openai/gsm8k":
        dataset = hf_load_dataset(dataset_name, "main", split=split)
    elif dataset_name == "hendrycks/competition_math":
        dataset = hf_load_dataset("hendrycks_math", "all", split=split)
    else:
        dataset = hf_load_dataset(dataset_name, split=split)

    # Convert to list of dicts
    data = list(dataset)

    # Sample if requested
    if sample_size is not None and sample_size < len(data):
        random.seed(seed)
        data = random.sample(data, sample_size)
        logger.info(f"Sampled {sample_size} examples from {len(dataset)}")

    logger.info(f"Loaded {len(data)} examples")

    return data


def create_control_dataset(
    size: int,
    task_type: str = "simple_qa",
    seed: int = 42
) -> List[Dict]:
    """
    Create a control dataset for contrastive analysis.

    Args:
        size: Number of examples to create
        task_type: Type of control task
        seed: Random seed

    Returns:
        List of control examples
    """
    random.seed(seed)

    control_data = []

    if task_type == "simple_qa":
        # Simple factual questions that don't require reasoning
        templates = [
            "What is the capital of {country}?",
            "Who wrote {book}?",
            "What color is {object}?",
            "In which year did {event} happen?",
            "What is the largest {category}?",
        ]

        # Simple answer templates
        countries = ["France", "Germany", "Japan", "Brazil", "Australia"]
        books = ["1984", "Pride and Prejudice", "The Great Gatsby"]
        objects = ["the sky", "grass", "an apple", "a banana"]
        events = ["World War II end", "the moon landing", "the fall of the Berlin Wall"]
        categories = ["ocean", "continent", "planet in our solar system"]

        for i in range(size):
            template = random.choice(templates)

            if "{country}" in template:
                question = template.format(country=random.choice(countries))
            elif "{book}" in template:
                question = template.format(book=random.choice(books))
            elif "{object}" in template:
                question = template.format(object=random.choice(objects))
            elif "{event}" in template:
                question = template.format(event=random.choice(events))
            elif "{category}" in template:
                question = template.format(category=random.choice(categories))
            else:
                question = template

            control_data.append({
                "question": question,
                "type": "simple_qa"
            })

    logger.info(f"Created {len(control_data)} control examples")

    return control_data


def prepare_prompts(
    examples: List[Dict],
    dataset_type: str = "gsm8k",
    include_cot_prompt: bool = True
) -> List[str]:
    """
    Prepare prompts from dataset examples.

    Args:
        examples: List of dataset examples
        dataset_type: Type of dataset ("gsm8k", "math", "control")
        include_cot_prompt: Whether to include chain-of-thought prompting

    Returns:
        List of formatted prompts
    """
    prompts = []

    cot_prefix = "Let's solve this step by step.\n\n" if include_cot_prompt else ""

    for example in examples:
        if dataset_type == "gsm8k":
            question = example["question"]
            prompt = f"{cot_prefix}Question: {question}\nAnswer:"

        elif dataset_type == "math":
            problem = example["problem"]
            prompt = f"{cot_prefix}Problem: {problem}\nSolution:"

        elif dataset_type == "control":
            question = example["question"]
            prompt = f"Question: {question}\nAnswer:"

        else:
            # Generic format
            if "question" in example:
                prompt = f"{cot_prefix}Question: {example['question']}\nAnswer:"
            elif "problem" in example:
                prompt = f"{cot_prefix}Problem: {example['problem']}\nSolution:"
            else:
                prompt = str(example)

        prompts.append(prompt)

    return prompts


def extract_reasoning_steps(
    generated_text: str,
    dataset_type: str = "gsm8k"
) -> List[str]:
    """
    Extract reasoning steps from generated text.

    Args:
        generated_text: Generated text containing reasoning
        dataset_type: Type of dataset

    Returns:
        List of reasoning steps
    """
    steps = []

    # Split by common delimiters
    lines = generated_text.split("\n")

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Common patterns for reasoning steps
        if any(marker in line.lower() for marker in ["step", "first", "second", "then", "next", "finally"]):
            steps.append(line)
        elif line and len(line) > 10:  # Heuristic: substantial lines are likely reasoning
            steps.append(line)

    return steps


def parse_answer(
    generated_text: str,
    dataset_type: str = "gsm8k"
) -> Optional[str]:
    """
    Parse the final answer from generated text.

    Args:
        generated_text: Generated text
        dataset_type: Type of dataset

    Returns:
        Extracted answer or None
    """
    if dataset_type == "gsm8k":
        # Look for #### pattern (GSM8K format)
        if "####" in generated_text:
            answer = generated_text.split("####")[-1].strip()
            return answer

        # Look for "The answer is" pattern
        if "answer is" in generated_text.lower():
            parts = generated_text.lower().split("answer is")
            if len(parts) > 1:
                answer = parts[-1].strip().split()[0]
                return answer

    elif dataset_type == "math":
        # Look for boxed answer
        if "\\boxed{" in generated_text:
            start = generated_text.rfind("\\boxed{") + 7
            # Find matching brace
            count = 1
            i = start
            while i < len(generated_text) and count > 0:
                if generated_text[i] == '{':
                    count += 1
                elif generated_text[i] == '}':
                    count -= 1
                i += 1
            answer = generated_text[start:i-1]
            return answer

    # Default: return last line
    lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return None


def create_dataset_splits(
    data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/val/test.

    Args:
        data: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train, val, test) splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    random.seed(seed)
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)

    n = len(data_shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data_shuffled[:train_end]
    val_data = data_shuffled[train_end:val_end]
    test_data = data_shuffled[val_end:]

    logger.info(f"Split data into train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    return train_data, val_data, test_data
