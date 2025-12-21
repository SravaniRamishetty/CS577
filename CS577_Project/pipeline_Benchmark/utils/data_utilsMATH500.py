"""
Data loading and processing utilities for MATH-500 benchmark.
Based on: https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/01_main-chapter-code/ch03_main.ipynb

Includes improved answer extraction and grading from Raschka's implementation.
"""

import json
import random
import re
from pathlib import Path
import requests
from datasets import load_dataset as hf_load_dataset
from typing import List, Dict, Tuple, Optional, Union
import logging
from sympy import simplify
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError

logger = logging.getLogger(__name__)

# Regular expression for extracting numbers (including fractions, decimals, scientific notation)
RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

# Regex to strip chat special tokens like <|assistant|>
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")


def load_math500_data(
    local_path: Union[str, Path] = 'math500_test.json',
    url: Optional[str] = None
) -> List[Dict]:
    """
    Load MATH-500 dataset from local file or URL.

    Args:
        local_path: Path to local JSON file
        url: URL to download data from if local file doesn't exist

    Returns:
        List of math problem dictionaries
    """
    local_path = Path(local_path)

    if local_path.exists():
        with local_path.open('r', encoding='utf-8') as f:
            math_data = json.load(f)
    else:
        if url is None:
            raise ValueError("URL must be provided if local file doesn't exist")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        math_data = r.json()

        # Optionally save locally
        with local_path.open('w', encoding='utf-8') as f:
            json.dump(math_data, f, indent=2)

    return math_data


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


def render_prompt(question: str) -> str:
    """
    Build the prompt for the model following Raschka's format.
    Instructs model to output answer in \\boxed{} format.

    Args:
        question: The math problem to solve

    Returns:
        Formatted prompt string
    """
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )
    return template


def prepare_prompts(
    examples: List[Dict],
    dataset_type: str = "gsm8k",
    include_cot_prompt: bool = True
) -> List[str]:
    """
    Prepare prompts from dataset examples.

    Args:
        examples: List of dataset examples
        dataset_type: Type of dataset ("gsm8k", "math", "math500", "control")
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

        elif dataset_type == "math500":
            problem = example.get("problem", example.get("question", ""))
            prompt = render_prompt(problem)

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


def get_last_boxed(text: str) -> Optional[str]:
    """
    Extract the content inside the last \\boxed{...} in the text.
    Handles nested braces correctly.

    Args:
        text: Generated text that may contain \\boxed{answer}

    Returns:
        Content inside the last \\boxed{} or None if not found
    """
    # Find the last occurrence of "\boxed"
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    # Get position after "\boxed"
    current_idx = boxed_start_idx + len(r"\boxed")

    # Skip any whitespace after "\boxed"
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    # Expect an opening brace "{"
    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    # Parse the braces with nesting support
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    # Check for unbalanced braces
    if brace_depth != 0:
        return None

    # Extract content inside the outermost braces
    return text[content_start_idx:current_idx-1]


def extract_final_candidate(
    text: str,
    fallback: str = 'number_then_full'
) -> str:
    """
    Extract the final answer from generated text.
    Prefers \\boxed{} format, falls back to extracting last number.

    Args:
        text: Generated text from the model
        fallback: Strategy when \\boxed{} not found ("number_then_full" or "number_only")

    Returns:
        Extracted answer string
    """
    result = ""

    if text:
        # Prefer the last boxed expression if present
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")
        # If no boxed expression, try fallback
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                # Use last number found
                result = m[-1]
            elif fallback == "number_then_full":
                # Return full text if no number found
                result = text
    return result


def split_into_parts(text: str) -> List[str]:
    """
    Split tuple/list answers into individual parts for comparison.
    Example: "(1, 2, 3)" -> ["1", "2", "3"]

    Args:
        text: Answer text that may be a tuple or list

    Returns:
        List of individual answer parts
    """
    result = [text]

    if text:
        # Check if text looks like a tuple or list, e.g. "(a, b)" or "[a, b]"
        if (
            len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            # Split on commas inside brackets and strip whitespace
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        # If text is empty, return an empty list
        result = []

    return result


# LaTeX formatting patterns to normalize
LATEX_FIXES = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]


def normalize_text(text: str) -> str:
    """
    Normalize mathematical text by handling LaTeX, unicode, fractions, etc.
    This is crucial for comparing model outputs with ground truth answers.

    Args:
        text: Raw mathematical expression (may contain LaTeX)

    Returns:
        Normalized lowercase string suitable for comparison
    """
    if not text:
        return ""

    text = RE_SPECIAL.sub("", text).strip()

    # Map for converting unicode superscripts to normal characters
    SUPERSCRIPT_MAP = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
    }

    # Remove angle-degree markers
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)   # ^{\circ}
    text = re.sub(r"\^\s*\\circ", "", text)             # ^\circ
    text = text.replace("°", "")                        # Unicode degree

    # Unwrap \text{...} if the whole string is wrapped
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    # Strip inline/display math wrappers \( \) \[ \]
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # Apply LaTeX canonicalization
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    def convert_superscripts(s, base=None):
        """Convert unicode superscripts to exponent notation"""
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )
        if base is None:
            return converted
        return f"{base}**{converted}"

    # Convert unicode superscripts into exponent form (e.g., 2² -> 2**2)
    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)

    # Handle percentages and dollar signs
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")

    # Convert \sqrt{...} to sqrt(...)
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    # Convert \frac{a}{b} to (a)/(b)
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    # Convert exponents: ^ to **
    text = text.replace("^", "**")

    # Handle mixed numbers (e.g., "2 1/2" -> "2+1/2")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    # Remove thousand separators (e.g., 1,234 -> 1234)
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )

    # Remove remaining braces and convert to lowercase
    return text.replace("{", "").replace("}", "").strip().lower()


def sympy_parser(expr: str) -> Optional['sympy.Expr']:
    """
    Parse a mathematical expression using SymPy with appropriate transformations.

    Args:
        expr: Normalized mathematical expression string

    Returns:
        SymPy expression object or None if parsing fails
    """
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                # Standard transformations like handling parentheses
                *spp.standard_transformations,
                # Allow omitted multiplication symbols (e.g., "2x" -> "2*x")
                spp.implicit_multiplication_application,
            ),
            # Evaluate during parsing so simple constants simplify (e.g., 2+3 -> 5)
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, IndexError):
        return None


def equality_check(expr_gtruth: str, expr_pred: str) -> bool:
    """
    Check if two mathematical expressions are equivalent.
    Uses both string comparison and symbolic SymPy comparison.

    Args:
        expr_gtruth: Ground truth expression (normalized)
        expr_pred: Predicted expression (normalized)

    Returns:
        True if expressions are mathematically equivalent
    """
    # First, check exact string match
    if expr_gtruth == expr_pred:
        return True

    # Parse both expressions into SymPy objects
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)

    # If both parsed successfully, try symbolic comparison
    if gtruth is not None and pred is not None:
        try:
            # If the difference simplifies to 0, they are equivalent
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass

    return False


def grade_answer(pred_text: str, gt_text: str) -> bool:
    """
    Grade a predicted answer against ground truth.
    Handles normalization, tuple/list splitting, and symbolic comparison.

    Args:
        pred_text: Model's predicted answer
        gt_text: Ground truth answer

    Returns:
        True if answer is correct, False otherwise
    """
    result = False  # Default outcome if checks fail

    # Only continue if both inputs are non-empty strings
    if pred_text is not None and gt_text is not None:
        # Normalize and split both answers into comparable parts
        gt_parts = split_into_parts(normalize_text(gt_text))
        pred_parts = split_into_parts(normalize_text(pred_text))

        # Ensure both sides have same number of valid parts
        if (gt_parts and pred_parts and len(gt_parts) == len(pred_parts)):
            # Check each part for mathematical equivalence
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )

    return result


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

    elif dataset_type in ("math", "math500"):
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


def format_problem(problem: Dict) -> Dict:
    """
    Format a MATH-500 problem for model input.

    Args:
        problem: Problem dictionary with 'problem' and 'solution' keys

    Returns:
        Formatted problem dictionary with 'prompt' key
    """
    return {
        'prompt': render_prompt(problem['problem']),
        'problem': problem['problem'],
        'solution': problem.get('solution', ''),
        'answer': problem.get('answer', '')
    }


def batch_problems(
    problems: List[Dict],
    batch_size: int
) -> List[List[Dict]]:
    """
    Split problems into batches.

    Args:
        problems: List of problem dictionaries
        batch_size: Number of problems per batch

    Returns:
        List of batches
    """
    return [
        problems[i:i + batch_size]
        for i in range(0, len(problems), batch_size)
    ]


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
