"""
Evaluation and analysis module
Measures intervention effects on reasoning behavior
"""

import torch
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class ReasoningEvaluator:
    """Evaluate and analyze reasoning behavior changes"""

    def __init__(self):
        """Initialize evaluator"""
        self.results = []

    def extract_think_tags(self, text: str) -> Tuple[str, str]:
        """
        Extract content inside and outside <think> tags

        FIXED: Now handles incomplete tags (missing </think> due to truncation)

        Args:
            text: Generated text

        Returns:
            Tuple of (think_content, non_think_content)
        """
        # Find all complete <think>...</think> blocks
        complete_pattern = r'<think>(.*?)</think>'
        complete_matches = re.findall(complete_pattern, text, re.DOTALL)

        # Find incomplete <think>... blocks (missing closing tag)
        incomplete_pattern = r'<think>(.*?)$'
        incomplete_match = re.search(incomplete_pattern, text, re.DOTALL)

        # Combine complete and incomplete matches
        all_think_content = []
        if complete_matches:
            all_think_content.extend(complete_matches)
        if incomplete_match and not complete_matches:
            # Only use incomplete match if no complete matches found
            # This handles truncated generations
            all_think_content.append(incomplete_match.group(1))

        # Extract think content
        think_content = ' '.join(all_think_content)

        # Remove both complete and incomplete think tags to get non-think content
        non_think_content = re.sub(complete_pattern, '', text, flags=re.DOTALL)
        if incomplete_match and not complete_matches:
            non_think_content = re.sub(incomplete_pattern, '', non_think_content, flags=re.DOTALL)

        return think_content, non_think_content

    def count_tokens(
        self,
        text: str,
        tokenizer,
        split_by_tags: bool = True
    ) -> Dict[str, int]:
        """
        Count tokens in generated text

        Args:
            text: Generated text
            tokenizer: Model tokenizer
            split_by_tags: Whether to separately count think vs non-think tokens

        Returns:
            Dictionary with token counts
        """
        counts = {}

        if split_by_tags:
            think_content, non_think_content = self.extract_think_tags(text)
            counts['think_tokens'] = len(tokenizer.encode(think_content))
            counts['non_think_tokens'] = len(tokenizer.encode(non_think_content))
            counts['total_tokens'] = len(tokenizer.encode(text))
        else:
            counts['total_tokens'] = len(tokenizer.encode(text))

        return counts

    def analyze_reasoning_quality(self, text: str) -> Dict[str, any]:
        """
        Analyze qualitative aspects of reasoning

        Args:
            text: Generated text

        Returns:
            Dictionary with reasoning quality metrics
        """
        think_content, non_think_content = self.extract_think_tags(text)

        # Count reasoning indicators
        backtracking_keywords = ['wait', 'actually', 'no', 'correction', 'mistake', 'reconsider']
        hesitation_keywords = ['maybe', 'perhaps', 'possibly', 'might', 'could be']

        backtracking_count = sum(1 for kw in backtracking_keywords if kw.lower() in think_content.lower())
        hesitation_count = sum(1 for kw in hesitation_keywords if kw.lower() in think_content.lower())

        # Check for presence of think tags
        has_think_tags = bool(think_content.strip())

        # Count steps (lines in think content)
        reasoning_steps = len([line for line in think_content.split('\n') if line.strip()])

        return {
            'has_think_tags': has_think_tags,
            'think_length': len(think_content),
            'reasoning_steps': reasoning_steps,
            'backtracking_count': backtracking_count,
            'hesitation_count': hesitation_count,
            'verbosity': len(think_content.split())
        }

    def compare_outputs(
        self,
        baseline_output: str,
        intervened_output: str,
        tokenizer
    ) -> Dict[str, any]:
        """
        Compare baseline and intervened outputs

        Args:
            baseline_output: Output without intervention
            intervened_output: Output with intervention
            tokenizer: Model tokenizer

        Returns:
            Comparison metrics
        """
        baseline_tokens = self.count_tokens(baseline_output, tokenizer)
        intervened_tokens = self.count_tokens(intervened_output, tokenizer)

        baseline_quality = self.analyze_reasoning_quality(baseline_output)
        intervened_quality = self.analyze_reasoning_quality(intervened_output)

        return {
            'token_delta': {
                'total': intervened_tokens['total_tokens'] - baseline_tokens['total_tokens'],
                'think': intervened_tokens.get('think_tokens', 0) - baseline_tokens.get('think_tokens', 0),
                'non_think': intervened_tokens.get('non_think_tokens', 0) - baseline_tokens.get('non_think_tokens', 0)
            },
            'quality_delta': {
                'reasoning_steps': intervened_quality['reasoning_steps'] - baseline_quality['reasoning_steps'],
                'backtracking': intervened_quality['backtracking_count'] - baseline_quality['backtracking_count'],
                'hesitation': intervened_quality['hesitation_count'] - baseline_quality['hesitation_count'],
                'verbosity': intervened_quality['verbosity'] - baseline_quality['verbosity']
            },
            'baseline': {
                'tokens': baseline_tokens,
                'quality': baseline_quality
            },
            'intervened': {
                'tokens': intervened_tokens,
                'quality': intervened_quality
            }
        }

    def analyze_layer_sensitivity(
        self,
        results: List[Dict],
        metric: str = 'token_count'
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze which layers are most sensitive to interventions

        Args:
            results: List of intervention results
            metric: Metric to analyze

        Returns:
            Layer sensitivity statistics
        """
        layer_stats = defaultdict(list)

        for result in results:
            layer = result.get('layer')
            value = result.get(metric, 0)
            layer_stats[layer].append(value)

        sensitivity = {}
        for layer, values in layer_stats.items():
            sensitivity[layer] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }

        return sensitivity

    def identify_critical_layers(
        self,
        layer_sensitivity: Dict[int, Dict[str, float]],
        threshold_percentile: float = 75
    ) -> List[int]:
        """
        Identify critical layers based on sensitivity

        Args:
            layer_sensitivity: Layer sensitivity statistics
            threshold_percentile: Percentile threshold for criticality

        Returns:
            List of critical layer indices
        """
        ranges = [stats['range'] for stats in layer_sensitivity.values()]
        threshold = np.percentile(ranges, threshold_percentile)

        # Fix: Only mark as critical if range > 0 AND above threshold
        critical_layers = [
            layer for layer, stats in layer_sensitivity.items()
            if stats['range'] > 0 and stats['range'] >= threshold
        ]

        return sorted(critical_layers)

    def generate_report(
        self,
        results: List[Dict],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate evaluation report

        Args:
            results: List of intervention results
            output_file: Optional file to save report

        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("REASONING DIRECTION EVALUATION REPORT")
        report_lines.append("="*60)
        report_lines.append("")

        # Summary statistics
        report_lines.append("Summary Statistics:")
        report_lines.append(f"Total experiments: {len(results)}")

        if results:
            layers = set(r.get('layer') for r in results)
            strengths = set(r.get('strength') for r in results)
            report_lines.append(f"Layers tested: {sorted(layers)}")
            report_lines.append(f"Strengths tested: {sorted(strengths)}")

        report_lines.append("")

        # Add more detailed analysis here as needed

        report = '\n'.join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")

        return report
