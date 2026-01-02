"""if-verifiable: Lightweight evaluation for IFBench and IFEval benchmarks.

Usage:
    >>> from if_verifiable import get_eval_data, evaluate_output_for_sample
    >>> 
    >>> # Load data from a benchmark
    >>> for sample in get_eval_data("ifeval"):
    ...     print(sample.prompt)
    ...     break
    >>> 
    >>> # Evaluate a model response
    >>> results, scores = evaluate_output_for_sample("ifeval", sample, "Model output here")
    >>> print(f"Strict: {scores['strict']:.2%}, Loose: {scores['loose']:.2%}")
"""

from if_verifiable.api import get_eval_data, evaluate_output_for_sample
from if_verifiable.types import IFEvalSample, IFBenchSample, BenchmarkSample
from if_verifiable.common import InstructionResult, EvaluationScores, RewardType

__version__ = "0.1.0"

__all__ = [
    # Main API
    "get_eval_data",
    "evaluate_output_for_sample",
    # Types
    "IFEvalSample",
    "IFBenchSample",
    "BenchmarkSample",
    "InstructionResult",
    "EvaluationScores",
    "RewardType",
    # Version
    "__version__",
]

