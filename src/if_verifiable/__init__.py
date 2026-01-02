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

from if_verifiable.api import (
    EvalResult,
    evaluate_output_for_sample,
    get_eval_data,
    run_eval,
    run_eval_async,
)
from if_verifiable.common import EvaluationScores, InstructionResult, RewardType
from if_verifiable.types import BenchmarkSample, IFBenchSample, IFEvalSample

__version__ = "0.1.1"

__all__ = [
    # Main API
    "get_eval_data",
    "evaluate_output_for_sample",
    "run_eval",
    "run_eval_async",
    # Types
    "IFEvalSample",
    "IFBenchSample",
    "BenchmarkSample",
    "InstructionResult",
    "EvaluationScores",
    "EvalResult",
    "RewardType",
    # Version
    "__version__",
]
