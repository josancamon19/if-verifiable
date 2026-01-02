"""Main API for if-verifiable package."""

from typing import Iterator

from datasets import load_dataset

from if_verifiable.common import EvaluationScores, InstructionResult, evaluate_instructions
from if_verifiable.types import (
    BenchmarkName,
    BenchmarkSample,
    IFBenchSample,
    IFEvalSample,
)

# Dataset names on HuggingFace
_DATASETS = {
    "ifeval": "google/IFEval",
    "ifbench": "allenai/IFBench_test",
}


def get_eval_data(benchmark: BenchmarkName) -> Iterator[BenchmarkSample]:
    """Load evaluation data from the specified benchmark.

    Args:
        benchmark: Either "ifbench" or "ifeval".

    Yields:
        Sample dataclass objects from the benchmark dataset.

    Raises:
        ValueError: If benchmark name is not recognized.

    Example:
        >>> for sample in get_eval_data("ifeval"):
        ...     print(sample.prompt)
    """
    benchmark = benchmark.lower()

    if benchmark not in _DATASETS:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. Must be one of: {list(_DATASETS.keys())}"
        )

    dataset_name = _DATASETS[benchmark]

    # Load dataset from HuggingFace
    # IFEval has "train" split, IFBench may have "test"
    dataset = load_dataset(dataset_name)

    # Get the appropriate split
    if "train" in dataset:
        split = dataset["train"]
    elif "test" in dataset:
        split = dataset["test"]
    else:
        # Use first available split
        split = dataset[list(dataset.keys())[0]]

    # Yield typed samples
    for row in split:
        if benchmark == "ifeval":
            yield IFEvalSample(
                key=row["key"],
                prompt=row["prompt"],
                instruction_id_list=row["instruction_id_list"],
                kwargs=row["kwargs"],
            )
        else:  # ifbench
            yield IFBenchSample(
                key=row["key"],
                prompt=row["prompt"],
                instruction_id_list=row["instruction_id_list"],
                kwargs=row["kwargs"],
            )


def evaluate_output_for_sample(
    benchmark: BenchmarkName,
    sample: BenchmarkSample,
    response: str,
) -> tuple[list[InstructionResult], EvaluationScores]:
    """Evaluate a model's response against a benchmark sample.

    Args:
        benchmark: Either "ifbench" or "ifeval".
        sample: A sample from the benchmark dataset.
        response: The model's response to evaluate.

    Returns:
        A tuple of:
        - List of InstructionResult with pass/fail for each instruction
        - EvaluationScores with all 4 metrics:
            - partial_strict: Fraction of instructions passed (strict mode)
            - partial_loose: Fraction of instructions passed (loose mode)
            - binary_strict: 1.0 if ALL instructions passed strict, else 0.0
            - binary_loose: 1.0 if ALL instructions passed loose, else 0.0

    Raises:
        ValueError: If benchmark name is not recognized.

    Example:
        >>> sample = next(get_eval_data("ifeval"))
        >>> results, scores = evaluate_output_for_sample("ifeval", sample, "Hello world!")
        >>> print(f"Partial strict: {scores.partial_strict:.2%}")
    """
    benchmark = benchmark.lower()

    if benchmark not in _DATASETS:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. Must be one of: {list(_DATASETS.keys())}"
        )

    # Get the appropriate instruction registry
    if benchmark == "ifeval":
        from if_verifiable.ifeval.instructions_registry import INSTRUCTION_DICT
    else:  # ifbench
        from if_verifiable.ifbench.instructions_registry import INSTRUCTION_DICT

    return evaluate_instructions(
        response=response,
        instruction_id_list=sample.instruction_id_list,
        kwargs_list=sample.kwargs,
        prompt=sample.prompt,
        instruction_registry=INSTRUCTION_DICT,
    )
