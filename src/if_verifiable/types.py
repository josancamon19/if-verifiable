"""Type definitions for IFBench and IFEval samples."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class IFEvalSample:
    """A sample from the IFEval dataset (google/IFEval).
    
    Attributes:
        key: Unique identifier for the sample.
        prompt: The input prompt to evaluate.
        instruction_id_list: List of instruction type IDs (e.g., "keywords:existence").
        kwargs: List of keyword argument dicts for each instruction.
    """
    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


@dataclass
class IFBenchSample:
    """A sample from the IFBench dataset (allenai/IFBench_test).
    
    Attributes:
        key: Unique identifier for the sample (string in IFBench).
        prompt: The input prompt to evaluate.
        instruction_id_list: List of instruction type IDs (e.g., "count:word_count_range").
        kwargs: List of keyword argument dicts for each instruction.
    """
    key: str
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


# Union type for any sample
BenchmarkSample = IFEvalSample | IFBenchSample
BenchmarkName = Literal["ifbench", "ifeval"]

