import enum
from dataclasses import dataclass
from typing import Any


class RewardType(enum.Enum):
    """Types of reward/scoring metrics."""

    BINARY_STRICT = "binary_strict"
    BINARY_LOOSE = "binary_loose"
    PARTIAL_STRICT = "partial_strict"
    PARTIAL_LOOSE = "partial_loose"


@dataclass
class InstructionResult:
    """Result of evaluating a single instruction."""

    instruction_id: str
    strict_pass: bool
    loose_pass: bool


@dataclass
class EvaluationScores:
    """Aggregated evaluation scores for a sample.

    Attributes:
        partial_strict: Fraction of instructions passed (strict mode, 0.0-1.0).
        partial_loose: Fraction of instructions passed (loose mode, 0.0-1.0).
        binary_strict: 1.0 if ALL instructions passed strict, else 0.0.
        binary_loose: 1.0 if ALL instructions passed loose, else 0.0.
    """

    partial_strict: float
    partial_loose: float
    binary_strict: float
    binary_loose: float


def get_loose_transformations(response: str) -> list[str]:
    """Generate response variants for loose evaluation.

    Based on official IFBench/IFEval methodology:
    - Try removing first line, last line, or both (handles formatting artifacts)
    - Also try removing markdown asterisks (handles bold/italic formatting)
    """
    lines = response.split("\n")
    response_remove_first = "\n".join(lines[1:]).strip()
    response_remove_last = "\n".join(lines[:-1]).strip()
    response_remove_both = "\n".join(lines[1:-1]).strip()

    # Also try with markdown asterisks removed
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")

    return [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]


def strip_thinking(content: str) -> str:
    """Remove thinking blocks from model output (for reasoning models)."""
    s = content.strip()
    if not s:
        return s
    if "</think>" in s:
        return s.split("</think>", 1)[-1].strip()
    if s.startswith("<think>"):
        return ""
    return s


def evaluate_instructions(
    response: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
    prompt: str,
    instruction_registry: dict[str, Any],
) -> tuple[list[InstructionResult], EvaluationScores]:
    """Evaluate response against all instructions.

    Returns:
        - List of InstructionResult with strict/loose pass status for each instruction
        - EvaluationScores with all 4 scoring metrics
    """
    response = strip_thinking(response)
    if not response.strip():
        return [], EvaluationScores(
            partial_strict=0.0,
            partial_loose=0.0,
            binary_strict=0.0,
            binary_loose=0.0,
        )

    results: list[InstructionResult] = []
    response_variants = get_loose_transformations(response)

    for idx, instruction_id in enumerate(instruction_id_list):
        # Get the checker class and instantiate it
        checker_class = instruction_registry.get(instruction_id)
        if checker_class is None:
            # Unknown instruction type - skip
            continue

        checker = checker_class(instruction_id)

        # Build the checker with the provided kwargs
        instruction_kwargs = kwargs_list[idx] if kwargs_list else {}
        filtered_kwargs = {k: v for k, v in instruction_kwargs.items() if v is not None}
        checker.build_description(**filtered_kwargs)

        # Some checkers need the original prompt
        args = checker.get_instruction_args()
        if args and "prompt" in args:
            checker.build_description(prompt=prompt)

        # Strict evaluation: check original response only
        strict_pass = bool(response.strip() and checker.check_following(response))

        # Loose evaluation: check if ANY variant passes
        loose_pass = False
        for variant in response_variants:
            if variant.strip() and checker.check_following(variant):
                loose_pass = True
                break

        results.append(
            InstructionResult(
                instruction_id=instruction_id,
                strict_pass=strict_pass,
                loose_pass=loose_pass,
            )
        )

    # Calculate scores
    num_instructions = len(results)
    if num_instructions == 0:
        return results, EvaluationScores(
            partial_strict=0.0,
            partial_loose=0.0,
            binary_strict=0.0,
            binary_loose=0.0,
        )

    strict_correct = sum(r.strict_pass for r in results)
    loose_correct = sum(r.loose_pass for r in results)

    return results, EvaluationScores(
        # Instruction-level: fraction of instructions that pass
        partial_strict=strict_correct / num_instructions,
        partial_loose=loose_correct / num_instructions,
        # Prompt-level: all instructions must pass (binary)
        binary_strict=float(strict_correct == num_instructions),
        binary_loose=float(loose_correct == num_instructions),
    )
