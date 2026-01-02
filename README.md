# if-verifiable

Lightweight Python library for evaluating LLM outputs against instruction-following benchmarks.

Supports:
- **IFEval** (`google/IFEval`) - Google's Instruction Following Eval
- **IFBench** (`allenai/IFBench_test`) - Allen AI's instruction-following benchmark

## Installation

```bash
pip install if-verifiable
```

## Usage

```python
from if_verifiable import get_eval_data, evaluate_output_for_sample

# Load samples from a benchmark
for sample in get_eval_data("ifeval"):
    print(f"Prompt: {sample.prompt[:100]}...")
    print(f"Instructions: {sample.instruction_id_list}")
    break

# Evaluate a model's response
sample = next(get_eval_data("ifeval"))
response = "Your model's response here..."

results, scores = evaluate_output_for_sample("ifeval", sample, response)

# Access scores (4 metrics available)
print(f"Partial strict: {scores.partial_strict:.2%}")
print(f"Partial loose: {scores.partial_loose:.2%}")
print(f"Binary strict (all passed): {scores.binary_strict}")
print(f"Binary loose (all passed): {scores.binary_loose}")

# Check individual instruction results
for result in results:
    print(f"  {result.instruction_id}: strict={result.strict_pass}, loose={result.loose_pass}")
```

## API

### `get_eval_data(benchmark: str) -> Iterator[BenchmarkSample]`

Load evaluation samples from a benchmark dataset.

- `benchmark`: Either `"ifeval"` or `"ifbench"`
- Returns: Iterator of `IFEvalSample` or `IFBenchSample` dataclasses

### `evaluate_output_for_sample(benchmark, sample, response) -> tuple[list[InstructionResult], EvaluationScores]`

Evaluate a model response against a benchmark sample.

- `benchmark`: Either `"ifeval"` or `"ifbench"`
- `sample`: A sample from `get_eval_data()`
- `response`: The model's text response

Returns:
- `list[InstructionResult]`: Per-instruction pass/fail results
- `EvaluationScores`: Aggregated scores dataclass with 4 metrics:
  - `partial_strict`: Fraction of instructions passed (strict evaluation)
  - `partial_loose`: Fraction of instructions passed (loose - allows formatting variations)
  - `binary_strict`: 1.0 if ALL instructions passed strict, else 0.0
  - `binary_loose`: 1.0 if ALL instructions passed loose, else 0.0

## Types

```python
@dataclass
class IFEvalSample:
    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]

@dataclass  
class IFBenchSample:
    key: str
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]

@dataclass
class EvaluationScores:
    partial_strict: float  # Fraction of instructions passed (strict)
    partial_loose: float   # Fraction of instructions passed (loose)
    binary_strict: float   # 1.0 if all passed strict, else 0.0
    binary_loose: float    # 1.0 if all passed loose, else 0.0

@dataclass
class InstructionResult:
    instruction_id: str
    strict_pass: bool
    loose_pass: bool
```

## License

Apache 2.0
