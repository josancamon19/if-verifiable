# if-verifiable

Lightweight Python library for evaluating LLM outputs against instruction-following benchmarks.

Supports:
- **IFEval** (`google/IFEval`) - Google's Instruction Following Eval
- **IFBench** (`allenai/IFBench`) - Allen AI's instruction-following benchmark

## Installation

```bash
pip install if-verifiable
```

For IFBench support (includes emoji/syllable checkers):
```bash
pip install if-verifiable[ifbench]
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

print(f"Strict score: {scores['strict']:.2%}")
print(f"Loose score: {scores['loose']:.2%}")
print(f"All instructions passed (strict): {scores['all_strict']}")
print(f"All instructions passed (loose): {scores['all_loose']}")

# Check individual instruction results
for result in results:
    print(f"  {result.instruction_id}: strict={result.strict_pass}, loose={result.loose_pass}")
```

## API

### `get_eval_data(benchmark: str) -> Iterator[BenchmarkSample]`

Load evaluation samples from a benchmark dataset.

- `benchmark`: Either `"ifeval"` or `"ifbench"`
- Returns: Iterator of `IFEvalSample` or `IFBenchSample` dataclasses

### `evaluate_output_for_sample(benchmark, sample, response) -> tuple[list[InstructionResult], dict]`

Evaluate a model response against a benchmark sample.

- `benchmark`: Either `"ifeval"` or `"ifbench"`
- `sample`: A sample from `get_eval_data()`
- `response`: The model's text response

Returns:
- `list[InstructionResult]`: Per-instruction pass/fail results
- `dict`: Aggregated scores with keys:
  - `strict`: Fraction of instructions passed (strict evaluation)
  - `loose`: Fraction of instructions passed (loose evaluation - allows minor formatting variations)
  - `all_strict`: 1.0 if all instructions passed strict, else 0.0
  - `all_loose`: 1.0 if all instructions passed loose, else 0.0

## Sample Types

```python
@dataclass
class IFEvalSample:
    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]

@dataclass  
class IFBenchSample:
    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]
```

## License

Apache 2.0

