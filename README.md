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

### Batch Evaluation

```python
from if_verifiable import run_eval, run_eval_async, get_eval_data

# Sync batch evaluation with multiprocessing
model_responses = ["response1", "response2", ...]  # One per sample
results = run_eval("ifeval", model_responses, max_workers=8)

for sample, response, instruction_results, scores in results:
    print(f"{sample.key}: {scores.partial_strict:.2%}")
```

### Async Evaluation

```python
import asyncio
from if_verifiable import run_eval_async, get_eval_data

async def get_model_response(prompt: str) -> dict:
    # Your async API call here
    return {"content": "model response", "usage": {...}}

samples = list(get_eval_data("ifeval"))
coroutines = [get_model_response(s.prompt) for s in samples]

# Evaluate concurrently with a map function to extract the response string
results = await run_eval_async(
    "ifeval",
    coroutines,
    map_fn=lambda r: r["content"]
)
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

### `run_eval(benchmark, model_responses, max_workers=None) -> list[EvalResult]`

Batch evaluate all responses with multiprocessing.

- `benchmark`: Either `"ifeval"` or `"ifbench"`
- `model_responses`: List of response strings, one per sample in dataset
- `max_workers`: Number of parallel workers (None = auto)

Returns list of `(sample, response, instruction_results, scores)` tuples.

### `run_eval_async(benchmark, coroutines, map_fn=str) -> list[EvalResult]`

Evaluate responses from async coroutines concurrently.

- `benchmark`: Either `"ifeval"` or `"ifbench"`
- `coroutines`: List of awaitables, one per sample
- `map_fn`: Function to extract response string from coroutine result

Returns list of `(sample, response, instruction_results, scores)` tuples in input order.

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

# Type alias for batch evaluation results
EvalResult = tuple[BenchmarkSample, str, list[InstructionResult], EvaluationScores]
```

## License

Apache 2.0
