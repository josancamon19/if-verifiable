"""Tests for the if-verifiable API."""

import asyncio
from unittest.mock import patch

import pytest

from if_verifiable import (
    EvalResult,
    EvaluationScores,
    IFBenchSample,
    IFEvalSample,
    InstructionResult,
    RewardType,
    evaluate_output_for_sample,
    get_eval_data,
    run_eval,
    run_eval_async,
)


class TestGetEvalData:
    def test_ifeval_returns_samples(self):
        sample = next(get_eval_data("ifeval"))
        assert isinstance(sample, IFEvalSample)
        assert isinstance(sample.key, int)
        assert isinstance(sample.prompt, str)
        assert isinstance(sample.instruction_id_list, list)
        assert isinstance(sample.kwargs, list)
        assert len(sample.prompt) > 0

    def test_ifbench_returns_samples(self):
        sample = next(get_eval_data("ifbench"))
        assert isinstance(sample, IFBenchSample)
        assert isinstance(sample.key, str)
        assert isinstance(sample.prompt, str)
        assert isinstance(sample.instruction_id_list, list)
        assert isinstance(sample.kwargs, list)
        assert len(sample.prompt) > 0

    def test_invalid_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            next(get_eval_data("invalid"))

    def test_case_insensitive(self):
        sample_lower = next(get_eval_data("ifeval"))
        sample_upper = next(get_eval_data("IFEVAL"))
        assert sample_lower.key == sample_upper.key


class TestEvaluateOutput:
    def test_ifeval_evaluation_returns_results(self):
        sample = next(get_eval_data("ifeval"))
        results, scores = evaluate_output_for_sample("ifeval", sample, "Test response")

        assert isinstance(results, list)
        assert isinstance(scores, EvaluationScores)

        for result in results:
            assert isinstance(result, InstructionResult)

    def test_ifbench_evaluation_returns_results(self):
        sample = next(get_eval_data("ifbench"))
        results, scores = evaluate_output_for_sample("ifbench", sample, "Test response")

        assert isinstance(results, list)
        assert isinstance(scores, EvaluationScores)
        assert 0.0 <= scores.partial_strict <= 1.0
        assert 0.0 <= scores.partial_loose <= 1.0

    def test_empty_response_returns_zero_scores(self):
        sample = next(get_eval_data("ifeval"))
        results, scores = evaluate_output_for_sample("ifeval", sample, "")

        assert scores.partial_strict == 0.0
        assert scores.partial_loose == 0.0
        assert scores.binary_strict == 0.0
        assert scores.binary_loose == 0.0

    def test_invalid_benchmark_raises(self):
        sample = next(get_eval_data("ifeval"))
        with pytest.raises(ValueError, match="Unknown benchmark"):
            evaluate_output_for_sample("invalid", sample, "Test")


class TestEvaluationScores:
    def test_all_score_fields_present(self):
        sample = next(get_eval_data("ifeval"))
        _, scores = evaluate_output_for_sample("ifeval", sample, "Test response")

        assert hasattr(scores, "partial_strict")
        assert hasattr(scores, "partial_loose")
        assert hasattr(scores, "binary_strict")
        assert hasattr(scores, "binary_loose")

    def test_score_values_are_floats(self):
        sample = next(get_eval_data("ifeval"))
        _, scores = evaluate_output_for_sample("ifeval", sample, "Test response")

        assert isinstance(scores.partial_strict, float)
        assert isinstance(scores.partial_loose, float)
        assert isinstance(scores.binary_strict, float)
        assert isinstance(scores.binary_loose, float)

    def test_partial_scores_are_fractions(self):
        sample = next(get_eval_data("ifeval"))
        _, scores = evaluate_output_for_sample("ifeval", sample, "Test response")

        assert 0.0 <= scores.partial_strict <= 1.0
        assert 0.0 <= scores.partial_loose <= 1.0

    def test_binary_scores_are_binary(self):
        sample = next(get_eval_data("ifeval"))
        _, scores = evaluate_output_for_sample("ifeval", sample, "Test response")

        assert scores.binary_strict in (0.0, 1.0)
        assert scores.binary_loose in (0.0, 1.0)

    def test_loose_at_least_as_good_as_strict(self):
        sample = next(get_eval_data("ifeval"))
        _, scores = evaluate_output_for_sample("ifeval", sample, "Test response")

        # Loose mode allows more transformations, so should be >= strict
        assert scores.partial_loose >= scores.partial_strict
        assert scores.binary_loose >= scores.binary_strict


class TestInstructionResults:
    def test_instruction_result_fields(self):
        sample = next(get_eval_data("ifeval"))
        results, _ = evaluate_output_for_sample("ifeval", sample, "Test response")

        if results:
            result = results[0]
            assert hasattr(result, "instruction_id")
            assert hasattr(result, "strict_pass")
            assert hasattr(result, "loose_pass")
            assert isinstance(result.strict_pass, bool)
            assert isinstance(result.loose_pass, bool)


class TestRewardType:
    def test_reward_type_values(self):
        assert RewardType.BINARY_STRICT.value == "binary_strict"
        assert RewardType.BINARY_LOOSE.value == "binary_loose"
        assert RewardType.PARTIAL_STRICT.value == "partial_strict"
        assert RewardType.PARTIAL_LOOSE.value == "partial_loose"

    def test_reward_type_matches_scores(self):
        sample = next(get_eval_data("ifeval"))
        _, scores = evaluate_output_for_sample("ifeval", sample, "Test response")

        # All RewardType values should be valid score keys
        for rt in RewardType:
            assert hasattr(scores, rt.value)


class TestRunEval:
    def test_run_eval_returns_correct_structure(self):
        # Get a small subset of samples for testing
        samples = list(get_eval_data("ifeval"))[:3]
        responses = ["Test response"] * len(samples)

        # Mock get_eval_data to return only our subset
        with patch("if_verifiable.api.get_eval_data", return_value=iter(samples)):
            results = run_eval("ifeval", responses, max_workers=1)

        assert len(results) == 3
        for sample, response, instruction_results, scores in results:
            assert isinstance(sample, IFEvalSample)
            assert isinstance(response, str)
            assert isinstance(instruction_results, list)
            assert isinstance(scores, EvaluationScores)

    def test_run_eval_mismatched_length_raises(self):
        samples = list(get_eval_data("ifeval"))[:3]

        with patch("if_verifiable.api.get_eval_data", return_value=iter(samples)):
            with pytest.raises(ValueError, match="Expected 3 responses, got 2"):
                run_eval("ifeval", ["a", "b"], max_workers=1)

    def test_run_eval_preserves_order(self):
        samples = list(get_eval_data("ifeval"))[:3]
        responses = ["response_0", "response_1", "response_2"]

        with patch("if_verifiable.api.get_eval_data", return_value=iter(samples)):
            results = run_eval("ifeval", responses, max_workers=1)

        for i, (sample, response, _, _) in enumerate(results):
            assert response == f"response_{i}"
            assert sample.key == samples[i].key


class TestRunEvalAsync:
    def test_run_eval_async_returns_correct_structure(self):
        samples = list(get_eval_data("ifeval"))[:3]

        async def mock_response(text: str) -> str:
            return text

        coroutines = [mock_response("Test response") for _ in samples]

        with patch("if_verifiable.api.get_eval_data", return_value=iter(samples)):
            results = asyncio.run(run_eval_async("ifeval", coroutines))

        assert len(results) == 3
        for sample, response, instruction_results, scores in results:
            assert isinstance(sample, IFEvalSample)
            assert response == "Test response"
            assert isinstance(instruction_results, list)
            assert isinstance(scores, EvaluationScores)

    def test_run_eval_async_with_map_fn(self):
        samples = list(get_eval_data("ifeval"))[:3]

        async def mock_api_response(idx: int) -> dict:
            return {"content": f"response_{idx}", "status": 200}

        coroutines = [mock_api_response(i) for i in range(len(samples))]

        with patch("if_verifiable.api.get_eval_data", return_value=iter(samples)):
            results = asyncio.run(
                run_eval_async("ifeval", coroutines, map_fn=lambda r: r["content"])
            )

        assert len(results) == 3
        for i, (_, response, _, _) in enumerate(results):
            assert response == f"response_{i}"

    def test_run_eval_async_mismatched_length_raises(self):
        samples = list(get_eval_data("ifeval"))[:3]

        async def run_with_wrong_count() -> None:
            async def mock_response() -> str:
                return "test"

            await run_eval_async("ifeval", [mock_response(), mock_response()])

        with patch("if_verifiable.api.get_eval_data", return_value=iter(samples)):
            with pytest.raises(ValueError, match="Expected 3 coroutines, got 2"):
                asyncio.run(run_with_wrong_count())

    def test_run_eval_async_preserves_order(self):
        samples = list(get_eval_data("ifeval"))[:3]

        async def mock_response(idx: int) -> str:
            # Add varying delays to test order preservation
            await asyncio.sleep(0.01 * (3 - idx))
            return f"response_{idx}"

        coroutines = [mock_response(i) for i in range(len(samples))]

        with patch("if_verifiable.api.get_eval_data", return_value=iter(samples)):
            results = asyncio.run(run_eval_async("ifeval", coroutines))

        # Results should be in input order, not completion order
        for i, (sample, response, _, _) in enumerate(results):
            assert response == f"response_{i}"
            assert sample.key == samples[i].key
