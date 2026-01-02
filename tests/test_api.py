"""Tests for the if-verifiable API."""

import pytest

from if_verifiable import (
    IFBenchSample,
    IFEvalSample,
    InstructionResult,
    evaluate_output_for_sample,
    get_eval_data,
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
        assert isinstance(scores, dict)
        assert "strict" in scores
        assert "loose" in scores
        assert "all_strict" in scores
        assert "all_loose" in scores

        for result in results:
            assert isinstance(result, InstructionResult)

    def test_ifbench_evaluation_returns_results(self):
        sample = next(get_eval_data("ifbench"))
        results, scores = evaluate_output_for_sample("ifbench", sample, "Test response")

        assert isinstance(results, list)
        assert isinstance(scores, dict)
        assert 0.0 <= scores["strict"] <= 1.0
        assert 0.0 <= scores["loose"] <= 1.0

    def test_empty_response_returns_zero_scores(self):
        sample = next(get_eval_data("ifeval"))
        results, scores = evaluate_output_for_sample("ifeval", sample, "")

        assert scores["strict"] == 0.0
        assert scores["loose"] == 0.0

    def test_invalid_benchmark_raises(self):
        sample = next(get_eval_data("ifeval"))
        with pytest.raises(ValueError, match="Unknown benchmark"):
            evaluate_output_for_sample("invalid", sample, "Test")


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
