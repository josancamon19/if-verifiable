# Copyright 2025 Joan Cabezas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive tests to verify all instruction checkers work correctly.

This test ensures every instruction type can be instantiated, have its description
built, and have check_following called without errors.
"""

import pytest


# Instructions that require special kwargs (like prompt_to_repeat)
SKIP_INSTRUCTIONS_BUILD = {
    "repeat:repeat_change",
    "repeat:repeat_span",
    "combination:repeat_prompt",
}

# Instructions that require build_description with specific kwargs before check_following
SKIP_INSTRUCTIONS_CHECK = {
    "repeat:repeat_change",
    "repeat:repeat_span",
    "ratio:overlap",  # requires reference_ngrams to be set
    "combination:repeat_prompt",
}


class TestIFBenchInstructions:
    """Test all IFBench instruction types."""

    def test_all_ifbench_instruction_classes_can_be_instantiated(self):
        """Ensure all IFBench instruction classes can be instantiated and used."""
        from if_verifiable.ifbench.instructions_registry import INSTRUCTION_DICT

        errors = []
        for instruction_id, instruction_class in INSTRUCTION_DICT.items():
            try:
                # Instantiate
                checker = instruction_class(instruction_id)
                assert checker is not None, f"Failed to instantiate {instruction_id}"
            except Exception as e:
                errors.append(f"{instruction_id}: instantiation failed - {e}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_all_ifbench_checkers_can_build_description(self):
        """Ensure all IFBench checkers can build their description."""
        from if_verifiable.ifbench.instructions_registry import INSTRUCTION_DICT

        errors = []
        for instruction_id, instruction_class in INSTRUCTION_DICT.items():
            if instruction_id in SKIP_INSTRUCTIONS_BUILD:
                continue
            try:
                checker = instruction_class(instruction_id)
                description = checker.build_description()
                assert description is not None, f"{instruction_id}: description is None"
                assert isinstance(description, str), (
                    f"{instruction_id}: description is not a string"
                )
            except Exception as e:
                errors.append(f"{instruction_id}: build_description failed - {e}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_all_ifbench_checkers_can_run_check_following(self):
        """Ensure all IFBench checkers can run check_following without crashing."""
        from if_verifiable.ifbench.instructions_registry import INSTRUCTION_DICT

        # Sample responses that might trigger various checks
        test_responses = [
            "Hello world. This is a test response with some words.",
            "Emma and Liam went to the store. They bought apples and oranges.",
            "1. First item\n2. Second item\n3. Third item",
            "The quick brown fox jumps over the lazy dog.",
            "日本語テスト文章です。",
            "Option A: Yes\nOption B: No\nOption C: Maybe",
            "🎉 This is exciting! 🎊 Very exciting indeed! 🎈",
        ]

        errors = []
        for instruction_id, instruction_class in INSTRUCTION_DICT.items():
            if instruction_id in SKIP_INSTRUCTIONS_CHECK:
                continue
            try:
                checker = instruction_class(instruction_id)
                checker.build_description()

                # Try with multiple test responses
                for response in test_responses:
                    result = checker.check_following(response)
                    assert isinstance(result, bool), (
                        f"{instruction_id}: check_following did not return bool"
                    )
            except Exception as e:
                errors.append(f"{instruction_id}: check_following failed - {e}")

        if errors:
            pytest.fail("\n".join(errors))


class TestIFEvalInstructions:
    """Test all IFEval instruction types."""

    def test_all_ifeval_instruction_classes_can_be_instantiated(self):
        """Ensure all IFEval instruction classes can be instantiated and used."""
        from if_verifiable.ifeval.instructions_registry import INSTRUCTION_DICT

        errors = []
        for instruction_id, instruction_class in INSTRUCTION_DICT.items():
            try:
                # Instantiate
                checker = instruction_class(instruction_id)
                assert checker is not None, f"Failed to instantiate {instruction_id}"
            except Exception as e:
                errors.append(f"{instruction_id}: instantiation failed - {e}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_all_ifeval_checkers_can_build_description(self):
        """Ensure all IFEval checkers can build their description."""
        from if_verifiable.ifeval.instructions_registry import INSTRUCTION_DICT

        errors = []
        for instruction_id, instruction_class in INSTRUCTION_DICT.items():
            if instruction_id in SKIP_INSTRUCTIONS_BUILD:
                continue
            try:
                checker = instruction_class(instruction_id)
                description = checker.build_description()
                assert description is not None, f"{instruction_id}: description is None"
                assert isinstance(description, str), (
                    f"{instruction_id}: description is not a string"
                )
            except Exception as e:
                errors.append(f"{instruction_id}: build_description failed - {e}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_all_ifeval_checkers_can_run_check_following(self):
        """Ensure all IFEval checkers can run check_following without crashing."""
        from if_verifiable.ifeval.instructions_registry import INSTRUCTION_DICT

        # Sample responses that might trigger various checks
        test_responses = [
            "Hello world. This is a test response with some words.",
            "The answer is: Yes, I agree with that statement.",
            "# Title\n\nFirst paragraph here.\n\nSecond paragraph here.",
            "P.S. This is a postscript.",
            '{"key": "value", "number": 42}',
            "***highlighted section*** and more text",
            "First response.\n\n******\n\nSecond response.",
        ]

        errors = []
        for instruction_id, instruction_class in INSTRUCTION_DICT.items():
            if instruction_id in SKIP_INSTRUCTIONS_CHECK:
                continue
            try:
                checker = instruction_class(instruction_id)
                checker.build_description()

                # Try with multiple test responses
                for response in test_responses:
                    result = checker.check_following(response)
                    assert isinstance(result, bool), (
                        f"{instruction_id}: check_following did not return bool"
                    )
            except Exception as e:
                errors.append(f"{instruction_id}: check_following failed - {e}")

        if errors:
            pytest.fail("\n".join(errors))


class TestIntegrationWithRealData:
    """Test using actual benchmark data samples."""

    def test_ifbench_with_real_samples(self):
        """Test IFBench checkers with real samples from the dataset."""
        from if_verifiable import evaluate_output_for_sample, get_eval_data

        # Get a small sample of real data
        count = 0
        for sample in get_eval_data("ifbench"):
            # Just make sure evaluation doesn't crash
            result = evaluate_output_for_sample(
                "ifbench",
                sample,
                "This is a test response that probably won't follow all instructions.",
            )
            assert result is not None
            # Result is a tuple of (instruction_results, scores)
            instruction_results, scores = result
            assert isinstance(instruction_results, list)
            assert scores is not None
            count += 1
            if count >= 5:
                break

    def test_ifeval_with_real_samples(self):
        """Test IFEval checkers with real samples from the dataset."""
        from if_verifiable import evaluate_output_for_sample, get_eval_data

        # Get a small sample of real data
        count = 0
        for sample in get_eval_data("ifeval"):
            # Just make sure evaluation doesn't crash
            result = evaluate_output_for_sample(
                "ifeval",
                sample,
                "This is a test response that probably won't follow all instructions.",
            )
            assert result is not None
            # Result is a tuple of (instruction_results, scores)
            instruction_results, scores = result
            assert isinstance(instruction_results, list)
            assert scores is not None
            count += 1
            if count >= 5:
                break

    def test_ifbench_all_instruction_types_in_dataset(self):
        """Verify all IFBench instruction types from the dataset can be evaluated."""
        from if_verifiable import evaluate_output_for_sample, get_eval_data

        # Collect all instruction types seen in the dataset
        seen_instructions = set()
        samples_by_instruction = {}

        for sample in get_eval_data("ifbench"):
            for instruction_id in sample.instruction_id_list:
                if instruction_id not in seen_instructions:
                    seen_instructions.add(instruction_id)
                    samples_by_instruction[instruction_id] = sample

        errors = []
        for instruction_id, sample in samples_by_instruction.items():
            try:
                result = evaluate_output_for_sample(
                    "ifbench",
                    sample,
                    "Test response with Emma and Liam. 🎉 Hello! 🎊 World! Option A: Yes",
                )
                assert result is not None
            except Exception as e:
                errors.append(f"{instruction_id}: evaluation failed - {e}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_ifeval_all_instruction_types_in_dataset(self):
        """Verify all IFEval instruction types from the dataset can be evaluated."""
        from if_verifiable import evaluate_output_for_sample, get_eval_data

        # Collect all instruction types seen in the dataset
        seen_instructions = set()
        samples_by_instruction = {}

        for sample in get_eval_data("ifeval"):
            for instruction_id in sample.instruction_id_list:
                if instruction_id not in seen_instructions:
                    seen_instructions.add(instruction_id)
                    samples_by_instruction[instruction_id] = sample

        errors = []
        for instruction_id, sample in samples_by_instruction.items():
            try:
                result = evaluate_output_for_sample(
                    "ifeval", sample, "Test response. P.S. This is a test."
                )
                assert result is not None
            except Exception as e:
                errors.append(f"{instruction_id}: evaluation failed - {e}")

        if errors:
            pytest.fail("\n".join(errors))
