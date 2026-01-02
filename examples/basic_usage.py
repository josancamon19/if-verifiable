from if_verifiable import evaluate_output_for_sample, get_eval_data


def main():
    # Example 1: Evaluate against IFEval
    print("=== IFEval Example ===\n")

    sample = next(get_eval_data("ifeval"))
    print(f"Prompt: {sample.prompt[:100]}...")
    print(f"Instructions: {sample.instruction_id_list}")
    print(f"Kwargs: {sample.kwargs}\n")

    # Simulate a model response
    response = "This is a sample response without any commas."

    results, scores = evaluate_output_for_sample("ifeval", sample, response)
    print(f"Partial strict: {scores.partial_strict:.2%}")
    print(f"Partial loose: {scores.partial_loose:.2%}")
    print(f"Binary strict (all passed): {bool(scores.binary_strict)}")
    print(f"Binary loose (all passed): {bool(scores.binary_loose)}\n")

    for result in results:
        status = "✓" if result.strict_pass else "✗"
        print(f"  {status} {result.instruction_id}")

    # Example 2: Evaluate against IFBench
    print("\n=== IFBench Example ===\n")

    sample = next(get_eval_data("ifbench"))
    print(f"Prompt: {sample.prompt[:100]}...")
    print(f"Instructions: {sample.instruction_id_list}\n")

    # This sample wants specific keyword counts
    response = """
    The kaleidoscope of life reveals many truths.
    Like a nebula in space, nebula formations guide us.
    We whisper secrets, whisper hopes, whisper dreams.
    Through the labyrinth we walk, labyrinth paths wind,
    labyrinth choices emerge, labyrinth doors open, labyrinth ends.
    Each paradox teaches us, paradox after paradox,
    paradox within paradox, paradox beyond paradox,
    paradox eternal, paradox infinite, paradox complete.
    """

    results, scores = evaluate_output_for_sample("ifbench", sample, response)
    print(f"Partial strict: {scores.partial_strict:.2%}")
    print(f"Partial loose: {scores.partial_loose:.2%}\n")

    for result in results:
        status = "✓" if result.strict_pass else "✗"
        print(f"  {status} {result.instruction_id}")


if __name__ == "__main__":
    main()
