import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import keras
import keras_hub
from dotenv import load_dotenv
from google.generativeai.generative_models import GenerativeModel
from keras_hub.models import Gemma3CausalLM

from utils import LoadedDataset, generate_from_model, truncate_sample

# Load environment variables for API access
load_dotenv()


def set_environment():
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"


def load_model(preset: str) -> Gemma3CausalLM:
    model = Gemma3CausalLM.from_preset(preset)
    print(f"Loaded model: {preset}")
    return model


def run_inference(
    model: Gemma3CausalLM,
    instruction: str,
    max_length: int,
    seed: int,
    template: str = "Instruction:{instruction}\n\nResponse:{response}",
):
    prompt = template.format(instruction=instruction, response="")
    if not hasattr(model, '_compiled_for_inference'):
        sampler = keras_hub.samplers.TopKSampler(k=5, seed=seed)
        model.compile(sampler=sampler)
        model._compiled_for_inference = True
    output = model.generate(prompt, max_length=max_length)
    print(f"Inference output: {output}")
    return output


def load_data(file_path: str) -> Dict[str, List[str]]:
    prompts = []
    responses = []

    script_dir = Path(__file__).parent
    full_path = script_dir / file_path

    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {full_path}")

    with open(full_path) as file:
        data_list = json.load(file)

    for examples in data_list:
        prompts.append(examples["text_input"])
        responses.append(examples["output"])

    return {"prompts": prompts, "responses": responses}


def fine_tune_model(
    model: Gemma3CausalLM,
    data: Dict[str, List[str]],
    rank: int,
    sequence_length: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
):
    model.backbone.enable_lora(rank=rank)
    print(model.summary())

    model.preprocessor.sequence_length = sequence_length
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    model.fit(data, epochs=epochs, batch_size=batch_size)


def generate_eval_template(question: str, answer_1: str, answer_2: str):
    return (
        "You are a judge model. Your task is to compare 2 answers to a given question and score them based on a scale of 1 through 10."
        + "You are provided the original question, as well, as the corresponding answers by the 2 models."
        + "Judge both the answers on their conciseness, clarity and most importantly - accuracy."
        + "You should output the scores at the end of your deliberation. The scores should be in format: `JUDGE_SCORES: (<score_1>, <score_2>)` where the tuples contain two `int`s between 0 and 10."
        + f"This is the original question: ```markdown{question}```"
        + f"\nModel Answer 1: {answer_1}"
        + f"\nModel Answer 2: {answer_2}"
    )


def extract_scores(judge_response: str) -> tuple[int | None, int | None]:
    """Extract the scores from the judge model's response."""
    pattern = r"JUDGE_SCORES:\s*\((\d+),\s*(\d+)\)"
    match = re.search(pattern, judge_response)

    if match:
        try:
            score_1 = int(match.group(1))
            score_2 = int(match.group(2))
            return score_1, score_2
        except (ValueError, IndexError):
            pass

    print(f"Failed to extract scores from judge response: {judge_response[:100]}...")
    return None, None


def evaluate_model_locally(
    model: Gemma3CausalLM,
    eval_data: LoadedDataset,
    max_samples: int = 20,
    max_length: int = 256,
):
    """Evaluate model locally on a dataset by generating responses."""
    results = []

    print(f"Evaluating on {min(max_samples, len(eval_data))} samples...")
    
    # Compile once at the start
    if not hasattr(model, '_compiled_for_eval'):
        sampler = keras_hub.samplers.TopKSampler(k=5, seed=42)
        model.compile(sampler=sampler)
        model._compiled_for_eval = True

    for i, sample in enumerate(eval_data[:max_samples]):
        question = sample["text_input"]

        # Generate response using the model
        prompt = f"Question: {question}\nAnswer:"
        try:
            generated_output = model.generate(prompt, max_length=max_length)

            # Extract just the answer part (remove the prompt)
            if prompt in generated_output:
                answer = generated_output[len(prompt) :].strip()
            else:
                answer = generated_output.strip()

        except Exception as e:
            print(f"Error generating for sample {i}: {e}")
            answer = f"Error: {str(e)}"

        results.append(answer)

        if i % 5 == 0:
            print(f"Completed {i + 1}/{min(max_samples, len(eval_data))} samples")

    return results


def evaluate_models(
    eval_data: LoadedDataset,
    unfinetuned_model_name: str,
    finetuned_model_name: str,
    judge_model_name: str,
    baseline_model: str,
    max_samples: int = 20,
):
    """Evaluate both unfinetuned and finetuned models locally, then judge with Gemini."""
    # Generate responses from both models
    print("\n--- Evaluating unfinetuned model ---")
    unfinetuned_model = load_model(unfinetuned_model_name)
    unfinetuned_outputs = evaluate_model_locally(
        unfinetuned_model, eval_data, max_samples
    )

    print("\n--- Loading finetuned model ---")
    finetuned_model = load_model(unfinetuned_model_name)

    # Load the finetuned weights
    weights_path = (
        finetuned_model_name
        if finetuned_model_name.endswith(".weights.h5")
        else f"{finetuned_model_name}.weights.h5"
    )
    try:
        finetuned_model.load_weights(weights_path)
        print(f"Loaded finetuned weights from {weights_path}")
    except Exception as e:
        print(f"Warning: Could not load finetuned weights: {e}")
        print("Using unfinetuned model for comparison")

    print("\n--- Evaluating finetuned model ---")
    finetuned_outputs = evaluate_model_locally(finetuned_model, eval_data, max_samples)

    # Now judge the outputs using Gemini
    print(f"\n--- Judging outputs with {judge_model_name} ---")
    judge_model = GenerativeModel(judge_model_name)

    results = []
    for i, sample in enumerate(eval_data[:max_samples]):
        question = sample["text_input"]
        unfinetuned_answer = unfinetuned_outputs[i]
        finetuned_answer = finetuned_outputs[i]

        # Create judge prompt
        judge_prompt = generate_eval_template(
            question, finetuned_answer, unfinetuned_answer
        )

        try:
            judge_response = generate_from_model(
                judge_model, judge_prompt, verbose=False
            )
            finetuned_score, unfinetuned_score = extract_scores(judge_response)
        except Exception as e:
            print(f"Error getting judge response for sample {i}: {e}")
            judge_response = f"Error: {str(e)}"
            finetuned_score, unfinetuned_score = None, None

        result = {
            "question": question,
            "finetuned_output": finetuned_answer,
            "unfinetuned_output": unfinetuned_answer,
            "finetuned_score": finetuned_score,
            "unfinetuned_score": unfinetuned_score,
            "judge_response": judge_response,
        }
        results.append(result)

        if i % 5 == 0:
            print(f"Judged {i + 1}/{max_samples} samples")

    return results


def compare_performance(results):
    """Compare the performance between unfinetuned and finetuned models using judge scores."""
    print("\n--- Performance Comparison ---")

    # Extract valid scores
    valid_results = [
        r
        for r in results
        if r["finetuned_score"] is not None and r["unfinetuned_score"] is not None
    ]

    if not valid_results:
        print("❌ No valid scores found from judge model")
        return

    finetuned_scores = [r["finetuned_score"] for r in valid_results]
    unfinetuned_scores = [r["unfinetuned_score"] for r in valid_results]

    avg_finetuned = sum(finetuned_scores) / len(finetuned_scores)
    avg_unfinetuned = sum(unfinetuned_scores) / len(unfinetuned_scores)

    print(f"Valid evaluations: {len(valid_results)}/{len(results)}")
    print(f"Finetuned model average score: {avg_finetuned:.2f}")
    print(f"Unfinetuned model average score: {avg_unfinetuned:.2f}")
    print(f"Improvement: {avg_finetuned - avg_unfinetuned:.2f}")

    # Win/loss/tie statistics
    wins = sum(
        1 for r in valid_results if r["finetuned_score"] > r["unfinetuned_score"]
    )
    losses = sum(
        1 for r in valid_results if r["finetuned_score"] < r["unfinetuned_score"]
    )
    ties = sum(
        1 for r in valid_results if r["finetuned_score"] == r["unfinetuned_score"]
    )

    print(f"Finetuned model: {wins} wins, {losses} losses, {ties} ties")

    if avg_finetuned > avg_unfinetuned:
        print("✅ Finetuning improved performance!")
    elif avg_finetuned < avg_unfinetuned:
        print("❌ Finetuning decreased performance")
    else:
        print("➖ No change in performance")

    # Show sample comparisons
    print("\n--- Sample Comparisons ---")
    for i, r in enumerate(valid_results[:3]):
        print(f"\nSample {i + 1}:")
        print(f"Question: {r['question'][:100]}...")
        print(f"Finetuned ({r['finetuned_score']}): {r['finetuned_output'][:150]}...")
        print(
            f"Unfinetuned ({r['unfinetuned_score']}): {r['unfinetuned_output'][:150]}..."
        )


def save_results(results, output_dir: str = "results"):
    script_dir = Path(__file__).parent
    full_output_dir = script_dir / output_dir
    os.makedirs(full_output_dir, exist_ok=True)

    with open(full_output_dir / "gemma_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {full_output_dir}/gemma_eval_results.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Gemma3 model")

    # Model configuration
    parser.add_argument(
        "--model",
        default="gemma3_instruct_1b",
        help="Model preset to use (default: gemma3_instruct_1b)",
    )
    parser.add_argument(
        "--output-model",
        default="finetuned_gemma3_1b",
        help="Path to save finetuned model (default: finetuned_gemma3_1b)",
    )

    # Data paths
    parser.add_argument(
        "--train-data",
        default="data/500_documented_commits.json",
        help="Training data path (default: data/500_documented_commits.json)",
    )
    parser.add_argument(
        "--eval-data",
        default="data/so_jax_qa_pairs.json",
        help="Evaluation data path (default: data/so_jax_qa_pairs.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)",
    )

    # Training hyperparameters
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=384,
        help="Sequence length (default: 384)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=6e-4,
        help="Learning rate (default: 6e-4)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-3, help="Weight decay (default: 1e-3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of epochs (default: 8)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )

    # Inference parameters
    parser.add_argument(
        "--test-prompt",
        default="Fix self-attention bug",
        help="Test prompt for inference (default: Fix self-attention bug)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max length for pre-training inference (default: 256)",
    )
    parser.add_argument(
        "--max-length-post",
        type=int,
        default=1024,
        help="Max length for post-training inference (default: 1024)",
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Random seed for inference (default: 2)"
    )

    # Evaluation models
    parser.add_argument(
        "--eval-judge",
        default="gemini-2.5-flash-preview-05-20",
        help="Judge model for evaluation (default: gemini-2.5-flash-preview-05-20)",
    )
    parser.add_argument(
        "--eval-baseline",
        default="gemini-2.5-flash-preview-05-20",
        help="Baseline model for evaluation (default: gemini-2.5-flash-preview-05-20)",
    )

    # Flags
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only evaluate existing models",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation and only do training",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_environment()

    if not args.skip_training:
        print("=== Loading unfinetuned model ===")
        gemma_lm = load_model(args.model)

        print("\n=== Testing inference before fine-tuning ===")
        run_inference(
            gemma_lm, args.test_prompt, max_length=args.max_length, seed=args.seed
        )

        print("\n=== Loading training data ===")
        data = load_data(args.train_data)
        print(f"Loaded {len(data['prompts'])} training samples")

        print("\n=== Starting fine-tuning ===")
        fine_tune_model(
            model=gemma_lm,
            data=data,
            rank=args.rank,
            sequence_length=args.sequence_length,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        print("\n=== Testing inference after fine-tuning ===")
        run_inference(
            gemma_lm, args.test_prompt, max_length=args.max_length_post, seed=1
        )

        print(f"\n=== Saving finetuned model to {args.output_model} ===")
        output_path = (
            args.output_model
            if args.output_model.endswith(".weights.h5")
            else f"{args.output_model}.weights.h5"
        )
        gemma_lm.save_weights(output_path)

    if not args.skip_evaluation:
        print("\n=== Loading evaluation data ===")
        script_dir = Path(__file__).parent
        eval_data_path = script_dir / args.eval_data
        eval_data = LoadedDataset(
            eval_data_path,
            truncate_sample,
        )
        print(f"Loaded {len(eval_data)} evaluation samples")

        print("\n=== Evaluating models ===")
        results = evaluate_models(
            eval_data,
            args.model,
            args.output_model,
            args.eval_judge,
            args.eval_baseline,
            20,  # max_samples
        )

        compare_performance(results)

        save_results(results, args.output_dir)

    print("\n🎉 Complete! Finetuning and evaluation finished.")


if __name__ == "__main__":
    main()
