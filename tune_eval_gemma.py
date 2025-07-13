import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import keras
import keras_hub
from keras_hub.models import Gemma3CausalLM

from eval_model import eval_model
from utils import LoadedDataset, truncate_sample


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
    sampler = keras_hub.samplers.TopKSampler(k=5, seed=seed)
    model.compile(sampler=sampler)
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


def evaluate_models(
    eval_data: LoadedDataset,
    unfinetuned_model_name: str,
    finetuned_model_name: str,
    judge_model: str,
    baseline_model: str,
):
    print("\n--- Evaluating unfinetuned model ---")
    unfinetuned_results = eval_model(
        eval_data,
        unfinetuned_model_name,
        judge_model,
        baseline_model,
    )

    print("\n--- Evaluating finetuned model ---")
    finetuned_results = eval_model(
        eval_data,
        finetuned_model_name,
        judge_model,
        baseline_model,
    )

    return unfinetuned_results, finetuned_results


def compare_performance(unfinetuned_results, finetuned_results):
    unfinetuned_scores = [
        r["tuned_model_score"]
        for r in unfinetuned_results
        if r["tuned_model_score"] is not None
    ]
    finetuned_scores = [
        r["tuned_model_score"]
        for r in finetuned_results
        if r["tuned_model_score"] is not None
    ]

    if unfinetuned_scores and finetuned_scores:
        avg_unfinetuned = sum(unfinetuned_scores) / len(unfinetuned_scores)
        avg_finetuned = sum(finetuned_scores) / len(finetuned_scores)

        print("\n--- Performance Comparison ---")
        print(f"Unfinetuned model average score: {avg_unfinetuned:.2f}")
        print(f"Finetuned model average score: {avg_finetuned:.2f}")
        print(f"Improvement: {avg_finetuned - avg_unfinetuned:.2f}")

        if avg_finetuned > avg_unfinetuned:
            print("✅ Finetuning improved performance!")
        elif avg_finetuned < avg_unfinetuned:
            print("❌ Finetuning decreased performance")
        else:
            print("➖ No change in performance")
    else:
        print("❌ Could not compare performance - no valid scores extracted")


def save_results(unfinetuned_results, finetuned_results, output_dir: str = "results"):
    script_dir = Path(__file__).parent
    full_output_dir = script_dir / output_dir
    os.makedirs(full_output_dir, exist_ok=True)

    with open(full_output_dir / "unfinetuned_eval_results.json", "w") as f:
        json.dump(unfinetuned_results, f, indent=2)

    with open(full_output_dir / "finetuned_eval_results.json", "w") as f:
        json.dump(finetuned_results, f, indent=2)

    print(f"Results saved to {full_output_dir}/")


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
        output_path = args.output_model if args.output_model.endswith('.weights.h5') else f"{args.output_model}.weights.h5"
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
        unfinetuned_results, finetuned_results = evaluate_models(
            eval_data,
            args.model,
            args.output_model,
            args.eval_judge,
            args.eval_baseline,
        )

        compare_performance(unfinetuned_results, finetuned_results)

        save_results(unfinetuned_results, finetuned_results, args.output_dir)

    print("\n🎉 Complete! Finetuning and evaluation finished.")


if __name__ == "__main__":
    main()
