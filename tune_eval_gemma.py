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
    os.environ["KERAS_BACKEND"] = "tensorflow"
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
    model.compile(sampler=sampler)  # pyright: ignore[reportArgumentType]
    output = model.generate(prompt, max_length=max_length)
    print(f"Inference output: {output}")
    return output


def load_data(file_path: str) -> Dict[str, List[str]]:
    prompts = []
    responses = []
    
    # Make path relative to script location
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

    model.preprocessor.sequence_length = sequence_length  # pyright: ignore[reportOptionalMemberAccess]
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
    eval_data: LoadedDataset, unfinetuned_model_name: str, finetuned_model_name: str
):
    print("\n--- Evaluating unfinetuned model ---")
    unfinetuned_results = eval_model(
        eval_data,
        unfinetuned_model_name,
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-preview-05-20",
    )

    print("\n--- Evaluating finetuned model ---")
    finetuned_results = eval_model(
        eval_data,
        finetuned_model_name,
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-preview-05-20",
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

        print(f"\n--- Performance Comparison ---")
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
    # Make output dir relative to script location  
    script_dir = Path(__file__).parent
    full_output_dir = script_dir / output_dir
    os.makedirs(full_output_dir, exist_ok=True)

    with open(full_output_dir / "unfinetuned_eval_results.json", "w") as f:
        json.dump(unfinetuned_results, f, indent=2)

    with open(full_output_dir / "finetuned_eval_results.json", "w") as f:
        json.dump(finetuned_results, f, indent=2)

    print(f"Results saved to {full_output_dir}/")


def main():
    set_environment()

    print("=== Loading unfinetuned model ===")
    gemma_lm = load_model("gemma3_instruct_1b")

    print("\n=== Testing inference before fine-tuning ===")
    run_inference(gemma_lm, "Fix self-attention bug", max_length=256, seed=2)

    print("\n=== Loading training data ===")
    data = load_data("data/500_documented_commits.json")
    print(f"Loaded {len(data['prompts'])} training samples")

    print("\n=== Starting fine-tuning ===")
    fine_tune_model(
        model=gemma_lm,
        data=data,
        rank=16,
        sequence_length=384,
        learning_rate=6e-4,
        weight_decay=1e-3,
        epochs=8,
        batch_size=4,
    )

    print("\n=== Testing inference after fine-tuning ===")
    run_inference(gemma_lm, "Fix self-attention bug", max_length=1024, seed=1)

    finetuned_model_path = "finetuned_gemma3_1b"
    print(f"\n=== Saving finetuned model to {finetuned_model_path} ===")
    gemma_lm.save_weights(finetuned_model_path)

    print("\n=== Loading evaluation data ===")
    script_dir = Path(__file__).parent
    eval_data_path = script_dir / "data/so_jax_qa_pairs.json"
    eval_data = LoadedDataset(
        eval_data_path,
        truncate_sample,
    )
    print(f"Loaded {len(eval_data)} evaluation samples")

    print("\n=== Evaluating models ===")
    unfinetuned_results, finetuned_results = evaluate_models(
        eval_data,
        "gemma3_instruct_1b",
        finetuned_model_path,
    )

    compare_performance(unfinetuned_results, finetuned_results)

    save_results(unfinetuned_results, finetuned_results)

    print("\n🎉 Complete! Finetuning and evaluation finished.")


if __name__ == "__main__":
    main()
