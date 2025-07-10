import json
import os
from typing import Dict, List

import keras
import keras_hub
from keras_hub.models import Gemma3CausalLM  # pyright: ignore[reportMissingImports]


def set_environment():
    """Sets the environment variables for Keras and XLA."""
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"


def load_model(preset: str) -> Gemma3CausalLM:
    """Loads the Gemma model from a preset."""
    model = Gemma3CausalLM.from_preset(preset)
    print(model.summary())
    return model


def run_inference(
    model: Gemma3CausalLM,
    instruction: str,
    max_length: int,
    seed: int,
    template: str = "Instruction:{instruction}\n\nResponse:{response}",
):
    """Runs inference on the model with a given instruction."""
    prompt = template.format(instruction=instruction, response="")
    sampler = keras_hub.samplers.TopKSampler(k=5, seed=seed)
    model.compile(sampler=sampler)
    print(model.generate(prompt, max_length=max_length))


def load_data(file_path: str) -> Dict[str, List[str]]:
    """Loads the training data from a JSON file."""
    prompts = []
    responses = []
    with open(file_path) as file:
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
    """Configures and runs the LoRA fine-tuning process."""
    # Configure LoRA tuning
    model.backbone.enable_lora(rank=rank)
    print(model.summary())

    # Configure fine-tuning settings
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

    # Run the fine-tune process
    model.fit(data, epochs=epochs, batch_size=batch_size)


def main():
    """Main function to run the fine-tuning process."""
    set_environment()
    gemma_lm = load_model("gemma3_instruct_1b")

    print("--- Running inference before fine-tuning ---")
    run_inference(gemma_lm, "Fix self-attention bug", max_length=256, seed=2)

    data = load_data("./data/500_documented_commits.json")

    print("\n--- Starting fine-tuning ---")
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

    print("\n--- Running inference after fine-tuning ---")
    run_inference(gemma_lm, "Fix self-attention bug", max_length=1024, seed=1)


if __name__ == "__main__":
    main()
