import logging
import os
import random
import time
from pathlib import Path
from typing import Optional, TypeVar

from dotenv import load_dotenv
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.models import create_tuned_model, get_tuned_model, list_models

from eval_model import run_async_eval
from utils import (
    LoadedDataset,
    generate_from_model,
    load_finetuned_model,
    split_dataset,
    truncate_sample,
)

# -- Config --
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"  # Disable MTLS

assert load_dotenv(), "Couldn't load envvars"
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
MODEL_NAME: None | str = None  # if already finetuned

# -- Setup --
K = TypeVar("K")
V = TypeVar("V")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def finetune_model(data: LoadedDataset, name: str) -> GenerativeModel:
    base_model = [
        m
        for m in list_models()
        if "createTunedModel" in m.supported_generation_methods and "flash" in m.name
    ][0]

    operation = create_tuned_model(
        source_model=base_model.name,
        training_data=data,
        id=name,
        epoch_count=1,
        batch_size=64,
        learning_rate=0.001,
    )

    model = get_tuned_model(f"tunedModels/{name}")

    print(f"Waiting for finetuning to complete... Model Name: {name}\n")

    while not operation.done():
        time.sleep(10)

    return model


def main():
    model_name = f"finetuned-{random.randint(0, 10000)}"

    qa_dataset = (
        LoadedDataset(Path("data/jax_issues_qa_pairs.json"), truncate_sample)  # type: ignore
        .extract_keys("conversation")
        .flatten_on_key("conversation")
        .extract_keys(["text_input", "output"])
    )

    commits_dataset = LoadedDataset(
        Path("data/500_documented_commits.json"),
        truncate_sample,  # type: ignore
    )

    so_dataset = LoadedDataset(
        Path("data/so_jax_qa_pairs.json"),
        truncate_sample,  # type: ignore
    ).extract_keys(["text_input", "output"])

    # combined_dataset = split_dataset(qa_dataset + commits_dataset + so_dataset)
    combined_dataset = split_dataset(so_dataset)
    train_data, eval_data = combined_dataset["train"], combined_dataset["test"]

    if MODEL_NAME:
        finetuned_model = load_finetuned_model(MODEL_NAME)
        finetuned_model_name = MODEL_NAME
    else:
        finetuned_model = finetune_model(train_data, model_name)
        finetuned_model_name = model_name

    output = generate_from_model(
        finetuned_model,
        "<p>What is a ConcretizationTypeError in JAX?</p>",
        is_str=True,
    )

    print(f"Model output: {output}")

    print("\nRunning evals...")

    run_async_eval(logger, eval_data, finetuned_model_name)


if __name__ == "__main__":
    main()
