import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, TypedDict, TypeVar

from dotenv import load_dotenv
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.models import create_tuned_model, get_tuned_model, list_models
from google.generativeai.types import GenerateContentResponse

# -- Config --
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"  # Disable MTLS

assert load_dotenv(), "Couldn't load envvars"
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
MODEL_NAME: None | str = "qa-commits-9543" # if already finetuned

# -- Setup --
K = TypeVar("K")
V = TypeVar("V")


class GeminiSample(TypedDict):
    """
    Actual sample that the Gemini API can consume.
    """

    text_input: str
    output: str


class IssuesSample(TypedDict):
    text_input: str
    output: str
    from_id: str
    to_id: str


class QAIssue(TypedDict):
    issue_id: str
    issue_title: str
    url: str
    conversation: list[IssuesSample]


IssuesDataset = list[QAIssue]


def truncate_sample(
    sample: GeminiSample,
    input_limit: int = 40_000,
    output_limit: int = 5_000,
) -> GeminiSample:
    """
    Modifier Function that truncates a dataset sample produced at `__getitem__`
    As per the limitations of the Gemini API
    """
    margin = 500  # API counts characters differently
    truncated_input: str = sample["text_input"][: input_limit - margin]
    truncated_output: str = sample["output"][: output_limit - margin]

    return {
        "text_input": truncated_input,
        "output": truncated_output,
    }


class LoadedDataset[T: Dict[str, Any]]:
    def __init__(
        self,
        data: Path | list[T],
        mod_fn: Callable[
            [T], T
        ] = lambda x: x,  # we assume the mod_fn returns the same shape T
    ) -> None:
        self._get_mod_fn = mod_fn

        if isinstance(data, Path):
            self.data: list[T] = json.loads(data.read_text())  # type: ignore
        else:
            self.data = data

    def extract_keys(
        self, keys: str | Iterable[str]
    ) -> "LoadedDataset[Dict[str, Any]]":
        key_list = [keys] if isinstance(keys, str) else list(keys)

        filtered: list[Dict[str, Any]] = [
            {k: item[k] for k in key_list if k in item} for item in self.data
        ]

        return LoadedDataset(filtered, mod_fn=self._get_mod_fn)  # type: ignore

    def flatten_on_key(
        self, key: str = "conversation"
    ) -> "LoadedDataset[Dict[str, Any]]":
        flattened: list[Dict[str, Any]] = []

        for item in self.data:
            if isinstance(item, dict) and key in item and isinstance(item[key], list):
                # we know item[key] is List[T], but we lose the original TypedDict at this point
                flattened.extend(item[key])
            else:
                flattened.append(item)

        return LoadedDataset(flattened, self._get_mod_fn)  # type: ignore

    @staticmethod
    def _filter_dict(keys: list[str], item: dict[str, Any]) -> dict[str, Any]:
        return {k: item[k] for k in keys if k in item}

    def get_keys(self, idx: int):
        return self.__getitem__(idx).keys()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__len__()} items)"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self._get_mod_fn(self.data[idx])

    def __add__(self, other: "LoadedDataset[T]") -> "LoadedDataset[T]":
        assert isinstance(other, LoadedDataset), (
            "Can only add LoadedDataset objects to another LoadedDataset object."
        )

        if isinstance(other.data, Iterable):
            combined = self.data + other.data
        else:
            raise TypeError(
                f"Expected `other.data` to be an iterable. Found: {type(other.data)}"
            )

        return LoadedDataset(combined, self._get_mod_fn)


def finetune_model(data: LoadedDataset, name: str):
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

    print("Waiting for finetuning to complete...")

    while not operation.done():
        time.sleep(20)

    return operation, model


def load_finetuned_model(model_name: str, question: str) -> GenerateContentResponse:
    model = GenerativeModel(model_name=f"tunedModels/{model_name}")
    return model.generate_content(question)


def main():
    model_name = f"qa-commits-{random.randint(0, 10000)}"

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

    combined_dataset = qa_dataset + commits_dataset
    # operation, _ = finetune_model(combined_dataset, model_name)

    finetuned_model_name = MODEL_NAME if MODEL_NAME else model_name

    output = load_finetuned_model(
        finetuned_model_name, "Can you do a deep dive into how XLA works?"
    )

    print(f"Model output: {output.text}")


if __name__ == "__main__":
    main()
