import json
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, TypedDict, overload

from google.generativeai.generative_models import GenerativeModel
from google.generativeai.models import list_models
from google.generativeai.types import GenerateContentResponse


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

    @overload
    def __getitem__(self, idx: int) -> T: ...

    @overload
    def __getitem__(self, idx: slice) -> "LoadedDataset[T]": ...

    def __getitem__(self, idx: int | slice) -> T | "LoadedDataset[T]":
        if isinstance(idx, int):
            return self._get_mod_fn(self.data[idx])
        elif isinstance(idx, slice):
            sliced_data = self.data[idx]
            return LoadedDataset(sliced_data, self._get_mod_fn)
        else:
            raise TypeError("Index must be an integer or a slice")

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


class SplitDataset(TypedDict):
    train: LoadedDataset
    test: LoadedDataset


def split_dataset(dataset: LoadedDataset, split: float = 0.9) -> SplitDataset:
    """
    Args:
        dataset: Combined `LoadedDataset` instance to split
        split: Float, determines how big the train set is

    Returns:
        `SplitDataset` object, TypedDict containing `train` and `test` fields.
    """
    raw_data = dataset.data

    train_data = raw_data[: int(split * len(dataset))]
    test_data = raw_data[int(split * len(dataset)) :]

    return {
        "train": LoadedDataset(train_data, dataset._get_mod_fn),
        "test": LoadedDataset(test_data, dataset._get_mod_fn),
    }


def load_finetuned_model(model_name: str) -> GenerativeModel:
    model = GenerativeModel(model_name=f"tunedModels/{model_name}")
    return model


def generate_from_model(
    model: GenerativeModel, question: str, is_str: bool = False
) -> GenerateContentResponse | str:
    """
    Samples from the model. `is_str` determines whether the output is a raw `str` or
    A `GenerateContentResponse` struct.
    """
    out = model.generate_content(question)

    if is_str:
        return out.text
    else:
        return out


def truncate_sample(
    sample: GeminiSample,
    input_limit: int = 40_000,
    output_limit: int = 5_000,
) -> GeminiSample:
    """
    Modifier Function that truncates a dataset sample produced at `__getitem__`
    As per the limitations of the Gemini API
    """
    margin = 50  # API counts characters differently
    truncated_input: str = sample["text_input"][: input_limit - margin]
    truncated_output: str = sample["output"][: output_limit - margin]

    return {
        "text_input": truncated_input,
        "output": truncated_output,
    }


def get_default_reference_model(logger: Logger) -> str:
    try:
        reference_model = [
            m
            for m in list_models()
            if "createTunedModel" in m.supported_generation_methods
            and "flash" in m.name
        ][0]
        return reference_model.name
    except (IndexError, Exception) as e:
        logger.warning(f"Failed to find default reference model: {str(e)}")
        return "gemini-1.5-flash"
