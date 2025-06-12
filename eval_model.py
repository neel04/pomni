import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    cast,
)

from dotenv import load_dotenv
from google.generativeai.generative_models import GenerativeModel

from utils import (
    LoadedDataset,
    generate_from_model,
    get_default_reference_model,
    load_finetuned_model,
    truncate_sample,
)

# -- Config --
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"  # Disable MTLS
assert load_dotenv(), "Couldn't load envvars"

EVAL_SAMPLES = 75
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvalResults(TypedDict):
    question: str
    tuned_model_output: str
    ref_model_output: str
    tuned_model_score: Optional[int]
    ref_model_score: Optional[int]
    judge_response: Optional[str]


class RequestTask:
    """A task to be executed with the model."""

    def __init__(
        self, index: int, func: Callable[..., Awaitable[str]], *args, **kwargs
    ):
        self.index = index
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result: Optional[str] = None
        self.attempts = 0
        self.max_attempts = 5
        self.error: Optional[Exception] = None

    async def execute(self) -> bool:
        """Execute the task and return True if successful, False otherwise."""
        self.attempts += 1
        try:
            self.result = await self.func(*self.args, **self.kwargs)
            return True
        except Exception as e:
            self.error = e
            return False

    @property
    def is_completed(self) -> bool:
        return self.result is not None

    @property
    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts and not self.is_completed


class RateLimitedQueue:
    """Queue that respects rate limits for API calls."""

    def __init__(self, name: str, requests_per_minute: int = 10):
        self.name = name
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60 / self.requests_per_minute  # seconds between requests
        self.last_request_time = 0
        self.queue: List[RequestTask] = []
        self.results: Dict[int, str] = {}

    def add_request(
        self, index: int, func: Callable[..., Awaitable[str]], *args, **kwargs
    ) -> None:
        """Add a request to the queue."""
        task = RequestTask(index, func, *args, **kwargs)
        self.queue.append(task)

    async def process_queue(self) -> Dict[int, str]:
        """Process all requests in the queue respecting rate limits."""
        pending_tasks = self.queue.copy()
        total_tasks = len(pending_tasks)
        completed_tasks = 0

        try:
            while pending_tasks:
                # Wait until we can make the next request
                now = time.time()
                time_since_last_request = now - self.last_request_time
                if time_since_last_request < self.min_interval:
                    await asyncio.sleep(self.min_interval - time_since_last_request)

                # Get the next task
                task = pending_tasks.pop(0)

                # Execute the task
                success = await task.execute()
                self.last_request_time = time.time()

                if success:
                    assert task.result is not None
                    self.results[task.index] = task.result
                    completed_tasks += 1
                    logger.info(
                        f"[{self.name}] Completed task {task.index + 1}/{total_tasks} ({completed_tasks}/{total_tasks})"
                    )
                elif task.can_retry:
                    # If task failed but can be retried, add it back to the end of the queue
                    pending_tasks.append(task)
                    logger.warning(
                        f"[{self.name}] Failed task {task.index + 1}/{total_tasks}, "
                        f"attempt {task.attempts}/{task.max_attempts}. Retrying..."
                    )
                    # Add a delay before retrying
                    await asyncio.sleep(2)
                else:
                    # Task failed and can't be retried
                    error_msg = (
                        f"Failed after {task.attempts} attempts: {str(task.error)}"
                    )
                    self.results[task.index] = f"Error: {error_msg}"
                    completed_tasks += 1
                    logger.error(
                        f"[{self.name}] Task {task.index + 1}/{total_tasks} failed permanently: {error_msg}"
                    )

        except KeyboardInterrupt:
            logger.warning(
                f"[{self.name}] Process interrupted by user. Completed {completed_tasks}/{total_tasks} tasks."
            )
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error: {str(e)}")
            raise

        return self.results


async def generate_from_model_async(model: GenerativeModel, prompt: str) -> str:
    return await asyncio.to_thread(generate_from_model, model, prompt, True)  # type: ignore


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


def extract_scores(judge_response: str) -> Tuple[Optional[int], Optional[int]]:
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

    # If the pattern doesn't match or there's an error parsing, log and return None
    logger.warning(
        f"Failed to extract scores from judge response: {judge_response[:100]}..."
    )
    return None, None


async def eval_model_async(
    eval_data: LoadedDataset,
    model_name: str,
    reference_model_name: str,
    judge_model_name: str,
) -> List[EvalResults]:
    """
    Asynchronously evaluate a fine-tuned model against a reference model.

    Args:
        eval_data: Dataset to evaluate on
        model_name: Name of the fine-tuned model
        reference_model_name: Name of the reference model (optional)
        judge_model_name: Name of the judge model (optional)

    Returns:
        List of evaluation results containing questions, model outputs, and scores
    """
    eval_data = eval_data[:EVAL_SAMPLES]  # truncate the dataset

    logger.info(
        f"Loading models: reference model ({reference_model_name}), fine-tuned model ({model_name}), and judge model ({judge_model_name})"
    )
    superior_model = GenerativeModel(reference_model_name)
    tuned_model = load_finetuned_model(model_name)
    judge_model = GenerativeModel(judge_model_name)

    superior_queue = RateLimitedQueue("Superior", requests_per_minute=10)
    tuned_queue = RateLimitedQueue("Tuned", requests_per_minute=60)

    logger.info(f"Preparing evaluation for {len(eval_data)} samples")

    for i, sample in enumerate(eval_data):  # type: ignore
        input_str = sample["text_input"]
        superior_queue.add_request(
            i, generate_from_model_async, superior_model, input_str
        )
        tuned_queue.add_request(i, generate_from_model_async, tuned_model, input_str)

    # Process queues concurrently
    logger.info("Starting concurrent processing of model queues")
    superior_results_task = asyncio.create_task(superior_queue.process_queue())
    tuned_results_task = asyncio.create_task(tuned_queue.process_queue())

    try:
        # Wait for both queues to be processed
        superior_results = await superior_results_task
        tuned_results = await tuned_results_task

        # Now queue up judge evaluations
        logger.info("Preparing judge model evaluations")
        judge_queue = RateLimitedQueue(
            "Judge", requests_per_minute=10
        )  # Rate limited to 10 RPM

        for i, sample in enumerate(eval_data):
            question = sample["text_input"]
            tuned_output = tuned_results.get(i, "Error: Failed to generate output")
            ref_output = superior_results.get(i, "Error: Failed to generate output")

            # Create eval template for the judge
            judge_prompt = generate_eval_template(question, tuned_output, ref_output)
            judge_queue.add_request(
                i, generate_from_model_async, judge_model, judge_prompt
            )

        # Process judge evaluations
        logger.info("Processing judge evaluations")
        judge_results = await judge_queue.process_queue()

        # Combine results and extract scores
        logger.info("Combining results and extracting scores")
        eval_outputs: list[EvalResults] = []
        all_scores: List[Tuple[int, int]] = []

        for i, sample in enumerate(eval_data):  # type: ignore
            question = sample["text_input"]
            tuned_output = tuned_results.get(i, "Error: Failed to generate output")
            ref_output = superior_results.get(i, "Error: Failed to generate output")
            judge_response = judge_results.get(
                i, "Error: Failed to generate judge response"
            )

            # Extract scores from judge response
            tuned_score, ref_score = extract_scores(judge_response)

            # Only add valid scores to the score list
            if tuned_score is not None and ref_score is not None:
                all_scores.append((tuned_score, ref_score))

            eval_infer = {
                "question": question,
                "tuned_model_output": tuned_output,
                "ref_model_output": ref_output,
                "tuned_model_score": tuned_score,
                "ref_model_score": ref_score,
                "judge_response": judge_response,
            }
            eval_outputs.append(cast(EvalResults, eval_infer))

        # Print the first 10 scores
        if all_scores:
            logger.info(f"First 10 scores (tuned, reference): {all_scores[:10]}")
            # Calculate average scores
            avg_tuned = sum(s[0] for s in all_scores) / len(all_scores)
            avg_ref = sum(s[1] for s in all_scores) / len(all_scores)
            logger.info(
                f"Average scores - Tuned model: {avg_tuned:.2f}, Reference model: {avg_ref:.2f}"
            )

            # Print win/loss/tie statistics
            wins = sum(1 for t, r in all_scores if t > r)
            losses = sum(1 for t, r in all_scores if t < r)
            ties = sum(1 for t, r in all_scores if t == r)
            logger.info(
                f"Tuned model: {wins} wins, {losses} losses, {ties} ties against reference model"
            )
        else:
            logger.warning("No valid scores were extracted")

        logger.info(f"Evaluation complete with {len(eval_outputs)} results")
        return eval_outputs

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        # Cancel any pending tasks
        superior_results_task.cancel()
        tuned_results_task.cancel()
        raise
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        # Cancel any pending tasks
        superior_results_task.cancel()
        tuned_results_task.cancel()
        raise


def eval_model(
    eval_data: LoadedDataset,
    model_name: str,
    reference_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
) -> List[EvalResults]:
    """
    Synchronous wrapper for the asynchronous eval_model_async function.
    Evaluates a fine-tuned model against a reference model on the given dataset.

    Args:
        eval_data: Dataset to evaluate on
        model_name: Name of the fine-tuned model
        reference_model_name: Name of the reference model (optional)
        judge_model_name: Name of the judge model (optional)

    Returns:
        List of evaluation results containing questions, model outputs, and scores
    """
    try:
        return asyncio.run(
            eval_model_async(
                eval_data, model_name, reference_model_name, judge_model_name
            )
        )
    except KeyboardInterrupt:
        logger.warning("Evaluation cancelled by user")
        return []


def run_async_eval(
    logger: logging.Logger, eval_data: LoadedDataset, finetuned_model_name: str
):
    reference_model_name = get_default_reference_model(logger)

    eval_results = eval_model(
        eval_data,
        finetuned_model_name,
        reference_model_name,
        "gemini-2.5-flash-preview-05-20",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/eval_results_{timestamp}.json"

    # Convert to serializable format
    serializable_results = []

    for result in eval_results:
        serializable_result = {
            "question": result["question"],
            "tuned_model_output": result["tuned_model_output"],
            "ref_model_output": result["ref_model_output"],
            "tuned_model_score": result["tuned_model_score"],
            "ref_model_score": result["ref_model_score"],
        }
        serializable_results.append(serializable_result)

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    commits_dataset = LoadedDataset(
        Path("data/so_jax_qa_pairs.json"),
        truncate_sample,  # type: ignore
    )
