import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# --- Configuration ---
assert load_dotenv(), "Couldn't load envvars"

STACK_API_URL: str = "https://api.stackexchange.com/2.3"
TAG: str = "jax"
MIN_VOTES: int = 0
MAX_QUESTIONS_TO_FETCH: int = 50  # Approx number of questions to fetch
QUESTIONS_PER_PAGE: int = 25
REQUEST_DELAY_SECONDS: float = 0.5  # Delay between API calls to be polite

OUTPUT_DIR: str = "data"
OUTPUT_FILENAME: str = os.path.join(OUTPUT_DIR, f"so_{TAG.lower()}_qa_pairs.json")


def make_stack_api_request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes a request to the Stack Exchange API.

    Args:
        endpoint: The API endpoint (e.g., "/questions").
        params: A dictionary of parameters for the API call.

    Returns:
        The JSON response from the API.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        ValueError: If the API returns an error.
    """
    url: str = f"{STACK_API_URL}{endpoint}"

    params["site"] = "stackoverflow"

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
    except requests.exceptions.Timeout:
        print(f"Timeout occurred while requesting {url} with params {params}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Request to {url} failed: {e}")
        print(
            f"Response content: {response.content if 'response' in locals() else 'No response object'}"
        )
        raise

    data: Dict[str, Any] = response.json()

    if data.get("error_id"):
        raise ValueError(
            f"Stack API error {data.get('error_id')}: {data.get('error_name')} - {data.get('error_message')}"
        )

    if "quota_remaining" in data and data["quota_remaining"] < 10:  # Be conservative
        print(f"Warning: Low API quota remaining: {data['quota_remaining']}")
    if data.get("backoff"):
        backoff_seconds = data["backoff"]
        print(f"API requested backoff for {backoff_seconds} seconds. Waiting...")
        time.sleep(backoff_seconds + 1)  # Add a small buffer

    return data


def get_questions(tag: str, num_questions: int, page_size: int) -> List[Dict[str, Any]]:
    """
    Fetches questions from Stack Overflow with a specific tag.

    Args:
        tag: The tag to filter questions by (e.g., "jax").
        num_questions: The approximate total number of questions to fetch.
        page_size: Number of questions to fetch per API call.

    Returns:
        A list of question objects from the API.
    """
    all_questions: List[Dict[str, Any]] = []
    page: int = 1
    fetched_count: int = 0

    print(f"Fetching questions tagged '{tag}'...")
    with tqdm(total=num_questions, unit="question") as pbar:
        while fetched_count < num_questions:
            params: Dict[str, Any] = {
                "page": page,
                "pagesize": min(
                    page_size, num_questions - fetched_count
                ),  # Adjust last page size
                "order": "desc",
                "sort": "votes",  # Sort by votes to get potentially higher quality questions first
                "tagged": tag,
                "filter": "withbody",  # Includes question body
            }
            try:
                data: Dict[str, Any] = make_stack_api_request("/questions", params)
                questions_on_page: List[Dict[str, Any]] = data.get("items", [])
                all_questions.extend(questions_on_page)

                newly_fetched = len(questions_on_page)
                fetched_count += newly_fetched
                pbar.update(newly_fetched)

                if not data.get("has_more") or not questions_on_page:
                    print("No more questions found or API limit reached.")
                    break
                page += 1
                time.sleep(REQUEST_DELAY_SECONDS)  # Polite delay
            except (requests.exceptions.RequestException, ValueError) as e:
                print(f"Error fetching questions on page {page}: {e}")
                # Decide if you want to retry or break. For simplicity, we break here.
                break
            if fetched_count >= num_questions:
                break

    # Ensure we don't exceed the requested number significantly due to page sizes
    return all_questions[:num_questions]


def get_answers_for_question(question_id: int) -> List[Dict[str, Any]]:
    """
    Fetches all answers for a given Stack Overflow question ID.

    Args:
        question_id: The ID of the question.

    Returns:
        A list of answer objects from the API.
    """
    all_answers: List[Dict[str, Any]] = []
    page: int = 1

    print(f"Fetching answers for question ID: {question_id}...")
    while True:
        params: Dict[str, Any] = {
            "page": page,
            "pagesize": 100,  # Max page size for answers
            "order": "desc",
            "sort": "votes",
            "filter": "withbody",  # Includes answer body
        }
        try:
            data: Dict[str, Any] = make_stack_api_request(
                f"/questions/{question_id}/answers", params
            )
            answers_on_page: List[Dict[str, Any]] = data.get("items", [])
            all_answers.extend(answers_on_page)

            if not data.get("has_more") or not answers_on_page:
                break
            page += 1
            time.sleep(REQUEST_DELAY_SECONDS)  # Polite delay
        except (requests.exceptions.RequestException, ValueError) as e:
            print(
                f"Error fetching answers for question {question_id} on page {page}: {e}"
            )
            break  # Stop trying for this question if an error occurs

    return all_answers


def create_qa_records(
    question: Dict[str, Any], answers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Creates serialized Q&A records from a question and its answers.

    Args:
        question: The question object from the API.
        answers: A list of answer objects for the question.

    Returns:
        A list of Q&A records in the desired format.
    """
    qa_records: List[Dict[str, Any]] = []
    question_votes: int = question.get("score", 0)
    question_owner_id: Optional[str] = str(
        question.get("owner", {}).get("user_id", "N/A")
    )
    question_body_html: str = question.get("body", "")
    question_id: int = question.get("question_id", -1)
    question_title: str = question.get("title", "N/A")
    question_link: str = question.get("link", "")

    if question_votes < MIN_VOTES:
        return []  # Skip if question itself doesn't meet vote criteria

    for answer in answers:
        answer_votes: int = answer.get("score", 0)
        if answer_votes < MIN_VOTES:
            continue  # Skip if answer doesn't meet vote criteria

        answer_owner_id: Optional[str] = str(
            answer.get("owner", {}).get("user_id", "N/A")
        )
        answer_body_html: str = answer.get("body", "")
        answer_id: int = answer.get("answer_id", -1)
        answer_link: str = (
            f"{question_link}/{answer_id}#{answer_id}"  # Construct answer link
        )

        record: Dict[str, Any] = {
            "text_input": question_body_html,
            "output": answer_body_html,
            "from_id": question_owner_id,
            "to_id": answer_owner_id,
            "answer_votes": answer_votes,
            "question_votes": question_votes,
            "question_id": question_id,
            "answer_id": answer_id,
            "question_title": question_title,
            "question_link": question_link,
            "answer_link": answer_link,
        }
        qa_records.append(record)
    return qa_records


def main() -> None:
    """
    Main function to orchestrate fetching Q&A pairs and saving them.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    questions: List[Dict[str, Any]] = get_questions(
        TAG, MAX_QUESTIONS_TO_FETCH, QUESTIONS_PER_PAGE
    )

    all_qa_data: List[Dict[str, Any]] = []

    if not questions:
        print(f"No questions found for tag '{TAG}'. Exiting.")
        return

    print(f"\nProcessing {len(questions)} questions to find Q&A pairs...")
    for question in tqdm(questions, unit="question processed"):
        question_id: Optional[int] = question.get("question_id")
        if question_id is None:
            print(f"Skipping question with missing ID: {question.get('title')}")
            continue

        answers: List[Dict[str, Any]] = get_answers_for_question(question_id)
        if answers:
            qa_records: List[Dict[str, Any]] = create_qa_records(question, answers)
            all_qa_data.extend(qa_records)
        time.sleep(
            REQUEST_DELAY_SECONDS
        )  # Delay even if no answers, for the question fetch itself

    print(f"\nCollected {len(all_qa_data)} Q&A pairs.")

    if all_qa_data:
        try:
            with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
                json.dump(all_qa_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved Q&A pairs to: {OUTPUT_FILENAME}")
        except IOError as e:
            print(f"Error saving data to {OUTPUT_FILENAME}: {e}")
    else:
        print("No Q&A pairs collected that meet the criteria.")


if __name__ == "__main__":
    main()
