import json
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# --- Configuration ---
assert load_dotenv(), "Couldn't load envvars"
GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")

GITHUB_GRAPHQL_URL: str = "https://api.github.com/graphql"
REPO_OWNER: str = "google"
REPO_NAME: str = "jax"

NUM_ISSUES_TO_FETCH: int = 10
MAX_ISSUES_PER_PAGE: int = 100
NUM_COMMENTS_PER_ISSUE: int = 20

OUTPUT_FILENAME: str = "data/jax_issues_qa_pairs.json"


# --- GraphQL Query ---
ISSUES_QUERY: str = """
query GetIssuesAndComments(
    $owner: String!,
    $name: String!,
    $numIssuesPerPage: Int!,
    $numComments: Int!,
    $afterCursor: String
) {
  repository(owner: $owner, name: $name) {
    issues(
        first: $numIssuesPerPage,
        after: $afterCursor,
        orderBy: {field: CREATED_AT, direction: DESC},
        states: [OPEN, CLOSED]
    ) {
      nodes {
        id
        number
        title
        url
        author {
          login
        }
        bodyText
        comments(first: $numComments) { # Default order is chronological (ASC)
          nodes {
            author {
              login
            }
            bodyText
            createdAt
          }
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""


def run_graphql_query(
    query: str, variables: Dict[str, Any], token: str
) -> Dict[str, Any]:
    """
    Executes a GraphQL query against the GitHub API.
    (Function remains the same as before)
    """
    headers = {
        "Authorization": f"bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        GITHUB_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers=headers,
        timeout=60,  # Increased timeout for potentially larger cumulative requests
    )
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        raise ValueError(f"GraphQL API errors: {data['errors']}")
    return data


def process_issues_to_qa(
    issues_nodes: List[Optional[Dict[str, Any]]], owner: str, name: str
) -> List[Dict[str, Any]]:
    """
    Processes raw issue data from GraphQL into the desired QA pair format.
    (Function remains the same as before)
    """
    processed_data: List[Dict[str, Any]] = []

    for issue in issues_nodes:
        if not issue:
            continue

        issue_number: Optional[int] = issue.get("number")
        issue_title: str = issue.get("title", "N/A")
        issue_url: str = issue.get("url", "")

        if issue_number is None:
            print(f"Skipping issue with missing number: {issue.get('id')}")
            continue

        issue_author_login: str = (issue.get("author") or {}).get(
            "login", "unknown_user"
        )
        issue_body: str = issue.get("bodyText", "")

        conversation_pairs: List[Dict[str, str]] = []
        current_input_text: str = issue_body
        current_input_author: str = issue_author_login
        comments_data: List[Optional[Dict[str, Any]]] = (
            issue.get("comments", {}).get("nodes") or []
        )

        for comment in comments_data:
            if not comment:
                continue
            comment_author_login: str = (comment.get("author") or {}).get(
                "login", "unknown_user"
            )
            comment_body: str = comment.get("bodyText", "")
            if not comment_body.strip():
                continue
            qa_pair: Dict[str, str] = {
                "text_input": current_input_text,
                "output": comment_body,
                "from_id": current_input_author,
                "to_id": comment_author_login,
            }
            conversation_pairs.append(qa_pair)
            current_input_text = comment_body
            current_input_author = comment_author_login

        if conversation_pairs:
            processed_data.append({
                "issue_id": f"{owner}/{name}#{issue_number}",
                "issue_title": issue_title,
                "url": issue_url,
                "conversation": conversation_pairs,
            })
        else:
            print(
                f"No QA pairs generated for issue {owner}/{name}#{issue_number} (e.g., no comments or only empty comments)."
            )
    return processed_data


def main() -> None:
    """
    Main function to fetch, process, and save GitHub issues as QA pairs.
    Handles pagination for fetching issues.
    """
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable not set.")
        print("Please set your GitHub Personal Access Token as GITHUB_TOKEN.")
        return

    all_issues_nodes: List[Optional[Dict[str, Any]]] = []
    current_cursor: Optional[str] = None
    has_next_page: bool = True
    fetched_issues_count: int = 0

    print(
        f"Attempting to fetch up to {NUM_ISSUES_TO_FETCH} issues from {REPO_OWNER}/{REPO_NAME}..."
    )
    print(f"(Issues will be fetched in batches of up to {MAX_ISSUES_PER_PAGE})")

    while fetched_issues_count < NUM_ISSUES_TO_FETCH and has_next_page:
        num_to_fetch_this_run = min(
            MAX_ISSUES_PER_PAGE, NUM_ISSUES_TO_FETCH - fetched_issues_count
        )

        print(
            f"Fetching next batch of {num_to_fetch_this_run} issues (cursor: {current_cursor})..."
        )

        variables: Dict[str, Any] = {
            "owner": REPO_OWNER,
            "name": REPO_NAME,
            "numIssuesPerPage": num_to_fetch_this_run,
            "numComments": NUM_COMMENTS_PER_ISSUE,
            "afterCursor": current_cursor,
        }

        try:
            raw_data = run_graphql_query(ISSUES_QUERY, variables, GITHUB_TOKEN)

            repository_data = raw_data.get("data", {}).get("repository", {})
            if not repository_data:
                print("Error: 'repository' field missing in GraphQL response.")
                break

            issues_connection = repository_data.get("issues", {})
            if not issues_connection:
                print("Error: 'issues' connection missing in GraphQL response.")
                break

            new_nodes: List[Optional[Dict[str, Any]]] = issues_connection.get(
                "nodes", []
            )
            all_issues_nodes.extend(new_nodes)
            fetched_issues_count += len(new_nodes)

            page_info: Dict[str, Any] = issues_connection.get("pageInfo", {})
            current_cursor = page_info.get("endCursor")
            has_next_page = page_info.get("hasNextPage", False)

            print(
                f"Fetched {len(new_nodes)} issues in this batch. Total fetched so far: {fetched_issues_count}"
            )

            if not has_next_page and fetched_issues_count < NUM_ISSUES_TO_FETCH:
                print("No more issues to fetch from the repository.")
                break

            # Optional: Add a small delay to be kind to the API, though GraphQL cost analysis is usually preferred
            # time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from GitHub API: {e}")
            break
        except ValueError as e:  # This catches our custom GraphQL error
            print(f"GraphQL API Error: {e}")
            break
        except Exception as e:  # Catch any other unexpected errors during the loop
            print(f"An unexpected error occurred during fetching: {e}")
            break

    if not all_issues_nodes:
        print("No issues were fetched successfully.")
        return

    print(f"\nSuccessfully fetched a total of {len(all_issues_nodes)} issues.")
    print("Processing issues into QA pairs...")
    qa_data = process_issues_to_qa(all_issues_nodes, REPO_OWNER, REPO_NAME)

    if not qa_data:
        print(
            "No QA data was generated. Check if issues had comments or if processing failed."
        )
        return

    print(f"Processed {len(qa_data)} issues with QA pairs.")

    try:
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved QA pairs to {OUTPUT_FILENAME}")
    except IOError as e:
        print(f"Error writing data to file {OUTPUT_FILENAME}: {e}")


if __name__ == "__main__":
    main()
