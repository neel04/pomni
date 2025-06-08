import json
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# --- Configuration ---
assert load_dotenv(), "Couldn't load envvars"
GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")

GITHUB_GRAPHQL_URL: str = "https://api.github.com/graphql"
REPO_OWNER: str = "google"
REPO_NAME: str = "jax"

MAX_DIFF_CHANGES: int = 500  # Maximum additions/deletions for "good" commits
MIN_COMMIT_MESSAGE_LENGTH: int = 10  # Minimum commit message length
NUM_COMMITS_TO_FETCH: int = 500
MAX_COMMITS_PER_PAGE: int = 50

OUTPUT_FILENAME: str = f"data/{NUM_COMMITS_TO_FETCH}_documented_commits.json"

# --- GraphQL Queries ---
COMMITS_QUERY: str = """
query GetCommits(
    $owner: String!,
    $name: String!,
    $numCommitsPerPage: Int!,
    $afterCursor: String
) {
  repository(owner: $owner, name: $name) {
    defaultBranchRef {
      target {
        ... on Commit {
          history(first: $numCommitsPerPage, after: $afterCursor) {
            nodes {
              oid
              message
              messageHeadline
              messageBody
              committedDate
              url
              additions
              deletions
              changedFiles
              author {
                user {
                  login
                }
                email
                name
              }
              committer {
                user {
                  login
                }
                email
                name
              }
            }
            pageInfo {
              endCursor
              hasNextPage
            }
          }
        }
      }
    }
  }
}
"""

COMMIT_DIFF_QUERY: str = """
query GetCommitDiff($owner: String!, $name: String!, $oid: String!) {
  repository(owner: $owner, name: $name) {
    object(oid: $oid) {
      ... on Commit {
        oid
        url
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

    Args:
        query: The GraphQL query string
        variables: Variables for the GraphQL query
        token: GitHub authentication token

    Returns:
        The JSON response from the GraphQL API

    Raises:
        ValueError: If GraphQL API returns errors
        requests.exceptions.RequestException: If HTTP request fails
    """
    headers = {
        "Authorization": f"bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        GITHUB_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        raise ValueError(f"GraphQL API errors: {data['errors']}")
    return data


def get_commit_diff_via_rest_api(
    owner: str, repo: str, sha: str, token: str
) -> Optional[str]:
    """
    Fetches the diff for a specific commit using GitHub's REST API.

    Args:
        owner: Repository owner
        repo: Repository name
        sha: Commit SHA
        token: GitHub authentication token

    Returns:
        The diff string in unified format, or None if failed
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.diff",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching diff for commit {sha}: {e}")
        return None


def is_well_documented_commit(commit: Dict[str, Any]) -> bool:
    """
    Determines if a commit is well-documented based on our heuristics.

    Args:
        commit: Commit data from GraphQL

    Returns:
        True if the commit meets our documentation criteria
    """
    message = commit.get("message", "")
    additions = commit.get("additions", 0)
    deletions = commit.get("deletions", 0)

    # Check message length
    if len(message.strip()) < MIN_COMMIT_MESSAGE_LENGTH:
        return False

    # Check diff size
    total_changes = additions + deletions
    if total_changes > MAX_DIFF_CHANGES * 2:  # Total of additions + deletions
        return False

    # Additional check: ensure changes aren't too minimal
    if total_changes < 5:  # Skip trivial changes
        return False

    return True


def extract_author_info(commit: Dict[str, Any]) -> List[str]:
    """
    Extracts author information from commit data.

    Args:
        commit: Commit data from GraphQL

    Returns:
        List of author identifiers (GitHub logins when available, emails otherwise)
    """
    authors = []

    # Check commit author
    author = commit.get("author", {})
    if author:
        author_user = author.get("user")
        if author_user and author_user.get("login"):
            authors.append(author_user["login"])
        elif author.get("email"):
            authors.append(author["email"])

    # Check committer (if different from author)
    committer = commit.get("committer", {})
    if committer:
        committer_user = committer.get("user")
        if committer_user and committer_user.get("login"):
            committer_login = committer_user["login"]
            # Only add if different from author
            if committer_login not in authors:
                authors.append(committer_login)
        elif committer.get("email"):
            committer_email = committer["email"]
            if committer_email not in authors:
                authors.append(committer_email)

    return authors if authors else ["unknown"]


def format_commit_diff(diff_text: str, commit: Dict[str, Any]) -> str:
    """
    Formats the commit diff with metadata and proper diff formatting.

    Args:
        diff_text: Raw diff text from GitHub API
        commit: Commit metadata

    Returns:
        List containing the formatted diff with metadata
    """
    if not diff_text:
        return "No diff available"

    # Add metadata header
    metadata = [
        f"Commit: {commit.get('oid', 'unknown')}",
        f"Date: {commit.get('committedDate', 'unknown')}",
        f"URL: {commit.get('url', 'unknown')}",
        f"Files changed: {commit.get('changedFiles', 0)}",
        f"Additions: +{commit.get('additions', 0)}, Deletions: -{commit.get('deletions', 0)}",
        "",
    ]

    # Format the diff with proper markdown
    formatted_diff = "```diff\n" + "\n".join(metadata) + diff_text + "\n```"

    return formatted_diff


def process_commits_to_training_data(
    commits_nodes: List[Optional[Dict[str, Any]]], owner: str, name: str, token: str
) -> List[Dict[str, Any]]:
    """
    Processes raw commit data from GraphQL into the desired training format.

    Args:
        commits_nodes: List of commit data from GraphQL
        owner: Repository owner
        name: Repository name
        token: GitHub authentication token

    Returns:
        List of training examples in the specified format
    """
    training_data: List[Dict[str, Any]] = []

    for commit in tqdm(commits_nodes):
        if not commit:
            continue

        # Check if commit meets our documentation criteria
        if not is_well_documented_commit(commit):
            continue

        commit_sha = commit.get("oid")
        if not commit_sha:
            continue

        print(f"Processing commit {commit_sha[:8]}...")

        # Get the diff for this commit
        diff_text = get_commit_diff_via_rest_api(owner, name, commit_sha, token)
        if not diff_text:
            print(f"Skipping commit {commit_sha[:8]} - could not fetch diff")
            continue

        # Extract commit message and author info
        commit_message = commit.get("message", "").strip()
        authors = extract_author_info(commit)

        # Format the diff output
        formatted_diff = format_commit_diff(diff_text, commit)

        # Create training example
        training_example = {
            "text_input": commit_message,
            "output": formatted_diff,
            "from_id": authors,
        }

        training_data.append(training_example)
        print(f"Added commit {commit_sha[:8]} to training data")

    return training_data


def main() -> None:
    """
    Main function to fetch, process, and save well-documented commits as training data.
    Handles pagination for fetching commits.
    """
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable not set.")
        print("Please set your GitHub Personal Access Token as GITHUB_TOKEN.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

    all_commits_nodes: List[Optional[Dict[str, Any]]] = []
    current_cursor: Optional[str] = None
    has_next_page: bool = True
    fetched_commits_count: int = 0

    print(
        f"Attempting to fetch up to {NUM_COMMITS_TO_FETCH} commits from {REPO_OWNER}/{REPO_NAME}..."
    )
    print("Looking for commits with:")
    print(f"  - Message length >= {MIN_COMMIT_MESSAGE_LENGTH} characters")
    print(f"  - Total changes <= {MAX_DIFF_CHANGES * 2} lines")
    print(f"(Commits will be fetched in batches of up to {MAX_COMMITS_PER_PAGE})")

    while fetched_commits_count < NUM_COMMITS_TO_FETCH and has_next_page:
        num_to_fetch_this_run = min(
            MAX_COMMITS_PER_PAGE, NUM_COMMITS_TO_FETCH - fetched_commits_count
        )

        print(
            f"Fetching next batch of {num_to_fetch_this_run} commits (cursor: {current_cursor})..."
        )

        variables: Dict[str, Any] = {
            "owner": REPO_OWNER,
            "name": REPO_NAME,
            "numCommitsPerPage": num_to_fetch_this_run,
            "afterCursor": current_cursor,
        }

        try:
            raw_data = run_graphql_query(COMMITS_QUERY, variables, GITHUB_TOKEN)

            repository_data = raw_data.get("data", {}).get("repository", {})
            if not repository_data:
                print("Error: 'repository' field missing in GraphQL response.")
                break

            default_branch = repository_data.get("defaultBranchRef", {})
            if not default_branch:
                print("Error: 'defaultBranchRef' missing in GraphQL response.")
                break

            target = default_branch.get("target", {})
            history = target.get("history", {})

            if not history:
                print("Error: 'history' missing in GraphQL response.")
                break

            new_nodes: List[Optional[Dict[str, Any]]] = history.get("nodes", [])
            all_commits_nodes.extend(new_nodes)
            fetched_commits_count += len(new_nodes)

            page_info: Dict[str, Any] = history.get("pageInfo", {})
            current_cursor = page_info.get("endCursor")
            has_next_page = page_info.get("hasNextPage", False)

            print(
                f"Fetched {len(new_nodes)} commits in this batch. Total fetched so far: {fetched_commits_count}"
            )

            if not has_next_page and fetched_commits_count < NUM_COMMITS_TO_FETCH:
                print("No more commits to fetch from the repository.")
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from GitHub API: {e}")
            break
        except ValueError as e:
            print(f"GraphQL API Error: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred during fetching: {e}")
            break

    if not all_commits_nodes:
        print("No commits were fetched successfully.")
        return

    print(f"\nSuccessfully fetched a total of {len(all_commits_nodes)} commits.")
    print("Processing commits into training data...")

    training_data = process_commits_to_training_data(
        all_commits_nodes, REPO_OWNER, REPO_NAME, GITHUB_TOKEN
    )

    if not training_data:
        print(
            "No training data was generated. Check if commits met the documentation criteria."
        )
        return

    print(f"Processed {len(training_data)} well-documented commits.")

    try:
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved training data to {OUTPUT_FILENAME}")
        print(f"Total training examples: {len(training_data)}")
    except IOError as e:
        print(f"Error writing data to file {OUTPUT_FILENAME}: {e}")


if __name__ == "__main__":
    main()
