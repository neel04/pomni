import argparse
import json
import os
import re
from typing import Any, Dict, List

import nbformat
from tqdm import tqdm

JAX_REPO_PATH = "/home/neel/Documents/work/pomni/jax"
NOTEBOOKS_DIR = os.path.join(JAX_REPO_PATH, "docs/notebooks")
OUTPUT_FILE = "data/jax_examples_qa_pairs.json"


def clean_markdown(markdown_text: str) -> str:
    """
    Clean markdown text by removing base64-encoded images and other unwanted elements.

    Args:
        markdown_text: The original markdown text

    Returns:
        Cleaned markdown text
    """
    # Remove HTML img tags with base64 data
    cleaned_text = re.sub(
        r'<img[^>]*src="data:image/[^"]*base64,[^"]*"[^>]*>', "[IMAGE]", markdown_text
    )

    # Remove Markdown style images with base64 data
    cleaned_text = re.sub(
        r"!\[.*?\]\(data:image/[^)]*base64,[^)]*\)",  # noqa: F821
        "[IMAGE]",
        cleaned_text,
    )

    # Handle general HTML img tags that might include base64
    cleaned_text = re.sub(r"<img[^>]*>", "[IMAGE]", cleaned_text)

    # Remove any remaining base64 chunks (might catch some rare cases)
    cleaned_text = re.sub(
        r'src="data:image/[^"]*base64,[^"]*"', 'src="[IMAGE]"', cleaned_text
    )

    # Replace multiple consecutive [IMAGE] tags with a single one
    cleaned_text = re.sub(r"(\[IMAGE\]\s*){2,}", "[IMAGE]\n", cleaned_text)

    # Replace multiple blank lines with a single one
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def extract_qa_pairs_from_notebook(notebook_path: str) -> List[Dict[str, Any]]:
    """
    Extract QA pairs from a Jupyter notebook where markdown cells are questions
    and subsequent code cells are answers.

    Args:
        notebook_path: Path to the Jupyter notebook file

    Returns:
        A list of dictionaries containing QA pairs
    """
    qa_pairs = []

    try:
        # Load the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        notebook_filename = os.path.basename(notebook_path)
        relative_path = os.path.relpath(notebook_path, start=JAX_REPO_PATH)
        cells = notebook.cells

        i = 0
        while i < len(cells):
            # Find a markdown cell to start a QA pair
            if cells[i].cell_type == "markdown":
                # Clean the markdown to remove base64 images
                question = clean_markdown(cells[i].source.strip())
                i += 1  # Move to the next cell

                # Collect all consecutive code cells
                code_blocks = []
                while i < len(cells) and cells[i].cell_type == "code":
                    if cells[i].source.strip():  # Skip empty code cells
                        code_blocks.append(cells[i].source.strip())
                    i += 1

                # Only create a QA pair if we have both a question and at least one code block
                if question and code_blocks:
                    combined_code = "\n\n".join(code_blocks)
                    qa_pairs.append({
                        "text_input": question,
                        "output": f"```python\n{combined_code}\n```",
                        "metadata": {
                            "source_file": notebook_filename,
                            "source_path": relative_path,
                            "type": "notebook_qa",
                        },
                    })
            else:
                # Skip cells that don't fit our pattern
                i += 1

    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")

    return qa_pairs


def find_notebooks(root_dir: str) -> List[str]:
    """
    Find all Jupyter notebook files in a directory recursively.

    Args:
        root_dir: Root directory to search

    Returns:
        List of paths to notebook files
    """
    notebook_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".ipynb") and not filename.endswith(
                "-checkpoint.ipynb"
            ):
                notebook_path = os.path.join(dirpath, filename)
                notebook_paths.append(notebook_path)

    return notebook_paths


def process_notebooks(notebooks_dir: str) -> List[Dict[str, Any]]:
    """
    Process all notebooks in the specified directory and extract QA pairs.

    Args:
        notebooks_dir: Directory containing notebooks

    Returns:
        List of QA pairs from all notebooks
    """
    all_qa_pairs = []

    # Find all notebooks
    notebook_paths = find_notebooks(notebooks_dir)
    print(f"Found {len(notebook_paths)} notebooks to process")

    # Process each notebook
    for notebook_path in tqdm(notebook_paths, desc="Processing notebooks"):
        qa_pairs = extract_qa_pairs_from_notebook(notebook_path)
        all_qa_pairs.extend(qa_pairs)

    return all_qa_pairs


def save_qa_pairs(qa_pairs: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save QA pairs to a JSONL file.

    Args:
        qa_pairs: List of QA pairs
        output_file: Path to output file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Saved {len(qa_pairs)} QA pairs to {output_file}")


def main() -> None:
    """Main function to process notebooks and extract QA pairs."""
    parser = argparse.ArgumentParser(description="Extract QA pairs from JAX notebooks.")
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE, help="Output file path"
    )
    parser.add_argument(
        "--notebooks-dir",
        type=str,
        default=NOTEBOOKS_DIR,
        help="Directory containing notebooks",
    )

    args = parser.parse_args()

    # Check if notebooks directory exists
    if not os.path.exists(args.notebooks_dir):
        print(f"Error: Notebooks directory {args.notebooks_dir} does not exist.")
        print("Please update JAX_REPO_PATH in the script or provide --notebooks-dir.")
        return

    # Process notebooks and extract QA pairs
    qa_pairs = process_notebooks(args.notebooks_dir)

    # Save QA pairs to file
    save_qa_pairs(qa_pairs, args.output)

    print(f"Extracted {len(qa_pairs)} QA pairs from notebooks in {args.notebooks_dir}")


if __name__ == "__main__":
    main()
