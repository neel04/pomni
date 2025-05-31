import ast
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

# --- Configuration ---
assert load_dotenv(), "Couldn't load envvars"

REPO_OWNER: str = "google"
REPO_NAME: str = "jax"
REPO_URL: str = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

TARGET_LANGUAGE: str = "python"
FILE_EXTENSIONS: List[str] = [".py"]
EXCLUDE_PATTERNS: List[str] = [
    "__pycache__",
    ".git",
    ".pytest_cache",
    "build",
    "dist",
    ".tox",
    "venv",
    "env",
    ".env",
    "node_modules",
    "test_",  # Exclude test files
    "_test.py",
    "tests/",
]

# Docstring filtering configuration
MIN_DOCSTRING_LENGTH: int = 10  # Minimum characters for meaningful docstrings
MIN_FUNCTION_LENGTH: int = 2  # Minimum lines for meaningful functions
MAX_FUNCTION_LENGTH: int = 350  # Maximum lines to avoid huge functions

OUTPUT_FILENAME: str = "data/func_docstrings.json"


class FunctionExtractor(ast.NodeVisitor):
    """
    AST visitor to extract functions with their docstrings and code.
    """

    def __init__(self, source_code: str, file_path: str):
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.file_path = file_path
        self.functions: List[Dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions and extract docstring + code."""
        self._process_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions and extract docstring + code."""
        self._process_function(node)
        self.generic_visit(node)

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """
        Process a function node to extract docstring and code.

        Args:
            node: AST function node (regular or async)
        """
        try:
            # Extract function name and basic info
            func_name = node.name
            start_line = node.lineno - 1  # Convert to 0-based indexing
            end_line = node.end_lineno if node.end_lineno else start_line + 1

            # Calculate function length
            func_length = end_line - start_line
            if func_length < MIN_FUNCTION_LENGTH or func_length > MAX_FUNCTION_LENGTH:
                return

            # Extract docstring
            docstring = ast.get_docstring(node)
            if not docstring or len(docstring.strip()) < MIN_DOCSTRING_LENGTH:
                return

            # Extract function code
            func_code_lines = self.source_lines[start_line:end_line]
            func_code = "\n".join(func_code_lines)

            # Extract function signature for additional context
            func_signature = self._extract_function_signature(node)

            # Extract decorator information
            decorators = self._extract_decorators(node)

            # Create function info dictionary
            func_info = {
                "function_name": func_name,
                "file_path": self.file_path,
                "start_line": start_line + 1,  # Convert back to 1-based for display
                "end_line": end_line,
                "function_length": func_length,
                "signature": func_signature,
                "decorators": decorators,
                "docstring": docstring.strip(),
                "code": func_code,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "is_method": self._is_method(node),
            }

            self.functions.append(func_info)

        except Exception as e:
            print(
                f"Error processing function {getattr(node, 'name', 'unknown')} in {self.file_path}: {e}"
            )

    def _extract_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str:
        """Extract function signature as string."""
        try:
            # Build signature manually from AST
            args = []

            # Regular arguments
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

            # *args
            if node.args.vararg:
                vararg_str = f"*{node.args.vararg.arg}"
                if node.args.vararg.annotation:
                    vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
                args.append(vararg_str)

            # **kwargs
            if node.args.kwarg:
                kwarg_str = f"**{node.args.kwarg.arg}"
                if node.args.kwarg.annotation:
                    kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
                args.append(kwarg_str)

            signature = f"{node.name}({', '.join(args)})"

            # Add return annotation
            if node.returns:
                signature += f" -> {ast.unparse(node.returns)}"

            return signature

        except Exception:
            # Fallback to simple signature
            return f"{node.name}(...)"

    def _extract_decorators(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> List[str]:
        """Extract decorator names."""
        decorators = []
        for decorator in node.decorator_list:
            try:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
                elif isinstance(decorator, ast.Attribute):
                    decorators.append(ast.unparse(decorator))
                elif isinstance(decorator, ast.Call):
                    decorators.append(ast.unparse(decorator))
                else:
                    decorators.append(ast.unparse(decorator))
            except Exception:
                decorators.append("unknown_decorator")
        return decorators

    def _is_method(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function is a method (has 'self' or 'cls' as first parameter)."""
        if not node.args.args:
            return False
        first_arg = node.args.args[0].arg
        return first_arg in ("self", "cls")


def clone_repository(repo_url: str, target_dir: str) -> bool:
    """
    Clone a Git repository to a temporary directory.

    Args:
        repo_url: URL of the repository to clone
        target_dir: Directory to clone into

    Returns:
        True if cloning was successful, False otherwise
    """
    try:
        print(f"Cloning repository {repo_url}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Successfully cloned repository to {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        print(f"Git output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: Git is not installed or not in PATH")
        return False


def find_python_files(repo_dir: str) -> List[str]:
    """
    Find all Python files in the repository, excluding common non-source directories.

    Args:
        repo_dir: Root directory of the repository

    Returns:
        List of Python file paths
    """
    python_files = []
    repo_path = Path(repo_dir)

    for file_path in repo_path.rglob("*.py"):
        # Convert to string for easier pattern matching
        file_str = str(file_path)

        # Check if file should be excluded
        should_exclude = False
        for pattern in EXCLUDE_PATTERNS:
            if pattern in file_str:
                should_exclude = True
                break

        if not should_exclude:
            python_files.append(file_str)

    return python_files


def extract_functions_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract functions with docstrings from a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        List of function information dictionaries
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Parse the AST
        tree = ast.parse(source_code)

        # Extract functions
        extractor = FunctionExtractor(source_code, file_path)
        extractor.visit(tree)

        return extractor.functions

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def format_function_code(code: str, language: str = "py") -> str:
    """
    Format function code with proper markdown code blocks.

    Args:
        code: Raw function code
        language: Programming language for syntax highlighting

    Returns:
        Formatted code with markdown code blocks
    """
    return f"```{language}\n{code}\n```"


def process_functions_to_training_data(
    all_functions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Process extracted functions into training data format.

    Args:
        all_functions: List of function information dictionaries

    Returns:
        List of training examples in the specified format
    """
    training_data = []

    for func_info in all_functions:
        try:
            # Create training example
            training_example = {
                "text_input": func_info["docstring"],
                "output": format_function_code(func_info["code"]),
                "metadata": {
                    "function_name": func_info["function_name"],
                    "file_path": func_info["file_path"],
                    "signature": func_info["signature"],
                    "start_line": func_info["start_line"],
                    "end_line": func_info["end_line"],
                    "function_length": func_info["function_length"],
                    "decorators": func_info["decorators"],
                    "is_async": func_info["is_async"],
                    "is_method": func_info["is_method"],
                    "repo_owner": REPO_OWNER,
                    "repo_name": REPO_NAME,
                },
            }

            training_data.append(training_example)

        except Exception as e:
            print(
                f"Error creating training example for function {func_info.get('function_name', 'unknown')}: {e}"
            )

    return training_data


def main() -> None:
    """
    Main function to clone repository, extract functions with docstrings, and save as training data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

    # Create temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, REPO_NAME)

        print(f"Starting extraction from {REPO_OWNER}/{REPO_NAME}")
        print(f"Target language: {TARGET_LANGUAGE}")
        print(f"File extensions: {FILE_EXTENSIONS}")
        print(f"Minimum docstring length: {MIN_DOCSTRING_LENGTH} characters")
        print(
            f"Function length range: {MIN_FUNCTION_LENGTH}-{MAX_FUNCTION_LENGTH} lines"
        )

        # Clone the repository
        if not clone_repository(REPO_URL, repo_dir):
            print("Failed to clone repository. Exiting.")
            return

        # Find all Python files
        print("Finding Python files...")
        python_files = find_python_files(repo_dir)
        print(f"Found {len(python_files)} Python files to process")

        # Extract functions from all files
        all_functions = []
        processed_files = 0

        for file_path in tqdm(python_files):
            print(f"Processing {file_path}...")
            functions = extract_functions_from_file(file_path)
            all_functions.extend(functions)
            processed_files += 1

            if processed_files % 10 == 0:
                print(
                    f"Processed {processed_files}/{len(python_files)} files, found {len(all_functions)} functions so far"
                )

        print(f"\nFinished processing {processed_files} files")
        print(f"Found {len(all_functions)} functions with docstrings")

        if not all_functions:
            print("No functions with docstrings were found.")
            return

        # Convert to training data format
        print("Converting to training data format...")
        training_data = process_functions_to_training_data(all_functions)

        print(f"Created {len(training_data)} training examples")

        # Save to file
        try:
            with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
                json.dump(training_data, f, indent=4, ensure_ascii=False)

            print(f"Successfully saved training data to {OUTPUT_FILENAME}")
            print(f"Total training examples: {len(training_data)}")

            # Print some statistics
            print("\n--- Statistics ---")
            print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
            print(f"Files processed: {processed_files}")
            print(f"Functions found: {len(all_functions)}")
            print(f"Training examples: {len(training_data)}")

            # Function type statistics
            methods = sum(1 for func in all_functions if func["is_method"])
            async_funcs = sum(1 for func in all_functions if func["is_async"])
            print(f"Methods: {methods}")
            print(f"Async functions: {async_funcs}")
            print(f"Regular functions: {len(all_functions) - methods - async_funcs}")

        except IOError as e:
            print(f"Error writing data to file {OUTPUT_FILENAME}: {e}")


if __name__ == "__main__":
    main()
