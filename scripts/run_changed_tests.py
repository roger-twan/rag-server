#!/usr/bin/env python3
"""
Run tests for changed files only.
Maps changed source files to their corresponding test files.
"""

import subprocess
import sys
from pathlib import Path


def get_test_for_source(source_path: str) -> str | None:
    """Map a source file to its corresponding test file."""
    path = Path(source_path)

    # Only process files in app/
    if not path.parts[0] == "app":
        return None

    # Build test file path
    test_name = f"test_{path.stem}.py"
    test_path = Path("tests") / test_name

    if test_path.exists():
        return str(test_path)
    return None


def main():
    """Run tests for changed files."""
    # Get changed files from pre-commit (passed as arguments)
    changed_files = sys.argv[1:] if len(sys.argv) > 1 else []

    if not changed_files:
        print("No files changed, skipping tests")
        sys.exit(0)

    # Map changed files to test files
    test_files = set()
    for file in changed_files:
        if file.startswith("tests/test_") and file.endswith(".py"):
            # If a test file was changed, run it directly
            test_files.add(file)
        elif file.startswith("app/") and file.endswith(".py"):
            # If a source file was changed, find its test
            test_file = get_test_for_source(file)
            if test_file:
                test_files.add(test_file)

    if not test_files:
        print("No relevant test files to run")
        sys.exit(0)

    # Run the identified tests
    cmd = [".venv/bin/python", "-m", "pytest", "-v"] + sorted(test_files)
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
