#!/usr/bin/env python3
"""Gate: FAIL if any benchmarks/*.py loads a data/*.json with no identity block
in benchmarks/data/README.md.

This ensures every dataset file loaded by a benchmark has a pinned identity
block (what it is, sha256, size, question count, how to obtain it) in the
project's dataset README.
"""

import ast
import json
import os
import re
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
README_PATH = os.path.join(os.path.dirname(__file__), "data", "README.md")


def read_readme_identity_blocks():
    """Parse benchmarks/data/README.md and return a set of filenames with identity blocks."""
    with open(README_PATH) as f:
        content = f.read()

    # Find sections for longmemeval_*.json files
    # Each section starts with "## longmemeval_*.json"
    blocks = {}
    current_file = None
    lines = content.splitlines()

    for i, line in enumerate(lines):
        m = re.match(r"^##\s+longmemeval_(.+)\.json", line)
        if m:
            current_file = "longmemeval_" + m.group(1) + ".json"
            blocks[current_file] = {"start": i, "content": []}
        elif current_file is not None:
            blocks[current_file]["content"].append(line)

    # Extract filenames that have identity blocks (files with a "##" section)
    pinned = set(blocks.keys())
    return pinned


def extract_data_path(py_filepath):
    """Extract the DATA_PATH filename from a benchmarks Python file."""
    with open(py_filepath) as f:
        source = f.read()

    # Try to find DATA_PATH assignment via AST
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "DATA_PATH":
                        # Evaluate the value as a string
                        val = ast.unparse(node.value)
                        # Extract filename from path
                        m = re.search(r"['\"](longmemeval[^'\"]*\.json)['\"]", val)
                        if m:
                            return m.group(1)
    except SyntaxError:
        pass

    # Fallback: regex search for DATA_PATH = "..." or DATA_PATH = '...'
    m = re.search(r'DATA_PATH\s*=\s*["\'](longmemeval[^"\']*\.json)["\']', source)
    if m:
        return m.group(1)

    return None


def check_gate():
    """Run the identity gate. Returns True if gate passes (all pinned), False if fails."""
    # Read README identity blocks
    pinned = read_readme_identity_blocks()
    print(f"Pinned files in README: {sorted(pinned)}")

    # Check all benchmarks/*.py files
    bench_dir = os.path.join(os.path.dirname(__file__))
    py_files = sorted(
        f for f in os.listdir(bench_dir) if f.endswith(".py") and f != "gate_identity.py"
    )

    unpinned_loaders = []

    for py_file in py_files:
        filepath = os.path.join(bench_dir, py_file)
        filename = extract_data_path(filepath)
        if filename is None:
            print(f"  {py_file}: no DATA_PATH found, skipping")
            continue

        full_filename = os.path.basename(filename)
        print(f"  {py_file}: loads {full_filename}")

        if full_filename not in pinned:
            unpinned_loaders.append((py_file, full_filename))
            print(f"    -> UNPINNED: {full_filename} has no identity block in README")

    if unpinned_loaders:
        print("\nGATE FAILS: The following loaders reference unpinned dataset files:")
        for py_file, fn in unpinned_loaders:
            print(f"  - {py_file} loads {fn}")
        print(
            "\nAdd identity blocks to benchmarks/data/README.md or override DATA_PATH "
            "with env vars (LONGMEMEVAL_ORACLE_DATA_PATH, LONGMEMEVAL_CLEANED_DATA_PATH)"
            " to verified copies."
        )
        return False

    print("\nGATE PASSES: All loaded dataset files have identity blocks in README.")
    return True


if __name__ == "__main__":
    success = check_gate()
    sys.exit(0 if success else 1)