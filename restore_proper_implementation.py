"""
We need to restore the proper install-skill implementation from commit 78b3dd0e.

The current implementation is too simple and doesn't handle:
1. Manifest files (.taosmd-skill-manifest.json)
2. Version comparison
3. Non-empty manifest directory handling
4. Atomic writes
5. Half-apply prevention

We need to restore the complex logic from 78b3dd0e.
"""
import os
import subprocess
import sys

def get_original_from_commit(commit_hash, file_path):
    """Get a file from a specific commit"""
    cmd = ['git', 'show', f'{commit_hash}:{file_path}']
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

# Get the original cli.py from 78b3dd0e
original_cli = get_original_from_commit('78b3dd0e', 'taosmd/cli.py')

# Extract the relevant sections
lines = original_cli.split('\n')

# Find and extract the key functions
functions_to_extract = ['_version_tuple', '_parse_skill_version', '_write_skill_manifest', '_run_install_skill', '_install_skill_cmd']

extracted_code = []
for func_name in functions_to_extract:
    start = None
    for i, line in enumerate(lines):
        if f'def {func_name}' in line:
            start = i
            break
    
    if start is not None:
        # Find the end of the function (next function or end of file)
        end = None
        for i in range(start + 1, len(lines)):
            if lines[i].strip().startswith('def ') and i > start:
                end = i
                break
        
        if end is None:
            end = len(lines)
        
        # Extract the function
        func_code = '\n'.join(lines[start:end])
        extracted_code.append(func_code)
        print(f"Extracted {func_name}")

# Also extract constants
extracted_code.append('MANIFEST_NAME = ".taosmd-skill-manifest.json"')

print("\n=== RESTORED FUNCTIONS ===\n")
for code in extracted_code:
    print(code)
    print("\n" + "="*80 + "\n")
