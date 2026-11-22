"""Test that .claude/skills/taosmd-a2a/ is byte-identical to taosmd/skills/taosmd-a2a/."""

import os
from pathlib import Path


def test_skill_copy_sync():
    """Assert .claude/skills/taosmd-a2a/ matches taosmd/skills/taosmd-a2a/ exactly."""
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "taosmd" / "skills" / "taosmd-a2a"
    dst_dir = repo_root / ".claude" / "skills" / "taosmd-a2a"

    assert src_dir.exists(), f"Source directory {src_dir} does not exist"
    assert dst_dir.exists(), f"Destination directory {dst_dir} does not exist"

    # Get all files in source directory (relative paths)
    src_files = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            rel_path = Path(root).relative_to(src_dir) / f
            src_files.append(rel_path)

    # Get all files in destination directory (relative paths)
    dst_files = []
    for root, dirs, files in os.walk(dst_dir):
        for f in files:
            rel_path = Path(root).relative_to(dst_dir) / f
            dst_files.append(rel_path)

    src_files_set = set(src_files)
    dst_files_set = set(dst_files)

    # Check for missing files in destination
    missing = src_files_set - dst_files_set
    assert not missing, (
        f"Missing files in {dst_dir}: {sorted(missing)}. "
        f"Copy them from {src_dir}/"
    )

    # Check for extra files in destination
    extra = dst_files_set - src_files_set
    assert not extra, (
        f"Extra files in {dst_dir}: {sorted(extra)}. "
        f"Remove them or add to {src_dir}/"
    )

    # Check byte-identical content for each file
    for rel_path in sorted(src_files_set):
        src_file = src_dir / rel_path
        dst_file = dst_dir / rel_path

        src_content = src_file.read_bytes()
        dst_content = dst_file.read_bytes()

        assert src_content == dst_content, (
            f"Content mismatch for {rel_path}. "
            f"Copy it from {src_dir}/{rel_path} to {dst_dir}/{rel_path}"
        )


if __name__ == "__main__":
    test_skill_copy_sync()
    print("All checks passed!")
