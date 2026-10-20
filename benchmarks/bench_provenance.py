#!/usr/bin/env python3
"""Shared benchmark provenance helpers.

Collects immutable run provenance (git, host, models, dataset, config) for
benchmark results to satisfy pre-registered experiment E-049.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Optional


_VOLATILE_KEYS = frozenset({"timestamp", "git_sha", "run_id", "failed_qa"})  # mirrors bench_checkpoint.py _HASH_SKIP


def config_hash(config: dict) -> str:
    filtered = {k: v for k, v in config.items() if k not in _VOLATILE_KEYS}
    return hashlib.sha256(
        json.dumps(filtered, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    ).hexdigest()


def file_sha256(path: Optional[str | Path]) -> str | None:
    if path is None:
        return None
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        raise FileNotFoundError(f"dataset file does not exist: {p}")
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def model_dir_digests(onnx_dir: Optional[str | Path]) -> dict[str, str]:
    if onnx_dir is None:
        return {}
    p = Path(onnx_dir) if not isinstance(onnx_dir, Path) else onnx_dir
    if not p.exists():
        raise ValueError(f"onnx_dir does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"onnx_dir is not a directory: {p}")
    patterns = ('*.onnx', '*.onnx_data', 'tokenizer.json')
    files: list[Path] = []
    for pat in patterns:
        files.extend(p.rglob(pat))
    if len(files) == 0:
        raise ValueError(f"no model files found in {p}")
    files.sort(key=lambda x: x.relative_to(p).as_posix())
    result: dict[str, str] = {}
    for f in files:
        result[f.relative_to(p).as_posix()] = file_sha256(f)  # type: ignore
    return result


def _git(args: list[str], repo_root: Path) -> str | None:
    """Stdout of a git command, or None when git is absent or the command fails."""
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect(
    config: dict,
    onnx_dir: Optional[str | Path],
    dataset_path: Optional[str | Path],
    repo_root: Optional[Path] = None,
) -> dict:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    # A failed git call yields "unknown"/None, never a false "clean".
    git_commit = _git(['rev-parse', 'HEAD'], repo_root) or "unknown"
    git_dirty = None
    if git_commit != "unknown":
        status = _git(['status', '--porcelain'], repo_root)
        if status is not None:
            git_dirty = bool(status)

    onnxruntime_version = None
    try:
        import onnxruntime  # type: ignore
        onnxruntime_version = onnxruntime.__version__
    except Exception:
        onnxruntime_version = None

    env_prefixes = {
        'TAOSMD_ONNX_QUERY_PREFIX': os.environ.get('TAOSMD_ONNX_QUERY_PREFIX'),
        'TAOSMD_ONNX_DOC_PREFIX': os.environ.get('TAOSMD_ONNX_DOC_PREFIX'),
    }

    return {
        'git_commit': git_commit,
        'git_dirty': git_dirty,
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': platform.python_version(),
        'onnxruntime_version': onnxruntime_version,
        'env_prefixes': env_prefixes,
        'model_files': model_dir_digests(onnx_dir),
        'dataset_sha256': file_sha256(dataset_path),
        'config_hash': config_hash(config),
    }

__all__ = [
    'config_hash',
    'file_sha256',
    'model_dir_digests',
    'collect',
]
