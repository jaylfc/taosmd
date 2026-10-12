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


def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    ).hexdigest()


def file_sha256(path: Optional[str | Path]) -> str | None:
    if path is None:
        return None
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        return None
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
    if not p.exists() or not p.is_dir():
        return {}
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


def collect(config: dict, onnx_dir: Optional[str | Path], dataset_path: Optional[str | Path]) -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        git_commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        ).stdout.strip()
    except Exception:
        git_commit = None
    try:
        git_dirty = bool(
            subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            ).stdout.strip()
        )
    except Exception:
        git_dirty = False

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
