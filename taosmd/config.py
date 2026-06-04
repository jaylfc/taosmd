"""Install-wide taOSmd config — a single JSON file at ``~/.taosmd/config.json``.

Historically the memory/Librarian model was a per-agent setting stored on
each agent record. That made the model awkward to manage on a single-user
install (set it once, get it everywhere) and impossible to use standalone
without first registering an agent. The model is now a system-wide setting
that lives here instead.

The data dir is resolved exactly like :mod:`taosmd.api` — ``TAOSMD_DATA_DIR``
when set, otherwise ``~/.taosmd``. The file is read/written defensively: a
missing or corrupt file is treated as "nothing configured" rather than an
error, so a hand-edited or half-written config never bricks ingest.

Stdlib only — this module must stay dependency-free so it can be imported
from anywhere in the package without pulling in heavier deps.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = os.path.expanduser("~/.taosmd")

# Key under which the global memory/Librarian model is stored.
_MEMORY_MODEL_KEY = "memory_model"


def _resolve_data_dir(data_dir=None) -> str:
    """Resolve the data dir — mirrors ``taosmd.api._resolve_data_dir``."""
    if data_dir is not None:
        return os.fspath(data_dir)
    env = os.environ.get("TAOSMD_DATA_DIR")
    if env:
        return env
    return _DEFAULT_DATA_DIR


def _config_path(data_dir=None) -> Path:
    return Path(_resolve_data_dir(data_dir)) / "config.json"


def _read(data_dir=None) -> dict:
    """Load the config dict. Missing/corrupt file → empty dict (unset)."""
    path = _config_path(data_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("taosmd: failed to read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: dict, data_dir=None) -> None:
    """Persist the config dict atomically (tmp + rename)."""
    path = _config_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def get_memory_model(data_dir=None) -> str | None:
    """Return the configured global memory model, or ``None`` if unset.

    The value is a ``provider:model`` string such as ``"ollama:qwen3:4b"``.
    """
    model = _read(data_dir).get(_MEMORY_MODEL_KEY)
    if isinstance(model, str) and model.strip():
        return model
    return None


def set_memory_model(model: str, clear: bool = False, data_dir=None) -> None:
    """Persist the global memory model.

    Args:
        model: ``provider:model`` string. Ignored when ``clear`` is True.
        clear: when True, remove the setting (unset → ``get`` returns None).

    Raises:
        ValueError: when ``clear`` is False and ``model`` is not a
            non-empty string.
    """
    data = _read(data_dir)
    if clear:
        data.pop(_MEMORY_MODEL_KEY, None)
    else:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string (or pass clear=True)")
        data[_MEMORY_MODEL_KEY] = model.strip()
    _write(data, data_dir)


def resolve_memory_model(fallback: str | None = None, data_dir=None) -> str | None:
    """Return the global memory model if set, else ``fallback``.

    Consumers call this so an unset global transparently falls back to
    their existing default — standalone installs that never set a model
    keep working exactly as before.
    """
    model = get_memory_model(data_dir)
    return model if model is not None else fallback


__all__ = [
    "get_memory_model",
    "set_memory_model",
    "resolve_memory_model",
]
