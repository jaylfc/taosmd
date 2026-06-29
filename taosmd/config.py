"""Install-wide taOSmd config: a single JSON file at ``~/.taosmd/config.json``.

Historically the memory/Librarian model was a per-agent setting stored on
each agent record. That made the model awkward to manage on a single-user
install (set it once, get it everywhere) and impossible to use standalone
without first registering an agent. The model is now a system-wide setting
that lives here instead.

The data dir is resolved exactly like :mod:`taosmd.api`: ``TAOSMD_DATA_DIR``
when set, otherwise ``~/.taosmd``. The file is read/written defensively: a
missing or corrupt file is treated as "nothing configured" rather than an
error, so a hand-edited or half-written config never bricks ingest.

Stdlib only: this module must stay dependency-free so it can be imported
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
# Key under which the optional remote server URL is stored.
_SERVER_URL_KEY = "server_url"
# Key under which the optional remote server bearer token is stored.
_SERVER_TOKEN_KEY = "server_token"
# Key under which the global default recipe id is stored.
_DEFAULT_RECIPE_KEY = "default_recipe"
_REGISTRY_URL_KEY = "registry_url"
# Key under which the registry auth token (taOS local/admin token) is stored.
# Used to poll the auth-gated registry revoked feed; the pubkey feed is public.
_REGISTRY_TOKEN_KEY = "registry_token"
# Who manages this taosmd instance: "standalone" (default) or "taos".
_MANAGED_BY_KEY = "managed_by"
# Override: serve the web dashboard even when managed_by=taos.
_SERVE_DASHBOARD_KEY = "serve_dashboard"
# Key under which the active global generator-profile id is stored.
_GENERATOR_PROFILE_KEY = "generator_profile"

MANAGED_BY_STANDALONE = "standalone"
MANAGED_BY_TAOS = "taos"


def _resolve_data_dir(data_dir=None) -> str:
    """Resolve the data dir, mirroring ``taosmd.api._resolve_data_dir``."""
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


def get_generator_profile(data_dir=None) -> str | None:
    """Return the active global generator-profile id, or None if unset."""
    pid = _read(data_dir).get(_GENERATOR_PROFILE_KEY)
    if isinstance(pid, str) and pid.strip():
        return pid
    return None


def set_generator_profile(profile_id: str, clear: bool = False, data_dir=None) -> None:
    """Persist the active global generator-profile id.

    Args:
        profile_id: a registered profile id. Ignored when clear is True.
        clear: when True, remove the setting (unset).

    Raises:
        ValueError: when clear is False and profile_id is not a non-empty str.
    """
    data = _read(data_dir)
    if clear:
        data.pop(_GENERATOR_PROFILE_KEY, None)
        _write(data, data_dir)
        return
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be a non-empty string")
    data[_GENERATOR_PROFILE_KEY] = profile_id.strip()
    _write(data, data_dir)


def get_default_recipe(data_dir=None) -> str | None:
    """Return the configured global default recipe id, or None if unset."""
    rid = _read(data_dir).get(_DEFAULT_RECIPE_KEY)
    if isinstance(rid, str) and rid.strip():
        return rid
    return None


def set_default_recipe(recipe_id: str, clear: bool = False, data_dir=None) -> None:
    """Persist the global default recipe id (or clear it).

    Args:
        recipe_id: Stable recipe slug. Ignored when ``clear`` is True.
        clear: when True, remove the setting (unset -> ``get`` returns None).

    Raises:
        ValueError: when ``clear`` is False and ``recipe_id`` is not a
            non-empty string.
    """
    data = _read(data_dir)
    if clear:
        data.pop(_DEFAULT_RECIPE_KEY, None)
    else:
        if not isinstance(recipe_id, str) or not recipe_id.strip():
            raise ValueError("recipe_id must be a non-empty string (or pass clear=True)")
        data[_DEFAULT_RECIPE_KEY] = recipe_id.strip()
    _write(data, data_dir)


def get_controls(data_dir=None) -> dict:
    """Resolved value for every memory control.

    Registry defaults (``taosmd.controls.default_controls``) overlaid with any
    persisted runtime-control overrides under the config ``controls`` section.
    Store-level and consumer-scope controls report their registry default, since
    they are not live toggles (see :mod:`taosmd.controls`).
    """
    from taosmd import controls as _c  # noqa: PLC0415
    resolved = _c.default_controls()
    stored = _read(data_dir).get("controls")
    if isinstance(stored, dict):
        for cid, val in stored.items():
            ctrl = _c.CONTROLS.get(cid)
            if ctrl is None or ctrl.scope != "runtime":
                continue
            try:
                resolved[cid] = _c.validate_control(cid, val)
            except ValueError:
                pass  # ignore a bad persisted value, keep the default
    return resolved


def set_control(control_id: str, value, data_dir=None) -> dict:
    """Validate and persist one runtime control; return the new resolved controls.

    Raises ValueError for an unknown control, a bad value, or a non-runtime
    control: store-level controls (embedder, binary_quant) require a re-index,
    and consumer-scope controls (self_verify) are applied in the consumer's
    answer-generation, not in taOSmd core.
    """
    from taosmd import controls as _c  # noqa: PLC0415
    ctrl = _c.CONTROLS.get(control_id)
    if ctrl is None:
        raise ValueError(f"unknown control: {control_id!r}")
    if ctrl.scope != "runtime":
        raise ValueError(
            f"{control_id} is {ctrl.scope}-scope, not a live toggle "
            "(store-level controls need a re-index; consumer controls apply in answer-gen)")
    coerced = _c.validate_control(control_id, value)
    data = _read(data_dir)
    section = data.get("controls")
    if not isinstance(section, dict):
        section = {}
    section[control_id] = coerced
    data["controls"] = section
    _write(data, data_dir)
    return get_controls(data_dir)


def get_runtime_overrides(data_dir=None) -> dict:
    """Persisted runtime-control overrides only, with no registry defaults filled.

    ``get_controls`` fills every control with its default (for display). This
    returns only the runtime controls a user has actually persisted, so the
    retrieval path can treat the recipe as the baseline and apply a control
    *only* when it was explicitly set. Bad or non-runtime persisted values are
    skipped. ``prefer_verified`` is excluded here because it has no recipe
    baseline and is resolved separately via :func:`get_controls`.
    """
    from taosmd import controls as _c  # noqa: PLC0415
    out: dict = {}
    stored = _read(data_dir).get("controls")
    if not isinstance(stored, dict):
        return out
    for cid, val in stored.items():
        if cid == "prefer_verified":
            continue
        ctrl = _c.CONTROLS.get(cid)
        if ctrl is None or ctrl.scope != "runtime":
            continue
        try:
            out[cid] = _c.validate_control(cid, val)
        except ValueError:
            pass
    return out


def resolve_memory_model(fallback: str | None = None, data_dir=None) -> str | None:
    """Return the global memory model if set, else ``fallback``.

    Consumers call this so an unset global transparently falls back to
    their existing default. Standalone installs that never set a model
    keep working exactly as before.
    """
    model = get_memory_model(data_dir)
    return model if model is not None else fallback


# ---------------------------------------------------------------------------
# Remote server URL
# ---------------------------------------------------------------------------

def get_server_url(data_dir=None) -> str | None:
    """Return the configured remote server URL, or ``None`` if unset.

    Resolution order (first non-empty wins):

    1. ``TAOSMD_SERVER_URL`` environment variable
    2. ``server_url`` key in ``~/.taosmd/config.json``

    A remote URL tells the service layer to delegate every data call to
    that server instead of running a local store.  Example value:
    ``"http://pi.local:7900"`` or a Tailscale MagicDNS URL.
    """
    env = os.environ.get("TAOSMD_SERVER_URL")
    if env and env.strip():
        return env.strip()
    url = _read(data_dir).get(_SERVER_URL_KEY)
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def set_server_url(url: str, clear: bool = False, data_dir=None) -> None:
    """Persist the remote server URL.

    Args:
        url: Base URL of the remote taOSmd server, e.g. ``"http://pi:7900"``.
            Ignored when ``clear`` is True.
        clear: when True, remove the setting.

    Raises:
        ValueError: when ``clear`` is False and ``url`` is not a non-empty string.
    """
    data = _read(data_dir)
    if clear:
        data.pop(_SERVER_URL_KEY, None)
    else:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string (or pass clear=True)")
        data[_SERVER_URL_KEY] = url.strip()
    _write(data, data_dir)


# ---------------------------------------------------------------------------
# Agent registry URL (opt-in A2A bus authentication)
# ---------------------------------------------------------------------------

def get_registry_url(data_dir=None) -> str | None:
    """Return the configured taOS agent-registry base URL, or ``None``.

    Resolution order (first non-empty wins):

    1. ``TAOSMD_REGISTRY_URL`` environment variable
    2. ``registry_url`` key in ``~/.taosmd/config.json``

    When set, the A2A bus authenticates senders against this registry (verify
    the EdDSA-JWT against ``<url>/api/agents/registry/pubkey`` and check the
    revocation feed). When unset, the bus keeps free-handle behaviour so a
    standalone install is unaffected.
    """
    env = os.environ.get("TAOSMD_REGISTRY_URL")
    if env and env.strip():
        return env.strip()
    url = _read(data_dir).get(_REGISTRY_URL_KEY)
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def set_registry_url(url: str, clear: bool = False, data_dir=None) -> None:
    """Persist the agent-registry base URL (or clear it).

    Args:
        url: Base URL of the taOS registry, e.g. ``"http://taos:8000"``.
            Ignored when ``clear`` is True.
        clear: when True, remove the setting (bus reverts to free handles).

    Raises:
        ValueError: when ``clear`` is False and ``url`` is not a non-empty string.
    """
    data = _read(data_dir)
    if clear:
        data.pop(_REGISTRY_URL_KEY, None)
    else:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string (or pass clear=True)")
        data[_REGISTRY_URL_KEY] = url.strip()
    _write(data, data_dir)


def get_registry_token(data_dir=None) -> str | None:
    """Return the configured registry auth token (taOS local/admin), or ``None``.

    Resolution order (first non-empty wins):

    1. ``TAOSMD_REGISTRY_TOKEN`` environment variable
    2. ``registry_token`` key in ``~/.taosmd/config.json``

    The token is sent as ``Authorization: Bearer <token>`` only on the
    auth-gated registry revoked feed; the pubkey feed is public and is fetched
    without it. Unset means the revoked poll is unauthenticated (pre-#710
    behaviour). The token is never logged or printed.
    """
    env = os.environ.get("TAOSMD_REGISTRY_TOKEN")
    if env and env.strip():
        return env.strip()
    token = _read(data_dir).get(_REGISTRY_TOKEN_KEY)
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def set_registry_token(token: str, clear: bool = False, data_dir=None) -> None:
    """Persist the registry auth token (or clear it).

    Args:
        token: The taOS local/admin token used to poll the revoked feed.
            Ignored when ``clear`` is True.
        clear: when True, remove the setting.

    Raises:
        ValueError: when ``clear`` is False and ``token`` is not a
            non-empty string.
    """
    data = _read(data_dir)
    if clear:
        data.pop(_REGISTRY_TOKEN_KEY, None)
    else:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("token must be a non-empty string (or pass clear=True)")
        data[_REGISTRY_TOKEN_KEY] = token.strip()
    _write(data, data_dir)


# ---------------------------------------------------------------------------
# Remote server bearer token
# ---------------------------------------------------------------------------

def get_server_token(data_dir=None) -> str | None:
    """Return the configured remote server bearer token, or ``None`` if unset.

    Resolution order (first non-empty wins):

    1. ``TAOSMD_TOKEN`` environment variable
    2. ``server_token`` key in ``~/.taosmd/config.json``

    When set, the token is sent as ``Authorization: Bearer <token>`` on every
    request to the remote server.  The token is never logged or printed.
    """
    env = os.environ.get("TAOSMD_TOKEN")
    if env and env.strip():
        return env.strip()
    token = _read(data_dir).get(_SERVER_TOKEN_KEY)
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def set_server_token(token: str, clear: bool = False, data_dir=None) -> None:
    """Persist the remote server bearer token.

    Args:
        token: Bearer token string.  Ignored when ``clear`` is True.
        clear: when True, remove the setting.

    Raises:
        ValueError: when ``clear`` is False and ``token`` is not a
            non-empty string.
    """
    data = _read(data_dir)
    if clear:
        data.pop(_SERVER_TOKEN_KEY, None)
    else:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("token must be a non-empty string (or pass clear=True)")
        data[_SERVER_TOKEN_KEY] = token.strip()
    _write(data, data_dir)


def get_managed_by(data_dir=None) -> str:
    """Return the managed_by value: ``"standalone"`` (default) or ``"taos"``.

    Resolution order (first non-empty wins):

    1. ``TAOSMD_MANAGED_BY`` environment variable
    2. ``managed_by`` key in ``~/.taosmd/config.json``
    3. ``"standalone"`` (default)

    When ``"taos"``, the taOS app owns the auth/permission UX and the web
    dashboard is hidden by default (controlled by :func:`get_serve_dashboard`).
    taOS writes this at provision time; standalone installs never set it.
    """
    env = os.environ.get("TAOSMD_MANAGED_BY")
    if env and env.strip() in (MANAGED_BY_STANDALONE, MANAGED_BY_TAOS):
        return env.strip()
    val = _read(data_dir).get(_MANAGED_BY_KEY)
    if val in (MANAGED_BY_STANDALONE, MANAGED_BY_TAOS):
        return val
    return MANAGED_BY_STANDALONE


def set_managed_by(value: str, data_dir=None) -> None:
    """Persist the managed_by value (``"standalone"`` or ``"taos"``).

    Raises:
        ValueError: when ``value`` is not one of the allowed strings.
    """
    if value not in (MANAGED_BY_STANDALONE, MANAGED_BY_TAOS):
        raise ValueError(f"managed_by must be 'standalone' or 'taos', got {value!r}")
    data = _read(data_dir)
    data[_MANAGED_BY_KEY] = value
    _write(data, data_dir)


def get_serve_dashboard(data_dir=None) -> bool:
    """Return whether the web dashboard should be served.

    Resolution order:

    1. ``TAOSMD_SERVE_DASHBOARD`` env var (``"1"`` or ``"true"`` = True)
    2. ``serve_dashboard`` bool in ``~/.taosmd/config.json``
    3. Derived default: True when managed_by=standalone, False when managed_by=taos.

    When managed_by=taos, the taOS apps render all UI. taOS can override this
    back to True via the config key or env var for debugging.
    """
    env = os.environ.get("TAOSMD_SERVE_DASHBOARD")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    data = _read(data_dir)
    if _SERVE_DASHBOARD_KEY in data:
        return bool(data[_SERVE_DASHBOARD_KEY])
    return get_managed_by(data_dir) == MANAGED_BY_STANDALONE


def set_serve_dashboard(value: bool, data_dir=None) -> None:
    """Persist the serve_dashboard override."""
    data = _read(data_dir)
    data[_SERVE_DASHBOARD_KEY] = bool(value)
    _write(data, data_dir)


__all__ = [
    "get_memory_model",
    "set_memory_model",
    "resolve_memory_model",
    "get_default_recipe",
    "set_default_recipe",
    "get_server_url",
    "set_server_url",
    "get_server_token",
    "set_server_token",
    "get_registry_url",
    "set_registry_url",
    "get_registry_token",
    "set_registry_token",
    "get_managed_by",
    "set_managed_by",
    "get_serve_dashboard",
    "set_serve_dashboard",
    "MANAGED_BY_STANDALONE",
    "MANAGED_BY_TAOS",
    "get_generator_profile",
    "set_generator_profile",
]
