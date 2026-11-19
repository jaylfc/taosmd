"""Contract probe for taOSmd releases.

A packaged probe that taOS CI can run against a candidate release to verify
the HTTP contract is intact. Standard library only so it runs from a bare
wheel install.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _get_json(base_url: str, path: str, timeout: int) -> tuple[int, dict[str, Any], str]:
    """GET a path and return (status, parsed_json_or_empty, content_type)."""
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            raw = resp.read().decode()
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {}
            return resp.status, data, ct
    except urllib.error.HTTPError as exc:
        ct = exc.headers.get("Content-Type", "")
        raw = exc.read().decode()
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = {}
        return exc.code, data, ct
    except Exception as exc:
        raise ConnectionError(str(exc)) from exc


def _post_json(base_url: str, path: str, payload: dict, timeout: int) -> tuple[int, dict[str, Any], str]:
    """POST JSON to a path and return (status, parsed_json_or_empty, content_type)."""
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            raw = resp.read().decode()
            try:
                resp_data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                resp_data = {}
            return resp.status, resp_data, ct
    except urllib.error.HTTPError as exc:
        ct = exc.headers.get("Content-Type", "")
        raw = exc.read().decode()
        try:
            resp_data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            resp_data = {}
        return exc.code, resp_data, ct
    except Exception as exc:
        raise ConnectionError(str(exc)) from exc


def _is_json_content_type(ct: str) -> bool:
    return ct.startswith("application/json")


# Known POST-only routes (no GET handler) that appear in capability route_markers.
# These return 200 text/html (SPA fallback) on GET, so we skip them.
# Also includes streaming endpoints that hang on GET.
_POST_ONLY_ROUTES = frozenset({
    "/ingest",
    "/ingest/batch",
    "/refs/fetch",
    "/a2a/send",
    "/a2a/import",
    "/a2a/inbox/advance",
    "/a2a/ack",
    "/a2a/threads",
    "/a2a/receipts",
    "/a2a/admin/prune-receipts",
    "/a2a/admin/delete-channel",
    "/a2a/admin/rename-channel",
    "/a2a/admin/supersede-message",
    "/a2a/admin/set-channel-acl",
    "/pending/resolve",
    "/controls",
    "/generator-profile",
    "/a2a/stream",
    "/grants",
    "/grants/",
    "/index",
    "/link",
})


def run(base_url: str, write_probe: bool = False, timeout: int = 10) -> dict[str, Any]:
    """Run the contract check against a running taOSmd server.

    Args:
        base_url: Base URL of the server (e.g. http://127.0.0.1:7900)
        write_probe: If True, perform a real write round-trip (ingest + search + A2A)
        timeout: Request timeout in seconds

    Returns:
        Dict with keys: ok, base_url, version, commit, checks, skipped, missing_on_server
    """
    from taosmd import capabilities

    checks: list[CheckResult] = []
    skipped: list[str] = []
    missing_on_server: list[str] = []

    # 1. GET /health
    try:
        status, health, ct = _get_json(base_url, "/health", timeout)
    except ConnectionError:
        raise
    if status != 200:
        checks.append(CheckResult("health_status", False, f"status {status}"))
    elif not isinstance(health, dict):
        checks.append(CheckResult("health_json", False, "response not a JSON object"))
    elif health.get("status") != "ok":
        checks.append(CheckResult("health_status_ok", False, f"status={health.get('status')!r}"))
    elif not isinstance(health.get("version"), str) or not health["version"]:
        checks.append(CheckResult("health_version", False, "version missing or empty"))
    elif not isinstance(health.get("capabilities"), list):
        checks.append(CheckResult("health_capabilities", False, "capabilities not a list"))
    else:
        checks.append(CheckResult("health", True, "ok"))

    server_caps = health.get("capabilities", []) if isinstance(health, dict) else []
    version = health.get("version", "") if isinstance(health, dict) else ""

    # 2. GET /version
    try:
        status, version_data, ct = _get_json(base_url, "/version", timeout)
        if status != 200:
            checks.append(CheckResult("version_status", False, f"status {status}"))
        elif not isinstance(version_data, dict):
            checks.append(CheckResult("version_json", False, "response not a JSON object"))
        else:
            v_caps = version_data.get("capabilities", [])
            if v_caps != server_caps:
                checks.append(CheckResult("version_capabilities_match", False, "capabilities differ from /health"))
            elif version_data.get("version") != version:
                checks.append(CheckResult("version_match", False, "version differs from /health"))
            else:
                checks.append(CheckResult("version", True, "ok"))
    except ConnectionError as exc:
        checks.append(CheckResult("version", False, f"connection failed: {exc}"))

    commit = version_data.get("commit", "") if isinstance(version_data, dict) else ""

    # 3. Probe each capability the server advertises that we know
    for probe in capabilities.all_probes():
        if probe.name not in server_caps:
            continue

        # Find GET route markers without path parameters
        get_markers = []
        for marker in probe.route_markers:
            # marker is a string like '"/search"' or '"/a2a/messages"'
            # strip quotes
            path = marker.strip('"\'')
            # skip if it's not a path (e.g., "as_of" query parameter)
            if not path.startswith("/"):
                skipped.append(f"{probe.name}:{path}")
                continue
            # skip if it contains path parameter patterns
            if "{" in path or "}" in path:
                skipped.append(f"{probe.name}:{path}")
                continue
            # skip known POST-only routes and non-GET routes
            if path in _POST_ONLY_ROUTES:
                skipped.append(f"{probe.name}:{path}")
                continue
            get_markers.append(path)

        for path in get_markers:
            try:
                status, data, ct = _get_json(base_url, path, timeout)
                if 200 <= status < 300:
                    if _is_json_content_type(ct):
                        checks.append(CheckResult(f"{probe.name}:{path}", True, "ok"))
                    else:
                        checks.append(CheckResult(f"{probe.name}:{path}", False, f"200 but Content-Type={ct!r}, expected application/json"))
                elif 400 <= status < 500:
                    if _is_json_content_type(ct):
                        checks.append(CheckResult(f"{probe.name}:{path}", True, f"ok (HTTP {status} with JSON error)"))
                    else:
                        checks.append(CheckResult(f"{probe.name}:{path}", False, f"HTTP {status} with non-JSON error"))
                else:
                    checks.append(CheckResult(f"{probe.name}:{path}", False, f"HTTP {status}"))
            except ConnectionError as exc:
                checks.append(CheckResult(f"{probe.name}:{path}", False, "connection failed: " + str(exc)))

    # 4. Capabilities we declare but server doesn't advertise
    declared = {p.name for p in capabilities.all_probes()}
    for cap in sorted(declared - set(server_caps)):
        missing_on_server.append(cap)

    # 5. Write probe if requested
    if write_probe:
        agent = f"contract-probe-{secrets.token_hex(4)}"
        thread = f"contract-probe-{secrets.token_hex(4)}"
        marker_word = f"probe-marker-{secrets.token_hex(4)}"

        # POST /ingest
        try:
            status, data, ct = _post_json(base_url, "/ingest", {"text": marker_word, "agent": agent}, timeout)
            if status != 200 or not _is_json_content_type(ct):
                checks.append(CheckResult("write_probe_ingest", False, f"status {status}, ct={ct}"))
            else:
                checks.append(CheckResult("write_probe_ingest", True, "ok"))
        except ConnectionError as exc:
            checks.append(CheckResult("write_probe_ingest", False, f"connection failed: {exc}"))

        # POST /search
        try:
            status, data, ct = _post_json(base_url, "/search", {"query": marker_word, "agent": agent, "limit": 5}, timeout)
            if status != 200 or not _is_json_content_type(ct):
                checks.append(CheckResult("write_probe_search", False, f"status {status}, ct={ct}"))
            else:
                hits = data.get("hits", []) if isinstance(data, dict) else []
                found = any(isinstance(h, dict) and marker_word in h.get("text", "") for h in hits)
                if found:
                    checks.append(CheckResult("write_probe_search", True, "ok"))
                else:
                    checks.append(CheckResult("write_probe_search", False, "marker not found in hits"))
        except ConnectionError as exc:
            checks.append(CheckResult("write_probe_search", False, f"connection failed: {exc}"))

        # POST /a2a/send then GET /a2a/threads/{thread}/messages
        try:
            status, data, ct = _post_json(base_url, "/a2a/send", {"from": agent, "body": marker_word, "thread": thread}, timeout)
            if status != 200 or not _is_json_content_type(ct):
                checks.append(CheckResult("write_probe_a2a_send", False, f"status {status}, ct={ct}"))
            else:
                checks.append(CheckResult("write_probe_a2a_send", True, "ok"))
        except ConnectionError as exc:
            checks.append(CheckResult("write_probe_a2a_send", False, f"connection failed: {exc}"))

        try:
            status, data, ct = _get_json(base_url, f"/a2a/threads/{thread}/messages", timeout)
            if status != 200 or not _is_json_content_type(ct):
                checks.append(CheckResult("write_probe_a2a_read", False, f"status {status}, ct={ct}"))
            else:
                messages = data.get("messages", []) if isinstance(data, dict) else []
                found = any(isinstance(m, dict) and marker_word in m.get("body", "") for m in messages)
                if found:
                    checks.append(CheckResult("write_probe_a2a_read", True, "ok"))
                else:
                    checks.append(CheckResult("write_probe_a2a_read", False, "marker not found in messages"))
        except ConnectionError as exc:
            checks.append(CheckResult("write_probe_a2a_read", False, f"connection failed: {exc}"))

    ok = all(c.ok for c in checks)
    # missing_on_server only fails with --require (handled in main)

    return {
        "ok": ok,
        "base_url": base_url,
        "version": version,
        "commit": commit,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
        "skipped": skipped,
        "missing_on_server": missing_on_server,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="taOSmd contract check")
    parser.add_argument("--base-url", required=True, help="Base URL of the taOSmd server")
    parser.add_argument("--write-probe", action="store_true", help="Perform a real write round-trip")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--require", action="append", default=[], help="Require a capability (fail if missing on server)")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args(argv)

    try:
        result = run(args.base_url, write_probe=args.write_probe, timeout=args.timeout)
    except ConnectionError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": f"connection failed: {exc}"}))
        else:
            print(f"CONNECTION FAILED: {exc}", file=sys.stderr)
        return 2

    # Check --require
    for req in args.require:
        if req in result["missing_on_server"]:
            result["ok"] = False
            result["checks"].append({"name": f"require:{req}", "ok": False, "detail": "required capability missing on server"})

    if args.json:
        print(json.dumps(result))
    else:
        for check in result["checks"]:
            if check["ok"]:
                print(f"PASS {check['name']}")
            else:
                print(f"FAIL {check['name']}: {check['detail']}")
        if result["skipped"]:
            print(f"SKIPPED: {', '.join(result['skipped'])}")
        if result["missing_on_server"]:
            print(f"MISSING ON SERVER: {', '.join(result['missing_on_server'])}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
