#!/usr/bin/env bash
# rebootwin.sh -- convenience wrapper over the taosmd bench pause-flag mechanism
#
# Subcommands:
#   pause   [--flag PATH]  Touch the pause flag and wait for the runner to exit cleanly.
#   status  [--flag PATH]  Report flag state, runner state, and newest sidecar.
#   resume                 Clear the flag and print instructions to re-run with --resume.
#
# This script never starts or kills the benchmark runner. It only manages the
# pause flag and reports process state. The runner checks the flag between
# conversations and exits 3 when it finds the flag present.
#
# Safe to run from any directory; uses absolute detection logic.

set -u

DEFAULT_FLAG="/tmp/taosmd-bench-pause"
RUNNER_PATTERN="locomo_runner.py"
PAUSE_TIMEOUT_SECONDS=1800  # 30 minutes; a single conversation should never take this long

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") <subcommand> [options]

Subcommands:
  pause   [--flag PATH]   Touch the pause flag and wait for the runner to exit.
                          Exits 0 on clean pause, 0 if no runner was running,
                          exits 1 on timeout (runner still alive after 30 min).
  status  [--flag PATH]   Report flag state, runner state, and newest sidecar info.
  resume                  Clear the pause flag and print resume instructions.

Options:
  --flag PATH   Override the default pause-flag path ($DEFAULT_FLAG).
EOF
    exit 2
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

runner_pid() {
    # Return the PID(s) of any running locomo_runner.py process, or empty string.
    pgrep -f "$RUNNER_PATTERN" 2>/dev/null || true
}

runner_alive() {
    local pid
    pid="$(runner_pid)"
    [ -n "$pid" ]
}

newest_sidecar() {
    # Print the path of the newest .ckpt.jsonl under benchmarks/results/, or empty.
    # Works regardless of cwd by searching relative paths from the script location.
    local script_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    local results_dir="$script_dir/results"
    if [ ! -d "$results_dir" ]; then
        return 0
    fi
    # Find the most recently modified sidecar file.
    find "$results_dir" -name "*.ckpt.jsonl" -maxdepth 1 2>/dev/null \
        | xargs ls -t 2>/dev/null \
        | head -1
}

conv_count() {
    # Count completed conversations in a sidecar. Handles both spacing variants.
    local path="$1"
    local n=0
    if [ -f "$path" ]; then
        # grep -c returns a non-zero exit code when nothing matches; treat that as 0.
        n=$(grep -c '"kind": "conv"' "$path" 2>/dev/null || true)
        if [ "$n" -eq 0 ]; then
            n=$(grep -c '"kind":"conv"' "$path" 2>/dev/null || true)
        fi
    fi
    echo "$n"
}

# ---------------------------------------------------------------------------
# Subcommand: pause
# ---------------------------------------------------------------------------

cmd_pause() {
    local flag_path="$DEFAULT_FLAG"

    while [ $# -gt 0 ]; do
        case "$1" in
            --flag) flag_path="$2"; shift 2 ;;
            *) echo "unknown option: $1"; usage ;;
        esac
    done

    if ! runner_alive; then
        echo "no benchmark running"
        exit 0
    fi

    # Touch the flag to signal the runner.
    touch "$flag_path"
    echo "pause flag set at $flag_path"
    echo "waiting for the current conversation to finish..."

    local elapsed=0
    while runner_alive; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$PAUSE_TIMEOUT_SECONDS" ]; then
            echo ""
            echo "WARNING: runner is still alive after ${PAUSE_TIMEOUT_SECONDS}s."
            echo "Something is wrong; investigate before rebooting."
            echo "Do NOT kill the process -- that would lose the in-flight conversation."
            exit 1
        fi
    done

    echo "bench paused cleanly; safe to reboot"
    exit 0
}

# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

cmd_status() {
    local flag_path="$DEFAULT_FLAG"

    while [ $# -gt 0 ]; do
        case "$1" in
            --flag) flag_path="$2"; shift 2 ;;
            *) echo "unknown option: $1"; usage ;;
        esac
    done

    # Flag state.
    if [ -e "$flag_path" ]; then
        echo "pause flag:  PRESENT ($flag_path)"
    else
        echo "pause flag:  not present ($flag_path)"
    fi

    # Runner state.
    local pid
    pid="$(runner_pid)"
    if [ -n "$pid" ]; then
        echo "runner:      RUNNING (pid $pid)"
    else
        echo "runner:      not running"
    fi

    # Newest sidecar.
    local sidecar
    sidecar="$(newest_sidecar)"
    if [ -n "$sidecar" ]; then
        local n
        n="$(conv_count "$sidecar")"
        echo "newest sidecar: $sidecar ($n completed conversations)"
    else
        echo "newest sidecar: none found in benchmarks/results/"
    fi

    exit 0
}

# ---------------------------------------------------------------------------
# Subcommand: resume
# ---------------------------------------------------------------------------

cmd_resume() {
    local flag_path="$DEFAULT_FLAG"

    # Remove the flag if present.
    if [ -e "$flag_path" ]; then
        rm -f "$flag_path"
        echo "pause flag cleared ($flag_path)"
    else
        echo "pause flag was not present ($flag_path)"
    fi

    echo ""
    echo "Re-run your original launch command with --resume appended (same --out and same recipe flags)."
    echo ""
    echo "Example:"
    echo "  python3 benchmarks/locomo_runner.py [your original flags] --resume"
    echo ""
    echo "The config hash must match; if you changed any flags, start a fresh run with a new --out path."
    exit 0
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if [ $# -eq 0 ]; then
    usage
fi

subcommand="$1"
shift

case "$subcommand" in
    pause)  cmd_pause "$@" ;;
    status) cmd_status "$@" ;;
    resume) cmd_resume ;;
    *)      echo "unknown subcommand: $subcommand"; usage ;;
esac
