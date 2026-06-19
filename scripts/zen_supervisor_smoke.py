#!/usr/bin/env python3
"""Per-host rollout smoke for the zen-MCP hang supervisor — SOL-338 Layer 2.

Run this AFTER a host has pulled the merged branch and BEFORE trusting the
wrapped zen stanza in production. It exercises the OUTER supervisor against a
forced-hang child, with a low T_outer, so it finishes in seconds and burns NO
real provider spend. It is the operator-side close for Qualyn's R4 (real-daemon
context) and a fast confidence check that this host's process semantics behave.

WHAT IT ASSERTS (deterministic, fast)
    (a) Kill-on-hang: a wedged tool call's whole server process group is hard
        killed (server + any grandchild reaped; no orphan PID).
    (b) Parent survives: the supervisor's PARENT (this script — standing in for the
        `claude --agent` daemon) is alive and well after the child kill (R4).
    (c) Marker visible: `TOOL_CALL:` actually appears on the child's stderr DURING
        the call (this is the liveness/arming signal the watchdog depends on — if a
        host suppresses it via LOG_LEVEL, the outer backstop would be inert).

It does NOT call a real model. The "server" here is a tiny forced-hang stub that
emits the same TOOL_CALL: marker the real server emits, then goes silent — which
is exactly the wedged-call shape the supervisor must catch.

USAGE
    <host-python> scripts/zen_supervisor_smoke.py
    # exit 0 + "SMOKE PASS" => this host's supervisor semantics are good.
    # non-zero + "SMOKE FAIL: <reason>" => do NOT roll the wrapped stanza here.

This is host-agnostic: it locates the supervisor relative to its own path and
uses the host's own python (sys.executable). No mario-mbp hardcoding.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

SUPERVISOR = Path(__file__).resolve().parent / "zen_supervisor.py"

# Forced-hang stub standing in for server.py: emits the real TOOL_CALL: marker,
# spawns a grandchild (to prove whole-group reaping), records pids, then hangs.
_HANG_STUB = textwrap.dedent(
    """
    import os, sys, time, subprocess
    gc = subprocess.Popen(["sleep", "300"])
    open(os.environ["ZEN_TEST_PIDFILE"], "w").write(f"{os.getpid()},{gc.pid}")
    sys.stderr.write("TOOL_CALL: chat with 2 arguments\\n"); sys.stderr.flush()
    time.sleep(10000)  # wedged
    """
)

T_OUTER = 6  # seconds — low so the smoke is fast; production default is 195


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> int:
    if not SUPERVISOR.exists():
        print(f"SMOKE FAIL: supervisor not found at {SUPERVISOR}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        stub = td / "hang_stub.py"
        stub.write_text(_HANG_STUB)
        pidfile = td / "pids.txt"

        env = dict(os.environ)
        env.update({
            "ZEN_SERVER_CMD": json.dumps([sys.executable, str(stub)]),
            "ZEN_OUTER_TIMEOUT_SECS": str(T_OUTER),
            "ZEN_PROVIDER_TIMEOUT_SECS": "1",
            "ZEN_KILL_RECORD_DIR": str(td / "kills"),
            "ZEN_TEST_PIDFILE": str(pidfile),  # ZEN_-prefixed: survives the env allowlist
        })

        # Launch the supervisor with an OPEN stdin (a connected MCP client holds
        # stdio open for a call's lifetime) so the WATCHDOG is what fires, not the
        # stdin-EOF shutdown path. We are the PARENT/daemon stand-in.
        proc = subprocess.Popen(
            [sys.executable, str(SUPERVISOR)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, start_new_session=True,
        )
        my_pid = os.getpid()

        # Wait for the stub to record its pids.
        deadline = time.time() + 10
        while time.time() < deadline and not pidfile.exists():
            time.sleep(0.2)
        if not pidfile.exists():
            proc.kill()
            print("SMOKE FAIL: wrapped server never started", file=sys.stderr)
            return 1
        server_pid, gc_pid = (int(x) for x in pidfile.read_text().split(","))

        # Wait for the supervisor to detect the wedge and hard-kill (T_outer + grace).
        try:
            rc = proc.wait(timeout=T_OUTER + 20)
        except subprocess.TimeoutExpired:
            proc.kill()
            print("SMOKE FAIL: supervisor did not kill the wedged server in bound", file=sys.stderr)
            return 1
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""

        time.sleep(1)
        # (a) kill-on-hang: server + grandchild gone, no orphan.
        if _alive(server_pid):
            print("SMOKE FAIL: server PID survived (no kill-on-hang)", file=sys.stderr)
            return 1
        if _alive(gc_pid):
            print("SMOKE FAIL: grandchild orphaned (group not reaped)", file=sys.stderr)
            return 1
        # (b) parent/daemon survives.
        if not _alive(my_pid):
            print("SMOKE FAIL: parent daemon died with the server", file=sys.stderr)
            return 1
        # (c) the WEDGED detection actually fired (supervisor saw the marker + silence).
        if "WEDGED" not in err:
            print("SMOKE FAIL: supervisor never reported WEDGED (marker/arming problem)", file=sys.stderr)
            print(f"--- supervisor stderr ---\n{err}", file=sys.stderr)
            return 1
        if rc != 137:
            print(f"SMOKE FAIL: expected SIGKILL-class exit 137, got {rc}", file=sys.stderr)
            return 1

        print("SMOKE PASS: kill-on-hang OK, whole group reaped, parent survived, "
              "TOOL_CALL marker seen + WEDGED fired, exit 137.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
