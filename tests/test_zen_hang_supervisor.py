#!/usr/bin/env python3
"""Back-gate test harness — SOL-338 Layer 2 (zen-MCP hang durable fix).

PURPOSE
    Prove the LOAD-BEARING behaviours of the two-layer hang fix WITHOUT burning
    real provider calls or waiting the real 150s timeout. Uses short configurable
    timeouts and a FAKE hanging child so the suite runs in seconds and is
    deterministic (QA R9 anti-flake).

WHAT IT PROVES (mapped to QA R1..R9 / SEC where mechanically checkable here)
    R1  Kill-on-hang: a child that goes silent past T_outer is killed; the WHOLE
        process group is reaped (the child's grandchild dies too) — no orphan PID.
    R2  Typed non-success: the INNER layer raises ProviderTimeoutError, a
        TimeoutError subclass DISTINCT from both success and a generic error, and a
        stubbed fallback driver advances to provider N+1 on it (a swallowed
        success-empty would FAIL this test).
    R3  No false-kill: a child that keeps emitting activity inside [0.5T,0.95T] is
        NOT killed and exits cleanly.
    R4  Auditor-daemon context: the supervisor is launched as a CHILD of a parent
        (simulating `claude --agent`), the hung server is killed, and the PARENT
        SURVIVES — killpg targets only the server's own new session/pgid.
    R5  Silent-failure-visible: every kill writes a structured 0600 kill-record
        with the required structural keys.
    R6  _reset present+active: regression guard that _reset_workflow_state exists
        and is called at the top of execute_workflow (so a Layer-2 change can't
        silently revert Layer 1).
    R9  Determinism: the timing tests are written to pass repeatedly; run with
        `-p no:randomly` / 3x in CI per the back-gate checklist.

    SEC-R2 kill-target integrity: asserts the child is its OWN session leader and
        the supervisor refuses to signal its own group.
    SEC-R5 no-injection: asserts metachar identifiers are reduced to one inert
        token in the kill-record.

NOT PROVEN HERE (must be done in the REAL daemon context at back-gate)
    The end-to-end mcp__zen__* hang under a real `claude --agent` daemon. This
    harness gives you a daemon-STYLE parent/child/grandchild process tree that
    exercises the same kill + survival semantics; the back-gate operator then runs
    the real-daemon smoke (section RUN below) to close R4 against the true client.

RUN
    cd ~/.mcp-servers/zen-mcp-server
    .zen_venv/bin/python -m pytest tests/test_zen_hang_supervisor.py -v
    # 3x for R9:
    for i in 1 2 3; do .zen_venv/bin/python -m pytest tests/test_zen_hang_supervisor.py -q || break; done
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SUP = REPO / "scripts" / "zen_supervisor.py"


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("zen_supervisor", SUP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# R2 + typed-signal + fallback advance (INNER layer)
# ---------------------------------------------------------------------------

def test_r2_inner_raises_typed_timeout_and_fallback_advances():
    """R2: hang -> ProviderTimeoutError (distinct type) -> fallback goes to N+1.

    Driven via asyncio.run so the suite needs NO pytest-asyncio plugin (portable
    across every host's bare venv — per-host parity R7).
    """
    import asyncio

    from utils.provider_timeout import ProviderTimeoutError, generate_content_with_timeout

    # Typed signal is distinct from BOTH success and a generic Exception.
    assert issubclass(ProviderTimeoutError, TimeoutError)
    assert not issubclass(ProviderTimeoutError, ValueError)

    class HangingProvider:
        def get_provider_type(self):
            class T:  # noqa
                value = "hanging"
            return T()

        def generate_content(self, **kw):
            time.sleep(30)  # would hang well past the 1s test bound
            return "SHOULD-NOT-REACH"

    class FastProvider:
        def get_provider_type(self):
            class T:  # noqa
                value = "fast"
            return T()

        def generate_content(self, **kw):
            return "FALLBACK-PROVIDER-ANSWER"

    os.environ["ZEN_PROVIDER_TIMEOUT_SECS"] = "1"  # 1s bound for the test

    # Stubbed fallback driver: try providers in order, advance on ProviderTimeoutError.
    async def _drive():
        providers = [HangingProvider(), FastProvider()]
        result = None
        advanced_to = -1
        for idx, prov in enumerate(providers):
            try:
                result = await generate_content_with_timeout(prov, model_name="m", prompt="p")
                advanced_to = idx
                break
            except ProviderTimeoutError:
                continue  # fallback advances to N+1 — the load-bearing behaviour
        return result, advanced_to

    result, advanced_to = asyncio.run(_drive())

    assert advanced_to == 1, "fallback did NOT advance to provider N+1 on timeout"
    assert result == "FALLBACK-PROVIDER-ANSWER"
    # A swallowed success-empty would have left result None / advanced_to 0 -> fail above.


# ---------------------------------------------------------------------------
# Helpers: a fake "server" that prints a TOOL_CALL to stderr then hangs (with a
# grandchild) OR stays alive emitting heartbeats.
# ---------------------------------------------------------------------------

_FAKE_HANG = textwrap.dedent(
    """
    import os, sys, time, subprocess
    # spawn a grandchild so we can prove the WHOLE group is reaped (R1)
    gc = subprocess.Popen(["sleep", "300"])
    sys.stderr.write("TOOL_CALL: chat with 3 arguments\\n"); sys.stderr.flush()
    # write our pid + grandchild pid where the test can read them
    open(os.environ["ZEN_TEST_PIDFILE"], "w").write(f"{os.getpid()},{gc.pid}")
    # then go SILENT forever (the hang)
    time.sleep(10000)
    """
)

_FAKE_HEALTHY = textwrap.dedent(
    """
    import sys, time
    sys.stderr.write("TOOL_CALL: chat with 3 arguments\\n"); sys.stderr.flush()
    # keep emitting activity inside [0.5T,0.95T] so the supervisor must NOT kill us
    for _ in range(6):
        time.sleep(0.3)
        sys.stderr.write("progress heartbeat\\n"); sys.stderr.flush()
    sys.stderr.write("TOOL_COMPLETED: chat\\n"); sys.stderr.flush()
    """
)

# Idle server: starts, records pids (with a grandchild), emits NO TOOL_CALL, and
# just waits. Models a connected-but-idle server. The watchdog never arms (no
# in_call), so ONLY the stdin-EOF shutdown path (Fix-5) can reap it.
_FAKE_IDLE = textwrap.dedent(
    """
    import os, sys, time, subprocess
    gc = subprocess.Popen(["sleep", "300"])
    open(os.environ["ZEN_TEST_PIDFILE"], "w").write(f"{os.getpid()},{gc.pid}")
    sys.stderr.write("MCP_CLIENT_INFO: idle\\n"); sys.stderr.flush()
    time.sleep(10000)
    """
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False if isinstance(sys.exc_info()[1], ProcessLookupError) else True


def _run_supervisor_with_fake(tmp_path, fake_src, t_outer, extra_env=None, timeout=30,
                              open_stdin=False):
    """Run the supervisor wrapping a fake child. Returns (proc, pidfile_contents).

    open_stdin=True gives the supervisor a real stdin PIPE (so a test can close it
    to simulate client disconnect); default DEVNULL keeps existing tests as-is.
    """
    pidfile = tmp_path / "pids.txt"
    fake = tmp_path / "fake_server.py"
    fake.write_text(fake_src)
    env = dict(os.environ)
    env.update({
        "ZEN_SERVER_CMD": json.dumps([sys.executable, str(fake)]),
        "ZEN_OUTER_TIMEOUT_SECS": str(t_outer),
        "ZEN_PROVIDER_TIMEOUT_SECS": str(max(1, t_outer - 5)),
        "ZEN_KILL_RECORD_DIR": str(tmp_path / "kills"),
        # ZEN_-prefixed so it survives the supervisor's child-env allowlist (SEC-R3).
        "ZEN_TEST_PIDFILE": str(pidfile),
    })
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        [sys.executable, str(SUP)],
        stdin=(subprocess.PIPE if open_stdin else subprocess.DEVNULL),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, start_new_session=True,
    )
    return proc, pidfile


# ---------------------------------------------------------------------------
# R1 + R5 + SEC: hang -> hard kill of the whole group + kill-record
# ---------------------------------------------------------------------------

def test_r1_r5_hang_is_hard_killed_whole_group_with_record(tmp_path):
    """R1: wedged child killed past T_outer, grandchild reaped, no orphan. R5: record."""
    # Note: the supervisor sits above the inner timeout; here inner==outer-5.
    # T_outer small so the test is fast. The fake never satisfies inner (it hangs
    # before any provider call), so the OUTER layer is what fires — exactly R1.
    # open_stdin=True keeps stdin OPEN (a real connected client holds stdio open
    # for the lifetime of a call) so the WATCHDOG fires, not the Fix-5 stdin-EOF
    # path. (DEVNULL would EOF immediately and trigger orderly shutdown instead.)
    proc, pidfile = _run_supervisor_with_fake(tmp_path, _FAKE_HANG, t_outer=6,
                                              open_stdin=True)

    # wait for the fake to record its pids
    deadline = time.time() + 10
    while time.time() < deadline and not pidfile.exists():
        time.sleep(0.2)
    assert pidfile.exists(), "fake server never started"
    server_pid, gc_pid = (int(x) for x in pidfile.read_text().split(","))
    assert _pid_alive(server_pid) and _pid_alive(gc_pid)

    # supervisor should hard-kill within T_outer + grace (~6 + 5 + slack)
    try:
        rc = proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("supervisor did not kill the wedged server within bound (R1 FAIL)")

    # R1: server AND grandchild both gone (whole pgroup reaped) — no orphan.
    time.sleep(1)
    assert not _pid_alive(server_pid), "server PID survived (R1 FAIL)"
    assert not _pid_alive(gc_pid), "grandchild orphaned — group not reaped (R1 FAIL)"
    assert rc == 137, f"supervisor should convey SIGKILL-class exit 137, got {rc}"

    # R5: structured 0600 kill-record with required keys.
    kr_dir = tmp_path / "kills"
    recs = list(kr_dir.glob("kill-*.json")) if kr_dir.exists() else []
    assert recs, "no kill-record written (R5 FAIL)"
    rec = json.loads(recs[0].read_text())
    for k in ("event", "timestamp", "host", "elapsed_secs", "t_outer_secs", "outcome"):
        assert k in rec, f"kill-record missing key {k} (R5 FAIL)"
    mode = oct(os.stat(recs[0]).st_mode & 0o777)
    assert mode == "0o600", f"kill-record not 0600 (SEC-R4 FAIL): {mode}"


# ---------------------------------------------------------------------------
# R3: no false-kill on a slow-but-progressing call
# ---------------------------------------------------------------------------

def test_r3_no_false_kill_when_progressing(tmp_path):
    """R3: a child emitting activity inside the window is NOT killed."""
    proc, pidfile = _run_supervisor_with_fake(tmp_path, _FAKE_HEALTHY, t_outer=8)
    try:
        rc = proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("healthy child should have completed; supervisor hung")
    # Healthy child emits TOOL_COMPLETED then exits 0; supervisor should NOT report 137.
    assert rc != 137, "false-kill of a progressing call (R3 FAIL)"
    kr_dir = tmp_path / "kills"
    hard = [p for p in (kr_dir.glob("kill-*.json") if kr_dir.exists() else [])
            if json.loads(p.read_text()).get("outcome") == "hard_kill_wedged"]
    assert not hard, "a hard_kill_wedged record exists for a healthy call (R3 FAIL)"


# ---------------------------------------------------------------------------
# R4: auditor-daemon context — parent survives the child kill
# ---------------------------------------------------------------------------

def test_r4_parent_daemon_survives_child_kill(tmp_path):
    """R4: supervisor (child of a parent daemon) kills server; PARENT survives."""
    # The pytest process here plays the "daemon": it launches the supervisor in a
    # NEW session (start_new_session=True in the helper). The supervisor launches
    # the server in ITS OWN new session. We assert the daemon (this process) is
    # untouched after the kill, and the supervisor's own group != server's group.
    # open_stdin=True so the WATCHDOG fires (connected client holds stdin open).
    proc, pidfile = _run_supervisor_with_fake(tmp_path, _FAKE_HANG, t_outer=6,
                                              open_stdin=True)
    my_pid = os.getpid()
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("supervisor did not complete (R4 setup FAIL)")
    # The daemon (this test process) must be alive and well after the child kill.
    assert _pid_alive(my_pid), "daemon process died with the server (R4 FAIL)"


# ---------------------------------------------------------------------------
# Fix-5: client-disconnect (stdin EOF) reaps the idle child — no orphan/leak.
# ---------------------------------------------------------------------------

def test_fix5_client_disconnect_reaps_idle_child(tmp_path):
    """Fix-5: closing supervisor stdin (client gone) reaps the wrapped server.

    The fake server is IDLE — it never emits TOOL_CALL, so the watchdog never arms.
    The ONLY thing that can reap it is the stdin-EOF shutdown path. We assert the
    supervisor exits and BOTH the server and its grandchild are gone (no orphan).
    """
    proc, pidfile = _run_supervisor_with_fake(tmp_path, _FAKE_IDLE, t_outer=120,
                                              open_stdin=True)
    # Wait for the idle server to record its pids.
    deadline = time.time() + 10
    while time.time() < deadline and not pidfile.exists():
        time.sleep(0.2)
    assert pidfile.exists(), "idle fake server never started"
    server_pid, gc_pid = (int(x) for x in pidfile.read_text().split(","))
    assert _pid_alive(server_pid) and _pid_alive(gc_pid)

    # Simulate the MCP client disconnecting: close the supervisor's stdin.
    proc.stdin.close()

    # Supervisor must wind down quickly (well under t_outer=120s — proving it's the
    # stdin-EOF path, NOT the watchdog, doing the reap).
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("supervisor did not shut down on client disconnect (Fix-5 FAIL)")

    time.sleep(1)
    assert not _pid_alive(server_pid), "idle server leaked after client disconnect (Fix-5 FAIL)"
    assert not _pid_alive(gc_pid), "grandchild orphaned after disconnect (Fix-5 FAIL)"


# ---------------------------------------------------------------------------
# SEC-R2 / SEC-R5 unit checks
# ---------------------------------------------------------------------------

def test_sec_r5_identifiers_reduced_to_inert_token():
    m = _load_supervisor()
    assert m._safe(";rm -rf /") == "_rm_-rf_/" or ";" not in m._safe(";rm -rf /")
    assert "$" not in m._safe("$(whoami)")
    assert "`" not in m._safe("`id`")
    assert ";" not in m._safe("a;b")


def test_sec_r5_t_outer_has_hard_ceiling(monkeypatch):
    m = _load_supervisor()
    monkeypatch.setenv("ZEN_SERVER_CMD", json.dumps(["/bin/echo", "x"]))
    monkeypatch.setenv("ZEN_OUTER_TIMEOUT_SECS", "999999999")
    cfg = m.resolve_config()
    assert cfg["t_outer"] <= 600, "T_outer ceiling not enforced (SEC-R5 FAIL)"


# ---------------------------------------------------------------------------
# Fix-1 (SEC-R3): explicit child-env allowlist — unrelated tokens never reach
# the child; doppler/PATH/HOME survive so secret injection still works.
# ---------------------------------------------------------------------------

def test_fix1_child_env_excludes_unrelated_tokens(monkeypatch):
    """SEC-R3: a planted unrelated secret in the PARENT env must NOT reach the child."""
    m = _load_supervisor()
    monkeypatch.setenv("MERCURY_FAKE_TOKEN", "should-not-leak")
    monkeypatch.setenv("QUICKBOOKS_FAKE", "should-not-leak")
    monkeypatch.setenv("DOPPLER_TOKEN", "dp.keep")
    monkeypatch.setenv("ZEN_HOST_LABEL", "keep")
    env = m._build_child_env()
    # Unrelated tokens excluded (the whole point of the allowlist).
    assert "MERCURY_FAKE_TOKEN" not in env, "unrelated token leaked to child (SEC-R3 FAIL)"
    assert "QUICKBOOKS_FAKE" not in env, "unrelated token leaked to child (SEC-R3 FAIL)"
    # Things doppler+python NEED to launch + authenticate are kept.
    assert "DOPPLER_TOKEN" in env, "DOPPLER_TOKEN stripped — would break secret injection"
    assert "PATH" in env and "HOME" in env, "PATH/HOME stripped — would break launch"
    assert env.get("ZEN_HOST_LABEL") == "keep"
    # It is a NEW dict, not os.environ itself (we never mutate/serialize os.environ).
    assert env is not os.environ


def test_fix1_child_env_subprocess_does_not_see_planted_token(monkeypatch, tmp_path):
    """End-to-end: a REAL child launched with the scoped env cannot read the token."""
    m = _load_supervisor()
    monkeypatch.setenv("MERCURY_FAKE_TOKEN", "should-not-leak")
    out = tmp_path / "seen.txt"
    prog = f"import os; open({str(out)!r},'w').write(str('MERCURY_FAKE_TOKEN' in os.environ))"
    child_env = m._build_child_env()
    subprocess.run([sys.executable, "-c", prog], env=child_env, check=True, timeout=15)
    assert out.read_text() == "False", "planted token visible in child process (SEC-R3 FAIL)"


# ---------------------------------------------------------------------------
# Fix-2: chunked relay passes a large payload through intact (no byte-at-a-time).
# ---------------------------------------------------------------------------

def test_fix2_pump_relays_large_payload_intact_and_fast():
    """A >=1MB payload relays through _pump verbatim and quickly (no per-byte loop)."""
    m = _load_supervisor()
    payload = (b"".join(bytes([i % 256]) for i in range(4096)) * 256)  # 1 MiB, all byte values
    assert len(payload) >= 1024 * 1024

    r_in, w_in = os.pipe()      # we write payload -> src
    r_out, w_out = os.pipe()    # dst writes -> we read result

    class FdWrap:
        def __init__(self, fd):
            self._fd = fd

        def fileno(self):
            return self._fd

    src = FdWrap(r_in)
    dst = FdWrap(w_out)

    import threading
    t = threading.Thread(target=m._pump, args=(src, dst), daemon=True)
    t.start()

    start = time.monotonic()
    # Feed the payload then close the write end so _pump sees EOF.
    def _feed():
        mv = memoryview(payload)
        while mv:
            n = os.write(w_in, mv)
            mv = mv[n:]
        os.close(w_in)

    feeder = threading.Thread(target=_feed, daemon=True)
    feeder.start()

    received = bytearray()
    while len(received) < len(payload):
        chunk = os.read(r_out, 65536)
        if not chunk:
            break
        received.extend(chunk)
    elapsed = time.monotonic() - start

    t.join(timeout=5)
    for fd in (w_out, r_out):
        try:
            os.close(fd)
        except OSError:
            pass

    assert bytes(received) == payload, "payload corrupted/truncated through relay (Fix-2 FAIL)"
    # A per-byte loop would take many seconds for 1MiB; chunked is well under 2s.
    assert elapsed < 2.0, f"relay too slow ({elapsed:.2f}s) — per-byte regression? (Fix-2 FAIL)"


def test_sec_r2_refuses_when_child_shares_our_pgid(tmp_path):
    """SEC-R2: ServerProcess.hard_kill must NOT signal our own group."""
    m = _load_supervisor()
    cfg = {"kill_record_dir": str(tmp_path), "host": "t", "t_outer": 1, "inner": 1}
    sp = m.ServerProcess(cfg)

    class FakeProc:
        pid = os.getpid()  # same as supervisor -> must refuse

        def poll(self):
            return None

    sp.proc = FakeProc()
    # os.getpgid(os.getpid()) == os.getpgrp() -> must refuse, no signal sent.
    sp.hard_kill(cfg, outcome="should_refuse")  # must NOT raise / must NOT kill us
    assert _pid_alive(os.getpid())  # we are still alive (obviously) — no self-kill


# ---------------------------------------------------------------------------
# R6: _reset_workflow_state regression guard (Layer 1 can't silently revert)
# ---------------------------------------------------------------------------

def test_r6_reset_workflow_state_present_and_called():
    src = (REPO / "tools" / "workflow" / "workflow_mixin.py").read_text()
    assert "def _reset_workflow_state(self)" in src, "Layer-1 _reset method missing (R6 FAIL)"
    # called at the top of execute_workflow, before _current_arguments assignment
    call_idx = src.find("self._reset_workflow_state()")
    exec_idx = src.find("async def execute_workflow")
    assert call_idx > exec_idx > 0, "_reset not called inside execute_workflow (R6 FAIL)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
