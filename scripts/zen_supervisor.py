#!/usr/bin/env python3
"""zen-MCP hang supervisor — SOL-338 Layer 2 OUTER layer (the hard-kill backstop).

WHAT THIS IS
    A thin external process supervisor that the MCP client launches IN PLACE OF
    launching `doppler run ... python server.py` directly. It launches the real
    zen server as a child in its OWN session/process-group, transparently relays
    the MCP stdio JSON-RPC byte stream in both directions (so the MCP client is
    unaware a supervisor exists), and watches the child's stderr activity stream
    as a per-process liveness heartbeat. If the child goes silent for longer than
    a hard outer bound after starting a tool call, the supervisor concludes the
    server is WEDGED, writes a structured kill-record, and KILLPGs the child's
    process group (SIGTERM grace -> SIGKILL). Process death reaps any leaked
    worker thread / socket FD that the in-server asyncio timeout (the INNER layer,
    utils/provider_timeout.py) cannot cancel.

WHY A SUPERVISOR AND NOT JUST THE IN-SERVER asyncio TIMEOUT
    The INNER layer (utils/provider_timeout.py, SOL-338 Layer 2 inner) bounds the
    blocking provider.generate_content() call with asyncio.wait_for and raises a
    TYPED ProviderTimeoutError so the existing cross-family fallback chain fires.
    That handles the COMMON case surgically, with no process restart. But by its
    own design it CANNOT cancel the blocking socket read inside the worker thread
    — that thread + FD leak until the SDK/OS TCP timeout (~15-30 min). Under a
    hang-storm those accumulate. Only process death gives the "whole pgroup
    reaped, no orphan PID" guarantee (QA R1). This supervisor is that backstop. It
    fires ONLY when the inner layer failed to fire or the process is otherwise
    wedged (T_outer > T_inner + grace), so a normal slow-but-recovering call is
    handled by the inner layer without a kill.

INTERPOSITION CHOICE (architecture tension, resolved)
    Agents call zen as in-process MCP stdio tools, not via a CLI — so there is no
    per-call subprocess to wrap. The ONLY subprocess boundary is the zen SERVER
    process itself (the doppler-launched child of the MCP client). Therefore the
    brief's setsid/killpg mandate maps to: kill-and-respawn the SERVER process
    group (interposition (c)), used as a BACKSTOP. The MCP client respawns a fresh
    server on its next call (stdio servers are cheap to respawn). This supervisor
    sits transparently in the stdio path to (1) own the child's session/pgid for a
    clean killpg and (2) own the child's stderr for a per-process liveness signal
    that does NOT depend on the shared logs/mcp_activity.log file (that file is
    written by EVERY server process on the host — it cannot attribute a line to a
    specific PID; the owned stderr pipe can).

AUDITOR-DAEMON SAFETY (QA R4)
    The auditor daemon (`claude --agent audelia`) is the PARENT that launches this
    supervisor, which launches the server child in its own NEW session. killpg
    targets ONLY the server child's process group. The daemon and the supervisor
    live in a DIFFERENT process group, so the daemon survives the child kill. We
    setsid the CHILD, never ourselves, and we NEVER signal our own group.

SECURITY (Selena SEC-R1..R6) — enforced here:
    - Child launched with an argv LIST and shell=False (no shell metachar surface).
    - Secrets are env-only (doppler injects them into the child env); this
      supervisor NEVER places secrets on argv and NEVER serializes os.environ.
    - Kill-records are written 0600, contain ONLY structural keys (no prompt/
      response/provider-secret payload), and the kill-record dir is created 0700.
    - Child is its own session leader (os.setsid in a preexec_fn); we hold the
      Popen handle and killpg(os.getpgid(child.pid)) — never a bare-PID kill,
      never a read-PID-then-sleep-then-signal race. If setsid setup FAILS we FAIL
      CLOSED: we refuse to run rather than fall back to a bare-PID kill (SEC-R2).
    - SIGTERM grace then SIGKILL; we reap the child (waitpid) so no zombie/orphan.
    - The outer timeout is a local config constant with a HARD upper bound; it can
      be tightened by env but NEVER set to infinity from outside (SEC-R5).
    - provider/model/host identifiers that land in the kill-record are validated
      against a strict charset allowlist so an injected `;rm`/`$()`/backtick lands
      as one inert token (SEC-R5). (We never pass them to a shell anyway.)

PER-HOST PARITY (QA R7)
    NOTHING is hardcoded to mario-mbp. The server command, python, doppler config,
    timeouts, and kill-record dir are ALL resolved from env / a config file, with
    portable defaults. There is no dependency on the GNU `timeout` binary — the
    kill mechanism is self-contained (Popen + os.killpg). See resolve_config().

This file is intentionally dependency-free (stdlib only) so it runs identically on
every host (Alan's node, Mac Mini, auditor boxes) under the host's own python.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration (ALL env-resolved; portable defaults; HARD upper bound on T_outer)
# ----------------------------------------------------------------------------

# Inner asyncio timeout (utils/provider_timeout.py) default is 150s. The outer
# hard kill must sit ABOVE inner + an unwind/grace window so we never hard-kill a
# server that is still cleanly unwinding an inner timeout / writing its fallback.
_DEFAULT_INNER_SECS = 150
_DEFAULT_GRACE_SECS = 45            # unwind + fallback-emit window above inner
# T_outer default = inner + grace = 195s. HARD ceiling so it can never be raised
# to "effectively infinite" from outside (SEC-R5).
_OUTER_HARD_CEILING_SECS = 600      # 10 min absolute max, non-overridable upward

# SIGTERM -> SIGKILL escalation window when we do kill (SEC-R4 bounded grace).
_SIGTERM_GRACE_SECS = 5

# Only these characters are allowed in identifiers that reach the kill-record.
_SAFE_IDENT = re.compile(r"[^A-Za-z0-9._:\-/]")

# Activity markers emitted by server.py on its stderr (propagated mcp_activity).
# A TOOL_CALL with no subsequent activity for T_outer == wedged.
_TOOL_CALL_MARK = "TOOL_CALL:"
_TOOL_DONE_MARKS = ("TOOL_COMPLETED:", "CONVERSATION_ERROR:", "Unknown tool:")

# Explicit child-env allowlist (SEC-R3). The child is
# `doppler run ... -- python server.py`; the PROVIDER SECRETS are injected by
# DOPPLER *inside* the child at launch — they are NOT in our inherited env — so
# this allowlist does NOT strip them. It exists to ensure unrelated tokens that
# DO sit in our env (Mercury/QuickBooks/Gmail/Slack/etc.) never reach the child.
# We keep exactly what doppler + python need to launch and authenticate:
#   - DOPPLER_*  : doppler's own token/config (required for secret injection)
#   - PATH/HOME  : locate the doppler+python binaries and the user homedir
#   - USER/LANG/LC_* : benign locale/identity python+doppler may read
#   - TMPDIR/TERM/SHLVL : doppler/python runtime niceties (harmless)
#   - ZEN_*      : our own supervisor/server config knobs
_ENV_ALLOW_EXACT = frozenset({
    "PATH", "HOME", "USER", "LANG", "TMPDIR", "TERM", "SHLVL", "LOGNAME", "PWD",
})
_ENV_ALLOW_PREFIX = ("ZEN_", "DOPPLER_", "LC_")


def _build_child_env() -> dict:
    """Scoped child env (SEC-R3): explicit allowlist, NOT full os.environ passthrough.

    Returns a NEW dict — we never mutate or serialize os.environ. Provider secrets
    are injected by doppler inside the child, so they are intentionally absent here.
    """
    out = {}
    for k, v in os.environ.items():
        if k in _ENV_ALLOW_EXACT or k.startswith(_ENV_ALLOW_PREFIX):
            out[k] = v
    return out


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        sys.stderr.write(f"[zen-supervisor] {name}={raw!r} not an int; using {default}\n")
        return default


def resolve_config() -> dict:
    """Resolve all runtime config from env with portable defaults. Nothing host-pinned.

    ZEN_SERVER_CMD : JSON array of the full child argv (doppler ... python server.py).
                     If unset, we build a default from ZEN_SERVER_DIR + the doppler
                     project/config envs below.
    """
    inner = _env_int("ZEN_PROVIDER_TIMEOUT_SECS", _DEFAULT_INNER_SECS)
    # If the inner timeout is disabled (<=0), fall back to a standalone outer bound.
    if inner <= 0:
        inner = _DEFAULT_INNER_SECS
    grace = _env_int("ZEN_OUTER_GRACE_SECS", _DEFAULT_GRACE_SECS)
    t_outer = _env_int("ZEN_OUTER_TIMEOUT_SECS", inner + grace)
    # HARD ceiling — outer can be tightened but never raised past the ceiling, and
    # never to infinity (SEC-R5). Also never below inner+5 (would false-kill).
    t_outer = max(inner + 5, min(t_outer, _OUTER_HARD_CEILING_SECS))

    repo_root = str(Path(__file__).resolve().parent.parent)  # portable, host-agnostic
    server_dir = os.getenv("ZEN_SERVER_DIR", repo_root)

    cmd_json = os.getenv("ZEN_SERVER_CMD")
    if cmd_json:
        try:
            cmd = json.loads(cmd_json)
            if not isinstance(cmd, list) or not all(isinstance(c, str) for c in cmd):
                raise ValueError("ZEN_SERVER_CMD must be a JSON array of strings")
        except Exception as e:
            sys.stderr.write(f"[zen-supervisor] bad ZEN_SERVER_CMD: {e}\n")
            sys.exit(2)
    else:
        python = os.getenv("ZEN_PYTHON", str(Path(server_dir) / ".zen_venv" / "bin" / "python"))
        doppler = os.getenv("ZEN_DOPPLER_BIN", "doppler")
        project = os.getenv("ZEN_DOPPLER_PROJECT", "miami-ai-lab")
        config = os.getenv("ZEN_DOPPLER_CONFIG", "dev")
        cmd = [
            doppler, "run", "--project", project, "--config", config, "--",
            python, str(Path(server_dir) / "server.py"),
        ]

    # Kill-record dir: defaults under the server repo logs/, but on auditor hosts
    # MUST be overridden to the auditor's own chmod-700 space (SEC-R6). Resolved
    # from env so the auditor lane points it at audit-svc-owned storage.
    kr_dir = os.getenv(
        "ZEN_KILL_RECORD_DIR",
        str(Path(server_dir) / "logs" / "zen-kills"),
    )

    return {
        "cmd": cmd,
        "t_outer": t_outer,
        "inner": inner,
        "kill_record_dir": kr_dir,
        "host": os.getenv("ZEN_HOST_LABEL", os.uname().nodename),
    }


def _safe(s: str | None) -> str:
    """Reduce an identifier to a single inert token (SEC-R5). Never shells out anyway."""
    if not s:
        return "unknown"
    return _SAFE_IDENT.sub("_", s)[:64]


def write_kill_record(cfg: dict, *, provider: str, model: str, elapsed: float, outcome: str) -> None:
    """Structured, redacted, 0600 kill-record (QA R5 / SEC-R4 / SEC-R6).

    Structural keys ONLY — never any prompt/response/auditor-activity payload.
    """
    try:
        d = Path(cfg["kill_record_dir"])
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, 0o700)  # dir owner-only (SEC-R6: no world-readable side channel)
        except OSError as e:
            # Fix-3: do NOT silently pass. A pre-existing world-writable kill-record
            # dir that we cannot tighten is an audit-integrity hole (another user
            # could delete/replace records). Warn loudly; we still write the record
            # (0600 file is created O_EXCL below) rather than crash the supervisor.
            sys.stderr.write(
                f"[zen-supervisor] WARNING: could not chmod 0700 kill-record dir "
                f"{d} ({e}); records may be exposed if the dir is world-writable. "
                f"On auditor hosts this dir MUST be owned by the audit service user.\n"
            )
        rec = {
            "event": "zen_supervisor_kill",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": _safe(cfg["host"]),
            "provider": _safe(provider),
            "model": _safe(model),
            "elapsed_secs": round(elapsed, 2),
            "t_outer_secs": cfg["t_outer"],
            "inner_timeout_secs": cfg["inner"],
            "outcome": _safe(outcome),
            "service_user": _safe(os.getenv("USER") or "unknown"),
        }
        fname = d / f"kill-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        # Write 0600 atomically.
        fd = os.open(str(fname), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(rec, fh)
            fh.write("\n")
    except Exception as e:  # kill-record failure must NOT crash the supervisor
        sys.stderr.write(f"[zen-supervisor] kill-record write failed: {e}\n")


class ServerProcess:
    """Owns the zen server child: own session/pgid, stderr-tail liveness, hard kill."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.proc = None
        self.last_activity = time.monotonic()
        self.in_call = False
        self.last_provider = "unknown"
        self.last_model = "unknown"
        self._lock = threading.Lock()

    # -- launch ---------------------------------------------------------------
    def start(self) -> subprocess.Popen:
        def _preexec():
            # Child becomes leader of a NEW session + process group (SEC-R2).
            # If this FAILS we raise — caller fails CLOSED, never bare-PID kill.
            os.setsid()

        # argv LIST + shell=False (SEC-R5). SCOPED env via explicit allowlist
        # (SEC-R3) — NOT full os.environ passthrough; unrelated tokens
        # (Mercury/QuickBooks/Gmail/Slack) never reach the child. Provider secrets
        # are injected by doppler INSIDE the child, so they are intentionally not
        # in this dict. stdin/stdout are the MCP stdio channel (relayed); stderr is
        # a PIPE WE OWN for per-process liveness (immune to shared mcp_activity.log).
        child_env = _build_child_env()
        self.proc = subprocess.Popen(
            self.cfg["cmd"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_preexec,
            shell=False,
            bufsize=0,
            close_fds=True,
            env=child_env,
        )
        # Verify the child actually got its own pgid; if not, FAIL CLOSED (SEC-R2).
        try:
            child_pgid = os.getpgid(self.proc.pid)
            if child_pgid == os.getpgrp():
                self._hard_kill_unconditional()
                raise RuntimeError("child did not get its own pgid; refusing bare-PID kill path")
        except ProcessLookupError:
            pass  # already exited; handled by the relay loops
        self.last_activity = time.monotonic()
        return self.proc

    # -- liveness from owned stderr ------------------------------------------
    def _touch_activity(self) -> None:
        """Bump last-activity on ANY child output (no marker parsing). Fix-4.

        Called per stderr chunk so liveness reflects raw data flow, decoupled from
        newline framing — a progressing-but-newline-less child is NOT misjudged.
        """
        with self._lock:
            self.last_activity = time.monotonic()

    def note_stderr_line(self, line: str) -> None:
        """Update liveness from a child stderr line; track call boundaries."""
        with self._lock:
            self.last_activity = time.monotonic()
            if _TOOL_CALL_MARK in line:
                self.in_call = True
                # Best-effort provider/model capture is NOT in this line; the
                # inner layer owns provider/model. We keep structural-only here.
            elif any(m in line for m in _TOOL_DONE_MARKS):
                self.in_call = False

    def seconds_since_activity(self) -> float:
        with self._lock:
            return time.monotonic() - self.last_activity

    def is_in_call(self) -> bool:
        with self._lock:
            return self.in_call

    # -- kill -----------------------------------------------------------------
    def hard_kill(self, cfg: dict, *, outcome: str) -> None:
        """Kill the WHOLE process group: SIGTERM grace -> SIGKILL, then reap (R1)."""
        if self.proc is None or self.proc.poll() is not None:
            return  # already exited; no signal to a dead/other pid (SEC-R2)
        elapsed = self.seconds_since_activity()
        write_kill_record(
            cfg, provider=self.last_provider, model=self.last_model,
            elapsed=elapsed, outcome=outcome,
        )
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            return
        # Never signal our own group (would kill the auditor daemon path) — SEC-R2.
        if pgid == os.getpgrp():
            sys.stderr.write("[zen-supervisor] REFUSING kill: child shares our pgid\n")
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + _SIGTERM_GRACE_SECS
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        if self.proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=5)  # reap — no zombie/orphan (SEC-R4)
        except Exception:
            pass

    def _hard_kill_unconditional(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass


_RELAY_CHUNK = 65536


def _pump(src, dst):
    """Relay raw bytes src->dst in CHUNKS until EOF (MCP stdio passthrough).

    Fix-2: the previous read(1) byte-at-a-time loop did a read/write/flush per
    BYTE. With bufsize=0 raw pipes and MB-scale MCP JSON-RPC payloads that is a
    CPU-thrash + throughput collapse that can MANUFACTURE the very timeouts the
    outer layer bounds. We read up to 64KiB per syscall on the raw fd (os.read),
    preserving exact raw-byte passthrough. We write to dst's underlying raw fd via
    os.write in a loop (handles partial writes) so the bytes are passed through
    verbatim with no added buffering/transcoding.
    """
    try:
        src_fd = src.fileno()
        dst_fd = dst.fileno()
    except Exception:
        return
    try:
        while True:
            chunk = os.read(src_fd, _RELAY_CHUNK)
            if not chunk:
                break  # EOF
            # os.write may write fewer bytes than requested — loop until drained.
            mv = memoryview(chunk)
            while mv:
                n = os.write(dst_fd, mv)
                mv = mv[n:]
    except (OSError, ValueError):
        pass  # pipe closed on either side -> relay ends


def main() -> int:
    cfg = resolve_config()

    # Self-check (LOAD-BEARING rollout precondition): the server emits the
    # TOOL_CALL: marker at logging.info, and the watchdog ARMS the outer hard-kill
    # ONLY while is_in_call() is True. is_in_call flips True solely on that
    # TOOL_CALL: marker (note_stderr_line); _touch_activity only refreshes the
    # liveness timestamp, it NEVER sets in_call. So if the host raises LOG_LEVEL
    # above INFO the marker is suppressed -> in_call stays False -> the outer
    # hard-kill NEVER fires: the OUTER backstop is FULLY INERT (the inner asyncio
    # timeout layer still protects each provider call — this is graceful, fail-safe
    # degradation, not a crash). LOG_LEVEL<=INFO is therefore the per-host
    # precondition that ARMS the outer backstop; the R7 per-host smoke asserts it.
    _lvl = (os.getenv("LOG_LEVEL", "DEBUG") or "DEBUG").upper()
    if _lvl in ("WARNING", "ERROR", "CRITICAL"):
        sys.stderr.write(
            f"[zen-supervisor] WARNING: LOG_LEVEL={_lvl} suppresses the INFO TOOL_CALL "
            f"marker; the outer hard-kill backstop will NOT arm (inner asyncio timeout "
            f"still active). Set LOG_LEVEL<=INFO on this host to arm the outer layer.\n"
        )

    server = ServerProcess(cfg)
    try:
        proc = server.start()
    except Exception as e:
        sys.stderr.write(f"[zen-supervisor] FAILED CLOSED at launch: {e}\n")
        return 3

    stop = threading.Event()

    # Relay client stdin -> child stdin, child stdout -> client stdout (raw bytes).
    # Fix-5: when the CLIENT disconnects (EOF on our stdin) the stdin relay returns.
    # We must then (a) close the child's stdin so it sees EOF and shuts down
    # orderly, and (b) wake the watchdog (stop) so the existing shutdown-reap path
    # runs — otherwise a disconnected client leaves an idle server lingering/leaking
    # (the exact leak class we're fixing). We do NOT add a new kill/signal path:
    # closing stdin is orderly, and the already-security-cleared hard_kill() reap at
    # the bottom of main() handles teardown — Selena's SEC-R2 clearance is intact.
    def _stdin_relay():
        try:
            _pump(sys.stdin.buffer, proc.stdin)
        finally:
            try:
                proc.stdin.close()  # propagate EOF to the child for orderly shutdown
            except Exception:
                pass
            stop.set()  # wake the watchdog -> shutdown-reap path runs

    t_in = threading.Thread(target=_stdin_relay, daemon=True)
    t_out = threading.Thread(target=_pump, args=(proc.stdout, sys.stdout.buffer), daemon=True)
    t_in.start()
    t_out.start()

    # Tail child stderr for liveness AND forward it (so the host still sees logs).
    # Fix-4: read in CHUNKS (not readline). readline blocks until a newline; a
    # large newline-less stderr burst fills the pipe buffer -> child blocks on
    # write while we block on read -> deadlock. Chunked os.read updates liveness on
    # ANY data and never waits for a newline. We keep a residual buffer to split
    # complete lines for marker detection (in_call boundaries), and forward EVERY
    # raw byte to our stderr verbatim so host logging is unchanged.
    def _stderr_tail():
        try:
            err_fd = proc.stderr.fileno()
            out_fd = sys.stderr.fileno()
        except Exception:
            stop.set()
            return
        residual = b""
        try:
            while True:
                chunk = os.read(err_fd, _RELAY_CHUNK)
                if not chunk:
                    break  # EOF
                # Liveness updates on ANY data (per byte received, not per line).
                server._touch_activity()
                # Forward raw bytes verbatim to host stderr (handle partial writes).
                mv = memoryview(chunk)
                while mv:
                    n = os.write(out_fd, mv)
                    mv = mv[n:]
                # Split complete lines for marker detection; carry the remainder.
                residual += chunk
                while b"\n" in residual:
                    raw_line, residual = residual.split(b"\n", 1)
                    try:
                        line = raw_line.decode("utf-8", "replace")
                    except Exception:
                        line = ""
                    server.note_stderr_line(line)
                # Bound the residual so a pathological newline-less stream can't
                # grow it without limit; scan the tail for markers, then trim.
                if len(residual) > 2 * _RELAY_CHUNK:
                    try:
                        server.note_stderr_line(residual.decode("utf-8", "replace"))
                    except Exception:
                        pass
                    residual = residual[-_RELAY_CHUNK:]
        except (OSError, ValueError):
            pass
        finally:
            stop.set()

    t_err = threading.Thread(target=_stderr_tail, daemon=True)
    t_err.start()

    # Watchdog: if a TOOL_CALL is in flight and the child has been SILENT on every
    # stream for >= T_outer, it is wedged -> hard kill the group, then exit so the
    # MCP client respawns a fresh server on its next call.
    exit_code = 0
    while not stop.is_set():
        if proc.poll() is not None:
            break  # child exited on its own
        if server.is_in_call() and server.seconds_since_activity() >= cfg["t_outer"]:
            sys.stderr.write(
                f"[zen-supervisor] WEDGED: no activity {server.seconds_since_activity():.0f}s "
                f">= T_outer {cfg['t_outer']}s during a tool call. Hard-killing server group.\n"
            )
            server.hard_kill(cfg, outcome="hard_kill_wedged")
            exit_code = 137  # convey SIGKILL-class outcome to the client
            break
        time.sleep(1.0)

    # Ensure no orphan on any exit path.
    server.hard_kill(cfg, outcome="shutdown_reap")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
