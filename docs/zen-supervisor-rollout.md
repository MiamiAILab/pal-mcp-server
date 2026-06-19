# zen-MCP hang supervisor — per-host rollout (SOL-338 Layer 2)

Turnkey steps to route a host's zen MCP calls **through** the hang supervisor
after the merged branch is pulled. The supervisor (`scripts/zen_supervisor.py`)
launches the real zen server in its own process group, hard-kills it if a tool
call wedges, and reaps it cleanly on client disconnect — so a hung provider call
becomes a bounded, recoverable event instead of an indefinite hang.

> **This is rollout tooling. Apply it per host, deliberately, with a backup.**
> Changing the stanza perturbs the live zen connection — the MCP client must be
> restarted / the session reopened for it to take effect.

Everything here is **config-driven and host-agnostic** — no host path is
hardcoded. Each host substitutes its own `<repo>` and `<python>`.

---

## 0. Prerequisites (every host)

- The host has pulled the merged branch (contains `scripts/zen_supervisor.py`,
  `scripts/zen_supervisor_smoke.py`, `scripts/zen_check_log_level.py`,
  `utils/provider_timeout.py`).
- `<repo>` = absolute path to this repo on the host (e.g. the dir containing
  `server.py`).
- `<python>` = the host's zen venv python (e.g. `<repo>/.zen_venv/bin/python`).
- `doppler` is on PATH and authenticated for project `miami-ai-lab` (the
  supervisor's env allowlist keeps `PATH` + `DOPPLER_*` so injection still works;
  provider secrets are injected by doppler **inside** the child, never passed
  through from the parent).

---

## 1. Install step — wrap the zen stanza

The supervisor itself runs under **plain python** (NOT doppler) — it must not hold
provider secrets; doppler injects them into the *child* the supervisor spawns.

### 1a. Back up the client config FIRST

```bash
cp ~/.claude.json ~/.claude.json.bak.$(date +%Y%m%d-%H%M%S)
```

### 1b. Edit the `mcpServers.zen` stanza

**BEFORE** (current, unwrapped):

```json
"zen": {
  "type": "stdio",
  "command": "doppler",
  "args": [
    "run", "--project", "miami-ai-lab", "--config", "dev", "--",
    "<repo>/.zen_venv/bin/python",
    "<repo>/server.py"
  ],
  "env": {}
}
```

**AFTER** (wrapped — recommended explicit form):

```json
"zen": {
  "type": "stdio",
  "command": "<repo>/.zen_venv/bin/python",
  "args": [
    "<repo>/scripts/zen_supervisor.py"
  ],
  "env": {
    "ZEN_SERVER_CMD": "[\"doppler\",\"run\",\"--project\",\"miami-ai-lab\",\"--config\",\"dev\",\"--\",\"<repo>/.zen_venv/bin/python\",\"<repo>/server.py\"]",
    "ZEN_HOST_LABEL": "<host-name>"
  }
}
```

Why the explicit `ZEN_SERVER_CMD`: it removes any ambiguity about which
python/server.py the supervisor launches on a host whose layout differs. If you
omit it, `resolve_config()` rebuilds the doppler child from defaults derived from
the supervisor's own location — also valid, but explicit is safer for rollout.

Optional tuning knobs (all have safe defaults; set in the same `env` if needed):

| Env var | Default | Meaning |
|---|---|---|
| `ZEN_OUTER_TIMEOUT_SECS` | inner+45 (=195) | hard-kill bound; clamped to [inner+5, 600] |
| `ZEN_PROVIDER_TIMEOUT_SECS` | 150 | inner asyncio per-call timeout |
| `ZEN_KILL_RECORD_DIR` | `<repo>/logs/zen-kills` | where kill-records are written (0600) |
| `ZEN_HOST_LABEL` | hostname | label stamped into kill-records |

### 1c. Restart / reopen

Restart the MCP client (or reopen the session) so it relaunches zen through the
supervisor. Verify with a normal zen call (e.g. `listmodels`) that responses still
flow. If anything is off, restore the backup from 1a and reopen.

---

## 2. Real-daemon smoke (Qualyn R4 close) — run BEFORE trusting the wrapped stanza

```bash
<python> <repo>/scripts/zen_supervisor_smoke.py
```

Forces a wedged call through the supervisor with a low timeout (seconds, **no real
provider spend**) and asserts kill-on-hang + parent-survives + marker-visible.

**PASS looks like** (exit 0):

```
SMOKE PASS: kill-on-hang OK, whole group reaped, parent survived, TOOL_CALL marker seen + WEDGED fired, exit 137.
```

Any `SMOKE FAIL: <reason>` (non-zero exit) ⇒ **do NOT** roll the wrapped stanza on
this host; capture the reason and report it.

---

## 3. LOG_LEVEL arming check (load-bearing R7) — run on every host

The outer hard-kill arms only after it sees the server's `TOOL_CALL:` marker,
which is emitted at INFO. If the host's zen server runs above INFO, the marker is
suppressed and the outer backstop is **inert** (inner asyncio timeout still
protects). Assert `LOG_LEVEL<=INFO`:

```bash
<python> <repo>/scripts/zen_check_log_level.py
```

**PASS** (exit 0): `LOG_LEVEL OK: 'DEBUG' <= INFO — outer backstop will arm.`

If it reports `LOG_LEVEL TOO HIGH`, set `LOG_LEVEL=INFO` (or `DEBUG`) in the zen
server's env (the doppler child env / the stanza) and re-check.

One-liner form (no script needed):

```bash
<python> -c "import os,sys; l=(os.getenv('LOG_LEVEL','DEBUG') or 'DEBUG').upper(); ok=l in {'DEBUG','INFO','NOTSET'}; print(('LOG_LEVEL OK: '+l) if ok else ('LOG_LEVEL TOO HIGH: '+l+' — outer backstop inert')); sys.exit(0 if ok else 1)"
```

---

## 4. Auditor-host variant — **Mario-direct; Sol/Genesis do NOT touch auditor hosts**

> This section is for the auditor boxes (`audit-mini` / `macmini-ailab`
> `audelia-svc`). Per audit independence, **Mario** performs these steps. Sol and
> Genesis do not reach into auditor isolation.

Same steps 1–3, with two auditor-specific requirements:

1. **Kill-records stay in the auditor's own space.** Point `ZEN_KILL_RECORD_DIR`
   at an audit-service-owned, chmod-700 directory — never a Sol-readable or
   world-readable path. In the wrapped stanza's `env`:

   ```json
   "ZEN_KILL_RECORD_DIR": "<auditor-700-dir>",
   "ZEN_HOST_LABEL": "<auditor-host>"
   ```

2. **Operator-side perms check** (Selena-flagged): confirm that directory is owned
   by the audit service user and is `700` BEFORE rolling the stanza, so kill-record
   integrity can't be tampered with by another user:

   ```bash
   # expect: drwx------  <audit-svc-user>  ...
   stat -f "%Sp %Su %N" "<auditor-700-dir>"
   ```

   If it is not `700` and audit-svc-owned, fix ownership/permissions first. The
   supervisor also emits a runtime WARNING if it cannot tighten the dir to 700, but
   the operator check is the authoritative gate.

The kill-records carry **only structural keys** (timestamp, host, provider/model
labels, elapsed, outcome, service user) — never any prompt/response or
auditor-activity payload — so even the record contents respect audit isolation.

---

## Rollback (any host)

Restore the backup taken in step 1a and restart/reopen the MCP client:

```bash
cp ~/.claude.json.bak.<timestamp> ~/.claude.json
```

The supervisor is purely additive plumbing; reverting the stanza returns the host
to the direct `doppler … server.py` launch with zero residue.
