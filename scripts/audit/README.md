# OpenRouter provider-drift re-audit gate (task-128)

Genesis, 2026-07-13. Design: `MiamiAILab/Team` → `agents/genesis/work/2026-07-13__openrouter-drift-reaudit-ci-gate.md`.
Canonical jurisdiction policy: `MiamiAILab/Team` → `governance/2026-07-13__llm-routing-jurisdiction-policy.md` (Complina, CR-001).

Two-tier, fail-closed, three-state (BLOCK / QUARANTINE_REQUIRED / INCONCLUSIVE).

## Tier A — static merge gate (this dir, shipped)

`provider_drift_static.py` — pure stdlib, NO network, NO secrets, deterministic.
Enforces two orthogonal axes (stricter wins):

- **Axis 1 (model lineage, load-bearing HIGH control):** a HIGH-eligible seat must be
  Western lineage. China-lineage prefixes force `chinese` (anti-mislabel-smuggle).
- **Axis 2 (inference broker, MODERATE control):** every China-lineage MODERATE seat needs
  `allow_fallbacks:false`, non-empty `only`, every provider `western` + non-expired review.

Also: deprecated/removed slugs (Grok stale-alias hygiene), quarantine list (live→static
bridge), diff-aware route-broadening detector.

```bash
# audit the live catalog
python3 scripts/audit/provider_drift_static.py

# with route-broadening detection vs origin/main baseline (CI)
git show origin/main:conf/openrouter_models.json > /tmp/base.json
python3 scripts/audit/provider_drift_static.py --baseline /tmp/base.json --json

# tests
python3 scripts/audit/test_provider_drift_static.py     # stdlib runner
python3 -m pytest scripts/audit/test_provider_drift_static.py -v
```

Exit 0 = PASS, 1 = BLOCK, 2 = IO error. WARN never fails CI.

## Policy data — `conf/policy/` (CODEOWNERS: @complina @selena)

| File | Axis | Owner |
|---|---|---|
| `model_lineage.json` | 1 — per-seat training lineage + max_sensitivity | Genesis (roster), mirrors policy §3 |
| `provider_jurisdictions.json` | 2 — OpenRouter sub-provider western/prc/unknown + expiry | Complina/Selena review |
| `deprecated_models.json` | removed/deprecated slugs + aliases | Genesis |
| `quarantined_models.json` | live→static bridge (Tier B writes) | automation |

**Fail-closed defaults:** missing/unknown provider entry, expired-and-used review, unstated
broker on a China-lineage seat → BLOCK / LOW-only. The seed classifications are marked
`PENDING-COMPLINA`; Tier A will BLOCK unverified providers (friendli/morph/parasail/gmicloud)
until Complina classifies them or they are stripped from the `only` lists — this is the gate
working as intended, not a bug.

## Tier B — live scheduled audit (follow-up, not in this PR)

`provider_drift_live.py` (to build) — probes protected seats with a minimal request, reads the
served `provider` + returned `model` (OpenRouter completion body + `GET /api/v1/generation?id=`),
classifies model_used!=requested / dead-seat / observed-provider-outside-`only`. Confirmed
findings materialize into `quarantined_models.json`. Runs nightly + `workflow_dispatch`; NEVER
in the merge path (network flake = INCONCLUSIVE, not BLOCK).

## CI wiring

`.github/workflows/provider-drift-audit.yml` — drafted, handed to **Sol** for the SSH-push lane
(workflow-file HTTPS pushes fail per fleet convention). Tier A runs on every PR touching
`conf/openrouter_models.json` or `conf/policy/**`; Tier B runs on schedule.
