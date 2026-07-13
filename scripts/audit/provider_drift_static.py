#!/usr/bin/env python3
"""Tier A — static, deterministic, fail-closed provider-drift gate.

task-128 (Genesis). Companion to the design at
Team/agents/genesis/work/2026-07-13__openrouter-drift-reaudit-ci-gate.md
and Complina's canonical policy
governance/2026-07-13__llm-routing-jurisdiction-policy.md (CR-001).

NO network, NO secrets. Parses conf/openrouter_models.json against four
policy files in conf/policy/ and enforces two orthogonal invariants
(stricter wins):

  Axis 1 (model lineage, load-bearing HIGH control):
    - No HIGH-eligible seat may be non-Western lineage.
    - Any seat matching a china_lineage_prefix is forced lineage=chinese
      and MUST carry a route pin (Axis 2) + must not be HIGH-eligible.

  Axis 2 (inference broker/residency, MODERATE control):
    - Every Chinese-lineage MODERATE-eligible seat: allow_fallbacks==false,
      non-empty `only`, every provider classified 'western' + non-expired.

Plus: deprecated/removed slugs (Grok stale-alias hygiene), quarantine list,
and a diff-aware route-broadening detector (vs a baseline catalog).

Exit codes:
  0  PASS (no BLOCK)
  1  BLOCK (>=1 deterministic policy violation)
  2  usage/IO error
WARN findings do not affect exit code.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# result model
# ---------------------------------------------------------------------------
BLOCK = "BLOCK"
WARN = "WARN"


class Finding:
    def __init__(self, level: str, drift_class: str, seat: str, msg: str):
        self.level = level
        self.drift_class = drift_class
        self.seat = seat
        self.msg = msg

    def __str__(self) -> str:
        return f"[{self.level}] ({self.drift_class}) {self.seat}: {self.msg}"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _today(as_of: str | None) -> _dt.date:
    if as_of:
        return _dt.date.fromisoformat(as_of)
    return _dt.date.today()


def _seats(catalog: dict) -> list[dict]:
    return [m for m in catalog.get("models", []) if isinstance(m, dict)]


# ---------------------------------------------------------------------------
# lineage resolution
# ---------------------------------------------------------------------------
def _matches_prefix(slug: str, prefixes: list[str]) -> bool:
    s = slug.lower()
    return any(s.startswith(p.lower()) or p.lower() in s for p in prefixes)


def _resolve_lineage(slug: str, aliases: list[str], lineage: dict) -> tuple[str, str, dict]:
    """Return (lineage_jurisdiction, max_sensitivity, entry) for a slug.

    Precedence: china_lineage_prefixes FORCE chinese (anti-smuggle) > exact
    seat entry > provider-prefix entry > unknown (fail-closed).
    """
    seats = lineage.get("seats", {})
    prefixes = lineage.get("_README", {}).get("china_lineage_prefixes", [])

    forced_chinese = _matches_prefix(slug, prefixes)

    # exact seat match (incl. the pre-staged ambiguous block)
    entry = seats.get(slug)
    if entry is None:
        amb = seats.get("_AMBIGUOUS_UNSTATED_BROKER", {})
        for alias in [slug, *aliases]:
            if alias in amb and isinstance(amb[alias], dict):
                entry = amb[alias]
                break
    # provider-prefix match (e.g. "openai/")
    if entry is None:
        for key, val in seats.items():
            if key.endswith("/") and slug.lower().startswith(key.lower()):
                entry = val
                break

    if entry is None:
        # no classification at all -> unknown, fail-closed
        return ("chinese" if forced_chinese else "unknown", "LOW", {})

    lj = entry.get("lineage_jurisdiction", "unknown")
    ms = entry.get("max_sensitivity", "LOW")
    if forced_chinese and lj != "chinese":
        # a china-prefix slug classified western/unknown = mislabel smuggle
        lj = "chinese"
    return (lj, ms, entry)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_catalog(
    catalog: dict,
    lineage: dict,
    jurisdictions: dict,
    deprecated: dict,
    quarantined: dict,
    as_of: str | None,
    baseline: dict | None,
) -> list[Finding]:
    findings: list[Finding] = []
    today = _today(as_of)
    prov = jurisdictions.get("providers", {})
    dep_slugs = {d["slug"] for d in deprecated.get("deprecated", [])}
    dep_aliases = {a.lower() for a in deprecated.get("deprecated_aliases", [])}
    dep_prefixes = deprecated.get("deprecated_slug_prefixes", [])
    quar = {q["slug"] for q in quarantined.get("quarantined", [])}
    china_prefixes = lineage.get("_README", {}).get("china_lineage_prefixes", [])

    for seat in _seats(catalog):
        slug = seat.get("model_name", "<no model_name>")
        aliases = [a.lower() for a in seat.get("aliases", [])]

        # (d) deprecated / removed (incl. Grok stale-alias hygiene)
        if slug in dep_slugs or _matches_prefix(slug, dep_prefixes):
            findings.append(Finding(BLOCK, "d-deprecated", slug,
                "deprecated/removed slug still wired (stale-alias hygiene, policy 3c/5.F)"))
            continue
        alias_hit = dep_aliases.intersection(aliases)
        if alias_hit:
            findings.append(Finding(BLOCK, "d-deprecated", slug,
                f"carries deprecated alias(es) {sorted(alias_hit)} (policy 5.F)"))

        # quarantine (live->static bridge)
        if slug in quar:
            findings.append(Finding(BLOCK, "quarantine", slug,
                "slug is quarantined by a confirmed live-audit finding"))

        lj, ms, _entry = _resolve_lineage(slug, aliases, lineage)
        route = seat.get("openrouter_provider_route") or {}
        only = route.get("only") or []
        allow_fb = route.get("allow_fallbacks", None)

        is_china = lj == "chinese" or _matches_prefix(slug, china_prefixes)

        # Axis 1 (HIGH invariant): HIGH-eligible must be western lineage
        if ms == "HIGH" and lj != "western":
            findings.append(Finding(BLOCK, "b-jurisdiction-HIGH", slug,
                f"HIGH-eligible but lineage={lj} (Axis 1, policy 1). "
                "Chinese/unknown lineage is excluded from HIGH — no broker exception."))

        # Axis 2 (MODERATE invariant): china-lineage pin completeness
        if is_china:
            if not only:
                findings.append(Finding(BLOCK, "b-lineage-pin", slug,
                    "Chinese-lineage seat has empty/missing openrouter_provider_route.only "
                    "(Axis 2 pin required — policy 4)."))
            if allow_fb is True:
                findings.append(Finding(BLOCK, "b-fallback", slug,
                    "Chinese-lineage seat has allow_fallbacks=true — a PRC endpoint could be "
                    "selected on fallback (fail-closed violation)."))
            if allow_fb is None and only:
                findings.append(Finding(WARN, "b-fallback-implicit", slug,
                    "route pin present but allow_fallbacks not explicitly false; set it explicitly."))

            # every provider in `only` must be western + non-expired
            for p in only:
                pj = prov.get(p)
                if pj is None:
                    findings.append(Finding(BLOCK, "b-provider-unknown", slug,
                        f"provider '{p}' in `only` has NO jurisdiction entry (unknown==non-Western, BLOCK)."))
                    continue
                cls = pj.get("jurisdiction_class", "unknown")
                if cls != "western":
                    findings.append(Finding(BLOCK, "b-provider-nonwestern", slug,
                        f"provider '{p}' classified '{cls}' (not western) — BLOCK."))
                    continue
                exp = pj.get("review_expires_at")
                if exp and _dt.date.fromisoformat(exp) < today:
                    findings.append(Finding(BLOCK, "b-provider-stale", slug,
                        f"provider '{p}' review expired {exp} and is used by a protected seat — BLOCK (stale==non-compliant)."))

            # MODERATE-eligible china seat must be verifiably brokered (handled by the
            # provider checks above); LOW-only ambiguous seats are safe as-is.
            if ms == "MODERATE" and not only:
                findings.append(Finding(BLOCK, "b-moderate-unbrokered", slug,
                    "MODERATE-eligible Chinese-lineage seat without a verified Western broker pin (policy 4)."))

    # unused-but-stale provider entries => WARN only
    used = set()
    for seat in _seats(catalog):
        for p in (seat.get("openrouter_provider_route") or {}).get("only", []) or []:
            used.add(p)
    for pname, pj in prov.items():
        exp = pj.get("review_expires_at")
        if exp and _dt.date.fromisoformat(exp) < today and pname not in used:
            findings.append(Finding(WARN, "provider-stale-unused", pname,
                f"jurisdiction review expired {exp} (unused by any seat — WARN)."))

    # (f) diff-aware route-broadening vs baseline
    if baseline is not None:
        findings.extend(_broadening(baseline, catalog))

    return findings


def _broadening(baseline: dict, catalog: dict) -> list[Finding]:
    out: list[Finding] = []
    base = {m.get("model_name"): m for m in _seats(baseline)}
    for seat in _seats(catalog):
        slug = seat.get("model_name")
        b = base.get(slug)
        if not b:
            continue
        br = b.get("openrouter_provider_route") or {}
        cr = seat.get("openrouter_provider_route") or {}
        b_only, c_only = set(br.get("only") or []), set(cr.get("only") or [])
        if c_only - b_only:
            out.append(Finding(BLOCK, "f-broadening", slug,
                f"PR ADDS provider(s) {sorted(c_only - b_only)} to `only` — route broadening, "
                "requires deliberate review (fail-closed)."))
        if br.get("allow_fallbacks") is False and cr.get("allow_fallbacks") is True:
            out.append(Finding(BLOCK, "f-broadening", slug,
                "PR flips allow_fallbacks false->true — route broadening (fail-closed)."))
        if (br.get("only") or []) and not (cr.get("only") or []):
            out.append(Finding(BLOCK, "f-broadening", slug,
                "PR empties/removes the `only` route pin — route broadening (fail-closed)."))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tier A static provider-drift gate (fail-closed).")
    root = Path(__file__).resolve().parents[2]  # zen-mcp-server/
    ap.add_argument("--catalog", type=Path, default=root / "conf" / "openrouter_models.json")
    ap.add_argument("--policy-dir", type=Path, default=root / "conf" / "policy")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="baseline catalog (e.g. origin/main copy) for route-broadening detection")
    ap.add_argument("--as-of", default=None, help="override 'today' (YYYY-MM-DD) for expiry testing")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = ap.parse_args(argv)

    try:
        catalog = _load(args.catalog)
        pd = args.policy_dir
        lineage = _load(pd / "model_lineage.json")
        jurisdictions = _load(pd / "provider_jurisdictions.json")
        deprecated = _load(pd / "deprecated_models.json")
        quarantined = _load(pd / "quarantined_models.json")
        baseline = _load(args.baseline) if args.baseline else None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR loading inputs: {exc}", file=sys.stderr)
        return 2

    findings = check_catalog(catalog, lineage, jurisdictions, deprecated,
                             quarantined, args.as_of, baseline)
    blocks = [f for f in findings if f.level == BLOCK]

    if args.json:
        print(json.dumps({
            "result": "BLOCK" if blocks else "PASS",
            "block_count": len(blocks),
            "findings": [f.__dict__ for f in findings],
        }, indent=2))
    else:
        for f in findings:
            print(f)
        print("-" * 60)
        print(f"RESULT: {'BLOCK' if blocks else 'PASS'}  "
              f"({len(blocks)} block, {len(findings) - len(blocks)} warn)")

    return 1 if blocks else 0


if __name__ == "__main__":
    raise SystemExit(main())
