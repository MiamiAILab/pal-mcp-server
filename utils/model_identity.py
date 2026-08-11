"""Served-model identity: surface what the provider ACTUALLY served, and assert on it.

GENESIS 2026-08-04, Mario ruling. Closes defect D-C from the zen-verdict-transport arc.

THE DEFECT
    `metadata.model_used` was stamped REQUEST-SIDE on both tool paths:
      - workflow tools: arguments["_resolved_model_name"]   (workflow_mixin._add_workflow_metadata)
      - simple tools:   self._current_model_name            (tools/simple/base.py)
    Neither is read back from the provider's response. So the long-standing
    GENESIS-097 healthcheck rule `model_used == model_requested` compared the
    request to a restatement of the request: it validated ALIAS RESOLUTION and
    could never detect a substituted seat. Two live incidents (Selena 2026-08-01
    requesting gpt-5.4-pro; Azul 2026-08-03 requesting gpt-5.6-terra) were
    detectable only as self-contradictory metadata, and not attributable.

WHY IT MATTERS
    Cross-family consensus is a CONTROL PRIMITIVE inside Tier-H. Both auditors
    permanently downgraded their own citations because of this. Audelia:
    "family independence is the entire basis of my role; I request the alias,
    I verify neither the model nor, strictly, the family." An unverifiable
    control is a decorative one — this is a live degradation of the
    three-lines-of-defense architecture, not a cosmetic issue.

THE FIX
    Providers ALREADY capture the served id — openai_compatible.py records
    metadata={"model": response.model, ...} straight from the provider's own
    response body — and it was then discarded at metadata-stamping time.
    We now surface it as `served_model` and assert against the requested id.

THREE OUTCOMES, DELIBERATELY DISTINCT (per the Audelia BOUND-UNKNOWN lesson:
"could not measure" must never be collapsed into "verified" or into "failed"):
    VERIFIED      served id reported and compatible with the request
    NOT_REPORTED  provider did not echo a model id -> UNVERIFIED, never "fine".
                  Does NOT raise: some providers legitimately omit it, and
                  raising would break working calls. It is made VISIBLE so a
                  healthcheck can refuse to count it as verification.
    MISMATCH      served id is a different model -> raises. Loud, never silent.
"""

import re

# A trailing snapshot/version suffix is an ACCEPTABLE difference:
#   "gpt-5.4" served as "gpt-5.4-2026-01-15" is the same seat.
# It must be purely date/version-like. This is deliberately strict so that
#   "gpt-5.4" served as "gpt-5.4-mini"  ->  MISMATCH, not "snapshot"
# A capability-changing suffix is exactly the substitution we exist to catch.
_SNAPSHOT_TAIL = re.compile(r"^[0-9][0-9._-]*$")


# === NOT_REPORTED BY DESIGN vs BY FAILURE (ruled 2026-08-04, Sol#1 question) =========
# A path that STRUCTURALLY cannot report a served model id is a different world from a
# path that should have reported one and did not. Both yield "no served identity", and
# collapsing them would (a) send operators chasing a repair that does not exist, and
# (b) let a genuine failure hide inside an expected condition.
#
# THE SAFETY PROPERTY, and it is the whole reason this is a declared list and not an
# inference: by-design status is ASSERTED HERE, never derived from a missing field.
# If absence itself were treated as "by design", every real reporting failure would
# silently reclassify as expected — reintroducing exactly the silent-pass hazard this
# module exists to remove.
#
# Current member: `clink`, which bridges to external AI CLIs as a SUBPROCESS. It
# overrides execute() and never enters the provider metadata path at all, so
# response.model does not exist for it in any form.
#
# CONSEQUENCE FOR VERDICTS (fail-closed on the property that actually matters):
# an UNVERIFIABLE_BY_DESIGN seat is NO_VOTE for cross-family consensus and Tier-H
# verdict independence — a seat whose family cannot be established cannot serve as an
# independent family vote. It is NOT an outage: it is excluded from the verifiable
# quorum and reported distinctly rather than halting the roster, and it stays usable
# for non-verdict work where family independence is not the property being claimed.
# Citable as "content delivered; seat NOT REPORTED" — NEVER as "verified <family> seat".
UNVERIFIABLE_BY_DESIGN_TOOLS = frozenset({"clink"})


class ServedModelMismatchError(ValueError):
    """The provider served a different seat than the one requested.

    A DEDICATED type, not a bare ValueError, and that is load-bearing. Both
    metadata-stamping sites sit inside pre-existing broad handlers that fail
    toward reassurance:
      - tools/simple/base.py       `except Exception: return ToolOutput(status="success")`
      - workflow_mixin             `except Exception: logger.warning(...)` and continue
    A generic exception raised at either site is SWALLOWED and the call returns
    looking successful — the exact silent-false-success shape this fix exists to
    remove, one layer down. Discovered 2026-08-04 because the D-C negative control
    did not fire; without that control an assertion that can never fire would have
    shipped. Both handlers now re-raise this type explicitly.
    """


def normalize_model_id(model_id) -> str:
    """Lowercase and drop any provider-routing prefix ('openai/gpt-5.2' -> 'gpt-5.2').

    Non-string input yields "" rather than raising. A provider that reports a
    non-string model id has not reported one, and that must degrade to
    NOT_REPORTED — never to an exception. (Found 2026-08-04: a non-str id raised
    a TypeError that an enclosing broad handler turned into EMPTY metadata,
    silently dropping model_used. Failing this open would have been a new
    silent-false-success of exactly the kind this module exists to remove.)
    """
    if not isinstance(model_id, str) or not model_id.strip():
        return ""
    m = model_id.strip().lower()
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m


def resolve_requested_alias(requested: str, provider=None) -> str:
    """Map a caller-facing ALIAS to the provider's own canonical model id.

    ADDED 2026-08-11 (Genesis) — closes a FALSE-REJECTION defect in the 2026-08-04
    D-C guard. Callers legitimately request seats by ALIAS: `opus`, `codex`, `k2.7`,
    `flash`. The provider then serves the CANONICAL id — `anthropic/claude-opus-4.8`,
    `gpt-5.3-codex`, `moonshotai/kimi-k2.7-code`, `gemini-3.5-flash`. The original
    `served_matches` compared the alias STRING against the canonical id, so every
    alias that is not a bare prefix of its own canonical name was reported as a
    SILENT SEAT SUBSTITUTION. `gpt-5.4` passed only by luck (its served snapshot
    `gpt-5.4-2026-03-05` happens to be prefix-extended).

    That took real seats out of real panels — `opus` is a consensus-roster seat and
    could never pass the guard when requested by its alias. Observed live 2026-08-11
    (Visionary 4 consecutive degraded passes; Marshall 3 false rejections; Complina
    3 sessions on the Gemini fallback).

    Direction matters: this defect is fail-CLOSED (healthy seat rejected), never
    fail-open. It cannot manufacture a false ACCEPT — see the module test.

    Degrades to the raw string on any resolver failure: an unresolvable alias must
    stay a mismatch, never become a silent pass.
    """
    if not isinstance(requested, str) or provider is None:
        return requested
    try:
        resolved = provider._resolve_model_name(requested)
    except Exception:  # noqa: BLE001 — a resolver failure must not weaken the guard
        return requested
    return resolved if isinstance(resolved, str) and resolved.strip() else requested


def _same_seat(requested: str, served: str) -> bool:
    """Pure string comparison of two model ids, allowing only snapshot suffixes."""
    r, s = normalize_model_id(requested), normalize_model_id(served)
    if not r or not s:
        return False
    if r == s:
        return True
    # snapshot/dated variant of the same seat, e.g. gpt-5.4 -> gpt-5.4-2026-01-15
    for a, b in ((r, s), (s, r)):
        if b.startswith(a):
            tail = b[len(a):].lstrip("-_:@")
            if tail and _SNAPSHOT_TAIL.match(tail):
                return True
    return False


def served_matches(requested: str, served: str, provider=None) -> bool:
    """True if `served` is the same seat as `requested`, allowing only snapshot suffixes.

    When `provider` is supplied the REQUEST is also compared in its canonical form, so
    an alias matches the canonical id the provider actually served. Only the REQUEST is
    canonicalised — never the served id — so this can add matches ONLY between an alias
    and its own declared canonical name. A genuine substitution (`gpt-5.4` served as
    `gpt-5.4-mini`) is still a mismatch, because canonicalising `gpt-5.4` yields
    `gpt-5.4` and the capability suffix still fails the snapshot rule.
    """
    if _same_seat(requested, served):
        return True
    canonical = resolve_requested_alias(requested, provider)
    if canonical != requested and _same_seat(canonical, served):
        return True
    return False


def resolve_served_model(model_response) -> str | None:
    """Pull the id the provider actually served out of a ModelResponse, if it reported one."""
    if model_response is None:
        return None
    meta = getattr(model_response, "metadata", None)
    served = meta.get("model") if isinstance(meta, dict) else None
    if not isinstance(served, str) or not served.strip():
        # some providers populate only the top-level field
        served = getattr(model_response, "model_name", None)
    # Anything that is not a non-empty string means "the provider did not report
    # a usable id" -> NOT_REPORTED. Never raise from extraction.
    if not isinstance(served, str) or not served.strip():
        return None
    return served.strip()


def stamp_served_model(metadata: dict, requested: str, model_response, tool_name: str = "", provider=None) -> dict:
    """Add served_model + served_model_status to `metadata`; raise on a genuine mismatch.

    Raises:
        ValueError: the provider served a different seat than the one requested.
    """
    return stamp_served_model_id(
        metadata, requested, resolve_served_model(model_response), tool_name, provider
    )


def stamp_served_model_id(
    metadata: dict, requested: str, served: str | None, tool_name: str = "", provider=None
) -> dict:
    """Same as stamp_served_model but takes the already-extracted served id.

    Raises:
        ValueError: the provider served a different seat than the one requested.
    """
    metadata["served_model"] = served
    if not served:
        # DECLARED structural inability vs an anomaly. Never inferred from absence.
        metadata["served_model_status"] = (
            "UNVERIFIABLE_BY_DESIGN" if tool_name in UNVERIFIABLE_BY_DESIGN_TOOLS else "NOT_REPORTED"
        )
        return metadata
    # Surface the canonical form the alias resolves to. An auditor reading this
    # metadata must be able to see WHY an alias and a canonical id were judged the
    # same seat, rather than take the verdict on trust.
    canonical = resolve_requested_alias(requested, provider)
    if canonical != requested:
        metadata["requested_model_canonical"] = canonical
    if served_matches(requested, served, provider):
        metadata["served_model_status"] = "VERIFIED"
        return metadata
    metadata["served_model_status"] = "MISMATCH"
    raise ServedModelMismatchError(
        "SERVED-MODEL MISMATCH (defect D-C) — the provider served a different seat "
        f"than the one requested{' in ' + tool_name if tool_name else ''}.\n"
        f"  requested : {requested}\n"
        + (f"  canonical : {canonical}   (alias resolved)\n" if canonical != requested else "")
        + f"  served    : {served}\n"
        + (
            "  NOTE: no provider was supplied to the guard, so an ALIAS could not be resolved to "
            "its canonical id. If 'requested' is an alias, re-check before treating this as a "
            "substitution.\n"
            if provider is None
            else ""
        )
        + "This call is NOT a valid cross-family pass. Do not record it as one, and do not "
        "attribute the response to the requested model or its family. Cross-family "
        "consensus is a Tier-H control primitive; a substituted seat silently voids it."
    )
