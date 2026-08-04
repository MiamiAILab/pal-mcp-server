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


def served_matches(requested: str, served: str) -> bool:
    """True if `served` is the same seat as `requested`, allowing only snapshot suffixes."""
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


def stamp_served_model(metadata: dict, requested: str, model_response, tool_name: str = "") -> dict:
    """Add served_model + served_model_status to `metadata`; raise on a genuine mismatch.

    Raises:
        ValueError: the provider served a different seat than the one requested.
    """
    return stamp_served_model_id(metadata, requested, resolve_served_model(model_response), tool_name)


def stamp_served_model_id(metadata: dict, requested: str, served: str | None, tool_name: str = "") -> dict:
    """Same as stamp_served_model but takes the already-extracted served id.

    Raises:
        ValueError: the provider served a different seat than the one requested.
    """
    metadata["served_model"] = served
    if not served:
        metadata["served_model_status"] = "NOT_REPORTED"
        return metadata
    if served_matches(requested, served):
        metadata["served_model_status"] = "VERIFIED"
        return metadata
    metadata["served_model_status"] = "MISMATCH"
    raise ServedModelMismatchError(
        "SERVED-MODEL MISMATCH (defect D-C) — the provider served a different seat "
        f"than the one requested{' in ' + tool_name if tool_name else ''}.\n"
        f"  requested : {requested}\n"
        f"  served    : {served}\n"
        "This call is NOT a valid cross-family pass. Do not record it as one, and do not "
        "attribute the response to the requested model or its family. Cross-family "
        "consensus is a Tier-H control primitive; a substituted seat silently voids it."
    )
