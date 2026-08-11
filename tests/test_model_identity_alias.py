"""Alias-vs-canonical seat identity: the 2026-08-11 roster-degradation defect.

Genesis, GENESIS-097. The 2026-08-04 D-C guard compared the caller's REQUEST STRING
against the id the provider served. Callers request seats by ALIAS (`opus`, `codex`,
`k2.7`, `flash`); providers serve CANONICAL ids (`anthropic/claude-opus-4.8`,
`gpt-5.3-codex`, `moonshotai/kimi-k2.7-code`, `gemini-3.5-flash`). Every alias that is
not a bare prefix of its own canonical name was reported as a SILENT SEAT SUBSTITUTION
and dropped from the panel.

BOTH DIRECTIONS ARE TESTED, because a guard that is wrong in the reassuring direction
is far worse than one that is wrong in the alarming direction:

  FALSE REJECT  (the observed defect)  — healthy seat called a substitution. Fixed here.
  FALSE ACCEPT  (the dangerous one)    — genuine substitution called VERIFIED. Must
                                         remain impossible. Asserted, not assumed.
"""

import pytest

from utils.model_identity import (
    ServedModelMismatchError,
    served_matches,
    stamp_served_model_id,
)


class _FakeProvider:
    """Stands in for a real provider's alias table (providers/base.py _resolve_model_name)."""

    def __init__(self, table=None, raises=False):
        self._table = table or {}
        self._raises = raises

    def _resolve_model_name(self, name):
        if self._raises:
            raise RuntimeError("registry unavailable")
        return self._table.get(name, name)


# Real alias -> canonical pairs, taken from a live resolver run on 2026-08-11.
LIVE_ALIASES = {
    "codex": "gpt-5.3-codex",
    "gpt5.3-codex": "gpt-5.3-codex",
    "k2.7": "moonshotai/kimi-k2.7-code",
    "opus": "anthropic/claude-opus-4.8",
    "flash": "gemini-3.5-flash",
    "gpt-5.4": "gpt-5.4",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "sonar-deep-research": "sonar-deep-research",
}


@pytest.fixture
def provider():
    return _FakeProvider(LIVE_ALIASES)


# --------------------------------------------------------------------------------
# DIRECTION 1 — the observed defect: healthy seats must stop being false-rejected.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested,served",
    [
        ("codex", "gpt-5.3-codex"),
        ("gpt5.3-codex", "gpt-5.3-codex"),
        ("k2.7", "moonshotai/kimi-k2.7-code"),
        ("opus", "anthropic/claude-opus-4.8"),
        ("flash", "gemini-3.5-flash"),
    ],
)
def test_alias_matches_its_own_canonical_id(requested, served, provider):
    """These are the SAME seat. Before the fix every one of them raised."""
    assert served_matches(requested, served, provider) is True


def test_alias_false_rejection_is_what_regressed_without_a_provider(provider):
    """Pin the exact regression: no provider => no alias resolution => false reject.

    This is why the defect existed at all — the call sites never passed a provider.
    """
    assert served_matches("opus", "anthropic/claude-opus-4.8", None) is False
    assert served_matches("opus", "anthropic/claude-opus-4.8", provider) is True


def test_seats_that_already_worked_still_work(provider):
    """No regression on the paths that passed by luck (prefix-extended snapshots)."""
    assert served_matches("gpt-5.4", "gpt-5.4-2026-03-05", provider) is True
    assert served_matches("deepseek-v4-pro", "deepseek/deepseek-v4-pro", provider) is True
    assert served_matches("sonar-deep-research", "sonar-deep-research", provider) is True


# --------------------------------------------------------------------------------
# DIRECTION 2 — the dangerous one: a real substitution must NEVER become VERIFIED.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested,served",
    [
        # capability-changing suffix: a different seat, not a snapshot
        ("gpt-5.4", "gpt-5.4-mini"),
        ("gpt-5.4", "gpt-5.4-pro"),
        # the two live incidents the guard exists to catch
        ("gpt-5.6-terra", "sonar-deep-research"),
        ("MiniMax-M2.7", "gemini-2.5-flash"),
        # an alias substituted with a DIFFERENT family's canonical id
        ("opus", "gemini-3.5-flash"),
        ("codex", "anthropic/claude-opus-4.8"),
    ],
)
def test_genuine_substitution_still_mismatches(requested, served, provider):
    """Alias resolution must not launder a substitution into a match."""
    assert served_matches(requested, served, provider) is False


def test_no_false_accept_when_resolver_lies_toward_the_served_model():
    """Only the REQUEST is canonicalised — never the served id.

    If canonicalisation were applied to BOTH sides, a resolver that mapped the served
    id onto the requested seat would manufacture a false ACCEPT. Assert the asymmetry
    holds by giving the resolver a mapping that would do exactly that.
    """
    hostile = _FakeProvider({"gpt-5.4": "gpt-5.4", "gpt-5.4-mini": "gpt-5.4"})
    assert served_matches("gpt-5.4", "gpt-5.4-mini", hostile) is False


def test_resolver_failure_degrades_to_mismatch_never_to_pass():
    """A broken registry must not become a silent pass — fail closed."""
    broken = _FakeProvider(raises=True)
    assert served_matches("opus", "anthropic/claude-opus-4.8", broken) is False
    assert served_matches("gpt-5.4", "gpt-5.4-2026-03-05", broken) is True  # pure-string path


# --------------------------------------------------------------------------------
# Stamping behaviour: status values and the auditor-facing canonical field.
# --------------------------------------------------------------------------------


def test_stamp_verifies_alias_and_records_canonical(provider):
    meta = {}
    stamp_served_model_id(meta, "opus", "anthropic/claude-opus-4.8", "chat", provider)
    assert meta["served_model_status"] == "VERIFIED"
    assert meta["served_model"] == "anthropic/claude-opus-4.8"
    # an auditor must be able to see WHY the two ids were judged one seat
    assert meta["requested_model_canonical"] == "anthropic/claude-opus-4.8"


def test_stamp_still_raises_on_real_substitution(provider):
    meta = {}
    with pytest.raises(ServedModelMismatchError) as exc:
        stamp_served_model_id(meta, "gpt-5.4", "gpt-5.4-mini", "chat", provider)
    assert meta["served_model_status"] == "MISMATCH"
    assert "gpt-5.4-mini" in str(exc.value)


def test_not_reported_is_still_distinct_from_verified(provider):
    """"Could not measure" must never collapse into "checked and fine"."""
    meta = {}
    stamp_served_model_id(meta, "opus", None, "chat", provider)
    assert meta["served_model_status"] == "NOT_REPORTED"


def test_unverifiable_by_design_preserved(provider):
    meta = {}
    stamp_served_model_id(meta, "opus", None, "clink", provider)
    assert meta["served_model_status"] == "UNVERIFIABLE_BY_DESIGN"


def test_mismatch_message_flags_a_missing_provider():
    """Without a provider the guard cannot rule out an alias — say so in the error."""
    with pytest.raises(ServedModelMismatchError) as exc:
        stamp_served_model_id({}, "opus", "anthropic/claude-opus-4.8", "chat", None)
    assert "alias" in str(exc.value).lower()
