#!/usr/bin/env python3
"""Tests for Tier A static provider-drift gate (task-128).

Proves the REQUIREMENTS_SET (front-gate) behaviors so Complina/Selena/Qualyn
can sign VERDICT_MET:
  1. BLOCK on allow_fallbacks flipped true          (Selena)
  2. BLOCK on unknown/prc provider added to `only`   (Complina)
  3. BLOCK on expired provider review used by seat    (Complina)
  4. BLOCK on China-lineage seat with no route pin    (Complina)
  5. PASS on clean, current, all-western config       (Qualyn — no false fail-closed)
  6. HIGH-eligible non-western lineage => BLOCK        (Axis 1)
  + Grok deprecated-alias hygiene, route-broadening diff.

Run: python3 -m pytest scripts/audit/test_provider_drift_static.py -v
(stdlib-only fallback runner at bottom if pytest unavailable.)
"""
from __future__ import annotations

import copy

import provider_drift_static as g

TODAY = "2026-07-13"

WESTERN_PROV = {
    "providers": {
        "deepinfra": {"jurisdiction_class": "western", "review_expires_at": "2026-10-13"},
        "atlas-cloud": {"jurisdiction_class": "western", "review_expires_at": "2026-10-13"},
        "prc-cloud": {"jurisdiction_class": "prc", "review_expires_at": "2026-10-13"},
        "unk-cloud": {"jurisdiction_class": "unknown", "review_expires_at": "2026-10-13"},
        "expired-west": {"jurisdiction_class": "western", "review_expires_at": "2026-01-01"},
    }
}
LINEAGE = {
    "_README": {"china_lineage_prefixes": ["deepseek/", "qwen/", "z-ai/", "minimax/"]},
    "seats": {
        "openai/": {"lineage_jurisdiction": "western", "max_sensitivity": "HIGH"},
        "anthropic/": {"lineage_jurisdiction": "western", "max_sensitivity": "HIGH"},
        "deepseek/deepseek-v3.2": {"lineage_jurisdiction": "chinese", "max_sensitivity": "MODERATE"},
    },
}
DEPRECATED = {
    "deprecated": [{"slug": "x-ai/grok-4"}],
    "deprecated_aliases": ["grok"],
    "deprecated_slug_prefixes": ["x-ai/"],
}
QUAR = {"quarantined": []}


def _run(catalog, baseline=None, lineage=None):
    return g.check_catalog(catalog, lineage or LINEAGE, WESTERN_PROV, DEPRECATED,
                           QUAR, TODAY, baseline)


def _blocks(findings):
    return [f for f in findings if f.level == g.BLOCK]


def _clean_western_seat():
    return {"model_name": "openai/gpt-5.2", "aliases": ["gpt5.2"]}


def _clean_china_seat():
    return {
        "model_name": "deepseek/deepseek-v3.2",
        "aliases": ["deepseek"],
        "openrouter_provider_route": {"only": ["deepinfra", "atlas-cloud"], "allow_fallbacks": False},
    }


# 5. clean config PASSes (no false fail-closed)
def test_clean_config_passes():
    cat = {"models": [_clean_western_seat(), _clean_china_seat()]}
    assert _blocks(_run(cat)) == []


# 1. allow_fallbacks true => BLOCK
def test_fallback_true_blocks():
    seat = _clean_china_seat()
    seat["openrouter_provider_route"]["allow_fallbacks"] = True
    b = _blocks(_run({"models": [seat]}))
    assert any(f.drift_class == "b-fallback" for f in b)


# 2. unknown provider in only => BLOCK ; prc provider => BLOCK
def test_unknown_provider_blocks():
    seat = _clean_china_seat()
    seat["openrouter_provider_route"]["only"] = ["deepinfra", "unk-cloud"]
    b = _blocks(_run({"models": [seat]}))
    assert any(f.drift_class == "b-provider-nonwestern" for f in b)


def test_prc_provider_blocks():
    seat = _clean_china_seat()
    seat["openrouter_provider_route"]["only"] = ["prc-cloud"]
    b = _blocks(_run({"models": [seat]}))
    assert any(f.drift_class == "b-provider-nonwestern" for f in b)


def test_missing_provider_entry_blocks():
    seat = _clean_china_seat()
    seat["openrouter_provider_route"]["only"] = ["deepinfra", "no-such-provider"]
    b = _blocks(_run({"models": [seat]}))
    assert any(f.drift_class == "b-provider-unknown" for f in b)


# 3. expired provider review used by seat => BLOCK
def test_expired_provider_blocks():
    seat = _clean_china_seat()
    seat["openrouter_provider_route"]["only"] = ["expired-west"]
    b = _blocks(_run({"models": [seat]}))
    assert any(f.drift_class == "b-provider-stale" for f in b)


# 4. china-lineage seat with no route pin => BLOCK
def test_china_no_pin_blocks():
    seat = {"model_name": "deepseek/deepseek-v3.2", "aliases": ["deepseek"]}
    b = _blocks(_run({"models": [seat]}))
    assert any(f.drift_class in ("b-lineage-pin", "b-moderate-unbrokered") for f in b)


def test_china_empty_only_blocks():
    seat = _clean_china_seat()
    seat["openrouter_provider_route"]["only"] = []
    b = _blocks(_run({"models": [seat]}))
    assert any(f.drift_class == "b-lineage-pin" for f in b)


# 6. HIGH-eligible non-western lineage => BLOCK (Axis 1)
def test_high_nonwestern_lineage_blocks():
    lin = copy.deepcopy(LINEAGE)
    # mislabel a china-prefix slug as HIGH western -> anti-smuggle forces chinese, then HIGH block
    lin["seats"]["qwen/qwen3.5-397b-a17b"] = {"lineage_jurisdiction": "western", "max_sensitivity": "HIGH"}
    seat = {
        "model_name": "qwen/qwen3.5-397b-a17b", "aliases": ["qwen"],
        "openrouter_provider_route": {"only": ["deepinfra"], "allow_fallbacks": False},
    }
    b = _blocks(_run({"models": [seat]}, lineage=lin))
    assert any(f.drift_class == "b-jurisdiction-HIGH" for f in b)


# Grok deprecated hygiene
def test_grok_slug_blocks():
    seat = {"model_name": "x-ai/grok-4", "aliases": ["grok"]}
    b = _blocks(_run({"models": [seat]}))
    assert any(f.drift_class == "d-deprecated" for f in b)


def test_grok_alias_on_other_seat_blocks():
    seat = {"model_name": "some/model", "aliases": ["grok"]}
    b = _blocks(_run({"models": [seat]}))
    assert any(f.drift_class == "d-deprecated" for f in b)


# route-broadening diff
def test_broadening_added_provider_blocks():
    base = {"models": [_clean_china_seat()]}
    cur = copy.deepcopy(base)
    cur["models"][0]["openrouter_provider_route"]["only"].append("unk-cloud")
    b = _blocks(_run(cur, baseline=base))
    assert any(f.drift_class == "f-broadening" for f in b)


def test_broadening_fallback_flip_blocks():
    base = {"models": [_clean_china_seat()]}
    cur = copy.deepcopy(base)
    cur["models"][0]["openrouter_provider_route"]["allow_fallbacks"] = True
    b = _blocks(_run(cur, baseline=base))
    assert any(f.drift_class == "f-broadening" for f in b)


if __name__ == "__main__":
    # stdlib fallback runner
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
