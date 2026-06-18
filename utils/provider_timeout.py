"""Bounded-timeout wrapper for synchronous provider.generate_content() calls.

zen-MCP hang durable-fix — Layer 2 (SOL-338 family, 2026-06-18).

PROBLEM
    provider.generate_content() is a SYNCHRONOUS blocking SDK call invoked
    directly from async tool code (4 callsites). When a provider hangs, the
    await never returns, the call sits forever (an audit agent observed a
    35-minute hang mid-audit), and — because the call never *raises* — the
    existing provider-error fallback chain never fires.

FIX
    Run the blocking call in a DEDICATED bounded thread pool and bound the
    await with asyncio.wait_for(). On timeout, RAISE ProviderTimeoutError (a
    TimeoutError subclass) so the existing broad `except Exception` at each
    callsite produces an error result and the fallback chain
    (gpt-5.4 -> gemini -> qwen -> ... -> host-Opus-degraded) finally fires.

DESIGN RATIFICATION (/pareto, Tier S, 2026-06-18)
    Mean confidence 0.84 (gpt-5.4 0.80 + gemini-3.1-pro 0.90 + Wren 0.82;
    qwen non-vote). MANDATORY amendment from that review, applied here:
    use a DEDICATED ThreadPoolExecutor, NOT asyncio.to_thread (which shares the
    process-default ~min(32, cpu+4) executor). Under a hang-storm the leaked
    worker threads — which wait_for cannot cancel — would otherwise saturate the
    default pool and block ALL to_thread work server-wide, turning a per-call
    hang into a fleet-wide stall (the exact opposite of this fix). A dedicated,
    generously sized pool (I/O-bound threads are cheap) isolates the blast
    radius so saturation degrades to fast-fail rather than silent block.

KNOWN LIMITATION (disclosed; follow-up tracked)
    wait_for cancels the await but cannot cancel the blocking socket read inside
    the worker thread — that thread (and its socket/FD) leaks until the SDK call
    returns or the OS TCP timeout (~15-30 min). The true reap is native
    per-provider httpx/SDK transport timeouts; that is the MANDATORY follow-up,
    not this stopgap.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECS = 150
_ENV_TIMEOUT = "ZEN_PROVIDER_TIMEOUT_SECS"
_MAX_WORKERS = max(8, int(os.getenv("ZEN_PROVIDER_MAX_WORKERS", "200")))

# --- Layer 3: native transport timeout knob ------------------------------------
# The TRUE reap. Layer 2 (above) frees the awaiting caller but CANNOT cancel the
# blocking socket read inside the worker thread — that thread + its FD leak until
# the SDK returns or the OS TCP timeout (~15-30 min). Layer 3 sets an explicit
# read/transport timeout on each provider's underlying HTTP client so a hung
# socket actually RAISES at the transport layer in ~seconds, the worker thread
# unwinds, and the FD closes.
#
# ORDERING (ratified via cross-family review, gpt-5.4 2026-06-18): the transport
# timeout is the PRIMARY reaper and must fire BEFORE the Layer-2 wait_for outer
# bound — otherwise wait_for fires first and we are back to leaking the thread.
# Hence the default (120s) is deliberately BELOW DEFAULT_TIMEOUT_SECS (150s):
#   transport read timeout (120s)  -> reaps the socket, unwinds the thread
#   Layer-2 wait_for      (150s)   -> secondary caller-protection guardrail
# Provider-specific CUSTOM_*_TIMEOUT envs still take precedence over this knob.
DEFAULT_TRANSPORT_TIMEOUT_SECS = 120
_ENV_TRANSPORT_TIMEOUT = "ZEN_PROVIDER_TRANSPORT_TIMEOUT_SECS"


def resolve_transport_timeout_secs(default=DEFAULT_TRANSPORT_TIMEOUT_SECS):
    """Resolve the per-provider native transport (read) timeout in SECONDS.

    Reads ZEN_PROVIDER_TRANSPORT_TIMEOUT_SECS; falls back to `default`
    (120s). Returns a positive float, or None if the operator set the value
    to <=0 (kill-switch -> let the provider client use its own default /
    unbounded behavior, matching the Layer-2 <=0 kill-switch convention).

    NOTE: this returns SECONDS. The Gemini SDK (google-genai) expects
    milliseconds on HttpOptions.timeout — callers there must multiply by 1000.
    The OpenAI-compatible httpx path uses seconds directly.
    """
    raw = os.getenv(_ENV_TRANSPORT_TIMEOUT)
    if raw is None:
        val = float(default)
    else:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "%s=%r is not a number; using default %ss",
                _ENV_TRANSPORT_TIMEOUT, raw, default,
            )
            val = float(default)
    if val <= 0:
        return None  # kill-switch: defer to the SDK/client default
    # Soft guard: warn if transport >= the Layer-2 outer bound, which weakens
    # the reap (Layer 2 would fire first and leak the worker thread).
    outer_raw = os.getenv(_ENV_TIMEOUT)
    try:
        outer = float(outer_raw) if outer_raw is not None else DEFAULT_TIMEOUT_SECS
    except (TypeError, ValueError):
        outer = DEFAULT_TIMEOUT_SECS
    if outer > 0 and val >= outer:
        logger.warning(
            "%s (%ss) >= %s (%ss): the transport timeout will not act as the "
            "primary socket reaper; the Layer-2 wait_for may fire first and leak "
            "the worker thread. Set transport below the outer bound.",
            _ENV_TRANSPORT_TIMEOUT, val, _ENV_TIMEOUT, outer,
        )
    return val

# Dedicated pool: isolates blocking provider calls from asyncio's default
# executor so a hang-storm cannot starve unrelated to_thread work. Per /pareto.
_PROVIDER_EXECUTOR = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS, thread_name_prefix="zen_provider"
)


class ProviderTimeoutError(TimeoutError):
    """Raised when provider.generate_content() exceeds the bounded timeout.

    Subclasses TimeoutError so existing broad `except Exception` handlers catch
    it and route to the provider-error fallback chain. Carries provider/model
    metadata for metrics + alerting.
    """

    def __init__(self, timeout_secs, provider=None, model_name=None):
        self.timeout_secs = timeout_secs
        self.provider = provider
        self.model_name = model_name
        try:
            ptype = provider.get_provider_type().value if provider is not None else "unknown"
        except Exception:
            ptype = type(provider).__name__ if provider is not None else "unknown"
        super().__init__(
            f"provider.generate_content timed out after {timeout_secs}s "
            f"(provider={ptype}, model={model_name or 'unknown'})"
        )


def _resolve_timeout():
    """Resolve the timeout (seconds). <=0 = kill-switch (no client-side bound)."""
    raw = os.getenv(_ENV_TIMEOUT)
    if raw is None:
        return DEFAULT_TIMEOUT_SECS
    try:
        val = int(raw)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not an int; using default %ss", _ENV_TIMEOUT, raw, DEFAULT_TIMEOUT_SECS)
        return DEFAULT_TIMEOUT_SECS
    return val if val > 0 else None


async def generate_content_with_timeout(provider, **kwargs):
    """Call provider.generate_content(**kwargs) with a bounded client-side timeout.

    Runs the blocking call in a dedicated thread pool; on timeout raises
    ProviderTimeoutError (a TimeoutError subclass) so existing fallback handling
    fires. Set ZEN_PROVIDER_TIMEOUT_SECS<=0 to disable (legacy unbounded behavior).
    """
    timeout_secs = _resolve_timeout()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_PROVIDER_EXECUTOR, lambda: provider.generate_content(**kwargs))
    if timeout_secs is None:
        return await fut
    try:
        return await asyncio.wait_for(fut, timeout=timeout_secs)
    except (asyncio.TimeoutError, TimeoutError):
        model_name = kwargs.get("model_name")
        logger.error(
            "provider.generate_content timed out after %ss (model=%s) — raising "
            "ProviderTimeoutError to trigger fallback. NOTE: the worker thread may "
            "leak until the SDK socket returns (see module docstring).",
            timeout_secs, model_name,
        )
        raise ProviderTimeoutError(timeout_secs, provider=provider, model_name=model_name)
