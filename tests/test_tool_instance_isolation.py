"""Concurrent MCP tool calls must NOT share a tool instance.

REGRESSION GUARD for the 2026-08-12 "voided Gemini seat" (Genesis, zen-seat goal-pack).

TOOLS in server.py holds one instance per tool for the process lifetime, and both
tool paths keep per-call state on `self` (tools/simple/base.py: _current_model_name,
_model_context; workflow_mixin: work_history, consolidated_findings, _served_model).
MCP dispatches CallToolRequests concurrently, so two overlapping calls used to share
one instance and the later starter overwrote the earlier one's model identity
mid-flight. Observed live as `requested: gemini-3.1-pro-preview / served: gpt-5.4-
2026-03-05` in two independent lanes (Lauren, Azul) on 2026-08-12 — a HEALTHY Gemini
seat reported as substituted, and, in the opposite interleaving, another family's
answer returned under a VERIFIED stamp.

This test asserts the property that actually prevents it: per-call instance isolation.
It deliberately does NOT assert "served_model is correct" — that is the symptom, and a
symptom-level test would pass again the moment a different shared attribute is added.
No network: the tool under test is model-free and its execute() is stubbed.
"""

import asyncio

import server


async def test_concurrent_calls_get_distinct_tool_instances(monkeypatch):
    """Two overlapping calls must each own their tool instance and their own state."""
    seen_ids = []
    a_is_running = asyncio.Event()
    b_has_mutated = asyncio.Event()

    tool_cls = type(server.TOOLS["version"])

    async def fake_execute(self, arguments):
        probe = arguments["probe"]
        seen_ids.append(id(self))
        # Per-call state written to `self` — the exact shape both real tool paths use.
        self._probe_marker = probe
        if probe == "A":
            a_is_running.set()
            await b_has_mutated.wait()  # force B to write its state inside A's flight
        else:
            await a_is_running.wait()
            b_has_mutated.set()
        # THE INVARIANT: the other in-flight call did not overwrite our identity.
        assert self._probe_marker == probe, (
            f"tool state bled across concurrent calls: {probe} observed {self._probe_marker}"
        )
        return []

    monkeypatch.setattr(tool_cls, "execute", fake_execute, raising=True)
    monkeypatch.setattr(tool_cls, "requires_model", lambda self: False, raising=False)

    await asyncio.wait_for(
        asyncio.gather(
            server.handle_call_tool("version", {"probe": "A"}),
            server.handle_call_tool("version", {"probe": "B"}),
        ),
        timeout=10,
    )

    assert len(seen_ids) == 2
    assert len(set(seen_ids)) == 2, "concurrent calls shared a single tool instance"
    # The process-global singleton must never be the object a call mutates.
    assert id(server.TOOLS["version"]) not in seen_ids


async def test_registry_singleton_is_not_mutated_by_a_call(monkeypatch):
    """Negative control: per-call writes must not land on the shared registry object."""
    tool_cls = type(server.TOOLS["version"])
    singleton = server.TOOLS["version"]

    async def fake_execute(self, arguments):
        self._probe_marker = "written"
        return []

    monkeypatch.setattr(tool_cls, "execute", fake_execute, raising=True)
    monkeypatch.setattr(tool_cls, "requires_model", lambda self: False, raising=False)

    await server.handle_call_tool("version", {"probe": "solo"})

    assert not hasattr(singleton, "_probe_marker"), (
        "a single call mutated the process-global tool instance"
    )
