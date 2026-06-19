#!/usr/bin/env python3
"""LOG_LEVEL arming check for the zen-MCP hang supervisor — SOL-338 Layer 2 (R7).

The OUTER supervisor's watchdog ARMS only while it has seen a `TOOL_CALL:` marker,
which the zen server emits at logging.INFO. If a host runs the server with an
effective level ABOVE INFO (WARNING/ERROR/CRITICAL), that marker is suppressed,
`is_in_call` never flips True, and the outer hard-kill NEVER fires — the backstop
is inert (the inner asyncio timeout still protects each provider call, but the
process-level reap is off). LOG_LEVEL<=INFO is therefore a LOAD-BEARING per-host
precondition that this check asserts.

This reads LOG_LEVEL exactly as server.py does:
    log_level = (get_env("LOG_LEVEL", "DEBUG") or "DEBUG").upper()
DEBUG and INFO both arm (<=INFO). WARNING/ERROR/CRITICAL do NOT.

USAGE
    <host-python> scripts/zen_check_log_level.py
    # exit 0 + "LOG_LEVEL OK" => outer backstop will arm on this host.
    # exit 1 + "LOG_LEVEL TOO HIGH" => fix before trusting the outer layer.
"""
import os
import sys

_ARMING = {"DEBUG", "INFO", "NOTSET"}


def main() -> int:
    lvl = (os.getenv("LOG_LEVEL", "DEBUG") or "DEBUG").upper()
    if lvl in _ARMING:
        print(f"LOG_LEVEL OK: '{lvl}' <= INFO — outer backstop will arm.")
        return 0
    print(
        f"LOG_LEVEL TOO HIGH: '{lvl}' suppresses the INFO TOOL_CALL marker — the "
        f"outer hard-kill backstop will NOT arm (inner asyncio timeout still "
        f"active). Set LOG_LEVEL=INFO (or DEBUG) on this host's zen server env.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
