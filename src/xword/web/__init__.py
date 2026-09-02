"""The web surface: session-oriented solving with a full agent trace.

The CLI solves one puzzle in the foreground. The web UI needs something the
library did not previously offer: solves that **outlive the request that
started them**, so a browser can start three puzzles, navigate between them,
watch any one of their traces from any point, and stop one without touching
the others.

That is what this package adds:

``trace``
    The event model. Coarse agent steps *and* individual model calls --
    prompt, tool definition, tool call, response, tokens -- as an
    append-only, cursor-addressable log.
``sessions``
    The registry. One background thread per solve, cooperative cancellation,
    and replay-from-cursor subscription so a late or returning subscriber
    sees the whole trace rather than only what happens next.

Nothing here is imported by the CLI or the solver, so the library keeps
working with no web dependencies installed.
"""

from xword.web.trace import (
    LLMCallRecord,
    SessionState,
    TraceEvent,
    TraceLog,
)

__all__ = [
    "LLMCallRecord",
    "SessionState",
    "TraceEvent",
    "TraceLog",
]
