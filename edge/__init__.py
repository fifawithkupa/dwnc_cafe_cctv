"""Edge box tooling (benchmarks, exports, box inspection).

Windows consoles here run cp949, which cannot encode the warning sign or an
em dash.  A measurement that took minutes must not die on its last print --
``bench_decode`` did exactly that, losing both the summary table and the report
file it had not written yet -- so every entry point makes its output replace
what the console cannot encode instead of raising.
"""

from __future__ import annotations

import sys


def tolerant_stdout() -> None:
    """Never let an unencodable character kill a finished measurement."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # detached or already-closed stream
            pass
