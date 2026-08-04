"""Tiny NDJSON debug logger for session 4b92ea.

Writes to the Cursor debug log path and to WORKING_DIR as a Docker-friendly fallback.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_SESSION = "4b92ea"
_PRIMARY = Path("/home/sanda/Desktop/project/cf_memgraphrag/.cursor/debug-4b92ea.log")


def agent_dbg(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    payload = {
        "sessionId": _SESSION,
        "id": f"log_{int(time.time() * 1000)}_{hypothesis_id}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data or {},
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    paths = [_PRIMARY]
    wd = os.getenv("WORKING_DIR")
    if wd:
        paths.append(Path(wd) / "debug-4b92ea.log")
    for path in paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            continue
    # #endregion
