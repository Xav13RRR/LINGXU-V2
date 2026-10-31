"""Content-safe deployment guard: report only active Meeting and Job counts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

_TERMINAL_MEETING_STATES = frozenset(
    {
        "COMPLETED",
        "PUBLISHED",
        "STOPPED",
        "INTERRUPTED",
        "FAILED_CAPTURE",
        "FAILED_REALTIME_ASR",
        "FAILED_TTS",
        "FAILED_POST_PROCESSING",
        "DELETED",
    }
)
_EXECUTABLE_JOB_STATES = frozenset({"PENDING", "RUNNING"})


def deployment_activity(database_path: Path) -> dict[str, int]:
    """Return only aggregate activity counts from an existing deployment database."""

    if not database_path.is_file():
        return {"active_meetings": 0, "executable_jobs": 0}
    connection = sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        meeting_counts = dict(
            connection.execute(
                "SELECT state, COUNT(*) FROM meeting_sessions GROUP BY state"
            ).fetchall()
        )
        job_counts = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM processing_jobs GROUP BY status"
            ).fetchall()
        )
    finally:
        connection.close()
    active_meetings = sum(
        count for state, count in meeting_counts.items() if state not in _TERMINAL_MEETING_STATES
    )
    executable_jobs = sum(
        count for state, count in job_counts.items() if state in _EXECUTABLE_JOB_STATES
    )
    return {
        "active_meetings": active_meetings,
        "executable_jobs": executable_jobs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    activity = deployment_activity(Path("/data/lingxu.db"))
    print(json.dumps(activity, separators=(",", ":")))
    return 2 if any(activity.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
