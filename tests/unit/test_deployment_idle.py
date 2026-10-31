import sqlite3
from pathlib import Path

from scripts.check_deployment_idle import deployment_activity


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE meeting_sessions (state TEXT NOT NULL)")
    connection.execute("CREATE TABLE processing_jobs (status TEXT NOT NULL)")
    return connection


def test_missing_deployment_database_is_idle(tmp_path: Path) -> None:
    assert deployment_activity(tmp_path / "missing.db") == {
        "active_meetings": 0,
        "executable_jobs": 0,
    }


def test_deployment_activity_reports_only_aggregate_active_counts(tmp_path: Path) -> None:
    path = tmp_path / "lingxu.db"
    connection = _database(path)
    try:
        connection.executemany(
            "INSERT INTO meeting_sessions(state) VALUES (?)",
            [("RUNNING",), ("PAUSED",), ("COMPLETED",), ("DELETED",)],
        )
        connection.executemany(
            "INSERT INTO processing_jobs(status) VALUES (?)",
            [("PENDING",), ("RUNNING",), ("COMPLETED",), ("FAILED",)],
        )
        connection.commit()
    finally:
        connection.close()

    assert deployment_activity(path) == {
        "active_meetings": 2,
        "executable_jobs": 2,
    }


def test_terminal_deployment_database_is_idle(tmp_path: Path) -> None:
    path = tmp_path / "lingxu.db"
    connection = _database(path)
    try:
        connection.execute("INSERT INTO meeting_sessions(state) VALUES ('PUBLISHED')")
        connection.execute("INSERT INTO processing_jobs(status) VALUES ('FAILED')")
        connection.commit()
    finally:
        connection.close()

    assert deployment_activity(path) == {
        "active_meetings": 0,
        "executable_jobs": 0,
    }
