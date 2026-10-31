"""Run a content-safe, read-only audit of retained Meeting records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lingxu.cli.retained_audit import RetainedAuditError, audit_retained_meetings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--sessions-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--probe-root", type=Path)
    args = parser.parse_args()
    try:
        result = audit_retained_meetings(
            args.database,
            args.sessions_root,
            backup_root=args.backup_root,
            probe_root=args.probe_root,
        )
    except RetainedAuditError as exc:
        result = {"content_safe": True, "failure_code": str(exc), "status": "FAILED"}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
