from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Make directory-based integration selection complete and deterministic."""

    for item in items:
        if "integration" in item.path.parts:
            item.add_marker(pytest.mark.integration)
