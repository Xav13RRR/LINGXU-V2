"""Packaged, read-only Lingxu runtime resources."""

from importlib.resources import files
from importlib.resources.abc import Traversable

_DEEPSEEK_FULL_TRANSCRIPT_MINUTES = "deepseek_full_transcript_minutes.md"


def deepseek_full_transcript_minutes_template() -> Traversable:
    """Return the packaged final-minutes template without assuming a filesystem install."""

    return files(__package__).joinpath(_DEEPSEEK_FULL_TRANSCRIPT_MINUTES)


__all__ = ["deepseek_full_transcript_minutes_template"]
