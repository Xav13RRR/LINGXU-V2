import inspect

import pytest
from discord.ext import voice_recv


@pytest.mark.integration
def test_pinned_voice_fork_exposes_inbound_dave_and_decode_hardening() -> None:
    assert hasattr(voice_recv, "parse_dave_payload")

    opus_source = inspect.getsource(voice_recv.opus)
    reader_source = inspect.getsource(voice_recv.reader)

    assert "dave_session" in opus_source or "dave_session" in reader_source
    assert "OpusError" in opus_source or "OpusError" in reader_source
