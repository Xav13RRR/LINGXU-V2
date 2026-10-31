from __future__ import annotations

from types import SimpleNamespace

from lingxu.voice.sink import AssistantOnlyPcmSink


class FakeAssistantReceiver:
    def __init__(self) -> None:
        self.received: list[tuple[int, int, bytes]] = []

    def receive(
        self,
        *,
        user_id: int,
        received_monotonic_ns: int,
        pcm_s16le_48k_stereo: bytes,
    ) -> bool:
        self.received.append((user_id, received_monotonic_ns, pcm_s16le_48k_stereo))
        return True


def test_assistant_only_sink_copies_human_pcm_without_other_dependencies() -> None:
    receiver = FakeAssistantReceiver()
    sink = AssistantOnlyPcmSink(assistant=receiver, clock_ns=lambda: 456)
    mutable = bytearray(b"\x01\x00\x02\x00")

    sink.write(None, SimpleNamespace(pcm=mutable))  # type: ignore[arg-type]
    sink.write(
        SimpleNamespace(id=1, bot=True),
        SimpleNamespace(pcm=mutable),  # type: ignore[arg-type]
    )
    sink.write(
        SimpleNamespace(id=2, bot=False),
        SimpleNamespace(pcm=mutable),  # type: ignore[arg-type]
    )
    mutable[:] = b"\0\0\0\0"

    assert receiver.received == [(2, 456, b"\x01\x00\x02\x00")]
