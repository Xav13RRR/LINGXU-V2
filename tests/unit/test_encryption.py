from pathlib import Path

from lingxu.storage.encryption import FileCipher


def test_encrypted_audio_is_not_directly_playable(tmp_path: Path) -> None:
    source = tmp_path / "speaker.flac"
    encrypted = tmp_path / "speaker.flac.enc"
    restored = tmp_path / "speaker.restored.flac"
    plaintext = b"fLaC" + (b"\x00\x01\x02\x03" * 100)
    source.write_bytes(plaintext)

    cipher = FileCipher.from_secret(FileCipher.generate_key())
    cipher.encrypt_file(source, encrypted)

    envelope = encrypted.read_bytes()
    assert not envelope.startswith(b"fLaC")
    assert plaintext not in envelope

    cipher.decrypt_file(encrypted, restored)
    assert restored.read_bytes() == plaintext
