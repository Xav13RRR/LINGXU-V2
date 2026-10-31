"""Authenticated encryption for persisted Session artifacts."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from lingxu.domain.errors import EncryptionConfigurationError
from lingxu.storage.session_files import atomic_write_bytes

_MAGIC = b"LXENC\x01"
_NONCE_BYTES = 12
_AAD = b"lingxu:file:v1"


class FileCipher:
    """AES-256-GCM file envelope with an explicit non-media header."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise EncryptionConfigurationError("FILE_ENCRYPTION_KEY must decode to 32 bytes")
        self._cipher = AESGCM(key)

    @staticmethod
    def generate_key() -> str:
        """Generate a URL-safe configuration value without logging it."""

        return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")

    @classmethod
    def from_secret(cls, secret: SecretStr | str | None) -> FileCipher:
        if secret is None:
            raise EncryptionConfigurationError("FILE_ENCRYPTION_KEY is not configured")
        encoded = secret.get_secret_value() if isinstance(secret, SecretStr) else secret
        try:
            key = base64.urlsafe_b64decode(encoded.encode("ascii"))
        except (ValueError, UnicodeError) as exc:
            raise EncryptionConfigurationError(
                "FILE_ENCRYPTION_KEY is not valid URL-safe base64"
            ) from exc
        return cls(key)

    def encrypt(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(_NONCE_BYTES)
        return _MAGIC + nonce + self._cipher.encrypt(nonce, plaintext, _AAD)

    def decrypt(self, envelope: bytes) -> bytes:
        if not envelope.startswith(_MAGIC):
            raise ValueError("encrypted file header is invalid")
        nonce_start = len(_MAGIC)
        nonce_end = nonce_start + _NONCE_BYTES
        if len(envelope) <= nonce_end:
            raise ValueError("encrypted file is truncated")
        return self._cipher.decrypt(
            envelope[nonce_start:nonce_end],
            envelope[nonce_end:],
            _AAD,
        )

    def encrypt_file(self, source: Path, destination: Path) -> str:
        """Encrypt a file atomically and return the encrypted SHA-256."""

        return atomic_write_bytes(destination, self.encrypt(source.read_bytes()))

    def decrypt_file(self, source: Path, destination: Path) -> str:
        """Decrypt a file atomically for Worker-only processing."""

        return atomic_write_bytes(destination, self.decrypt(source.read_bytes()))
