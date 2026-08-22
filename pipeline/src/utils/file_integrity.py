"""Small file-integrity primitives shared across artifact owners."""

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_file_sha256 = file_sha256


__all__ = ["_file_sha256", "file_sha256"]
