import hashlib
from pathlib import Path


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rel(path, base):
    """Chemin relatif à BASE (portable d'une machine à l'autre)."""
    return Path(path).relative_to(base).as_posix()
