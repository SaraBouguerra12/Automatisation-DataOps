"""Intégrité d'image (Pillow), taille minimale, SHA-256 et hash perceptuel."""
from pathlib import Path

import imagehash
from PIL import Image

from src.common.io_utils import sha256_file


def _info(ok, reason="", detail="", sha256=None, phash=None, w=None, h=None):
    return dict(ok=ok, reason=reason, detail=detail, sha256=sha256, phash=phash, w=w, h=h)


def inspect_image(path, min_side):
    p = Path(path)
    if not p.exists():
        return _info(False, "image absente")
    try:
        if p.stat().st_size == 0:
            return _info(False, "fichier image vide")
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < min_side:
                return _info(False, "image trop petite", f"{w}x{h}", w=w, h=h)
            return _info(True, sha256=sha256_file(p), phash=str(imagehash.phash(im)), w=w, h=h)
    except Exception as e:
        return _info(False, "image illisible", repr(e)[:100])
