"""Nettoyage de texte et détection de langue (texte original conservé en raw ; version normalisée ajoutée)."""
import html
import re
import unicodedata

from langdetect import DetectorFactory, detect_langs

URL_RE, MENTION_RE = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+")
INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]")


def set_seed(seed):
    DetectorFactory.seed = seed


def clean_text(t):
    t = unicodedata.normalize("NFKC", html.unescape(str(t)))
    t = INVISIBLE_RE.sub("", t)
    t = URL_RE.sub("[URL]", t)
    t = MENTION_RE.sub("@user", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_key(t):
    """Clé de comparaison pour détecter les textes identiques."""
    return re.sub(r"[\W_]+", " ", clean_text(t).lower()).strip()


def detect_lang(text):
    """langdetect est peu fiable sur les textes courts : 'und' (indéterminé) plutôt que de forcer."""
    if len(re.sub(r"[\W\d_]+", "", text)) < 3:
        return "und", 0.0
    try:
        best = detect_langs(text)[0]
        return best.lang, round(float(best.prob), 3)
    except Exception:
        return "und", 0.0
