import numpy as np
import pandas as pd

from src.alignment.dedup_groups import hamming_roots, make_groups
from src.alignment.split import assign_splits
from src.common.config import load_config
from src.common.runlog import RunLog
from src.preprocessing.text import clean_text, detect_lang, norm_key


def test_clean_text():
    assert clean_text("Hello&amp;  WORLD http://x.co @bob\u200b") == "Hello& WORLD [URL] @user"
    assert norm_key("Hello,  World!!") == "hello world"


def test_detect_lang_short_text_is_undetermined():
    assert detect_lang("ok") == ("und", 0.0)


def test_hamming_roots_groups_near_identical_hashes():
    roots = hamming_roots(["0000000000000000", "0000000000000001", "ffffffffffffffff"], max_dist=4)
    assert roots[0] == roots[1] and roots[0] != roots[2]


def test_make_groups_by_text_and_image():
    df = pd.DataFrame(dict(img_hash=["0" * 16, "f" * 16, "0" * 15 + "1", "a5" * 8],
                           text_norm=["a long enough identical text", "a long enough identical text", "x", "y"]))
    g = make_groups(df, 4, 15)
    assert g[0] == g[1] == g[2] and g[3] != g[0]  # 0~2 par pHash, 0~1 par texte identique


def test_split_has_no_group_leak():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(dict(sample_id=[f"s{i}" for i in range(n)], source="hateful_memes",
                           hate_binary=pd.array(rng.integers(0, 2, n), dtype="Int64"),
                           group_id=[f"g{i // 3}" for i in range(n)], label="x"))
    out = assign_splits(df, load_config())
    assert out.groupby("group_id").split.nunique().max() == 1
    assert set(out.split) == {"train", "val", "test"}


def test_runlog_counters_and_rejects(tmp_path):
    rl = RunLog("t", tmp_path)
    rl.reject("reddit", "p1", "texte vide")
    rl.reject_bulk("sentiment140", [1, 2, 3], "id dupliqué")
    rl.save()
    cnt = pd.DataFrame(rl.counters_records())
    assert cnt[cnt.kind == "rejected"].n.sum() == 4
    assert (tmp_path / "t_rejects.csv").exists() and (tmp_path / "t.json").exists()
