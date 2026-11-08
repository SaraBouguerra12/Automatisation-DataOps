"""Passe avant sur un VRAI batch issu du pipeline (vérification des dimensions).

Usage : python -m src.fusion.demo_forward     (nécessite : pip install -r requirements-fusion.txt ; `dvc pull`)
Le premier lancement télécharge CLIP (environ 600 Mo).
"""
import pandas as pd
import torch
from PIL import Image

from src.common.config import get_paths, load_config
from src.fusion.encoders import encode_batch, load_clip
from src.fusion.model import MultimodalFusion, fusion_loss


def main(n_img=6, n_txt_only=2):
    cfg = load_config()
    P = get_paths(cfg)
    fc = cfg["fusion"]
    torch.manual_seed(cfg["seed"])
    m = pd.read_parquet(P.processed / "manifest.parquet")
    hm = m[(m.source == "hateful_memes") & (m.split == "train")].head(n_img)
    s140 = m[m.source == "sentiment140"].head(n_txt_only)

    clip, proc = load_clip(fc["encoder"])
    images = [Image.open(P.base / p).convert("RGB") for p in hm.img_path]
    blank = Image.new("RGB", (224, 224))  # remplaçant neutre : l'image d'un tweet est ABSENTE (has_img = False)
    images += [blank] * len(s140)
    texts = hm.text_clean.tolist() + s140.text_clean.tolist()
    v, t, tmask = encode_batch(clip, proc, images, texts)
    has_img = torch.tensor([True] * len(hm) + [False] * len(s140))
    y = torch.tensor(hm.hate_binary.astype(float).tolist() + [0.0] * len(s140))

    model = MultimodalFusion(d_img=v.size(-1), d_txt=t.size(-1), d=fc["d"], n_heads=fc["n_heads"],
                             n_layers=fc["n_layers"], dropout=fc["dropout"], p_modality_dropout=0.0).eval()
    img_mask = torch.ones(v.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        out = model(v, t, img_mask, tmask, has_img=has_img)
    loss, parts = fusion_loss(out, y)
    print("tokens image :", tuple(v.shape), "| tokens texte :", tuple(t.shape), "| masque texte :", tuple(tmask.shape))
    print("logit :", tuple(out["logit"].shape), "| porte :", tuple(out["gate"].shape), "| perte :", float(loss))
    assert out["logit"].shape == (len(texts),) and torch.isfinite(loss)
    print("OK : passe avant valide sur un batch réel du pipeline.")


if __name__ == "__main__":
    main()
