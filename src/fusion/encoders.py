"""Encodeurs pré-entraînés (CLIP) : extraction des représentations de tokens image et texte (import paresseux)."""
import torch


def load_clip(name="openai/clip-vit-base-patch32", device="cpu"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(name).eval().to(device)
    return model, CLIPProcessor.from_pretrained(name)


@torch.no_grad()
def encode_batch(model, processor, images, texts, device="cpu"):
    """Retourne (tokens image [B,50,768], tokens texte [B,m,512], masque texte [B,m] bool)."""
    inp = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True,
                    truncation=True, max_length=77).to(device)
    v = model.vision_model(pixel_values=inp["pixel_values"]).last_hidden_state
    t = model.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]).last_hidden_state
    return v, t, inp["attention_mask"].bool()
