"""Tests du module de fusion (ignorés si PyTorch n'est pas installé : pip install -r requirements-fusion.txt)."""
import pytest

torch = pytest.importorskip("torch")

from src.fusion.model import MultimodalFusion, PooledBaseline, fusion_loss, late_fusion  # noqa: E402

B, N, M, DI, DT = 4, 50, 12, 768, 512


@pytest.fixture()
def batch():
    g = torch.Generator().manual_seed(0)
    img, txt = torch.randn(B, N, DI, generator=g), torch.randn(B, M, DT, generator=g)
    img_mask = torch.ones(B, N, dtype=torch.bool)
    txt_mask = torch.ones(B, M, dtype=torch.bool)
    txt_mask[:, 8:] = False  # padding
    return img, txt, img_mask, txt_mask


def make(p=0.0):
    torch.manual_seed(0)
    return MultimodalFusion(d_img=DI, d_txt=DT, d=64, n_heads=4, n_layers=2, p_modality_dropout=p).eval()


def test_forward_shapes(batch):
    out = make()(*batch)
    assert out["logit"].shape == (B,) and out["gate"].shape == (B, 64)
    assert torch.isfinite(out["logit"]).all()


def test_missing_image_is_ignored_for_that_sample(batch):
    img, txt, mi, mt = batch
    has_img = torch.tensor([False, True, True, True])
    model = make()
    a = model(img, txt, mi, mt, has_img=has_img)["logit"]
    b = model(torch.randn_like(img), txt, mi, mt, has_img=has_img)["logit"]
    assert torch.allclose(a[0], b[0], atol=1e-5)      # image absente : son contenu n'a aucun effet
    assert not torch.allclose(a[1], b[1], atol=1e-5)  # image présente : le contenu compte


def test_text_padding_does_not_change_output(batch):
    img, txt, mi, mt = batch
    txt2 = txt.clone()
    txt2[:, 8:] = torch.randn_like(txt2[:, 8:])
    model = make()
    assert torch.allclose(model(img, txt, mi, mt)["logit"], model(img, txt2, mi, mt)["logit"], atol=1e-5)


def test_modality_dropout_never_removes_both(batch):
    model = make(p=0.9).train()
    for _ in range(20):
        out = model(*batch)
        assert (out["has_img"] | out["has_txt"]).all()


def test_loss_backward_and_gradients(batch):
    model = make(p=0.2).train()
    out = model(*batch)
    loss, parts = fusion_loss(out, torch.tensor([0.0, 1.0, 1.0, 0.0]), pos_weight=torch.tensor(1.7))
    loss.backward()
    assert torch.isfinite(loss) and model.head[0].weight.grad is not None and parts["aux"] >= 0


def test_baselines_and_late_fusion(batch):
    for mode in ("image", "text", "concat"):
        assert PooledBaseline(mode, DI, DT, 64)(*batch).shape == (B,)
    assert late_fusion(torch.zeros(B), torch.ones(B)).allclose(torch.full((B,), 0.5))
