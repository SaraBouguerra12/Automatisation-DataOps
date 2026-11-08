"""Fusion multimodale image + texte : attention croisée bidirectionnelle + unité à porte (GMU).

Entrées = représentations de TOKENS déjà extraites par des encodeurs pré-entraînés (CLIP) et mises en cache.
Une modalité absente est remplacée par un jeton « absent » appris (cas Sentiment140 : texte seul).

Notation (voir rapport, section 5) : h_I in R^{n x d}, h_T in R^{m x d}, d = dimension commune.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(h, mask):
    """Moyenne des tokens valides. h : [B, L, d], mask : [B, L] (True = valide)."""
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class CrossBlock(nn.Module):
    """Un bloc : texte <- image, image <- texte (attention croisée), puis réseau feed-forward."""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.ca_t = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)  # Q = texte, K/V = image
        self.ca_i = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)  # Q = image, K/V = texte
        self.ln_t1, self.ln_i1, self.ln_t2, self.ln_i2 = (nn.LayerNorm(d) for _ in range(4))
        self.ffn_t = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.ffn_i = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, h_t, h_i, m_t, m_i):
        a_t, _ = self.ca_t(h_t, h_i, h_i, key_padding_mask=~m_i)
        a_i, _ = self.ca_i(h_i, h_t, h_t, key_padding_mask=~m_t)
        h_t, h_i = self.ln_t1(h_t + a_t), self.ln_i1(h_i + a_i)
        return self.ln_t2(h_t + self.ffn_t(h_t)), self.ln_i2(h_i + self.ffn_i(h_i))


class MultimodalFusion(nn.Module):
    def __init__(self, d_img=768, d_txt=512, d=256, n_heads=4, n_layers=2, dropout=0.1, p_modality_dropout=0.15):
        super().__init__()
        self.p_md = p_modality_dropout
        self.proj_i = nn.Sequential(nn.Linear(d_img, d), nn.LayerNorm(d))
        self.proj_t = nn.Sequential(nn.Linear(d_txt, d), nn.LayerNorm(d))
        self.type_i, self.type_t = nn.Parameter(torch.zeros(1, 1, d)), nn.Parameter(torch.zeros(1, 1, d))
        self.absent_i, self.absent_t = nn.Parameter(torch.zeros(1, 1, d)), nn.Parameter(torch.zeros(1, 1, d))
        for p in (self.type_i, self.type_t, self.absent_i, self.absent_t):
            nn.init.normal_(p, std=0.02)
        self.blocks = nn.ModuleList(CrossBlock(d, n_heads, dropout) for _ in range(n_layers))
        self.gate, self.w_i, self.w_t = nn.Linear(2 * d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.aux_i, self.aux_t = nn.Linear(d, 1), nn.Linear(d, 1)  # têtes unimodales auxiliaires (entraînement)

    def _modality_dropout(self, has_i, has_t):
        """Retire aléatoirement une modalité (jamais les deux) : robustesse et équilibre entre modalités."""
        drop_i = torch.rand(has_i.shape, device=has_i.device) < self.p_md
        drop_t = (torch.rand(has_t.shape, device=has_t.device) < self.p_md) & ~drop_i
        return has_i & ~drop_i, has_t & ~drop_t

    def forward(self, img_tokens, txt_tokens, img_mask, txt_mask, has_img=None, has_txt=None):
        B = img_tokens.size(0)
        ones = torch.ones(B, dtype=torch.bool, device=img_tokens.device)
        has_img = ones if has_img is None else has_img
        has_txt = ones if has_txt is None else has_txt
        if self.training and self.p_md > 0:
            has_img, has_txt = self._modality_dropout(has_img, has_txt)

        h_i = self.proj_i(img_tokens) + self.type_i
        h_t = self.proj_t(txt_tokens) + self.type_t
        # modalité absente : jeton appris en première position, tout le reste masqué
        h_i = torch.where(has_img[:, None, None], h_i, self.absent_i.expand_as(h_i))
        h_t = torch.where(has_txt[:, None, None], h_t, self.absent_t.expand_as(h_t))
        m_i, m_t = img_mask & has_img[:, None], txt_mask & has_txt[:, None]
        m_i, m_t = m_i.clone(), m_t.clone()
        m_i[:, 0] |= ~has_img
        m_t[:, 0] |= ~has_txt

        for blk in self.blocks:
            h_t, h_i = blk(h_t, h_i, m_t, m_i)
        p_i, p_t = masked_mean(h_i, m_i), masked_mean(h_t, m_t)
        g = torch.sigmoid(self.gate(torch.cat([p_i, p_t], -1)))  # GMU : porte apprise par dimension
        z = g * torch.tanh(self.w_i(p_i)) + (1 - g) * torch.tanh(self.w_t(p_t))
        return dict(logit=self.head(z).squeeze(-1), logit_i=self.aux_i(p_i).squeeze(-1),
                    logit_t=self.aux_t(p_t).squeeze(-1), gate=g, has_img=has_img, has_txt=has_txt)


def _masked_bce(logit, y, keep, pos_weight):
    if keep.sum() == 0:
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[keep], y[keep], pos_weight=pos_weight)


def fusion_loss(out, y, pos_weight=None, lambda_aux=0.3):
    """L = L_tâche + lambda * (L_image + L_texte) ; les pertes auxiliaires ne portent que sur les modalités présentes."""
    main = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos_weight)
    aux = (_masked_bce(out["logit_i"], y, out["has_img"], pos_weight)
           + _masked_bce(out["logit_t"], y, out["has_txt"], pos_weight))
    return main + lambda_aux * aux, dict(main=main.detach(), aux=aux.detach())


class PooledBaseline(nn.Module):
    """Baselines : mode 'image' ou 'text' (unimodal) et 'concat' (fusion simple par concaténation)."""

    def __init__(self, mode, d_img=768, d_txt=512, d=256, dropout=0.1):
        super().__init__()
        assert mode in ("image", "text", "concat")
        self.mode = mode
        din = {"image": d_img, "text": d_txt, "concat": d_img + d_txt}[mode]
        self.mlp = nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, img_tokens, txt_tokens, img_mask, txt_mask, **_):
        p_i, p_t = masked_mean(img_tokens, img_mask), masked_mean(txt_tokens, txt_mask)
        x = {"image": p_i, "text": p_t, "concat": torch.cat([p_i, p_t], -1)}[self.mode]
        return self.mlp(x).squeeze(-1)


def late_fusion(logit_i, logit_t, w=0.5):
    """Fusion tardive : moyenne pondérée des décisions des modèles unimodaux."""
    return w * logit_i + (1 - w) * logit_t
