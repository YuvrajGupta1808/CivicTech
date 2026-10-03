"""M4 (stretch): ordinal Transformer with OSM <-> DDOT cross-attention for tabular data.

Ported (copy-adapt, no import) from AVB-Engage / AVT-CA:
  Attention, Mlp          <- AVTCA-Research/models/transformer.py (torch only; SDPA instead of manual softmax)
  AttentionPool           <- AVTCA-Research/models/multimodal_cnn.py
  OrdinalDistanceCrossEntropy <- AVTCA-Research/src/engine/runtime.py
  source ("modality") dropout + eval-time ablation <- multimodal_cnn.py forward()

Tokenisation is FT-Transformer style: one token per feature.
  numeric  -> standardised value * w_f + b_f   (NaN -> learned per-feature missing token)
  category -> per-feature embedding row         (index 0 = missing, 1 = unseen/rare)
  every token gets a learned feature-identity embedding.
Streams: DDOT tokens (ddot_*) + net_* ; OSM tokens (osmf_*) + net_*  (net tokens are appended to BOTH).
`mode="xattn"`: `exchanges` rounds of simultaneous X_o += CA(q=X_o, kv=X_d); X_d += CA(q=X_d, kv=X_o),
each followed by a pre-norm FFN residual per stream, then AttentionPool per stream -> concat -> Linear(3).
`mode="concat"` (E8): 2-layer self-attention encoder over all tokens (same tokeniser / pools / head).
`ablate` in {'none','no_osm','no_ddot'}: replace one stream's own tokens by their missing tokens (E7).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

STREAMS = {"ddot": "ddot_", "osm": "osmf_", "net": "net_"}
STREAM_ID = {"ddot": 0, "osm": 1, "net": 2}
ABLATIONS = ("none", "no_osm", "no_ddot")


# --------------------------------------------------------------------------- ported blocks
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    """Cross/self attention: queries from x_q, keys/values from x (AVTCA signature kept)."""

    def __init__(self, in_dim_k, in_dim_q, out_dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.q = nn.Linear(in_dim_q, out_dim, bias=qkv_bias)
        self.kv = nn.Linear(in_dim_k, out_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_q):
        B, Nk, _ = x.shape
        _, Nq, _ = x_q.shape
        h = self.num_heads
        q = self.q(x_q).reshape(B, Nq, h, -1).transpose(1, 2)
        k, v = self.kv(x).reshape(B, Nk, 2, h, -1).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.proj_drop(self.proj(out))


class AttentionPool(nn.Module):
    """Learned weighted sum over tokens."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.proj(x), dim=1)
        return (w * x).sum(dim=1)


class OrdinalDistanceCrossEntropy(nn.Module):
    def __init__(self, n_classes, distance_weight=0.35, label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.n_classes = int(n_classes)
        self.distance_weight = float(distance_weight)
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        pos = torch.arange(self.n_classes, dtype=torch.float32)
        self.register_buffer("class_positions", pos / max(self.n_classes - 1, 1))

    def forward(self, logits, targets):
        ce = self.cross_entropy(logits, targets)
        probs = torch.softmax(logits, dim=1)
        tpos = self.class_positions.index_select(0, targets)
        dist = torch.abs(self.class_positions.unsqueeze(0) - tpos.unsqueeze(1))
        return ce + self.distance_weight * (probs * dist).sum(dim=1).mean()


class CrossBlock(nn.Module):
    """Pre-norm cross-attention: returns CA(q=norm(xq), kv=norm(xkv)); caller adds the residual."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, d, d, heads, attn_drop=drop, proj_drop=drop)

    def forward(self, xq, xkv):
        return self.attn(self.nk(xkv), self.nq(xq))


class FFNResidual(nn.Module):
    def __init__(self, d, drop, ratio=2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = Mlp(d, d * ratio, drop=drop)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class SelfBlock(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = Attention(d, d, d, heads, attn_drop=drop, proj_drop=drop)
        self.ffn = FFNResidual(d, drop)

    def forward(self, x):
        h = self.norm(x)
        return self.ffn(x + self.attn(h, h))


# --------------------------------------------------------------------------- tabular encoding
class TabularEncoder:
    """Fit on the train fold only. Numeric -> standardised float32 (+NaN mask); category -> int codes."""

    MIN_COUNT = 5

    def __init__(self, df: pd.DataFrame):
        self.num_cols, self.cat_cols = [], []
        for s in ("ddot", "osm", "net"):
            for c in df.columns:
                if not c.startswith(STREAMS[s]):
                    continue
                dt = df[c].dtype
                if isinstance(dt, pd.CategoricalDtype) or dt == object or pd.api.types.is_string_dtype(dt):
                    self.cat_cols.append((c, s))
                else:
                    self.num_cols.append((c, s))
        self.stream_ids = [STREAM_ID[s] for _, s in self.num_cols] + [STREAM_ID[s] for _, s in self.cat_cols]
        self.names = [c for c, _ in self.num_cols] + [c for c, _ in self.cat_cols]

    @property
    def n_num(self):
        return len(self.num_cols)

    def fit(self, df: pd.DataFrame):
        self.mean, self.std = [], []
        for c, _ in self.num_cols:
            x = pd.to_numeric(df[c], errors="coerce").astype("float64").to_numpy()
            x[~np.isfinite(x)] = np.nan
            m = np.nanmean(x) if np.isfinite(x).any() else 0.0
            s = np.nanstd(x) if np.isfinite(x).any() else 1.0
            self.mean.append(m)
            self.std.append(s if s > 1e-8 else 1.0)
        self.vocab = []
        for c, _ in self.cat_cols:
            vc = df[c].astype("object").where(df[c].notna(), None).value_counts()
            keep = [k for k, n in vc.items() if n >= self.MIN_COUNT]
            self.vocab.append({k: i + 2 for i, k in enumerate(keep)})
        self.cat_sizes = [len(v) + 2 for v in self.vocab]  # 0 = missing, 1 = unseen/rare
        return self

    def transform(self, df: pd.DataFrame, device="cpu"):
        n = len(df)
        xnum = np.zeros((n, len(self.num_cols)), dtype=np.float32)
        nan = np.zeros((n, len(self.num_cols)), dtype=bool)
        for j, (c, _) in enumerate(self.num_cols):
            x = pd.to_numeric(df[c], errors="coerce").astype("float64").to_numpy()
            bad = ~np.isfinite(x)
            z = np.clip((x - self.mean[j]) / self.std[j], -6.0, 6.0)
            z[bad] = 0.0
            xnum[:, j], nan[:, j] = z, bad
        xcat = np.zeros((n, len(self.cat_cols)), dtype=np.int64)
        for j, (c, _) in enumerate(self.cat_cols):
            s = df[c].astype("object")
            codes = s.map(self.vocab[j]).fillna(1).astype(np.int64).to_numpy()
            codes[s.isna().to_numpy()] = 0
            xcat[:, j] = codes
        return {
            "xnum": torch.from_numpy(xnum).to(device),
            "nan": torch.from_numpy(nan).to(device),
            "xcat": torch.from_numpy(xcat).to(device),
        }


# --------------------------------------------------------------------------- model
class FTTokenizer(nn.Module):
    def __init__(self, n_num, cat_sizes, d):
        super().__init__()
        self.n_num, self.n_cat = n_num, len(cat_sizes)
        bound = 1.0 / math.sqrt(d)
        if n_num:
            self.w = nn.Parameter(torch.empty(n_num, d).uniform_(-bound, bound))
            self.b = nn.Parameter(torch.empty(n_num, d).uniform_(-bound, bound))
            self.miss_num = nn.Parameter(torch.empty(n_num, d).uniform_(-bound, bound))
        if self.n_cat:
            self.cat_emb = nn.Embedding(int(sum(cat_sizes)), d)
            nn.init.uniform_(self.cat_emb.weight, -bound, bound)
            off = np.concatenate([[0], np.cumsum(cat_sizes)[:-1]]).astype(np.int64)
            self.register_buffer("offsets", torch.from_numpy(off))
        self.ident = nn.Parameter(torch.randn(n_num + self.n_cat, d) * 0.02)

    def forward(self, xnum, nan, xcat):
        parts = []
        if self.n_num:
            t = xnum.unsqueeze(-1) * self.w + self.b
            parts.append(torch.where(nan.unsqueeze(-1), self.miss_num, t))
        if self.n_cat:
            parts.append(self.cat_emb(xcat + self.offsets))
        return torch.cat(parts, dim=1) + self.ident

    def missing_tokens(self):
        parts = []
        if self.n_num:
            parts.append(self.miss_num)
        if self.n_cat:
            parts.append(self.cat_emb(self.offsets))  # code 0 of every categorical feature
        return torch.cat(parts, dim=0) + self.ident


class XAttnModel(nn.Module):
    def __init__(self, n_num, cat_sizes, stream_ids, d=32, heads=4, exchanges=2, dropout=0.1,
                 source_dropout=0.15, mode="xattn", n_classes=3):
        super().__init__()
        assert mode in ("xattn", "concat")
        self.mode, self.p_src, self.ablate = mode, source_dropout, "none"
        self.tok = FTTokenizer(n_num, cat_sizes, d)
        sid = torch.tensor(stream_ids)
        self.register_buffer("idx_d", torch.nonzero(sid == 0).flatten())
        self.register_buffer("idx_o", torch.nonzero(sid == 1).flatten())
        self.register_buffer("idx_n", torch.nonzero(sid == 2).flatten())
        if mode == "xattn":
            self.ca_o = nn.ModuleList(CrossBlock(d, heads, dropout) for _ in range(exchanges))
            self.ca_d = nn.ModuleList(CrossBlock(d, heads, dropout) for _ in range(exchanges))
            self.ffn_o = nn.ModuleList(FFNResidual(d, dropout) for _ in range(exchanges))
            self.ffn_d = nn.ModuleList(FFNResidual(d, dropout) for _ in range(exchanges))
        else:
            self.enc = nn.ModuleList(SelfBlock(d, heads, dropout) for _ in range(2))
        self.norm_o, self.norm_d = nn.LayerNorm(d), nn.LayerNorm(d)
        self.pool_o, self.pool_d = AttentionPool(d), AttentionPool(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d, n_classes))

    def _drop_flags(self, B, device):
        do = torch.zeros(B, dtype=torch.bool, device=device)
        dd = torch.zeros(B, dtype=torch.bool, device=device)
        if self.training and self.p_src > 0:
            do = torch.rand(B, device=device) < self.p_src
            dd = torch.rand(B, device=device) < self.p_src
            both = do & dd  # never both: keep one of the two at random
            flip = torch.rand(B, device=device) < 0.5
            do = do & ~(both & flip)
            dd = dd & ~(both & ~flip)
        if self.ablate == "no_osm":
            do = torch.ones_like(do)
            dd = torch.zeros_like(dd)
        elif self.ablate == "no_ddot":
            dd = torch.ones_like(dd)
            do = torch.zeros_like(do)
        elif self.ablate != "none":
            raise ValueError(f"unsupported ablate {self.ablate!r}; expected one of {ABLATIONS}")
        return do, dd

    def forward(self, xnum, nan, xcat):
        B = xnum.shape[0] if xnum.shape[1] else xcat.shape[0]
        tok = self.tok(xnum, nan, xcat)  # B x F x d
        miss = self.tok.missing_tokens().unsqueeze(0)  # 1 x F x d
        do, dd = self._drop_flags(B, tok.device)
        # replace a whole stream's own tokens by its missing tokens (net tokens are context and stay)
        keep_o = (~do).view(B, 1, 1)
        keep_d = (~dd).view(B, 1, 1)
        t_o = torch.where(keep_o, tok.index_select(1, self.idx_o), miss.index_select(1, self.idx_o))
        t_d = torch.where(keep_d, tok.index_select(1, self.idx_d), miss.index_select(1, self.idx_d))
        t_n = tok.index_select(1, self.idx_n)
        no, nd = t_o.shape[1], t_d.shape[1]
        if self.mode == "xattn":
            xo, xd = torch.cat([t_o, t_n], 1), torch.cat([t_d, t_n], 1)
            for ca_o, ca_d, ffn_o, ffn_d in zip(self.ca_o, self.ca_d, self.ffn_o, self.ffn_d):
                xo, xd = xo + ca_o(xo, xd), xd + ca_d(xd, xo)  # simultaneous exchange
                xo, xd = ffn_o(xo), ffn_d(xd)
        else:
            h = torch.cat([t_o, t_d, t_n], 1)
            for blk in self.enc:
                h = blk(h)
            h_o, h_d, h_n = h[:, :no], h[:, no:no + nd], h[:, no + nd:]
            xo, xd = torch.cat([h_o, h_n], 1), torch.cat([h_d, h_n], 1)
        z = torch.cat([self.pool_o(self.norm_o(xo)), self.pool_d(self.norm_d(xd))], dim=1)
        return self.head(z)


# --------------------------------------------------------------------------- decoding helper
def expected_level(P):
    return np.asarray(P) @ np.arange(np.asarray(P).shape[1])


def fit_thresholds_macro_f1(mu, y, step=0.01, n_classes=3):
    """Vectorised 2-threshold grid search on expected level maximising macro-F1. Returns (thr, f1)."""
    mu, y = np.asarray(mu, dtype=np.float64), np.asarray(y)
    order = np.argsort(mu, kind="stable")
    ms, ys = mu[order], y[order]
    n = len(ms)
    C = np.zeros((n + 1, n_classes))
    C[1:] = np.cumsum(np.eye(n_classes)[ys], axis=0)
    true = C[n]
    grid = np.arange(max(0.0, ms[0]) + step / 2, ms[-1] + step, step)
    pos = np.searchsorted(ms, grid, side="left")  # items with mu < grid[g]
    a, b = np.triu_indices(len(grid), k=1)
    pa, pb = pos[a], pos[b]
    tp = np.stack([C[pa, 0], C[pb, 1] - C[pa, 1], true[2] - C[pb, 2]], 1)
    pred = np.stack([pa, pb - pa, n - pb], 1).astype(np.float64)
    den = pred + true[None]
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    present = true > 0
    score = f1[:, present].mean(1)
    k = int(np.argmax(score))
    return np.array([grid[a[k]], grid[b[k]]]), float(score[k])


def decode(mu, thr):
    return np.digitize(mu, thr)


def macro_f1(y, yhat, n_classes=3):
    f = []
    for c in range(n_classes):
        t = (y == c)
        if not t.any():
            continue
        tp = ((yhat == c) & t).sum()
        f.append(2 * tp / max((yhat == c).sum() + t.sum(), 1))
    return float(np.mean(f))


# --------------------------------------------------------------------------- training / prediction
@torch.no_grad()
def predict_proba(model, data, ablate="none", bs=4096):
    model.eval()
    prev, model.ablate = model.ablate, ablate
    n = data["xnum"].shape[0]
    out = [torch.softmax(model(data["xnum"][i:i + bs], data["nan"][i:i + bs], data["xcat"][i:i + bs]), 1)
           for i in range(0, n, bs)]
    model.ablate = prev
    return torch.cat(out).float().cpu().numpy()


def make_model(enc: TabularEncoder, mode, cfg):
    return XAttnModel(enc.n_num, enc.cat_sizes, enc.stream_ids, d=cfg.get("d_model", 32),
                      heads=cfg.get("heads", 4), exchanges=cfg.get("exchanges", 2),
                      dropout=cfg.get("dropout", 0.1), source_dropout=cfg.get("source_dropout", 0.15), mode=mode)


def fit_model(enc, train, y_train, val, y_val, mode, cfg, seed, device, epochs=40):
    """Train one seed; keep the epoch with best validation macro-F1 after threshold decoding."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(enc, mode, cfg).to(device)
    crit = OrdinalDistanceCrossEntropy(3, cfg.get("ordinal_lambda", 0.15), cfg.get("label_smoothing", 0.1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 0.05))
    bs = cfg.get("batch_size", 256)
    n = len(y_train)
    steps_per_epoch = max(n // bs, 1)
    total, warm = steps_per_epoch * epochs, max(int(0.05 * steps_per_epoch * epochs), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    sw = torch.tensor((1.0 / np.sqrt(np.maximum(counts, 1)))[y_train], dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    best_f1, best_state, best_ep = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.multinomial(sw, steps_per_epoch * bs, replacement=True)
        for i in range(steps_per_epoch):
            b = idx[i * bs:(i + 1) * bs]
            logits = model(train["xnum"][b], train["nan"][b], train["xcat"][b])
            loss = crit(logits, yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if ep >= 3:
            _, f1 = fit_thresholds_macro_f1(expected_level(predict_proba(model, val)), y_val)
            if f1 > best_f1:
                best_f1, best_ep = f1, ep
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_ep, "val_f1": best_f1}
