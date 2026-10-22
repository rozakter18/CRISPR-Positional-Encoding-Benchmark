"""
03_models.py
────────────────────────────────────────────────────────────────────────────────
Four identical transformer encoders for token-level classification, differing
only in their positional encoding:

  'learned'     — Standard learned absolute embeddings (nn.Embedding)
  'sinusoidal'  — Fixed sin/cos embeddings (Vaswani et al. 2017)
  'rope'        — Rotary Position Embedding (Su et al. 2021 / LLaMA-style)
  'alibi'       — Attention with Linear Biases (Press et al. 2021)

Architecture (all four):
  d_model=64, n_heads=4, n_layers=2, FFN dim=256, dropout=0.1
  Token classifier head: Linear(d_model -> 2)
  Vocabulary: A T G C N PAD (6 tokens, PAD ignored in loss)

EMBEDDING SCALING (v2 fix)
──────────────────────────
Token embeddings are multiplied by sqrt(d_model) before the positional
encoding is added, following Vaswani et al. (2017) §3.4.

Without this, nn.Embedding initialised at std=0.02 produces a token signal of
RMS ~0.02 while sinusoidal PE has RMS ~0.707 — the positional signal is ~34x
larger, so nucleotide identity accounts for ~0.1% of input variance.  Learned
absolute PE escapes this because gradients can shrink its table; fixed
sinusoidal PE cannot, and collapses to chance (loss = ln 2 = 0.693).

Set scale_emb=False to reproduce the unscaled v1 behaviour as an ablation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import VOCAB_SIZE, PAD_IDX

PE_TYPES = ('learned', 'sinusoidal', 'rope', 'alibi')


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding modules
# ──────────────────────────────────────────────────────────────────────────────

class LearnedPE(nn.Module):
    """
    Learned absolute positional embedding.

    `scale` must match the scale applied to the token embedding, so the
    positional signal enters the residual stream at comparable magnitude.
    Scaling only the token embedding (v2) leaves the learned table ~7x
    smaller than the tokens and the model loses positional information.
    """
    def __init__(self, d_model: int, max_len: int, scale: float = 1.0):
        super().__init__()
        self.pe    = nn.Embedding(max_len, d_model)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions).unsqueeze(0) * self.scale


class SinusoidalPE(nn.Module):
    """Fixed sin/cos positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ── RoPE helpers ──────────────────────────────────────────────────────────────

def build_rope_cache(max_len: int, head_dim: int, device: torch.device):
    """Precompute cos/sin tables for RoPE (LLaMA concat convention)."""
    assert head_dim % 2 == 0
    theta = 1.0 / (10000 ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim
    ))
    t     = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, theta)
    emb   = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ── ALiBi helper ──────────────────────────────────────────────────────────────

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """ALiBi head slopes (Press et al., 2021)."""
    def _slopes_pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _slopes_pow2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes  = _slopes_pow2(closest)
        extra   = _slopes_pow2(2 * closest)[0::2][: n_heads - closest]
        slopes  = slopes + extra
    return torch.tensor(slopes, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-head attention (supports all four PE types)
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, pe_type: str, max_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.pe_type  = pe_type
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if pe_type == 'rope':
            cos, sin = build_rope_cache(max_len, self.head_dim, torch.device('cpu'))
            self.register_buffer('rope_cos', cos)
            self.register_buffer('rope_sin', sin)
        elif pe_type == 'alibi':
            self.register_buffer('alibi_slopes', get_alibi_slopes(n_heads))

    def forward(self, x, pad_mask=None):
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pe_type == 'rope':
            if T > self.rope_cos.size(0):
                cos, sin = build_rope_cache(T, self.head_dim, x.device)
                self.rope_cos, self.rope_sin = cos, sin
            Q, K = apply_rope(Q, K, self.rope_cos[:T], self.rope_sin[:T])

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if self.pe_type == 'alibi':
            pos  = torch.arange(T, device=x.device).float()
            dist = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1))
            bias = -self.alibi_slopes.view(self.n_heads, 1, 1) * dist
            attn = attn + bias.unsqueeze(0)

        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask[:, None, None, :], float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, pe_type, ffn_dim, max_len, dropout):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, n_heads, pe_type, max_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), pad_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Top-level model
# ──────────────────────────────────────────────────────────────────────────────

class CRISPRTransformer(nn.Module):
    """
    Token-level binary classifier for CRISPR repeat vs. spacer.

    pe_type    : 'learned' | 'sinusoidal' | 'rope' | 'alibi'
    scale_emb  : multiply token embeddings by sqrt(d_model) before adding PE
                 (Vaswani et al. 2017 §3.4).  True by default; set False to
                 reproduce the unscaled v1 behaviour.
    """

    def __init__(
        self,
        pe_type:   str,
        d_model:   int   = 64,
        n_heads:   int   = 4,
        n_layers:  int   = 2,
        ffn_dim:   int   = 256,
        max_len:   int   = 1024,
        dropout:   float = 0.1,
        scale_emb: bool  = True,
        emb_init_std: float | None = None,
    ):
        super().__init__()
        assert pe_type in PE_TYPES, f"pe_type must be one of {PE_TYPES}"
        self.pe_type   = pe_type
        self.scale_emb = scale_emb
        self.emb_scale = math.sqrt(d_model) if scale_emb else 1.0
        # Vaswani convention: embeddings ~ N(0, d_model^-0.5), then x sqrt(d_model),
        # which gives unit-RMS token signal comparable to the PE.  Falling back to
        # std=0.02 (the v1 value) leaves the PE ~4x larger even when scaled.
        self.emb_init_std = (
            emb_init_std if emb_init_std is not None
            else (d_model ** -0.5 if scale_emb else 0.02)
        )

        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        self.emb_drop  = nn.Dropout(dropout)

        if pe_type == 'learned':
            self.pos_enc = LearnedPE(d_model, max_len, scale=self.emb_scale)
        elif pe_type == 'sinusoidal':
            self.pos_enc = SinusoidalPE(d_model, max_len)
        else:
            self.pos_enc = None   # rope / alibi act inside attention

        attn_pe = pe_type if pe_type in ('rope', 'alibi') else 'none'
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, attn_pe, ffn_dim, max_len, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.emb_init_std)
        # padding_idx row must stay zero after re-init
        with torch.no_grad():
            self.embedding.weight[PAD_IDX].fill_(0)

    def forward(self, input_ids, pad_mask=None):
        """input_ids (B,T) -> logits (B,T,2)"""
        x = self.embedding(input_ids) * self.emb_scale
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.emb_drop(x)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.head(self.norm(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Sanity check + signal-balance diagnostic ──────────────────────────────────

if __name__ == "__main__":
    B, T = 4, 512
    ids  = torch.randint(0, 4, (B, T))
    mask = torch.zeros(B, T, dtype=torch.bool)

    print("Forward pass check:\n")
    for pe in PE_TYPES:
        model  = CRISPRTransformer(pe_type=pe)
        logits = model(ids, mask)
        assert logits.shape == (B, T, 2)
        print(f"  {pe:12s}  out {tuple(logits.shape)}  params {model.n_params:,}")

    # Token-vs-position signal balance.  The controlled variable across PE
    # types is that both signals enter the residual stream at comparable
    # magnitude; a ratio far from 1 handicaps that encoding.
    print("\nToken-vs-position signal balance (token RMS : positional RMS):\n")
    print(f"  {'config':<30}{'token':>9}{'position':>10}{'tok:pos':>10}")
    print("  " + "-" * 59)
    for tag, kw in [("v1  unscaled, std 0.02", dict(scale_emb=False)),
                    ("v3  scaled, matched",    dict(scale_emb=True))]:
        for pe in ("learned", "sinusoidal"):
            m   = CRISPRTransformer(pe_type=pe, **kw)
            tok = (m.embedding(ids) * m.emb_scale).pow(2).mean().sqrt().item()
            pos = (m.pos_enc.pe.weight * m.pos_enc.scale if pe == "learned"
                   else m.pos_enc.pe).pow(2).mean().sqrt().item()
            flag = "ok" if 0.5 <= tok / pos <= 2.0 else "IMBALANCED"
            print(f"  {tag + ' / ' + pe:<30}{tok:>9.3f}{pos:>10.3f}"
                  f"{tok/pos:>9.2f}  {flag}")
    print("\n  RoPE and ALiBi inject position inside attention and add nothing")
    print("  to the residual stream, so this balance does not apply to them.")