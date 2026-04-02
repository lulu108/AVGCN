import torch
from torch import nn


def masked_mean(features, time_mask, eps=1e-6):
    valid = time_mask.float().unsqueeze(-1)
    denom = valid.sum(1).clamp(min=eps)
    return (features * valid).sum(1) / denom


def safe_time_pool_temporal(features, target_t, actual_lens):
    """Mask-aware temporal pooling from (B, T_long, D) to (B, target_t, D)."""
    bsz, t_long, dim = features.shape
    device = features.device

    frame_idx = torch.arange(t_long, device=device)
    frame_mask = (frame_idx.unsqueeze(0) < actual_lens.unsqueeze(1)).float()

    if t_long == target_t:
        return features * frame_mask.unsqueeze(-1)

    bin_idx = (frame_idx.float() * target_t / t_long).long().clamp(max=target_t - 1)
    bin_idx_feat = bin_idx.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, dim)

    weighted = features * frame_mask.unsqueeze(-1)
    output = torch.zeros(bsz, target_t, dim, device=device, dtype=features.dtype)
    output.scatter_add_(1, bin_idx_feat, weighted)

    bin_idx_mask = bin_idx.unsqueeze(0).expand(bsz, -1)
    bin_counts = torch.zeros(bsz, target_t, device=device, dtype=features.dtype)
    bin_counts.scatter_add_(1, bin_idx_mask, frame_mask)
    bin_counts = bin_counts.clamp(min=1e-6).unsqueeze(-1)

    return output / bin_counts


def mask_safe_attention_pool(features, time_mask, attn_proj):
    """Attention pooling with all-invalid fallback."""
    safe_mask = time_mask.clone()
    all_invalid = ~safe_mask.any(dim=1)
    if all_invalid.any():
        safe_mask[all_invalid] = True

    scores = attn_proj(features).squeeze(-1)
    scores = scores.masked_fill(~safe_mask, -1e9)
    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
    return (features * weights).sum(dim=1)


class LateFusionHead(nn.Module):
    def __init__(self, dim, gcn_feat_dim, dropout=0.1):
        super().__init__()
        self.proj_gcn = nn.Sequential(
            nn.Linear(gcn_feat_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, vit_repr, gcn_repr):
        gcn_proj = self.proj_gcn(gcn_repr)
        gate = self.gate(torch.cat([vit_repr, gcn_proj], dim=-1))
        fused = gate * vit_repr + (1.0 - gate) * gcn_proj
        return self.norm(fused)


class IntermediateCrossFusion(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, vit_tokens, gcn_tokens, time_mask):
        key_pad = ~time_mask
        fused, _ = self.cross_attn(vit_tokens, gcn_tokens, gcn_tokens, key_padding_mask=key_pad)
        fused = self.norm(vit_tokens + fused)
        return mask_safe_attention_pool(fused, time_mask, self.attn_proj)


class IntermediateBiGateFusion(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.cross_v2g = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm_v = nn.LayerNorm(dim)
        self.cross_g2v = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm_g = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.fused_norm = nn.LayerNorm(dim)
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, vit_tokens, gcn_tokens, time_mask):
        key_pad = ~time_mask
        vit_ctx, _ = self.cross_v2g(vit_tokens, gcn_tokens, gcn_tokens, key_padding_mask=key_pad)
        vit_ctx = self.norm_v(vit_tokens + vit_ctx)

        gcn_ctx, _ = self.cross_g2v(gcn_tokens, vit_tokens, vit_tokens, key_padding_mask=key_pad)
        gcn_ctx = self.norm_g(gcn_tokens + gcn_ctx)

        alpha = self.gate(torch.cat([vit_ctx, gcn_ctx], dim=-1))
        fused = alpha * vit_ctx + (1.0 - alpha) * gcn_ctx
        fused = self.fused_norm(fused)
        return mask_safe_attention_pool(fused, time_mask, self.attn_proj)


class ConcatFusionHead(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2)
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, vit_repr, gcn_repr):
        z = torch.cat([vit_repr, gcn_repr], dim=-1)
        return self.mlp(self.norm(z))


class IAResidualFusionHead(nn.Module):
    def __init__(self, dim, hidden_ratio=0.5, dropout=0.05, alpha_init=0.0, mode='g2v'):
        super().__init__()
        hidden_dim = max(1, int(dim * float(hidden_ratio)))
        self.mode = str(mode).lower()
        valid_modes = {'g2v', 'v2g', 'bi'}
        if self.mode not in valid_modes:
            raise ValueError(f"ia recalibration mode must be one of {sorted(valid_modes)}, got {mode!r}")
        self.use_g2v = self.mode in {'g2v', 'bi'}
        self.use_v2g = self.mode in {'v2g', 'bi'}

        self.norm_v = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(dim)

        if self.use_g2v:
            self.v_gate = nn.Sequential(
                nn.Linear(dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, dim),
            )
            self.alpha_v = nn.Parameter(torch.tensor(float(alpha_init)))
            self.out_norm_v = nn.LayerNorm(dim)
        else:
            self.v_gate = None
            self.alpha_v = None
            self.out_norm_v = None

        if self.use_v2g:
            self.g_gate = nn.Sequential(
                nn.Linear(dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, dim),
            )
            self.alpha_g = nn.Parameter(torch.tensor(float(alpha_init)))
            self.out_norm_g = nn.LayerNorm(dim)
        else:
            self.g_gate = None
            self.alpha_g = None
            self.out_norm_g = None

        self.fused_norm = nn.LayerNorm(dim) if self.mode == 'bi' else None
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, vit_tokens, gcn_tokens, time_mask):
        v = self.norm_v(vit_tokens)
        g = self.norm_g(gcn_tokens)

        v_out = None
        g_out = None
        if self.use_g2v:
            gate_v = torch.sigmoid(self.v_gate(torch.cat([v, g], dim=-1)))
            v_out = self.out_norm_v(vit_tokens + torch.tanh(self.alpha_v) * (v * gate_v))
        if self.use_v2g:
            gate_g = torch.sigmoid(self.g_gate(torch.cat([g, v], dim=-1)))
            g_out = self.out_norm_g(gcn_tokens + torch.tanh(self.alpha_g) * (g * gate_g))

        if self.mode == 'bi':
            fused = self.fused_norm(0.5 * (v_out + g_out))
        elif self.mode == 'g2v':
            fused = v_out
        else:
            fused = g_out

        return mask_safe_attention_pool(fused, time_mask, self.attn_proj)


class MutualAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, ff_mult=2):
        super().__init__()
        self.norm_v = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(dim)
        self.cross_v2g = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.cross_g2v = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.alpha_v = nn.Parameter(torch.zeros(1))
        self.alpha_g = nn.Parameter(torch.zeros(1))

        hidden = max(dim, int(dim * ff_mult))
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_g2 = nn.LayerNorm(dim)
        self.ffn_v = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.ffn_g = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, v, g, time_mask):
        key_pad = ~time_mask
        v_n = self.norm_v(v)
        g_n = self.norm_g(g)
        v_ctx, _ = self.cross_v2g(v_n, g_n, g_n, key_padding_mask=key_pad)
        g_ctx, _ = self.cross_g2v(g_n, v_n, v_n, key_padding_mask=key_pad)
        v = v + torch.tanh(self.alpha_v) * v_ctx
        g = g + torch.tanh(self.alpha_g) * g_ctx
        v = v + self.ffn_v(self.norm_v2(v))
        g = g + self.ffn_g(self.norm_g2(g))
        return v, g


class AFIFusionHead(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            MutualAttentionBlock(dim=dim, heads=heads, dropout=dropout, ff_mult=2)
            for _ in range(n_blocks)
        ])
        self.gate_scalar = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())
        self.fused_norm = nn.LayerNorm(dim)
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, vit_tokens, gcn_tokens, time_mask):
        v, g = vit_tokens, gcn_tokens
        for blk in self.blocks:
            v, g = blk(v, g, time_mask)

        v_mean = masked_mean(v, time_mask)
        g_mean = masked_mean(g, time_mask)
        alpha = self.gate_scalar(torch.cat([v_mean, g_mean], dim=-1))

        fused = alpha.unsqueeze(1) * v + (1.0 - alpha).unsqueeze(1) * g
        fused = self.fused_norm(fused)
        return mask_safe_attention_pool(fused, time_mask, self.attn_proj)
