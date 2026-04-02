import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class _FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, dim_in),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class _Attention1d(nn.Module):
    def __init__(self, dim_in, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_in),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class _TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_mlp=1024, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention1d(dim, heads=heads, dim_head=max(1, dim // heads), dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, dim_mlp, dropout=dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ff(self.norm2(x))
        return x


class _TemporalTransformerStack(nn.Module):
    def __init__(self, dim, depth, heads, dim_mlp, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TransformerBlock(dim=dim, heads=heads, dim_mlp=dim_mlp, dropout=dropout)
            for _ in range(max(1, int(depth)))
        ])

    def forward(self, x, valid_mask=None):
        attn_mask = None
        if valid_mask is not None:
            safe_mask = valid_mask.clone()
            all_invalid = ~safe_mask.any(dim=1)
            if all_invalid.any():
                safe_mask[all_invalid] = True
            attn_mask = (~safe_mask).float() * (-1e9)
            attn_mask = attn_mask[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, mask=attn_mask)
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).float()
        return x


class _ResidualTemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        if kernel_size <= 0 or (kernel_size % 2) == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = dilation * (kernel_size // 2)
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pw_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return x + out


class _AttnPool1D(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, tokens, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        safe_mask = valid_mask.clone()
        all_invalid = ~safe_mask.any(dim=1)
        if all_invalid.any():
            safe_mask[all_invalid] = True
        scores = self.score(self.norm(tokens)).squeeze(-1)
        scores = scores.masked_fill(~safe_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(self.drop(tokens) * weights.unsqueeze(-1), dim=1)
        return pooled, weights


def _build_frame_mask(actual_lens, seq_len, device, batch_size):
    if actual_lens is None:
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    if not isinstance(actual_lens, torch.Tensor):
        actual_lens = torch.as_tensor(actual_lens, device=device)
    else:
        actual_lens = actual_lens.to(device=device)
    actual_lens = actual_lens.long().clamp(min=1, max=seq_len)
    frame_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    return frame_ids < actual_lens.unsqueeze(1)


def _pool_mask(mask, target_len):
    if mask.shape[1] == target_len:
        return mask
    pooled = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), target_len).squeeze(1)
    return pooled > 0.0


class StrongAudioEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        target_len=128,
        local_blocks=2,
        heads=4,
        global_depth=4,
        dim_mlp=1024,
        dropout=0.1,
        stem_kernel=3,
    ):
        super().__init__()
        if stem_kernel <= 0 or (stem_kernel % 2) == 0:
            raise ValueError("stem_kernel must be a positive odd integer")
        self.target_len = int(target_len)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, kernel_size=stem_kernel, padding=stem_kernel // 2),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
        )
        self.local_encoder = nn.Sequential(*[
            _ResidualTemporalConvBlock(model_dim, kernel_size=3, dilation=1, dropout=dropout * 0.5)
            for _ in range(max(1, int(local_blocks)))
        ])
        self.length_pool = nn.AdaptiveAvgPool1d(self.target_len)
        self.global_encoder = _TemporalTransformerStack(
            dim=model_dim,
            depth=global_depth,
            heads=heads,
            dim_mlp=dim_mlp,
            dropout=dropout,
        )
        self.pool = _AttnPool1D(model_dim, dropout=dropout * 0.5)

    def forward(self, x, actual_lens=None):
        bsz, _ch, src_len = x.shape
        frame_mask = _build_frame_mask(actual_lens, src_len, x.device, bsz)
        token_mask = _pool_mask(frame_mask, self.target_len)

        x = self.input_proj(x)
        x = self.local_encoder(x)
        x = self.length_pool(x)
        tokens = x.transpose(1, 2)

        tokens = self.global_encoder(tokens, valid_mask=token_mask)
        pooled, attn = self.pool(tokens, valid_mask=token_mask)
        return {
            'tokens': tokens,
            'repr': pooled,
            'attn': attn,
            'token_mask': token_mask,
        }


class StrongLandmarkVideoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        target_len=128,
        use_delta=True,
        local_blocks=3,
        heads=4,
        global_depth=4,
        dim_mlp=1024,
        dropout=0.1,
        stem_kernel=5,
    ):
        super().__init__()
        if stem_kernel <= 0 or (stem_kernel % 2) == 0:
            raise ValueError("stem_kernel must be a positive odd integer")
        self.target_len = int(target_len)
        self.use_delta = bool(use_delta)
        in_dim = input_dim * (2 if self.use_delta else 1)

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_dim, model_dim, kernel_size=stem_kernel, padding=stem_kernel // 2),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
        )
        dilations = [1, 2, 4]
        self.local_encoder = nn.Sequential(*[
            _ResidualTemporalConvBlock(
                model_dim,
                kernel_size=3,
                dilation=dilations[i % len(dilations)],
                dropout=dropout * 0.5,
            )
            for i in range(max(1, int(local_blocks)))
        ])
        self.length_pool = nn.AdaptiveAvgPool1d(self.target_len)
        self.global_encoder = _TemporalTransformerStack(
            dim=model_dim,
            depth=global_depth,
            heads=heads,
            dim_mlp=dim_mlp,
            dropout=dropout,
        )
        self.pool = _AttnPool1D(model_dim, dropout=dropout * 0.5)

    def forward(self, x, actual_lens=None):
        bsz, _ch, src_len = x.shape
        frame_mask = _build_frame_mask(actual_lens, src_len, x.device, bsz)
        token_mask = _pool_mask(frame_mask, self.target_len)

        if self.use_delta:
            delta = torch.diff(x, dim=-1, prepend=x[:, :, :1])
            x = torch.cat([x, delta], dim=1)

        x = self.input_proj(x)
        x = self.local_encoder(x)
        x = self.length_pool(x)
        tokens = x.transpose(1, 2)

        tokens = self.global_encoder(tokens, valid_mask=token_mask)
        pooled, attn = self.pool(tokens, valid_mask=token_mask)
        return {
            'tokens': tokens,
            'repr': pooled,
            'attn': attn,
            'token_mask': token_mask,
        }
