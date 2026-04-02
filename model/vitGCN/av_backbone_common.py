import math
import types

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from unimodal_backbones import StrongAudioEncoder, StrongLandmarkVideoEncoder


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, p, **kwargs):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)
        return x


class Lambda(nn.Module):
    def __init__(self, lmd):
        super().__init__()
        if not isinstance(lmd, types.LambdaType):
            raise Exception("'lmd' should be lambda ftn.")
        self.lmd = lmd

    def forward(self, x):
        return self.lmd(x)


class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.0, f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.net = nn.Sequential(
            f(dim_in, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(hidden_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class Attention1d(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        dim_out = dim_in if dim_out is None else dim_out
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dim_mlp=1024, dropout=0.0, sd=0.0,
                 attn=Attention1d, norm=nn.LayerNorm, f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.norm1 = norm(dim_in)
        self.attn = attn(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd1 = DropPath(sd) if sd > 0.0 else nn.Identity()

        self.norm2 = norm(dim_out)
        self.ff = FeedForward(dim_out, dim_mlp, dim_out, dropout=dropout, f=f, activation=activation)
        self.sd2 = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, _attn = self.attn(x, mask=mask)
        x = self.sd1(x) + skip

        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.sd2(x) + skip
        return x


class PatchEmbdding(nn.Module):
    def __init__(self, spectra_size, patch_size, dim_out, channel=1):
        super().__init__()
        if not spectra_size % patch_size == 0:
            raise Exception('Spectra dimensions must be divisible by the patch size!')
        patch_dim = channel * patch_size
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (d p) -> b d (p c)', p=patch_size),
            nn.Linear(patch_dim, dim_out),
        )

    def forward(self, x):
        return self.patch_embedding(x)


class CLSToken(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x):
        b, _n, _d = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        return torch.cat((cls_tokens, x), dim=1)


class AbsPosEmbedding(nn.Module):
    def __init__(self, spectra_size, patch_size, dim, stride=None, cls=True):
        super().__init__()
        if not spectra_size % patch_size == 0:
            raise Exception('Spectra dimensions must be divisible by the patch size!')
        stride = patch_size if stride is None else stride
        output_size = self._conv_output_size(spectra_size, patch_size, stride)
        num_patches = output_size * 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + int(cls), dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding

    @staticmethod
    def _conv_output_size(spectra_size, kernel_size, stride, padding=0):
        return int(((spectra_size - kernel_size + (2 * padding)) / stride) + 1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, alpha_init=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x_vid, x_aud):
        query = self.norm_q(x_vid).permute(1, 0, 2)
        key = self.norm_kv(x_aud).permute(1, 0, 2)
        value = self.norm_kv(x_aud).permute(1, 0, 2)
        attn_output, _ = self.attn(query, key, value)
        attn_output = attn_output.permute(1, 0, 2)
        output = x_vid + self.alpha * attn_output
        return self.norm_out(output)


class ShallowAudioTCNEncoder(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.05):
        super().__init__()
        k = int(kernel_size)
        if k <= 0 or (k % 2) == 0:
            raise ValueError("audio_tcn_kernel_size must be a positive odd integer")
        d = float(dropout)
        if d < 0.0 or d >= 1.0:
            raise ValueError("audio_tcn_dropout must be in [0, 1)")
        p = k // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, stride=1, padding=p)
        self.act = nn.GELU()
        self.drop = nn.Dropout(d) if d > 0.0 else nn.Identity()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, stride=1, padding=p)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        return residual + x


class ResidualTemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.05):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = dilation * (kernel_size // 2)
        self.dw_conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=channels
        )
        self.pw_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return x + out


class TemporalTransformerStack(nn.Module):
    def __init__(self, dim, depth, heads, dim_mlp, dropout=0.1, sd=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Transformer(
                dim, heads=heads, dim_head=max(1, dim // heads),
                dim_mlp=dim_mlp, dropout=dropout,
                sd=(sd * i / max(depth - 1, 1))
            )
            for i in range(depth)
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


class AttnPool1D(nn.Module):
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


class FeatureSequenceEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, patch_size, local_blocks=2, local_kernel=3,
                 dilations=(1, 2), global_depth=4, heads=4, dim_mlp=1024,
                 dropout=0.1, sd=0.0, stem_kernel=1):
        super().__init__()
        if stem_kernel <= 0 or stem_kernel % 2 == 0:
            raise ValueError("stem_kernel must be a positive odd integer")
        stem_padding = stem_kernel // 2
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, kernel_size=stem_kernel, padding=stem_padding),
            nn.GroupNorm(1, model_dim),
            nn.GELU(),
        )
        blocks = []
        use_dilations = list(dilations) if len(dilations) > 0 else [1]
        for idx in range(int(local_blocks)):
            blocks.append(
                ResidualTemporalConvBlock(
                    model_dim,
                    kernel_size=local_kernel,
                    dilation=use_dilations[idx % len(use_dilations)],
                    dropout=dropout * 0.5,
                )
            )
        self.local_encoder = nn.Sequential(*blocks)
        self.patch_embed = nn.Conv1d(model_dim, model_dim, kernel_size=patch_size, stride=patch_size)
        self.global_encoder = TemporalTransformerStack(
            dim=model_dim,
            depth=max(1, int(global_depth)),
            heads=heads,
            dim_mlp=dim_mlp,
            dropout=dropout,
            sd=sd,
        )
        self.pool = AttnPool1D(model_dim, dropout=dropout * 0.5)

    def forward(self, x, token_mask=None):
        x = self.input_proj(x)
        x = self.local_encoder(x)
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = self.global_encoder(tokens, valid_mask=token_mask)
        pooled, weights = self.pool(tokens, valid_mask=token_mask)
        return {
            'tokens': tokens,
            'repr': pooled,
            'attn': weights,
        }


class PostEncoderAVFusion(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, use_cross_attn=True, alpha_init=0.1):
        super().__init__()
        self.use_cross_attn = bool(use_cross_attn)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        if self.use_cross_attn:
            self.cross_v = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)
            self.cross_a = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)
            self.alpha_v = nn.Parameter(torch.tensor(float(alpha_init)))
            self.alpha_a = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.cross_v = None
            self.cross_a = None
            self.alpha_v = None
            self.alpha_a = None
        hidden = max(dim, dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, video_tokens, audio_tokens, token_mask=None):
        key_pad = None if token_mask is None else (~token_mask)
        v = self.norm_v(video_tokens)
        a = self.norm_a(audio_tokens)
        if self.use_cross_attn:
            v_ctx, _ = self.cross_v(v, a, a, key_padding_mask=key_pad)
            a_ctx, _ = self.cross_a(a, v, v, key_padding_mask=key_pad)
            v = video_tokens + torch.tanh(self.alpha_v) * v_ctx
            a = audio_tokens + torch.tanh(self.alpha_a) * a_ctx
        else:
            v = video_tokens
            a = audio_tokens
        gate = self.gate(torch.cat([v, a], dim=-1))
        fused = gate * v + (1.0 - gate) * a
        fused = self.out_norm(fused)
        if token_mask is not None:
            fused = fused * token_mask.unsqueeze(-1).float()
        return fused


class AVBackboneCore(nn.Module):
    def __init__(self, spectra_size, patch_size, num_classes, dim, depth, heads, dim_mlp,
                 channel=1, dim_head=16, dropout=0.0, emb_dropout=0.3, sd=0.0,
                 embedding=None, classifier=None, name='vit',
                 video_dim=171, audio_dim=128, dataset='LMVD', use_av_cross_attn=True,
                 av_cross_heads=None, av_cross_alpha_init=0.1,
                 use_temporal_conv_stem=False,
                 temporal_conv_stem_kernel=3,
                 temporal_conv_stem_for='both',
                 temporal_conv_stem_dvlog_only=False,
                 use_shallow_audio_tcn_encoder=False,
                 audio_tcn_kernel_size=3,
                 audio_tcn_dropout=0.05,
                 audio_tcn_encoder_dvlog_only=True,
                 use_legacy_av_backbone=False,
                 use_feature_sequence_encoder=False,
                 fs_video_local_blocks=2,
                 fs_audio_local_blocks=3,
                 fs_video_global_depth=None,
                 fs_audio_global_depth=None,
                 fs_dilated_audio=True,
                 fs_fusion_dropout=None,
                 modality_mode='fusion',
                 single_modality_clean_path=True,
                 use_strong_audio_encoder=False,
                 use_strong_video_encoder=False,
                 audio_fixed_len=128,
                 video_fixed_len=128,
                 video_use_delta=True,
                 audio_strong_stem_kernel=3,
                 video_strong_stem_kernel=5,
                 **block_kwargs):
        super().__init__()
        self.name = name
        self.dataset = dataset
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.use_av_cross_attn = use_av_cross_attn
        self.patch_size = patch_size
        self.dim = dim
        self.modality_mode = str(modality_mode).lower()
        valid_modes = {'fusion', 'av_only', 'video_only', 'audio_only', 'gcn_only'}
        if self.modality_mode not in valid_modes:
            raise ValueError(f"modality_mode must be one of {sorted(valid_modes)}")
        self.single_modality_clean_path = bool(single_modality_clean_path)
        self.use_strong_audio_encoder = bool(use_strong_audio_encoder)
        self.use_strong_video_encoder = bool(use_strong_video_encoder)
        self.audio_fixed_len = int(audio_fixed_len)
        self.video_fixed_len = int(video_fixed_len)
        self.video_use_delta = bool(video_use_delta)
        self.audio_strong_stem_kernel = int(audio_strong_stem_kernel)
        self.video_strong_stem_kernel = int(video_strong_stem_kernel)
        self.use_legacy_av_backbone = bool(use_legacy_av_backbone)
        self.use_feature_sequence_encoder = bool(use_feature_sequence_encoder)
        self.use_feature_sequence_encoder_effective = (
            self.use_feature_sequence_encoder and (not self.use_legacy_av_backbone)
        )

        stem_for = str(temporal_conv_stem_for).lower()
        if stem_for not in {'video', 'audio', 'both'}:
            raise ValueError("temporal_conv_stem_for must be one of ['video', 'audio', 'both']")
        stem_k = int(temporal_conv_stem_kernel)
        if stem_k <= 0 or (stem_k % 2) == 0:
            raise ValueError("temporal_conv_stem_kernel must be a positive odd integer")
        stem_enabled = bool(use_temporal_conv_stem)
        if bool(temporal_conv_stem_dvlog_only) and str(dataset).upper() != 'DVLOG':
            stem_enabled = False
        video_k = stem_k if (stem_enabled and stem_for in {'video', 'both'}) else 1
        audio_proj_k = stem_k if (stem_enabled and stem_for in {'audio', 'both'}) else 1

        audio_tcn_enabled = bool(use_shallow_audio_tcn_encoder)
        if bool(audio_tcn_encoder_dvlog_only) and str(dataset).upper() != 'DVLOG':
            audio_tcn_enabled = False
        self.audio_tcn_enabled = audio_tcn_enabled
        self.audio_tcn_kernel_size = int(audio_tcn_kernel_size)
        self.audio_tcn_dropout = float(audio_tcn_dropout)
        self.audio_tcn_encoder = (
            ShallowAudioTCNEncoder(
                channels=self.audio_dim,
                kernel_size=audio_tcn_kernel_size,
                dropout=audio_tcn_dropout,
            )
            if self.audio_tcn_enabled else nn.Identity()
        )

        d_projection = dim // 2
        self.legacy_proj_video = nn.Sequential(
            nn.Conv1d(self.video_dim, d_projection, kernel_size=video_k, padding=video_k // 2),
            nn.GroupNorm(1, d_projection),
            nn.ReLU(),
        )
        self.legacy_proj_audio = nn.Sequential(
            nn.Conv1d(self.audio_dim, d_projection, kernel_size=audio_proj_k, padding=audio_proj_k // 2),
            nn.GroupNorm(1, d_projection),
            nn.ReLU(),
        )
        self.legacy_ln_video = nn.LayerNorm(d_projection)
        self.legacy_ln_audio = nn.LayerNorm(d_projection)
        cross_heads = heads if av_cross_heads is None else av_cross_heads
        self.legacy_fusion = (
            CrossAttentionFusion(dim=d_projection, heads=cross_heads, dropout=dropout, alpha_init=av_cross_alpha_init)
            if self.use_av_cross_attn else None
        )
        self.legacy_embedding = nn.Sequential(
            PatchEmbdding(spectra_size=spectra_size, patch_size=patch_size, dim_out=dim, channel=channel),
            CLSToken(dim=dim),
            AbsPosEmbedding(spectra_size=spectra_size, patch_size=patch_size, dim=dim, cls=True),
            nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity(),
        )
        self.legacy_transformers = nn.ModuleList([
            Transformer(dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp, dropout=dropout,
                        sd=(sd * i / max(depth - 1, 1)))
            for i in range(depth)
        ])

        fusion_drop = float(dropout if fs_fusion_dropout is None else fs_fusion_dropout)
        video_global_depth = int(depth if fs_video_global_depth is None else fs_video_global_depth)
        audio_global_depth = int(max(2, depth // 2) if fs_audio_global_depth is None else fs_audio_global_depth)
        audio_dilations = (1, 2, 4) if fs_dilated_audio else (1, 2)

        if self.use_strong_video_encoder:
            self.video_encoder = StrongLandmarkVideoEncoder(
                input_dim=self.video_dim,
                model_dim=dim,
                target_len=self.video_fixed_len,
                use_delta=self.video_use_delta,
                local_blocks=max(3, int(fs_video_local_blocks)),
                heads=heads,
                global_depth=video_global_depth,
                dim_mlp=dim_mlp,
                dropout=dropout,
                stem_kernel=self.video_strong_stem_kernel,
            )
        else:
            self.video_encoder = FeatureSequenceEncoder(
                input_dim=self.video_dim,
                model_dim=dim,
                patch_size=patch_size,
                local_blocks=fs_video_local_blocks,
                local_kernel=3,
                dilations=(1, 2),
                global_depth=video_global_depth,
                heads=heads,
                dim_mlp=dim_mlp,
                dropout=dropout,
                sd=sd,
                stem_kernel=video_k,
            )

        if self.use_strong_audio_encoder:
            self.audio_encoder = StrongAudioEncoder(
                input_dim=self.audio_dim,
                model_dim=dim,
                target_len=self.audio_fixed_len,
                local_blocks=max(2, int(fs_audio_local_blocks)),
                heads=heads,
                global_depth=audio_global_depth,
                dim_mlp=dim_mlp,
                dropout=dropout,
                stem_kernel=self.audio_strong_stem_kernel,
            )
        else:
            self.audio_encoder = FeatureSequenceEncoder(
                input_dim=self.audio_dim,
                model_dim=dim,
                patch_size=patch_size,
                local_blocks=fs_audio_local_blocks,
                local_kernel=3,
                dilations=audio_dilations,
                global_depth=audio_global_depth,
                heads=heads,
                dim_mlp=dim_mlp,
                dropout=dropout,
                sd=sd,
                stem_kernel=audio_proj_k,
            )

        self.video_stem = self.video_encoder.input_proj
        self.video_local_encoder = self.video_encoder.local_encoder
        self.video_global_encoder = self.video_encoder.global_encoder
        self.video_pool = self.video_encoder.pool
        self.audio_stem = self.audio_encoder.input_proj
        self.audio_local_encoder = self.audio_encoder.local_encoder
        self.audio_global_encoder = self.audio_encoder.global_encoder
        self.audio_pool = self.audio_encoder.pool

        self.av_post_encoder_fusion = PostEncoderAVFusion(
            dim=dim,
            heads=cross_heads,
            dropout=fusion_drop,
            use_cross_attn=self.use_av_cross_attn,
            alpha_init=av_cross_alpha_init,
        )
        self.av_pool = AttnPool1D(dim, dropout=fusion_drop * 0.5)

        if not self.use_feature_sequence_encoder_effective:
            self.proj_video = self.legacy_proj_video
            self.proj_audio = self.legacy_proj_audio
            self.ln_video = self.legacy_ln_video
            self.ln_audio = self.legacy_ln_audio
            self.fusion = self.legacy_fusion
        else:
            self.proj_video = self.video_encoder.input_proj
            self.proj_audio = self.audio_encoder.input_proj
            self.ln_video = nn.LayerNorm(dim)
            self.ln_audio = nn.LayerNorm(dim)
            self.fusion = self.av_post_encoder_fusion

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) if classifier is None else classifier

    def _build_token_mask(self, actual_lens, token_count, device, batch_size):
        if actual_lens is None:
            return torch.ones(batch_size, token_count, dtype=torch.bool, device=device)
        if not isinstance(actual_lens, torch.Tensor):
            actual_lens = torch.as_tensor(actual_lens, device=device)
        else:
            actual_lens = actual_lens.to(device=device)
        actual_lens = actual_lens.long()
        token_lens = ((actual_lens + self.patch_size - 1) // self.patch_size).clamp(min=1, max=token_count)
        return torch.arange(token_count, device=device).unsqueeze(0) < token_lens.unsqueeze(1)

    @staticmethod
    def _temporal_global_pool(x):
        return x.mean(dim=-1)

    def _legacy_embed_with_dynamic_pos(self, x_fused):
        x = self.legacy_embedding[0](x_fused)
        x = self.legacy_embedding[1](x)
        pos_full = self.legacy_embedding[2].pos_embedding
        n_cur = x.shape[1]
        n_full = pos_full.shape[1]
        if n_cur == n_full:
            pos = pos_full
        elif n_cur < n_full:
            pos = pos_full[:, :n_cur, :]
        else:
            cls_pos = pos_full[:, :1, :]
            patch_pos = pos_full[:, 1:, :].transpose(1, 2)
            patch_pos = F.interpolate(patch_pos, size=n_cur - 1, mode='linear', align_corners=False).transpose(1, 2)
            pos = torch.cat([cls_pos, patch_pos], dim=1)
        x = x + pos
        x = self.legacy_embedding[3](x)
        return x

    @staticmethod
    def _resize_tokens(tokens, target_len):
        if tokens.shape[1] == target_len:
            return tokens
        return F.adaptive_avg_pool1d(tokens.transpose(1, 2), target_len).transpose(1, 2)

    @staticmethod
    def _resize_mask(mask, target_len, device, batch_size):
        if mask is None:
            return torch.ones(batch_size, target_len, dtype=torch.bool, device=device)
        if mask.shape[1] == target_len:
            return mask
        pooled = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), target_len).squeeze(1)
        return pooled > 0.0

    def _run_video_encoder(self, video_x, token_mask=None, actual_lens=None):
        if self.use_strong_video_encoder:
            return self.video_encoder(video_x, actual_lens=actual_lens)
        out = self.video_encoder(video_x, token_mask=token_mask)
        out['token_mask'] = token_mask
        return out

    def _run_audio_encoder(self, audio_x, token_mask=None, actual_lens=None):
        if self.use_strong_audio_encoder:
            return self.audio_encoder(audio_x, actual_lens=actual_lens)
        out = self.audio_encoder(audio_x, token_mask=token_mask)
        out['token_mask'] = token_mask
        return out

    def _legacy_encode_unimodal_clean(self, video_x, audio_x, actual_lens, mode):
        token_count = max(1, int(video_x.shape[-1] // self.patch_size))
        token_mask = self._build_token_mask(actual_lens, token_count, video_x.device, video_x.size(0))
        zero_repr = torch.zeros(video_x.size(0), self.dim // 2, device=video_x.device, dtype=video_x.dtype)

        if mode == 'audio_only':
            audio_proj = self.legacy_proj_audio(audio_x)
            audio_tokens = F.adaptive_avg_pool1d(audio_proj, token_count).transpose(1, 2)
            audio_repr = self._temporal_global_pool(audio_proj)
            zero_tokens = torch.zeros_like(audio_tokens)
            return {
                'video_tokens': zero_tokens,
                'audio_tokens': audio_tokens,
                'av_tokens': audio_tokens,
                'video_repr': zero_repr,
                'audio_repr': audio_repr,
                'av_repr': audio_repr,
                'token_mask': token_mask,
                'av_attn': None,
            }

        video_proj = self.legacy_proj_video(video_x)
        video_tokens = F.adaptive_avg_pool1d(video_proj, token_count).transpose(1, 2)
        video_repr = self._temporal_global_pool(video_proj)
        zero_tokens = torch.zeros_like(video_tokens)
        return {
            'video_tokens': video_tokens,
            'audio_tokens': zero_tokens,
            'av_tokens': video_tokens,
            'video_repr': video_repr,
            'audio_repr': zero_repr,
            'av_repr': video_repr,
            'token_mask': token_mask,
            'av_attn': None,
        }

    def _encode_feature_sequences_legacy(self, video_x, audio_x, actual_lens=None):
        video_x = video_x.permute(0, 2, 1)
        audio_x = audio_x.permute(0, 2, 1)
        audio_x = self.audio_tcn_encoder(audio_x)

        video_proj = self.legacy_proj_video(video_x)
        audio_proj = self.legacy_proj_audio(audio_x)
        audio_repr = self._temporal_global_pool(audio_proj)

        if self.use_av_cross_attn:
            v_in = self.legacy_ln_video(video_proj.permute(0, 2, 1))
            a_in = self.legacy_ln_audio(audio_proj.permute(0, 2, 1))
            fused_v = self.legacy_fusion(v_in, a_in).permute(0, 2, 1)
            x_fused = torch.cat([fused_v, audio_proj], dim=1)
        else:
            x_fused = torch.cat([video_proj, audio_proj], dim=1)

        x = self._legacy_embed_with_dynamic_pos(x_fused)
        token_count = x.size(1) - 1
        token_mask = self._build_token_mask(actual_lens, token_count, x.device, x.size(0))
        cls_valid = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)
        full_valid = torch.cat([cls_valid, token_mask], dim=1)
        attn_mask = (~full_valid).float() * (-1e9)
        attn_mask = attn_mask[:, None, None, :]
        for blk in self.legacy_transformers:
            x = blk(x, mask=attn_mask)

        av_repr = x[:, 0]
        av_tokens = x[:, 1:]
        video_tokens = F.adaptive_avg_pool1d(video_proj, token_count).transpose(1, 2)
        audio_tokens = F.adaptive_avg_pool1d(audio_proj, token_count).transpose(1, 2)
        return {
            'video_tokens': video_tokens,
            'audio_tokens': audio_tokens,
            'av_tokens': av_tokens,
            'video_repr': av_repr,
            'audio_repr': audio_repr,
            'av_repr': av_repr,
            'token_mask': token_mask,
            'av_attn': None,
        }

    def encode_feature_sequences(self, video_x, audio_x, actual_lens=None):
        if not self.use_feature_sequence_encoder_effective:
            if self.single_modality_clean_path and self.modality_mode in {'audio_only', 'video_only'}:
                video_x = video_x.permute(0, 2, 1)
                audio_x = audio_x.permute(0, 2, 1)
                audio_x = self.audio_tcn_encoder(audio_x)
                return self._legacy_encode_unimodal_clean(video_x, audio_x, actual_lens, self.modality_mode)
            return self._encode_feature_sequences_legacy(video_x, audio_x, actual_lens=actual_lens)

        video_x = video_x.permute(0, 2, 1)
        audio_x = audio_x.permute(0, 2, 1)
        audio_x = self.audio_tcn_encoder(audio_x)

        token_count = max(1, video_x.shape[-1] // self.patch_size)
        base_mask = self._build_token_mask(actual_lens, token_count, video_x.device, video_x.size(0))

        if self.single_modality_clean_path and self.modality_mode == 'audio_only':
            audio_out = self._run_audio_encoder(audio_x, token_mask=base_mask, actual_lens=actual_lens)
            audio_tokens = audio_out['tokens']
            token_mask = audio_out.get('token_mask', None)
            if token_mask is None:
                token_mask = torch.ones(audio_tokens.shape[:2], dtype=torch.bool, device=audio_tokens.device)
            zero_tokens = torch.zeros_like(audio_tokens)
            zero_repr = torch.zeros(audio_tokens.size(0), self.dim, device=audio_tokens.device, dtype=audio_tokens.dtype)
            return {
                'video_tokens': zero_tokens,
                'audio_tokens': audio_tokens,
                'av_tokens': audio_tokens,
                'video_repr': zero_repr,
                'audio_repr': audio_out['repr'],
                'av_repr': audio_out['repr'],
                'token_mask': token_mask,
                'av_attn': audio_out.get('attn', None),
            }

        if self.single_modality_clean_path and self.modality_mode == 'video_only':
            video_out = self._run_video_encoder(video_x, token_mask=base_mask, actual_lens=actual_lens)
            video_tokens = video_out['tokens']
            token_mask = video_out.get('token_mask', None)
            if token_mask is None:
                token_mask = torch.ones(video_tokens.shape[:2], dtype=torch.bool, device=video_tokens.device)
            zero_tokens = torch.zeros_like(video_tokens)
            zero_repr = torch.zeros(video_tokens.size(0), self.dim, device=video_tokens.device, dtype=video_tokens.dtype)
            return {
                'video_tokens': video_tokens,
                'audio_tokens': zero_tokens,
                'av_tokens': video_tokens,
                'video_repr': video_out['repr'],
                'audio_repr': zero_repr,
                'av_repr': video_out['repr'],
                'token_mask': token_mask,
                'av_attn': video_out.get('attn', None),
            }

        video_out = self._run_video_encoder(video_x, token_mask=base_mask, actual_lens=actual_lens)
        audio_out = self._run_audio_encoder(audio_x, token_mask=base_mask, actual_lens=actual_lens)

        t_video = int(video_out['tokens'].shape[1])
        t_audio = int(audio_out['tokens'].shape[1])
        target_len = min(t_video, t_audio)
        video_tokens = self._resize_tokens(video_out['tokens'], target_len)
        audio_tokens = self._resize_tokens(audio_out['tokens'], target_len)
        v_mask = self._resize_mask(video_out.get('token_mask', None), target_len, video_tokens.device, video_tokens.size(0))
        a_mask = self._resize_mask(audio_out.get('token_mask', None), target_len, video_tokens.device, video_tokens.size(0))
        token_mask = v_mask & a_mask

        av_tokens = self.av_post_encoder_fusion(video_tokens, audio_tokens, token_mask=token_mask)
        av_repr, av_attn = self.av_pool(av_tokens, valid_mask=token_mask)

        return {
            'video_tokens': video_tokens,
            'audio_tokens': audio_tokens,
            'av_tokens': av_tokens,
            'video_repr': video_out['repr'],
            'audio_repr': audio_out['repr'],
            'av_repr': av_repr,
            'token_mask': token_mask,
            'av_attn': av_attn,
        }

    def forward(self, x1, x2, actual_lens=None, mask=None, return_repr=False, return_dict=False):
        if actual_lens is not None and mask is None:
            if isinstance(actual_lens, torch.Tensor) and actual_lens.ndim >= 3:
                mask = actual_lens
                actual_lens = None

        enc = self.encode_feature_sequences(x1, x2, actual_lens=actual_lens)
        av_repr = enc['av_repr']
        logits = self.classifier(av_repr)

        if return_dict:
            return {
                'logits': logits,
                'video_tokens': enc['video_tokens'],
                'audio_tokens': enc['audio_tokens'],
                'av_tokens': enc['av_tokens'],
                'video_repr': enc['video_repr'],
                'audio_repr': enc['audio_repr'],
                'av_repr': enc['av_repr'],
                'token_mask': enc['token_mask'],
            }
        if return_repr:
            return logits, av_repr, enc['audio_repr']
        return logits
