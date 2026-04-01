import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
import types
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
""" 定义了核心的Vision Transformer架构，
用于接收音频和视频特征序列，并将它们进行拼接融合，
然后进行时序建模和分类。 """
# layers:

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
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

    def extra_repr(self):
        return 'p=%s' % repr(self.p)


class Lambda(nn.Module):
    def __init__(self, lmd):
        super(Lambda, self).__init__()
        if not isinstance(lmd, types.LambdaType):
            raise Exception("'lmd' should be lambda ftn.")
        self.lmd = lmd

    def forward(self, x):
        return self.lmd(x)

# attentions:

class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.0, f=nn.Linear, activation=nn.GELU):
        super(FeedForward, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(
            f(dim_in, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(hidden_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention1d(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0):
        super(Attention1d, self).__init__()
        inner_dim = heads * dim_head
        dim_out = dim_in if dim_out is None else dim_out

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # (2, 16, 11, 32)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (2, 16, 11, 11)
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)  # (2, 16, 11, 11)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (2, 16, 11, 32)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (2, 11, 512)
        out = self.to_out(out)  # (2, 11, 512)

        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dim_mlp=1024, dropout=0.0, sd=0.0,
                 attn=Attention1d, norm=nn.LayerNorm, f=nn.Linear, activation=nn.GELU):
        super(Transformer, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.norm1 = norm(dim_in)
        self.attn = attn(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout, )
        self.sd1 = DropPath(sd) if sd > 0.0 else nn.Identity()

        self.norm2 = norm(dim_out)
        self.ff = FeedForward(dim_out, dim_mlp, dim_out, dropout=dropout, f=f, activation=activation)
        self.sd2 = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, attn = self.attn(x, mask=mask)
        x = self.sd1(x) + skip

        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.sd2(x) + skip

        return x


# embeddings:

class PatchEmbdding(nn.Module):
    def __init__(self, spectra_size, patch_size, dim_out, channel=1):
        super(PatchEmbdding, self).__init__()
        if not spectra_size % patch_size == 0:
            raise Exception('Spectra dimensions must be divisible by the patch size!')
        patch_dim = channel * patch_size
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (d p) -> b d (p c)', p=patch_size),
            nn.Linear(patch_dim, dim_out),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        return x


class CLSToken(nn.Module):
    def __init__(self, dim):
        super(CLSToken, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        return x


class AbsPosEmbedding(nn.Module):
    def __init__(self, spectra_size, patch_size, dim, stride=None, cls=True):
        super(AbsPosEmbedding, self).__init__()
        if not spectra_size % patch_size == 0:
            raise Exception('Spectra dimensions must be divisible by the patch size!')
        stride = patch_size if stride is None else stride
        output_size = self._conv_output_size(spectra_size, patch_size, stride)
        num_patches = output_size * 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + int(cls), dim) * 0.02)

    def forward(self, x):
        x = x + self.pos_embedding
        return x

    @staticmethod
    def _conv_output_size(spectra_size, kernel_size, stride, padding=0):
        return int(((spectra_size - kernel_size + (2 * padding)) / stride) + 1)

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, alpha_init=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        
        # 1. 删除 batch_first=True，因为旧版本不支持
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x_vid, x_aud):
        # 输入 x_vid, x_aud 形状为: (Batch, Sequence_Length, Dim)
        
        # LayerNorm
        query = self.norm_q(x_vid) 
        key = self.norm_kv(x_aud) 
        value = self.norm_kv(x_aud)
        
        # 2. 手动转置维度：(Batch, Seq, Dim) -> (Seq, Batch, Dim)
        # 旧版 MultiheadAttention 要求 Seq 在第一维
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        
        # Attention
        attn_output, _ = self.attn(query, key, value)
        
        # 3. 转置回来：(Seq, Batch, Dim) -> (Batch, Seq, Dim)
        attn_output = attn_output.permute(1, 0, 2)
        
        # 残差连接（带可学习缩放）
        output = x_vid + self.alpha * attn_output 
        
        return self.norm_out(output)


class ShallowAudioTCNEncoder(nn.Module):
    """轻量音频时序编码器：Conv1d -> GELU -> Dropout -> Conv1d + residual。"""
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
    """Local temporal modeling block for feature-sequence encoding."""
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
    """Pre-norm temporal transformer stack over token sequences."""
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
    """Mask-aware attention pooling over token sequences."""
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
    """
    Single-modality feature-sequence encoder:
      stem -> local temporal conv encoder -> patch/downsample -> global transformer -> attention pool
    """
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
    """
    AV interaction after modality-specific encoders are finished.
    Supports lightweight post-encoder cross-attn or gated token fusion.
    """
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


class ViT(nn.Module):
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
                 use_temporal_attn_pool=False,
                 temporal_attn_pool_dvlog_only=True,
                 temporal_attn_pool_alpha_init=0.2,
                 use_legacy_av_backbone=False,
                 use_feature_sequence_encoder=False,
                 fs_video_local_blocks=2,
                 fs_audio_local_blocks=3,
                 fs_video_global_depth=None,
                 fs_audio_global_depth=None,
                 fs_dilated_audio=True,
                 fs_fusion_dropout=None,
                 **block_kwargs):
        super(ViT, self).__init__()
        dataset_upper = str(dataset).upper()
        if dataset_upper == 'DVLOG':
            from av_backbone_dvlog import DVLOGAVBackbone
            backbone_cls = DVLOGAVBackbone
        else:
            from av_backbone_lmvd import LMVDAVBackbone
            backbone_cls = LMVDAVBackbone

        self.backbone = backbone_cls(
            spectra_size=spectra_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_mlp=dim_mlp,
            channel=channel,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            sd=sd,
            embedding=embedding,
            classifier=classifier,
            name=name,
            video_dim=video_dim,
            audio_dim=audio_dim,
            dataset=dataset_upper,
            use_av_cross_attn=use_av_cross_attn,
            av_cross_heads=av_cross_heads,
            av_cross_alpha_init=av_cross_alpha_init,
            use_temporal_conv_stem=use_temporal_conv_stem,
            temporal_conv_stem_kernel=temporal_conv_stem_kernel,
            temporal_conv_stem_for=temporal_conv_stem_for,
            temporal_conv_stem_dvlog_only=temporal_conv_stem_dvlog_only,
            use_shallow_audio_tcn_encoder=use_shallow_audio_tcn_encoder,
            audio_tcn_kernel_size=audio_tcn_kernel_size,
            audio_tcn_dropout=audio_tcn_dropout,
            audio_tcn_encoder_dvlog_only=audio_tcn_encoder_dvlog_only,
            use_legacy_av_backbone=use_legacy_av_backbone,
            use_feature_sequence_encoder=use_feature_sequence_encoder,
            fs_video_local_blocks=fs_video_local_blocks,
            fs_audio_local_blocks=fs_audio_local_blocks,
            fs_video_global_depth=fs_video_global_depth,
            fs_audio_global_depth=fs_audio_global_depth,
            fs_dilated_audio=fs_dilated_audio,
            fs_fusion_dropout=fs_fusion_dropout,
            **block_kwargs,
        )

        self.name = self.backbone.name
        self.dataset = self.backbone.dataset
        self.video_dim = self.backbone.video_dim
        self.audio_dim = self.backbone.audio_dim
        self.use_av_cross_attn = self.backbone.use_av_cross_attn
        self.patch_size = self.backbone.patch_size
        self.dim = self.backbone.dim
        self.use_legacy_av_backbone = self.backbone.use_legacy_av_backbone
        self.use_feature_sequence_encoder = self.backbone.use_feature_sequence_encoder
        self.use_feature_sequence_encoder_effective = self.backbone.use_feature_sequence_encoder_effective
        self.audio_tcn_encoder = self.backbone.audio_tcn_encoder
        self.proj_video = self.backbone.proj_video
        self.proj_audio = self.backbone.proj_audio

    @property
    def classifier(self):
        return self.backbone.classifier

    @classifier.setter
    def classifier(self, module):
        self.backbone.classifier = module

    def encode_feature_sequences(self, video_x, audio_x, actual_lens=None):
        return self.backbone.encode_feature_sequences(video_x, audio_x, actual_lens=actual_lens)

    def forward(self, X1, X2, actual_lens=None, mask=None, return_repr=False, return_dict=False):
        return self.backbone(
            X1,
            X2,
            actual_lens=actual_lens,
            mask=mask,
            return_repr=return_repr,
            return_dict=return_dict,
        )

# ===========================================================================
# Fusion Modules: 三种可切换的 ViT-GCN 融合策略
# ===========================================================================

class LateFusionHead(nn.Module):
    """
    Late Fusion: 各模态独立池化后门控融合（最简单、最稳）。
    Input:
        vit_repr : (B, D)   — ViT CLS-token 表示
        gcn_repr : (B, Dg)  — GCN 时序池化后的表示
    Output:
        fused_repr : (B, D)
    """
    def __init__(self, dim, gcn_feat_dim, dropout=0.1):
        super().__init__()
        self.proj_gcn = nn.Sequential(
            nn.Linear(gcn_feat_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 门控: 学习每个维度偏向哪个模态
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, vit_repr, gcn_repr):
        gcn_proj = self.proj_gcn(gcn_repr)                             # (B, D)
        gate     = self.gate(torch.cat([vit_repr, gcn_proj], dim=-1))  # (B, D)
        fused    = gate * vit_repr + (1.0 - gate) * gcn_proj           # (B, D)
        return self.norm(fused)


class IntermediateCrossFusion(nn.Module):
    """
    Intermediate Cross Fusion（单向）: Q=ViT, K/V=GCN，再做 Attention Pooling。
    Input:
        vit_features        : (B, T_p, D)
        gcn_features_pooled : (B, T_p, D)
        time_mask           : (B, T_p)  True=有效帧
    Output:
        video_repr : (B, D)
    """
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm       = nn.LayerNorm(dim)
        self.attn_proj  = nn.Linear(dim, 1)
        self._mask_debug_printed = False

    def forward(self, vit_features, gcn_features_pooled, time_mask):
        key_pad = ~time_mask
        if not self._mask_debug_printed:
            B, T_p, _ = vit_features.shape
            assert key_pad.shape == (B, T_p), (
                f"[CrossFusion] key_padding_mask shape mismatch: got {tuple(key_pad.shape)}, expected ({B}, {T_p})"
            )
            print(f"[CrossFusion] key_pad_ratio={key_pad.float().mean().item():.4f}")
            self._mask_debug_printed = True
        fused, _ = self.cross_attn(
            vit_features, gcn_features_pooled, gcn_features_pooled,
            key_padding_mask=key_pad
        )
        fused    = self.norm(vit_features + fused)              # 残差 (B, T_p, D)
        return self._attn_pool(fused, time_mask)

    def _attn_pool(self, features, time_mask):
        scores  = self.attn_proj(features).squeeze(-1)          # (B, T_p)
        scores  = scores.masked_fill(~time_mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)    # (B, T_p, 1)
        return (features * weights).sum(dim=1)                  # (B, D)


class IntermediateBiGateFusion(nn.Module):
    """
    Intermediate Bidirectional Gate Fusion（双向 cross-attn + 门控融合）:
        vit_ctx = Attn(Q=vit, K/V=gcn)
        gcn_ctx = Attn(Q=gcn, K/V=vit)
        alpha   = sigmoid(MLP([vit_ctx, gcn_ctx]))  per (B, T, D)
        fused   = alpha * vit_ctx + (1-alpha) * gcn_ctx
    再做 Attention Pooling -> (B, D)。
    Input:
        vit_features        : (B, T_p, D)
        gcn_features_pooled : (B, T_p, D)
        time_mask           : (B, T_p)
    Output:
        video_repr : (B, D)
    """
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        # vit → gcn 方向
        self.cross_v2g = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm_v = nn.LayerNorm(dim)
        # gcn → vit 方向
        self.cross_g2v = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm_g = nn.LayerNorm(dim)
        # 门控: per (B, T, D)，学习每个时间步每个维度偏向哪个模态
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.fused_norm = nn.LayerNorm(dim)
        # Attention Pooling
        self.attn_proj = nn.Linear(dim, 1)
        self._mask_debug_printed = False

    def forward(self, vit_features, gcn_features_pooled, time_mask):
        # 双向 cross-attn + 残差
        key_pad = ~time_mask
        if not self._mask_debug_printed:
            B, T_p, _ = vit_features.shape
            assert key_pad.shape == (B, T_p), (
                f"[BiGateFusion] key_padding_mask shape mismatch: got {tuple(key_pad.shape)}, expected ({B}, {T_p})"
            )
            print(f"[BiGateFusion] key_pad_ratio={key_pad.float().mean().item():.4f}")
            self._mask_debug_printed = True
        vit_ctx, _ = self.cross_v2g(
            vit_features, gcn_features_pooled, gcn_features_pooled,
            key_padding_mask=key_pad
        )
        vit_ctx    = self.norm_v(vit_features + vit_ctx)          # (B, T_p, D)

        gcn_ctx, _ = self.cross_g2v(
            gcn_features_pooled, vit_features, vit_features,
            key_padding_mask=key_pad
        )
        gcn_ctx    = self.norm_g(gcn_features_pooled + gcn_ctx)   # (B, T_p, D)

        # 门控融合
        alpha = self.gate(torch.cat([vit_ctx, gcn_ctx], dim=-1))  # (B, T_p, D)
        fused = alpha * vit_ctx + (1.0 - alpha) * gcn_ctx
        fused = self.fused_norm(fused)                            # (B, T_p, D)

        return self._attn_pool(fused, time_mask)

    def _attn_pool(self, features, time_mask):
        scores  = self.attn_proj(features).squeeze(-1)            # (B, T_p)
        scores  = scores.masked_fill(~time_mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)      # (B, T_p, 1)
        return (features * weights).sum(dim=1)                    # (B, D)


class ConcatFusionHead(nn.Module):
    """
    Concat Fusion（强约束低参数）:
        z = concat(vit_repr, gcn_repr) -> MLP -> (B, D)
    适合小数据集稳定性对照。
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, vit_repr, gcn_repr):
        z = torch.cat([vit_repr, gcn_repr], dim=-1)  # (B, 2D)
        return self.mlp(self.norm(z))                # (B, D)


class IAResidualFusionHead(nn.Module):
    """
    Lightweight IA-style token recalibration for concat fusion.

    Supported modes:
        'g2v': GCN -> ViT recalibration, then pool recalibrated ViT tokens
        'v2g': ViT -> GCN recalibration, then pool recalibrated GCN tokens
        'bi' : do both directions and average the recalibrated token streams
    """
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
        self._debug_printed = False

    def forward(self, vit_features, gcn_features_pooled, time_mask):
        v = self.norm_v(vit_features)
        g = self.norm_g(gcn_features_pooled)

        scale_v = None
        scale_g = None
        v_out = None
        g_out = None

        if self.use_g2v:
            gate_v = torch.sigmoid(self.v_gate(torch.cat([v, g], dim=-1)))
            scale_v = torch.tanh(self.alpha_v)
            v_out = self.out_norm_v(vit_features + scale_v * (v * gate_v))

        if self.use_v2g:
            gate_g = torch.sigmoid(self.g_gate(torch.cat([g, v], dim=-1)))
            scale_g = torch.tanh(self.alpha_g)
            g_out = self.out_norm_g(gcn_features_pooled + scale_g * (g * gate_g))

        if self.mode == 'bi':
            fused = self.fused_norm(0.5 * (v_out + g_out))
        elif self.mode == 'g2v':
            fused = v_out
        else:
            fused = g_out

        if not self._debug_printed:
            msg = f"[IAResidualFusion] mode={self.mode}"
            if scale_v is not None:
                msg += f" alpha_v={scale_v.item():.4f}"
            if scale_g is not None:
                msg += f" alpha_g={scale_g.item():.4f}"
            print(msg)
            self._debug_printed = True

        return self._attn_pool(fused, time_mask)

    def _attn_pool(self, features, time_mask):
        scores = self.attn_proj(features).squeeze(-1)            # (B, T_p)
        scores = scores.masked_fill(~time_mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)     # (B, T_p, 1)
        return (features * weights).sum(dim=1)                   # (B, D)


class MutualAttentionBlock(nn.Module):
    """
    B2: AFI (Adaptive Feature Interaction) 基本单元。
    双向 pre-norm cross-attn + 标量门控残差 + 轻量 FFN。

    与 IntermediateBiGateFusion 的核心区别：
      - gate = tanh(scalar)，初始化为 0 → tanh(0)=0，初始完全无跨模态影响，渐进开门
      - pre-norm 结构：梯度更稳定，收敛更快
      - 附带轻量 FFN（per-branch），捕获交互后非线性变换
    """
    def __init__(self, dim, heads=8, dropout=0.1, ff_mult=2):
        super().__init__()
        # Pre-norm
        self.norm_v    = nn.LayerNorm(dim)
        self.norm_g    = nn.LayerNorm(dim)
        # Bidirectional cross-attn
        self.cross_v2g = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                               dropout=dropout, batch_first=True)
        self.cross_g2v = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                               dropout=dropout, batch_first=True)
        # 标量门控（初始化为 0 → tanh(0)=0，不扰动初始输出，逐步学习开门程度）
        self.alpha_v   = nn.Parameter(torch.zeros(1))
        self.alpha_g   = nn.Parameter(torch.zeros(1))
        # Pre-norm FFN（轻量）
        hidden         = max(dim, int(dim * ff_mult))
        self.norm_v2   = nn.LayerNorm(dim)
        self.norm_g2   = nn.LayerNorm(dim)
        self.ffn_v     = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.ffn_g     = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self._debug_printed = False

    def forward(self, v, g, time_mask):
        """
        v, g      : (B, T_p, dim)
        time_mask : (B, T_p) True=有效帧
        Returns   : v', g' 同 shape
        """
        key_pad = ~time_mask
        if not self._debug_printed:
            print(f"[MutualAttnBlock] key_pad_ratio={key_pad.float().mean().item():.4f} "
                  f"alpha_v={self.alpha_v.item():.4f} alpha_g={self.alpha_g.item():.4f}")
            self._debug_printed = True

        # Pre-norm
        v_n = self.norm_v(v)
        g_n = self.norm_g(g)

        # Bidirectional cross-attn + scalar-gate residual
        v_ctx, _ = self.cross_v2g(v_n, g_n, g_n, key_padding_mask=key_pad)
        g_ctx, _ = self.cross_g2v(g_n, v_n, v_n, key_padding_mask=key_pad)
        v = v + torch.tanh(self.alpha_v) * v_ctx  # 初始 alpha=0 → 纯恒等映射
        g = g + torch.tanh(self.alpha_g) * g_ctx

        # Pre-norm FFN
        v = v + self.ffn_v(self.norm_v2(v))
        g = g + self.ffn_g(self.norm_g2(g))
        return v, g


class AFIFusionHead(nn.Module):
    """
    B2: AFI Fusion Head（Adaptive Feature Interaction）。
    n_blocks 层 MutualAttentionBlock + 全局摘要标量门控 + Attention Pooling。

    设计原则：
      - 多轮渐进交互：每层 block 均含 pre-norm + bidirectional cross-attn + FFN
      - 标量门初始为 0：比 it_bi_gate per-dim gate 更稳健，不易被单个极端 token 支配
      - 最终样本级 scalar gate（Linear(2D,1)）：参数极少，小数据更稳
      - 与 IntermediateBiGateFusion / IntermediateCrossFusion 接口完全兼容

    Args:
        dim      : token 维度
        heads    : attention 头数（须整除 dim）
        dropout  : attention/FFN dropout
        n_blocks : MutualAttentionBlock 堆叠层数（建议 DVLOG=2, LMVD=2）
    """
    def __init__(self, dim, heads=8, dropout=0.1, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            MutualAttentionBlock(dim=dim, heads=heads, dropout=dropout, ff_mult=2)
            for _ in range(n_blocks)
        ])
        # 最终门控：scalar per sample（比 it_bi_gate 的 per-dim gate 参数少 dim 倍）
        # 用全局 masked-mean 摘要计算一个 [0,1] 标量，决定偏向哪个模态
        self.gate_scalar = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid(),
        )
        self.fused_norm  = nn.LayerNorm(dim)
        # Attention Pooling（与其他 Fusion Head 对齐）
        self.attn_proj   = nn.Linear(dim, 1)

    def forward(self, vit_features, gcn_features_pooled, time_mask):
        """
        vit_features        : (B, T_p, dim)
        gcn_features_pooled : (B, T_p, dim)
        time_mask           : (B, T_p) True=有效帧
        Returns: (B, dim)
        """
        v, g = vit_features, gcn_features_pooled
        for blk in self.blocks:
            v, g = blk(v, g, time_mask)

        # 全局摘要（masked-mean）用于 scalar gate 计算
        valid_f = time_mask.float().unsqueeze(-1)           # (B, T_p, 1)
        denom   = valid_f.sum(1).clamp(min=1e-6)            # (B, 1)
        v_mean  = (v * valid_f).sum(1) / denom              # (B, dim)
        g_mean  = (g * valid_f).sum(1) / denom              # (B, dim)

        # Scalar gate：(B, 1) → broadcast 到 token 维
        alpha = self.gate_scalar(torch.cat([v_mean, g_mean], dim=-1))  # (B, 1)
        fused = alpha.unsqueeze(1) * v + (1.0 - alpha).unsqueeze(1) * g  # (B, T_p, dim)
        fused = self.fused_norm(fused)
        return self._attn_pool(fused, time_mask)

    def _attn_pool(self, features, time_mask):
        scores  = self.attn_proj(features).squeeze(-1)       # (B, T_p)
        scores  = scores.masked_fill(~time_mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1) # (B, T_p, 1)
        return (features * weights).sum(dim=1)               # (B, dim)


class ViT_GCN_Fusion(nn.Module):
    """
    ViT-GCN 融合模型：
    - ViT 分支: 提取视频+音频时序特征
    - GCN 分支: 提取面部关键点空间特征
    - 跨注意力: Query(ViT) + Key/Value(GCN) 动态融合
    - 注意力池化: 聚焦关键帧
    """
    def __init__(self, 
                 # ViT 参数
                 spectra_size=915, patch_size=15, dim=256, depth=8, heads=8, dim_mlp=1024,
                 sd=0.0,           # Step3: stochastic depth rate；DVLOG 传 0.0 关闭
                 # GCN 参数
                 gcn_out_dim=32, gcn_nhead=4,
                 # 全局支路开关
                 use_global_branch=True,
                 global_alpha_init=0.2,  # DVLOG 建议 0.2，LMVD 建议 0.5
                 # 通用参数
                 num_classes=2, dropout=0.45, channel=256,
                 # 数据集参数
                 video_dim=171, audio_dim=128, dataset='LMVD',
                 # 融合策略
                 fusion_mode='it_cross',
                 # AV cross-attn 开关（默认按数据集自动）
                 use_av_cross_attn=None,
                 av_cross_heads=None,
                 av_cross_alpha_init=0.1,
                 # Temporal Conv Stem 开关
                 use_temporal_conv_stem=False,
                 temporal_conv_stem_kernel=3,
                 temporal_conv_stem_for='both',
                 temporal_conv_stem_dvlog_only=False,
                 use_shallow_audio_tcn_encoder=False,
                 audio_tcn_kernel_size=3,
                 audio_tcn_dropout=0.05,
                 audio_tcn_encoder_dvlog_only=True,
                 use_temporal_attn_pool=False,
                 temporal_attn_pool_dvlog_only=True,
                 temporal_attn_pool_alpha_init=0.2,
                 use_legacy_av_backbone=False,
                 use_feature_sequence_encoder=False,
                 fs_video_local_blocks=2,
                 fs_audio_local_blocks=3,
                 fs_video_global_depth=None,
                 fs_audio_global_depth=None,
                 fs_dilated_audio=True,
                 fs_fusion_dropout=None,
                 # A4: 人脸关键点有效度阈值（对照实验用，默认 0.2）
                 face_valid_thresh=0.2,
                 # 分区方案："legacy6" 原始6区 | "symptom7" 症状卨7区
                 region_scheme="legacy6",
                 # GCN 分支内部消融开关
                 region_fusion_mode='cross_attn',
                 gcn_temporal_mode='transformer',
                 region_mlp_dropout=0.1,
                 tcn_kernel_size=3,
                 modality_mode='fusion',
                 use_ia_recalibration=False,
                 ia_recal_bidirectional=False,
                 ia_recal_mode='g2v',
                 ia_hidden_ratio=0.5,
                 ia_dropout=0.05,
                 ia_alpha_init=0.0,
                 concat_blend_init=0.3):  # 'late' | 'it_cross' | 'it_bi_gate' | 'concat' | 'afi'
        """
        Args:
            video_dim:   视频特征维度 (LMVD: 171, D-Vlog: 136)
            audio_dim:   音频特征维度 (LMVD: 128, D-Vlog: 25)
            dataset:     数据集类型 ('LMVD' 或 'DVLOG')
            fusion_mode: 融合策略 ('late' | 'it_cross' | 'it_bi_gate' | 'concat' | 'afi')
        """
        super().__init__()
        
        self.dim         = dim
        self.dataset     = dataset
        self.fusion_mode = fusion_mode
        self.modality_mode = str(modality_mode).lower()
        if self.modality_mode not in {'fusion', 'gcn_only'}:
            raise ValueError("modality_mode for ViT_GCN_Fusion must be 'fusion' or 'gcn_only'")
        self.use_ia_recalibration = bool(use_ia_recalibration)
        ia_mode = str(ia_recal_mode).lower()
        if ia_mode == 'off':
            ia_mode = 'bi' if ia_recal_bidirectional else 'g2v'
        self.ia_recal_mode = ia_mode
        self.ia_recal_bidirectional = (self.ia_recal_mode == 'bi')
        self.concat_head = None
        self.concat_blend_logit = None
        self._concat_debug_printed = False
        self.patch_size  = patch_size  # 保留：用于 forward 中构造 patch-level padding mask
        self.face_valid_thresh = face_valid_thresh  # A4: 保存以便子模块引用
        self.region_scheme     = region_scheme       # 分区方案保存
        if use_av_cross_attn is None:
            use_av_cross_attn = (str(dataset).upper() != 'DVLOG')
        self.use_av_cross_attn = use_av_cross_attn
        
        # ===== 1. ViT 分支（提取视频时序特征）=====
        self.vit_branch = ViT(
            spectra_size=spectra_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_mlp=dim_mlp,
            channel=channel,
            dim_head=dim // heads,
            dropout=dropout,
            sd=sd,                # Step3: 传递 stochastic depth rate
            video_dim=video_dim,  # 传入视频维度
            audio_dim=audio_dim,  # 传入音频维度
            dataset=dataset,      # 传入数据集类型
            use_av_cross_attn=use_av_cross_attn,
            av_cross_heads=av_cross_heads,
            av_cross_alpha_init=av_cross_alpha_init,
            use_temporal_conv_stem=use_temporal_conv_stem,
            temporal_conv_stem_kernel=temporal_conv_stem_kernel,
            temporal_conv_stem_for=temporal_conv_stem_for,
            temporal_conv_stem_dvlog_only=temporal_conv_stem_dvlog_only,
            use_shallow_audio_tcn_encoder=use_shallow_audio_tcn_encoder,
            audio_tcn_kernel_size=audio_tcn_kernel_size,
            audio_tcn_dropout=audio_tcn_dropout,
            audio_tcn_encoder_dvlog_only=audio_tcn_encoder_dvlog_only,
            use_legacy_av_backbone=use_legacy_av_backbone,
            use_feature_sequence_encoder=use_feature_sequence_encoder,
            fs_video_local_blocks=fs_video_local_blocks,
            fs_audio_local_blocks=fs_audio_local_blocks,
            fs_video_global_depth=fs_video_global_depth,
            fs_audio_global_depth=fs_audio_global_depth,
            fs_dilated_audio=fs_dilated_audio,
            fs_fusion_dropout=fs_fusion_dropout,
        )
        # 移除 ViT 的分类器，只保留特征提取
        self.vit_branch.classifier = nn.Identity()

        # ===== 1.5 Temporal Attention Pool (Version A) =====
        self.use_temporal_attn_pool = use_temporal_attn_pool
        self.temporal_attn_pool_dvlog_only = temporal_attn_pool_dvlog_only
        self.temporal_attn_pool_effective = (
            self.use_temporal_attn_pool
            and (not self.temporal_attn_pool_dvlog_only or (str(dataset).upper() == 'DVLOG'))
        )
        if self.temporal_attn_pool_effective:
            self.temporal_attn_fc = nn.Linear(dim, 1)
            self.temporal_attn_alpha = nn.Parameter(torch.tensor(float(temporal_attn_pool_alpha_init)))
        else:
            self.temporal_attn_fc = None
            self.temporal_attn_alpha = None
        
        # ===== 2. GCN 分支（提取面部空间特征）=====
        from _02GCN_Transformer import AnatomicalGCN
        self.gcn_branch = AnatomicalGCN(
            out_dim=gcn_out_dim,
            nhead=gcn_nhead,
            num_classes=num_classes,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            face_valid_thresh=face_valid_thresh,   # A4: 透传阈值
            use_global_branch=use_global_branch,   # 全局支路开关
            global_alpha_init=global_alpha_init,   # 初始地标权重
            region_scheme=region_scheme,           # 分区方案透传
            region_fusion_mode=region_fusion_mode,
            gcn_temporal_mode=gcn_temporal_mode,
            region_mlp_dropout=region_mlp_dropout,
            tcn_kernel_size=tcn_kernel_size,
        )
        # GCN 输出维度 = fused_dim (区域64 + 全局64 = 128)
        # 注意：不覆盖 gcn_branch.classifier；通过 return_dict=True 获取中间特征
        gcn_feat_dim = gcn_out_dim * 4  # 128
        
        # GCN 特征投影到 ViT 维度
        self.gcn_proj = nn.Sequential(
            nn.Linear(gcn_feat_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gcn_only_proj = nn.Sequential(
            nn.LayerNorm(gcn_feat_dim),
            nn.Linear(gcn_feat_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ===== 3. 融合模块（根据 fusion_mode 切换）=====
        if fusion_mode == 'late':
            self.fusion_head = LateFusionHead(
                dim=dim, gcn_feat_dim=gcn_feat_dim, dropout=dropout
            )
            # 轻量重构头：用 fused_repr 重构 vit_repr / gcn_repr
            self.recon_v = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            self.recon_g = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, gcn_feat_dim),
                nn.ReLU(),
                nn.Linear(gcn_feat_dim, gcn_feat_dim)
            )
        elif fusion_mode == 'it_bi_gate':
            self.fusion_head = IntermediateBiGateFusion(
                dim=dim, heads=heads, dropout=dropout
            )
            # Intermediate 重构头（目标：masked-mean 的 vit / gcn 摘要，均为 dim 维）
            # gcn_pooled 已由 gcn_proj 投影到 dim，因此 recon_g 目标维度也是 dim
            self.recon_v = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
            self.recon_g = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
        elif fusion_mode == 'concat':
            # concat 升级：token 级双向 cross-attn + 摘要级 concat
            if self.use_ia_recalibration:
                self.fusion_head = IAResidualFusionHead(
                    dim=dim,
                    hidden_ratio=ia_hidden_ratio,
                    dropout=ia_dropout,
                    alpha_init=ia_alpha_init,
                    mode=self.ia_recal_mode,
                )
            else:
                self.fusion_head = IntermediateBiGateFusion(
                    dim=dim, heads=heads, dropout=dropout
                )
            self.concat_head = ConcatFusionHead(dim=dim, dropout=dropout)
            blend_init = float(min(max(concat_blend_init, 1e-4), 1.0 - 1e-4))
            blend_init_t = torch.tensor(blend_init, dtype=torch.float32)
            self.concat_blend_logit = nn.Parameter(torch.log(blend_init_t / (1.0 - blend_init_t)))
            # concat 使用 masked-mean 表征（vit/gcn 均为 dim）
            self.recon_v = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
            self.recon_g = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
        elif fusion_mode == 'afi':
            # B2: AFI Fusion Head - n_blocks 层 MutualAttentionBlock（pre-norm + scalar gate + FFN）
            # 比 it_bi_gate 更稳健：标量门初始化 0，防止训练开始手扰动融合路径
            self.fusion_head = AFIFusionHead(
                dim=dim, heads=heads, dropout=dropout, n_blocks=2
            )
            # recon heads（目标维度与 it_bi_gate 相同：dim）
            self.recon_v = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
            self.recon_g = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
        else:  # 'it_cross'（默认）
            self.fusion_head = IntermediateCrossFusion(
                dim=dim, heads=heads, dropout=dropout
            )
            # Intermediate 重构头（同 it_bi_gate，目标均为 dim 维）
            self.recon_v = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
            self.recon_g = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
        
        # ===== 4. 最终分类器 =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
        
    def _temporal_attention_pool(self, temporal_tokens, patch_valid):
        """Temporal attention pooling over non-CLS tokens."""
        scores = self.temporal_attn_fc(temporal_tokens).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~patch_valid, -1e9)
        attn = torch.softmax(scores, dim=1)                          # (B, N)
        pooled = torch.sum(temporal_tokens * attn.unsqueeze(-1), dim=1)  # (B, D)
        return pooled, attn

    def forward(self, video_features, audio_features, face_regions, actual_lens, quality=None):
        """
        Args:
            video_features: (B, T, video_dim)
            audio_features: (B, T, audio_dim)
            face_regions:   dict of {region: (B, T, N, C)}
            actual_lens:    (B,)
        Returns:
            dict with keys:
                'logits'     : (B, num_classes)  融合头分类 logits
                'logits_gcn' : (B, num_classes)  GCN 辅助头 logits（Phase 2-1 使用）
        """
        B      = video_features.size(0)
        device = video_features.device

        # ===== Quality Gate =====
        # quality 支持：
        #   1) Tensor (B,2) -> [:,0]=q_v, [:,1]=q_g
        #   2) dict {'q_v':(B,), 'q_g':(B,)}
        #   3) None -> 默认全 1（不降权）
        if quality is None:
            q_v = torch.ones(B, device=device, dtype=video_features.dtype)
            q_g = torch.ones(B, device=device, dtype=video_features.dtype)
        elif isinstance(quality, dict):
            q_v = quality.get('q_v', None)
            q_g = quality.get('q_g', None)
            if q_v is None:
                q_v = torch.ones(B, device=device, dtype=video_features.dtype)
            if q_g is None:
                q_g = torch.ones(B, device=device, dtype=video_features.dtype)
            q_v = q_v.to(device=device, dtype=video_features.dtype)
            q_g = q_g.to(device=device, dtype=video_features.dtype)
        else:
            q_v = quality[:, 0].to(device=device, dtype=video_features.dtype)
            q_g = quality[:, 1].to(device=device, dtype=video_features.dtype)

        if self.modality_mode == 'gcn_only':
            gcn_out = self.gcn_branch(face_regions, actual_lens, return_dict=True)
            gcn_repr = gcn_out['gcn_repr'] * q_g.unsqueeze(-1)
            video_repr = self.gcn_only_proj(gcn_repr)
            logits_fusion = self.classifier(video_repr)
            vit_repr = torch.zeros(
                gcn_repr.size(0), self.dim,
                device=device, dtype=video_features.dtype
            )
            audio_repr = torch.zeros(
                gcn_repr.size(0), self.dim,
                device=device, dtype=video_features.dtype
            )
            token_steps = max(1, int(math.ceil(float(video_features.size(1)) / float(self.patch_size))))
            token_mask = torch.ones(gcn_repr.size(0), token_steps, dtype=torch.bool, device=device)
            zero_tokens = torch.zeros(
                gcn_repr.size(0), token_steps, self.dim,
                device=device, dtype=video_features.dtype
            )
            return {
                'logits': logits_fusion,
                'logits_gcn': gcn_out['logits_gcn'],
                'recon_v': None,
                'recon_g': None,
                'video_tokens': zero_tokens,
                'audio_tokens': zero_tokens,
                'av_tokens': zero_tokens,
                'vit_repr': vit_repr,
                'gcn_repr': gcn_repr,
                'audio_repr': audio_repr,
                'av_repr': video_repr,
                'token_mask': token_mask,
            }

        # ===== Step 1: AV feature-sequence encoder =====
        av_out = self.vit_branch(
            video_features, audio_features,
            actual_lens=actual_lens,
            return_dict=True
        )
        vit_features = av_out['av_tokens']
        vit_repr = av_out['av_repr']
        audio_repr = av_out['audio_repr']
        patch_valid = av_out['token_mask']
        T_p = vit_features.size(1)
        valid_patch_ratio = patch_valid.float().mean().item() if T_p > 0 else 0.0  # noqa

        # ===== Step 2 & 3: 根据 fusion_mode 分发 =====
        if self.fusion_mode == 'late':
            # ── Late Fusion ──────────────────────────────────────────────
            # ViT CLS-token → (B, dim)
            # AV backbone summary repr (post-encoder pooled)
            if self.temporal_attn_pool_effective:
                temporal_tokens = vit_features  # (B, T_p, dim)
                attn_pool, _attn = self._temporal_attention_pool(temporal_tokens, patch_valid)
                alpha = torch.tanh(self.temporal_attn_alpha)
                alpha = torch.clamp(alpha, 0.0, 1.0)
                vit_repr = (1.0 - alpha) * vit_repr + alpha * attn_pool
            # GCN 完整前向（含时序 Transformer + Attention Pooling）
            gcn_out  = self.gcn_branch(face_regions, actual_lens, return_dict=True)
            gcn_repr = gcn_out['gcn_repr']   # (B, gcn_feat_dim)
            # 质量门控：坏模态降权
            vit_repr = vit_repr * q_v.unsqueeze(-1)
            gcn_repr = gcn_repr * q_g.unsqueeze(-1)
            # 门控融合 → (B, dim)
            video_repr = self.fusion_head(vit_repr, gcn_repr)
            # 轻量重构（fused -> vit/gcn）
            recon_v = self.recon_v(video_repr)
            recon_g = self.recon_g(video_repr)

        else:
            # ── Intermediate Fusion ('it_cross' | 'it_bi_gate' | 'concat' | 'afi') ─────────
            # ViT patch 序列特征（去掉 CLS token）→ (B, T_p, dim)
            # T_p, actual_lens_patches, patch_valid 均已在上方 mask 构造处算好

            # GCN 时序特征（temporal_transformer 输出）
            gcn_out      = self.gcn_branch(face_regions, actual_lens, return_dict=True)
            gcn_temporal = gcn_out['temporal_out']                         # (B, T, fused_dim)
            gcn_features = self.gcn_proj(gcn_temporal)                     # (B, T, dim)
            gcn_pooled   = self._mask_aware_pool_temporal(gcn_features, T_p, actual_lens)  # (B, T_p, dim)

            # 质量门控（token-level）：在 cross-attn / gate 前屏蔽低质量模态
            vit_features = vit_features * q_v[:, None, None]
            gcn_pooled   = gcn_pooled * q_g[:, None, None]

            # cross-attn mask 对齐：patch_valid ∩ pooled_face_mask
            # gcn_out['time_mask'] 为帧级有效掩码（已融合 face_qual_mask）
            frame_mask = gcn_out.get('time_mask', None)
            if frame_mask is None:
                pooled_face_mask = patch_valid
            else:
                pooled_mask = self._mask_aware_pool_temporal(
                    frame_mask.float().unsqueeze(-1),
                    T_p,
                    actual_lens
                ).squeeze(-1)
                pooled_face_mask = (pooled_mask > 0.0)

            time_mask = patch_valid & pooled_face_mask                     # (B, T_p) True=有效
            # 保底：避免某些样本整段都无效
            _all_invalid = ~time_mask.any(dim=1)
            if _all_invalid.any():
                time_mask[_all_invalid] = patch_valid[_all_invalid]

            # 同步屏蔽无效 query token，避免无效 token 参与 cross-attn
            _valid_tok = time_mask.float().unsqueeze(-1)
            vit_features = vit_features * _valid_tok
            gcn_pooled = gcn_pooled * _valid_tok

            # ── Intermediate 重构目标：各模态特征的 masked-mean 摘要 ────────────────
            # vit_features 和 gcn_pooled 均为 (B, T_p, dim)；masked mean → (B, dim)
            valid_float = time_mask.float().unsqueeze(-1)           # (B, T_p, 1)
            denom       = valid_float.sum(1).clamp(min=1e-6)        # (B, 1)
            vit_repr    = (vit_features * valid_float).sum(1) / denom   # (B, dim)
            gcn_repr    = (gcn_pooled   * valid_float).sum(1) / denom   # (B, dim)

            # 融合头
            if self.fusion_mode == 'concat':
                # concat 升级：token 级双向交互 + 摘要级低参融合
                # IA-enabled concat uses token and summary branches with a learnable blend.
                video_repr_token = self.fusion_head(vit_features, gcn_pooled, time_mask)
                video_repr_sum = self.concat_head(vit_repr, gcn_repr)
                blend = torch.sigmoid(self.concat_blend_logit)
                video_repr = (1.0 - blend) * video_repr_sum + blend * video_repr_token
                if not self._concat_debug_printed:
                    print(
                        f"[ConcatFusion] use_ia={self.use_ia_recalibration} "
                        f"ia_mode={self.ia_recal_mode} blend={blend.item():.4f}"
                    )
                    self._concat_debug_printed = True
            else:
                # cross / bi-gate: token 级交互 + attention pooling
                video_repr = self.fusion_head(vit_features, gcn_pooled, time_mask)
            # 通过融合后的 video_repr 重构各模态摘要（信息保留正则）
            recon_v     = self.recon_v(video_repr)                   # (B, dim)
            recon_g     = self.recon_g(video_repr)                   # (B, dim)

        # ===== Step 4: 分类 =====
        logits_fusion = self.classifier(video_repr)
        return {
            'logits':     logits_fusion,          # (B, num_classes) 融合头 logits
            'logits_gcn': gcn_out['logits_gcn'],  # (B, num_classes) GCN 辅助头 logits
            # 重构分支（所有 fusion mode 均填充，训练时用于信息保留正则）
            # late:  recon_g 目标维度为 gcn_feat_dim；intermediate: recon_g 目标维度为 dim
            'recon_v':    recon_v,    # (B, dim)
            'recon_g':    recon_g,    # (B, dim or gcn_feat_dim) 取决于 fusion_mode
            'video_tokens': av_out['video_tokens'],
            'audio_tokens': av_out['audio_tokens'],
            'av_tokens': av_out['av_tokens'],
            'vit_repr':   vit_repr,   # (B, dim) CLS-token 或 masked-mean(vit_features)
            'gcn_repr':   gcn_repr,   # (B, gcn_feat_dim or dim) 取决于 fusion_mode
            'audio_repr': audio_repr,
            'av_repr':    av_out['av_repr'],
            'token_mask': av_out['token_mask'],
        }
    
    def _mask_aware_pool_temporal(self, features, target_T, actual_lens):
        """
        掩码感知时序池化（Phase 2-2）：将 (B, T_long, D) 分箱池化到 (B, T_short, D)。

        与普通 AdaptiveAvgPool1d 的关键区别：
        - 对每个目标 bin，只对 actual_lens 范围内的有效帧求均值；
        - padding 帧权重置 0，不参与统计，避免稀释真实信号。

        Args:
            features    : (B, T_long, D)
            target_T    : int, 目标时间步数
            actual_lens : (B,), 每条样本的有效帧数（帧粒度，非 patch 粒度）
        Returns:
            (B, target_T, D)
        """
        B, T_long, D = features.shape
        device = features.device

        # ── 1. 帧级有效掩码 (B, T_long)，float 型供乘权 ──
        frame_idx  = torch.arange(T_long, device=device)             # (T_long,)
        frame_mask = (frame_idx.unsqueeze(0) < actual_lens.unsqueeze(1)).float()  # (B, T_long)

        if T_long == target_T:
            # 长度已匹配：仅做掩码归一化，将 padding 帧清零后返回
            return features * frame_mask.unsqueeze(-1)

        # ── 2. 每帧分配到目标 bin（均匀分箱）──
        # bin_idx ∈ [0, target_T-1]，shape: (T_long,)
        bin_idx = (frame_idx.float() * target_T / T_long).long().clamp(max=target_T - 1)

        # ── 3. scatter_add 累积有效帧特征 ──
        # 扩展 bin_idx 到 (B, T_long, D) 供 scatter
        bin_idx_feat = bin_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)  # (B, T_long, D)

        # 加权特征（掩码置 0 padding 帧）
        weighted = features * frame_mask.unsqueeze(-1)   # (B, T_long, D)

        # 累积到 (B, target_T, D)
        output = torch.zeros(B, target_T, D, device=device, dtype=features.dtype)
        output.scatter_add_(1, bin_idx_feat, weighted)

        # ── 4. 统计每 bin 的有效帧数，做归一化均值 ──
        bin_idx_mask = bin_idx.unsqueeze(0).expand(B, -1)            # (B, T_long)
        bin_counts   = torch.zeros(B, target_T, device=device, dtype=features.dtype)
        bin_counts.scatter_add_(1, bin_idx_mask, frame_mask)         # (B, target_T)

        # clamp 防零除；对空 bin（全 padding）输出 0 向量
        bin_counts = bin_counts.clamp(min=1e-6).unsqueeze(-1)        # (B, target_T, 1)

        return output / bin_counts                                    # (B, target_T, D)


if __name__ == '__main__':
    #ViT(spectra_size=1400, patch_size=140, num_classes=40, dim=512, depth=8, heads=16, dim_mlp=1400, channel=1,dim_head=32)
    model = ViT(spectra_size=915,patch_size=15,num_classes=2,dim=256,depth=8,heads=8,dim_mlp=1024,channel=256,dim_head=32,dropout=0.1).cuda()
    print(model)
    x1  = torch.randn(4,915,171).cuda()
    x2 = torch.randn(4,915,128).cuda()
    y = model(x1,x2)

    print(y.shape)
