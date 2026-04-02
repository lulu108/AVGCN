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
                 modality_mode='fusion',
                 single_modality_clean_path=True,
                 use_strong_audio_encoder=False,
                 use_strong_video_encoder=False,
                 audio_fixed_len=128,
                 video_fixed_len=128,
                 video_use_delta=True,
                 **block_kwargs):
        super(ViT, self).__init__()
        dataset_upper = str(dataset).upper()
        from av_backbone_common import AVBackboneCore
        backbone_cls = AVBackboneCore

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
            modality_mode=modality_mode,
            single_modality_clean_path=single_modality_clean_path,
            use_strong_audio_encoder=use_strong_audio_encoder,
            use_strong_video_encoder=use_strong_video_encoder,
            audio_fixed_len=audio_fixed_len,
            video_fixed_len=video_fixed_len,
            video_use_delta=video_use_delta,
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

# fusion heads and helpers were moved to fusion_common.py


class ViT_GCN_Fusion(nn.Module):
    """Dataset-router wrapper.

    Keep external API unchanged while routing to dataset-specific fusion assembly:
    - LMVD: LMVDViTGCNFusion
    - DVLOG: DVLOGViTGCNFusion
    """

    def __init__(self,
                 spectra_size=915, patch_size=15, dim=256, depth=8, heads=8, dim_mlp=1024,
                 sd=0.0,
                 gcn_out_dim=32, gcn_nhead=4,
                 use_global_branch=True,
                 global_alpha_init=0.2,
                 num_classes=2, dropout=0.45, channel=256,
                 video_dim=171, audio_dim=128, dataset='LMVD',
                 fusion_mode='it_cross',
                 use_av_cross_attn=None,
                 av_cross_heads=None,
                 av_cross_alpha_init=0.1,
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
                 single_modality_clean_path=True,
                 use_strong_audio_encoder=False,
                 use_strong_video_encoder=False,
                 audio_fixed_len=128,
                 video_fixed_len=128,
                 video_use_delta=True,
                 face_valid_thresh=0.2,
                 region_scheme="legacy6",
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
                 concat_blend_init=0.3):
        super().__init__()

        dataset_upper = str(dataset).upper()
        if dataset_upper == 'DVLOG':
            from vit_gcn_fusion_dvlog import DVLOGViTGCNFusion as _Impl
        else:
            from vit_gcn_fusion_lmvd import LMVDViTGCNFusion as _Impl

        self.impl = _Impl(
            spectra_size=spectra_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_mlp=dim_mlp,
            sd=sd,
            gcn_out_dim=gcn_out_dim,
            gcn_nhead=gcn_nhead,
            use_global_branch=use_global_branch,
            global_alpha_init=global_alpha_init,
            num_classes=num_classes,
            dropout=dropout,
            channel=channel,
            video_dim=video_dim,
            audio_dim=audio_dim,
            dataset=dataset,
            fusion_mode=fusion_mode,
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
            use_temporal_attn_pool=use_temporal_attn_pool,
            temporal_attn_pool_dvlog_only=temporal_attn_pool_dvlog_only,
            temporal_attn_pool_alpha_init=temporal_attn_pool_alpha_init,
            use_legacy_av_backbone=use_legacy_av_backbone,
            use_feature_sequence_encoder=use_feature_sequence_encoder,
            fs_video_local_blocks=fs_video_local_blocks,
            fs_audio_local_blocks=fs_audio_local_blocks,
            fs_video_global_depth=fs_video_global_depth,
            fs_audio_global_depth=fs_audio_global_depth,
            fs_dilated_audio=fs_dilated_audio,
            fs_fusion_dropout=fs_fusion_dropout,
            single_modality_clean_path=single_modality_clean_path,
            use_strong_audio_encoder=use_strong_audio_encoder,
            use_strong_video_encoder=use_strong_video_encoder,
            audio_fixed_len=audio_fixed_len,
            video_fixed_len=video_fixed_len,
            video_use_delta=video_use_delta,
            face_valid_thresh=face_valid_thresh,
            region_scheme=region_scheme,
            region_fusion_mode=region_fusion_mode,
            gcn_temporal_mode=gcn_temporal_mode,
            region_mlp_dropout=region_mlp_dropout,
            tcn_kernel_size=tcn_kernel_size,
            modality_mode=modality_mode,
            use_ia_recalibration=use_ia_recalibration,
            ia_recal_bidirectional=ia_recal_bidirectional,
            ia_recal_mode=ia_recal_mode,
            ia_hidden_ratio=ia_hidden_ratio,
            ia_dropout=ia_dropout,
            ia_alpha_init=ia_alpha_init,
            concat_blend_init=concat_blend_init,
        )

    @property
    def classifier(self):
        return self.impl.classifier

    @classifier.setter
    def classifier(self, module):
        self.impl.classifier = module

    def forward(self, video_features, audio_features, face_regions, actual_lens, quality=None):
        return self.impl(video_features, audio_features, face_regions, actual_lens, quality=quality)


if __name__ == '__main__':
    #ViT(spectra_size=1400, patch_size=140, num_classes=40, dim=512, depth=8, heads=16, dim_mlp=1400, channel=1,dim_head=32)
    model = ViT(spectra_size=915,patch_size=15,num_classes=2,dim=256,depth=8,heads=8,dim_mlp=1024,channel=256,dim_head=32,dropout=0.1).cuda()
    print(model)
    x1  = torch.randn(4,915,171).cuda()
    x2 = torch.randn(4,915,128).cuda()
    y = model(x1,x2)

    print(y.shape)
