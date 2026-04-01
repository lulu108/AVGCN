import torch
from torch import nn, einsum
import torch.nn.functional as F
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

class ViT(nn.Module):
    def __init__(self, spectra_size, patch_size, num_classes, dim, depth, heads, dim_mlp, 
                 channel=1, dim_head=16, dropout=0.0, emb_dropout=0.3, sd=0.0, 
                 embedding=None, classifier=None, name='vit', 
                 video_dim=171, audio_dim=128, dataset='LMVD', use_av_cross_attn=True,
                 av_cross_heads=None, av_cross_alpha_init=0.1, **block_kwargs):
        """
        Args:
            video_dim: 视频特征维度 (LMVD: 171, D-Vlog: 136)
            audio_dim: 音频特征维度 (LMVD: 128, D-Vlog: 25)
            dataset: 数据集类型 ('LMVD' 或 'DVLOG')
        """
        super(ViT, self).__init__()
        self.name = name
        self.dataset = dataset
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.use_av_cross_attn = use_av_cross_attn
        
        self.embedding =nn.Sequential(
            PatchEmbdding(spectra_size=spectra_size, patch_size=patch_size, dim_out=dim, channel=channel),
            CLSToken(dim=dim),
            AbsPosEmbedding(spectra_size=spectra_size, patch_size=patch_size, dim=dim, cls=True),
            nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity(),
        )if embedding is None else embedding
    
        self.transformers = nn.ModuleList([
            Transformer(dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
                        dropout=dropout, sd=(sd * i / (depth - 1)))
            for i in range(depth)
        ])

        D_PROJECTION = dim // 2
        # 视频特征投影层: (B, video_dim, T) -> (B, D_PROJECTION, T)
        # LMVD: 171 -> 128, D-Vlog: 136 -> 128
        # GroupNorm(1, C): 单组归一化 = Instance Norm，不依赖 batch 统计，比 BatchNorm1d 更稳定
        self.proj_video = nn.Sequential(
            nn.Conv1d(in_channels=self.video_dim, out_channels=D_PROJECTION, kernel_size=1, stride=1),
            nn.GroupNorm(1, D_PROJECTION),  # 已替换 BatchNorm1d，消除 batch 统计波动
            nn.ReLU()
        )

        # 音频特征投影层: (B, audio_dim, T) -> (B, D_PROJECTION, T)
        # LMVD: 128 -> 128, D-Vlog: 25 -> 128
        self.proj_audio = nn.Sequential(
            nn.Conv1d(in_channels=self.audio_dim, out_channels=D_PROJECTION, kernel_size=1, stride=1),
            nn.GroupNorm(1, D_PROJECTION),  # 已替换 BatchNorm1d，消除 batch 统计波动
            nn.ReLU()
        )

        #LayerNorm 层，用于在 Cross-Attention 之前对齐特征
        # 注意：LayerNorm 的参数是特征维度，即 128
        self.ln_video = nn.LayerNorm(D_PROJECTION)
        self.ln_audio = nn.LayerNorm(D_PROJECTION)

        # ================== AV Cross Attention（可开关）==================
        # 输入维度是 D_PROJECTION (128)，不是 dim (256)
        if self.use_av_cross_attn:
            cross_heads = heads if av_cross_heads is None else av_cross_heads
            self.fusion = CrossAttentionFusion(dim=D_PROJECTION, heads=cross_heads,
                                               dropout=dropout, alpha_init=av_cross_alpha_init)
        # ==============================================================
        
        self.classifier = nn.Sequential(
            # Lambda(lambda x: x[:, 0]),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )if classifier is None else classifier

    def _embed_with_dynamic_pos(self, x_fused):
        """
        兼容可变长度输入（如 Segment-MIL）的位置编码：
        - token 数变短：裁剪 pos embedding
        - token 数变长：对非 CLS 部分做 1D 线性插值
        """
        # self.embedding = [PatchEmbdding, CLSToken, AbsPosEmbedding, Dropout]
        x = self.embedding[0](x_fused)  # (B, N, D)
        x = self.embedding[1](x)        # (B, N+1, D)

        pos_full = self.embedding[2].pos_embedding  # (1, N_full+1, D)
        n_cur = x.shape[1]
        n_full = pos_full.shape[1]

        if n_cur == n_full:
            pos = pos_full
        elif n_cur < n_full:
            # 变短：直接裁剪（保留 CLS + 前 n_cur-1 个 patch）
            pos = pos_full[:, :n_cur, :]
        else:
            # 变长：对 patch 位置编码插值，CLS 位置编码保持不变
            cls_pos = pos_full[:, :1, :]                          # (1,1,D)
            patch_pos = pos_full[:, 1:, :].transpose(1, 2)       # (1,D,N_full-1)
            patch_pos = F.interpolate(
                patch_pos, size=n_cur - 1, mode='linear', align_corners=False
            ).transpose(1, 2)                                     # (1,n_cur-1,D)
            pos = torch.cat([cls_pos, patch_pos], dim=1)         # (1,n_cur,D)

        x = x + pos
        x = self.embedding[3](x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight) #
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)    
                    
    def forward(self, X1, X2, mask=None):
        # X1: (B, 915, 171)
        # X2: (B, 915, 128)

        # 1. 维度转置 (B, T, D) -> (B, D, T) 以适配 Conv1d
        X1 = X1.permute(0, 2, 1) 
        X2 = X2.permute(0, 2, 1) 

        # 2. 特征投影 (B, 128, 915)
        X1_proj = self.proj_video(X1)
        X2_proj = self.proj_audio(X2)

        if self.use_av_cross_attn:
            # 3. 准备 Cross Attention 输入
            # CrossAttention 需要 (Batch, Seq, Dim)，所以要转置回来
            x_vid_in = X1_proj.permute(0, 2, 1) # (B, T, 128)
            x_aud_in = X2_proj.permute(0, 2, 1) # (B, T, 128)

            #在进入 Cross-Attention 融合模块前进行归一化
            # 这步能确保视频和音频特征的均值和方差在一个量级，防止某种模态“霸权”
            x_vid_in = self.ln_video(x_vid_in)
            x_aud_in = self.ln_audio(x_aud_in)

            # ================== 使用 Cross Attention 融合 ==================
            # 融合后的视频特征 (包含了音频上下文)
            x_vid_fused = self.fusion(x_vid_in, x_aud_in) # (B, T, 128)
            
            # 为了适配后面的 PatchEmbedding (它期望 channel=256)
            # 我们将 [融合后的视频] 和 [原始投影音频] 拼接
            # 这样总维度依然是 128 + 128 = 256
            
            # 此时需要转回 (B, Dim, Seq) 才能进行 cat 和传入 PatchEmbedding
            x_vid_fused = x_vid_fused.permute(0, 2, 1) # (B, 128, T)
            
            # 拼接: (B, 128, T) + (B, 128, T) -> (B, 256, T)
            # 注意：X2_proj 还是原来的 (B, 128, T)
            X_fused = torch.cat([x_vid_fused, X2_proj], dim=1) 
            # ===================================================================
        else:
            # 关闭 AV cross-attn：直接拼接 video/audio 投影（轻量融合）
            X_fused = torch.cat([X1_proj, X2_proj], dim=1)

        # 4. Patch Embedding (输入形状 B, 256, 915)
        X = self._embed_with_dynamic_pos(X_fused)

        # 5. Transformer Block 和分类（ModuleList for-loop，支持可选 mask）
        for blk in self.transformers:
            X = blk(X, mask=mask)
        X = self.classifier(X[:, 0])
        
        return X

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
                 # A4: 人脸关键点有效度阈值（对照实验用，默认 0.2）
                 face_valid_thresh=0.2,
                 # GCN 分支内部开关
                 region_fusion_mode='cross_attn',
                 gcn_temporal_mode='transformer',
                 region_mlp_dropout=0.1,
                 tcn_kernel_size=3,
                 # 分区方案："legacy6" 原始6区 | "symptom7" 症状卨7区
                 region_scheme="legacy6"):  # 'late' | 'it_cross' | 'it_bi_gate' | 'concat' | 'afi'
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
        self.patch_size  = patch_size  # 保留：用于 forward 中构造 patch-level padding mask
        self.face_valid_thresh = face_valid_thresh  # A4: 保存以便子模块引用
        self.region_scheme     = region_scheme       # 分区方案保存
        self.region_fusion_mode = region_fusion_mode
        self.gcn_temporal_mode = gcn_temporal_mode
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
            av_cross_alpha_init=av_cross_alpha_init
        )
        # 移除 ViT 的分类器，只保留特征提取
        self.vit_branch.classifier = nn.Identity()
        
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
            region_fusion_mode=region_fusion_mode,
            gcn_temporal_mode=gcn_temporal_mode,
            region_mlp_dropout=region_mlp_dropout,
            tcn_kernel_size=tcn_kernel_size,
            region_scheme=region_scheme            # 分区方案透传
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
            self.fusion_head = IntermediateBiGateFusion(
                dim=dim, heads=heads, dropout=dropout
            )
            self.concat_head = ConcatFusionHead(dim=dim, dropout=dropout)
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

        # ===== Step 1: ViT 特征提取（三种模式共用）=====
        X1 = video_features.permute(0, 2, 1)          # (B, video_dim, T)
        X2 = audio_features.permute(0, 2, 1)          # (B, audio_dim, T)

        X1_proj = self.vit_branch.proj_video(X1)      # (B, 128, T)
        X2_proj = self.vit_branch.proj_audio(X2)      # (B, 128, T)

        if self.vit_branch.use_av_cross_attn:
            x_vid_in = X1_proj.permute(0, 2, 1)           # (B, T, 128)
            x_aud_in = X2_proj.permute(0, 2, 1)           # (B, T, 128)
            x_vid_in = self.vit_branch.ln_video(x_vid_in)
            x_aud_in = self.vit_branch.ln_audio(x_aud_in)

            x_vid_fused = self.vit_branch.fusion(x_vid_in, x_aud_in)  # (B, T, 128)
            x_vid_fused = x_vid_fused.permute(0, 2, 1)                # (B, 128, T)
            X_fused = torch.cat([x_vid_fused, X2_proj], dim=1)        # (B, 256, T)
        else:
            X_fused = torch.cat([X1_proj, X2_proj], dim=1)            # (B, 256, T)

        vit_embedded = self.vit_branch._embed_with_dynamic_pos(X_fused)  # (B, T_p+1, dim)

        # ── Patch 级 padding mask（修正：让 padding token 不污染 ViT 自注意力）──────
        T_p = vit_embedded.size(1) - 1                            # 去掉 CLS，得 patch 数
        # ceil 除法：与 _mask_aware_pool_temporal 完全对齐
        actual_lens_patches = ((actual_lens + self.patch_size - 1) // self.patch_size
                               ).clamp(min=1, max=T_p)            # (B,)
        # (B, T_p)：True = 有效 patch
        patch_valid = (torch.arange(T_p, device=device).unsqueeze(0)
                       < actual_lens_patches.unsqueeze(1))        # (B, T_p)
        # CLS token 永远有效 → (B, T_p+1)
        cls_valid  = torch.ones(B, 1, dtype=torch.bool, device=device)
        full_valid = torch.cat([cls_valid, patch_valid], dim=1)   # (B, T_p+1)
        # 加性 key mask：pad 位置 -1e9，有效位置 0
        # shape (B, 1, 1, T_p+1) 可 broadcast 到 Attention1d 的 dots (B, H, N, N)
        attn_mask = (~full_valid).float() * (-1e9)                # (B, T_p+1)
        attn_mask = attn_mask[:, None, None, :]                   # (B, 1, 1, T_p+1)

        # ── 运行时接线断言（首次 forward 即可发现 shape 错误，无性能损耗后可删）──
        assert attn_mask.shape == (B, 1, 1, T_p + 1), (
            f"[ViT attn_mask] shape mismatch: "
            f"expected ({B}, 1, 1, {T_p + 1}), got {tuple(attn_mask.shape)}"
        )
        # 诊断：有效 patch 占比（接近 1.0 = padding 少；接近 0.5 = padding 严重）
        valid_patch_ratio = actual_lens_patches.float().mean().item() / T_p  # noqa

        # ── 带 mask 逐层前向，防止 padding token 污染注意力 ──────────────────────
        # 注意：这里用的是 self.vit_branch.transformers（ModuleList），
        #       而 ViT.forward() 独立路径不传 mask（因为没有 actual_lens）——两条路不冲突。
        vit_transformed = vit_embedded
        for blk in self.vit_branch.transformers:
            vit_transformed = blk(vit_transformed, mask=attn_mask)  # (B, T_p+1, dim)

        # ===== Step 2 & 3: 根据 fusion_mode 分发 =====
        if self.fusion_mode == 'late':
            # ── Late Fusion ──────────────────────────────────────────────
            # ViT CLS-token → (B, dim)
            vit_repr = vit_transformed[:, 0, :]
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
            vit_features = vit_transformed[:, 1:, :]                      # (B, T_p, dim)

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
                video_repr_token = self.fusion_head(vit_features, gcn_pooled, time_mask)
                video_repr_sum = self.concat_head(vit_repr, gcn_repr)
                video_repr = 0.5 * (video_repr_token + video_repr_sum)
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
            'vit_repr':   vit_repr,   # (B, dim) CLS-token 或 masked-mean(vit_features)
            'gcn_repr':   gcn_repr,   # (B, gcn_feat_dim or dim) 取决于 fusion_mode
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