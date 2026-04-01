import torch
from torch import nn, einsum
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
        dots = dots.clamp(min=-1e4, max=1e4)  # 防止极端值导致 softmax 溢出
        attn = dots.softmax(dim=-1)  # (2, 16, 11, 11)
        attn = attn.nan_to_num(nan=0.0)  # 安全网: 防止全 -inf 行产生 NaN

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
    """
    多层跨模态注意力融合模块（改进版）
    特点:
    1. 多层堆叠（层次化学习）- 优化为2层防止过拟合
    2. 双向注意力（视频⟷音频互查询）
    3. 残差连接 + 层归一化
    4. Feed-Forward Network 增强表达能力
    5. 增强正则化（Dropout=0.2）
    """
    def __init__(self, dim, heads=8, layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.layers = layers
        
        # 多层注意力堆叠
        self.vid_to_aud_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        
        if bidirectional:
            self.aud_to_vid_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
                for _ in range(layers)
            ])
        
        # 每层的归一化和FFN
        self.vid_norms1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.vid_norms2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.aud_norms1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.aud_norms2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        
        self.vid_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            ) for _ in range(layers)
        ])
        
        if bidirectional:
            self.aud_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout)
                ) for _ in range(layers)
            ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_vid, x_aud, vid_kpm=None, aud_kpm=None):
        """
        Args:
            x_vid: (B, T, D) 视频特征
            x_aud: (B, T, D) 音频特征
            vid_kpm: (B, T) bool, True=padding（视频 key padding mask）
            aud_kpm: (B, T) bool, True=padding（音频 key padding mask）
        Returns:
            融合后的视频和音频特征（如果bidirectional=True）
            否则只返回增强的视频特征
        """
        # ==== 预检测全 mask 样本（v_missing 导致 vid_kpm 全 True）====
        # 避免 MultiheadAttention 在全 -inf softmax 时产生 NaN
        vid_all_masked = None
        if vid_kpm is not None:
            vid_all_masked = vid_kpm.all(dim=1)  # (B,) True=该样本视频完全缺失
        aud_all_masked = None
        if aud_kpm is not None:
            aud_all_masked = aud_kpm.all(dim=1)  # (B,)

        for i in range(self.layers):
            # === 视频查询音频（Video queries Audio）===
            vid_residual = x_vid
            # LayerNorm -> Attention（各模态用独立 norm，避免尺度耦合）
            vid_normed = self.vid_norms1[i](x_vid)
            aud_normed = self.aud_norms1[i](x_aud)
            
            # 转置为 (T, B, D)
            q = vid_normed.permute(1, 0, 2)
            k = aud_normed.permute(1, 0, 2)
            v = aud_normed.permute(1, 0, 2)
            
            attn_out, _ = self.vid_to_aud_layers[i](
                q, k, v, key_padding_mask=aud_kpm
            )
            attn_out = attn_out.permute(1, 0, 2)  # 转回 (B, T, D)
            
            # 保护：audio 全 mask 的样本，该支输出置 0（保留 residual）
            if aud_all_masked is not None and aud_all_masked.any():
                mask = aud_all_masked.unsqueeze(1).unsqueeze(2)  # (B,1,1)
                attn_out = torch.where(mask, torch.zeros_like(attn_out), attn_out)
            
            # 残差 + Dropout
            x_vid = vid_residual + self.dropout(attn_out)
            
            # FFN增强
            vid_residual = x_vid
            ffn_out = self.vid_ffns[i](self.vid_norms2[i](x_vid))
            x_vid = vid_residual + ffn_out
            
            # === 音频查询视频（Audio queries Video）双向融合 ===
            if self.bidirectional:
                aud_residual = x_aud
                # LayerNorm -> Attention（各模态用独立 norm）
                aud_normed = self.aud_norms1[i](x_aud)
                vid_normed = self.vid_norms1[i](x_vid)
                
                # 转置为 (T, B, D)
                q = aud_normed.permute(1, 0, 2)
                k = vid_normed.permute(1, 0, 2)
                v = vid_normed.permute(1, 0, 2)
                
                attn_out, _ = self.aud_to_vid_layers[i](
                    q, k, v, key_padding_mask=vid_kpm
                )
                attn_out = attn_out.permute(1, 0, 2)
                
                # 保护：video 全 mask 的样本，audio→video attention 输出置 0
                if vid_all_masked is not None and vid_all_masked.any():
                    mask = vid_all_masked.unsqueeze(1).unsqueeze(2)  # (B,1,1)
                    attn_out = torch.where(mask, torch.zeros_like(attn_out), attn_out)
                
                # 残差 + Dropout
                x_aud = aud_residual + self.dropout(attn_out)
                
                # FFN增强
                aud_residual = x_aud
                ffn_out = self.aud_ffns[i](self.aud_norms2[i](x_aud))
                x_aud = aud_residual + ffn_out
        
        if self.bidirectional:
            return x_vid, x_aud
        else:
            return x_vid


class IntermediateAttentionFusion(nn.Module):
    """
    IA (Intermediate Attention): 跨模态注意力门控模块
    
    核心思想：
    - 计算 Q(self)·K(other) 得到跨模态注意力 logits
    - **不使用**对方的 Value 生成新 token（避免噪声注入）
    - 用注意力 logits 的 per-token 统计量（沿 key 维度均值）生成门控信号
    - 用 gate 残差重加权自己的 token: x = x + dropout(x * gate)
    
    与 CrossAttentionFusion 的区别：
    - CrossAttn: output = softmax(Q·K^T) @ V_other → 注入对方 Value
    - IA:        gate = σ(MLP(mean_j(Q·K^T/√d))) → x = x + x * gate（自我增强）
    
    适用场景：
    - 小数据/跨域场景（DVLOG），避免低质量模态噪声注入高质量模态
    - 模态质量差异大时更稳定
    """
    def __init__(self, dim, heads=8, layers=2, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.layers = layers
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        
        # 每层的跨模态 Q/K 投影 + gate MLP + LayerNorm
        self.vid_q_projs = nn.ModuleList()
        self.aud_k_for_vid_projs = nn.ModuleList()
        self.aud_q_projs = nn.ModuleList()
        self.vid_k_for_aud_projs = nn.ModuleList()
        self.vid_gate_mlps = nn.ModuleList()
        self.aud_gate_mlps = nn.ModuleList()
        self.vid_norms = nn.ModuleList()
        self.aud_norms = nn.ModuleList()
        
        for _ in range(layers):
            self.vid_q_projs.append(nn.Linear(dim, dim, bias=False))
            self.aud_k_for_vid_projs.append(nn.Linear(dim, dim, bias=False))
            self.aud_q_projs.append(nn.Linear(dim, dim, bias=False))
            self.vid_k_for_aud_projs.append(nn.Linear(dim, dim, bias=False))
            
            # gate: per-head relevance (H 维) → 标量门控 (1 维)
            self.vid_gate_mlps.append(nn.Sequential(
                nn.Linear(heads, heads * 2),
                nn.ReLU(),
                nn.Linear(heads * 2, 1),
                nn.Sigmoid()
            ))
            self.aud_gate_mlps.append(nn.Sequential(
                nn.Linear(heads, heads * 2),
                nn.ReLU(),
                nn.Linear(heads * 2, 1),
                nn.Sigmoid()
            ))
            
            self.vid_norms.append(nn.LayerNorm(dim))
            self.aud_norms.append(nn.LayerNorm(dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_vid, x_aud, vid_kpm=None, aud_kpm=None):
        """
        Args:
            x_vid: (B, T, D) 视频特征
            x_aud: (B, T, D) 音频特征
            vid_kpm: (B, T) bool, True=padding（视频 key padding mask）
            aud_kpm: (B, T) bool, True=padding（音频 key padding mask）
        Returns:
            门控增强后的 (x_vid, x_aud)
        """
        # ==== 预检测全 mask 样本 ====
        vid_all_masked = None
        if vid_kpm is not None:
            vid_all_masked = vid_kpm.all(dim=1)  # (B,)
        aud_all_masked = None
        if aud_kpm is not None:
            aud_all_masked = aud_kpm.all(dim=1)  # (B,)

        # 预构造 key 维度的有效帧计数（用于 masked mean，避免 padding 位置污染 gate）
        # aud_valid_count: 计算 video-queries-audio 的 relevance 时，audio 有效帧数
        # vid_valid_count: 计算 audio-queries-video 的 relevance 时，video 有效帧数
        if aud_kpm is not None:
            # aud_kpm: (B, T) True=padding → ~aud_kpm: True=valid
            aud_valid_count = (~aud_kpm).sum(dim=-1, keepdim=True).unsqueeze(1).float().clamp(min=1)  # (B, 1, 1)
        else:
            aud_valid_count = None
        if vid_kpm is not None:
            vid_valid_count = (~vid_kpm).sum(dim=-1, keepdim=True).unsqueeze(1).float().clamp(min=1)  # (B, 1, 1)
        else:
            vid_valid_count = None
        
        for i in range(self.layers):
            # === Video tokens gated by Audio relevance ===
            vid_normed = self.vid_norms[i](x_vid)
            aud_normed = self.vid_norms[i](x_aud)  # 用同一 LN 保持尺度一致
            
            q = rearrange(self.vid_q_projs[i](vid_normed),
                          'b t (h d) -> b h t d', h=self.heads)
            k = rearrange(self.aud_k_for_vid_projs[i](aud_normed),
                          'b t (h d) -> b h t d', h=self.heads)
            
            # (B, H, Tv, Ta) — raw attention logits (未 softmax)
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale
            # 将 padding 位置的 logits 置零，避免污染 mean
            if aud_kpm is not None:
                # aud_kpm: (B, Ta) → (B, 1, 1, Ta) 广播到 (B, H, Tv, Ta)
                attn_logits = attn_logits.masked_fill(
                    aud_kpm.unsqueeze(1).unsqueeze(2), 0.0
                )
            # Per-token 跨模态相关性: 沿 audio 位置取 masked mean → (B, H, T)
            if aud_valid_count is not None:
                relevance = attn_logits.sum(dim=-1) / aud_valid_count  # (B, H, Tv)
            else:
                relevance = attn_logits.mean(dim=-1)
            relevance = relevance.permute(0, 2, 1)  # (B, T, H)
            gate_vid = self.vid_gate_mlps[i](relevance)  # (B, T, 1)
            # 保护：audio 全 mask 时，gate 置 0（不增强 video）
            if aud_all_masked is not None and aud_all_masked.any():
                mask = aud_all_masked.unsqueeze(1).unsqueeze(2)  # (B,1,1)
                gate_vid = torch.where(mask, torch.zeros_like(gate_vid), gate_vid)
            x_vid = x_vid + self.dropout(x_vid * gate_vid)
            
            # === Audio tokens gated by Video relevance ===
            aud_normed = self.aud_norms[i](x_aud)
            vid_normed = self.aud_norms[i](x_vid)
            
            q = rearrange(self.aud_q_projs[i](aud_normed),
                          'b t (h d) -> b h t d', h=self.heads)
            k = rearrange(self.vid_k_for_aud_projs[i](vid_normed),
                          'b t (h d) -> b h t d', h=self.heads)
            
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale
            # 将 padding 位置的 logits 置零
            if vid_kpm is not None:
                attn_logits = attn_logits.masked_fill(
                    vid_kpm.unsqueeze(1).unsqueeze(2), 0.0
                )
            if vid_valid_count is not None:
                relevance = attn_logits.sum(dim=-1) / vid_valid_count  # (B, H, Ta)
            else:
                relevance = attn_logits.mean(dim=-1)
            relevance = relevance.permute(0, 2, 1)  # (B, T, H)
            gate_aud = self.aud_gate_mlps[i](relevance)  # (B, T, 1)
            # 保护：video 全 mask 时，gate 置 0（不增强 audio）
            if vid_all_masked is not None and vid_all_masked.any():
                mask = vid_all_masked.unsqueeze(1).unsqueeze(2)  # (B,1,1)
                gate_aud = torch.where(mask, torch.zeros_like(gate_aud), gate_aud)
            x_aud = x_aud + self.dropout(x_aud * gate_aud)
        
        return x_vid, x_aud


class ViT(nn.Module):
    """
    改进的 Vision Transformer（独立编码版）
    特点:
    1. 视频和音频独立编码（避免简单拼接）
    2. 使用跨模态注意力深度融合
    3. 时序对齐确保同步
    """
    def __init__(self, spectra_size, patch_size, num_classes, dim, depth, heads, dim_mlp, 
                 channel=1, dim_head=16, dropout=0.0, emb_dropout=0.3, sd=0.0, 
                 embedding=None, classifier=None, name='vit', 
                 video_dim=171, audio_dim=128, dataset='LMVD', 
                 cross_attn_layers=3, **block_kwargs):
        """
        Args:
            video_dim: 视频特征维度 (LMVD: 171, D-Vlog: 136)
            audio_dim: 音频特征维度 (LMVD: 128, D-Vlog: 25)
            dataset: 数据集类型 ('LMVD' 或 'DVLOG')
        """
        super(ViT, self).__init__()
        self.name = name
        self.patch_size = patch_size
        self.dataset = dataset
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        
        self.embedding =nn.Sequential(
            PatchEmbdding(spectra_size=spectra_size, patch_size=patch_size, dim_out=dim, channel=channel),
            CLSToken(dim=dim),
            AbsPosEmbedding(spectra_size=spectra_size, patch_size=patch_size, dim=dim, cls=True),
            nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity(),
        )if embedding is None else embedding
    
        self.transformers = []
        for i in range(depth):
            self.transformers.append(
                Transformer(dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp, dropout=dropout, sd=(sd * i / (depth -1)))
            )
        self.transformers = nn.ModuleList(self.transformers)

        D_PROJECTION = dim // 2
        # 视频特征投影层: (B, video_dim, T) -> (B, D_PROJECTION, T)
        # LMVD: 171 -> 128, D-Vlog: 136 -> 128
        self.proj_video = nn.Sequential(
            nn.Conv1d(in_channels=self.video_dim, out_channels=D_PROJECTION, kernel_size=1, stride=1),
            nn.BatchNorm1d(D_PROJECTION),
            nn.ReLU()
        )

        # 音频特征投影层: (B, audio_dim, T) -> (B, D_PROJECTION, T)
        # LMVD: 128 -> 128, D-Vlog: 25 -> 128
        self.proj_audio = nn.Sequential(
            nn.Conv1d(in_channels=self.audio_dim, out_channels=D_PROJECTION, kernel_size=1, stride=1),
            nn.BatchNorm1d(D_PROJECTION),
            nn.ReLU()
        )

        # ================== Temporal Conv1D Refine ==================
        # 在 cross-attn 前对投影特征做局部时序整形，减少噪声尖峰
        # 用 GroupNorm(1, D) ≈ LayerNorm，对小 batch 更稳定
        self.refine_video = nn.Sequential(
            nn.Conv1d(D_PROJECTION, D_PROJECTION, kernel_size=3, padding=1, groups=1),
            nn.GroupNorm(1, D_PROJECTION),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.refine_audio = nn.Sequential(
            nn.Conv1d(D_PROJECTION, D_PROJECTION, kernel_size=3, padding=1, groups=1),
            nn.GroupNorm(1, D_PROJECTION),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # ==============================================================

        #LayerNorm 层，用于在 Cross-Attention 之前对齐特征
        # 注意：LayerNorm 的参数是特征维度，即 128
        self.ln_video = nn.LayerNorm(D_PROJECTION)
        self.ln_audio = nn.LayerNorm(D_PROJECTION)

        # ================== 新增：多层Cross Attention 模块 ==================
        # 【优化】2层堆叠的跨模态注意力,平衡性能与过拟合风险
        self.fusion = CrossAttentionFusion(
            dim=D_PROJECTION, 
            heads=heads, 
            layers=2,  # 【关键】从3层减至2层,降低复杂度防止过拟合
            dropout=0.2,  # 【关键】增大Dropout防止过拟合
            bidirectional=True  # 双向注意力
        )
        
        self.classifier = nn.Sequential(
            # Lambda(lambda x: x[:, 0]),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )if classifier is None else classifier
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight) #
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)    
                    
    def forward(self, X1, X2, actual_lens=None):
        # X1: (B, T, D_VIDEO)
        # X2: (B, T, D_AUDIO)

        # 1. 维度转置 (B, T, D) -> (B, D, T) 以适配 Conv1d
        X1 = X1.permute(0, 2, 1) 
        X2 = X2.permute(0, 2, 1) 

        # 2. 特征投影 (B, 128, 915)
        X1_proj = self.proj_video(X1)
        X2_proj = self.proj_audio(X2)

        # 2.5 Temporal Conv1D Refine（局部时序整形，减少噪声尖峰）
        X1_proj = self.refine_video(X1_proj)  # (B, D_PROJ, T)
        X2_proj = self.refine_audio(X2_proj)  # (B, D_PROJ, T)

        # 3. 准备 Cross Attention 输入
        # CrossAttention 需要 (Batch, Seq, Dim)，所以要转置回来
        x_vid_in = X1_proj.permute(0, 2, 1) # (B, 915, 128)
        x_aud_in = X2_proj.permute(0, 2, 1) # (B, 915, 128)

        #在进入 Cross-Attention 融合模块前进行归一化
        # 这步能确保视频和音频特征的均值和方差在一个量级，防止某种模态“霸权”
        x_vid_in = self.ln_video(x_vid_in)
        x_aud_in = self.ln_audio(x_aud_in)

        # ================== 构造帧级 key_padding_mask ==================
        vid_kpm = aud_kpm = None
        if actual_lens is not None:
            B_f, T_f = x_vid_in.shape[0], x_vid_in.shape[1]
            time_valid = (torch.arange(T_f, device=x_vid_in.device).unsqueeze(0)
                          < actual_lens.unsqueeze(1))  # (B, T) True=有效
            vid_kpm = ~time_valid  # True=padding
            aud_kpm = ~time_valid

        # ================== 修改：使用多层双向 Cross Attention 融合 ==================
        # 双向融合: 视频⟷音频互相查询,获得深度交互的特征
        x_vid_fused, x_aud_fused = self.fusion(
            x_vid_in, x_aud_in, vid_kpm=vid_kpm, aud_kpm=aud_kpm
        ) 
        # x_vid_fused: (B, 915, 128) - 包含音频上下文的视频特征
        # x_aud_fused: (B, 915, 128) - 包含视频上下文的音频特征
        
        # 为了适配后面的 PatchEmbedding (它期望 channel=256)
        # 我们将 [融合后的视频] 和 [融合后的音频] 拼接，这样总维度依然是 128 + 128 = 256
        
        # 此时需要转回 (B, Dim, Seq) 才能进行 cat 和传入 PatchEmbedding
        x_vid_fused = x_vid_fused.permute(0, 2, 1) # (B, 128, 915)
        x_aud_fused = x_aud_fused.permute(0, 2, 1) # (B, 128, 915)
        
        # 拼接: (B, 128, 915) + (B, 128, 915) -> (B, 256, 915)
        X_fused = torch.cat([x_vid_fused, x_aud_fused], dim=1) 
        # ===================================================================

        # 4. Patch Embedding (输入形状 B, 256, 915)
        X = self.embedding(X_fused)

        # 5. Transformer Block（可选 padding mask，抑制对无效 patch 的注意力）
        attn_mask = None
        if actual_lens is not None:
            B_size = X.size(0)
            T_patches = X.size(1) - 1  # 减去 CLS token
            # 向上取整：只要 patch 内含有 ≥1 帧真实数据就视为有效，避免尾部信息丢失
            actual_lens_patches = ((actual_lens + self.patch_size - 1) // self.patch_size).clamp(min=1, max=T_patches)
            patch_valid = (torch.arange(T_patches, device=X.device).expand(B_size, -1)
                           < actual_lens_patches.unsqueeze(1))
            cls_valid = torch.ones(B_size, 1, dtype=torch.bool, device=X.device)
            valid_mask = torch.cat([cls_valid, patch_valid], dim=1)   # (B, T_patches+1)
            # Attention1d 使用加性掩码: padding 位置为 -1e9，有效位置为 0
            attn_mask = torch.zeros(B_size, 1, 1, X.size(1), device=X.device)
            attn_mask = attn_mask.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(1), -1e9)

        for transformer_layer in self.transformers:
            X = transformer_layer(X, mask=attn_mask)

        # 6. 分类（取 CLS token）
        X = self.classifier(X[:, 0])
        
        return X

class GatedFusion(nn.Module):
    """
    轻量级门控融合模块（Softmax Gate + LayerNorm + Residual）
    
    核心思想：
    - 把三个模态的 pooled 向量拼接后通过 MLP，产出 3 个标量门控权重
    - 用 softmax 归一化（天然 w1+w2+w3=1，输出尺度稳定，概率校准更好）
    - 加权求和后加 residual（以 video 为锚点）再 LayerNorm
    
    公式：
        w1, w2, w3 = softmax(MLP([v; a; g]))
        f0 = w1 * v + w2 * a + w3 * g
        f  = LayerNorm(v + f0)
    
    相比 cross-attention 融合：
    - 参数量极小（只有一个小 MLP + LN）
    - 不容易让 logit 尺度爆/塌（softmax 保证权重归一化）
    - 在 DVLOG 等小数据场景下更稳定
    """
    def __init__(self, dim, num_modalities=3, dropout=0.1):
        super().__init__()
        self.num_modalities = num_modalities
        
        # 标量门控 MLP：[v;a;g] (dim*3) → hidden → 3
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * num_modalities, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_modalities)  # 输出 3 个 logit
        )
        
        # 输出 LayerNorm（稳定尺度）
        self.ln = nn.LayerNorm(dim)
    
    def forward(self, v, a, g):
        """
        Args:
            v: (B, dim) - 视频 pooled 向量（base / anchor 模态）
            a: (B, dim) - 音频 pooled 向量
            g: (B, dim) - GCN pooled 向量
        Returns:
            f: (B, dim) - 融合后的向量
            gate_weights: (B, 3) - 门控权重（可用于日志/可视化）
        """
        # 1. 拼接三模态向量
        concat = torch.cat([v, a, g], dim=-1)  # (B, dim*3)
        
        # 2. 计算门控权重（softmax 归一化，天然 w1+w2+w3=1）
        gate_logits = self.gate_mlp(concat)     # (B, 3)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, 3)
        
        w1 = gate_weights[:, 0:1]  # (B, 1)
        w2 = gate_weights[:, 1:2]  # (B, 1)
        w3 = gate_weights[:, 2:3]  # (B, 1)
        
        # 3. 加权融合
        f0 = w1 * v + w2 * a + w3 * g  # (B, dim)
        
        # 4. Residual（以 video 为锚点）+ LayerNorm
        f = self.ln(v + f0)  # (B, dim)
        
        return f, gate_weights


class ViT_GCN_Fusion(nn.Module):
    """
    ViT-GCN 门控融合模型：
    - ViT 分支: 提取视频+音频时序特征，CLS token 作为 video pooled 向量
    - Audio 分支: 从 ViT 内部分离音频 pooled 向量
    - GCN 分支: 提取面部关键点空间特征，masked mean pooling 后投影
    - 门控融合: softmax(MLP([v;a;g])) 加权 + residual + LayerNorm
    """
    def __init__(self, 
                 # ViT 参数
                 spectra_size=915, patch_size=15, dim=256, depth=8, heads=8, dim_mlp=1024,
                 # GCN 参数
                 gcn_out_dim=32, gcn_nhead=4,
                 # 通用参数
                 num_classes=2, dropout=0.45, channel=256,
                 emb_dropout=0.1,
                 # 数据集参数
                 video_dim=171, audio_dim=128, dataset='LMVD',
                 # 融合策略
                 fusion_strategy='ET'):
        """
        Args:
            video_dim: 视频特征维度 (LMVD: 171, D-Vlog: 136)
            audio_dim: 音频特征维度 (LMVD: 128, D-Vlog: 25)
            dataset: 数据集类型 ('LMVD' 或 'DVLOG')
            emb_dropout: ViT embedding dropout（原硬编码 0.3，现可配置）
            fusion_strategy: 融合策略（'ET'=早融合, 'LT'=晚融合, 'IT'=中融合, 'IA'=注意力门控）
        """
        super().__init__()
        
        self.dim = dim
        self.patch_size = patch_size
        self.dataset = dataset
        self.fusion_strategy = fusion_strategy
        D_PROJ = dim // 2  # 128
        
        # ===== 1. ViT 分支（提取视频+音频时序特征）=====
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
            emb_dropout=emb_dropout,  # 【优化】从外部传入，不再默认 0.3
            video_dim=video_dim,
            audio_dim=audio_dim,
            dataset=dataset
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
            dropout=dropout,
            spatial_dropout=dropout,
            temporal_dropout=dropout,
            classifier_dropout=dropout
        )
        # GCN 输出维度 = fused_dim (区域64 + 全局64 = 128)
        gcn_feat_dim = gcn_out_dim * 4  # 128
        self.gcn_branch.classifier = nn.Identity()
        
        # ===== 3. 模态投影层（统一到 dim 维度）=====
        # GCN pooled → dim
        self.gcn_proj = nn.Sequential(
            nn.Linear(gcn_feat_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Audio pooled → dim（输入维度取决于融合策略）
        if fusion_strategy in ('ET', 'IA'):
            # ET/IA: audio pooled 来自 embedding 前 (D_PROJ=128)，需投影到 dim=256
            self.audio_proj = nn.Sequential(
                nn.Linear(D_PROJ, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:  # LT, IT
            # LT/IT: audio pooled 来自 transformer 输出 (已是 dim=256)，只做 LN+Dropout
            self.audio_proj = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Dropout(dropout)
            )
        
        # ===== 4. 融合策略专用模块 =====
        if fusion_strategy == 'IA':
            # IA: 跨模态注意力门控（替代 CrossAttentionFusion）
            self.ia_fusion = IntermediateAttentionFusion(
                dim=D_PROJ, heads=heads, layers=2, dropout=dropout
            )
        
        elif fusion_strategy == 'LT':
            # LT: 各模态独立 embedding + 独立 transformer + 晚期 cross-attention
            self.embedding_video = nn.Sequential(
                PatchEmbdding(spectra_size, patch_size, dim, channel=D_PROJ),
                CLSToken(dim),
                AbsPosEmbedding(spectra_size, patch_size, dim, cls=True),
                nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity(),
            )
            self.embedding_audio = nn.Sequential(
                PatchEmbdding(spectra_size, patch_size, dim, channel=D_PROJ),
                CLSToken(dim),
                AbsPosEmbedding(spectra_size, patch_size, dim, cls=True),
                nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity(),
            )
            self.transformers_video = nn.ModuleList([
                Transformer(dim, heads=heads, dim_head=dim // heads,
                            dim_mlp=dim_mlp, dropout=dropout)
                for _ in range(depth)
            ])
            self.transformers_audio = nn.ModuleList([
                Transformer(dim, heads=heads, dim_head=dim // heads,
                            dim_mlp=dim_mlp, dropout=dropout)
                for _ in range(depth)
            ])
            self.late_fusion = CrossAttentionFusion(
                dim=dim, heads=heads, layers=2, dropout=dropout, bidirectional=True
            )
        
        elif fusion_strategy == 'IT':
            # IT: 前 k 层独立编码 → cross-attention → 后 (depth-k) 层共享精炼
            self.it_split_k = max(1, depth // 2)
            
            self.embedding_video = nn.Sequential(
                PatchEmbdding(spectra_size, patch_size, dim, channel=D_PROJ),
                CLSToken(dim),
                AbsPosEmbedding(spectra_size, patch_size, dim, cls=True),
                nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity(),
            )
            self.embedding_audio = nn.Sequential(
                PatchEmbdding(spectra_size, patch_size, dim, channel=D_PROJ),
                CLSToken(dim),
                AbsPosEmbedding(spectra_size, patch_size, dim, cls=True),
                nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity(),
            )
            self.transformers_pre_video = nn.ModuleList([
                Transformer(dim, heads=heads, dim_head=dim // heads,
                            dim_mlp=dim_mlp, dropout=dropout)
                for _ in range(self.it_split_k)
            ])
            self.transformers_pre_audio = nn.ModuleList([
                Transformer(dim, heads=heads, dim_head=dim // heads,
                            dim_mlp=dim_mlp, dropout=dropout)
                for _ in range(self.it_split_k)
            ])
            self.it_fusion = CrossAttentionFusion(
                dim=dim, heads=heads, layers=2, dropout=dropout, bidirectional=True
            )
            self.transformers_post = nn.ModuleList([
                Transformer(dim, heads=heads, dim_head=dim // heads,
                            dim_mlp=dim_mlp, dropout=dropout)
                for _ in range(depth - self.it_split_k)
            ])
            self.it_merge_norm = nn.LayerNorm(dim)
        
        # else: ET — 使用 vit_branch 中已有的 CrossAttentionFusion + 共享 embedding/transformers
        
        # ===== 4. 门控融合（替代 cross-attn + deep_fusion）=====
        self.gated_fusion = GatedFusion(
            dim=dim,
            num_modalities=3,
            dropout=dropout
        )
        
        # ===== 5. 【A策略】可学习视觉缺失 token =====
        # 当 v_missing=True 时，将 video pooled 替换为此向量再做门控融合，
        # 让模型通过梯度学到 "缺失视觉" 的表示，使 gate 自然偏向 audio。
        self.v_missing_embed = nn.Parameter(torch.zeros(dim))
        # 小幅随机初始化，避免初始时恰好与真实 video 表征混淆
        nn.init.normal_(self.v_missing_embed, mean=0.0, std=0.02)
        
        # ===== 6. 最终分类器 =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
        
    def forward(self, video_features, audio_features, face_regions, actual_lens, v_missing=None):
        """
        根据 fusion_strategy 分派到对应的前向流程：
        - ET: 早融合（当前默认）  - IA: 注意力门控（最小改动版 ET）
        - LT: 晚融合（各模态独立编码后交互）
        - IT: 中融合（前 k 层独立 → 交互 → 后 depth-k 层精炼）
        
        Args:
            v_missing: (B,) bool, True 表示该样本视觉模态缺失
        """
        if self.fusion_strategy in ('ET', 'IA'):
            return self._forward_early_fusion(video_features, audio_features,
                                              face_regions, actual_lens, v_missing=v_missing)
        elif self.fusion_strategy == 'LT':
            return self._forward_lt(video_features, audio_features,
                                    face_regions, actual_lens, v_missing=v_missing)
        elif self.fusion_strategy == 'IT':
            return self._forward_it(video_features, audio_features,
                                    face_regions, actual_lens, v_missing=v_missing)
        else:
            raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")
    
    # ---------- 共享辅助方法 ----------
    
    def _project_and_normalize(self, video_features, audio_features):
        """特征投影 + LayerNorm（所有策略共享的前处理）"""
        X1 = video_features.permute(0, 2, 1)   # (B, D_video, T)
        X2 = audio_features.permute(0, 2, 1)   # (B, D_audio, T)
        X1_proj = self.vit_branch.proj_video(X1)  # (B, D_PROJ, T)
        X2_proj = self.vit_branch.proj_audio(X2)  # (B, D_PROJ, T)
        # Temporal Conv1D Refine（局部时序整形）
        X1_proj = self.vit_branch.refine_video(X1_proj)  # (B, D_PROJ, T)
        X2_proj = self.vit_branch.refine_audio(X2_proj)  # (B, D_PROJ, T)
        x_vid = X1_proj.permute(0, 2, 1)  # (B, T, D_PROJ)
        x_aud = X2_proj.permute(0, 2, 1)  # (B, T, D_PROJ)
        x_vid = self.vit_branch.ln_video(x_vid)
        x_aud = self.vit_branch.ln_audio(x_aud)
        return X1_proj, X2_proj, x_vid, x_aud
    
    def _build_attn_mask(self, B, num_tokens, actual_lens, device):
        """构造 padding mask（所有策略共享）"""
        T_patches = num_tokens - 1  # 减去 CLS token
        actual_lens_patches = ((actual_lens + self.patch_size - 1) // self.patch_size).clamp(min=1, max=T_patches)
        patch_valid = (torch.arange(T_patches, device=device).expand(B, -1)
                       < actual_lens_patches.unsqueeze(1))
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        full_valid = torch.cat([cls_valid, patch_valid], dim=1)
        attn_mask = torch.zeros(B, 1, 1, num_tokens, device=device)
        attn_mask = attn_mask.masked_fill(~full_valid.unsqueeze(1).unsqueeze(1), -1e9)
        return attn_mask
    
    def _extract_gcn_pooled(self, face_regions, actual_lens, device, B):
        """GCN 特征提取 + masked mean pooling + 投影（所有策略共享）"""
        gcn_temporal = self._extract_gcn_temporal_features(face_regions, actual_lens)
        T_gcn = gcn_temporal.size(1)
        gcn_time_mask = (torch.arange(T_gcn, device=device).expand(B, -1)
                         < actual_lens.unsqueeze(1))
        gcn_mask = gcn_time_mask.unsqueeze(-1).float()
        g_pooled = (gcn_temporal * gcn_mask).sum(dim=1) / gcn_mask.sum(dim=1).clamp(min=1)
        g = self.gcn_proj(g_pooled)  # (B, dim)
        return g
    
    def _fuse_and_classify(self, v, a, g, v_missing=None):
        """门控融合 + 分类（所有策略共享的后处理）
        
        v_missing: (B,) bool — 视觉缺失的样本标记
        
        策略组合（B 主 + A 辅）：
          A（辅）：用 v_missing_embed 替换 video 表征，让门控融合可以学到
                   "视觉缺失" 时的自适应权重分配。
          B（主）：融合后仍强制将 fused 回退为 audio 表征，
                   作为安全兜底，防止早期训练未收敛时视觉噪声泄漏。
        """
        # 【A策略】用可学习缺失 token 替换 video 向量
        if v_missing is not None and v_missing.any():
            v = v.clone()
            v[v_missing] = self.v_missing_embed  # broadcast (dim,) → (n_missing, dim)
        
        fused, gate_weights = self.gated_fusion(v, a, g)
        
        # 【B策略】视觉缺失时，融合表征回退到音频表征（安全兜底）
        if v_missing is not None and v_missing.any():
            fused = fused.clone()
            fused[v_missing] = a[v_missing]
        
        logits = self.classifier(fused)
        return logits
    
    # ---------- ET / IA: 早融合（当前默认 + 注意力门控变体）----------
    
    def _forward_early_fusion(self, video_features, audio_features,
                               face_regions, actual_lens, v_missing=None):
        """
        ET: proj → CrossAttnFusion(128d) → concat → PatchEmbed → Transformer → CLS
        IA: proj → IAFusion(128d, gating) → concat → PatchEmbed → Transformer → CLS
        
        v_missing: (B,) bool — 视觉模态缺失标记
        """
        B = video_features.size(0)
        T = video_features.size(1)
        device = video_features.device
        
        # 1. 特征投影 + LN
        X1_proj, X2_proj, x_vid_in, x_aud_in = self._project_and_normalize(
            video_features, audio_features)
        
        # 2. 构造帧级 key_padding_mask
        vid_kpm = aud_kpm = None
        if actual_lens is not None:
            time_valid = (torch.arange(T, device=device).unsqueeze(0)
                          < actual_lens.unsqueeze(1))  # (B, T) True=有效
            vid_kpm = ~time_valid  # True=padding
            aud_kpm = ~time_valid
        
        # 2.5 v_missing: 视觉缺失样本的 vid_kpm 全置 True
        #     CrossAttention 中 audio→video 分支会看到全 mask → 保护逻辑置 0
        if v_missing is not None and v_missing.any():
            if vid_kpm is None:
                vid_kpm = torch.zeros(B, T, dtype=torch.bool, device=device)
            vid_kpm = vid_kpm.clone()
            vid_kpm[v_missing] = True
        
        # 3. 跨模态融合（ET: cross-attention / IA: gating）
        if self.fusion_strategy == 'IA':
            x_vid_fused, x_aud_fused = self.ia_fusion(
                x_vid_in, x_aud_in, vid_kpm=vid_kpm, aud_kpm=aud_kpm)
        else:  # ET
            x_vid_fused, x_aud_fused = self.vit_branch.fusion(
                x_vid_in, x_aud_in, vid_kpm=vid_kpm, aud_kpm=aud_kpm
            )
        
        # 3. Concat → PatchEmbed → Transformer
        x_vid_fused_t = x_vid_fused.permute(0, 2, 1)  # (B, D_PROJ, T)
        x_aud_fused_t = x_aud_fused.permute(0, 2, 1)
        X_fused = torch.cat([x_vid_fused_t, x_aud_fused_t], dim=1)  # (B, 256, T)
        
        vit_embedded = self.vit_branch.embedding(X_fused)
        attn_mask = self._build_attn_mask(B, vit_embedded.size(1), actual_lens, device)
        
        vit_transformed = vit_embedded
        for layer in self.vit_branch.transformers:
            vit_transformed = layer(vit_transformed, mask=attn_mask)
        
        v = vit_transformed[:, 0, :]  # (B, dim) — CLS token
        
        # 4. Audio pooled: 使用融合后音频做 masked mean pooling
        time_mask = (torch.arange(T, device=device).expand(B, -1)
                     < actual_lens.unsqueeze(1)).unsqueeze(-1).float()
        a_pooled = (x_aud_fused * time_mask).sum(dim=1) / time_mask.sum(dim=1).clamp(min=1)
        a = self.audio_proj(a_pooled)  # (B, dim)
        
        # 5. GCN → GatedFusion → Classifier
        g = self._extract_gcn_pooled(face_regions, actual_lens, device, B)
        return self._fuse_and_classify(v, a, g, v_missing=v_missing)
    
    # ---------- LT: 晚融合 ----------
    
    @staticmethod
    def _nan_check(tensor, name):
        """NaN/Inf 诊断工具（训练调试用，找到 NaN 源头后可删除）"""
        if torch.isnan(tensor).any():
            cnt = torch.isnan(tensor).sum().item()
            print(f"⚠️ NaN detected in [{name}], shape={list(tensor.shape)}, count={cnt}")
            return True
        if torch.isinf(tensor).any():
            cnt = torch.isinf(tensor).sum().item()
            print(f"⚠️ Inf detected in [{name}], shape={list(tensor.shape)}, count={cnt}")
            return True
        return False

    def _forward_lt(self, video_features, audio_features,
                    face_regions, actual_lens, v_missing=None):
        """
        LT: proj → [embed_v → transformers_v] + [embed_a → transformers_a]
            → late cross-attn → CLS_v = v, CLS_a = a
            → GatedFusion(v, a, g) → classifier
        """
        B = video_features.size(0)
        device = video_features.device
        
        # == 输入 NaN 检查 ==
        self._nan_check(video_features, 'LT:input_video')
        self._nan_check(audio_features, 'LT:input_audio')
        
        # 1. 特征投影（不做 cross-attn，保持独立）
        X1_proj, X2_proj, _, _ = self._project_and_normalize(
            video_features, audio_features)
        
        # 2. 各模态独立 PatchEmbed（输入 channel=D_PROJ=128）
        self._nan_check(X1_proj, 'LT:X1_proj')  # NaN诊断
        self._nan_check(X2_proj, 'LT:X2_proj')
        vid_tokens = self.embedding_video(X1_proj)   # (B, T'+1, dim)
        aud_tokens = self.embedding_audio(X2_proj)   # (B, T'+1, dim)
        self._nan_check(vid_tokens, 'LT:vid_tokens_after_embed')  # NaN诊断
        self._nan_check(aud_tokens, 'LT:aud_tokens_after_embed')
        
        attn_mask = self._build_attn_mask(B, vid_tokens.size(1), actual_lens, device)
        
        # 3. 各模态独立 Transformer（全部 depth 层）
        for i_layer, layer in enumerate(self.transformers_video):
            vid_tokens = layer(vid_tokens, mask=attn_mask)
            if self._nan_check(vid_tokens, f'LT:vid_tokens_after_transformer_{i_layer}'):
                break  # 找到第一个 NaN 层即停止
        for i_layer, layer in enumerate(self.transformers_audio):
            aud_tokens = layer(aud_tokens, mask=attn_mask)
            if self._nan_check(aud_tokens, f'LT:aud_tokens_after_transformer_{i_layer}'):
                break
        
        # 4. 构造 token 级 key_padding_mask（CLS 始终有效 + patch 级 padding）
        T_p = vid_tokens.size(1) - 1  # 减去 CLS
        actual_lens_p = ((actual_lens + self.patch_size - 1) // self.patch_size).clamp(min=1, max=T_p)
        patch_valid = (torch.arange(T_p, device=device).expand(B, -1)
                       < actual_lens_p.unsqueeze(1))
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        token_kpm = ~torch.cat([cls_valid, patch_valid], dim=1)  # (B, T'+1) True=padding
        
        # v_missing: 视觉缺失样本的 video token 全 mask
        vid_token_kpm = token_kpm
        if v_missing is not None and v_missing.any():
            vid_token_kpm = token_kpm.clone()
            vid_token_kpm[v_missing] = True  # 所有 video token（含 CLS）标记为 padding
        
        # 晚期 Cross-Attention（高层 token 级交互）
        vid_enhanced, aud_enhanced = self.late_fusion(
            vid_tokens, aud_tokens, vid_kpm=vid_token_kpm, aud_kpm=token_kpm
        )
        self._nan_check(vid_enhanced, 'LT:vid_enhanced_after_late_fusion')  # NaN诊断
        self._nan_check(aud_enhanced, 'LT:aud_enhanced_after_late_fusion')
        
        # 5. 提取 pooled 向量
        v = vid_enhanced[:, 0, :]    # (B, dim) — video CLS
        a = self.audio_proj(aud_enhanced[:, 0, :])  # (B, dim) — audio CLS + LN+Dropout
        
        # 6. GCN → GatedFusion → Classifier
        g = self._extract_gcn_pooled(face_regions, actual_lens, device, B)
        self._nan_check(v, 'LT:v_pooled')  # NaN诊断
        self._nan_check(a, 'LT:a_pooled')
        self._nan_check(g, 'LT:g_pooled')
        return self._fuse_and_classify(v, a, g, v_missing=v_missing)
    
    # ---------- IT: 中融合 ----------
    
    def _forward_it(self, video_features, audio_features,
                    face_regions, actual_lens, v_missing=None):
        """
        IT: proj → [embed_v, embed_a]
            → transformers_pre(独立 k 层) → cross-attn fusion
            → merge(add+LN) → transformers_post(共享 depth-k 层)
            → CLS = v, aud_CLS(pre-merge) = a
            → GatedFusion(v, a, g) → classifier
        """
        B = video_features.size(0)
        device = video_features.device
        
        # 1. 特征投影（不做 cross-attn，保持独立）
        X1_proj, X2_proj, _, _ = self._project_and_normalize(
            video_features, audio_features)
        
        # 2. 各模态独立 PatchEmbed
        vid_tokens = self.embedding_video(X1_proj)
        aud_tokens = self.embedding_audio(X2_proj)
        
        attn_mask = self._build_attn_mask(B, vid_tokens.size(1), actual_lens, device)
        
        # 3. Pre-fusion: 前 k 层独立编码
        for layer in self.transformers_pre_video:
            vid_tokens = layer(vid_tokens, mask=attn_mask)
        for layer in self.transformers_pre_audio:
            aud_tokens = layer(aud_tokens, mask=attn_mask)
        
        # 4. 构造 token 级 key_padding_mask（CLS 始终有效 + patch 级 padding）
        T_p = vid_tokens.size(1) - 1
        actual_lens_p = ((actual_lens + self.patch_size - 1) // self.patch_size).clamp(min=1, max=T_p)
        patch_valid = (torch.arange(T_p, device=device).expand(B, -1)
                       < actual_lens_p.unsqueeze(1))
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        token_kpm = ~torch.cat([cls_valid, patch_valid], dim=1)  # (B, T'+1) True=padding
        
        # v_missing: 视觉缺失样本的 video token 全 mask
        vid_token_kpm = token_kpm
        if v_missing is not None and v_missing.any():
            vid_token_kpm = token_kpm.clone()
            vid_token_kpm[v_missing] = True  # 所有 video token（含 CLS）标记为 padding
        
        # 中间 Cross-Attention 交互
        vid_enhanced, aud_enhanced = self.it_fusion(
            vid_tokens, aud_tokens, vid_kpm=vid_token_kpm, aud_kpm=token_kpm
        )
        
        # 保存 audio CLS（merge 前），作为 GatedFusion 的 audio pooled
        a_cls = aud_enhanced[:, 0, :]  # (B, dim)
        
        # 5. 合并两路: element-wise 平均 + LayerNorm
        merged = self.it_merge_norm((vid_enhanced + aud_enhanced) / 2.0)
        
        # 6. Post-fusion: 共享 transformer 精炼
        for layer in self.transformers_post:
            merged = layer(merged, mask=attn_mask)
        
        # 7. 提取 pooled 向量
        v = merged[:, 0, :]  # (B, dim) — 融合后 CLS
        a = self.audio_proj(a_cls)  # (B, dim) — audio CLS + LN+Dropout
        
        # 8. GCN → GatedFusion → Classifier
        g = self._extract_gcn_pooled(face_regions, actual_lens, device, B)
        return self._fuse_and_classify(v, a, g, v_missing=v_missing)
    
    def _extract_gcn_temporal_features(self, face_regions, actual_lens):
        """
        从 GCN 中提取时序特征（在 attention pooling 之前）
        
        Returns:
            temporal_features: (B, T, gcn_feat_dim)
        """
        # 这里需要修改 GCN 的 forward 方法以返回中间特征
        # 临时方案：直接复制 GCN 的前向逻辑到这里
        
        device = next(self.gcn_branch.parameters()).device
        face_regions = {k: v.to(device) for k, v in face_regions.items()}
        actual_lens = actual_lens.to(device)
        
        B, T = face_regions['mouth'].shape[0], face_regions['mouth'].shape[1]
        
        # 1. 各区域 GCN 提取
        ljaw = self.gcn_branch._region_forward(
            face_regions['ljaw'], 
            self.gcn_branch.gcn_ljaw, 
            self.gcn_branch.gcn_ljaw_2, 
            getattr(self.gcn_branch, 'edge_index_ljaw'), 
            'ljaw'
        )
        rjaw = self.gcn_branch._region_forward(
            face_regions['rjaw'], 
            self.gcn_branch.gcn_rjaw, 
            self.gcn_branch.gcn_rjaw_2, 
            getattr(self.gcn_branch, 'edge_index_rjaw'), 
            'rjaw'
        )
        leye = self.gcn_branch._region_forward(
            face_regions['leye'], 
            self.gcn_branch.gcn_leye, 
            self.gcn_branch.gcn_leye_2, 
            getattr(self.gcn_branch, 'edge_index_leye'), 
            'leye'
        )
        reye = self.gcn_branch._region_forward(
            face_regions['reye'], 
            self.gcn_branch.gcn_reye, 
            self.gcn_branch.gcn_reye_2, 
            getattr(self.gcn_branch, 'edge_index_reye'), 
            'reye'
        )
        nose = self.gcn_branch._region_forward(
            face_regions['nose'], 
            self.gcn_branch.gcn_nose, 
            self.gcn_branch.gcn_nose_2, 
            getattr(self.gcn_branch, 'edge_index_nose'), 
            'nose'
        )
        mouth = self.gcn_branch._region_forward(
            face_regions['mouth'], 
            self.gcn_branch.gcn_mouth, 
            self.gcn_branch.gcn_mouth_2, 
            getattr(self.gcn_branch, 'edge_index_mouth'), 
            'mouth'
        )
        
        # 2. 跨区域注意力
        regions = [ljaw, rjaw, leye, reye, nose, mouth]
        all_regions = torch.stack(regions, dim=2)  # (B, T, 6, D)
        
        cross_attention_results = []
        for i, query_region in enumerate(regions):
            kv = all_regions.clone()
            kv[:, :, i, :] = 0
            
            q = query_region.view(B * T, 1, -1)
            kv_flat = kv.view(B * T, 6, -1)
            
            attn_out, _ = self.gcn_branch.cross_attention(q, kv_flat, kv_flat)
            attn_out = attn_out.view(B, T, -1)
            
            enhanced_region = self.gcn_branch.cross_attn_norm(query_region + attn_out)
            cross_attention_results.append(enhanced_region)
        
        cross_attention_output = torch.stack(cross_attention_results, dim=2).mean(dim=2)  # (B, T, D)
        
        # 3. 全局支路
        all_pts = torch.cat([face_regions[k] for k in ['ljaw', 'rjaw', 'leye', 'reye', 'nose', 'mouth']], dim=2)
        global_feat = self.gcn_branch.global_branch(all_pts.view(B*T, -1)).view(B, T, -1)
        
        # 4. 特征融合
        per_frame = torch.cat([cross_attention_output, global_feat], dim=-1)  # (B, T, fused_dim=128)
        per_frame = self.gcn_branch.fused_norm(per_frame)
        
        return per_frame  # (B, T, 128)
    
    def _adaptive_pool_temporal(self, features, target_T):
        """
        自适应时序池化：将 (B, T_long, D) 池化到 (B, T_short, D)
        """
        B, T_long, D = features.shape
        
        if T_long == target_T:
            return features
        
        # 使用平均池化
        # 重塑为 (B, D, T_long)
        features_transposed = features.permute(0, 2, 1)
        
        # AdaptiveAvgPool1d
        pool = nn.AdaptiveAvgPool1d(target_T)
        pooled = pool(features_transposed)  # (B, D, target_T)
        
        # 转回 (B, target_T, D)
        return pooled.permute(0, 2, 1)


if __name__ == '__main__':
    #ViT(spectra_size=1400, patch_size=140, num_classes=40, dim=512, depth=8, heads=16, dim_mlp=1400, channel=1,dim_head=32)
    model = ViT(spectra_size=915,patch_size=15,num_classes=2,dim=256,depth=8,heads=8,dim_mlp=1024,channel=256,dim_head=32,dropout=0.1).cuda()
    print(model)
    x1  = torch.randn(4,915,171).cuda()
    x2 = torch.randn(4,915,128).cuda()
    y = model(x1,x2)

    print(y.shape)