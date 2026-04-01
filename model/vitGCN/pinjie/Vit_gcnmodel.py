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
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
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
        
        # 残差连接
        output = x_vid + attn_output 
        
        return self.norm_out(output)

class ViT(nn.Module):
    def __init__(self, spectra_size, patch_size, num_classes, dim, depth, heads, dim_mlp, 
                 channel=1, dim_head=16, dropout=0.0, emb_dropout=0.3, sd=0.0, 
                 embedding=None, classifier=None, name='vit', 
                 video_dim=171, audio_dim=128, dataset='LMVD', **block_kwargs):
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
        self.transformers = nn.Sequential(*self.transformers)

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

        #LayerNorm 层，用于在 Cross-Attention 之前对齐特征
        # 注意：LayerNorm 的参数是特征维度，即 128
        self.ln_video = nn.LayerNorm(D_PROJECTION)
        self.ln_audio = nn.LayerNorm(D_PROJECTION)

        # ================== 新增：Cross Attention 模块 ==================
        # 输入维度是 D_PROJECTION (128)，不是 dim (256)
        self.fusion = CrossAttentionFusion(dim=D_PROJECTION, heads=heads, dropout=dropout)
        # ==============================================================
        
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
                    
    def forward(self, X1, X2):
        # X1: (B, 915, 171)
        # X2: (B, 915, 128)

        # 1. 维度转置 (B, T, D) -> (B, D, T) 以适配 Conv1d
        X1 = X1.permute(0, 2, 1) 
        X2 = X2.permute(0, 2, 1) 

        # 2. 特征投影 (B, 128, 915)
        X1_proj = self.proj_video(X1)
        X2_proj = self.proj_audio(X2)

        # 3. 准备 Cross Attention 输入
        # CrossAttention 需要 (Batch, Seq, Dim)，所以要转置回来
        x_vid_in = X1_proj.permute(0, 2, 1) # (B, 915, 128)
        x_aud_in = X2_proj.permute(0, 2, 1) # (B, 915, 128)

        #在进入 Cross-Attention 融合模块前进行归一化
        # 这步能确保视频和音频特征的均值和方差在一个量级，防止某种模态“霸权”
        x_vid_in = self.ln_video(x_vid_in)
        x_aud_in = self.ln_audio(x_aud_in)

        # ================== 修改：使用 Cross Attention 融合 ==================
        # 融合后的视频特征 (包含了音频上下文)
        x_vid_fused = self.fusion(x_vid_in, x_aud_in) # (B, 915, 128)
        
        # 为了适配后面的 PatchEmbedding (它期望 channel=256)
        # 我们将 [融合后的视频] 和 [原始投影音频] 拼接
        # 这样总维度依然是 128 + 128 = 256
        
        # 此时需要转回 (B, Dim, Seq) 才能进行 cat 和传入 PatchEmbedding
        x_vid_fused = x_vid_fused.permute(0, 2, 1) # (B, 128, 915)
        
        # 拼接: (B, 128, 915) + (B, 128, 915) -> (B, 256, 915)
        # 注意：X2_proj 还是原来的 (B, 128, 915)
        X_fused = torch.cat([x_vid_fused, X2_proj], dim=1) 
        # ===================================================================

        # 4. Patch Embedding (输入形状 B, 256, 915)
        X = self.embedding(X_fused)

        # 5. Transformer Block 和分类
        X = self.transformers(X)
        X = self.classifier(X[:, 0])
        
        return X

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
                 # GCN 参数
                 gcn_out_dim=32, gcn_nhead=4,
                 # 通用参数
                 num_classes=2, dropout=0.45, channel=256,
                 # 数据集参数
                 video_dim=171, audio_dim=128, dataset='LMVD'):
        """
        Args:
            video_dim: 视频特征维度 (LMVD: 171, D-Vlog: 136)
            audio_dim: 音频特征维度 (LMVD: 128, D-Vlog: 25)
            dataset: 数据集类型 ('LMVD' 或 'DVLOG')
        """
        super().__init__()
        
        self.dim = dim
        self.dataset = dataset
        
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
            video_dim=video_dim,  # 传入视频维度
            audio_dim=audio_dim,  # 传入音频维度
            dataset=dataset       # 传入数据集类型
        )
        # 移除 ViT 的分类器，只保留特征提取
        self.vit_branch.classifier = nn.Identity()
        
        # ===== 2. GCN 分支（提取面部空间特征）=====
        from _02GCN_Transformer import AnatomicalGCN
        self.gcn_branch = AnatomicalGCN(
            out_dim=gcn_out_dim,
            nhead=gcn_nhead,
            num_classes=num_classes,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        # 修改 GCN 输出：返回时序特征而非分类 logits
        # GCN 输出维度 = fused_dim (区域64 + 全局64 = 128)
        gcn_feat_dim = gcn_out_dim * 4  # 128
        self.gcn_branch.classifier = nn.Identity()
        
        # GCN 特征投影到 ViT 维度
        self.gcn_proj = nn.Sequential(
            nn.Linear(gcn_feat_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ===== 3. 跨注意力融合层 =====
        # Query: ViT时序特征, Key/Value: GCN空间特征
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(dim)
        
        # ===== 4. 注意力池化层 =====
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.attn_proj = nn.Linear(dim, 1)
        
        # ===== 5. 最终分类器 =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
        
    def forward(self, video_features, audio_features, face_regions, actual_lens):
        """
        Args:
            video_features: (B, T, 171)
            audio_features: (B, T, 128)
            face_regions: dict of {region: (B, T, N, C)}
            actual_lens: (B,)
        
        Returns:
            logits: (B, num_classes)
        """
        B = video_features.size(0)
        T = video_features.size(1)
        device = video_features.device
        
        # ===== Step 1: ViT 提取视频时序特征 =====
        # 【关键修复】不直接调用 vit_branch.forward()，而是手动提取中间特征
        # 因为原始 ViT.forward() 会在最后做池化并通过分类器，输出 (B, num_classes)
        
        # 1.1 特征投影和融合（复制 ViT 的预处理逻辑）
        X1 = video_features.permute(0, 2, 1)  # (B, 171, T)
        X2 = audio_features.permute(0, 2, 1)  # (B, 128, T)
        
        X1_proj = self.vit_branch.proj_video(X1)  # (B, 128, T)
        X2_proj = self.vit_branch.proj_audio(X2)  # (B, 128, T)
        
        x_vid_in = X1_proj.permute(0, 2, 1)  # (B, T, 128)
        x_aud_in = X2_proj.permute(0, 2, 1)  # (B, T, 128)
        
        x_vid_in = self.vit_branch.ln_video(x_vid_in)
        x_aud_in = self.vit_branch.ln_audio(x_aud_in)
        
        x_vid_fused = self.vit_branch.fusion(x_vid_in, x_aud_in)  # (B, T, 128)
        x_vid_fused = x_vid_fused.permute(0, 2, 1)  # (B, 128, T)
        
        X_fused = torch.cat([x_vid_fused, X2_proj], dim=1)  # (B, 256, T)
        
        # 1.2 Patch Embedding + Transformer（但不经过分类器）
        vit_embedded = self.vit_branch.embedding(X_fused)  # (B, T_patches+1, dim)
        vit_transformed = self.vit_branch.transformers(vit_embedded)  # (B, T_patches+1, dim)
        
        # 1.3 去掉 CLS token，保留序列特征
        vit_features = vit_transformed[:, 1:, :]  # (B, T_patches, dim)
        # T_patches = T // patch_size = 915 // 15 = 61
        
        # ===== Step 2: GCN 提取面部空间特征 =====
        # 需要修改 GCN.forward 返回时序特征，暂时使用 GCN 的中间输出（temporal_out）
        # 这需要修改 _02GCN_Transformer.py 的 forward 方法
        
        # 临时方案：调用 GCN forward 并获取 temporal_transformer 之前的输出
        gcn_temporal_features = self._extract_gcn_temporal_features(face_regions, actual_lens)  # (B, T, gcn_feat_dim)
        
        # 投影到 ViT 维度
        gcn_features = self.gcn_proj(gcn_temporal_features)  # (B, T, dim)
        
        # ===== Step 3: 跨注意力融合 =====
        # Query: ViT特征, Key/Value: GCN特征
        # 需要对齐时间维度：ViT是 T_patches=61, GCN是 T=915
        # 方案：将 GCN 特征池化到 T_patches
        gcn_features_pooled = self._adaptive_pool_temporal(gcn_features, vit_features.size(1))  # (B, T_patches, dim)
        
        fused_features, _ = self.cross_attention(
            vit_features,           # Query: (B, T_patches, dim)
            gcn_features_pooled,    # Key: (B, T_patches, dim)
            gcn_features_pooled     # Value: (B, T_patches, dim)
        )
        
        # 残差连接 + LayerNorm
        fused_features = self.cross_attn_norm(vit_features + fused_features)  # (B, T_patches, dim)
        
        # ===== Step 4: 注意力池化 =====
        # 计算有效长度（patch后的）
        actual_lens_patches = (actual_lens // 15).clamp(min=1)
        time_mask = (torch.arange(fused_features.size(1), device=device).expand(B, -1) 
                     < actual_lens_patches.unsqueeze(1))
        
        # 全局上下文
        valid_mask = time_mask.unsqueeze(-1).float()
        global_context = (fused_features * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        global_context = global_context.unsqueeze(1)  # (B, 1, dim)
        
        # 自注意力池化
        attn_out, attn_weights = self.attention_pooling(
            global_context, 
            fused_features, 
            fused_features,
            key_padding_mask=(~time_mask)
        )
        
        # 加权池化
        frame_scores = self.attn_proj(fused_features).squeeze(-1)
        frame_scores = frame_scores.masked_fill(~time_mask, -1e9)
        attn_weights_combined = torch.softmax(frame_scores, dim=1).unsqueeze(-1)
        
        video_repr = (fused_features * attn_weights_combined).sum(dim=1)  # (B, dim)
        
        # ===== Step 5: 分类 =====
        logits = self.classifier(video_repr)
        
        return logits
    
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