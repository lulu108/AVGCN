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
    def __init__(self, spectra_size, patch_size, num_classes, dim, depth, heads, dim_mlp, channel=1, dim_head=16, dropout=0.0, emb_dropout=0.3, sd=0.0, embedding=None, classifier=None, name='vit', **block_kwargs):
        super(ViT, self).__init__()
        self.name = name
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
        # 视频特征投影层： (B, 171, T) -> (B, 128, T)
        self.proj_video = nn.Sequential(
            # in_channels=171 (D_VIDEO), out_channels=128
            nn.Conv1d(in_channels=171, out_channels=D_PROJECTION, kernel_size=1, stride=1),
            nn.BatchNorm1d(D_PROJECTION),
            nn.ReLU()
        )

        # 音频特征投影层： (B, 128, T) -> (B, 128, T)
        self.proj_audio = nn.Sequential(
            # in_channels=128 (D_AUDIO), out_channels=128
            nn.Conv1d(in_channels=128, out_channels=D_PROJECTION, kernel_size=1, stride=1),
            nn.BatchNorm1d(D_PROJECTION),
            nn.ReLU()
        )

        #LayerNorm 层，用于在 Cross-Attention 之前对齐特征
        # 注意：LayerNorm 的参数是特征维度，即 128
        self.ln_video = nn.LayerNorm(D_PROJECTION)
        self.ln_audio = nn.LayerNorm(D_PROJECTION)

        # ================== 新增：Cross Attention 模块 ==================
        # 输入维度是 D_PROJECTION (128)，不是 dim (256)
        self.fusion = CrossAttentionFusion(dim=D_PROJECTION, heads=4, dropout=dropout)
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

if __name__ == '__main__':
    #ViT(spectra_size=1400, patch_size=140, num_classes=40, dim=512, depth=8, heads=16, dim_mlp=1400, channel=1,dim_head=32)
    model = ViT(spectra_size=915,patch_size=15,num_classes=2,dim=256,depth=8,heads=8,dim_mlp=1024,channel=256,dim_head=32,dropout=0.1).cuda()
    print(model)
    x1  = torch.randn(4,915,171).cuda()
    x2 = torch.randn(4,915,128).cuda()
    y = model(x1,x2)

    print(y.shape)