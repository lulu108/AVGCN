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

#Transformer 架构中的前馈神经网络层
class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.0, f=nn.Linear, activation=nn.GELU):
        super(FeedForward, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(# 构建前馈网络的序列结构
            f(dim_in, hidden_dim), # 第一层线性变换：从输入维度映射到隐藏层维度
            activation(),# 激活函数（默认 GELU），引入非线性
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),# 若 dropout 概率 > 0，则添加 dropout 层；否则用 Identity（无操作）
            f(hidden_dim, dim_out),# 第二层线性变换：从隐藏层维度映射到输出维度
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.net(x)# 前向传播：输入 x 经过 self.net 处理
        return x


class Attention1d(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0):
        super(Attention1d, self).__init__()
        inner_dim = heads * dim_head#计算内部特征维度
        dim_out = dim_in if dim_out is None else dim_out

        self.heads = heads
        self.scale = dim_head ** -0.5#定义缩放因子：1 / sqrt(dim_head)，用于缩放注意力分数

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)#定义一个线性层 to_qkv，用于将输入特征同时映射为QKV；输出维度：inner_dim * 3（Q、K、V 各占 inner_dim 维度）。  

#定义输出处理模块 to_out
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):#前向传播函数，定义注意力计算逻辑
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)#.chunk(3, dim=-1)：沿最后一个维度（特征维度）将张量拆分为 3 份，每份形状为 (b, n, inner_dim)，分别对应 Q、K、V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # (2, 16, 11, 32)

# 计算注意力分数（点积注意力）
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (2, 16, 11, 11)
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)  #对 dots 沿最后一个维度（j 维度）执行 softmax，将分数归一化为概率分布

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (2, 16, 11, 32)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (2, 11, 512)
        out = self.to_out(out)  # (2, 11, 512)

        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dim_mlp=1024, dropout=0.0, sd=0.0,
                 attn=Attention1d, norm=nn.LayerNorm, f=nn.Linear, activation=nn.GELU):
        super(Transformer, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out#确定输出维度（默认与输入维度相同）

# 2. 定义 shortcut（残差连接适配层）——解决 输入特征维度dim_in 和 输出特征维度dim_out 不一致
        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))# 向空列表里加第一个元素：归一化层
            self.shortcut.append(nn.Linear(dim_in, dim_out)) # 向列表末尾加第二个元素：线性层——再映射维度
        self.shortcut = nn.Sequential(*self.shortcut)# 封装为序列

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
        # 「复制同一个 cls_token 到每个样本」
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b) #原张量的维度是(无,1,dim)，把第一个维度复制成b份，其他维度不变
        x = torch.cat((cls_tokens, x), dim=1)# 在「序列维度 (dim=1)」做拼接
        return x


class AbsPosEmbedding(nn.Module):
    def __init__(self, spectra_size, patch_size, dim, stride=None, cls=True):
        super(AbsPosEmbedding, self).__init__()
        if not spectra_size % patch_size == 0:
            raise Exception('Spectra dimensions must be divisible by the patch size!')
        stride = patch_size if stride is None else stride
        output_size = self._conv_output_size(spectra_size, patch_size, stride)
        num_patches = output_size * 1
        # 随机生成一个张量，形状 [1,1,dim]，数值服从「标准正态分布」
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + int(cls), dim) * 0.02)

    def forward(self, x):
        x = x + self.pos_embedding#注入位置信息的方式是「张量的逐元素相加」
        return x

    @staticmethod
    def _conv_output_size(spectra_size, kernel_size, stride, padding=0):
        return int(((spectra_size - kernel_size + (2 * padding)) / stride) + 1)

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)#实例化一个层归一化层，值对后续的查询特征做归一化处理
        self.norm_kv = nn.LayerNorm(dim)#归一化层同时对后续的键特征 (x_aud) 和值特征 (x_aud) 做归一化处理
        
        self.attn = Attention1d(dim_in=dim, dim_out=dim, heads=heads, dropout=dropout)
        
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x_vid, x_aud):
        # x_vid, x_aud: (Batch, Seq, Dim)
        q = self.norm_q(x_vid)
        kv = self.norm_kv(x_aud)
        
        # Attention1d 内部处理了维度变换，直接传入即可
        # 返回 (out, attn_map)
        attn_output, _ = self.attn(q) 
        
        output = x_vid + attn_output 
        return self.norm_out(output)

class ViT(nn.Module):
    def __init__(self, spectra_size, patch_size, num_classes, dim, depth, heads, dim_mlp, video_dim=171, audio_dim=128,channel=1, dim_head=16, dropout=0.0, emb_dropout=0.3, sd=0.0, embedding=None, classifier=None, name='vit', **block_kwargs):
        super(ViT, self).__init__()
        self.name = name#将传入的模型名称赋值给实例属性
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

        D_PROJECTION = dim // 2 #所有模态的特征都会被投影到128 维
        # 视频投影：使用 video_dim 变量
        self.proj_video = nn.Sequential(
            nn.Conv1d(in_channels=video_dim, out_channels=D_PROJECTION, kernel_size=1),
            # nn.BatchNorm1d(D_PROJECTION),
            nn.ReLU()
        )

        # 音频投影：使用 audio_dim 变量
        self.proj_audio = nn.Sequential(
            nn.Conv1d(in_channels=audio_dim, out_channels=D_PROJECTION, kernel_size=1),
            # nn.BatchNorm1d(D_PROJECTION),
            nn.ReLU()
        )
        self.proj_video_norm = nn.LayerNorm(D_PROJECTION) # D_PROJECTION 为 128
        self.proj_audio_norm = nn.LayerNorm(D_PROJECTION)
        
        #融合后的归一化层 (对拼接后的 256 维进行标准化)
        self.norm_fusion = nn.LayerNorm(channel) 

        self.fusion = CrossAttentionFusion(dim=D_PROJECTION, heads=4, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )if classifier is None else classifier
        
    def forward(self, X1, X2):

        # 1. 维度转置 (B, T, D) -> (B, D, T) 以适配 Conv1d
        X1 = X1.permute(0, 2, 1) 
        X2 = X2.permute(0, 2, 1) 

        # 2. 特征投影 (B, 128, 915)
        X1_proj = self.proj_video(X1)
        X2_proj = self.proj_audio(X2)

        #在融合和拼接前进行特征对齐（关键步骤）
        # LayerNorm 期望维度在最后一维，所以需要转置：(B, 128, 915) -> (B, 915, 128)
        X1_proj = X1_proj.permute(0, 2, 1).contiguous() # 【增加 contiguous】
        X1_proj = self.proj_video_norm(X1_proj)
        X1_proj = X1_proj.permute(0, 2, 1).contiguous() # 【增加 contiguous】

        X2_proj = X2_proj.permute(0, 2, 1).contiguous() # 【增加 contiguous】
        X2_proj = self.proj_audio_norm(X2_proj)
        X2_proj = X2_proj.permute(0, 2, 1).contiguous() # 【增加 contiguous】

        # 3. 准备 Cross Attention 输入
        # CrossAttention 需要 (Batch, Seq, Dim)，所以要转置回来
        x_vid_in = X1_proj.permute(0, 2, 1) # (B, 915, 128)
        x_aud_in = X2_proj.permute(0, 2, 1) # (B, 915, 128)

        # ================== 使用 Cross Attention 融合==================
        # 融合后的视频特征 (包含了音频上下文)
        x_vid_fused = self.fusion(x_vid_in, x_aud_in) # (B, 915, 128)
        
        # 此时需要转回 (B, Dim, Seq) 才能进行 cat 和传入 PatchEmbedding
        x_vid_fused = x_vid_fused.permute(0, 2, 1).contiguous() # (B, 128, 915)
        
        # 拼接后的特征: (B, 256, 915)
        X_fused = torch.cat([x_vid_fused, X2_proj], dim=1).contiguous() 
        
        #执行归一化，确保模态公平
        X_fused = X_fused.permute(0, 2, 1) # 转为 (B, T, C)
        X_fused = self.norm_fusion(X_fused)
        X_fused = X_fused.permute(0, 2, 1) # 转回 (B, C, T) 适配 PatchEmbedding
        # 4. Patch Embedding (输入形状 B, 256, 915)
        X = self.embedding(X_fused)

        # 5. Transformer Block 和分类
        X = self.transformers(X)
        X = self.classifier(X[:, 0])
        
        return X

if __name__ == '__main__':
    model = ViT(spectra_size=915,patch_size=15,num_classes=2,dim=256,depth=8,heads=8,dim_mlp=1024,channel=256,dim_head=32,dropout=0.1).cuda()
    print(model)
    x1  = torch.randn(4,915,171).cuda()
    x2 = torch.randn(4,915,128).cuda()
    y = model(x1,x2)

    print(y.shape)

