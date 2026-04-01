from collections import OrderedDict
import torch.nn as nn
import torch
import numpy as np
import logging
from torch.autograd import Function
from math import sqrt
import torch
import torch.nn.functional as F

class Conv1d(nn.Module):
    def __init__(self) -> None:
        super(Conv1d, self).__init__()
        # 第一层卷积：输入通道171 → 输出通道32
        self.layer1 = nn.Sequential(
            nn.Conv1d(171, 32, 3, padding=1),  # 171维输入→32维输出，卷积核3，填充1（等长输出）
            nn.BatchNorm1d(32),  # 批归一化：加速训练，稳定数值分布
            nn.ReLU(),  # 激活函数：引入非线性，增强特征表达能力
            nn.MaxPool1d(2),  # 最大池化：窗口大小2，时间维度压缩为1/2
        )
        # 第二层卷积：输入通道32 → 输出通道64
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),  # 32维→64维，等长卷积
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 时间维度再压缩为1/2（累计压缩为1/4）
        )
        # 第三层卷积：输入通道64 → 输出通道128
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),  # 64维→128维，等长卷积
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 时间维度再压缩为1/2（累计压缩为1/8）
        )

    def forward(self, input):
        x = self.layer1(input)  # 经过第一层处理
        x = self.layer2(x)      # 经过第二层处理
        x = self.layer3(x)      # 经过第三层处理
        return x

class AstroModel(nn.Module):
    def __init__(self) -> None:
        super(AstroModel, self).__init__()
        self.conv = nn.Conv1d(128, 256, 1)
        self.dropout = nn.Dropout(0.2)      # 20% 的概率随机丢弃神经元输出，防止模型过拟合

        self.conv1_1 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.conv1_2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)

        self.conv2_1 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)

        self.conv3_1 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)
        self.conv3_2 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)

        self.conv4_1 = nn.Conv1d(128, 256, 3, padding=16, dilation=16)      #用 256 个 1x1 卷积核，将 128 通道的输入线性映射为 256 通道的输出
        self.conv4_2 = nn.Conv1d(256, 256, 3, padding=16, dilation=16)
    
    def forward(self, x):
        raw = x #第一组空洞卷积加残差连接
        x = F.relu(self.conv1_1(x)) #用 dilation=2 的空洞卷积提取特征，再通过 ReLU 引入非线性
        x = self.dropout(x)
        x = self.dropout(self.conv1_2(x))       #第一次 dropout 是对 conv1_1 的输出做正则化，防止过拟合；然后用 conv1_2（和 conv1_1 参数一样）再做一次卷积，进一步强化特征，之后再 dropout 一次 
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv2_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv2_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv3_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv3_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv4_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv4_2(x))
        raw = self.conv(raw)        #先用 self.conv(raw)（1x1 卷积，128→256 通道）把 raw 的通道对齐到 256，再和 x 相加
        raw = F.relu(x + raw)

        return raw
        
class TCNModel(nn.Module):
    def __init__(self) -> None:
        super(TCNModel, self).__init__()
        self.Conv1d = Conv1d()  # 实例化基础卷积特征提取模块
        self.AstroModel = AstroModel()  # 实例化空洞卷积残差增强模块

    def forward(self, input):
        input = input.transpose(1,2)
        x = self.Conv1d(input)
        x = self.AstroModel(x)
        x = x.transpose(1,2)
        return x

class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Qx, Kx, Vx,Qy,Ky,Vy):
        attentionx = torch.matmul(Qx, torch.transpose(Kx, -1, -2))
        attentiony = torch.matmul(Qy, torch.transpose(Ky, -1, -2))
        attention = torch.cat((attentionx,attentiony),dim=1)
        B,C,H,W = attention.size()      #将各维度大小赋值给对应变量
        attention = attention.reshape(B,2,C//2,H,W) #拆分为两组
        attention = torch.mean(attention,dim=1).squeeze()   #两组的分数求平均值
        attention1= torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention1 = torch.matmul(attention1, Vx)
        attention2 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention2 = torch.matmul(attention2, Vy)
        return attention1,attention2

class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.0, f=nn.Conv1d, activation=nn.ELU):
        super(FeedForward, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(
            f(in_channels=dim_in, out_channels=hidden_dim,kernel_size=1,padding=0,stride=1),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(in_channels=hidden_dim, out_channels=dim_out,kernel_size=1,padding=0,stride=1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Multi_CrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        assert all_head_size % head_num == 0
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.norm = sqrt(all_head_size)
    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)
    def forward(self, x,y):
        batch_size = x.size(0)
        q_sx = self.linear_q(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sx = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sx = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        q_sy = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sy = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sy = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)


        attention1,attention2 = CalculateAttention()(q_sx, k_sx, v_sx,q_sy,k_sy,v_sy)
        attention1 = attention1.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)+x
        attention2 = attention2.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)+y

        return attention1,attention2

class ConvNet1d(nn.Module):
    def __init__(self) -> None:
        super(ConvNet1d, self).__init__()
        # TCNModel 的输出 feature dim 为 128，所以这里输入应为 128
        self.fc = nn.Linear(128, 128)
    
    def forward(self,input):
        sizeTmp = input.size(1)
        batch_size = input.size(0)
        outConv1d = input.contiguous().view(input.size(0)*input.size(1),-1)
        output = self.fc(outConv1d)
        output = output.view(batch_size, sizeTmp, -1)

        return output



class gateRegress():
    def __init__(self) -> None:
        pass
    def forward(self, ):
        pass

class Regress2(nn.Module):
    def __init__(self) -> None:
        super(Regress2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2),
            nn.ELU())

            
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.TCNModel = TCNModel() 
        self.Conv1dModel = ConvNet1d()
        self.Regress = Regress2()

        self.softmax = torch.nn.Softmax(dim=1)
        # 移除错误的 conv，统一 feature 维为 128 -> 拼接后 256
        self.mhca = Multi_CrossAttention(hidden_size=128, all_head_size=128, head_num=4)
        # 规范化维度改为 256
        self.norm = nn.LayerNorm(256)
        # FFN 使用 channels-first (Conv1d) 实现，输入通道改为 256
        self.FFN = FeedForward(dim_in=256, hidden_dim=256*2, dim_out=256)
        self.norm2 = nn.LayerNorm(256)
        self.pooling = nn.AdaptiveAvgPool1d(1)
    def forward(self,inputVideo,inputAudio): 

        inputVideo = self.TCNModel(inputVideo)

        outputConv1dVideo = self.Conv1dModel(inputVideo)

        # outputConv1dVideo: (B, T_v, 128)
        T_v = outputConv1dVideo.size(1)

        # 把 audio 池化/对齐到 video 的时间步长 T_v
        # inputAudio: (B, T_a, 128) -> (B,128,T_a)
        audio_c_first = inputAudio.permute(0,2,1)
        audio_pooled = F.adaptive_avg_pool1d(audio_c_first, output_size=T_v)  # (B,128,T_v)
        audio_pooled = audio_pooled.permute(0,2,1)  # (B, T_v, 128)

        # Multi-head cross attention：输入均为 (B, T_v, 128)
        output1, output2 = self.mhca(outputConv1dVideo, audio_pooled)

        # 拼接 feature dim -> (B, T_v, 256)
        outputFeature = torch.cat((output1, output2), dim=2)

        # FFN 要求 channels-first (B,C,T)，所以先转置
        tmp = outputFeature.permute(0,2,1)  # (B, 256, T_v)
        ffn_out = self.FFN(tmp)             # (B, 256, T_v)
        ffn_out = ffn_out.permute(0,2,1)    # (B, T_v, 256)
        outputFeature = self.norm(ffn_out + outputFeature)

        # 全局池化：先转换为 (B, C, T) 再池化
        pooled = self.pooling(outputFeature.permute(0,2,1)).reshape(outputFeature.size(0), -1)  # (B, 256)

        result = self.Regress(pooled)  # (B, 2)
        result = self.softmax(result)
        return result

if __name__ == '__main__':

    # 统一 device：支持 GPU 或 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 把模型和示例张量移动到 device
    model = Net().to(device)
    Conv1dModel = ConvNet1d().to(device)
    x1 = torch.randn(4, 915, 128).to(device)
    x2 = torch.randn(4, 915, 171).to(device)
    y = model(x2, x1)
    print(y.shape)