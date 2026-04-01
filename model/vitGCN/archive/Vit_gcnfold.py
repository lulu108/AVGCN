import copy
import numpy as np
from re import T
import random
import torch.nn.functional as F
import torch
import logging
from kfoldLoader import MyDataLoader 
from kfoldLoader_multimodal import MultiModalDataLoader, collate_fn_multimodal  # 【融合】导入多模态加载器
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
from math import cos
from tqdm import tqdm
import torch.nn as nn
import time
import os
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_fscore_support,
                             classification_report)
from Vit_gcnmodel import ViT, ViT_GCN_Fusion  # 【融合】导入融合模型（相对导入）
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from numpy.random import beta
import matplotlib.pyplot as plt
import json
import inspect
""" 实现K折交叉验证的全部逻辑，包括
数据集索引划分、数据加载、模型初始化、训练、测试、性能指标记录和结果保存。 """

# ==================== 数据集选择开关 ====================
DATASET_SELECT = "DVLOG"  # 可选: "LMVD" 或 "DVLOG"
# python model/vitGCN/Vit_gcnfold.py | tee lmvd_itBiGate.log
# 根据数据集类型动态配置路径和参数
if DATASET_SELECT == "DVLOG":
    # D-Vlog 数据集配置
    VIDEO_FEATURE_PATH = "data/dvlog-dataset/dvlog-dataset"  # 视频和音频在同一目录
    AUDIO_FEATURE_PATH = "data/dvlog-dataset/dvlog-dataset"
    FACE_FEATURE_PATH = "data/dvlog-dataset/dvlog-dataset"   # 面部关键点路径
    LABEL_PATH = "data/dvlog-dataset/dvlog-dataset/labels.csv"  # 统一的CSV标签文件
    
    T = 870           # D-Vlog 帧数：870 = 15×58 patches，能被 patch_size 整除；
                      # 相比 915 减少 padding 帧，提升有效 token 占比
    D_VIDEO = 136     # D-Vlog视频特征维度
    D_AUDIO = 25      # D-Vlog音频特征维度
    BATCH_SIZE = 16   # D-Vlog序列短,可用大batch
    
else:  # LMVD
    # LMVD 数据集配置
    VIDEO_FEATURE_PATH = "data/LMVD_Feature/tcnfeature" 
    AUDIO_FEATURE_PATH = "data/LMVD_Feature/Audio_feature"
    FACE_FEATURE_PATH = "data/LMVD_Feature/Video_landmarks_npy"
    LABEL_PATH = "label/label"
    
    T = 915           # 序列长度
    D_VIDEO = 171     # LMVD视频特征维度
    D_AUDIO = 128     # LMVD音频特征维度
    BATCH_SIZE = 8    # LMVD序列较长,使用小batch

# 视频特征路径
VIDEO_FEATURE_PATH = VIDEO_FEATURE_PATH
# 音频特征路径
AUDIO_FEATURE_PATH = AUDIO_FEATURE_PATH
# 【融合】面部关键点路径
FACE_FEATURE_PATH = FACE_FEATURE_PATH
# 标签路径
LABEL_PATH = LABEL_PATH

# 【融合】控制是否使用融合模型的开关
USE_FUSION_MODEL = True  # 设为 False 则使用原始 ViT 模型

# ==================== 融合模式开关（数据集感知）====================
# 'late'       : Late Fusion (CLS-token + GCN pool, 门控融合，参数少，最稳)LMVD
# 'it_cross'   : Intermediate Cross Fusion (Q=ViT, K/V=GCN，单向，当前默认)
# 'it_bi_gate' : Intermediate Bidirectional Gate Fusion (双向 + 门控，建议重点尝试)DVLOG
# 'concat'     : Concat Fusion（低参数强约束，对小数据更稳）
# 'afi'        : AFI Fusion（MutualAttentionBlock ×2 + scalar gate，稳健首选）建议 DVLOG
if DATASET_SELECT == "DVLOG":
    FUSION_MODE = 'late'     # 实验6：DVLOG 先用 late 作为对照基线
else:
    FUSION_MODE = 'late'     # LMVD: 保持 late baseline

# ==================== AV Cross-Attn 开关（ViT 内部视频↔音频交互）====================
# DVLOG：低成本开启；LMVD：保留作为 baseline
DVLOG_USE_AV_CROSS = True
LMVD_USE_AV_CROSS = True
USE_AV_CROSS_ATTN = DVLOG_USE_AV_CROSS if DATASET_SELECT == "DVLOG" else LMVD_USE_AV_CROSS
# 低成本 AV cross-attn 头数（DVLOG 降头数，LMVD 保持默认）
DVLOG_AV_CROSS_HEADS = 2
LMVD_AV_CROSS_HEADS = None
AV_CROSS_HEADS = DVLOG_AV_CROSS_HEADS if DATASET_SELECT == "DVLOG" else LMVD_AV_CROSS_HEADS
AV_CROSS_ALPHA_INIT = 0.1

# ==================== Mixup 开关 ====================
# DVLOG: face landmark 空间结构 mixup 后易破坏关键点图结构，先关闭做对照
# LMVD: 多模态时序+GCN 图结构场景下，Mixup 会搞乱 face_regions/actual_lens 等多路输入
# 先关掉跨此 baseline，后续单独做 Mixup 对照实验
DVLOG_USE_MIXUP = False
USE_MIXUP = DVLOG_USE_MIXUP if DATASET_SELECT == "DVLOG" else False

# ==================== 数据增强配置（数据集分支）====================
# LMVD: 特征规整，无需弱增强，关闭 noise/mask 避免引入不必要抖动
# DVLOG: 数据量小，使用 weak 增强（弱高斯噪声 + 3% 音频 mask）
if DATASET_SELECT == "DVLOG":
    # DVLOG 增强开关：'baseline' | 'weak'
    DVLOG_AUG_PRESET = 'weak'
    if DVLOG_AUG_PRESET == 'baseline':
        DVLOG_AUG_NOISE_STD = 0.0
        DVLOG_AUDIO_MASK_RATIO = 0.0
    else:  # 'weak'
        DVLOG_AUG_NOISE_STD = 0.005        # 范围 0.003~0.01
        DVLOG_AUDIO_MASK_RATIO = 0.03      # 范围 0.03~0.05
    # DVLOG 关键帧采样偏置控制：提高 uniform 覆盖
    DVLOG_UNIFORM_RATIO = 0.7
else:  # LMVD
    DVLOG_AUG_PRESET = 'none'              # LMVD 不使用 DVLOG 式 preset
    DVLOG_AUG_NOISE_STD = 0.0              # LMVD: 关闭噪声增强
    DVLOG_AUDIO_MASK_RATIO = 0.0           # LMVD: 关闭音频 mask
    DVLOG_UNIFORM_RATIO = 0.7              # LMVD 保持采样参数一致性

# ==================== Segment-MIL（可选结构升级）====================
# 关闭时完全不影响现有流程；开启后对每个 vlog 抽 K 个 segment 并聚合 logits
# LMVD 使用完整序列 → Transformer → 分类，不需要 MIL 分段聚合
USE_SEGMENT_MIL = True if DATASET_SELECT == "DVLOG" else False
SEGMENT_LEN = 300              # 建议提升到 300/330（需被 PATCH_SIZE 整除）；默认 300
SEGMENT_K_TRAIN = 4
SEGMENT_K_EVAL = 8
SEGMENT_AGG = 'logsumexp'      # 'max' | 'logsumexp'（训练聚合，已被 TRAIN_MIL_AGG 接管）
# Step1: 训练聚合改为与 eval 对称的 valid_ratio 加权均值
# 原理：logsumexp 会放大"碰巧好的单段"，使 train/eval 分布不一致，引入 seed 方差
# weighted_mean：有效帧多的段权重大，padding 多的段自然降权，test 更稳、std 更小
TRAIN_MIL_AGG = 'weighted_mean'  # 'logsumexp' | 'max' | 'weighted_mean'

# ==================== Symptom-aware MIL 加权（先改聚合，不改采样）====================
# 最终段权重：final_w = (1-a) * valid_ratio_w + a * symptom_w
# 仅在 weighted_mean 聚合路径生效；用于 Phase B 第一版稳健消融
USE_SYMPTOM_AWARE_MIL_WEIGHT = True
if DATASET_SELECT == "DVLOG":
    SYMPTOM_WEIGHT_ALPHA = 0.3  # DVLOG 噪声更大，适度降低 symptom 权重
else:
    SYMPTOM_WEIGHT_ALPHA = 0.5  # LMVD 稳定度更高，维持等权

# ==================== 固定长度 segment 采样（优先打掉 length confound）====================
# 训练：DataLoader 随机切一段固定长度（orig_T < L 时 pad）
# 评估：滑窗多段 logits 平均，降低“某一段很空”引起的方差
USE_FIXED_SEGMENT_TRAIN = True if DATASET_SELECT == "DVLOG" else False
# DVLOG 专用滑窗策略；LMVD face_valid≈1.0，无 padding 问题，不需要滑窗
USE_SLIDING_SEGMENT_EVAL = True if DATASET_SELECT == "DVLOG" else False
SEGMENT_EVAL_STRIDE = 128
SEGMENT_EVAL_MIN_VALID_RATIO = 0.4           # 仅保留有效比例>=阈值的窗口（建议 0.4~0.5）
# LMVD face_valid≈1.0，不存在 padding 问题，不需要 valid_ratio 加权
SEGMENT_EVAL_WEIGHT_BY_VALID_RATIO = True if DATASET_SELECT == "DVLOG" else False

# ==================== Two/Three-Clip Training（DVLOG 专用）====================
# Two-clip：center + random/jitter/quality
# Three-clip：center + left/right or center + quality + jitter
USE_TWO_CLIP_TRAIN = False if DATASET_SELECT == "DVLOG" else False
TWO_CLIP_MODE = 'center_jitter'   # 'center_random' | 'center_jitter' | 'center_quality'

USE_THREE_CLIP_TRAIN = True if DATASET_SELECT == "DVLOG" else False
THREE_CLIP_MODE = 'center_lr'     # 'center_lr' | 'center_quality_jitter'
QUALITY_CANDIDATE_COUNT = 5       # 候选窗口数量（质量挑选用）

# ==================== 分区方案 ====================
# 控制 face 分支 partition_regions() 的分区策略，同时控制 GCN 图结构
#   "legacy6"  : 原始 6 区（双眉+眼混合、鼻整段）用于基线对照实验
#   "symptom7" : 症状导向 7 区（纯眉+眼、鼻梁桥接、下鼻、嘴闭环）
REGION_PARTITION_SCHEME = "legacy6"   # "legacy6" | "symptom7"
REGION_FUSION_MODE = 'cross_attn'      # 'cross_attn' | 'concat_linear' | 'mlp'
GCN_TEMPORAL_MODE = 'transformer'      # 'transformer' | 'meanmax' | 'tcn'
USE_GLOBAL_BRANCH = True
REGION_MLP_DROPOUT = 0.1
TCN_KERNEL_SIZE = 3
USE_QUALITY_GATE = True
MODALITY_DROPOUT_P_V = 0.20     # 训练时随机屏蔽视频模态概率
MODALITY_DROPOUT_P_G = 0.10     # 训练时随机屏蔽人脸/GCN 模态概率

# ==================== A4: 人脸关键点有效度阈值（数据集感知）====================
# valid_ratio < 阈值 的帧被 GCN 时序建模视为无效帧
# DVLOG: validity 分布有差异，降低阈值保留更多帧可能减少方差
# LMVD:  validity ≈ 1.0 全有效，阈值对实际训练无影响
if DATASET_SELECT == "DVLOG":
    FACE_VALID_THRESH = 0.1  # DVLOG: 宽松阈值，减少 padding 造成的帧丢失
else:
    FACE_VALID_THRESH = 0.1  # LMVD: 同步阈值，便于对照实验

# ==================== 强缺失样本推理温和化 ====================
# 当 (q_v<0.5 且 q_g<0.05) 时，将 logits 置 0（两类 0.5/0.5），避免高置信随机预测
HARD_MISSING_NEUTRALIZE = True

# ==================== 重构目标是否 detach ====================
# True: 仅训练重构头拟合当前表征，不反向改动 backbone 表征（更稳）
RECON_TARGET_DETACH = True

# ==================== 辅助 GCN 损失权重（Phase 2-1）====================
# 实际 λ_gcn 由 get_lambda_gcn() 动态调度（数据集感知）：
#   DVLOG : 前30% 0.20 → 后70% 0.10  （gcn_acc 低，轻量辅助）
#   LMVD  : 前30% 0.40 → 中40% 0.20 → 后30% 0.10  （数据量大，强监督GCN）
AUX_GCN_LAMBDA = 0.3     # 参考起始值（训练中自动调度）

# ==================== 诊断开关 A：临时将 λ_gcn 强制置 0 ====================
# 用途：单独跳过 aux_gcn loss，验证它对 DVLOG 是帮护还是拖累
#   True  → 全程 λ_gcn=0，只训练 fusion loss（诊断用）
#   False → 正常调度（默认）
DVLOG_AUX_ZERO = False  # DVLOG 默认不强制置 0：保留 aux 正则

# ==================== 诊断开关 B：DVLOG GCN 辅助损失强度 ====================
# 用途：判断 DVLOG GCN 分支是否「学不会」还是「没让它认真学」
#   True  → 强监督（前30%: 0.15, 后70%: 0.05）— 诊断 GCN 学习能力
#   False → 弱正则（前30%: 0.05, 后70%: 0.02）— 默认保守策略
# [DIAG] 第二步优先诊断：启用强监督，判断 GCN 是否能学到有效特征
DVLOG_STRONG_AUX_GCN = True  # 诊断模式：强监督 GCN，观察 gcn_acc 和 fusion F1 变化

# ==================== 轻量重构损失（late fusion 辅助）====================
# L = L_cls + λ * (MSE(recon_v, vit_repr) + MSE(recon_g, gcn_repr))
# LMVD 数据量大且稳定，recon 不是主要增益来源，保留小値正则即可
# 若后期发现 recon_v_mse 飘可再将 LMVD 分支置 0
if DATASET_SELECT == "DVLOG":
    RECON_LAMBDA = 0.02
else:
    RECON_LAMBDA = 0.02

# ==================== Step2: 损失函数校准对照 ====================
# 诊断目的：判断 focal + pos_weight 是否在当前 pipeline 制造不稳定性
# 对照方案（建议各跑 3 seeds，比较 mean/std/thr 稳定性）:
#   'focal'       : Focal BCE + pos_weight (sqrt-inverse)  ← 当前默认
#   'focal_no_pw' : Focal BCE + pos_weight=1.0             ← 去掉类别权重（诊断类别偏置影响）
#   'smooth_ce'   : Label Smoothing CE (0.05) + class_w    ← CE 基线（thr 是否更稳）
LOSS_MODE = 'focal_no_pw'    # 实验7：Label Smoothing CE，对照 focal_no_pw 的 thr 稳定性

# ==================== Step3: AFI 专用超参（稳健训练）====================
# AFI scalar gate 初始=0 (tanh(0)=0)，开门过快易致 recon_v_mse 爆炸
# 建议：lr 下探一档（DVLOG 1e-4→5e-5）+ recon_lambda 小幅降低（0.05→0.02）
# 其他 fusion mode（it_bi_gate/late/it_cross）使用数据集默认 lr 和全局 RECON_LAMBDA
AFI_LR_OVERRIDE  = 5e-5   # None = 不覆盖，使用数据集默认 lr；FUSION_MODE='afi' 时生效
AFI_RECON_LAMBDA = 0.02   # AFI 专用 recon_lambda；其他 mode 用全局 RECON_LAMBDA=0.05

# ----------------- 模型的超参数设置 -----------------
# T, D_VIDEO, D_AUDIO 已在上面根据数据集动态设置
if DATASET_SELECT == "DVLOG":
    D_EMB = 192        # DVLOG：缩水 ViT
    HEADS = 4
    DEPTH = 4
    DROPOUT = 0.10     # Step3: DVLOG 进一步压低 dropout，减少随机性
    SD_RATE = 0.0      # Step3: 关闭 stochastic depth，消除震荡噪声源
else:
    D_EMB = 256        # LMVD：保持原样
    HEADS = 8
    DEPTH = 8
    DROPOUT = 0.25     # LMVD 推荐 0.2~0.3；原 0.45 过大易欠拟合
    SD_RATE = 0.1      # LMVD 保留 stochastic depth

PATCH_SIZE = 15    # Patch 大小
DIM_MLP = 1024     # FFN 的隐藏层维度 (dim_mlp)

if DATASET_SELECT == "DVLOG":
    lr = 1e-4               # DVLOG 缩水配置：更高 LR
    epochSize = 180         # DVLOG 更短训练
    warmupEpoch = 10        # DVLOG 更短 warmup
    schedule = 'cosine'     # DVLOG 更稳定的 cosine
    # Step4: DVLOG 早停改为短 patience + Macro-F1 EMA 监控
    EARLY_STOP_PATIENCE = 20
    EARLY_STOP_MONITOR  = 'macro_f1'  # 'loss' | 'macro_f1'
    EMA_ALPHA = 0.3          # 指标 EMA 平滑系数（越大越跟当前值，越小越平滑）
    # Step5: 模型权重 EMA（SWA-lite）
    USE_MODEL_EMA   = True
    MODEL_EMA_DECAY = 0.995  # 越接近 1 越平滑；DVLOG 建议 0.99~0.999
else:
    lr = 7e-5               # LMVD 保持原配置
    epochSize = 300
    warmupEpoch = 20
    schedule = 'cyclic'
    EARLY_STOP_PATIENCE = 50
    # LMVD 改为 macro_f1 监控：loss 最低不等于分类边界最优，F1 是实际目标
    EARLY_STOP_MONITOR  = 'macro_f1'  # 'loss' | 'macro_f1'
    EMA_ALPHA = 0.3
    USE_MODEL_EMA   = False
    MODEL_EMA_DECAY = 0.999
testRows = 1
classes = ['Normal','Depression']
ps = []
rs = []
f1s = []
totals = []

total_pre = []
total_label = []

# ----------------- 日志和保存路径的修复 -----------------
tim = time.strftime('%m_%d__%H_%M', time.localtime())
# 路径格式: {数据集}_{融合模式}_{时间}
# 同时跑多个实验（不同数据集或不同融合模式）时不会互相覆盖
filepath  = os.path.join('logs',   f"{DATASET_SELECT}_{FUSION_MODE}_{tim}")
savePath1 = os.path.join('models', f"{DATASET_SELECT}_{FUSION_MODE}_{tim}")
if not os.path.exists(filepath):
        os.makedirs(filepath)
if not os.path.exists(savePath1): # 确保模型保存路径也创建
        os.makedirs(savePath1)
logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    # 修复日志文件路径
                    filename=os.path.join(filepath, 'training.log'),
                    filemode='w')

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 确定性设置 & 种子函数 ====================
USE_STRICT_DETERMINISTIC_ALGOS = True

def set_all_seeds(seed):
    """统一设置所有随机种子，确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 启用确定性模式（可能会稍慢，但保证复现性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if USE_STRICT_DETERMINISTIC_ALGOS:
        # CUDA matmul 确定性建议配置（部分算子需要）
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    print(
        f"[set_all_seeds] seed={seed}, cudnn.deterministic=True, "
        f"strict_deterministic_algos={USE_STRICT_DETERMINISTIC_ALGOS}"
    )

def plot_confusion_matrix(y_true, y_pred, labels_name, savename,title=None, thresh=0.6, axis_labels=None):

    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()


    if title is not None:
        plt.title(title)

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = classes
    plt.xticks(num_local, ['Normal','Depression'])
    plt.yticks(num_local, ['Normal','Depression'],rotation=90,va='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if cm[i][j] * 100 > 0:
                plt.text(j, i, format(cm[i][j] * 100 , '0.2f') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")

    plt.savefig(savename, format='png')
    plt.clf()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Weighted Label Smoothing Cross-Entropy（Phase 3）。

    同时解决两个问题：
      1. 过度自信（label smoothing）
      2. 类别频率偏差（class weight）

    Args:
        smoothing : float，平滑幺度，建议 0.05
        weight    : Tensor (C,) 或 None。
                    若为 None 则退化为普通 label-smoothing CE；
                    若传入应将 sum≈1的归一化权重，且应已在调用方自行 move_to_device。
    """
    def __init__(self, smoothing=0.05, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        # weight 登记为 buffer（to(device) 时自动跟随设备移动）
        if weight is not None:
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

    def forward(self, x, target, sample_weight=None):
        """
        x             : (B, C) logits
        target        : (B,)   long
        sample_weight : (B,)   可选样本权重（与 FocalBCELoss 接口对齐，Step2 smooth_ce 对照用）
        """
        confidence = 1.0 - self.smoothing
        logprobs   = F.log_softmax(x, dim=-1)          # (B, C)

        # 真实标签的 NLL loss
        nll_loss   = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)  # (B,)
        # 均匀分布的 smooth loss
        smooth_loss = -logprobs.mean(dim=-1)                                          # (B,)

        loss = confidence * nll_loss + self.smoothing * smooth_loss                  # (B,)

        # 类别加权：少数类违分被拨重一点
        if self.weight is not None:
            class_w = self.weight[target]   # (B,)
            loss = loss * class_w

        # 样本权重（Step2: smooth_ce 对照实验时也支持 sample_weight）
        if sample_weight is not None:
            sw = sample_weight.float()
            denom = sw.sum().clamp(min=1e-6)
            return (loss * sw).sum() / denom

        return loss.mean()


class FocalBCELoss(nn.Module):
    """
    Focal BCE Loss for 2-class logit output (B, 2).

    内部将 2-logit 转为单二元 logit: logit_bin = logits[:,1] - logits[:,0]
    再施加 focal reweighting + pos_weight，对难例和少数类同时聚焦。

    相比 LabelSmoothingCE 的优势：
      - 自动降低易分样本权重，梯度集中在"偏一类"的难例上
      - pos_weight 直接控制正类（Depression）的损失倍率

    Args:
        gamma     : focal 调制因子（标准值 2.0）
        pos_weight: 正类权重标量 Tensor；None = 不加权
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        if pos_weight is not None:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.tensor(float(pos_weight))
            self.register_buffer('pos_weight', pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits, target, sample_weight=None):
        """
        logits: (B, 2) — 2-class logits
        target: (B,)   — 0/1 integer labels
        """
        # 二分类单 logit：Depression 相对 Normal 的优势得分
        logit_bin = logits[:, 1] - logits[:, 0]   # (B,)
        target_f  = target.float()

        # 基础 BCE（reduction='none' 供 focal 加权）
        bce_raw = F.binary_cross_entropy_with_logits(
            logit_bin, target_f,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        # Focal 权重: (1 - p_t)^gamma
        with torch.no_grad():
            p       = torch.sigmoid(logit_bin)
            p_t     = p * target_f + (1.0 - p) * (1.0 - target_f)
            focal_w = (1.0 - p_t).pow(self.gamma)

        loss_vec = focal_w * bce_raw

        if sample_weight is None:
            return loss_vec.mean()

        sw = sample_weight.float()
        denom = sw.sum().clamp(min=1e-6)
        return (loss_vec * sw).sum() / denom


def compute_class_weights(trainSet, num_classes=2, sqrt_inverse=True):
    """
    从训练集的 _label_cache 中统计类别分布，返回平方根反比类别权重。

    公式（sqrt-inverse）：
        w_c = sqrt( N / (K * n_c) )
    再導款sum=1 为加权形式。

    Args:
        trainSet      : MultiModalDataLoader 实例，内含 _label_cache
        num_classes   : 类别数，默认 2
        sqrt_inverse  : 是否用平方根反比，默认 True（避免权重过大）

    Returns:
        Tensor (num_classes,)，已归一化到 sum=1
    """
    file_list = trainSet.file_list
    cache     = trainSet._label_cache

    counts = torch.zeros(num_classes)
    for fn in file_list:
        file_root = os.path.splitext(fn)[0]
        lbl = cache.get(file_root)
        if lbl is not None:
            counts[int(lbl)] += 1

    n_total = counts.sum().clamp(min=1)
    if sqrt_inverse:
        # w_c = sqrt(N / (K * n_c))
        raw_w = torch.sqrt(n_total / (num_classes * counts.clamp(min=1)))
    else:
        # 纯反比
        raw_w = n_total / (num_classes * counts.clamp(min=1))

    # 归一化：mean=1（让平均权重=1，保持梯度尺度，避免学习率隐式缩水）
    raw_w = raw_w / raw_w.mean()
    print(f"[class_weight] counts={counts.tolist()}  weights={raw_w.tolist()}")
    return raw_w

class AffectnetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset):
        print('initial balance sampler ...')

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        expression_count = [0] * 63
        for idx in self.indices:
            label = dataset.label[idx]
            expression_count[int(label)] += 1

        self.weights = torch.zeros(self.num_samples)
        for idx in self.indices:
            label = dataset.label[idx]
            self.weights[idx] = 1. / expression_count[int(label)]

        print('initial balance sampler OK...')


    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))


    def __len__(self):
        return self.num_samples


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (cos(min((current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps),1) * math.pi) + 1)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lambda_gcn(epoch, total_epochs):
    """
    动态 λ_gcn 调度（Phase 2-1）：前期大权重强监督 GCN，后期让融合头主导。

    DVLOG（小数据集，gcn_acc 偏低）：
      【强监督模式】前 30%: 0.15 → 后 70%: 0.05  — 诊断 GCN 是否能学到有效特征
      【弱正则模式】前 30%: 0.05 → 后 70%: 0.02  — 保守策略，避免拉偏融合头

    LMVD（大数据集，aux 头更稳定）：
      前 30% epoch : 0.40  — GCN 分支先学好判别特征
      中间 40% epoch : 0.20  — GCN 与融合头协同优化
      后 30% epoch : 0.10  — 融合头主导，GCN 仅提供辅助正则

    Args:
        epoch        : 当前 epoch 编号（1-based）
        total_epochs : epochSize
    Returns:
        float: 当前 epoch 的 λ_gcn
    """
    # 诊断开关 A：强制关闭 aux loss（判断它是否拖累）
    if DVLOG_AUX_ZERO and DATASET_SELECT == 'DVLOG':
        return 0.0

    ratio = epoch / max(total_epochs, 1)
    if DATASET_SELECT == 'DVLOG':
        # 诊断开关 B：强监督 vs 弱正则（判断 GCN 学习能力）
        if DVLOG_STRONG_AUX_GCN:
            # 强监督：让 GCN 认真学，判断它能否学会
            return 0.15 if ratio < 0.30 else 0.05
        else:
            # 弱正则（默认）：保守策略，避免弱 aux 头拉偏
            return 0.05 if ratio < 0.30 else 0.02
    else:
        # LMVD：三段式，前期强监督 GCN
        if ratio < 0.30:
            return 0.40
        elif ratio < 0.70:
            return 0.20
        else:
            return 0.10


# ===========================================================================
# Step5: 轻量模型权重 EMA（指数移动平均，SWA-lite）
# 原理：ema_params = decay * ema_params + (1-decay) * model_params
# 效果：平均掉权重漂移，落到更平坦的 basin，降低 seed 敏感性
# ===========================================================================
class ModelEMA:
    """对 model.parameters() 做指数移动平均。
    用法：
        ema = ModelEMA(model, decay=0.995)
        # 每个 optimizer.step() 后调用：
        ema.update(model)
        # 评估时切换到 EMA 权重：
        with ema.average_parameters():
            output = model(x)
    """
    def __init__(self, model, decay=0.995):
        self.decay = decay
        # 存储 EMA 参数（detach，不参与梯度计算）
        self.shadow = {name: param.data.clone().detach()
                       for name, param in model.named_parameters()
                       if param.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        """将 EMA 参数写入 model（用于保存/评估）。"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model, backup):
        """还原 model 参数（评估完毕后恢复训练参数）。"""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    from contextlib import contextmanager
    @contextmanager
    def average_parameters(self, model):
        """上下文管理器：进入时切换为 EMA 权重，退出时自动还原。"""
        backup = {name: param.data.clone()
                  for name, param in model.named_parameters()
                  if param.requires_grad}
        self.apply_shadow(model)
        try:
            yield
        finally:
            self.restore(model, backup)


def compute_segment_symptom_score(face_regions_seg, actual_lens_seg, region_scheme):
    """
    显式规则分数（可解释、稳定）：
      score = 0.40*mouth + 0.30*eye + 0.20*head + 0.10*global
    Args:
        face_regions_seg: dict of region -> (B, L, N, C)
        actual_lens_seg: (B,) tensor, valid length per segment
        region_scheme: 'legacy6' or 'symptom7'
    Returns:
        (B,) tensor
    """
    any_region = next(iter(face_regions_seg.values()))
    B_loc, L_loc = any_region.shape[0], any_region.shape[1]
    if L_loc <= 1:
        return torch.ones(B_loc, device=any_region.device, dtype=any_region.dtype)

    device = any_region.device
    dtype = any_region.dtype
    t_idx = torch.arange(L_loc, device=device).unsqueeze(0)
    mask = (t_idx < actual_lens_seg.unsqueeze(1)).to(dtype)  # (B, L)

    def _masked_mean(x, m, dims):
        denom = m.sum(dim=dims, keepdim=False).clamp(min=1.0)
        return (x * m).sum(dim=dims, keepdim=False) / denom

    def _masked_var(x, m, dims):
        mu = _masked_mean(x, m, dims)
        mu_keep = mu
        for _ in dims:
            mu_keep = mu_keep.unsqueeze(-1)
        denom = m.sum(dim=dims, keepdim=False).clamp(min=1.0)
        return ((x - mu_keep) ** 2 * m).sum(dim=dims, keepdim=False) / denom

    def _speed_stats(x):
        # x: (B, L, N, C)
        d = x[:, 1:, :, :2] - x[:, :-1, :, :2]
        sp = torch.norm(d, dim=-1)  # (B, L-1, N)
        m = mask[:, 1:].unsqueeze(-1)  # (B, L-1, 1)
        mean = _masked_mean(sp, m, dims=(1, 2))
        var = _masked_var(sp, m, dims=(1, 2))
        sp_masked = sp.masked_fill(m == 0, float('-inf'))
        peak = sp_masked.amax(dim=(1, 2))
        peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
        return mean, var, peak

    def _get_region_list(scheme, kind):
        if scheme == 'symptom7':
            if kind == 'global':
                return ['ljaw', 'rjaw', 'leye', 'reye', 'brow_glabella', 'nose_lower', 'mouth']
            if kind == 'head':
                return ['brow_glabella', 'nose_lower']
        # legacy6 fallback
        if kind == 'global':
            return ['ljaw', 'rjaw', 'leye', 'reye', 'nose', 'mouth']
        if kind == 'head':
            return ['nose']
        return []

    # 1) mouth activity
    m_mean, m_var, _ = _speed_stats(face_regions_seg['mouth'])
    mouth_activity = 0.7 * m_mean + 0.3 * m_var

    # 2) eye activity（左右眼平均）
    le_mean, le_var, le_peak = _speed_stats(face_regions_seg['leye'])
    re_mean, re_var, re_peak = _speed_stats(face_regions_seg['reye'])
    left_eye = 0.6 * le_mean + 0.2 * le_var + 0.2 * le_peak
    right_eye = 0.6 * re_mean + 0.2 * re_var + 0.2 * re_peak
    eye_activity = 0.5 * (left_eye + right_eye)

    # 3) head motion proxy（区域中心轨迹）
    head_keys = [k for k in _get_region_list(region_scheme, 'head') if k in face_regions_seg]
    if len(head_keys) == 0:
        head_keys = list(face_regions_seg.keys())
    head_pts = torch.cat([face_regions_seg[k] for k in head_keys], dim=2)[..., :2]
    head_centroid = head_pts.mean(dim=2)  # (B, L, 2)
    d_centroid = head_centroid[:, 1:, :] - head_centroid[:, :-1, :]
    head_sp = torch.norm(d_centroid, dim=-1)  # (B, L-1)
    head_motion = _masked_mean(head_sp, mask[:, 1:], dims=(1,))

    # 4) global face energy
    global_keys = [k for k in _get_region_list(region_scheme, 'global') if k in face_regions_seg]
    if len(global_keys) == 0:
        global_keys = list(face_regions_seg.keys())
    all_pts = torch.cat([face_regions_seg[k] for k in global_keys], dim=2)[..., :2]
    d_all = all_pts[:, 1:, :, :] - all_pts[:, :-1, :, :]
    g_sp = torch.norm(d_all, dim=-1)  # (B, L-1, N)
    g_m = mask[:, 1:].unsqueeze(-1)
    global_energy = _masked_mean(g_sp, g_m, dims=(1, 2))

    score = 0.40 * mouth_activity + 0.30 * eye_activity + 0.20 * head_motion + 0.10 * global_energy
    return score.clamp(min=1e-6)


def train(VideoPath, AudioPath, FacePath, X_train, X_dev, X_final_test, labelPath, fold_name):  # 【融合】新增 FacePath
    mytop = 0
    topacc = 0
    top_p=0
    top_r=0
    top_f1=0
    top_pre=[]
    top_label=[]

    patience = EARLY_STOP_PATIENCE  # Step4: 数据集感知 patience
    # --- EarlyStop state ---
    counter = 0
    if EARLY_STOP_MONITOR == 'macro_f1':
        best_monitor = float('-inf')  # 越大越好
        ema_monitor  = float('-inf')
        ema_loss_val = None
    else:
        best_monitor = float('inf')   # 越小越好
        ema_loss_val = float('inf')
        ema_monitor  = None

    # 选模与阈值解耦：选模固定看 argmax macro-F1；阈值 sweep 只做校准记录
    best_argmax_f1 = float('-inf')
    best_thr_for_best_model = 0.5
    best_epoch_for_best_model = -1

    # debug: early-stop 初始化态自检（防止出现 EMA Macro-F1=-inf 长期不变）
    print(
        f"[ES-INIT] monitor={EARLY_STOP_MONITOR} best_monitor={best_monitor} "
        f"ema_monitor={ema_monitor} ema_loss_val={ema_loss_val}"
    )

    # 运行时来源与签名核验：防止同名旧文件被意外 import
    try:
        _mm_sig = str(inspect.signature(MultiModalDataLoader.__init__))
        _fbce_sig = str(inspect.signature(FocalBCELoss.forward))
        print(f"[IMPORT-CHECK] MultiModalDataLoader from {inspect.getsourcefile(MultiModalDataLoader)}")
        print(f"[IMPORT-CHECK] collate_fn_multimodal from {inspect.getsourcefile(collate_fn_multimodal)}")
        print(f"[IMPORT-CHECK] ViT_GCN_Fusion from {inspect.getsourcefile(ViT_GCN_Fusion)}")
        print(f"[IMPORT-CHECK] MultiModalDataLoader.__init__ signature: {_mm_sig}")
        print(f"[IMPORT-CHECK] FocalBCELoss.forward signature: {_fbce_sig}")
        assert 'segment_len' in _mm_sig, f"[IMPORT-CHECK] segment_len missing in MultiModalDataLoader.__init__: {_mm_sig}"
        assert 'sample_weight' in _fbce_sig, f"[IMPORT-CHECK] sample_weight missing in FocalBCELoss.forward: {_fbce_sig}"
    except Exception as _e:
        raise RuntimeError(f"[IMPORT-CHECK] failed: {_e}")

    # 1. 加载训练集 (80%)
    seg_len_eff = (SEGMENT_LEN // PATCH_SIZE) * PATCH_SIZE if SEGMENT_LEN >= PATCH_SIZE else 0
    train_t_target = T
    train_segment_len = 0
    use_fixed_segment_train = USE_FIXED_SEGMENT_TRAIN
    if DATASET_SELECT == "DVLOG" and USE_SEGMENT_MIL and USE_FIXED_SEGMENT_TRAIN:
        # 避免“双重切片”：MIL 已在 forward 中做 K 段抽样
        use_fixed_segment_train = False
        print("[SEG] USE_SEGMENT_MIL=True -> disable dataset fixed segment slicing (segment_len=0)")

    if DATASET_SELECT == "DVLOG" and use_fixed_segment_train and seg_len_eff > 0:
        train_t_target = seg_len_eff
        train_segment_len = seg_len_eff
        if seg_len_eff != SEGMENT_LEN:
            print(f"[SEG] SEGMENT_LEN={SEGMENT_LEN} not divisible by PATCH_SIZE={PATCH_SIZE}, use {seg_len_eff}")

    if USE_THREE_CLIP_TRAIN and USE_TWO_CLIP_TRAIN:
        raise RuntimeError("USE_THREE_CLIP_TRAIN and USE_TWO_CLIP_TRAIN cannot both be True.")

    if USE_FUSION_MODEL or DATASET_SELECT == "DVLOG":
        # DVLOG 无 .npy 平铺文件：即使纯 ViT 也使用多模态加载器读取
        trainSet = MultiModalDataLoader(X_train, VideoPath, AudioPath, FacePath, labelPath, 
                        T_target=train_t_target, mode='train', dataset=DATASET_SELECT,
                        uniform_ratio=DVLOG_UNIFORM_RATIO,
                        dvlog_aug_noise_std=DVLOG_AUG_NOISE_STD,
                        dvlog_audio_mask_ratio=DVLOG_AUDIO_MASK_RATIO,
                        segment_len=train_segment_len,
                        use_two_clip_train=USE_TWO_CLIP_TRAIN,
                        two_clip_mode=TWO_CLIP_MODE,
                        use_three_clip_train=USE_THREE_CLIP_TRAIN,
                        three_clip_mode=THREE_CLIP_MODE,
                        quality_candidate_count=QUALITY_CANDIDATE_COUNT,
                        region_partition_scheme=REGION_PARTITION_SCHEME)
        trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_multimodal)
    else:
        trainSet = MyDataLoader(X_train, VideoPath, AudioPath, labelPath, T_target=T, mode='train')
        trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. 加载验证集 (10%) -> 这里的变量名原来叫 X_test，现在对应 X_dev
    if USE_FUSION_MODEL or DATASET_SELECT == "DVLOG":
        devSet = MultiModalDataLoader(X_dev, VideoPath, AudioPath, FacePath, labelPath, 
                          T_target=T, mode='test', dataset=DATASET_SELECT,
                          uniform_ratio=DVLOG_UNIFORM_RATIO,
                          dvlog_aug_noise_std=DVLOG_AUG_NOISE_STD,
                          dvlog_audio_mask_ratio=DVLOG_AUDIO_MASK_RATIO,
                          use_three_clip_train=False,
                          use_two_clip_train=False,
                          quality_candidate_count=QUALITY_CANDIDATE_COUNT,
                          region_partition_scheme=REGION_PARTITION_SCHEME)
        devLoader = DataLoader(devSet, batch_size=BATCH_SIZE//2, shuffle=False, collate_fn=collate_fn_multimodal)
    else:
        devSet = MyDataLoader(X_dev, VideoPath, AudioPath, labelPath, T_target=T, mode='test')
        devLoader = DataLoader(devSet, batch_size=BATCH_SIZE//2, shuffle=False)
    
    # 3. 加载最终测试集 (10%) -> 新增
    if USE_FUSION_MODEL or DATASET_SELECT == "DVLOG":
        finalTestSet = MultiModalDataLoader(X_final_test, VideoPath, AudioPath, FacePath, labelPath, 
                           T_target=T, mode='test', dataset=DATASET_SELECT,
                           uniform_ratio=DVLOG_UNIFORM_RATIO,
                           dvlog_aug_noise_std=DVLOG_AUG_NOISE_STD,
                           dvlog_audio_mask_ratio=DVLOG_AUDIO_MASK_RATIO,
                           use_three_clip_train=False,
                           use_two_clip_train=False,
                           quality_candidate_count=QUALITY_CANDIDATE_COUNT,
                           region_partition_scheme=REGION_PARTITION_SCHEME)
        finalTestLoader = DataLoader(finalTestSet, batch_size=BATCH_SIZE//2, shuffle=False, collate_fn=collate_fn_multimodal)
    else:
        finalTestSet = MyDataLoader(X_final_test, VideoPath, AudioPath, labelPath, T_target=T, mode='test')
        finalTestLoader = DataLoader(finalTestSet, batch_size=BATCH_SIZE//2, shuffle=False)

    print("DataLoaders Ready: Train={}, Dev={}, Test={}".format(
        len(trainLoader), len(devLoader), len(finalTestLoader)))
    print(f"[SEG] train_T={train_t_target}, train_segment_len={train_segment_len}, eval_sliding={USE_SLIDING_SEGMENT_EVAL}, eval_stride={SEGMENT_EVAL_STRIDE}")

    # 创建模型并移动到 device（单 GPU 环境）
    D_PROJECTION = D_EMB // 2 # 256 // 2 = 128
    FEATURE_DIM_AFTER_CONCAT = D_PROJECTION * 2 # 128 * 2 = 256

    if torch.cuda.is_available():
        # 拼接后的特征维度 = 2 * D_PROJECTION，与 ViT 内部投影保持一致
        # DVLOG: D_EMB=192 -> D_PROJECTION=96 -> concat=192
        # LMVD:  D_EMB=256 -> D_PROJECTION=128 -> concat=256
        if USE_FUSION_MODEL:
            # 【融合】使用 ViT-GCN 融合模型
            model = ViT_GCN_Fusion(
                # ViT 参数
                spectra_size=T,
                patch_size=PATCH_SIZE,
                dim=D_EMB,
                depth=DEPTH,
                heads=HEADS,
                dim_mlp=DIM_MLP,
                sd=SD_RATE,        # Step3: DVLOG=0.0 关闭随机深度
                # GCN 参数
                gcn_out_dim=32,
                gcn_nhead=4,
                # 通用参数
                num_classes=2,
                dropout=DROPOUT,
                channel=FEATURE_DIM_AFTER_CONCAT,
                # 数据集参数
                video_dim=D_VIDEO,
                audio_dim=D_AUDIO,
                dataset=DATASET_SELECT,
                # 融合策略
                fusion_mode=FUSION_MODE,
                # ViT 内部 AV cross-attn 开关
                use_av_cross_attn=USE_AV_CROSS_ATTN,
                av_cross_heads=AV_CROSS_HEADS,
                av_cross_alpha_init=AV_CROSS_ALPHA_INIT,
                # A4: 人脸有效度阈值（对照实验用）
                face_valid_thresh=FACE_VALID_THRESH,
                # 分区方案："legacy6" | "symptom7"
                region_scheme=REGION_PARTITION_SCHEME,
                # GCN 分支内部开关
                region_fusion_mode=REGION_FUSION_MODE,
                gcn_temporal_mode=GCN_TEMPORAL_MODE,
                region_mlp_dropout=REGION_MLP_DROPOUT,
                tcn_kernel_size=TCN_KERNEL_SIZE,
                # 全局支路开关：True=开启，False=关闭（填零向量，fused_dim 不变）
                # global_alpha 初值：DVLOG=0.2（保守），LMVD=0.5（适中）
                # [ABLATION] 可通过 USE_GLOBAL_BRANCH 控制是否关闭 global branch
                use_global_branch=USE_GLOBAL_BRANCH,
                global_alpha_init=0.2 if DATASET_SELECT == "DVLOG" else 0.5
            ).to(device)
            print(f"[Model] Fusion mode: {FUSION_MODE}, sd={SD_RATE}, dropout={DROPOUT}")
            print(
                f"[GCN-CFG] region_fusion={REGION_FUSION_MODE}, temporal={GCN_TEMPORAL_MODE}, "
                f"global_branch={USE_GLOBAL_BRANCH}, region_scheme={REGION_PARTITION_SCHEME}"
            )
            
            # ====== [DIAG] 第一步：验证 FACE_VALID_THRESH 是否真的生效 ======
            print(f"[CFG-CHECK] FACE_VALID_THRESH={FACE_VALID_THRESH}")
            if hasattr(model, 'face_valid_thresh'):
                print(f"[CFG-CHECK] model.face_valid_thresh={model.face_valid_thresh}")
            if hasattr(model, 'gcn_branch'):
                print(f"[CFG-CHECK] gcn_branch.face_valid_thresh={model.gcn_branch.face_valid_thresh}")
            else:
                print(f"[CFG-CHECK] model has no gcn_branch attribute")
            
            # ====== [DIAG] 第一步：验证 FACE_VALID_THRESH 是否真的生效 ======
            print(f"[CFG-CHECK] FACE_VALID_THRESH={FACE_VALID_THRESH}")
            if hasattr(model, 'face_valid_thresh'):
                print(f"[CFG-CHECK] model.face_valid_thresh={model.face_valid_thresh}")
            if hasattr(model, 'gcn_branch'):
                print(f"[CFG-CHECK] gcn_branch.face_valid_thresh={model.gcn_branch.face_valid_thresh}")
            else:
                print(f"[CFG-CHECK] model has no gcn_branch attribute")
        else:
            # 使用原始 ViT 模型
            model = ViT(
                spectra_size=T, # T=915, 序列长度
                patch_size=PATCH_SIZE, # 15
                num_classes=2,
                dim=D_EMB, # dim 修正为 256
                depth=DEPTH, # depth 提升至 8 或 12
                heads=HEADS,       # heads 设置为 4（已优化）
                dim_mlp=DIM_MLP,  # dim_mlp 修正为 1024
                # 修复: 这里的 channel 必须是特征融合后的维度256
                channel=FEATURE_DIM_AFTER_CONCAT, 
                # dim_head 必须满足 dim / heads = dim_head, 即 256 / 8 = 32
                dim_head=D_EMB // HEADS, # 32（因为HEADS=8）
                dropout=DROPOUT,  # 【优化】适度提高Dropout，控制过拟合
                video_dim=D_VIDEO,
                audio_dim=D_AUDIO,
                dataset=DATASET_SELECT,
                use_av_cross_attn=USE_AV_CROSS_ATTN,
                av_cross_heads=AV_CROSS_HEADS,
                av_cross_alpha_init=AV_CROSS_ALPHA_INIT
            ).to(device)

    # Step5: 初始化模型权重 EMA
    ema = ModelEMA(model, decay=MODEL_EMA_DECAY) if USE_MODEL_EMA else None
    print(f"[EMA] USE_MODEL_EMA={USE_MODEL_EMA}, decay={MODEL_EMA_DECAY if USE_MODEL_EMA else 'N/A'}")
    # 数据集感知的增强日志打印
    if DATASET_SELECT == "DVLOG":
        print(f"[AUG] DVLOG preset={DVLOG_AUG_PRESET}, noise_std={DVLOG_AUG_NOISE_STD}, audio_mask_ratio={DVLOG_AUDIO_MASK_RATIO}, uniform_ratio={DVLOG_UNIFORM_RATIO}")
    else:
        print(f"[AUG] LMVD augmentation: noise_std={DVLOG_AUG_NOISE_STD}, audio_mask_ratio={DVLOG_AUDIO_MASK_RATIO}, uniform_ratio={DVLOG_UNIFORM_RATIO}")
    print(f"[MIL] USE_SEGMENT_MIL={USE_SEGMENT_MIL}, seg_len={SEGMENT_LEN}, k_train={SEGMENT_K_TRAIN}, k_eval={SEGMENT_K_EVAL}, train_agg={TRAIN_MIL_AGG}, eval_agg=weighted_mean")
    if USE_SEGMENT_MIL and USE_SYMPTOM_AWARE_MIL_WEIGHT:
        print(
            f"[MIL] Symptom-aware weighting enabled: alpha={SYMPTOM_WEIGHT_ALPHA}, "
            f"region_scheme={REGION_PARTITION_SCHEME}"
        )

    # ── 损失函数初始化（Step2: LOSS_MODE 分支）────────────────────────────────
    if USE_FUSION_MODEL:
        class_w = compute_class_weights(trainSet, num_classes=2, sqrt_inverse=True).to(device)
        if LOSS_MODE == 'focal':
            pos_weight = class_w[1] / class_w[0].clamp(min=1e-6)
            lossFunc   = FocalBCELoss(gamma=2.0, pos_weight=pos_weight).to(device)
            print(f"[train/{fold_name}] LOSS_MODE=focal  gamma=2.0  pos_weight={pos_weight.item():.4f}")
        elif LOSS_MODE == 'focal_no_pw':
            # 诊断：去掉 pos_weight，判断类别权重是否是 thr 乱跳根源
            lossFunc   = FocalBCELoss(gamma=2.0, pos_weight=None).to(device)
            print(f"[train/{fold_name}] LOSS_MODE=focal_no_pw  gamma=2.0  pos_weight=1.0 (disabled)")
        else:  # 'smooth_ce'
            # 诊断：CE 基线——thr 是否更稳定、test 方差是否更小
            lossFunc   = LabelSmoothingCrossEntropy(smoothing=0.05, weight=class_w).to(device)
            print(f"[train/{fold_name}] LOSS_MODE=smooth_ce  smoothing=0.05  class_w={[round(v,4) for v in class_w.tolist()]}")
    else:
        lossFunc = FocalBCELoss(gamma=2.0).to(device)
        print(f"[train/{fold_name}] FocalBCE  gamma=2.0  (no pos_weight, ViT-only)")
    # Step3: AFI 专用 lr（scalar gate init=0，开门过快易导致 recon_v_mse 爆炸；降一档更稳）
    eff_lr = (AFI_LR_OVERRIDE if (FUSION_MODE == 'afi' and AFI_LR_OVERRIDE is not None) else lr)
    print(f"[train/{fold_name}] eff_lr={eff_lr:.2e}  (base_lr={lr:.2e}, fusion_mode={FUSION_MODE}, LOSS_MODE={LOSS_MODE})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=eff_lr,
                                    weight_decay=1e-2
                                    )

    def _aggregate_logits(logits_stack):
        """logits_stack: (B, K, C) -> (B, C)"""
        if SEGMENT_AGG == 'logsumexp':
            return torch.logsumexp(logits_stack, dim=1)
        return logits_stack.max(dim=1).values

    eval_window_debug_printed = False

    def _forward_fusion_with_optional_mil(video_x, audio_x, face_x, lens_x, quality_x=None, is_train=False):
        """融合模型前向，支持可选 segment-MIL。"""

        def _mix_segment_weights(valid_w, symptom_w):
            """valid_ratio 权重与 symptom 权重做 convex mix，并归一化。"""
            valid_w = valid_w.clamp(min=1e-6)
            valid_w = valid_w / valid_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            if (not USE_SYMPTOM_AWARE_MIL_WEIGHT) or (symptom_w is None):
                return valid_w

            symptom_w = symptom_w.clamp(min=1e-6)
            symptom_w = symptom_w / symptom_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            a = float(max(0.0, min(1.0, SYMPTOM_WEIGHT_ALPHA)))
            final_w = (1.0 - a) * valid_w + a * symptom_w
            final_w = final_w / final_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return final_w

        if (not USE_SEGMENT_MIL) or (SEGMENT_LEN <= 0):
            return model(video_x, audio_x, face_x, lens_x, quality=quality_x)

        k_seg = SEGMENT_K_TRAIN if is_train else SEGMENT_K_EVAL
        k_seg = max(1, int(k_seg))

        T_cur = int(video_x.shape[1])
        # 固定 segment 长度（不再受 min_len 约束），短样本交给 actual_lens/mask 处理
        # 同时对齐到 patch_size，避免 PatchEmbedding 的整除报错。
        if T_cur < PATCH_SIZE:
            return model(video_x, audio_x, face_x, lens_x, quality=quality_x)
        seg_len_raw = int(max(1, min(SEGMENT_LEN, T_cur)))
        seg_len = int((seg_len_raw // PATCH_SIZE) * PATCH_SIZE)
        if seg_len <= 0:
            # 极短样本回退：至少一个 patch（即 15 帧）
            seg_len = PATCH_SIZE
        seg_len = min(seg_len, T_cur)

        B = int(video_x.shape[0])
        # 每个样本独立起点（避免分布偏差），但按 k 维度组织为 batch 前向（避免 B=1 触发 BN 报错）
        starts_mat = np.zeros((B, k_seg), dtype=np.int64)
        for i in range(B):
            max_valid_i = int(lens_x[i].item()) if lens_x.numel() > 0 else T_cur
            max_valid_i = max(1, min(max_valid_i, T_cur))
            max_start_full = max(0, T_cur - seg_len)
            max_start_valid = max(0, max_valid_i - 1)
            max_start = min(max_start_full, max_start_valid)

            if is_train:
                if max_start > 0:
                    starts_i = np.random.randint(0, max_start + 1, size=k_seg)
                else:
                    starts_i = np.zeros((k_seg,), dtype=np.int64)
            else:
                if k_seg <= 1 or max_start == 0:
                    starts_i = np.zeros((k_seg,), dtype=np.int64)
                else:
                    starts_i = np.linspace(0, max_start, num=k_seg, dtype=np.int64)
            starts_mat[i] = starts_i

        logits_k = []      # list[(B,2)] * K
        gcn_logits_k = []  # list[(B,2)] * K
        valid_ratios_k = []  # A3: list[(B,)] * K，记录每段有效帧占比（评估加权用）
        symptom_scores_k = []  # list[(B,)] * K，症状感知段分数（Phase B）
        # B1: 训练期收集各段 recon 输出，K 段均値作为 recon loss 目标，提供稳定的多段信息保留正则信号
        recon_v_k  = []   # list[(B, dim)] * K
        recon_g_k  = []   # list[(B, dim)] * K
        vit_repr_k = []   # list[(B, dim)] * K
        gcn_repr_k = []   # list[(B, dim)] * K
        for kk in range(k_seg):
            starts_k = starts_mat[:, kk]
            # 构建第 kk 个 segment 的 batch（B, seg_len, ...）
            v_parts, a_parts = [], []
            f_parts = {rk: [] for rk in face_x.keys()}
            for i in range(B):
                s = int(starts_k[i]); e = s + seg_len
                v_i = video_x[i:i+1, s:e, :]
                a_i = audio_x[i:i+1, s:e, :]
                assert int(v_i.shape[1]) == seg_len, (
                    f"[MIL] segment length mismatch: got {v_i.shape[1]}, expect {seg_len}, s={s}, e={e}, T_cur={T_cur}"
                )
                v_parts.append(v_i)
                a_parts.append(a_i)
                for rk, rv in face_x.items():
                    f_parts[rk].append(rv[i:i+1, s:e, ...])

            v_seg = torch.cat(v_parts, dim=0)
            a_seg = torch.cat(a_parts, dim=0)
            f_seg = {rk: torch.cat(vv, dim=0) for rk, vv in f_parts.items()}

            starts_t = torch.as_tensor(starts_k, device=lens_x.device, dtype=lens_x.dtype)
            lens_seg = (lens_x - starts_t).clamp(min=1, max=seg_len)
            # A3: 每段有效帧比例 (B,)：评估加权均值用，每段 pad 多则权重小
            _valid_r = (lens_x.float() - starts_t.float()).clamp(0.0, float(seg_len)) / float(max(seg_len, 1))
            valid_ratios_k.append(_valid_r)
            symptom_scores_k.append(
                compute_segment_symptom_score(f_seg, lens_seg, REGION_PARTITION_SCHEME)
            )

            if quality_x is None:
                q_seg = None
            elif isinstance(quality_x, dict):
                q_seg = {k: v for k, v in quality_x.items()}
            else:
                q_seg = quality_x

            out = model(v_seg, a_seg, f_seg, lens_seg, quality=q_seg)
            logits_k.append(out['logits'])
            gcn_logits_k.append(out['logits_gcn'])
            # B1: 训练期收集 recon（不在 eval 期收集，节省内存且 eval 无据操作 loss）
            if is_train and out.get('recon_v') is not None:
                recon_v_k.append(out['recon_v'])
                recon_g_k.append(out['recon_g'])
                vit_repr_k.append(out['vit_repr'])
                gcn_repr_k.append(out['gcn_repr'])

        logits_stack = torch.stack(logits_k, dim=1)       # (B,K,2)
        gcn_stack = torch.stack(gcn_logits_k, dim=1)      # (B,K,2)

        if not is_train:
            # A3: 评估期改用 valid_ratio 加权均值（替代 logsumexp/max）
            # 原理：有效帧多的段贡献大，padding 多的段被自然降权，减少窗口抖动
            valid_w = torch.stack(valid_ratios_k, dim=1)  # (B, K)
            symptom_w = torch.stack(symptom_scores_k, dim=1) if len(symptom_scores_k) == k_seg else None
            w_stack = _mix_segment_weights(valid_w, symptom_w)
            logits_agg = (logits_stack * w_stack.unsqueeze(-1)).sum(dim=1)  # (B, 2)
            gcn_agg    = (gcn_stack   * w_stack.unsqueeze(-1)).sum(dim=1)   # (B, 2)
        elif TRAIN_MIL_AGG == 'weighted_mean':
            # Step1: 训练聚合改为 valid_ratio 加权均值，与 eval 保持一致
            # 减少"某一段碰巧好→logsumexp 放大"的现象，test 更稳，方差更小
            valid_w = torch.stack(valid_ratios_k, dim=1)  # (B, K)
            symptom_w = torch.stack(symptom_scores_k, dim=1) if len(symptom_scores_k) == k_seg else None
            w_stack = _mix_segment_weights(valid_w, symptom_w)
            logits_agg = (logits_stack * w_stack.unsqueeze(-1)).sum(dim=1)  # (B, 2)
            gcn_agg    = (gcn_stack   * w_stack.unsqueeze(-1)).sum(dim=1)   # (B, 2)
        elif TRAIN_MIL_AGG == 'logsumexp' or SEGMENT_AGG == 'logsumexp':
            # 对照用：保留 logsumexp 选项（可与 weighted_mean 做消融对比）
            logits_agg = torch.logsumexp(logits_stack, dim=1)
            gcn_agg    = torch.logsumexp(gcn_stack, dim=1)
        else:
            logits_agg = logits_stack.max(dim=1).values
            gcn_agg    = gcn_stack.max(dim=1).values

        # B1: 训练期对 K 段 recon 取均値（各段时序窗口的融合表征均値），提供稳定的信息保留正则信号
        # 评估期：recon 为 None（无 loss 计算需求，不影响推理）
        if is_train and len(recon_v_k) == k_seg:
            recon_v_agg  = torch.stack(recon_v_k,  dim=0).mean(dim=0)  # (B, dim)
            recon_g_agg  = torch.stack(recon_g_k,  dim=0).mean(dim=0)  # (B, dim)
            vit_repr_agg = torch.stack(vit_repr_k, dim=0).mean(dim=0)  # (B, dim)
            gcn_repr_agg = torch.stack(gcn_repr_k, dim=0).mean(dim=0)  # (B, dim)
        else:
            recon_v_agg = recon_g_agg = vit_repr_agg = gcn_repr_agg = None

        return {
            'logits':     logits_agg,
            'logits_gcn': gcn_agg,
            'recon_v':    recon_v_agg,   # B1: 训练期 K 段均値；评估期 None
            'recon_g':    recon_g_agg,
            'vit_repr':   vit_repr_agg,
            'gcn_repr':   gcn_repr_agg,
        }

    def _forward_fusion_eval_logits(video_x, audio_x, face_x, lens_x, quality_x=None):
        """评估期滑窗多段 logits 平均；关闭时回退到常规前向。"""
        nonlocal eval_window_debug_printed
        if (not USE_SLIDING_SEGMENT_EVAL) or DATASET_SELECT != 'DVLOG' or SEGMENT_LEN <= 0:
            return _forward_fusion_with_optional_mil(
                video_x, audio_x, face_x, lens_x, quality_x=quality_x, is_train=False
            )['logits']

        T_cur = int(video_x.shape[1])
        seg_len = int((min(SEGMENT_LEN, T_cur) // PATCH_SIZE) * PATCH_SIZE)
        if seg_len < PATCH_SIZE:
            return _forward_fusion_with_optional_mil(
                video_x, audio_x, face_x, lens_x, quality_x=quality_x, is_train=False
            )['logits']
        assert seg_len % PATCH_SIZE == 0, (
            f"[EVAL-SLIDE] seg_len={seg_len} must be divisible by PATCH_SIZE={PATCH_SIZE}"
        )

        if not eval_window_debug_printed:
            max_valid_vec = lens_x.float().clamp(min=1, max=T_cur)
            max_valid_mean = float(max_valid_vec.mean().item())
            actual_lens_mean = float(lens_x.float().mean().item())
            pad_ratio_excluded_batch = max(0.0, 1.0 - max_valid_mean / float(max(1, T_cur)))
            print(
                f"[EVAL-BATCH] T_cur={T_cur}, seg_len={seg_len}, "
                f"max_valid_mean={max_valid_mean:.2f}, actual_lens_mean={actual_lens_mean:.2f}, "
                f"pad_ratio_excluded={pad_ratio_excluded_batch:.4f}"
            )

        def _slice_quality(_q, i):
            if _q is None:
                return None
            if isinstance(_q, dict):
                return {k: v[i:i+1] for k, v in _q.items()}
            return _q[i:i+1]

        B = int(video_x.shape[0])
        logits_per_sample = []
        for i in range(B):
            max_valid_i = int(lens_x[i].item()) if lens_x.numel() > 0 else T_cur
            max_valid_i = max(1, min(max_valid_i, T_cur))

            # 起点约束：
            # 1) s < max_valid_i（只在有效范围启动窗口）
            # 2) s <= T_cur - seg_len（保证切片长度恒为 seg_len，避免尾部短片触发 patch 报错）
            max_start_full = max(0, T_cur - seg_len)
            max_start_valid = max(0, max_valid_i - 1)
            max_start = min(max_start_full, max_start_valid)

            starts = list(range(0, max_start + 1, SEGMENT_EVAL_STRIDE))
            tail = min(max_start_full, max(0, max_valid_i - seg_len))
            if len(starts) == 0:
                starts = [0]
            if starts[-1] != tail:
                starts.append(tail)
            starts = sorted(set(int(s) for s in starts if 0 <= int(s) <= max_start))
            if len(starts) == 0:
                starts = [0]

            weighted_sum = None
            weight_sum = 0.0
            kept_windows = 0

            for s in starts:
                e = int(s) + seg_len
                valid_frames = max(0, min(max_valid_i, e) - int(s))
                valid_ratio = float(valid_frames) / float(max(1, seg_len))
                if valid_ratio < SEGMENT_EVAL_MIN_VALID_RATIO:
                    continue

                v_seg = video_x[i:i+1, s:e, :]
                a_seg = audio_x[i:i+1, s:e, :]
                f_seg = {rk: rv[i:i+1, s:e, ...] for rk, rv in face_x.items()}
                assert int(v_seg.shape[1]) == seg_len, (
                    f"[EVAL-SLIDE] segment length mismatch: got {v_seg.shape[1]}, expect {seg_len}, s={s}, e={e}, T_cur={T_cur}"
                )
                lens_seg = (lens_x[i:i+1] - int(s)).clamp(min=1, max=seg_len)
                q_seg = _slice_quality(quality_x, i)

                out = model(v_seg, a_seg, f_seg, lens_seg, quality=q_seg)
                w = valid_ratio if SEGMENT_EVAL_WEIGHT_BY_VALID_RATIO else 1.0
                if weighted_sum is None:
                    weighted_sum = out['logits'] * w
                else:
                    weighted_sum = weighted_sum + out['logits'] * w
                weight_sum += float(w)
                kept_windows += 1

            # 回退：若全部窗口被过滤，至少保留一个起点窗口
            if weighted_sum is None:
                s = min(max(0, max_valid_i - seg_len), max_valid_i - 1)
                e = int(s) + seg_len
                v_seg = video_x[i:i+1, s:e, :]
                a_seg = audio_x[i:i+1, s:e, :]
                f_seg = {rk: rv[i:i+1, s:e, ...] for rk, rv in face_x.items()}
                assert int(v_seg.shape[1]) == seg_len, (
                    f"[EVAL-SLIDE-FALLBACK] segment length mismatch: got {v_seg.shape[1]}, expect {seg_len}, s={s}, e={e}, T_cur={T_cur}"
                )
                lens_seg = (lens_x[i:i+1] - int(s)).clamp(min=1, max=seg_len)
                q_seg = _slice_quality(quality_x, i)
                out = model(v_seg, a_seg, f_seg, lens_seg, quality=q_seg)
                weighted_sum = out['logits']
                weight_sum = 1.0
                kept_windows = 1

            if not eval_window_debug_printed and i == 0:
                pad_ratio = max(0.0, 1.0 - float(max_valid_i) / float(max(1, T_cur)))
                print(
                    f"[EVAL-WINDOW] T_cur={T_cur}, seg_len={seg_len}, max_valid={max_valid_i}, "
                    f"pad_ratio_excluded={pad_ratio:.4f}, starts={len(starts)}, kept={kept_windows}, "
                    f"min_valid_ratio={SEGMENT_EVAL_MIN_VALID_RATIO}, weighted={SEGMENT_EVAL_WEIGHT_BY_VALID_RATIO}"
                )
                eval_window_debug_printed = True

            logits_per_sample.append(weighted_sum / max(weight_sum, 1e-6))

        return torch.cat(logits_per_sample, dim=0)

    def _assert_fusion_batch_schema(batch, stage):
        if not isinstance(batch, (tuple, list)):
            raise TypeError(f"[{stage}] fusion batch type must be tuple/list, got {type(batch)}")
        if len(batch) != 7:
            raise RuntimeError(
                f"[{stage}] expected 7 items (video,audio,face_regions,actual_lens,label,quality,sample_weight), got {len(batch)}"
            )

    def _move_fusion_batch_to_device(batch, dev):
        """将单 clip 7-tuple batch 整体移到指定 device。"""
        v, a, fr, al, lbl, q, sw = batch
        v   = v.to(dev)
        a   = a.to(dev)
        fr  = {k: val.to(dev) for k, val in fr.items()}
        al  = al.to(dev)
        lbl = lbl.to(dev)
        q   = q.to(dev)
        sw  = sw.to(dev)
        return v, a, fr, al, lbl, q, sw

    train_steps = len(trainLoader)*epochSize
    warmup_steps = len(trainLoader)*warmupEpoch
    target_steps = len(trainLoader)*epochSize
    
    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    elif schedule == 'cyclic':
        # 【优化】实现Cyclic LR调度器：帮助跳过局部最优点，适合训练振荡的情况
        # base_lr: 最小学习率 (7e-5的1/10 = 7e-6)
        # max_lr: 最大学习率 (已优化的7e-5)
        # step_size_up: 上升步数(从base_lr到max_lr)，设置为5个epoch内的步数
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=lr / 10,          # 7e-6
            max_lr=lr,                 # 7e-5
            step_size_up=len(trainLoader) * 8,  # 8个epoch内升到max_lr
            step_size_down=len(trainLoader) * 8,  # 再8个epoch降回base_lr
            cycle_momentum=False
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=target_steps)

    logging.info('The {} training begins!'.format(fold_name))
    savePath = os.path.join(str(savePath1), str(fold_name))
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    
    for epoch in range(1, epochSize):
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        traloss_one = 0
        correct = 0
        gcn_correct = 0   # GCN 辅助头训练准确数
        total = 0
        last_batch_idx = -1
        lable1 = []
        pre1 = []
        # 数据侧诊断：face validity 统计（DVLOG 关键点追踪质量）
        valid_ratio_running = 0.0
        low_valid_running = 0.0
        valid_ratio_steps = 0
        first_batch_T = None
        lens_mean_running = 0.0
        lens_mean_steps = 0
        sw_mean_running = 0.0
        sw_low_running = 0.0
        sw_steps = 0
        # A1: recon 诊断—分别跟踪两个分支的 MSE，确认 recon 头真在训练
        recon_v_mse_running = 0.0
        recon_g_mse_running = 0.0
        recon_steps = 0
        
        model.train()
        
        # 【融合】根据是否使用融合模型，选择不同的数据解包方式
        if USE_FUSION_MODEL:
            _is_two_clip = USE_TWO_CLIP_TRAIN and DATASET_SELECT == "DVLOG"
            _is_three_clip = USE_THREE_CLIP_TRAIN and DATASET_SELECT == "DVLOG"
            if USE_MIXUP and _is_three_clip:
                raise RuntimeError("Mixup is not supported for three-clip training.")
            for batch_idx, batch in loop:
                # ---- 解包：支持 Three-Clip / Two-Clip / 单-Clip ----
                if _is_three_clip:
                    batch1, batch2, batch3 = batch
                    if batch_idx == 0:
                        _assert_fusion_batch_schema(batch1, stage='train-clip1')
                        _assert_fusion_batch_schema(batch2, stage='train-clip2')
                        _assert_fusion_batch_schema(batch3, stage='train-clip3')
                    videoData,  audioData,  face_regions,  actual_lens,  label,  quality,  sample_weight  = _move_fusion_batch_to_device(batch1, device)
                    videoData2, audioData2, face_regions2, actual_lens2, _lb2,   quality2, _sw2            = _move_fusion_batch_to_device(batch2, device)
                    videoData3, audioData3, face_regions3, actual_lens3, _lb3,   quality3, _sw3            = _move_fusion_batch_to_device(batch3, device)
                elif _is_two_clip:
                    batch1, batch2 = batch
                    if batch_idx == 0:
                        _assert_fusion_batch_schema(batch1, stage='train-clip1')
                        _assert_fusion_batch_schema(batch2, stage='train-clip2')
                    videoData,  audioData,  face_regions,  actual_lens,  label,  quality,  sample_weight  = _move_fusion_batch_to_device(batch1, device)
                    videoData2, audioData2, face_regions2, actual_lens2, _lb2,   quality2, _sw2            = _move_fusion_batch_to_device(batch2, device)
                else:
                    videoData, audioData, face_regions, actual_lens, label, quality, sample_weight = batch
                    if batch_idx == 0:
                        _assert_fusion_batch_schema(
                            (videoData, audioData, face_regions, actual_lens, label, quality, sample_weight),
                            stage='train'
                        )
                    if torch.cuda.is_available():
                        videoData = videoData.to(device)
                        audioData = audioData.to(device)
                        # face_regions 是字典，每个値需要移到 device
                        face_regions = {k: v.to(device) for k, v in face_regions.items()}
                        actual_lens = actual_lens.to(device)
                        label = label.to(device)
                        quality = quality.to(device)
                        sample_weight = sample_weight.to(device)

                last_batch_idx = batch_idx
                # 样本权重过滤：支持跳过法(weight=0)与降权法(0<weight<1)并存
                keep = sample_weight > 0
                if keep.sum() == 0:
                    continue
                if keep.sum() < keep.numel():
                    videoData = videoData[keep]
                    audioData = audioData[keep]
                    face_regions = {k: v[keep] for k, v in face_regions.items()}
                    actual_lens = actual_lens[keep]
                    label = label[keep]
                    quality = quality[keep]
                    sample_weight = sample_weight[keep]
                    if _is_two_clip:
                        videoData2    = videoData2[keep]
                        audioData2    = audioData2[keep]
                        face_regions2 = {k: v[keep] for k, v in face_regions2.items()}
                        actual_lens2  = actual_lens2[keep]
                        quality2      = quality2[keep]

                if first_batch_T is None:
                    first_batch_T = int(videoData.shape[1])
                    if DATASET_SELECT == "DVLOG" and train_segment_len > 0 and USE_FIXED_SEGMENT_TRAIN:
                        assert first_batch_T == int(train_segment_len), (
                            f"[SEG-CHECK] first_batch_T={first_batch_T} != train_segment_len={train_segment_len}"
                        )
                lens_mean_running += float(actual_lens.float().mean().item())
                lens_mean_steps += 1
                # 默认先使用原始 sample_weight，若后续进入 Mixup 会在分支内覆写 mixed_sw
                mixed_sw = sample_weight
                sw_ref = mixed_sw if USE_MIXUP else sample_weight
                sw_mean_running += float(sw_ref.float().mean().item())
                sw_low_running += float((sw_ref < 1.0).float().mean().item())
                sw_steps += 1

                # 训练期 modality dropout（仅作用于 quality gate）
                if USE_QUALITY_GATE:
                    quality_train = quality.clone()
                    drop_v = (torch.rand(quality_train.size(0), device=quality_train.device) < MODALITY_DROPOUT_P_V)
                    drop_g = (torch.rand(quality_train.size(0), device=quality_train.device) < MODALITY_DROPOUT_P_G)
                    quality_train[drop_v, 0] = 0.0
                    quality_train[drop_g, 1] = 0.0
                else:
                    quality_train = None

                lam_gcn = get_lambda_gcn(epoch, epochSize)   # 动态 GCN 辅助权重（Phase 2-1）
                loss_recon = 0.0  # 提前初始化，two-clip / single-clip 路径按需覆写

                if USE_MIXUP:
                    # ── Mixup 数据增强 ─────────────────────────────────────────────
                    alpha = 0.4
                    lam   = beta(alpha, alpha)
                    index = torch.randperm(videoData.size(0)).to(device)

                    mixed_video = lam * videoData + (1 - lam) * videoData[index, :]
                    mixed_audio = lam * audioData + (1 - lam) * audioData[index, :]
                    mixed_face_regions = {
                        k: lam * v + (1 - lam) * v[index, :]
                        for k, v in face_regions.items()
                    }
                    mixed_actual_lens = torch.max(actual_lens, actual_lens[index])
                    label_a, label_b  = label, label[index]
                    mixed_sw = torch.minimum(sample_weight, sample_weight[index])

                    out_dict   = _forward_fusion_with_optional_mil(
                        mixed_video, mixed_audio, mixed_face_regions, mixed_actual_lens,
                        quality_x=(torch.minimum(quality_train, quality_train[index]) if quality_train is not None else None),
                        is_train=True
                    )
                    output     = out_dict['logits']
                    output_gcn = out_dict['logits_gcn']

                    loss_fusion = (lam       * lossFunc(output,     label_a.long(), sample_weight=mixed_sw)
                                   + (1-lam) * lossFunc(output,     label_b.long(), sample_weight=mixed_sw))
                    loss_gcn    = (lam       * lossFunc(output_gcn, label_a.long(), sample_weight=mixed_sw)
                                   + (1-lam) * lossFunc(output_gcn, label_b.long(), sample_weight=mixed_sw))
                    vp_ref_lens = mixed_actual_lens
                else:
                    if _is_three_clip or _is_two_clip:
                        # ── N-Clip Forward: batch concat -> single forward -> reshape avg ──
                        def _apply_quality_dropout(q):
                            if not USE_QUALITY_GATE:
                                return None
                            q_out = q.clone()
                            dv = (torch.rand(q_out.size(0), device=q_out.device) < MODALITY_DROPOUT_P_V)
                            dg = (torch.rand(q_out.size(0), device=q_out.device) < MODALITY_DROPOUT_P_G)
                            q_out[dv, 0] = 0.0
                            q_out[dg, 1] = 0.0
                            return q_out

                        clips = [
                            (videoData, audioData, face_regions, actual_lens, _apply_quality_dropout(quality))
                        ]
                        if _is_two_clip:
                            clips.append(
                                (videoData2, audioData2, face_regions2, actual_lens2, _apply_quality_dropout(quality2))
                            )
                        else:
                            clips.append(
                                (videoData2, audioData2, face_regions2, actual_lens2, _apply_quality_dropout(quality2))
                            )
                            clips.append(
                                (videoData3, audioData3, face_regions3, actual_lens3, _apply_quality_dropout(quality3))
                            )

                        n_clips = len(clips)
                        video_cat = torch.cat([c[0] for c in clips], dim=0)
                        audio_cat = torch.cat([c[1] for c in clips], dim=0)
                        actual_lens_cat = torch.cat([c[3] for c in clips], dim=0)
                        face_regions_cat = {
                            k: torch.cat([c[2][k] for c in clips], dim=0)
                            for k in face_regions.keys()
                        }
                        quality_list = [c[4] for c in clips]
                        quality_cat = torch.cat(quality_list, dim=0) if (quality_list[0] is not None) else None

                        out_cat = _forward_fusion_with_optional_mil(
                            video_cat, audio_cat, face_regions_cat, actual_lens_cat,
                            quality_x=quality_cat, is_train=True
                        )

                        B = int(label.shape[0])
                        output = out_cat['logits'].view(n_clips, B, -1).mean(dim=0)
                        output_gcn = out_cat['logits_gcn'].view(n_clips, B, -1).mean(dim=0)
                        loss_fusion = lossFunc(output,     label.long(), sample_weight=sample_weight)
                        loss_gcn    = lossFunc(output_gcn, label.long(), sample_weight=sample_weight)
                        vp_ref_lens = actual_lens
                        if out_cat.get('recon_v') is not None:
                            recon_v = out_cat['recon_v'].view(n_clips, B, -1)
                            recon_g = out_cat['recon_g'].view(n_clips, B, -1)
                            vit_repr = out_cat['vit_repr'].view(n_clips, B, -1)
                            gcn_repr = out_cat['gcn_repr'].view(n_clips, B, -1)
                            if RECON_TARGET_DETACH:
                                vit_repr = vit_repr.detach()
                                gcn_repr = gcn_repr.detach()
                            _rv_mse_val = F.mse_loss(recon_v, vit_repr)
                            _rg_mse_val = F.mse_loss(recon_g, gcn_repr)
                            _recon_lam  = AFI_RECON_LAMBDA if FUSION_MODE == 'afi' else RECON_LAMBDA
                            loss_recon  = _recon_lam * (_rv_mse_val + _rg_mse_val)
                            recon_v_mse_running += _rv_mse_val.item()
                            recon_g_mse_running += _rg_mse_val.item()
                            recon_steps += 1
                    else:
                        # ── 单段直接前向（DVLOG 默认关闭 Mixup）──────────────────
                        out_dict   = _forward_fusion_with_optional_mil(
                            videoData, audioData, face_regions, actual_lens,
                            quality_x=quality_train,
                            is_train=True
                        )
                        output     = out_dict['logits']
                        output_gcn = out_dict['logits_gcn']

                        loss_fusion = lossFunc(output,     label.long(), sample_weight=sample_weight)
                        loss_gcn    = lossFunc(output_gcn, label.long(), sample_weight=sample_weight)
                        vp_ref_lens = actual_lens

                # 重构损失（单段路径；two-clip / Mixup 已在各自分支内处理）
                # Step3: AFI 使用专用 lambda（更小，防止 recon 反拉 backbone 导致爆炸）
                if not _is_two_clip and not _is_three_clip and not USE_MIXUP:
                    if out_dict.get('recon_v') is not None:
                        target_v = out_dict['vit_repr'].detach() if RECON_TARGET_DETACH else out_dict['vit_repr']
                        target_g = out_dict['gcn_repr'].detach() if RECON_TARGET_DETACH else out_dict['gcn_repr']
                        _rv_mse_val = F.mse_loss(out_dict['recon_v'], target_v)
                        _rg_mse_val = F.mse_loss(out_dict['recon_g'], target_g)
                        _recon_lam  = AFI_RECON_LAMBDA if FUSION_MODE == 'afi' else RECON_LAMBDA
                        loss_recon  = _recon_lam * (_rv_mse_val + _rg_mse_val)
                        # A1: 离线统计（no_grad已由计算图提供）
                        recon_v_mse_running += _rv_mse_val.item()
                        recon_g_mse_running += _rg_mse_val.item()
                        recon_steps += 1
                traLoss = loss_fusion + lam_gcn * loss_gcn + loss_recon

                traloss_one += traLoss.item()   # .item() 防止 tensor 图堆积
                optimizer.zero_grad()
                traLoss.backward()

                # 梯度裁剪：限制梯度范数，防止数值爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                if ema is not None:          # Step5: 更新 EMA 权重
                    ema.update(model)

                _, predicted     = torch.max(output.data,     1)
                _, gcn_predicted = torch.max(output_gcn.data, 1)
                total       += label.size(0)
                correct     += predicted.eq(label.data).cpu().sum()
                gcn_correct += gcn_predicted.eq(label.data).cpu().sum()

                # ── 数据侧诊断：统计当前 batch 的 face 有效帧比例与低质量帧比例 ─────────
                # validity 通道约定在最后一维（index=-1），节点维任取一个即可（如节点0）
                _face_src = mixed_face_regions if USE_MIXUP else face_regions
                _lens_src = mixed_actual_lens if USE_MIXUP else actual_lens
                _any_region = _face_src.get('mouth', next(iter(_face_src.values())))  # (B,T,N,C)
                if _any_region.shape[-1] >= 11:
                    _v = _any_region[:, :, 0, -1]  # (B,T) in [0,1]
                    _T = _v.shape[1]
                    _mask = (torch.arange(_T, device=_v.device).unsqueeze(0) < _lens_src.unsqueeze(1))
                    _den = _mask.sum().clamp(min=1)
                    _valid_mean = ((_v * _mask.float()).sum() / _den).item()
                    _low_ratio = (((_v < 0.2) & _mask).sum().float() / _den.float()).item()
                    valid_ratio_running += _valid_mean
                    low_valid_running += _low_ratio
                    valid_ratio_steps += 1
                else:
                    _valid_mean = float('nan')
                    _low_ratio = float('nan')

                loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
                # vp_r: 当前 batch 平均有效帧占比（进而反映有效 patch 占比）
                vp_r = vp_ref_lens.float().clamp(max=T).mean().item() / T
                loop.set_postfix(
                    loss   =f'{traloss_one/(batch_idx+1):.4f}',
                    lam_g  =f'{lam_gcn:.2f}',
                    gcn_acc=f'{100.0*gcn_correct/total:.1f}%',
                    vp_r   =f'{vp_r:.2f}',
                    f_valid=f'{_valid_mean:.2f}' if np.isfinite(_valid_mean) else 'n/a',
                    f_low =f'{_low_ratio:.2f}' if np.isfinite(_low_ratio) else 'n/a',
                    rec    =f'{(loss_recon if isinstance(loss_recon, float) else float(loss_recon)):.4f}'
                )
        else:
            # ViT-only 训练路径（纯 ViT 对照实验）
            for batch_idx, batch in loop:
                if DATASET_SELECT == "DVLOG":
                    videoData, audioData, _face_regions, _actual_lens, label, _quality, _sample_weight = batch
                else:
                    videoData, audioData, label = batch
                last_batch_idx = batch_idx
                if torch.cuda.is_available():
                    videoData = videoData.to(device)
                    audioData = audioData.to(device)
                    label = label.to(device)

                # 1. 生成 Mixup 系数 lam (0到1之间)
                alpha = 0.4
                lam = beta(alpha, alpha)

                # 2. 生成随机打乱的索引
                index = torch.randperm(videoData.size(0)).to(device)

                # 3. 混合输入 (Mix Data)
                mixed_video = lam * videoData + (1 - lam) * videoData[index, :]
                mixed_audio = lam * audioData + (1 - lam) * audioData[index, :]

                # 4. 混合标签 (Mix Labels)
                label_a, label_b = label, label[index]

                # 5. 前向传播（纯 ViT）
                output = model(mixed_video, mixed_audio)

                # 6. 计算 Mixup Loss
                loss_fusion = (lam       * lossFunc(output, label_a.long())
                               + (1-lam) * lossFunc(output, label_b.long()))
                traLoss = loss_fusion

                traloss_one += traLoss.item()
                optimizer.zero_grad()
                traLoss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                _, predicted = torch.max(output.data, 1)
                total   += label.size(0)
                correct += predicted.eq(label.data).cpu().sum()

                loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
                loop.set_postfix(
                    loss=f'{traloss_one/(batch_idx+1):.4f}',
                    acc=f'{100.0*correct/total:.1f}%'
                )

        _face_valid_epoch = (valid_ratio_running / valid_ratio_steps) if valid_ratio_steps > 0 else float('nan')
        _face_low_epoch = (low_valid_running / valid_ratio_steps) if valid_ratio_steps > 0 else float('nan')
        _lens_mean_epoch = (lens_mean_running / lens_mean_steps) if lens_mean_steps > 0 else float('nan')
        _sw_mean_epoch = (sw_mean_running / sw_steps) if sw_steps > 0 else float('nan')
        _sw_low_epoch = (sw_low_running / sw_steps) if sw_steps > 0 else float('nan')
        logging.info(
            'Epoch:{}, Batch:{}, Loss:{:.4f}, FusionAcc:{:.1f}%, GCN_Acc:{:.1f}%, lam_gcn:{:.2f}, FaceValid:{:.3f}, FaceLow<{:.1f}:{:.3f}'.format(
                epoch, (last_batch_idx + 1), traloss_one / len(trainLoader),
                100.0 * correct / total, 100.0 * gcn_correct / total,
                get_lambda_gcn(epoch, epochSize),
                _face_valid_epoch if np.isfinite(_face_valid_epoch) else -1.0,
                FACE_VALID_THRESH,  # 动态显示实际使用的阈值
                _face_low_epoch if np.isfinite(_face_low_epoch) else -1.0,
            )
        )
        if np.isfinite(_face_valid_epoch):
            print(f"[FaceValid][Train Epoch {epoch}] valid_mean={_face_valid_epoch:.3f}, low(<{FACE_VALID_THRESH})={_face_low_epoch:.3f}")
        # A1: recon 诊断—每 epoch 打印两个分支的 MSE，确认 recon 头真实在训练
        _rv_mean = recon_v_mse_running / max(recon_steps, 1)
        _rg_mean = recon_g_mse_running / max(recon_steps, 1)
        _recon_lam_diag = AFI_RECON_LAMBDA if FUSION_MODE == 'afi' else RECON_LAMBDA
        print(
            f"[Recon][Epoch {epoch}] recon_v_mse={_rv_mean:.6f}  recon_g_mse={_rg_mean:.6f}  "
            f"recon_total_weighted={_recon_lam_diag * (_rv_mean + _rg_mean):.6f}  "
            f"recon_lam={_recon_lam_diag}  steps={recon_steps}"
        )
        logging.info(
            f'[Recon] Epoch:{epoch} recon_v_mse={_rv_mean:.6f}, recon_g_mse={_rg_mean:.6f}, '
            f'recon_total_weighted={_recon_lam_diag * (_rv_mean + _rg_mean):.6f}, recon_lam={_recon_lam_diag}'
        )
        if USE_FUSION_MODEL and DATASET_SELECT == "DVLOG":
            print(f"[SEG-CHECK][Epoch {epoch}] seg_len_eff={train_segment_len if train_segment_len > 0 else T}")
            print(f"[SEG-CHECK][Epoch {epoch}] first_batch_T={first_batch_T if first_batch_T is not None else 'n/a'}")
            print(f"[SEG-CHECK][Epoch {epoch}] actual_lens_mean={_lens_mean_epoch:.2f}")
            print(f"[SW-CHECK][Epoch {epoch}] sw_mean={_sw_mean_epoch:.4f}, sw_lt1_ratio={_sw_low_epoch:.4f}")

        if epoch-warmupEpoch >=0 and epoch % testRows == 0:
            train_num = 0
            correct = 0
            total = 0
            dictt, labelDict = {},{}
            
            
            label2=[]
            pre2 = []
            
            model.eval()
            print("*******dev********")
            loop = tqdm(enumerate(devLoader), total=len(devLoader))
            dev_all_preds  = []   # 收集 dev 预测（逐 batch append）
            dev_all_labels = []   # 收集 dev 真实标签
            dev_all_p1     = []   # 收集 Depression softmax 概率，用于 threshold sweep
            with torch.no_grad():
                loss_one = 0

                for batch_idx, batch in loop:
                    if USE_FUSION_MODEL:
                        if batch_idx == 0:
                            _assert_fusion_batch_schema(batch, stage='dev')
                            print(
                                f"[EVAL-PATH][dev] using _forward_fusion_eval_logits "
                                f"(sliding={USE_SLIDING_SEGMENT_EVAL}, seg_len={SEGMENT_LEN}, stride={SEGMENT_EVAL_STRIDE})"
                            )
                        videoData, audioData, face_regions, actual_lens, label, quality, sample_weight = batch
                        if torch.cuda.is_available():
                            videoData    = videoData.to(device)
                            audioData    = audioData.to(device)
                            face_regions = {k: v.to(device) for k, v in face_regions.items()}
                            actual_lens  = actual_lens.to(device)
                            label        = label.to(device)
                            quality      = quality.to(device)
                            sample_weight = sample_weight.to(device)
                        devOutput = _forward_fusion_eval_logits(
                            videoData, audioData, face_regions, actual_lens,
                            quality_x=(quality if USE_QUALITY_GATE else None)
                        )
                        if HARD_MISSING_NEUTRALIZE:
                            hard_mask = (quality[:, 0] < 0.5) & (quality[:, 1] < 0.05)
                            if hard_mask.any():
                                devOutput = devOutput.clone()
                                devOutput[hard_mask] = 0.0
                    else:
                        if DATASET_SELECT == "DVLOG":
                            videoData, audioData, _face_regions, _actual_lens, label, _quality, _sample_weight = batch
                        else:
                            videoData, audioData, label = batch
                        if torch.cuda.is_available():
                            videoData = videoData.to(device)
                            audioData = audioData.to(device)
                            label     = label.to(device)
                        devOutput = model(videoData, audioData)
                    if USE_FUSION_MODEL:
                        loss = lossFunc(devOutput, label.long(), sample_weight=sample_weight)
                    else:
                        loss = lossFunc(devOutput, label.long())
                    loss_one += loss.item()
                    train_num += label.size(0)

                    _, predicted = torch.max(devOutput.data, 1)
                    total    += label.size(0)
                    correct  += predicted.eq(label.data).cpu().sum()

                    label2.append(label.data)
                    pre2.append(predicted)

                    lable1 += label.data.tolist()
                    pre1   += predicted.tolist()

                    dev_all_preds.append(predicted.detach().cpu().numpy())
                    dev_all_labels.append(label.detach().cpu().numpy())
                    # 收集 Depression 概率，用于 threshold sweep
                    _probs = torch.softmax(devOutput, dim=1)
                    dev_all_p1.append(_probs[:, 1].detach().cpu().numpy())
            
            acc = 100.0*correct/total
            lable1 = np.array(lable1)
            pre1 = np.array(pre1)

            p = precision_score(lable1, pre1, average='weighted')
            r = recall_score(lable1, pre1, average='weighted')
            f1score = f1_score(lable1, pre1, average='weighted')
            logging.info('precision:{}'.format(p))
            logging.info('recall:{}'.format(r))
            logging.info('f1:{}'.format(f1score))

            # ── 详细 per-class 指标（混淆矩阵 + 逐类 P/R/F1 + macro）──
            _y_pred = np.concatenate(dev_all_preds)
            _y_true = np.concatenate(dev_all_labels)
            _cm = confusion_matrix(_y_true, _y_pred, labels=[0, 1])
            _p, _r, _f1, _ = precision_recall_fscore_support(
                _y_true, _y_pred, labels=[0, 1], average=None, zero_division=0)
            _pm, _rm, _f1m, _ = precision_recall_fscore_support(
                _y_true, _y_pred, average='macro', zero_division=0)
            argmax_macro_f1 = _f1m
            print(f"\n[DEV Epoch {epoch}] Confusion Matrix (rows=true, cols=pred, label order [0,1]):")
            print(_cm)
            print(f"[DEV] Class-0(Normal)    : P={_p[0]:.4f}  R={_r[0]:.4f}  F1={_f1[0]:.4f}")
            print(f"[DEV] Class-1(Depression): P={_p[1]:.4f}  R={_r[1]:.4f}  F1={_f1[1]:.4f}")
            print(f"[DEV] Macro              : P={_pm:.4f}  R={_rm:.4f}  F1={_f1m:.4f}\n")
            logging.info(f'[DEV] CM={_cm.tolist()} | Macro P={_pm:.4f} R={_rm:.4f} F1={_f1m:.4f}')

            # ── Threshold Sweep（Dev softmax 概率阈值 0.1~0.9 → 最佳 macro-F1）──────
            # 用 p1 = softmax(logits)[:,1] 替代 argmax，使模型在类别偏置时也能找到平衡点
            _y_p1 = np.concatenate(dev_all_p1)   # (N,) Depression 概率
            best_thr, best_f1m_sweep_ep = 0.5, -1.0
            for _thr in np.linspace(0.1, 0.9, 81):
                _pred_thr = (_y_p1 >= _thr).astype(int)
                _, _, _f1m_thr, _ = precision_recall_fscore_support(
                    _y_true, _pred_thr, average='macro', zero_division=0
                )
                if _f1m_thr > best_f1m_sweep_ep:
                    best_f1m_sweep_ep = _f1m_thr
                    best_thr = float(_thr)
            print(f"[DEV] Threshold sweep: best_thr={best_thr:.2f}  sweep_F1={best_f1m_sweep_ep:.4f}  argmax_F1={_f1m:.4f}")
            logging.info(f'[DEV] Threshold sweep: best_thr={best_thr:.2f}  sweep_F1={best_f1m_sweep_ep:.4f}')

            # 选模规则（固定指标）：argmax macro-F1；sweep 仅作阈值校准
            if argmax_macro_f1 > best_argmax_f1:
                best_argmax_f1 = argmax_macro_f1
                best_thr_for_best_model = best_thr
                best_epoch_for_best_model = epoch

                model_path = os.path.join(savePath, 'best_model_by_argmax_f1.pth')
                if ema is not None:
                    _backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                    ema.apply_shadow(model)
                    torch.save({'net': model.state_dict(), 'epoch': epoch,
                                'argmax_macro_f1': best_argmax_f1}, model_path)
                    ema.restore(model, _backup)
                else:
                    torch.save({'net': model.state_dict(), 'epoch': epoch,
                                'argmax_macro_f1': best_argmax_f1}, model_path)

                with open(os.path.join(savePath, 'best_thr_for_best_model.json'), 'w') as _f:
                    json.dump({'best_thr': best_thr_for_best_model,
                               'sweep_f1': best_f1m_sweep_ep,
                               'argmax_f1': best_argmax_f1,
                               'epoch': best_epoch_for_best_model}, _f, indent=2)
                print(f"  [BestModel] argmax_F1={best_argmax_f1:.4f} @epoch {epoch} | thr={best_thr_for_best_model:.2f}")
            # ── Sweep/选模处理结束 ───────────────────────────────────────────────────

            logging.debug('Dev epoch:{}, Loss:{}, Acc:{}%'.format(epoch,loss_one/len(devLoader), acc))
            loop.set_description(f'__Dev Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=loss)
            print('Dev epoch:{}, Loss:{},Acc:{}%'.format(epoch,loss_one/len(devLoader),acc))
            if acc> mytop:
                mytop = max(acc,mytop)
                top_p = p
                top_r = r
                top_f1 = f1score
                top_pre = pre2
                top_label = label2
            
            # ================== Step4/5: EMA 指标早停逻辑 ==================
            avg_dev_loss  = loss_one / len(devLoader)
            cur_macro_f1 = float('nan')

            if EARLY_STOP_MONITOR == 'macro_f1':
                # 固定规则：用 argmax macro-F1 做早停监控（避免与阈值 sweep 耦合）
                cur_macro_f1 = argmax_macro_f1
                if not np.isfinite(cur_macro_f1):
                    print(f"[ES-WARN] epoch={epoch} cur_macro_f1 is non-finite: {cur_macro_f1}")
                if ema_monitor == float('-inf'):
                    ema_monitor = cur_macro_f1
                else:
                    ema_monitor = EMA_ALPHA * cur_macro_f1 + (1 - EMA_ALPHA) * ema_monitor
                improved = (ema_monitor > best_monitor)
                best_display = f'EMA Macro-F1={ema_monitor:.4f}'
            else:
                if ema_loss_val == float('inf'):
                    ema_loss_val = avg_dev_loss
                else:
                    ema_loss_val = EMA_ALPHA * avg_dev_loss + (1 - EMA_ALPHA) * ema_loss_val
                improved = (ema_loss_val < best_monitor)
                best_display = f'EMA Loss={ema_loss_val:.4f}'

            # 必要自检：按 monitor 分支打印，避免 loss 模式下误显示 F1 变量
            if EARLY_STOP_MONITOR == 'macro_f1':
                print(f"[ES] epoch={epoch} monitor=macro_f1 "
                    f"cur_macro_f1={cur_macro_f1 if np.isfinite(cur_macro_f1) else float('nan'):.4f} "
                    f"ema_macro_f1={ema_monitor if ema_monitor is not None else float('nan'):.4f} "
                    f"best_macro_f1={best_monitor:.4f} improved={improved}")
            else:
                print(f"[ES] epoch={epoch} monitor=loss "
                    f"cur_loss={avg_dev_loss:.4f} "
                    f"ema_loss={ema_loss_val if ema_loss_val is not None else float('nan'):.4f} "
                    f"best_loss={best_monitor:.4f} improved={improved}")

            if improved:
                best_monitor = ema_monitor if EARLY_STOP_MONITOR == 'macro_f1' else ema_loss_val
                counter = 0
                # Step5: 当监控指标创新高时，用 EMA 权重保存 best_ema checkpoint
                if ema is not None:
                    _backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                    ema.apply_shadow(model)
                    torch.save({
                        'net': model.state_dict(),
                        'epoch': epoch,
                        'ema_monitor': best_monitor,
                    }, os.path.join(savePath, 'best_ema_model.pth'))

                    # EMA 模型对应阈值（与 best_ema_model.pth 同源），避免模型-阈值错配
                    with open(os.path.join(savePath, 'best_thr_for_best_ema_model.json'), 'w') as _f:
                        json.dump({
                            'best_thr': float(best_thr),
                            'sweep_f1': float(best_f1m_sweep_ep),
                            'argmax_f1': float(argmax_macro_f1),
                            'epoch': int(epoch)
                        }, _f, indent=2)

                    ema.restore(model, _backup)
                    print(f'  [EMA] best_ema checkpoint saved at epoch {epoch} ({best_display})')
                # Fix2: best_model.pth 也跟随 monitor 指标创新高时保存
                # 这样即使 USE_MODEL_EMA=False，fallback 加载的也是 monitor 最优模型
                checkpoint_monitor = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scheduler': scheduler.state_dict(),
                    'monitor': EARLY_STOP_MONITOR,
                    'best_val': best_monitor,
                }
                torch.save(checkpoint_monitor, os.path.join(savePath, 'best_model.pth'))
                logging.info(f'  [Checkpoint] best_model.pth saved at epoch {epoch} ({best_display})')
            else:
                counter += 1
                logging.info(f'EarlyStopping counter: {counter}/{patience} | {best_display}')
                if counter >= patience:
                    logging.info('Early stopping triggered!')
                    print(f'Early stopping triggered! ({best_display})')
                    break
            # ================== 早停逻辑结束 ==================

            if acc > topacc:
                topacc = max(acc, topacc)
                # 保留按 acc 记录 top_xxx（供日志/打印用），但不再覆盖 best_model.pth
                # best_model.pth 现在由 monitor 指标（macro_f1 或 loss）导向保存（见上方 improved 块）
                checkpoint_acc = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'scheduler':scheduler.state_dict()}
                torch.save(checkpoint_acc, savePath+'/'+"mdn+tcn"+'_'+str(epoch)+'_'+ str(acc)+'_'+ str(p)+'_'+str(r)+'_'+str(f1score)+'.pth')
    
    # ================== 【新增】加载最佳权重逻辑 ==================
    print("Training Finished. Loading Best Model for Final Testing...")
    
    # ====== [DIAG] 第三步：记录训练后的 global_alpha 值 ======
    if USE_FUSION_MODEL and hasattr(model, 'gcn_branch'):
        try:
            if hasattr(model.gcn_branch, 'global_alpha'):
                _ga_raw = model.gcn_branch.global_alpha.item()
                _ga_effective = float(torch.tanh(model.gcn_branch.global_alpha).item())
                print(f"[DIAG-GlobalAlpha] raw={_ga_raw:.4f}, tanh(alpha)={_ga_effective:.4f}")
            else:
                print(f"[DIAG-GlobalAlpha] gcn_branch has no global_alpha attribute")
        except Exception as e:
            print(f"[DIAG-GlobalAlpha] failed to retrieve: {e}")
    # 加载优先级: best_ema > best_model_by_argmax_f1 > best_model fallback
    # 若加载 EMA，则读取 best_thr_for_best_ema_model.json；否则读取 argmax-f1 的阈值 JSON。
    best_model_by_f1_path = os.path.join(savePath, 'best_model_by_argmax_f1.pth')
    best_ema_path         = os.path.join(savePath, 'best_ema_model.pth')
    best_model_path       = os.path.join(savePath, 'best_model.pth')
    best_thr_json = os.path.join(savePath, 'best_thr_for_best_model.json')

    if USE_MODEL_EMA and os.path.exists(best_ema_path):
        checkpoint = torch.load(best_ema_path)
        model.load_state_dict(checkpoint['net'])
        print(f"Successfully loaded EMA model (epoch {checkpoint['epoch']}) from {best_ema_path}")
        best_thr_json = os.path.join(savePath, 'best_thr_for_best_ema_model.json')
    elif os.path.exists(best_model_by_f1_path):
        checkpoint = torch.load(best_model_by_f1_path)
        model.load_state_dict(checkpoint['net'])
        print(f"Successfully loaded argmax-f1 selected model (epoch {checkpoint.get('epoch','?')}) from {best_model_by_f1_path}")
        best_thr_json = os.path.join(savePath, 'best_thr_for_best_model.json')
    elif os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['net'])
        print(f"Successfully loaded best model from {best_model_path}")
        best_thr_json = os.path.join(savePath, 'best_thr_for_best_model.json')
    else:
        print("Warning: No best model found! Using model from last epoch.")
        best_thr_json = os.path.join(savePath, 'best_thr_for_best_model.json')

    # ── 加载 threshold sweep 保存的最佳阈值 ──────────────────────────────────
    test_thr = 0.5  # 默认回退阈值
    if os.path.exists(best_thr_json):
        with open(best_thr_json, 'r') as _f:
            _thr_data = json.load(_f)
        test_thr = float(_thr_data.get('best_thr', 0.5))
        print(f"[Test] Loaded sweep threshold: {test_thr:.2f}  (sweep_F1={_thr_data.get('sweep_f1', 0):.4f})")
    else:
        print(f"[Test] No best_thr.json found, using default thr=0.5")

    model.eval()
    test_correct = 0
    test_total = 0
    test_label = []
    test_pre = []
    
    print("******* FINAL TEST ********")
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(finalTestLoader)):
            if USE_FUSION_MODEL:
                if batch_idx == 0:
                    _assert_fusion_batch_schema(batch, stage='test')
                    print(
                        f"[EVAL-PATH][test] using _forward_fusion_eval_logits "
                        f"(sliding={USE_SLIDING_SEGMENT_EVAL}, seg_len={SEGMENT_LEN}, stride={SEGMENT_EVAL_STRIDE})"
                    )
                videoData, audioData, face_regions, actual_lens, label, quality, sample_weight = batch
                if torch.cuda.is_available():
                    videoData    = videoData.to(device)
                    audioData    = audioData.to(device)
                    face_regions = {k: v.to(device) for k, v in face_regions.items()}
                    actual_lens  = actual_lens.to(device)
                    label        = label.to(device)
                    quality      = quality.to(device)
                    sample_weight = sample_weight.to(device)
                output = _forward_fusion_eval_logits(
                    videoData, audioData, face_regions, actual_lens,
                    quality_x=(quality if USE_QUALITY_GATE else None)
                )
                if HARD_MISSING_NEUTRALIZE:
                    hard_mask = (quality[:, 0] < 0.5) & (quality[:, 1] < 0.05)
                    if hard_mask.any():
                        output = output.clone()
                        output[hard_mask] = 0.0
            else:
                if DATASET_SELECT == "DVLOG":
                    videoData, audioData, _face_regions, _actual_lens, label, _quality, _sample_weight = batch
                else:
                    videoData, audioData, label = batch
                if torch.cuda.is_available():
                    videoData = videoData.to(device)
                    audioData = audioData.to(device)
                    label     = label.to(device)
                output = model(videoData, audioData)
            # 使用 sweep 阈值（代替 argmax 固定阈值），修正"偏一类"问题
            _test_probs = torch.softmax(output, dim=1)
            predicted   = (_test_probs[:, 1] >= test_thr).long()

            test_total   += label.size(0)
            test_correct += predicted.eq(label.data).cpu().sum()

            test_label += label.data.tolist()
            test_pre   += predicted.tolist()
            
    final_acc = 100.0 * test_correct / test_total
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    
    # 保存 Confusion Matrix（保存到 fold 级子目录，文件名含 fold_name 防止多 seed/fold 覆盖）
    plot_confusion_matrix(test_label, test_pre, [0, 1], 
                          savename=os.path.join(savePath, f'cm_{fold_name}.png'),
                          title=f'Final Test Acc: {final_acc:.2f}%')
                          
    # 计算最终测试集的各项指标
    final_acc = 100.0 * test_correct / test_total
    final_p = precision_score(test_label, test_pre, average='weighted')
    final_r = recall_score(test_label, test_pre, average='weighted')
    final_f1 = f1_score(test_label, test_pre, average='weighted')
    
    print(f"--- Fold Final Metrics ---")
    print(f"Acc: {final_acc:.2f}%, Precision: {final_p:.4f}, Recall: {final_r:.4f}, F1: {final_f1:.4f}")
    
    # 关键修改：返回所有核心指标
    return {
        'acc': final_acc.item() if torch.is_tensor(final_acc) else final_acc,
        'precision': final_p,
        'recall': final_r,
        'f1': final_f1
    }

def count(string):
    dig = sum(1 for char in string if char.isdigit())
    return dig


def read_lmvd_label_value(label_csv_path):
    """鲁棒读取 LMVD 单标签文件，兼容 UTF-8/UTF-16/GBK 等编码。"""
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "gbk", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            # 常见格式：文件内容只有一格（如 "1"）
            df = pd.read_csv(label_csv_path, header=None, encoding=enc)
            if df.shape[0] > 0 and df.shape[1] > 0:
                val = pd.to_numeric(df.iloc[0, 0], errors='coerce')
                if not pd.isna(val):
                    return int(val)
        except Exception as e:
            last_err = e

        try:
            # 兼容旧写法：标签可能在列名里
            df = pd.read_csv(label_csv_path, encoding=enc)
            if len(df.columns) > 0:
                col0 = pd.to_numeric(str(df.columns[0]).strip(), errors='coerce')
                if not pd.isna(col0):
                    return int(col0)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to parse LMVD label file: {label_csv_path}. "
        f"Tried encodings={encodings}. Last error: {last_err}"
    )

if __name__ == '__main__':
    from sklearn.model_selection import KFold, StratifiedKFold
    import glob
    
    # 使用全局配置的路径
    tcn = VIDEO_FEATURE_PATH
    mdnAudioPath = AUDIO_FEATURE_PATH
    facePath = FACE_FEATURE_PATH
    labelPath = LABEL_PATH
    
    print(f"Dataset: {DATASET_SELECT}")
    print(f"Video Path: {tcn}")
    print(f"Audio Path: {mdnAudioPath}")
    print(f"Face Path: {facePath}")
    print(f"Label Path: {labelPath}")
    
    if DATASET_SELECT == "DVLOG":
        # =============================================================
        # DVLOG: 固定 labels.csv 划分 + 多随机种子重复实验 (41-46)
        # 评估训练随机性下的稳健性（robustness）
        # 划分来自 labels.csv 的 fold 列，seed 只控制训练随机性
        # =============================================================
        
        # 1) 读取并统一清洗 labels.csv
        label_df = pd.read_csv(labelPath)
        label_df['index'] = label_df['index'].astype(str)
        label_df['fold'] = label_df['fold'].astype(str).str.strip().str.lower()
        label_df['label'] = label_df['label'].astype(str).str.strip().str.lower()
        
        # 2) 只保留磁盘上真实存在的样本（避免 labels.csv 与目录不一致）
        all_folders = [d for d in os.listdir(tcn)
                       if os.path.isdir(os.path.join(tcn, d)) and d.isdigit()]
        folder_set = set(all_folders)
        before_filter = len(label_df)
        label_df = label_df[label_df['index'].isin(folder_set)].copy()
        print(f"labels.csv 过滤: {before_filter} -> {len(label_df)} (磁盘上存在的样本)")
        
        # 3) 按 labels.csv 的 fold 列固定划分
        train_df = label_df[label_df['fold'] == 'train']
        dev_df   = label_df[label_df['fold'] == 'valid']
        test_df  = label_df[label_df['fold'] == 'test']
        
        X_train = train_df['index'].to_numpy()
        X_dev   = dev_df['index'].to_numpy()
        X_test  = test_df['index'].to_numpy()
        
        # 打印类别分布（强烈建议核实）
        print(f"\n--- DVLOG 固定划分统计 ---")
        print(f"Train: {len(X_train)} samples, 分布: {train_df['label'].value_counts().to_dict()}")
        print(f"Dev:   {len(X_dev)} samples,   分布: {dev_df['label'].value_counts().to_dict()}")
        print(f"Test:  {len(X_test)} samples,  分布: {test_df['label'].value_counts().to_dict()}")
        
        # 4) 多随机种子重复实验
        seeds = [41, 42, 43, 44, 45, 46]
        metrics_history = {'acc': [], 'precision': [], 'recall': [], 'f1': []}
        
        for seed in seeds:
            set_all_seeds(seed)
            print(f"\n{'='*20} Seed {seed} {'='*20}")
            
            results = train(tcn, mdnAudioPath, facePath,
                            X_train, X_dev, X_test,
                            labelPath, fold_name=f"seed_{seed}")
            
            for k in metrics_history:
                metrics_history[k].append(results[k])
        
        # 5) 汇总 mean ± std
        print(f"\n{'*'*15} FINAL MULTI-SEED RESULTS (DVLOG, seeds={seeds}) {'*'*15}")
        print(f"{'Seed':<8} {'Acc':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
        print("-" * 54)
        for i, seed in enumerate(seeds):
            print(f"{seed:<8} {metrics_history['acc'][i]:>9.2f}% "
                  f"{metrics_history['precision'][i]:>12.4f} "
                  f"{metrics_history['recall'][i]:>10.4f} "
                  f"{metrics_history['f1'][i]:>10.4f}")
        print("-" * 54)
        for key, values in metrics_history.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            if key == 'acc':
                print(f"Mean {key.upper():<10}: {mean_val:.2f}% ± {std_val:.2f}%")
            else:
                print(f"Mean {key.upper():<10}: {mean_val:.4f} ± {std_val:.4f}")
    
    else:
        # LMVD: 保持原有 10 折交叉验证逻辑
        seed = 42
        set_all_seeds(seed)
        
        X = os.listdir(tcn)
        X.sort(key=lambda x: int(x.split(".")[0]))
        X = np.array(X)
        
        Y = []
        for i in X:
            label_file = os.path.join(labelPath, (str(i.split('.npy')[0]) + "_Depression.csv"))
            Y.append(read_lmvd_label_value(label_file))
        Y = np.array(Y)

        # 切分出 10% 的固定测试集
        X_train_val_pool, X_test_holdout, Y_train_val_pool, Y_test_holdout = train_test_split(
            X, Y, test_size=0.10, stratify=Y, random_state=seed
        )

        print(f"Total: {len(X)}, Train-Val Pool: {len(X_train_val_pool)}, Fixed Test Set: {len(X_test_holdout)}")

        # 10 折交叉验证
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        metrics_history = {'acc': [], 'precision': [], 'recall': [], 'f1': []}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_pool, Y_train_val_pool)):
            print(f"\n{'='*20} Fold {fold+1} / 10 {'='*20}")
            
            X_train_fold = X_train_val_pool[train_idx]
            X_val_fold = X_train_val_pool[val_idx]
            
            fold_results = train(tcn, mdnAudioPath, facePath,
                                 X_train_fold,
                                 X_val_fold,
                                 X_test_holdout,
                                 labelPath, 
                                 fold_name=f"Fold_{fold+1}")
            
            for key in metrics_history.keys():
                metrics_history[key].append(fold_results[key])

        # 统计最终结果
        print(f"\n{'*'*15} FINAL 10-FOLD CROSS VALIDATION RESULTS {'*'*15}")
        for key, values in metrics_history.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            if key == 'acc':
                print(f"{key.upper():<10}: {mean_val:.2f}% (+/- {std_val:.2f}%)")
            else:
                print(f"{key.upper():<10}: {mean_val:.4f} (+/- {std_val:.4f})")