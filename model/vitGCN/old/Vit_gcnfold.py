import numpy as np
import torch.nn.functional as F
import torch
import logging
from kfoldLoader import MyDataLoader 
from kfoldLoader_multimodal import MultiModalDataLoader, collate_fn_multimodal  # 【融合】导入多模态加载器
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR
from math import cos
from tqdm import tqdm
import torch.nn as nn
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix
from Vit_gcnmodel import ViT, ViT_GCN_Fusion  # 【融合】导入融合模型（相对导入）
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from numpy.random import beta
import matplotlib.pyplot as plt
import argparse
""" 实现K折交叉验证的全部逻辑，包括：
数据集索引划分、数据加载、模型初始化、训练、测试、性能指标记录和结果保存。"""

# ==================== 数据集选择开关 ====================
#python model/vitGCN/Vit_gcnfold.py --dataset DVLOG --fusion_strategy IA --label_smoothing 0.00
# 支持命令行覆盖:python model/vitGCN/Vit_gcnfold.py --dataset LMVD --fusion_strategy LT
import sys as _sys
_dataset_default = "LMVD"
for _i, _arg in enumerate(_sys.argv[:-1]):
    if _arg == '--dataset' and _sys.argv[_i + 1] in ('DVLOG', 'LMVD'):
        _dataset_default = _sys.argv[_i + 1]
        break
DATASET_SELECT = _dataset_default

# 根据数据集类型动态配置路径和参数
if DATASET_SELECT == "DVLOG":
    # D-Vlog 数据集配置
    VIDEO_FEATURE_PATH = "data/dvlog-dataset/dvlog-dataset"  # 视频和音频在同一目录
    AUDIO_FEATURE_PATH = "data/dvlog-dataset/dvlog-dataset"
    FACE_FEATURE_PATH = "data/dvlog-dataset/dvlog-dataset"   # 面部关键点路径
    LABEL_PATH = "data/dvlog-dataset/dvlog-dataset/labels.csv"  # 统一的CSV标签文件
    
    T = 915           # 统一帧数指标
    D_VIDEO = 136     # D-Vlog视频特征维度
    D_AUDIO = 25      # D-Vlog音频特征维度
    BATCH_SIZE = 16   # D-Vlog序列较短，可用大batch
    MIXUP_ALPHA = 0.0  # 【消融】关闭 Mixup，观察纯模型性能
    LABEL_SMOOTHING = 0.02  # 【优化】DVLOG 二分类+小数据，过强平滑会模糊决策边界
    DROPOUT = 0.20            # 【优化】D-Vlog 进一步降低 Dropout，充分释放模型容量
    EMB_DROPOUT = 0.10          # 【优化】D-Vlog embedding dropout（原默认 0.3 太大导致 token 表征抖动）
    TEMPORAL_MASK_RATIO = 0.05  # 【增强】D-Vlog 微表情依赖时间连续性，低比例时序掩码
    LOGIT_L2_LAM = 1e-4         # 【新增】Logit L2 惩罚系数，压住极端 logit 幅度（0=关闭）
    CONF_PENALTY_BETA = 1e-3    # 【新增】Confidence penalty（熵惩罚），鼓励输出别太尖（0=关闭）
    LOSS_TYPE = 'ce'            # 【新增】损失函数类型：'ce'=LabelSmoothingCE, 'focal'=FocalLoss
    FOCAL_GAMMA = 2.0           # 【新增】FocalLoss gamma 参数（仅 loss_type='focal' 时生效）
    V_MISSING_WEIGHT = 0.5      # 【新增】视觉缺失样本的 loss 降权系数（0.5 起步）
else:  # LMVD
    # LMVD 数据集配置
    VIDEO_FEATURE_PATH = "data/LMVD_Feature/tcnfeature" 
    AUDIO_FEATURE_PATH = "data/LMVD_Feature/Audio_feature"
    FACE_FEATURE_PATH = "data/LMVD_Feature/Video_landmarks_npy"
    LABEL_PATH = "label/label"
    
    T = 915           # 序列长度
    D_VIDEO = 171     # LMVD视频特征维度
    D_AUDIO = 128     # LMVD音频特征维度
    BATCH_SIZE = 8    # LMVD序列较长，使用小batch
    MIXUP_ALPHA = 0.0  # 【消融】关闭 Mixup，观察纯模型性能
    LABEL_SMOOTHING = 0.0   # 【优化】LMVD 数据量大，关闭 Label Smoothing
    DROPOUT = 0.10            # 【优化】LMVD 数据量充足，极低 Dropout 最大化模型容量
    EMB_DROPOUT = 0.15          # 【优化】LMVD embedding dropout（保守偏高于 DROPOUT，做对照）
    TEMPORAL_MASK_RATIO = 0.10  # 【增强】LMVD 使用较强时序掩码，提升泛化能力
    LOGIT_L2_LAM = 0.0          # 【新增】LMVD 数据量充足，默认关闭 Logit L2 惩罚
    CONF_PENALTY_BETA = 0.0     # 【新增】LMVD 默认关闭 Confidence penalty
    LOSS_TYPE = 'ce'            # 【新增】默认使用 LabelSmoothingCE
    FOCAL_GAMMA = 2.0           # 【新增】FocalLoss gamma（仅 loss_type='focal' 时生效）
    V_MISSING_WEIGHT = 1.0      # 【新增】LMVD 视觉通常不缺失，不降权

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

# ----------- 模型的超参数设置 -----------------
# T, D_VIDEO, D_AUDIO 已在上面根据数据集动态设置
D_EMB = 256        # 嵌入维度 (dim)
HEADS = 8          # 【优化】增加到8个头，提升多模态建模能力
PATCH_SIZE = 15    # Patch 大小
DEPTH = 6          # Transformer 深度/层数（降低以抑制过拟合）
DIM_MLP = 1024     # FFN 的隐层维度 (dim_mlp)


lr = 5e-5                  # 【优化】降低学习率提高稳定性，防止损失波动
WEIGHT_DECAY_CANDIDATES = [2e-3, 5e-3, 1e-2]  # 可网格搜索的正则强度
weight_decay = 5e-3        # 当前默认值（在原本2e-3基础上上调一档）
EARLY_STOP_PATIENCE = 25
EARLY_STOP_MIN_DELTA = 0.002
epochSize = 300            # 【优化】从200增加到300，给模型更多学习机会
warmupEpoch = 30           # 【关键】增加预热轮数从20增加30，稳定训练初期
testRows = 1
schedule = 'cosine'        # 【优化】使用cosine调度器替代cyclic，更平稳

# ============== Threshold / Calibration switches ==============
EARLYSTOP_FIXED_THRESH = 0.5      # 早停/选 best epoch 用固定阈值（不扫描）
SCAN_THRESH_EACH_EPOCH = False    # 训练中不再每 epoch 扫阈值，避免 dev 上高方差超参搜索
FINAL_DO_TEMP_SCALING  = True     # 训练结束后做一次 Temperature Scaling
FINAL_SCAN_THRESH_RANGE = (0.01, 0.99)  # 训练结束后再扫阈值，覆盖极端概率分布
FINAL_SCAN_STEPS = 199            # 扫描步数（步长≈0.005，精度足够）

# ============== Fusion Strategy ==============
# 'ET' = 早融合（默认，proj→CrossAttn→concat→ViT）
# 'LT' = 晚融合（各模态独立编码→高层 CrossAttn，更适配 LMVD）
# 'IT' = 中融合（前k层独立→CrossAttn→后depth-k层共享，更适配 DVLOG）
# 'IA' = 注意力门控（不注入对方 Value，用相关性 gate 自己 token，跨域更稳）
FUSION_STRATEGY = 'ET'

# ============== 命令行参数覆盖 ==============
# 用法示例: python model/vitGCN/Vit_gcnfold.py --dataset LMVD --fusion_strategy LT
#          python model/vitGCN/Vit_gcnfold.py --label_smoothing 0.00  (DVLOG 网格搜索)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--dataset', type=str, default=None,
                     choices=['DVLOG', 'LMVD'],
                     help='Select dataset (already applied at top, registered here for --help)')
_parser.add_argument('--label_smoothing', type=float, default=None,
                     help='Override LABEL_SMOOTHING (e.g., 0.00, 0.02, 0.05)')
_parser.add_argument('--fusion_strategy', type=str, default=None,
                     choices=['ET', 'LT', 'IT', 'IA'],
                     help='Override FUSION_STRATEGY (ET/LT/IT/IA)')
_parser.add_argument('--logit_l2_lam', type=float, default=None,
                     help='Override LOGIT_L2_LAM (e.g., 1e-4, 1e-3, 0=off)')
_parser.add_argument('--conf_penalty_beta', type=float, default=None,
                     help='Override CONF_PENALTY_BETA (e.g., 1e-3, 1e-2, 0=off)')
_parser.add_argument('--loss_type', type=str, default=None,
                     choices=['ce', 'focal'],
                     help='Loss function type: ce=LabelSmoothingCE, focal=FocalLoss')
_parser.add_argument('--focal_gamma', type=float, default=None,
                     help='FocalLoss gamma (only used when --loss_type=focal, default=2.0)')
_args, _ = _parser.parse_known_args()
if _args.label_smoothing is not None:
    LABEL_SMOOTHING = _args.label_smoothing
    print(f'[CLI Override] LABEL_SMOOTHING = {LABEL_SMOOTHING}')
if _args.fusion_strategy is not None:
    FUSION_STRATEGY = _args.fusion_strategy
    print(f'[CLI Override] FUSION_STRATEGY = {FUSION_STRATEGY}')
if _args.logit_l2_lam is not None:
    LOGIT_L2_LAM = _args.logit_l2_lam
    print(f'[CLI Override] LOGIT_L2_LAM = {LOGIT_L2_LAM}')
if _args.conf_penalty_beta is not None:
    CONF_PENALTY_BETA = _args.conf_penalty_beta
    print(f'[CLI Override] CONF_PENALTY_BETA = {CONF_PENALTY_BETA}')
if _args.loss_type is not None:
    LOSS_TYPE = _args.loss_type
    print(f'[CLI Override] LOSS_TYPE = {LOSS_TYPE}')
if _args.focal_gamma is not None:
    FOCAL_GAMMA = _args.focal_gamma
    print(f'[CLI Override] FOCAL_GAMMA = {FOCAL_GAMMA}')

classes = ['Normal','Depression']
ps = []
rs = []
f1s = []
totals = []

total_pre = []
total_label = []

# ----------------- 日志和保存路径的相关配置 -----------------
# 【关键】以脚本所在目录为基准，避免 CWD 不同导致路径错误
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
tim = time.strftime('%m_%d__%H_%M', time.localtime())
# 按数据集分目录：output/dvlog/ 或 output/lmvd/
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output', DATASET_SELECT.lower())
filepath = os.path.join(OUTPUT_DIR, 'logs', str(tim)) 
savePath1 = os.path.join(OUTPUT_DIR, 'models', str(tim))
os.makedirs(filepath, exist_ok=True)
os.makedirs(savePath1, exist_ok=True)
logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    # 修正日志文件路径
                    filename=os.path.join(filepath, 'training.log'),
                    filemode='w')

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    手动实现支持 Label Smoothing 的交叉熵损失函数
    适用于 PyTorch 版本 < 1.10 的环境
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        # x: (batch_size, num_classes) -> 模型输出 (logits)
        # target: (batch_size) -> 真实标签 (long)
        
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        
        # 计算真实标签对应的 NLL Loss
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        # 计算平滑项（即均匀分布的 CrossEntropy）
        smooth_loss = -logprobs.mean(dim=-1)
        
        # 加权融合
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss: 降低易分样本的权重，聚焦于难分样本。
    公式: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma=0 退化为标准 CE；gamma=2 (默认) 可有效抑制极端 hard sample 对梯度的支配。
    在存在 label noise / 系统性 hard sample 时比标准 CE 更稳定。
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重，None=不加权, list/tensor=[w0, w1]
        self.reduction = reduction
    
    def forward(self, x, target):
        # x: (B, C) logits, target: (B,) int labels
        logprobs = F.log_softmax(x, dim=-1)  # (B, C)
        probs = torch.exp(logprobs)  # (B, C)
        
        # 获取真实类的概率和 log 概率
        target_one_hot = F.one_hot(target, num_classes=x.size(-1)).float()  # (B, C)
        p_t = (probs * target_one_hot).sum(dim=-1)       # (B,)
        log_p_t = (logprobs * target_one_hot).sum(dim=-1) # (B,)
        
        # Focal 调制因子
        focal_weight = (1.0 - p_t) ** self.gamma  # (B,)
        
        # 类别权重（可选）
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=x.device, dtype=x.dtype)
            else:
                alpha_t = self.alpha
            alpha_t = alpha_t.gather(0, target)  # (B,)
            focal_weight = focal_weight * alpha_t
        
        loss = -focal_weight * log_p_t  # (B,)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_auxiliary_loss(logits, logit_l2_lam=0.0, conf_penalty_beta=0.0):
    """
    计算辅助正则损失：Logit L2 惩罚 + Confidence Penalty（熵惩罚）
    
    Args:
        logits: (B, C) 模型原始输出
        logit_l2_lam: Logit L2 系数，把 logit 幅度压住，降低 P≈0.999/0.001 的频率
        conf_penalty_beta: 熵惩罚系数，鼓励输出分布别太尖
    Returns:
        aux_loss: 辅助损失标量（0 如果两个系数都为 0）
    """
    aux_loss = 0.0
    
    # A) Logit L2 penalty: loss += lam * mean(logits^2)
    if logit_l2_lam > 0:
        aux_loss = aux_loss + logit_l2_lam * (logits ** 2).mean()
    
    # B) Confidence penalty (entropy regularization): loss -= beta * H(softmax(logits))
    #    H 越大 = 分布越均匀 → 减去 beta*H 等于鼓励更大的熵
    if conf_penalty_beta > 0:
        p = torch.softmax(logits, dim=-1)
        entropy = -(p * (p + 1e-12).log()).sum(dim=-1).mean()
        aux_loss = aux_loss - conf_penalty_beta * entropy
    
    return aux_loss


# ==================== Offender Bank：持久化 top-loss 样本追踪 ====================
class OffenderBank:
    """
    追踪训练过程中反复出现在 dev top-loss 的"惯犯"样本。
    
    两种导出模式：
    - Mode A (全量)：每个 epoch 的 top-k 记录全部保存，训练结束后按 id 聚合导出
    - Mode B (惯犯)：只导出出现次数 >= min_count 的样本
    
    使用方式：
        bank = OffenderBank(topk=10)
        # 每个 dev epoch 结束后调用：
        bank.update(epoch, records)  # records: list of dict
        # 训练结束后：
        bank.export_full_csv(path)       # Mode A
        bank.export_offenders_csv(path)  # Mode B
    """
    
    def __init__(self, topk=10, min_count=3):
        self.topk = topk
        self.min_count = min_count
        self.all_records = []       # Mode A: [(epoch, record_dict), ...]
        self.counter = {}           # id -> 出现次数
        self.latest_info = {}       # id -> 最新一次的完整 record（用于 Mode B 导出）
    
    def update(self, epoch, records):
        """
        每个 dev epoch 调用一次，传入该 epoch 的 top-k 样本信息。
        
        Args:
            epoch: 当前 epoch
            records: list of dict, 每个 dict 包含：
                id, dev_idx, label, logit0, logit1, p_dep, pred, correct,
                margin, ce_loss, actual_lens, v_path, a_path, l_path,
                v_len, a_len, v_norm, a_norm, has_nan_inf, has_allzero
        """
        for rec in records:
            sid = rec['id']
            # Mode A: 记录全量
            self.all_records.append({'epoch': epoch, **rec})
            # Counter
            self.counter[sid] = self.counter.get(sid, 0) + 1
            # 更新最新信息
            self.latest_info[sid] = {'epoch_last_seen': epoch, **rec}
    
    def get_persistent(self):
        """返回出现次数 >= min_count 的 id 列表，按出现次数降序"""
        return sorted(
            [(sid, cnt) for sid, cnt in self.counter.items() if cnt >= self.min_count],
            key=lambda x: -x[1]
        )
    
    def export_full_csv(self, out_path):
        """
        Mode A：导出每个 epoch 的 top-k 全量记录。
        列：epoch, rank, id, dev_idx, label, logit0, logit1, p_dep, pred, correct,
            margin, ce_loss, actual_lens, v_len, a_len, v_norm, a_norm,
            has_nan_inf, has_allzero, v_path, a_path, l_path
        """
        if not self.all_records:
            print(f"[OffenderBank] No records to export.")
            return
        df = pd.DataFrame(self.all_records)
        # 按 epoch + ce_loss 降序排列
        if 'ce_loss' in df.columns:
            df = df.sort_values(['epoch', 'ce_loss'], ascending=[True, False])
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
        df.to_csv(out_path, index=False, float_format='%.6f')
        print(f"[OffenderBank] Mode A full records exported: {out_path} ({len(df)} rows)")
    
    def export_offenders_csv(self, out_path):
        """
        Mode B：按 id 聚合，导出惯犯排序表。
        列：id, appear_count, mean_ce, max_ce, mean_margin, min_margin,
            label, p_dep_last, pred_last, correct_last, logit0_last, logit1_last,
            actual_lens, v_len, a_len, v_norm_last, a_norm_last,
            has_nan_inf, has_allzero, v_path, a_path, l_path
        """
        if not self.all_records:
            print(f"[OffenderBank] No records to export.")
            return
        
        df_all = pd.DataFrame(self.all_records)
        
        # 按 id 聚合统计
        agg_rows = []
        for sid, cnt in sorted(self.counter.items(), key=lambda x: -x[1]):
            subset = df_all[df_all['id'] == sid]
            latest = self.latest_info.get(sid, {})
            row = {
                'id': sid,
                'appear_count': cnt,
                'mean_ce': subset['ce_loss'].mean() if 'ce_loss' in subset.columns else None,
                'max_ce': subset['ce_loss'].max() if 'ce_loss' in subset.columns else None,
                'mean_margin': subset['margin'].mean() if 'margin' in subset.columns else None,
                'min_margin': subset['margin'].min() if 'margin' in subset.columns else None,
                'dev_idx': latest.get('dev_idx', ''),
                'label': latest.get('label', ''),
                'p_dep_last': latest.get('p_dep', ''),
                'pred_last': latest.get('pred', ''),
                'correct_last': latest.get('correct', ''),
                'logit0_last': latest.get('logit0', ''),
                'logit1_last': latest.get('logit1', ''),
                'actual_lens': latest.get('actual_lens', ''),
                'v_len': latest.get('v_len', ''),
                'a_len': latest.get('a_len', ''),
                'v_norm_last': latest.get('v_norm', ''),
                'a_norm_last': latest.get('a_norm', ''),
                'has_nan_inf': latest.get('has_nan_inf', ''),
                'has_allzero': latest.get('has_allzero', ''),
                'epoch_last_seen': latest.get('epoch_last_seen', ''),
                'v_path': latest.get('v_path', ''),
                'a_path': latest.get('a_path', ''),
                'l_path': latest.get('l_path', ''),
            }
            agg_rows.append(row)
        
        df_agg = pd.DataFrame(agg_rows)
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
        df_agg.to_csv(out_path, index=False, float_format='%.6f')
        persistent_count = sum(1 for _, cnt in self.counter.items() if cnt >= self.min_count)
        print(f"[OffenderBank] Mode B offenders exported: {out_path} "
              f"({len(df_agg)} unique, {persistent_count} persistent >= {self.min_count})")
    
    def summary_str(self, total_epochs=None):
        """返回当前惯犯统计的简要摘要字符串"""
        persistent = self.get_persistent()
        lines = [f"[OffenderBank Summary] {len(self.counter)} unique, "
                 f"{len(persistent)} persistent (>={self.min_count}x)"]
        for sid, cnt in persistent[:10]:
            info = self.latest_info.get(sid, {})
            ep_str = f", last_epoch={info.get('epoch_last_seen', '?')}" if info else ''
            label_str = f", label={info.get('label', '?')}" if info else ''
            lines.append(f"  id={sid}: {cnt}x{ep_str}{label_str}")
        return '\n'.join(lines)


def _build_topk_records(topk_idxs, topk_losses, dev_logits_cat, dev_probs,
                        dev_labels_all, logit_margin, dataset, threshold=0.5):
    """
    为 top-k 样本构建完整的 record 字典列表。
    
    Args:
        topk_idxs: top-k 样本的 dev 索引 (Tensor or list)
        topk_losses: 对应的 CE loss 值 (Tensor or list)
        dev_logits_cat: (N, 2) 所有 dev 样本的 logits
        dev_probs: list/array, P(dep) 概率
        dev_labels_all: list, 真实标签
        logit_margin: (N,) |logit1 - logit0|
        dataset: dev 数据集对象（需有 get_meta 方法）
        threshold: 分类阈值
    
    Returns:
        list of dict
    """
    records = []
    for rank, (idx, ce_val) in enumerate(zip(
            topk_idxs if isinstance(topk_idxs, list) else topk_idxs.tolist(),
            topk_losses if isinstance(topk_losses, list) else topk_losses.tolist()), 1):
        
        label = dev_labels_all[idx]
        p_dep = dev_probs[idx]
        pred = int(p_dep >= threshold)
        correct = int(pred == label)
        logits = dev_logits_cat[idx].tolist()
        margin = logit_margin[idx].item() if hasattr(logit_margin[idx], 'item') else float(logit_margin[idx])
        
        rec = {
            'rank': rank,
            'dev_idx': idx,
            'id': str(idx),  # fallback, 下面会用 meta 覆盖
            'label': label,
            'logit0': logits[0],
            'logit1': logits[1],
            'p_dep': p_dep,
            'pred': pred,
            'correct': correct,
            'margin': margin,
            'ce_loss': ce_val,
            'actual_lens': '',
            'v_path': '', 'a_path': '', 'l_path': '',
            'v_len': '', 'a_len': '',
            'v_norm': '', 'a_norm': '',
            'has_nan_inf': False, 'has_allzero': False,
        }
        
        # 从 dataset 获取 meta 信息
        if hasattr(dataset, 'get_meta'):
            meta = dataset.get_meta(idx)
            rec['id'] = meta['id']
            rec['v_path'] = meta.get('video_path', '')
            rec['a_path'] = meta.get('audio_path', '')
            rec['l_path'] = meta.get('landmark_path', '')
            
            # 快速数据质量检查
            try:
                vid = np.load(meta['video_path'])
                aud = np.load(meta['audio_path'])
                rec['v_len'] = vid.shape[0]
                rec['a_len'] = aud.shape[0]
                rec['v_norm'] = float(np.linalg.norm(vid))
                rec['a_norm'] = float(np.linalg.norm(aud))
                has_nan = np.isnan(vid).any() or np.isnan(aud).any()
                has_inf = np.isinf(vid).any() or np.isinf(aud).any()
                rec['has_nan_inf'] = has_nan or has_inf
                rec['has_allzero'] = (rec['v_norm'] < 1e-6) or (rec['a_norm'] < 1e-6)
            except Exception:
                rec['has_nan_inf'] = 'LOAD_ERROR'
                rec['has_allzero'] = 'LOAD_ERROR'
        
        records.append(rec)
    
    return records


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


def find_optimal_threshold(labels, probs, low=0.01, high=0.99, steps=197):
    """在验证集上扫描阈值，找到使 macro-F1 最大化的最优分类阈值（二分类）
    扫描范围 [0.01, 0.99]，步长约 0.005，覆盖极端概率分布"""
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(low, high, steps):
        preds = [1 if p >= thresh else 0 for p in probs]
        mf1 = f1_score(labels, preds, average='macro', zero_division=0)
        if mf1 > best_f1:
            best_f1 = mf1
            best_thresh = float(thresh)
    # 边界警告：阈值命中扫描区间边界说明概率校准可能有漂移
    if best_thresh <= low + 0.01 or best_thresh >= high - 0.01:
        print(f"⚠️ WARNING: Optimal threshold {best_thresh:.3f} hit scan boundary [{low}, {high}]. "
              f"Probability calibration may be drifting!")
    return best_thresh, best_f1


def scan_best_threshold(probs, labels, lo=0.01, hi=0.99, steps=199):
    """独立的阈值扫描函数，返回最优阈值、对应 F1 和是否命中边界
    
    与 find_optimal_threshold 的区别：
    - 返回 hit_boundary 标志，方便调用方判断是否需要警告
    - 参数名/顺序统一为 (probs, labels)，与其他工具函数一致
    """
    best_t, best_f1 = 0.5, -1.0
    ts = np.linspace(lo, hi, steps)
    for t in ts:
        preds = (np.array(probs) >= t).astype(np.int64)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    hit_boundary = (abs(best_t - lo) < 1e-3) or (abs(best_t - hi) < 1e-3)
    if hit_boundary:
        print(f"⚠️ WARNING: scan_best_threshold hit boundary t={best_t:.3f} in [{lo}, {hi}]")
    return best_t, best_f1, hit_boundary

def dump_dropouts(model):
    """【诊断工具】打印模型中所有 Dropout 层及 TransformerEncoderLayer 的 dropout 参数。
    用于一锤定音地确认实际运行时每个子模块的 dropout 是否符合预期。
    只需在第一次训练时调用一次即可。"""
    print("\n" + "="*60)
    print("[Dropout 诊断] 模型中所有 Dropout 配置：")
    print("="*60)
    for name, m in model.named_modules():
        # 检测所有 nn.Dropout 层
        if isinstance(m, nn.Dropout):
            print(f"  [Dropout]  {name}: p={m.p}")
        # 检测 DropPath (Stochastic Depth)
        if hasattr(m, 'p') and m.__class__.__name__ == 'DropPath':
            print(f"  [DropPath] {name}: p={m.p}")
        # 检测 TransformerEncoderLayer 内置 dropout
        if m.__class__.__name__ == 'TransformerEncoderLayer':
            # PyTorch TEL 将 dropout 存储在 self.dropout 或 self.dropout1/dropout2
            dp_val = getattr(m, 'dropout', None)
            if hasattr(dp_val, 'p'):
                dp_val = dp_val.p
            print(f"  [TEL]      {name}: dropout={dp_val}")
        # 检测 MultiheadAttention 内置 dropout
        if m.__class__.__name__ == 'MultiheadAttention':
            print(f"  [MHA]      {name}: dropout={m.dropout}")
    print("="*60 + "\n")


def temperature_scaling(model, dev_loader, device, use_fusion=True):
    """
    【后处理】Temperature Scaling — 训练结束后在 dev 上学一个温度参数 T
    
    核心思想：
    - 不修改模型权重，只学一个标量 T，对 logits 做 logits / T 再 softmax
    - 解决"模型过度自信导致概率分布极端、阈值漂移"的问题
    - 对 DVLOG 这种"错得很自信导致 CE 爆炸"的症状尤其对症
    Args:
        model: 已训练好的模型（eval 模式）
        dev_loader: 验证集 DataLoader
        device: 计算设备
        use_fusion: 是否使用融合模型
    Returns:
        temperature: 学到的最优温度值（float）
    """
    model.eval()
    
    # 【关键】冻结模型所有参数，确保 LBFGS 只优化温度 T，不动模型权重
    original_requires_grad = {}
    for name, p in model.named_parameters():
        original_requires_grad[name] = p.requires_grad
        p.requires_grad_(False)
    
    # 1. 收集 dev 上所有 logits 和 labels
    all_logits = []
    all_labels = []
    with torch.no_grad():
        if use_fusion:
            for videoData, audioData, face_regions, actual_lens, label, v_missing in dev_loader:
                videoData = videoData.to(device)
                audioData = audioData.to(device)
                face_regions = {k: v.to(device) for k, v in face_regions.items()}
                actual_lens = actual_lens.to(device)
                v_missing = v_missing.to(device)
                logits = model(videoData, audioData, face_regions, actual_lens, v_missing=v_missing)
                all_logits.append(logits.cpu())
                all_labels.append(label)
        else:
            for videoData, audioData, label in dev_loader:
                videoData, audioData = videoData.to(device), audioData.to(device)
                logits = model(videoData, audioData)
                all_logits.append(logits.cpu())
                all_labels.append(label)
    
    all_logits = torch.cat(all_logits, dim=0)   # (N, num_classes)
    all_labels = torch.cat(all_labels, dim=0)    # (N,)
    
    # 2. 用 LBFGS 优化温度 T（只有 1 个参数，收敛极快）
    temperature = nn.Parameter(torch.ones(1) * 1.5)  # 初始 T=1.5
    optimizer_t = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    nll_criterion = nn.CrossEntropyLoss()
    
    def eval_temp():
        optimizer_t.zero_grad()
        # T 必须 > 0，用 clamp 保护
        t = temperature.clamp(min=0.01)
        loss = nll_criterion(all_logits / t, all_labels.long())
        loss.backward()
        return loss
    
    optimizer_t.step(eval_temp)
    
    optimal_T = temperature.item()
    optimal_T = max(optimal_T, 0.01)  # 安全下界
    print(f"[Temperature Scaling] Optimal T = {optimal_T:.4f}")
    
    # 【关键】恢复模型参数的 requires_grad 状态
    for name, p in model.named_parameters():
        p.requires_grad_(original_requires_grad.get(name, True))
    
    return optimal_T

def train(VideoPath, AudioPath, FacePath, X_train, X_dev, X_final_test, labelPath, fold_name, seed=42):  # 【融合】新增FacePath；seed 控制 DataLoader 可复现
    mytop = 0
    top_macro_f1 = -float('inf')
    top_p=0
    top_r=0
    top_f1_weighted=0
    top_f1_macro=0
    top_pre=[]
    top_label=[]

    patience = EARLY_STOP_PATIENCE
    min_delta = EARLY_STOP_MIN_DELTA
    counter = 0
    best_monitor = -float('inf')  # 统一使用 macro-F1 作为早停和最优模型监控指标
    best_dev_loss = float('inf')  # 【tie-break】F1 相近时选 dev loss 更低的 epoch
    best_threshold = 0.5  # 默认阈值，后续在 dev 上搜索最优值
    toploss_counter = {}   # 【诊断】追踪反复出现在 top-loss 的 dev 样本 idx（保留兼容）
    offender_bank = OffenderBank(topk=10, min_count=3)  # 【诊断】持久化 top-loss 追踪与 CSV 导出

    # 【关键】为 DataLoader 创建独立的 Generator，确保 shuffle 顺序可复现
    g_train = torch.Generator()
    g_train.manual_seed(seed)

    # 1. 加载训练集 (80%)
    if USE_FUSION_MODEL:
        # 【融合】使用多模态加载器 (视频+音频+面部关键点)
        trainSet = MultiModalDataLoader(X_train, VideoPath, AudioPath, FacePath, labelPath,
                                    T_target=T, mode='train', dataset=DATASET_SELECT,
                                    temporal_mask_ratio=TEMPORAL_MASK_RATIO)
        trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                                 collate_fn=collate_fn_multimodal,
                                 generator=g_train, worker_init_fn=_worker_init_fn)
    else:
        trainSet = MyDataLoader(X_train, VideoPath, AudioPath, labelPath, T_target=T, mode='train',
                                 temporal_mask_ratio=TEMPORAL_MASK_RATIO)
        trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                                 generator=g_train, worker_init_fn=_worker_init_fn)
    
    # 2. 加载验证集 (10%) -> 这里的变量名原来叫 X_test，现在改叫 X_dev
    if USE_FUSION_MODEL:
        devSet = MultiModalDataLoader(X_dev, VideoPath, AudioPath, FacePath, labelPath, 
                                          T_target=T, mode='test', dataset=DATASET_SELECT)
        devLoader = DataLoader(devSet, batch_size=BATCH_SIZE//2, shuffle=False,
                               collate_fn=collate_fn_multimodal, worker_init_fn=_worker_init_fn)
    else:
        devSet = MyDataLoader(X_dev, VideoPath, AudioPath, labelPath, T_target=T, mode='test')
        devLoader = DataLoader(devSet, batch_size=BATCH_SIZE//2, shuffle=False,
                               worker_init_fn=_worker_init_fn)
    
    # 3. 加载最终测试集 (10%) -> 新增
    if USE_FUSION_MODEL:
        finalTestSet = MultiModalDataLoader(X_final_test, VideoPath, AudioPath, FacePath, labelPath, 
                                           T_target=T, mode='test', dataset=DATASET_SELECT)
        finalTestLoader = DataLoader(finalTestSet, batch_size=BATCH_SIZE//2, shuffle=False,
                                     collate_fn=collate_fn_multimodal, worker_init_fn=_worker_init_fn)
    else:
        finalTestSet = MyDataLoader(X_final_test, VideoPath, AudioPath, labelPath, T_target=T, mode='test')
        finalTestLoader = DataLoader(finalTestSet, batch_size=BATCH_SIZE//2, shuffle=False,
                                     worker_init_fn=_worker_init_fn)

    print("DataLoaders Ready: Train={}, Dev={}, Test={}".format(
        len(trainLoader), len(devLoader), len(finalTestLoader)))

    # 创建模型并移动到 device（单 GPU 环境）
    D_PROJECTION = D_EMB // 2 # 256 // 2 = 128
    FEATURE_DIM_AFTER_CONCAT = D_PROJECTION * 2 # 128 * 2 = 256

    if torch.cuda.is_available():
        # 拼接后的特征维度: (186 + 128) = 314. 这是 PatchEmbdding 的 channel (c) 维度
        FEATURE_DIM_AFTER_CONCAT = 256
        
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
                # GCN 参数
                gcn_out_dim=32,  # GCN 输出维度
                gcn_nhead=4,     # GCN 注意力头数
                # 通用参数
                num_classes=2,
                dropout=DROPOUT,
                channel=FEATURE_DIM_AFTER_CONCAT,
                emb_dropout=EMB_DROPOUT,  # 【优化】从配置区传入（DVLOG=0.10, LMVD=0.15）
                # 数据集参数
                video_dim=D_VIDEO,
                audio_dim=D_AUDIO,
                dataset=DATASET_SELECT,
                # 融合策略
                fusion_strategy=FUSION_STRATEGY,
            ).to(device)
        else:
            # 使用原始 ViT 模型
            model = ViT(
                spectra_size=T, # T=915, 序列长度
                patch_size=PATCH_SIZE, # 15
                num_classes=2,
                dim=D_EMB, # dim 修改为 256
                depth=DEPTH, # depth 提升至 8 或 12
                heads=HEADS,       # heads 设置为 4（已优化）
                dim_mlp=DIM_MLP,  # dim_mlp 修改为 1024
                # 注意: 这里的 channel 必须是特征融合后的维度 256
                channel=FEATURE_DIM_AFTER_CONCAT, 
                # dim_head 必须满足 dim / heads = dim_head，即 256 / 8 = 32
                dim_head=D_EMB // HEADS, # 32（因为 HEADS=8）
                dropout=DROPOUT,  # 按数据集配置（DVLOG=0.20, LMVD=0.10）
                emb_dropout=EMB_DROPOUT,  # 【优化】从配置区传入（DVLOG=0.10, LMVD=0.15）
                video_dim=D_VIDEO,
                audio_dim=D_AUDIO,
                dataset=DATASET_SELECT
            ).to(device)

    # 【诊断】打印模型中所有 Dropout 配置，确认实际生效的值
    dump_dropouts(model)

    # 使用全局配置的路径
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr,
                                    weight_decay=weight_decay
                                    )
    
    # 【关键】定义损失函数：按 LOSS_TYPE 选择 CE 或 Focal Loss
    if LOSS_TYPE == 'focal':
        lossFunc = FocalLoss(gamma=FOCAL_GAMMA)
        print(f'[Loss] FocalLoss(gamma={FOCAL_GAMMA})')
    else:
        lossFunc = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
        print(f'[Loss] LabelSmoothingCE(smoothing={LABEL_SMOOTHING})')
    
    # 打印 Logit 正则化配置
    if LOGIT_L2_LAM > 0 or CONF_PENALTY_BETA > 0:
        print(f'[Logit Reg] L2_lam={LOGIT_L2_LAM}, ConfPenalty_beta={CONF_PENALTY_BETA}')

    train_steps = len(trainLoader)*epochSize
    warmup_steps = len(trainLoader)*warmupEpoch
    target_steps = len(trainLoader)*epochSize
    
    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    elif schedule == 'cosine':
        # 【优化】使用余弦退火调度器,比cyclic更平稳
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    elif schedule == 'cyclic':
        # 【优化】实现Cyclic LR调度器，帮助跳过局部最优点，适合训练卡顿的情况
        # base_lr: 最小学习率 (7e-5的1/10 = 7e-6)
        # max_lr: 最大学习率 (已优化的7e-5)
        # step_size_up: 上升步数(从base_lr到max_lr)，设置为8个epoch内的步数
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=lr / 10,          # 7e-6
            max_lr=lr,                 # 7e-5
            step_size_up=len(trainLoader) * 8,  # 8个epoch内上升到max_lr
            step_size_down=len(trainLoader) * 8,  # 8个epoch降回base_lr
            cycle_momentum=False
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=target_steps)

    logging.info('The {} training begins!'.format(fold_name))
    savePath = os.path.join(str(savePath1), str(fold_name))
    os.makedirs(savePath, exist_ok=True)
    
    # == NaN 诊断：捕捉反向传播中第一个产生 NaN 的操作 ==
    # 找到 NaN 根因后请注释掉此行以恢复训练速度
    torch.autograd.set_detect_anomaly(True)
    
    
    for epoch in range(1, epochSize):
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        traloss_one = 0
        correct = 0
        total = 0
        lable1 = []
        pre1 = []
        
        model.train()
        
        
        # 【融合】根据是否使用融合模型，选择不同的数据解包方式
        if USE_FUSION_MODEL:
            for batch_idx, (videoData, audioData, face_regions, actual_lens, label, v_missing) in loop:
                if torch.cuda.is_available():
                    videoData = videoData.to(device)
                    audioData = audioData.to(device)
                    # face_regions 是字典，每个值需要移到 device
                    face_regions = {k: v.to(device) for k, v in face_regions.items()}
                    actual_lens = actual_lens.to(device)
                    label = label.to(device)
                    v_missing = v_missing.to(device)

                # Mixup 开关：alpha > 0 时启用混合，alpha = 0 时直接前向
                if MIXUP_ALPHA > 0:
                    lam = beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    index = torch.randperm(videoData.size(0)).to(device)
                    mixed_video = lam * videoData + (1 - lam) * videoData[index, :]
                    mixed_audio = lam * audioData + (1 - lam) * audioData[index, :]
                    mixed_face_regions = {k: lam * v + (1 - lam) * v[index, :] for k, v in face_regions.items()}
                    mixed_actual_lens = torch.min(actual_lens, actual_lens[index])
                    # Mixup 时 v_missing 取并集（任一侧缺失则混合样本也视为缺失）
                    mixed_v_missing = v_missing | v_missing[index]
                    label_a, label_b = label, label[index]
                    output = model(mixed_video, mixed_audio, mixed_face_regions, mixed_actual_lens, v_missing=mixed_v_missing)
                    traLoss = lam * lossFunc(output, label_a.long()) + (1 - lam) * lossFunc(output, label_b.long())
                else:
                    output = model(videoData, audioData, face_regions, actual_lens, v_missing=v_missing)
                    
                    # == NaN 诊断：检查模型输出（找到根因后可删除）==
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print(f"\n⚠️ NaN/Inf in model OUTPUT at epoch={epoch}, batch={batch_idx}")
                        print(f"  output: {output}")
                        print(f"  video range: [{videoData.min():.4f}, {videoData.max():.4f}]")
                        print(f"  audio range: [{audioData.min():.4f}, {audioData.max():.4f}]")
                        for rk, rv in face_regions.items():
                            has_nan = torch.isnan(rv).any().item()
                            print(f"  face[{rk}] range: [{rv.min():.4f}, {rv.max():.4f}], nan={has_nan}")
                        print(f"  actual_lens: {actual_lens.tolist()}")
                        print(f"  v_missing: {v_missing.tolist()}")
                        # 检查模型权重
                        for pname, p in model.named_parameters():
                            if torch.isnan(p).any() or torch.isinf(p).any():
                                print(f"  ⚠️ NaN/Inf in weight: {pname}")
                    
                    # 【核心】per-sample loss + v_missing 降权
                    loss_per = F.cross_entropy(output, label.long(), reduction='none')  # (B,)
                    w = torch.ones_like(loss_per)
                    if v_missing.any():
                        w[v_missing] = V_MISSING_WEIGHT
                    traLoss = (w * loss_per).sum() / w.sum()
                # 【新增】Logit L2 惩罚 + Confidence Penalty（抑制过尖 logits）
                aux = compute_auxiliary_loss(output, LOGIT_L2_LAM, CONF_PENALTY_BETA)
                if aux != 0.0:
                    traLoss = traLoss + aux
                traloss_one += traLoss
                optimizer.zero_grad()
                traLoss.backward()

                # 梯度裁剪：限制梯度范数，防止数值爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += predicted.eq(label.data).cpu().sum()

                loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
                loop.set_postfix(loss = traloss_one/(batch_idx+1))
        else:
            # 原始 ViT 模型训练逻辑
            for batch_idx, (videoData, audioData, label) in loop:
                if torch.cuda.is_available():
                    videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)

                # Mixup 开关：alpha > 0 时启用混合，alpha = 0 时直接前向
                if MIXUP_ALPHA > 0:
                    lam = beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    index = torch.randperm(videoData.size(0)).to(device)
                    mixed_video = lam * videoData + (1 - lam) * videoData[index, :]
                    mixed_audio = lam * audioData + (1 - lam) * audioData[index, :]
                    label_a, label_b = label, label[index]
                    output = model(mixed_video, mixed_audio)
                    traLoss = lam * lossFunc(output, label_a.long()) + (1 - lam) * lossFunc(output, label_b.long())
                else:
                    output = model(videoData, audioData)
                    traLoss = lossFunc(output, label.long())
                # 【新增】Logit L2 惩罚 + Confidence Penalty（抑制过尖 logits）
                aux = compute_auxiliary_loss(output, LOGIT_L2_LAM, CONF_PENALTY_BETA)
                if aux != 0.0:
                    traLoss = traLoss + aux
                traloss_one += traLoss
                optimizer.zero_grad()
                traLoss.backward()

                # 梯度裁剪：限制梯度范数，防止数值爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += predicted.eq(label.data).cpu().sum()

                loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
                loop.set_postfix(loss = traloss_one/(batch_idx+1))

        logging.info('EpochSize: {}, Train batch: {}, Loss:{}, Acc:{}%'.format(epoch, batch_idx+1, traloss_one/len(trainLoader), 100.0*correct/total))

        if epoch-warmupEpoch >=0 and epoch % testRows == 0:
            train_num = 0
            correct = 0
            total = 0
            dictt, labelDict = {},{}
            
            
            label2=[]
            pre2 = []
            dev_probs = []        # 【新增】收集 softmax P(Depression) 用于阈值搜索
            dev_labels_all = []
            dev_logits_all = []   # 【新增】收集原始 logits 用于 top-k per-sample loss 诊断
            
            model.eval()
            print("*******dev********")
            loop = tqdm(enumerate(devLoader), total=len(devLoader))
            with torch.no_grad():
                loss_one = 0
                
                # 銆愯瀺鍚堛€戞牴鎹槸鍚︿娇鐢ㄨ瀺鍚堟ā鍨嬮€夋嫨鏁版嵁瑙ｅ寘鏂瑰紡
                if USE_FUSION_MODEL:
                    for batch_idx, (videoData, audioData, face_regions, actual_lens, label, v_missing) in loop:
                        if torch.cuda.is_available():
                            videoData = videoData.to(device)
                            audioData = audioData.to(device)
                            face_regions = {k: v.to(device) for k, v in face_regions.items()}
                            actual_lens = actual_lens.to(device)
                            label = label.to(device)
                            v_missing = v_missing.to(device)
                        
                        devOutput = model(videoData, audioData, face_regions, actual_lens, v_missing=v_missing)
                        loss = lossFunc(devOutput, label.long())
                        loss_one += loss
                        train_num += label.size(0)
                        
                        _, predicted = torch.max(devOutput.data, 1)
                        total += label.size(0)
                        correct += predicted.eq(label.data).cpu().sum()
                        
                        label2.append(label.data)
                        pre2.append(predicted)
                        
                        lable1 += label.data.tolist()
                        pre1 += predicted.tolist()
                        dev_probs += F.softmax(devOutput, dim=-1)[:, 1].cpu().tolist()
                        dev_labels_all += label.data.cpu().tolist()
                        dev_logits_all.append(devOutput.detach().cpu())
                else:
                    for batch_idx, (videoData, audioData, label) in loop:
                        if torch.cuda.is_available():
                            videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)
                        
                        devOutput = model(videoData, audioData)
                        loss = lossFunc(devOutput, label.long())
                        loss_one += loss
                        train_num += label.size(0)
                        
                        _, predicted = torch.max(devOutput.data, 1)
                        total += label.size(0)
                        correct += predicted.eq(label.data).cpu().sum()
                        
                        label2.append(label.data)
                        pre2.append(predicted)
                        
                        lable1 += label.data.tolist()
                        pre1 += predicted.tolist()
                        dev_probs += F.softmax(devOutput, dim=-1)[:, 1].cpu().tolist()
                        dev_labels_all += label.data.cpu().tolist()
                        dev_logits_all.append(devOutput.detach().cpu())
            
            acc = 100.0*correct/total
            lable1 = np.array(lable1)
            pre1 = np.array(pre1)

            p = precision_score(lable1, pre1, average='weighted')
            r = recall_score(lable1, pre1, average='weighted')
            f1score = f1_score(lable1, pre1, average='weighted')
            macro_f1 = f1_score(lable1, pre1, average='macro', zero_division=0)
            logging.info('precision:{}'.format(p))
            logging.info('recall:{}'.format(r))
            logging.info('f1:{}'.format(f1score))
            logging.info('macro_f1:{}'.format(macro_f1))

            # 【核心改动】阈值扫描受开关控制，早停只用固定 0.5 的 macro_f1
            opt_thresh = EARLYSTOP_FIXED_THRESH
            opt_thresh_f1 = macro_f1  # 固定阈值下的 macro-F1
            if SCAN_THRESH_EACH_EPOCH:
                # 可选：训练中也扫描（仅日志/诊断用，不影响早停）
                scan_t, scan_f1, scan_hit = scan_best_threshold(dev_probs, dev_labels_all)
                opt_thresh = scan_t
                opt_thresh_f1 = scan_f1
                logging.info(f'Scan threshold: {scan_t:.3f}, Macro-F1@scan: {scan_f1:.4f}, hit_boundary={scan_hit}')
            else:
                logging.info(f'Fixed threshold: {opt_thresh:.3f}, Macro-F1@fixed: {opt_thresh_f1:.4f}')

            logging.debug('Dev epoch:{}, Loss:{}, Acc:{}%'.format(epoch,loss_one/len(devLoader), acc))
            loop.set_description(f'__Dev Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=loss)
            print('Dev epoch:{}, Loss:{},Acc:{}%,MacroF1:{:.4f},Thresh:{:.3f}'.format(epoch,loss_one/len(devLoader),acc,macro_f1,opt_thresh))

            # 【诊断】Top-k 最大 loss 样本检测 + Logit 幅度统计 + Persistent Offenders 追踪
            dev_logits_cat_diag = torch.cat(dev_logits_all, dim=0)   # [N, 2]
            dev_labels_t = torch.tensor(dev_labels_all, dtype=torch.long)  # [N]
            per_sample_loss = F.cross_entropy(dev_logits_cat_diag, dev_labels_t, reduction='none')
            
            # Logit 幅度统计（始终打印，诊断 L2 惩罚是否生效）
            logit_abs = dev_logits_cat_diag.abs()
            dev_probs_t = torch.tensor(dev_probs)
            logit_margin = (dev_logits_cat_diag[:, 1] - dev_logits_cat_diag[:, 0]).abs()
            print(f'  [Logit Stats] mean|logit|={logit_abs.mean():.3f}, '
                  f'max|logit|={logit_abs.max():.3f}, '
                  f'std|logit|={logit_abs.std():.3f}, '
                  f'mean_margin={logit_margin.mean():.3f}, '
                  f'P(dep)>0.99: {(dev_probs_t>0.99).sum().item()}, '
                  f'P(dep)<0.01: {(dev_probs_t<0.01).sum().item()}')
            
            # Top-k 详细诊断 + OffenderBank 追踪（始终运行，不限 DVLOG / loss 阈值）
            topk_k = min(10, len(per_sample_loss))
            topk_vals, topk_idxs = torch.topk(per_sample_loss, k=topk_k)
            top_margins = logit_margin[topk_idxs]
            
            # 控制台简要输出（始终打印）
            print(f'  [TopLoss] Dev avg_loss={float(loss_one/len(devLoader)):.4f}, '
                  f'top-{topk_k} mean_margin={top_margins.mean():.3f}, per-sample CE losses:')
            for rank, (idx, val) in enumerate(zip(topk_idxs.tolist(), topk_vals.tolist()), 1):
                true_label = dev_labels_all[idx]
                pred_prob = dev_probs[idx]
                logit_str = [f'{v:.2f}' for v in dev_logits_cat_diag[idx].tolist()]
                meta_str = ''
                if hasattr(devLoader.dataset, 'get_meta'):
                    meta = devLoader.dataset.get_meta(idx)
                    meta_str = f', id={meta["id"]}'
                print(f'    #{rank}: dev_idx={idx}{meta_str}, CE={val:.4f}, '
                      f'true={true_label}, P(dep)={pred_prob:.4f}, logits=[{logit_str[0]}, {logit_str[1]}]')
            
            # ── OffenderBank: 构建完整 records 并更新 ──
            topk_records = _build_topk_records(
                topk_idxs, topk_vals, dev_logits_cat_diag, dev_probs,
                dev_labels_all, logit_margin, devLoader.dataset, threshold=0.5)
            offender_bank.update(epoch, topk_records)
            
            # 兼容旧 toploss_counter（保留控制台日志）
            top10_set = set(topk_idxs.tolist())
            for idx in top10_set:
                toploss_counter[idx] = toploss_counter.get(idx, 0) + 1
            
            # 每 10 个 epoch 打印惯犯摘要 + 中间 CSV 快照
            if epoch % 10 == 0:
                persistent = offender_bank.get_persistent()
                if persistent:
                    print(f'  [Persistent Offenders] 出现 >={offender_bank.min_count} 次在 top-{topk_k}:')
                    for sid, cnt in persistent[:15]:
                        info = offender_bank.latest_info.get(sid, {})
                        print(f'    id={sid}: {cnt}x, label={info.get("label","?")}, '
                              f'last_CE={info.get("ce_loss","?"):.4f}, '
                              f'last_P(dep)={info.get("p_dep","?"):.4f}')
                # 中间快照导出
                snap_dir = os.path.join(savePath, 'offender_snapshots')
                offender_bank.export_full_csv(
                    os.path.join(snap_dir, f'topk_full_epoch{epoch}.csv'))
                
            if acc> mytop:
                mytop = max(acc,mytop)
                top_p = p
                top_r = r
                top_f1_weighted = f1score
                top_pre = pre2
                top_label = label2
            
            # ================== 插入早停逻辑 (开始) ==================
            # 【关键】早停监控指标用固定阈值 0.5 的 macro_f1（即 argmax 预测）
            # 避免在 dev 小数据上因阈值扫描方差大而选到不泛化的 epoch
            # 训练结束后再做一次 Temperature Scaling + 阈值校准
            monitor_metric = macro_f1  # macro_f1 来自 argmax(=固定 0.5) 预测
            curr_loss = float(loss_one / len(devLoader))  # 当前 epoch 的 dev 平均 loss
            f1_tie_eps = 1e-3  # F1 "接近" 的判断阈值

            # 【改进】F1 为主指标 + dev loss 为 tie-break
            #  - is_better: F1 显著提升（超过 best + min_delta）
            #  - is_tie_better: F1 接近但 dev loss 更低（同等 F1 下选更不"过拟合"的 epoch）
            is_better = (monitor_metric > best_monitor + min_delta)
            is_tie_better = (abs(monitor_metric - best_monitor) <= f1_tie_eps) and (curr_loss < best_dev_loss - 1e-4)

            if is_better or is_tie_better:
                reason = 'F1↑' if is_better else 'F1≈+Loss↓'
                best_monitor = monitor_metric
                best_dev_loss = min(best_dev_loss, curr_loss)
                counter = 0
                top_macro_f1 = monitor_metric
                top_f1_macro = macro_f1
                # 【注意】不在早停阶段保存 opt_thresh，避免不稳定阈值影响模型选择
                # best_threshold 将在训练结束后通过 Temperature Scaling 统一校准
                checkpoint = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'scheduler':scheduler.state_dict()}
                torch.save(checkpoint, savePath+'/'+"mdn+tcn"+'_'+str(epoch)+'_'+ str(acc)+'_'+ str(p)+'_'+str(r)+'_'+str(f1score)+'_macrof1_'+str(macro_f1)+'.pth')
                torch.save(checkpoint, os.path.join(savePath, 'best_model.pth'))
                logging.info(f'New best ({reason}): macro-F1={best_monitor:.4f}, dev_loss={best_dev_loss:.4f} at epoch {epoch}')
            else:
                # 同步更新 best_dev_loss（即使 F1 没提升，记录最低 loss 供后续 tie-break 参考）
                best_dev_loss = min(best_dev_loss, curr_loss)
                counter += 1
                logging.info(
                    f'EarlyStopping counter: {counter} out of {patience} | '
                    f'best macro-F1: {best_monitor:.4f}, current: {monitor_metric:.4f}, '
                    f'dev_loss: {curr_loss:.4f} (best: {best_dev_loss:.4f})'
                )
                
                if counter >= patience:
                    logging.info("Early stopping triggered! Stop training.")
                    print("Early stopping triggered!")
                    break  # <--- 关键：跳出最外层的 epoch 循环
            # ================== 插入早停逻辑 (结束) ==================
            # best checkpoint is already saved when macro-F1 improves
    
    # ================== 【新增】加载最佳权重逻辑 ==================
    print("Training Finished. Loading Best Model for Final Testing...")
    
    # ── OffenderBank 最终导出 ──
    print(offender_bank.summary_str())
    offender_bank.export_full_csv(os.path.join(savePath, 'offender_full_records.csv'))
    offender_bank.export_offenders_csv(os.path.join(savePath, 'offender_aggregated.csv'))
    
    best_model_path = os.path.join(savePath, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        # 加载 checkpoint
        checkpoint = torch.load(best_model_path)
        # 覆盖当前模型的参数
        model.load_state_dict(checkpoint['net'])
        print(f"Successfully loaded best model (Macro-F1: {top_macro_f1:.4f}) from {best_model_path}")
    else:
        print("Warning: No best model found! Using model from last epoch.")

    # ================== 【新增】Temperature Scaling 校准 ==================
    # 在 dev 上学一个温度 T，用于校准概率分布
    # 这对"错得很自信导致 CE 爆炸 / 阈值漂移"的症状尤其对症
    if FINAL_DO_TEMP_SCALING:
        optimal_temperature = temperature_scaling(model, devLoader, device, use_fusion=USE_FUSION_MODEL)
    else:
        optimal_temperature = 1.0
        print("[TempScaling] Skipped (FINAL_DO_TEMP_SCALING=False), T=1.0")
    
    # 在校准后的概率上重新扫描 dev 最优阈值
    model.eval()
    dev_logits_cal = []
    dev_labels_cal = []
    with torch.no_grad():
        if USE_FUSION_MODEL:
            for videoData, audioData, face_regions, actual_lens, label, v_missing in devLoader:
                videoData = videoData.to(device)
                audioData = audioData.to(device)
                face_regions = {k: v.to(device) for k, v in face_regions.items()}
                actual_lens = actual_lens.to(device)
                v_missing = v_missing.to(device)
                logits = model(videoData, audioData, face_regions, actual_lens, v_missing=v_missing)
                dev_logits_cal.append(logits.cpu())
                dev_labels_cal.append(label)
        else:
            for videoData, audioData, label in devLoader:
                videoData, audioData = videoData.to(device), audioData.to(device)
                logits = model(videoData, audioData)
                dev_logits_cal.append(logits.cpu())
                dev_labels_cal.append(label)
    
    dev_logits_cal = torch.cat(dev_logits_cal, dim=0)
    dev_labels_cal = torch.cat(dev_labels_cal, dim=0).numpy()
    # 用校准后的概率扫描最优阈值
    dev_probs_calibrated = F.softmax(dev_logits_cal / optimal_temperature, dim=-1)[:, 1].numpy()
    best_threshold, best_thresh_f1, thresh_hit = scan_best_threshold(
        dev_probs_calibrated.tolist(), dev_labels_cal.tolist(),
        lo=FINAL_SCAN_THRESH_RANGE[0], hi=FINAL_SCAN_THRESH_RANGE[1], steps=FINAL_SCAN_STEPS
    )
    print(f"[Post-Training Calibration] T={optimal_temperature:.4f}, Best threshold={best_threshold:.3f}, "
          f"Dev Macro-F1@thresh={best_thresh_f1:.4f}, hit_boundary={thresh_hit}")

    model.eval()
    test_label = []
    test_logits_all = []   # 【改为收集 logits，后续用 T 校准】
    
    print(f"Using calibrated threshold from dev: {best_threshold:.3f} (T={optimal_temperature:.4f})")
    print("******* FINAL TEST ********")
    with torch.no_grad():
        # 【融合】根据是否使用融合模型选择数据解包方式
        if USE_FUSION_MODEL:
            for batch_idx, (videoData, audioData, face_regions, actual_lens, label, v_missing) in tqdm(enumerate(finalTestLoader)):
                if torch.cuda.is_available():
                    videoData = videoData.to(device)
                    audioData = audioData.to(device)
                    face_regions = {k: v.to(device) for k, v in face_regions.items()}
                    actual_lens = actual_lens.to(device)
                    label = label.to(device)
                    v_missing = v_missing.to(device)
                
                output = model(videoData, audioData, face_regions, actual_lens, v_missing=v_missing)
                test_label += label.data.cpu().tolist()
                test_logits_all.append(output.cpu())
        else:
            for batch_idx, (videoData, audioData, label) in tqdm(enumerate(finalTestLoader)):
                if torch.cuda.is_available():
                    videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)
                
                output = model(videoData, audioData)
                test_label += label.data.cpu().tolist()
                test_logits_all.append(output.cpu())
    
    # ===== 用 Temperature Scaling 校准后的概率做预测 =====
    test_logits_cat = torch.cat(test_logits_all, dim=0)  # (N, 2)
    test_probs_arr = F.softmax(test_logits_cat / optimal_temperature, dim=-1)[:, 1].numpy()
    test_label_arr = np.array(test_label)
    
    # --- A. 固定阈值 0.5（纯模型能力，不受校准影响）---
    pred_fixed = (test_probs_arr >= 0.5).astype(int)
    fixed_acc = 100.0 * np.mean(pred_fixed == test_label_arr)
    fixed_p = precision_score(test_label_arr, pred_fixed, average='weighted')
    fixed_r = recall_score(test_label_arr, pred_fixed, average='weighted')
    fixed_f1 = f1_score(test_label_arr, pred_fixed, average='weighted')
    fixed_mf1 = f1_score(test_label_arr, pred_fixed, average='macro', zero_division=0)
    
    # --- B. Dev 最优阈值（模型+校准的综合效果）---
    pred_best = (test_probs_arr >= best_threshold).astype(int)
    best_acc = 100.0 * np.mean(pred_best == test_label_arr)
    best_p = precision_score(test_label_arr, pred_best, average='weighted')
    best_r = recall_score(test_label_arr, pred_best, average='weighted')
    best_f1 = f1_score(test_label_arr, pred_best, average='weighted')
    best_mf1 = f1_score(test_label_arr, pred_best, average='macro', zero_division=0)
    
    # 保存 Confusion Matrix（用 best_threshold 版本）
    plot_confusion_matrix(test_label, pred_best.tolist(), [0, 1], 
                          savename=filepath + '/final_test_confusion_matrix.png',
                          title=f'Final Test Acc: {best_acc:.2f}% (t={best_threshold:.3f})')
    
    print(f"--- Final Test @ fixed threshold=0.500 (model capacity) ---")
    print(f"Acc: {fixed_acc:.2f}%, P: {fixed_p:.4f}, R: {fixed_r:.4f}, F1: {fixed_f1:.4f}, Macro-F1: {fixed_mf1:.4f}")
    print(f"--- Final Test @ best  threshold={best_threshold:.3f} (model+calibration) ---")
    print(f"Acc: {best_acc:.2f}%, P: {best_p:.4f}, R: {best_r:.4f}, F1: {best_f1:.4f}, Macro-F1: {best_mf1:.4f}")
    delta_mf1 = best_mf1 - fixed_mf1
    print(f"    Threshold gain: Macro-F1 {'+' if delta_mf1>=0 else ''}{delta_mf1:.4f}")
    
    # 返回两套指标
    # 【关键】DVLOG 主结果用固定阈值 0.5（dev 太小，校准阈值方差大）
    #        LMVD 主结果用校准阈值（dev 足够大，校准有增益）
    if DATASET_SELECT == 'DVLOG':
        return {
            'acc': fixed_acc,           # 主结果: 固定 0.5
            'precision': fixed_p,
            'recall': fixed_r,
            'f1': fixed_f1,
            'macro_f1': fixed_mf1,
            'threshold': 0.5,
            # 校准阈值指标（附加分析，不作为主结论）
            'acc_calibrated': best_acc,
            'macro_f1_calibrated': best_mf1,
            'threshold_calibrated': best_threshold,
            # 兼容原有 key
            'acc_fixed': fixed_acc,
            'macro_f1_fixed': fixed_mf1,
        }
    else:
        return {
            'acc': best_acc,
            'precision': best_p,
            'recall': best_r,
            'f1': best_f1,
            'macro_f1': best_mf1,
            'threshold': best_threshold,
            # 固定阈值指标（用于诊断波动来源）
            'acc_fixed': fixed_acc,
            'macro_f1_fixed': fixed_mf1,
        }

def count(string):
    dig = sum(1 for char in string if char.isdigit())
    return dig

def _worker_init_fn(worker_id):
    """DataLoader worker 初始化函数，确保每个 worker 的随机种子可控"""
    import random as _random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)

def set_seed(seed: int):
    """统一设置所有随机种子，确保实验可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 【关键】保证 CUDA 卷积等算子的确定性，消除 GPU 非确定性噪声
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    from sklearn.model_selection import KFold, StratifiedKFold
    import glob

    # DVLOG: 官方划分固定，多 seed 跑 5 次消除随机性；LMVD: 单 seed + 10 折
    seeds = [42, 43, 44, 45, 46] if DATASET_SELECT == "DVLOG" else [42]

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
        # ============================================================
        # D-Vlog: 使用 labels.csv 官方 fold 列划分 (train/valid/test)
        # 数据划分固定不变，多 seed 只影响模型初始化和训练随机性
        # ============================================================
        label_df = pd.read_csv(labelPath)

        def parse_dvlog_label(label_value):
            if isinstance(label_value, str):
                return 1 if label_value.strip().lower() == 'depression' else 0
            return int(label_value)

        fold_col = label_df['fold'].astype(str).str.strip().str.lower()

        train_rows = label_df[fold_col == 'train']
        valid_rows = label_df[fold_col == 'valid']
        test_rows  = label_df[fold_col == 'test']

        X_train_all = np.array(train_rows['index'].astype(int).astype(str).tolist())
        X_dev_all   = np.array(valid_rows['index'].astype(int).astype(str).tolist())
        X_test_all  = np.array(test_rows['index'].astype(int).astype(str).tolist())

        Y_train = np.array([parse_dvlog_label(v) for v in train_rows['label'].tolist()])
        Y_dev   = np.array([parse_dvlog_label(v) for v in valid_rows['label'].tolist()])
        Y_test  = np.array([parse_dvlog_label(v) for v in test_rows['label'].tolist()])

        # 过滤不存在的样本目录，避免运行时路径错误
        def filter_existing(X, Y):
            keep = [i for i, sid in enumerate(X) if os.path.isdir(os.path.join(tcn, sid))]
            return X[keep], Y[keep]

        X_train_all, Y_train = filter_existing(X_train_all, Y_train)
        X_dev_all,   Y_dev   = filter_existing(X_dev_all,   Y_dev)
        X_test_all,  Y_test  = filter_existing(X_test_all,  Y_test)

        print("[DVLOG] Official split from labels.csv:")
        print(f"  Train: {len(X_train_all)} (Dep: {Y_train.sum()}, Nor: {len(Y_train)-Y_train.sum()})")
        print(f"  Valid: {len(X_dev_all)}   (Dep: {Y_dev.sum()}, Nor: {len(Y_dev)-Y_dev.sum()})")
        print(f"  Test : {len(X_test_all)}  (Dep: {Y_test.sum()}, Nor: {len(Y_test)-Y_test.sum()})")

        # 多 seed 跑 5 次，消除模型初始化和训练过程的随机性
        all_runs = []
        for seed in seeds:
            set_seed(seed)
            print(f"\n{'='*20} DVLOG official split | seed={seed} {'='*20}")

            res = train(
                tcn, mdnAudioPath, facePath,
                X_train_all,   # 训练集 (fold=="train", 约 70%)
                X_dev_all,     # 验证集/早停 (fold=="valid", 约 10%)
                X_test_all,    # 固定测试集 (fold=="test", 约 20%)
                labelPath,
                fold_name=f"DVLOG_official_seed{seed}",
                seed=seed
            )
            all_runs.append(res)

        # 汇总 5 次运行的 mean ± std
        def mean_std(key):
            vals = np.array([r[key] for r in all_runs], dtype=float)
            return vals.mean(), vals.std()

        # 主结果（fixed 0.5，因为 DVLOG dev 太小，校准不稳定）
        acc_m, acc_s = mean_std('acc')          # = fixed_acc
        p_m,   p_s   = mean_std('precision')    # = fixed_p
        r_m,   r_s   = mean_std('recall')       # = fixed_r
        f1_m,  f1_s  = mean_std('f1')           # = fixed_f1
        mf1_m, mf1_s = mean_std('macro_f1')     # = fixed_mf1
        # 校准阈值指标（附加分析）
        acc_cal_m, acc_cal_s = mean_std('acc_calibrated')
        mf1_cal_m, mf1_cal_s = mean_std('macro_f1_calibrated')
        thr_cal_m, thr_cal_s = mean_std('threshold_calibrated')

        print(f"\n{'*'*15} DVLOG OFFICIAL SPLIT ({len(seeds)} seeds) {'*'*15}")
        print(f"=== Primary results (fixed threshold=0.5) ===")
        print(f"ACC       : {acc_m:.2f}% (+/- {acc_s:.2f}%)")
        print(f"PRECISION : {p_m:.4f} (+/- {p_s:.4f})")
        print(f"RECALL    : {r_m:.4f} (+/- {r_s:.4f})")
        print(f"F1        : {f1_m:.4f} (+/- {f1_s:.4f})")
        print(f"MACRO-F1  : {mf1_m:.4f} (+/- {mf1_s:.4f})")
        print(f"--- Supplementary: calibrated threshold (reference only) ---")
        print(f"ACC@cal   : {acc_cal_m:.2f}% (+/- {acc_cal_s:.2f}%)")
        print(f"MACRO-F1@cal: {mf1_cal_m:.4f} (+/- {mf1_cal_s:.4f})")
        print(f"Threshold : {thr_cal_m:.3f} (+/- {thr_cal_s:.3f})")
        print(f"Calibration gain: Macro-F1 {mf1_cal_m - mf1_m:+.4f}")

    else:  # LMVD
        # ============================================================
        # LMVD: 10% holdout 测试集 + 90% 上做 10 折交叉验证
        # ============================================================
        seed = seeds[0]
        set_seed(seed)

        X = os.listdir(tcn)
        X.sort(key=lambda x: int(x.split(".")[0]))
        X = np.array(X)

        Y = []
        for i in X:
            file_csv = pd.read_csv(os.path.join(labelPath, (str(i.split('.npy')[0])+"_Depression.csv")))
            Y.append(int(file_csv.columns[0]))
        Y = np.array(Y)

        # 切分出 10% 固定测试集
        X_train_val_pool, X_test_holdout, Y_train_val_pool, Y_test_holdout = train_test_split(
            X, Y, test_size=0.10, stratify=Y, random_state=seed
        )

        print(f"[LMVD] Total: {len(X)}, Train-Val Pool: {len(X_train_val_pool)}, Fixed Test Set: {len(X_test_holdout)}")

        # 在 90% 数据上做 10 折交叉验证
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        metrics_history = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'macro_f1': [],
                           'acc_fixed': [], 'macro_f1_fixed': []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_pool, Y_train_val_pool)):
            print(f"\n{'='*20} Fold {fold+1} / 10 {'='*20}")

            X_train_fold = X_train_val_pool[train_idx]
            X_val_fold   = X_train_val_pool[val_idx]

            fold_results = train(tcn, mdnAudioPath, facePath,
                                 X_train_fold,      # 训练用 (81%)
                                 X_val_fold,         # 验证/早停用 (9%)
                                 X_test_holdout,     # 最终测试用 (固定 10%)
                                 labelPath,
                                 fold_name=f"Fold_{fold+1}",
                                 seed=seed)

            for key in metrics_history.keys():
                metrics_history[key].append(fold_results[key])

        # 统计最终结果（均值和标准差）
        print(f"\n{'*'*15} FINAL 10-FOLD CROSS VALIDATION RESULTS {'*'*15}")
        for key, values in metrics_history.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            if 'acc' in key:
                print(f"{key.upper():<15}: {mean_val:.2f}% (+/- {std_val:.2f}%)")
            else:
                print(f"{key.upper():<15}: {mean_val:.4f} (+/- {std_val:.4f})")
        # 诊断：阈值校准增益
        mf1_vals = np.array(metrics_history['macro_f1'])
        mf1_fix_vals = np.array(metrics_history['macro_f1_fixed'])
        print(f"Threshold calibration gain: Macro-F1 {mf1_vals.mean() - mf1_fix_vals.mean():+.4f}")