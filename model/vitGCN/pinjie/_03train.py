import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
# 引入 sklearn 进行列表切分
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
# 引入 余弦退火调度器
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
# 导入模块
from _01DatasetLoader import DepressionFaceDataset
from _02GCN_Transformer import AnatomicalGCN
from torch.optim.lr_scheduler import LambdaLR
import math
# --- 全局配置 ---
# DATASET_SELECT = "LMVD"  # <--- 切换数据集
# DATASET_SELECT = "DVLOG"  # <--- 切换数据集
DATASET_SELECT = "DVLOG"

LMVD_FEATURE_DIR = "data/LMVD_Feature/Video_landmarks_npy"
DVLOG_ROOT_DIR   = "data/dvlog-dataset/dvlog-dataset"
DVLOG_LABEL_CSV  = "data/dvlog-dataset/dvlog-dataset/labels.csv"

LR = 1e-4
EPOCHS = 300           
# T_TARGET = 150
# BATCH_SIZE = 32
# --- 核心修改：动态参数分配 ---
if DATASET_SELECT == "LMVD":
    T_TARGET = 500       # LMVD 较长，采样后保留 500 帧足以覆盖核心动作
    STRIDE = 5           # LMVD 原始帧率高，每 5 帧采 1 帧（大幅降维）
    BATCH_SIZE = 16      # 采样后序列依然较长，缩小 Batch 保护显存
    NUM_WORKERS = 8      # LMVD 文件多，开启更多进程并行预处理
    USE_ADAPTIVE_SAMPLING = True  # LMVD数据量大，可用自适应采样
else:
    T_TARGET = 300       # DVLOG 较短，300 帧左右即可
    STRIDE = 1           # 【关键】DVLOG 微表情重要，不跳帧！
    BATCH_SIZE = 32      # 序列短，可以跑大 Batch
    NUM_WORKERS = 4      # 保持适中并行度
    USE_ADAPTIVE_SAMPLING = False  # 【关键】DVLOG禁用自适应采样，保留微表情连续性
    
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {DEVICE}")

class EarlyStopping:
    def __init__(self, patience=30, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# 在 main 中初始化
early_stopping = EarlyStopping(patience=50) # 50 轮不升则停
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    实现带预热的余弦退火调度器
    """
    def lr_lambda(current_step):
        # 1. 线性预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 2. 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# 1. 数据准备函数 (Data Preparation)
def get_lmvd_data(feature_dir):
    """
    修改为扫描 .npy 文件
    """
    print(f"Scanning LMVD processed features from: {feature_dir} ...")
    # 修改后缀为 .npy
    npy_files = glob.glob(os.path.join(feature_dir, "*.npy")) 
    
    file_paths = []
    labels = []
    dep_count, norm_count = 0, 0
    
    for path in npy_files:
        filename = os.path.basename(path)
        try:
            # 提取 ID (例如 "001.npy" -> 1)
            file_id = int(filename.split('.')[0])
            
            # 标签判定逻辑保持不变
            is_depression = (1 <= file_id <= 601) or (1117 <= file_id <= 1423)
            is_normal = (602 <= file_id <= 1116) or (1425 <= file_id <= 1824)
            
            if is_depression:
                file_paths.append(path)
                labels.append(1)
                dep_count += 1
            elif is_normal:
                file_paths.append(path)
                labels.append(0)
                norm_count += 1
        except:
            continue
            
    print(f"[LMVD] Found {len(file_paths)} samples (Depression: {dep_count}, Normal: {norm_count})")
    return file_paths, labels

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, weight=None, reduction='mean'): # 增加 weight 参数
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.weight = weight # 保存权重
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        
        # 修正：在 nll_loss 中传入权重
        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(
            log_preds, target, weight=self.weight, reduction=self.reduction
        )

def get_dvlog_data(root_dir, label_csv):
    """
    解析 D-vlog 数据：处理嵌套文件夹结构 0/0_visual.npy
    """
    print(f"Scanning D-vlog features from: {root_dir} using labels: {label_csv} ...")
    
    if not os.path.exists(label_csv):
        print(f"Warning: Label file not found at {label_csv}. Skipping D-vlog.")
        return [], []

    df = pd.read_csv(label_csv)
    # CSV 列结构: index, label, duration, gender, fold
    
    file_paths = []
    labels = []
    
    missing_count = 0
    
    for _, row in df.iterrows():
        try:
            # 获取索引 ID (例如 0)
            idx = int(row[0]) # 第一列是 index
            label_str = str(row[1]).strip().lower() # 假设第二列是 label
            
            # 标签转换
            if label_str == 'depression':
                label = 1
            elif label_str == 'normal':
                label = 0
            else:
                continue # 跳过其他标签
            
            # 关键：构建路径 ../data/d-vlog/{id}/{id}_visual.npy
            # 注意 idx 需要转字符串
            folder_name = str(idx)
            file_name = f"{idx}_visual.npy"
            npy_path = os.path.join(root_dir, folder_name, file_name)
            
            # 检查文件是否存在
            if os.path.exists(npy_path):
                file_paths.append(npy_path)
                labels.append(label)
            else:
                # 某些数据集可能缺失部分文件
                missing_count += 1
                
        except Exception as e:
            print(f"Error parsing row: {row}. Error: {e}")
            continue

    print(f"[D-vlog] Found {len(file_paths)} samples. (Missing files: {missing_count})")
    return file_paths, labels

# 2. Collate Function (处理 Batch堆叠)
def my_collate_fn(batch):
    """
    将 list of (dict, label) 转换为 (batched_dict, batched_labels)
    """
    # 1. 收集标签
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    # 2. 收集 Regions
    # batch[0][0] 是 regions 字典: {'mouth': np.array, ...}
    batch_regions = {}
    keys = batch[0][0].keys() # ['mouth', 'nose', ...]
    
    for k in keys:
        # 收集当前 key 对应的所有样本 numpy array
        arrays = [item[0][k] for item in batch] 
        # 转换为 Tensor 并堆叠 -> (Batch, T, N, 2)
        # Dataset 里的 pad_or_crop 已经保证了 T 一致
        batch_regions[k] = torch.stack([torch.from_numpy(a) for a in arrays])
        
    actual_lens = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return batch_regions, labels, actual_lens

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # 将 alpha 转换为 Tensor（如果是列表的话）
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 使用 log_softmax 提高稳定性
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)
        
        # 针对目标的 log_p
        log_p_target = log_p.gather(1, targets.view(-1, 1))
        p_target = p.gather(1, targets.view(-1, 1))
        
        eps = 1e-8
        
        # 处理 alpha：如果是 Tensor，根据 targets 索引获取对应值
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(targets.device)[targets].view(-1, 1)
        else:
            alpha_t = self.alpha
        
        loss = -alpha_t * (1 - p_target + eps)**self.gamma * (log_p_target + eps)
        
        if self.weight is not None:
            # 加上类别权重
            loss = loss * self.weight[targets].view(-1, 1)

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

# --- 全局配置区补充 ---
K_FOLDS = 10  # 明确定义 10 折

if __name__ == '__main__':
    # 1. 加载数据
    if DATASET_SELECT == "LMVD":
        paths, labels = get_lmvd_data("data/LMVD_Feature/Video_landmarks_npy")
        curr_mode = 'lmvd'
    else:
        paths, labels = get_dvlog_data("data/dvlog-dataset/dvlog-dataset", "data/dvlog-dataset/dvlog-dataset/labels.csv")
        curr_mode = 'dvlog'

    # 2. 固定切出 10% 独立测试集 (绝对不参与训练和验证)
    paths_90, test_p, labels_90, test_l = train_test_split(
        paths, labels, test_size=0.1, random_state=42, stratify=labels
    )
    paths_90, labels_90 = np.array(paths_90), np.array(labels_90)
    
    #提前准备好独立测试集 Loader
    test_ds = DepressionFaceDataset(
        test_p, test_l, 
        mode=curr_mode, 
        T_target=T_TARGET,
        stride=STRIDE, 
        augment=False,
        adaptive_sampling=False  # 测试集不用自适应采样
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate_fn)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results_f1 = [] # 记录每一折的最佳 F1

    print(f"\n开始 {K_FOLDS} 折交叉验证流程...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths_90, labels_90)):
        print(f"\n{'='*20} Fold [{fold+1}/{K_FOLDS}] {'='*20}")
        
        # 准备数据加载器 (逻辑保持不变)
        curr_train_p, curr_val_p = paths_90[train_idx], paths_90[val_idx]
        curr_train_l, curr_val_l = labels_90[train_idx], labels_90[val_idx]
        
        train_ds = DepressionFaceDataset(
            curr_train_p, 
            curr_train_l, 
            mode=curr_mode, 
            T_target=T_TARGET, 
            stride=STRIDE,
            augment=True,
            adaptive_sampling=USE_ADAPTIVE_SAMPLING  # 【关键】根据数据集动态控制
        )
        val_ds = DepressionFaceDataset(
            curr_val_p, curr_val_l, 
            mode=curr_mode, 
            T_target=T_TARGET,
            stride=STRIDE, 
            augment=False,
            adaptive_sampling=False  # 验证集不用自适应采样
        )
        
        # 采样器平衡类别
        t_tensor = torch.tensor(curr_train_l)
        counts = torch.tensor([(t_tensor == t).sum() for t in torch.unique(t_tensor)])
        weights = 1. / counts.float()
        t_tensor = torch.tensor(curr_train_l)
        counts = torch.tensor([(t_tensor == t).sum() for t in torch.unique(t_tensor)])
        class_weights = 1. / counts.float()  # 类别反比例分布
        sample_weights = [class_weights[t] for t in curr_train_l]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_ds, 
            batch_size=BATCH_SIZE, 
            sampler=sampler,  # 替代 shuffle
            collate_fn=my_collate_fn
        )
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate_fn)

        # 初始化模型与优化器
        model = AnatomicalGCN(out_dim=32, nhead=4, num_classes=2).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        # 【策略四】改进学习率调度：增加warmup + 使用ReduceLROnPlateau
        num_training_steps = EPOCHS * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        warmup_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, verbose=True
        )
        
        # 【策略三】使用标准CrossEntropyLoss + 类别权重（避免FocalLoss过度偏向）
        class_weights = torch.tensor([0.55, 0.45], dtype=torch.float32).to(DEVICE)  # 轻微平衡
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        early_stopping = EarlyStopping(patience=50)
        best_fold_f1 = 0.0
        os.makedirs(DATASET_SELECT, exist_ok=True)
        fold_model_path = os.path.join(DATASET_SELECT, f"best_model_fold_{fold+1}.pth")

        for epoch in range(EPOCHS):
            # --- Training ---
            model.train()
            for regions, targets, actual_lens in train_loader:
                regions = {k: v.to(DEVICE) for k, v in regions.items()}
                targets, actual_lens = targets.to(DEVICE), actual_lens.to(DEVICE)
                optimizer.zero_grad()
                logits = model(regions, actual_lens)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                warmup_scheduler.step()  # 使用warmup调度器

            # --- Validation ---
            model.eval()
            val_correct, val_total = 0, 0
            all_val_preds, all_val_targets = [], []
            total_dep_preds, total_norm_preds = 0, 0

            with torch.no_grad():
                for regions, targets, actual_lens in val_loader:
                    regions = {k: v.to(DEVICE) for k, v in regions.items()}
                    targets, actual_lens = targets.to(DEVICE), actual_lens.to(DEVICE)
                    logits = model(regions, actual_lens)
                    _, pred = torch.max(logits, 1)
                    
                    all_val_preds.extend(pred.cpu().numpy())
                    all_val_targets.extend(targets.cpu().numpy())
                    total_dep_preds += (pred == 1).sum().item()
                    total_norm_preds += (pred == 0).sum().item()
                    val_correct += (pred == targets).sum().item()
                    val_total += targets.size(0)

            # 计算指标
            epoch_val_acc = 100.0 * val_correct / val_total
            epoch_precision = precision_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
            epoch_recall = recall_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
            epoch_f1 = f1_score(all_val_targets, all_val_preds, average='macro')
            
            # 【策略四】在验证后调用ReduceLROnPlateau
            plateau_scheduler.step(epoch_f1)
            
            # 仅打印简洁日志，不在这里跑全量测试
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}] Val F1: {epoch_f1:.4f} | Acc: {epoch_val_acc:.2f}% | "
                      f"Precision: {epoch_precision:.4f} | Recall: {epoch_recall:.4f} | "
                      f"LR: {current_lr:.6f} | Dist: D{total_dep_preds}/N{total_norm_preds}")
            
            # 保存该折最佳模型
            if epoch_f1 > best_fold_f1:
                best_fold_f1 = epoch_f1
                torch.save(model.state_dict(), fold_model_path)

            # 【修正】早停法也观察 F1，保持一致性
            early_stopping(epoch_f1)
            if early_stopping.early_stop: break
            
        # 计算最终折内验证的完整指标
        fold_precision = precision_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
        fold_recall = recall_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
        fold_accuracy = 100.0 * accuracy_score(all_val_targets, all_val_preds)
        print(f"Fold {fold+1} 结束. "
            f"Final Metrics | Acc: {fold_accuracy:.2f}%, Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}, F1: {best_fold_f1:.4f}")
        fold_results_f1.append(best_fold_f1)

    # ================================================================
    # 7. 期刊级汇总评估 (这一块必须放在所有 Fold 循环结束后)
    # ================================================================
    print("\n" + "="*50)
    print("所有折训练结束，正在进行独立测试集汇总评估...")
    print("="*50)
    
    test_metrics = {'acc': [], 'f1': [], 'recall': [], 'precision': []}

    for f in range(1, K_FOLDS + 1):
        model.load_state_dict(torch.load(os.path.join(DATASET_SELECT, f"best_model_fold_{f}.pth")))
        model.eval()
        f_preds, f_targets = [], []
        with torch.no_grad():
            for regions, targets, actual_lens in test_loader:
                regions = {k: v.to(DEVICE) for k, v in regions.items()}
                logits = model(regions, actual_lens.to(DEVICE))
                _, preds = torch.max(logits, 1)
                f_preds.extend(preds.cpu().numpy())
                f_targets.extend(targets.numpy())
        
        test_metrics['acc'].append(accuracy_score(f_targets, f_preds))
        test_metrics['f1'].append(f1_score(f_targets, f_preds, average='macro'))
        test_metrics['recall'].append(recall_score(f_targets, f_preds))
        test_metrics['precision'].append(precision_score(f_targets, f_preds, zero_division=0))

    # 打印最终统计数据
    print("\n******* FINAL TEST ********")
    # 逐折打印每折的最终结果
    for f in range(1, K_FOLDS + 1):
        print(f"--- Fold {f:02d} Final Metrics ---\n"
            f"Acc: {test_metrics['acc'][f-1]:.2f}, "
            f"Precision: {test_metrics['precision'][f-1]:.4f}, "
            f"Recall: {test_metrics['recall'][f-1]:.4f}, "
            f"F1: {test_metrics['f1'][f-1]:.4f}")

    # 打印汇总指标（平均 + 标准差）
    print("\n--- Final Test Set Mean Metrics ---")
    print(f"Mean Accuracy:  {np.mean(test_metrics['acc']):.4f} ± {np.std(test_metrics['acc']):.4f}")
    print(f"Mean Macro-F1:  {np.mean(test_metrics['f1']):.4f} ± {np.std(test_metrics['f1']):.4f}")
    print(f"Mean Recall: {np.mean(test_metrics['recall']):.4f} ± {np.std(test_metrics['recall']):.4f}")
    print(f"Mean Precision:  {np.mean(test_metrics['precision']):.4f} ± {np.std(test_metrics['precision']):.4f}")