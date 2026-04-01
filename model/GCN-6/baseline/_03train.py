import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import pandas as pd
import os
import glob
import numpy as np

# ---------------------------------------------------------
# 导入自定义模块
# ---------------------------------------------------------
# 确保 01DatasetLoader.py 和 02GCN_Transformer.py 在同一目录下
from _01DatasetLoader import DepressionFaceDataset
from _02GCN_Transformer import AnatomicalGCN

# ---------------------------------------------------------
# 全局配置 (Configuration)
# ---------------------------------------------------------
# 路径配置 (根据你的描述修改)
LMVD_FEATURE_DIR = "data/LMVD_Feature/Video_feature"
DVLOG_ROOT_DIR = "data/dvlog-dataset"
# D-vlog 的标签文件路径如下，如果不同请修改
DVLOG_LABEL_CSV = "data/dvlog-dataset/labels.csv" 

# 训练参数
BATCH_SIZE = 8          # 显存较小(如6G-8G)建议8-16
LR = 1e-4               # 学习率
EPOCHS = 50             # 训练轮数
T_TARGET = 300          # 时间帧统一长度 (不够补零，超长截断)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running on device: {DEVICE}")

# ---------------------------------------------------------
# 1. 数据准备函数 (Data Preparation)
# ---------------------------------------------------------

def get_lmvd_data(feature_dir):
    """
    解析 LMVD 数据：基于文件名 ID 范围判断标签
    """
    print(f"Scanning LMVD features from: {feature_dir} ...")
    csv_files = glob.glob(os.path.join(feature_dir, "*.csv"))
    
    file_paths = []
    labels = []
    
    # 计数器
    dep_count = 0
    norm_count = 0
    
    for path in csv_files:
        filename = os.path.basename(path)
        try:
            # 文件名可能是 "001.csv"，提取 "001" -> 1
            file_id = int(filename.split('.')[0])
            
            # 根据提供的 ID 规则判断
            # Depression: 001-601, 1117-1423
            is_depression = (1 <= file_id <= 601) or (1117 <= file_id <= 1423)
            # Normal: 0602-1116, 1425-1824
            is_normal = (602 <= file_id <= 1116) or (1425 <= file_id <= 1824)
            
            if is_depression:
                file_paths.append(path)
                labels.append(1)
                dep_count += 1
            elif is_normal:
                file_paths.append(path)
                labels.append(0)
                norm_count += 1
                
        except ValueError:
            print(f"Skipping invalid filename: {filename}")
            continue
            
    print(f"[LMVD] Found {len(file_paths)} samples (Depression: {dep_count}, Normal: {norm_count})")
    return file_paths, labels

def get_dvlog_data(root_dir, label_csv):
    """
    解析 D-vlog 数据：处理嵌套文件夹结构 0/0_visual.npy
    """
    print(f"Scanning D-vlog features from: {root_dir} using labels: {label_csv} ...")
    
    if not os.path.exists(label_csv):
        print(f"Warning: Label file not found at {label_csv}. Skipping D-vlog.")
        return [], []

    df = pd.read_csv(label_csv)
    # 假设 CSV 列结构: index, label, duration, gender, fold
    
    file_paths = []
    labels = []
    
    missing_count = 0
    
    for _, row in df.iterrows():
        try:
            # 获取索引 ID (例如 0)
            idx = int(row[0]) # 假设第一列是 index
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

# ---------------------------------------------------------
# 2. Collate Function (处理 Batch堆叠)
# ---------------------------------------------------------
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
        # 注意：这里假设 Dataset 里的 pad_or_crop 已经保证了 T 一致
        batch_regions[k] = torch.stack([torch.from_numpy(a) for a in arrays])
        
    return batch_regions, labels

# ---------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------
if __name__ == '__main__':
    # --- Step 1: 获取数据路径 ---
    lmvd_paths, lmvd_labels = get_lmvd_data(LMVD_FEATURE_DIR)
    dvlog_paths, dvlog_labels = get_dvlog_data(DVLOG_ROOT_DIR, DVLOG_LABEL_CSV)
    
    # --- Step 2: 创建数据集实例 ---
    # 定义 LMVD 数据集
    ds_lmvd = DepressionFaceDataset(
        data_list=lmvd_paths, 
        labels=lmvd_labels, 
        mode='lmvd', 
        T_target=T_TARGET,
        interpolate_invalid=True # 开启插值修复无效帧
    )
    
    # 定义 D-vlog 数据集
    ds_dvlog = DepressionFaceDataset(
        data_list=dvlog_paths, 
        labels=dvlog_labels, 
        mode='dvlog', 
        T_target=T_TARGET
    )
    
    # 合并数据集 (混合训练)
    # 如果只想用其中一个，可以只用 ds_lmvd 或 ds_dvlog
    if len(ds_dvlog) > 0:
        full_dataset = ConcatDataset([ds_lmvd, ds_dvlog])
        print(f"Combined Dataset Size: {len(full_dataset)}")
    else:
        full_dataset = ds_lmvd
        print("Using LMVD only (D-vlog empty or not found).")
    
    # --- Step 3: 划分训练集和验证集 (80% / 20%) ---
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])
    
    # --- Step 4: 创建 DataLoader ---
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,        # Windows下如果报错改为0
        collate_fn=my_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        collate_fn=my_collate_fn,
        pin_memory=True
    )
    
    # --- Step 5: 初始化模型、优化器、损失函数 ---
    model = AnatomicalGCN(out_dim=64, num_classes=2).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5) # 加一点正则化
    criterion = nn.CrossEntropyLoss()
    
    # --- Step 6: 训练循环 ---
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        # === Training Phase ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (regions, targets) in enumerate(train_loader):
            # 数据移至 GPU
            targets = targets.to(DEVICE)
            # regions 字典在 model.forward 内部会自动处理 device，
            # 但为了保险，也可以在这里手动处理，不过 model 代码里已经写了防御性逻辑，这里不需要动
            
            # Forward
            optimizer.zero_grad()
            logits = model(regions) # (B, 2)
            
            # Loss
            loss = criterion(logits, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(logits, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        epoch_loss = train_loss / train_total
        epoch_acc = 100.0 * train_correct / train_total
        
        # === Validation Phase ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for regions, targets in val_loader:
                targets = targets.to(DEVICE)
                logits = model(regions)
                loss = criterion(logits, targets)
                
                val_loss += loss.item() * targets.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")
        
    print("Training Finished!")
    # 保存模型
    torch.save(model.state_dict(), "depression_model_final.pth")
    print("Model saved to depression_model_final.pth")