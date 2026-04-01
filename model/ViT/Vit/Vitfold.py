import numpy as np
from re import T
import torch.nn.functional as F
import torch
import logging
from kfoldLoader import MyDataLoader 
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
from math import cos
from tqdm import tqdm
import torch.nn as nn
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from Vitmodel import ViT
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from numpy.random import beta
import matplotlib.pyplot as plt
""" 实现K折交叉验证的全部逻辑，包括
数据集索引划分、数据加载、模型初始化、训练、测试、性能指标记录和结果保存。 """
# 视频特征路径
VIDEO_FEATURE_PATH = "data/LMVD_Feature/tcnfeature" 
# 音频特征路径
AUDIO_FEATURE_PATH = "data/LMVD_Feature/Audio_feature" 
# 标签路径
LABEL_PATH = "label/label"

# ----------------- 模型的超参数设置 -----------------
T = 915            # 序列长度（来自 tcnfeature.py 中的 915）
D_VIDEO = 171      # 视频特征维度
D_AUDIO = 128     

D_EMB = 256        # 嵌入维度 (dim)
HEADS = 4          # Attention 头的数量，每个头处理的维度为D_EMB / HEADS = 64
PATCH_SIZE = 15    # Patch 大小 
DEPTH = 4          # Transformer深度/层数 (修改为 8)
DIM_MLP = 512     # FFN 的隐藏层维度 (dim_mlp)


lr = 5e-6       #初始学习率，控制参数更新的步长
epochSize = 200 #训练总轮数（Epoch），即整个训练集被模型学习的次数
warmupEpoch = 20        #学习率预热轮数。在预热阶段，学习率从 0 线性增长到初始学习率，避免初始高学习率对模型的冲击。
testRows = 1            #测试间隔，即每训练 1 个 Epoch 后在验证集上评估一次模型性能。
schedule = 'cosine'
classes = ['Normal','Depression']   #分类任务的类别标签
ps = []
rs = []
f1s = []
totals = []

total_pre = []
total_label = []

# ----------------- 日志和保存路径的修复 -----------------
tim = time.strftime('%m_%d__%H_%M', time.localtime())
# 实际的日志文件路径
filepath = os.path.join('logs', str(tim)) 
savePath1 = os.path.join('models', str(tim))
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
def seed_everything(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        
        # 计算平滑项 (即均匀分布的 CrossEntropy)
        smooth_loss = -logprobs.mean(dim=-1)
        
        # 加权融合
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class AffectnetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset):
        print('initial balance sampler ...')

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        expression_count = [0] * 2
        for idx in self.indices:
            label = dataset.labels[idx]
            expression_count[int(label)] += 1

        self.weights = torch.zeros(self.num_samples)
        for idx in self.indices:
            label = dataset.labels[idx]
            self.weights[idx] = 1. / (expression_count[int(label)]+ 1e-6)

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

def train(VideoPath, AudioPath, X_train, X_dev, X_final_test, labelPath, fold_name):
    mytop = 0
    topacc = 0

    patience = 80           # 忍耐 50 个 epoch
    counter = 0             # 计数器归零
    best_loss = float('inf') # 初始化最佳 Loss 为无穷大

    # 1. 加载训练集 (80%)
    trainSet = MyDataLoader(X_train, VideoPath, AudioPath, labelPath, mode='train')
    # 真正使用采样器，解决平衡问题
    sampler = AffectnetSampler(trainSet) 
    trainLoader = DataLoader(trainSet, batch_size=15, sampler=sampler) # 使用 sampler 时不可设置 shuffle=True
    
    # 2. 加载验证集 (10%) -> 这里的变量名原来叫 X_test，现在对应 X_dev
    devSet = MyDataLoader(X_dev, VideoPath, AudioPath, labelPath, mode='test')
    devLoader = DataLoader(devSet, batch_size=4, shuffle=False)
    
    # 3. 加载最终测试集 (10%) -> 新增
    finalTestSet = MyDataLoader(X_final_test, VideoPath, AudioPath, labelPath, mode='test')
    finalTestLoader = DataLoader(finalTestSet, batch_size=4, shuffle=False)

    print("DataLoaders Ready: Train={}, Dev={}, Test={}".format(
        len(trainLoader), len(devLoader), len(finalTestLoader)))

    # 创建模型并移动到 device（单 GPU 环境）
    D_PROJECTION = D_EMB // 2 # 256 // 2 = 128
    FEATURE_DIM_AFTER_CONCAT = D_PROJECTION * 2 # 128 * 2 = 256

    if torch.cuda.is_available():
        # 拼接后的特征维度: (186 + 128) = 314. 这是 PatchEmbdding 的 channel (c) 维度
        FEATURE_DIM_AFTER_CONCAT = 256
        model = ViT(
            spectra_size=T, # T=915, 序列长度
            patch_size=PATCH_SIZE, # 15
            num_classes=2,
            dim=D_EMB, # dim 修正为 256
            depth=DEPTH, 
            heads=HEADS,       
            dim_mlp=DIM_MLP,  # dim_mlp 修正为 1024
            # 修复: 这里的 channel 必须是特征融合后的维度256
            channel=FEATURE_DIM_AFTER_CONCAT, 
            # dim_head 必须满足 dim / heads = dim_head, 即 256 / 8 = 32
            dim_head=D_EMB // HEADS, # 32 
            dropout=0.3 # 优化：降低 dropout
        ).to(device)

    lossFunc = LabelSmoothingCrossEntropy(smoothing=0.05).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr,
                                    betas=(0.9,0.999),
                                    eps=1e-8,
                                    weight_decay=1e-4,
                                    amsgrad=False
                                    )

    train_steps = len(trainLoader)*epochSize
    warmup_steps = len(trainLoader)*warmupEpoch
    target_steps = len(trainLoader)*epochSize
    
    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=target_steps)

    logging.info('The {} training begins!'.format(fold_name))
    savePath = os.path.join(str(savePath1), str(fold_name))
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    
    for epoch in range(1, epochSize):
        # 1. 修正进度条包装逻辑，使其包含 enumerate
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        traloss_one = 0
        correct = 0
        total = 0
        
        model.train()
        # 2. 修正循环头，定义 batch_idx 并从 loop 中取值
        for batch_idx, (videoData, audioData, label) in loop:
            videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)

            output = model(videoData, audioData)
            traLoss = lossFunc(output, label.long())
            
            optimizer.zero_grad()
            traLoss.backward()
            #梯度裁剪：防止梯度爆炸导致 nan
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()

            # 3. 修正损失累加逻辑
            traloss_one += traLoss.item() # 将当前 batch 的 loss 累加到总和
            
            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()

            # 4. 修正进度条显示，计算当前的平均 Loss (traloss_one / 已处理的 batch 数)
            loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss = traloss_one / (batch_idx + 1), acc = f"{100.0 * correct / total:.2f}%")

        # 5. 修正日志记录中的 batch_idx 使用
        logging.info('EpochSize: {}, Train batch: {}, Loss:{}, Acc:{}%'.format(
            epoch, batch_idx + 1, traloss_one / len(trainLoader), 100.0 * correct / total))

        if epoch-warmupEpoch >= 0 and epoch % testRows == 0:
            # 1. 统一初始化变量名，防止 NameError
            val_labels_collect = []
            val_preds_collect = []
            
            model.eval()
            print("*******dev********")
            loop_dev = tqdm(enumerate(devLoader), total=len(devLoader))
            loss_one = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (videoData, audioData, label) in loop_dev:
                    videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)
                    
                    devOutput = model(videoData, audioData)
                    loss = lossFunc(devOutput, label.long())
                    loss_one += loss.item() 
                    
                    _, predicted = torch.max(devOutput.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()
                    
                    # 2. 收集数据
                    val_labels_collect.extend(label.data.cpu().tolist())
                    val_preds_collect.extend(predicted.cpu().tolist())
            
            acc = 100.0 * correct / total
            # 3. 计算指标
            f1score = f1_score(val_labels_collect, val_preds_collect, average='weighted')
            p = precision_score(val_labels_collect, val_preds_collect, average='weighted')
            r = recall_score(val_labels_collect, val_preds_collect, average='weighted')
            logging.info('precision:{}'.format(p))
            logging.info('recall:{}'.format(r))
            logging.info('f1:{}'.format(f1score))

            logging.debug('Dev epoch:{}, Loss:{}, Acc:{}%'.format(epoch,loss_one/len(devLoader), acc))
            loop.set_description(f'__Dev Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=loss)
            print('Dev epoch:{}, Loss:{},Acc:{}%'.format(epoch,loss_one/len(devLoader),acc))
            if acc > mytop:
                mytop = max(acc,mytop)
                top_p, top_r, top_f1 = p, r, f1score
                top_pre, top_label = val_preds_collect, val_labels_collect

                # 2. 触发保存逻辑（只要是目前最好的，就保存）
                logging.info(f"New Best Accuracy: {acc:.2f}%. Saving model...")
                checkpoint = {
                    'net': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'epoch': epoch, 
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(savePath, f"ViT_best_{fold_name}.pth"))
                        
            # 插入早停逻辑 (开始)
            # 计算当前 epoch 验证集的平均 loss
            avg_dev_loss = loss_one / len(devLoader)
            
            # 打印一下，确认逻辑在运行
            # print(f'Validation Loss: {avg_dev_loss:.4f}') 

            if avg_dev_loss < best_loss:
                best_loss = avg_dev_loss
                counter = 0  # 如果 loss 创新低，重置计数器
                # 这里通常不需要额外 save，因为后面有 acc > topacc 的保存逻辑
                # 但如果你想基于 Loss 保存最佳模型，可以在这里 save
            else:
                counter += 1
                logging.info(f'EarlyStopping counter: {counter} out of {patience}')
                
                if counter >= patience:
                    logging.info("Early stopping triggered! Stop training.")
                    print("Early stopping triggered!")
                    break  # <--- 关键：跳出最外层的 epoch 循环
            #  插入早停逻辑 (结束) 
    
    #加载最佳权重逻辑
    print("Training Finished. Loading Best Model for Final Testing...")
    best_model_path = os.path.join(savePath, f"ViT_best_{fold_name}.pth")
    
    if os.path.exists(best_model_path):
        # 加载 checkpoint
        checkpoint = torch.load(best_model_path)
        # 覆盖当前模型的参数
        model.load_state_dict(checkpoint['net'])
        print(f"Successfully loaded best model (Acc: {mytop}%) from {best_model_path}")
    else:
        print("Warning: No best model found! Using model from last epoch.")

    model.eval()
    test_correct = 0
    test_total = 0
    test_label = []
    test_pre = []
    
    print("******* FINAL TEST ********")
    with torch.no_grad():
        for batch_idx, (videoData, audioData, label) in tqdm(enumerate(finalTestLoader)):
            if torch.cuda.is_available():
                videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)
            
            output = model(videoData, audioData)
            _, predicted = torch.max(output.data, 1)
            
            test_total += label.size(0)
            test_correct += predicted.eq(label.data).cpu().sum()
            
            test_label += label.data.tolist()
            test_pre += predicted.tolist()
            
    final_acc = 100.0 * test_correct / test_total
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    
    # 保存 Confusion Matrix
    plot_confusion_matrix(test_label, test_pre, [0, 1], 
                          savename=filepath + '/final_test_confusion_matrix.png',
                          title=f'Final Test Acc: {final_acc:.2f}%')
                          
    return test_label, test_pre

def count(string):
    dig = sum(1 for char in string if char.isdigit())
    return dig

def ensemble_evaluate(model_paths, test_loader, device):
    """
    加载多个模型，对同一个测试集进行预测，取平均概率
    """
    print(f"\n================ START ENSEMBLE ({len(model_paths)} Models) ================")
    
    # 1. 加载所有模型
    models = []
    # 获取必要的参数用于初始化模型架构 (这里复用全局变量，或者你应该从 args 传进来)
    # 注意：这里必须保证模型架构参数与训练时完全一致
    D_PROJECTION = D_EMB // 2
    FEATURE_DIM_AFTER_CONCAT = D_PROJECTION * 2
    
    for path in model_paths:
        print(f"Loading model from: {path}")
        # 初始化空模型
        model = ViT(
            spectra_size=T, patch_size=PATCH_SIZE, num_classes=2, dim=D_EMB, 
            depth=DEPTH, heads=HEADS, dim_mlp=DIM_MLP, 
            channel=FEATURE_DIM_AFTER_CONCAT, dim_head=D_EMB // HEADS, video_dim=D_VIDEO,audio_dim=D_AUDIO,
            dropout=0.3
        ).to(device)
        
        # 加载权重
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['net'])
        model.eval()
        models.append(model)

    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    # 2. 开始预测
    with torch.no_grad():
        for batch_idx, (videoData, audioData, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if torch.cuda.is_available():
                videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)

            # 累加所有模型的输出概率 (Logits)
            ensemble_output = torch.zeros(label.size(0), 2).to(device)
            
            for model in models:
                output = model(videoData, audioData)
                # 将 Logits 转为概率 (Softmax)
                prob = F.softmax(output, dim=1)
                ensemble_output += prob

            # 取平均 (其实不除以 len 也可以，因为 argmax 结果一样)
            ensemble_output /= len(models)

            # 3. 最终决策
            _, predicted = torch.max(ensemble_output.data, 1)
            
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()
            
            all_labels += label.data.tolist()
            all_preds += predicted.tolist()

    acc = 100.0 * correct / total
    print(f"\n★ Ensemble Final Accuracy: {acc:.2f}% ")
    
    # 画混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, [0, 1], 
                          savename=filepath + '/ENSEMBLE_confusion_matrix.png',
                          title=f'Ensemble Acc: {acc:.2f}%')
    return acc

if __name__ == '__main__':
    # 1. 初始化设置
    seed_everything(42)
    
    # 2. 准备文件列表、分组 ID 和 标签
    X_files = np.array([f for f in os.listdir(VIDEO_FEATURE_PATH) if f.endswith('.npy')])
    X_files.sort()

    groups = []
    Y = []
    for f in X_files:
        subject_id = int(f.split('.')[0].split('_')[0])
        groups.append(subject_id)
        # 根据提供的范围打标签
        if (1 <= subject_id <= 601) or (1117 <= subject_id <= 1423):
            Y.append(1) # Depression
        else:
            Y.append(0) # Normal
    
    groups = np.array(groups)
    Y = np.array(Y)

    # 3. 10折交叉验证循环
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    
    all_fold_f1 = []
    all_fold_acc = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_files, Y, groups=groups)):
        print(f"\n>>>>>>> FOLD {fold+1} / 10 <<<<<<<")
        X_train_fold, X_val_fold = X_files[train_idx], X_files[val_idx]

        # 4. 调用单折训练逻辑
        #train 函数返回的是最终测试的 (y_true, y_pred)
        test_labels, test_preds = train(
            VIDEO_FEATURE_PATH, AUDIO_FEATURE_PATH, 
            X_train_fold, X_val_fold, X_val_fold, 
            LABEL_PATH, f"Fold_{fold}"
        )
        
        # 5. 计算并存储指标
        f1 = f1_score(test_labels, test_preds, average='weighted')
        acc = accuracy_score(test_labels, test_preds)
        
        all_fold_f1.append(f1)
        all_fold_acc.append(acc)
        print(f"Fold {fold+1} Finished. Acc: {acc:.4f}, F1: {f1:.4f}")

    # 6. 输出汇总统计 (顶会标准：均值 ± 标准差)
    print("\n" + "="*30)
    print(f"Final 10-Fold Results:")
    print(f"Accuracy: {np.mean(all_fold_acc):.4f} ± {np.std(all_fold_acc):.4f}")
    print(f"F1-Score: {np.mean(all_fold_f1):.4f} ± {np.std(all_fold_f1):.4f}")
    print("="*30)