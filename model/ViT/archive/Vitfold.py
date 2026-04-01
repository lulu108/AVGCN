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
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix
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
HEADS = 8          # Attention 头的数量
PATCH_SIZE = 15    # Patch 大小 
DEPTH = 4          # Transformer 深度/层数 (修改为 8)
DIM_MLP = 512     # FFN 的隐藏层维度 (dim_mlp)


lr = 3e-5
epochSize = 200
warmupEpoch = 10
testRows = 1
schedule = 'cosine'
classes = ['Normal','Depression']
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

def train(VideoPath, AudioPath, X_train, X_dev, X_final_test, labelPath, fold_name):
    mytop = 0
    topacc = 0
    top_p=0
    top_r=0
    top_f1=0
    top_pre=[]
    top_label=[]

    patience = 50           # 忍耐 50 个 epoch
    counter = 0             # 计数器归零
    best_loss = float('inf') # 初始化最佳 Loss 为无穷大

    # 1. 加载训练集 (80%)
    trainSet = MyDataLoader(X_train, VideoPath, AudioPath, labelPath, mode='train')
    trainLoader = DataLoader(trainSet, batch_size=8, shuffle=True)
    
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
            depth=DEPTH, # depth 提升至 8 或 12
            heads=HEADS,       # heads 设置为 8
            dim_mlp=DIM_MLP,  # dim_mlp 修正为 1024
            # 修复: 这里的 channel 必须是特征融合后的维度256
            channel=FEATURE_DIM_AFTER_CONCAT, 
            # dim_head 必须满足 dim / heads = dim_head, 即 256 / 8 = 32
            dim_head=D_EMB // HEADS, # 32 
            dropout=0.35 # 优化：降低 dropout
        ).to(device)

    lossFunc = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr,
                                    weight_decay=1e-2
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
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        traloss_one = 0
        correct = 0
        total = 0
        lable1 = []
        pre1 = []
        
        model.train()
        for batch_idx, (videoData, audioData, label) in loop:
            if torch.cuda.is_available():
                videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)

            # 1. 生成 Mixup 系数 lam (0到1之间)
            alpha = 0.3
            lam = beta(alpha, alpha)
            
            # 2. 生成随机打乱的索引
            index = torch.randperm(videoData.size(0)).to(device)
            
            # 3. 混合输入 (Mix Data)
            mixed_video = lam * videoData + (1 - lam) * videoData[index, :]
            mixed_audio = lam * audioData + (1 - lam) * audioData[index, :]
            
            # 4. 混合标签 (Mix Labels)
            label_a, label_b = label, label[index]
            
            # 5. 前向传播
            output = model(mixed_video, mixed_audio)
            
            # 6. 计算 Mixup Loss
            # Loss = lam * Loss(label_a) + (1-lam) * Loss(label_b)
            traLoss = lam * lossFunc(output, label_a.long()) + (1 - lam) * lossFunc(output, label_b.long())
            traloss_one += traLoss
            optimizer.zero_grad()
            traLoss.backward()

            # 梯度裁剪：限制梯度范数，防止数值爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #

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
            
            model.eval()
            print("*******dev********")
            loop = tqdm(enumerate(devLoader), total=len(devLoader))
            with torch.no_grad():
                loss_one = 0
                for batch_idx, (videoData, audioData, label) in loop:
                    if torch.cuda.is_available():
                        videoData, audioData,label = videoData.to(device), audioData.to(device),label.to(device)
                    devOutput = model(videoData, audioData)
                    loss = lossFunc(devOutput, label.long())
                    loss_one += loss
                    train_num+=label.size(0)
                    
                    _, predicted = torch.max(devOutput.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()
                    
                    label2.append(label.data)
                    pre2.append(predicted)
                    
                    lable1 += label.data.tolist()
                    pre1 += predicted.tolist()
            
            acc = 100.0*correct/total
            lable1 = np.array(lable1)
            pre1 = np.array(pre1)

            p = precision_score(lable1, pre1, average='weighted')
            r = recall_score(lable1, pre1, average='weighted')
            f1score = f1_score(lable1, pre1, average='weighted')
            logging.info('precision:{}'.format(p))
            logging.info('recall:{}'.format(r))
            logging.info('f1:{}'.format(f1score))

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
            
            # ================== 插入早停逻辑 (开始) ==================
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
            # ================== 插入早停逻辑 (结束) ==================

            if acc > topacc:
                topacc = max(acc, topacc)
                checkpoint = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'scheduler':scheduler.state_dict()}
                torch.save(checkpoint, savePath+'/'+"mdn+tcn"+'_'+str(epoch)+'_'+ str(acc)+'_'+ str(p)+'_'+str(r)+'_'+str(f1score)+'.pth')
                
                torch.save(checkpoint, os.path.join(savePath, 'best_model.pth'))
    
    # ================== 【新增】加载最佳权重逻辑 ==================
    print("Training Finished. Loading Best Model for Final Testing...")
    best_model_path = os.path.join(savePath, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        # 加载 checkpoint
        checkpoint = torch.load(best_model_path)
        # 覆盖当前模型的参数
        model.load_state_dict(checkpoint['net'])
        print(f"Successfully loaded best model (Acc: {topacc}%) from {best_model_path}")
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
    # top_pre = torch.cat(top_pre,axis=0).cpu()
    # top_label=torch.cat(top_label,axis=0).cpu()
    
    # totals.append(mytop)
    # ps.append(top_p)
    # rs.append(top_r)
    # f1s.append(top_f1)
    # logging.info('topacc:'.format(mytop))
    # logging.info('')
    
    # print("Training Finished. Loading Best Model for Final Testing...")
    # print("train end")
    
    # return top_label,top_pre

def count(string):
    dig = sum(1 for char in string if char.isdigit())
    return dig

if __name__ == '__main__':
    import random
    from sklearn.model_selection import KFold,StratifiedKFold
    seed = 42
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    tcn = "data/LMVD_Feature/tcnfeature"
    mdnAudioPath = "data/LMVD_Feature/Audio_feature"
    labelPath="label/label"

    
    # 1. 加载所有样本路径和标签
    X = os.listdir(tcn)
    X.sort(key=lambda x: int(x.split(".")[0]))
    X = np.array(X)
    
    Y = []
    for i in X:
        # 注意路径拼接逻辑需与你的文件夹匹配
        file_csv = pd.read_csv(os.path.join(labelPath, (str(i.split('.npy')[0])+"_Depression.csv")))
        Y.append(int(file_csv.columns[0]))
    Y = np.array(Y)

    # 2. 【关键修改】首先切分出 10% 的“固定测试集”
    # 使用 stratify=Y 确保测试集中的抑郁/正常比例与全集一致
    X_train_val_pool, X_test_holdout, Y_train_val_pool, Y_test_holdout = train_test_split(
        X, Y, test_size=0.10, stratify=Y, random_state=seed
    )

    print(f"Total: {len(X)}, Train-Val Pool: {len(X_train_val_pool)}, Fixed Test Set: {len(X_test_holdout)}")

    # 3. 在剩下的 90% 数据上设置十折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    metrics_history = {'acc': [], 'precision': [], 'recall': [], 'f1': []}
    
    # 4. 开始 K-Fold 循环 (在池子内循环)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_pool, Y_train_val_pool)):
        print(f"\n{'='*20} Fold {fold+1} / 10 {'='*20}")
        
        # 从池子中提取当前折的训练集(81%)和验证集(9%)
        X_train_fold = X_train_val_pool[train_idx]
        X_val_fold = X_train_val_pool[val_idx]
        
        # 5. 执行训练
        # 这里的 X_test_holdout 是全局固定的 10%
        fold_results = train(tcn, mdnAudioPath, 
                             X_train_fold,  # 训练用 (81%)
                             X_val_fold,    # 验证/早停用 (9%)
                             X_test_holdout, # 最终测试用 (固定 10%)
                             labelPath, 
                             fold_name=f"Fold_{fold+1}")
        
        # 记录指标
        for key in metrics_history.keys():
            metrics_history[key].append(fold_results[key])

    # 5. 统计最终结果（均值和标准差）
    print(f"\n{'*'*15} FINAL 10-FOLD CROSS VALIDATION RESULTS {'*'*15}")
    for key, values in metrics_history.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        # Accuracy 打印百分比，其他打印小数
        if key == 'acc':
            print(f"{key.upper():<10}: {mean_val:.2f}% (+/- {std_val:.2f}%)")
        else:
            print(f"{key.upper():<10}: {mean_val:.4f} (+/- {std_val:.4f})")