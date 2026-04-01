import pandas as pd
import os
import numpy as np
import random
from pydub import AudioSegment
import math
import joblib
from sklearn.decomposition import PCA
                            
def validFrame(frames):
    """
    frames: numpy array-like
    行结构与 CSV 对应，success 在索引4。
    行为：
      - 尝试把 success 转为整数（不能转的当 0）
      - 把所有元素转为 float32（不能转的置 0）
      - 若存在 success==1，用最近的 success==1 行填充 success==0 的行（从列5开始）
      - 若不存在 success==1，打印警告并用第一行作为填充源继续处理
    """
    frames = np.asarray(frames)
    if frames.ndim != 2 or frames.shape[1] <= 5:
        raise ValueError("invalid frames shape")
    # 安全读取 success 列
    success = np.zeros(frames.shape[0], dtype=int)
    for i in range(frames.shape[0]):
        try:
            success[i] = int(float(frames[i, 4]))
        except Exception:
            success[i] = 0
    # 确保数值矩阵
    try:
        frames = frames.astype(np.float32)
    except Exception:
        tmp = np.zeros((frames.shape[0], frames.shape[1]), dtype=np.float32)
        for i in range(frames.shape[0]):
            for j in range(frames.shape[1]):
                try:
                    tmp[i, j] = float(frames[i, j])
                except Exception:
                    tmp[i, j] = 0.0
        frames = tmp
    valid_idxs = np.where(success == 1)[0]
    if valid_idxs.size == 0:#情况 1：没有有效帧
        print("Warning: no frames with success==1 — using first row to fill")
        last_valid = frames[0].copy()
        for r in range(frames.shape[0]): # 遇到有效帧，更新填充源为当前帧
            if success[r] == 1:
                last_valid = frames[r].copy()
            else:                           # 无效帧，用最近的有效帧（last_valid）填充特征列
                frames[r, 5:] = last_valid[5:]
        return frames
    # 用最近的 success==1 行填充
    last_valid = frames[valid_idxs[0]].copy()#存在有效帧
    for r in range(frames.shape[0]):
        if success[r] == 1:
            last_valid = frames[r].copy()
        else:
            frames[r, 5:] = last_valid[5:]
    return frames

def chouzhen(_feature):#每隔 6 帧抽取一帧（下采样），把长序列压缩到原来的 1/6。
    flag = 0
    for i in range(0, len(_feature), 6):
        if flag == 0:#第一次直接赋值
            feature = _feature[i]
            flag = 1
        else:
            feature = np.vstack((feature, _feature[i]))
    return feature

def split(data):#截取前 5490 帧（若有），下采样为 ~915 行，若不足 915 行则尾部补零，保证固定长度 915
    _data = chouzhen(data[:5490, ])
    if _data.shape[0] < 915:
        zeros = np.zeros([(915 - _data.shape[0]), _data.shape[1]])
        _data = np.vstack((_data, zeros))
    return _data        

def getTCNVideoFeature(trainPath, targetPath):#遍历 trainPath（支持直接一层文件或子目录结构），对每个 CSV 文件读取并生成对应的 npy 特征文件保存在 targetPath 下
    os.makedirs(targetPath, exist_ok=True)
    entries = os.listdir(trainPath)
    for entry in entries:
        entry_path = os.path.join(trainPath, entry)
        # 如果是目录则遍历目录内文件，否则直接处理该文件（支持两种结构）
        if os.path.isdir(entry_path):
            iter_files = [(os.path.join(entry_path, f), f) for f in os.listdir(entry_path)]
        else:
            iter_files = [(entry_path, entry)]
        for fullpath, fname in iter_files:
            if not fname.lower().endswith(".csv"):
                continue
            try:
                df = pd.read_csv(fullpath, low_memory=False)
            except Exception as e:
                print(f"read csv failed: {fullpath} -> {e}")
                continue
            # 强制所有列为数值，无法转的置 NaN，随后填充
            df = df.apply(pd.to_numeric, errors='coerce')
            if df.isnull().values.any():# 填充缺失值：前向填充→后向填充→剩余用0填充
                df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)#先用前向填充补全缺失值，剩余未补全的（如开头的缺失值）用后向填充，最后仍有缺失的用 0 填充，确保所有数据都是数值，无缺失
            try:
                data = np.array(df, dtype=np.float32)
            except Exception as e:
                print(f"convert to float failed: {fullpath} -> {e}")
                continue

            try:
                data = validFrame(data)
            except Exception as e:
                print(f"Video issues {fname} -> {e}")
                continue

            try:
                data = split(data) # 抽帧并固定长度为915
                data = np.delete(data, [0,1,2,3,4], axis=1)#删除前5列非特征信息
            except Exception as e:
                print(f"processing failed for {fname}: {e}")
                continue

            try:
                gaze = data[:, 0:6]# 眼球凝视特征
                gaze_zero = np.zeros_like(gaze)# 生成与gaze同形状的零矩阵
                gaze = np.hstack((gaze, gaze_zero))# 拼接凝视特征和零矩阵（扩展特征维度）
                pose = data[:, 288:294]
                features = data[:, 294:430]
                au = data[:, 447:465]
                au = np.delete(au, [5], axis=1)
                au[:, [12,13]] = au[:, [13,12]]
                au[:, [13,14]] = au[:, [14,13]]
                feature = np.hstack((au, features, gaze, pose)).astype(np.float32)# 拼接所有特征，最终转为float32
            except Exception as e:
                print(f"feature assembly failed for {fname}: {e}")
                continue

            if np.isnan(feature).any():#确保无缺失值
                print('There is a null value present：', fname)
                continue
            base_name = os.path.splitext(fname)[0]#去除文件名后缀
            np.save(os.path.join(targetPath, base_name), feature)#转换为.npy文件

if __name__ == "__main__":
    
    trainPath = r"F:\asd\LMVD\data\LMVD_Feature\Video_feature"   # 源 CSV 目录（支持直接放 csv 或每个 id 子文件夹）
    targetPath = r"F:\asd\LMVD\data\LMVD_Feature\tcnfeature"     # 保存 .npy 的目录
    if not os.path.exists(trainPath):
        raise FileNotFoundError(f"trainPath not found: {trainPath}")
    os.makedirs(targetPath, exist_ok=True)
    getTCNVideoFeature(trainPath, targetPath)