import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

#核心数据加载类

class MyDataLoader(Dataset):
    """
    自定义 PyTorch Dataset 类，用于加载 K-Fold 划分后的视频和音频特征。
    要求传入的 file_list 元素可以带或不带扩展名（'001' 或 '001.npy' 都支持）。
    返回 (video_tensor, audio_tensor, label_tensor)，其中 video_tensor/audio_tensor 形状为 (D, T)。
    """
    def __init__(self, file_list, video_feature_path, audio_feature_path, label_path, T_target=915, mode='train'):
        self.file_list = list(file_list)
        self.video_feature_path = video_feature_path
        self.audio_feature_path = audio_feature_path
        self.label_path = label_path
        self.T_target = int(T_target)
        
        # 2. 保存 mode 到 self 中
        self.mode = mode
        self.labels = [] # 用于存储所有标签，供 Sampler 使用
        for file_name in self.file_list:
            file_root = os.path.splitext(file_name)[0]
            label_file = os.path.join(self.label_path, file_root + '_Depression.csv')
            l = int(pd.read_csv(label_file, header=None).iloc[0, 0])
            self.labels.append(l)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]  # 可能是 '001' 或 '001.npy'
        file_root = os.path.splitext(file_name)[0]  # 去掉扩展名，得到 '001'

        # 构造路径
        video_file = os.path.join(self.video_feature_path, file_root + '.npy')
        audio_file = os.path.join(self.audio_feature_path, file_root + '.npy')
        label_file = os.path.join(self.label_path, file_root + '_Depression.csv')

        # 检查文件存在性
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video feature not found: {video_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio feature not found: {audio_file}")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        # 1. 加载视频特征 (.npy)：期望形状 (T, D_video)
        input_video = np.load(video_file).astype(np.float32)
        # 2. 加载音频特征 (.npy)：期望形状 (T, D_audio)
        input_audio = np.load(audio_file).astype(np.float32)

        if np.isnan(input_video).any() or np.isnan(input_audio).any():
            print(f"Warning: NaN values found in {file_root}, zeroing out...")
            input_video = np.nan_to_num(input_video)
            input_audio = np.nan_to_num(input_audio)

        # 统一裁剪/填充到目标长度 T_target（对 video 和 audio 都处理）
        T_target = self.T_target

        def pad_or_crop(arr, target_T):
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            t = arr.shape[0]
            if t > target_T:
                return arr[:target_T, :]
            elif t < target_T:
                pad = np.zeros((target_T - t, arr.shape[1]), dtype=arr.dtype)
                return np.vstack((arr, pad))
            else:
                return arr
        
        def simple_norm(arr):
            # 均值 0，标准差 1 归一化
            return (arr - arr.mean()) / (arr.std() + 1e-5)

        input_video = pad_or_crop(input_video, T_target)
        input_audio = pad_or_crop(input_audio, T_target)

        #显式调用归一化，防止 loss=nan
        input_video = simple_norm(input_video)
        input_audio = simple_norm(input_audio)
        # 3. 读取标签
        try:
            label_df = pd.read_csv(label_file, header=None)
            label = int(label_df.iloc[0, 0])
        except Exception as e:
            raise RuntimeError(f"Failed to read label file {label_file}: {e}")

        # 4. 转为 Tensor 并调整形状到 (D, T)
        # 输入目前为 (T, D)，这是 Transformer 标准序列输入 (Length, Dimension)
        input_video = torch.from_numpy(input_video).float()
        input_audio = torch.from_numpy(input_audio).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        # if self.mode == 'train': 
        # 1. 高斯噪声 (Gaussian Noise): 模拟特征提取时的微小扰动
        # 给视频特征加噪声 (强度 0.01 ~ 0.05)
            # noise_video = torch.randn_like(input_video) * 0.02 
            # input_video += noise_video
            
            # # 给音频特征加噪声
            # noise_audio = torch.randn_like(input_audio) * 0.02
            # input_audio += noise_audio

            # 2. 时序掩码 (Time Masking): 强迫模型利用上下文，而不是依赖特定帧
            # 随机把一段连续的时间步置为 0 (类似 SpecAugment)
        #     T = input_video.shape[0] # 915
        #     mask_len = int(T * 0.1) # 遮挡 10%
        #     start = np.random.randint(0, T - mask_len)
        #     input_video[start : start + mask_len, :] = 0
        #     input_audio[start : start + mask_len, :] = 0
        # # ------------------------------------------------

        return input_video, input_audio, label_tensor