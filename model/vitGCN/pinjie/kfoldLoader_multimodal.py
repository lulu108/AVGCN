import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

class MultiModalDataLoader(Dataset):
    """
    多模态数据加载器：同时加载视频特征、音频特征和面部关键点
    用于 ViT-GCN 融合模型训练
    支持 LMVD 和 D-Vlog 两种数据集格式
    """
    def __init__(self, file_list, video_feature_path, audio_feature_path, 
                 face_landmark_path, label_path, T_target=915, mode='train', dataset='LMVD'):
        """
        Args:
            file_list: 文件名列表（如 ['001.npy', '002.npy', ...] for LMVD or ['0', '1', ...] for D-Vlog）
            video_feature_path: 视频特征目录（tcnfeature for LMVD, dvlog-dataset for D-Vlog）
            audio_feature_path: 音频特征目录（Audio_feature for LMVD, dvlog-dataset for D-Vlog）
            face_landmark_path: 面部关键点目录（Video_landmarks_npy）
            label_path: 标签目录（label for LMVD, labels.csv for D-Vlog）
            T_target: 目标序列长度
            mode: 'train' 或 'test'（控制数据增强）
            dataset: 'LMVD' 或 'DVLOG'（控制数据加载逻辑）
        """
        self.file_list = list(file_list)
        self.video_feature_path = video_feature_path
        self.audio_feature_path = audio_feature_path
        self.face_landmark_path = face_landmark_path
        self.label_path = label_path
        self.T_target = int(T_target)
        self.mode = mode
        self.dataset = dataset.upper()  # 统一大写
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_root = os.path.splitext(file_name)[0]
        
        # ===== 1. 加载视频和音频特征 =====
        if self.dataset == 'DVLOG':
            # D-Vlog 格式: {index}/{index}_visual.npy 和 {index}_acoustic.npy
            video_file = os.path.join(self.video_feature_path, file_root, f"{file_root}_visual.npy")
            audio_file = os.path.join(self.audio_feature_path, file_root, f"{file_root}_acoustic.npy")
        else:
            # LMVD 格式: {index}.npy
            video_file = os.path.join(self.video_feature_path, file_root + '.npy')
            audio_file = os.path.join(self.audio_feature_path, file_root + '.npy')
        
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video feature not found: {video_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio feature not found: {audio_file}")
        
        input_video = np.load(video_file)
        input_audio = np.load(audio_file)
        input_video_raw = input_video.copy()
        
        # 处理 NaN
        input_video = np.nan_to_num(input_video, nan=0.0, posinf=1.0, neginf=-1.0)
        input_audio = np.nan_to_num(input_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Pad/Crop 到目标长度
        input_video = self._pad_or_crop(input_video, self.T_target)
        input_audio = self._pad_or_crop(input_audio, self.T_target)
        input_video_raw = self._pad_or_crop(input_video_raw, self.T_target)

        # DVLOG 数值归一化（避免数值过大导致 NaN）
        if self.dataset == 'DVLOG':
            input_video = self._normalize_features(input_video)
            input_audio = self._normalize_features(input_audio)
        
        # ===== 2. 加载/提取面部关键点并处理 =====
        if self.dataset == 'DVLOG':
            # D-Vlog: 面部关键点已包含在 visual 文件中（136维 = 68点×2坐标）
            # input_video 的形状是 (T, 136)
            # 直接在内存中处理，避免临时文件并发问题
            face_landmarks = input_video_raw.reshape(-1, 68, 2)  # (T, 68, 2)
            face_regions, actual_len = self._process_face_landmarks_array(face_landmarks)
        else:
            # LMVD 格式: {index}.npy (单独的面部关键点文件)
            face_file = os.path.join(self.face_landmark_path, file_root + '.npy')
            
            if not os.path.exists(face_file):
                raise FileNotFoundError(f"Face landmark not found: {face_file}")
            
            # 调用 _01DatasetLoader 的处理逻辑
            from _01DatasetLoader import DepressionFaceDataset
            face_dataset = DepressionFaceDataset(
                [face_file], 
                [0],  # dummy label
                mode='lmvd', 
                T_target=self.T_target, 
                augment=(self.mode == 'train'),
                adaptive_sampling=False,
                stride=1
            )
            face_regions, _, actual_len = face_dataset[0]
        
        # ===== 3. 读取标签 =====
        if self.dataset == 'DVLOG':
            # D-Vlog 使用统一的 labels.csv 文件
            label_df = pd.read_csv(self.label_path)
            # 查找对应 index 的标签 (label_df 有 'index' 和 'label' 列)
            row = label_df[label_df['index'] == int(file_root)]
            if row.empty:
                raise FileNotFoundError(f"Label not found for index {file_root} in {self.label_path}")
            # 处理字符串标签: 'depression' -> 1, 'normal' -> 0
            label_value = row['label'].iloc[0]
            if isinstance(label_value, str):
                label_value = label_value.strip().lower()
                label = 1 if label_value == 'depression' else 0
            else:
                label = int(label_value)
        else:
            # LMVD 格式: {index}_Depression.csv
            label_file = os.path.join(self.label_path, file_root + '_Depression.csv')
            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Label file not found: {label_file}")
            try:
                label_df = pd.read_csv(label_file, header=None)
                label = int(label_df.iloc[0, 0])
            except Exception as e:
                raise RuntimeError(f"Failed to read label: {e}")
        
        # ===== 4. 转为 Tensor =====
        input_video = torch.from_numpy(input_video).float()
        input_audio = torch.from_numpy(input_audio).float()
        actual_len_tensor = torch.tensor(actual_len, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # ===== 5. 训练模式数据增强 =====
        if self.mode == 'train':
            # 高斯噪声
            noise_video = torch.randn_like(input_video) * 0.02
            input_video += noise_video
            
            noise_audio = torch.randn_like(input_audio) * 0.02
            input_audio += noise_audio
            
            # 时序掩码
            T = input_video.shape[0]
            mask_len = int(T * 0.1)
            start = np.random.randint(0, T - mask_len)
            input_video[start:start + mask_len, :] = 0
            input_audio[start:start + mask_len, :] = 0
        
        return input_video, input_audio, face_regions, actual_len_tensor, label_tensor
    
    def _process_face_landmarks_array(self, raw_landmarks):
        """
        直接处理面部关键点数组，复用 DepressionFaceDataset 的处理逻辑（插值、归一化、区域划分）
        raw_landmarks: (T, 68, 2)
        Returns:
            face_regions: dict
            actual_len: int
        """
        from _01DatasetLoader import DepressionFaceDataset

        raw = raw_landmarks

        # 1) 插值修复（与 _01DatasetLoader 保持一致）
        T_len, _, _ = raw.shape
        flat_data = raw.reshape(T_len, -1)
        flat_data[flat_data == 0] = np.nan
        for i in range(flat_data.shape[1]):
            col = flat_data[:, i]
            mask = np.isnan(col)
            if np.any(mask) and not np.all(mask):
                valid_idxs = np.where(~mask)[0]
                nan_idxs = np.where(mask)[0]
                flat_data[mask, i] = np.interp(nan_idxs, valid_idxs, col[~mask])
        raw = np.nan_to_num(flat_data).reshape(T_len, 68, 2)

        # 2) 关键帧采样（D-Vlog 禁用自适应采样，使用固定策略）
        if self.T_target is not None and T_len > self.T_target:
            diff = np.diff(raw, axis=0)
            motion_scores = np.linalg.norm(diff, axis=(1, 2))
            motion_scores = np.concatenate(([0], motion_scores))
            top_indices = np.argsort(motion_scores)[-self.T_target:]
            top_indices = np.sort(top_indices)
            raw = raw[top_indices]

        # 3) 归一化与区域划分
        helper_ds = DepressionFaceDataset([], [], mode='dvlog', T_target=self.T_target,
                                          augment=(self.mode == 'train'), adaptive_sampling=False, stride=1)
        landmarks = helper_ds.normalize_face(raw)
        actual_len = min(len(landmarks), self.T_target)
        landmarks = helper_ds.pad_or_crop(landmarks, self.T_target)
        regions = helper_ds.partition_regions(landmarks)

        return regions, actual_len

    def _normalize_features(self, arr, eps=1e-6):
        """
        对特征做每维标准化，抑制极端值，避免数值溢出
        arr: (T, D)
        """
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.maximum(std, eps)
        arr = (arr - mean) / std
        arr = np.clip(arr, -5.0, 5.0)
        return arr.astype(np.float32)

    def _pad_or_crop(self, arr, target_T):
        """统一序列长度"""
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        t = arr.shape[0]
        if t > target_T:
            return arr[:target_T, :]
        elif t < target_T:
            pad = np.zeros((target_T - t, arr.shape[1]), dtype=arr.dtype)
            return np.vstack((arr, pad))
        return arr


def collate_fn_multimodal(batch):
    """
    自定义 collate 函数，处理多模态数据
    
    Args:
        batch: list of (video, audio, face_regions, actual_len, label)
    
    Returns:
        video_batch: (B, T, D_video)
        audio_batch: (B, T, D_audio)
        face_regions_batch: dict of {region: (B, T, N, C)}
        actual_lens: (B,)
        labels: (B,)
    """
    video_list, audio_list, face_list, len_list, label_list = zip(*batch)
    
    # 视频和音频直接堆叠
    video_batch = torch.stack(video_list)
    audio_batch = torch.stack(audio_list)
    actual_lens = torch.stack(len_list)
    labels = torch.stack(label_list)
    
    # 面部区域需要特殊处理
    region_keys = face_list[0].keys()
    face_regions_batch = {}
    
    for key in region_keys:
        # 收集所有样本的当前区域
        regions = [face[key] for face in face_list]
        # 堆叠成 (B, T, N, C)
        face_regions_batch[key] = torch.stack([torch.from_numpy(r) for r in regions])
    
    return video_batch, audio_batch, face_regions_batch, actual_lens, labels
