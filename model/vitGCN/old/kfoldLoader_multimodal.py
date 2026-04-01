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
                 face_landmark_path, label_path, T_target=915, mode='train', dataset='LMVD',
                 temporal_mask_ratio=0.0):
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
            temporal_mask_ratio: 时序掩码比例（仅训练模式生效，0.0 表示不掩码）
        """
        self.file_list = list(file_list)
        self.video_feature_path = video_feature_path
        self.audio_feature_path = audio_feature_path
        self.face_landmark_path = face_landmark_path
        self.label_path = label_path
        self.T_target = int(T_target)
        self.mode = mode
        self.dataset = dataset.upper()  # 统一大写
        self.temporal_mask_ratio = temporal_mask_ratio
        
        # ================== Step0: 构建 metas 列表（用于 top-loss 诊断）==================
        # 在 __init__ 中预计算每个样本的元信息，方便训练脚本按 idx 查询
        self.metas = []
        for fname in self.file_list:
            froot = os.path.splitext(fname)[0]
            if self.dataset == 'DVLOG':
                vp = os.path.join(video_feature_path, froot, f"{froot}_visual.npy")
                ap = os.path.join(audio_feature_path, froot, f"{froot}_acoustic.npy")
                fp = vp  # DVLOG 面部关键点嵌在 visual 里
            else:
                vp = os.path.join(video_feature_path, froot + '.npy')
                ap = os.path.join(audio_feature_path, froot + '.npy')
                fp = os.path.join(face_landmark_path, froot + '.npy')
            self.metas.append({
                'id': froot,
                'video_path': vp,
                'audio_path': ap,
                'landmark_path': fp,
            })
    
    def get_meta(self, idx):
        """返回第 idx 个样本的元信息（不加载数据，O(1)）"""
        return self.metas[idx]
    
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
        
        # ===== 1.5 DVLOG 维度纠错：期望 video=(T,136), audio=(T,25) =====
        # 某些样本的 visual/acoustic 文件名可能与实际内容对调
        if self.dataset == 'DVLOG':
            input_video, input_audio, _swapped = self._maybe_swap_by_dim(
                input_video, input_audio, expected_vid_d=136, expected_aud_d=25)
        
        input_video_raw = input_video.copy()
        
        # 处理 NaN
        input_video = np.nan_to_num(input_video, nan=0.0, posinf=1.0, neginf=-1.0)
        input_audio = np.nan_to_num(input_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Pad/Crop 到目标长度
        input_video = self._pad_or_crop(input_video, self.T_target)
        input_audio = self._pad_or_crop(input_audio, self.T_target)
        input_video_raw = self._pad_or_crop(input_video_raw, self.T_target)

        # ===== v_missing 判定（在归一化前，基于原始特征）=====
        # 对 DVLOG 特别关键：部分样本视觉特征全零、近零或近常数，需标记为缺失
        v_missing = self._is_allzero_or_constant(input_video)

        # 数值归一化（避免数值过大导致 Transformer / BN 内部 NaN）
        # 对所有数据集统一执行：per-sample z-normalization + clip 到 [-5, 5]
        # LMVD 的 TCN / Audio 特征若未预归一化，原始数值可能极大，
        # 或 float64→float32 溢出为 inf，从而在第一个 forward 就产生 NaN。
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
        

        # ===== 5. 转为 Tensor =====
        input_video = torch.from_numpy(input_video).float()
        input_audio = torch.from_numpy(input_audio).float()
        # 安全网：防止 float64→float32 溢出产生 inf / nan
        input_video = torch.nan_to_num(input_video, nan=0.0, posinf=5.0, neginf=-5.0)
        input_audio = torch.nan_to_num(input_audio, nan=0.0, posinf=5.0, neginf=-5.0)
        actual_len_tensor = torch.tensor(actual_len, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        v_missing_tensor = torch.tensor(v_missing, dtype=torch.bool)
        
        # ===== 6. 训练模式数据增强 =====
        if self.mode == 'train':
            # 高斯噪声
            noise_video = torch.randn_like(input_video) * 0.02
            input_video += noise_video
            
            noise_audio = torch.randn_like(input_audio) * 0.02
            input_audio += noise_audio
            
            # 时序掩码：随机散布式掩码（由外部传入比例控制）
            input_video = self._temporal_mask(input_video, self.temporal_mask_ratio)
            input_audio = self._temporal_mask(input_audio, self.temporal_mask_ratio)
        
        return input_video, input_audio, face_regions, actual_len_tensor, label_tensor, v_missing_tensor
    
    @staticmethod
    def _temporal_mask(x, mask_ratio=0.1):
        """随机散布式时序掩码增强（单样本版）
        
        随机选取 mask_ratio 比例的时间步，将对应帧的所有特征维度置零。
        相比连续块掩码，散布式掩码更温和，不会破坏长段连续特征。
        
        Args:
            x: 输入张量 (T, D)，单个样本
            mask_ratio: 掩码比例，0.0 表示不掩码
        Returns:
            掩码后的张量，形状不变
        """
        if mask_ratio <= 0.0:
            return x
        T = x.shape[0]
        num_mask = max(1, int(T * mask_ratio))
        indices = torch.randperm(T)[:num_mask]
        x[indices, :] = 0.0
        return x

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

        # 1.5) DVLOG 几何归一化：以鼻尖为中心、IOD（两眼间距）为尺度
        raw = self._geometric_normalize_landmarks(raw)

        # 2) 关键帧采样（D-Vlog 禁用自适应采样，使用固定策略）
        if self.T_target is not None and T_len > self.T_target:
            diff = np.diff(raw, axis=0)
            motion_scores = np.linalg.norm(diff, axis=(1, 2))
            motion_scores = np.concatenate(([0], motion_scores))
            top_indices = np.argsort(motion_scores)[-self.T_target:]
            top_indices = np.sort(top_indices)
            raw = raw[top_indices]

        # 3) 特征工程（速度、距离、对称性）
        #    skip_geometry=True：上游 _geometric_normalize_landmarks 已用 IOD 完成几何归一化
        #    这里只需要在归一化坐标基础上提取动态特征，避免面宽再次缩放
        helper_ds = DepressionFaceDataset([], [], mode='dvlog', T_target=self.T_target,
                                          augment=(self.mode == 'train'), adaptive_sampling=False, stride=1)
        landmarks = helper_ds.normalize_face(raw, skip_geometry=True)
        actual_len = min(len(landmarks), self.T_target)
        landmarks = helper_ds.pad_or_crop(landmarks, self.T_target)
        regions = helper_ds.partition_regions(landmarks)

        return regions, actual_len

    @staticmethod
    def _geometric_normalize_landmarks(landmarks, eps=1e-6):
        """
        DVLOG 专用几何归一化：以鼻尖为中心、两眼间距 (IOD) 为尺度。
        消除不同视频中相机距离 / 人脸位置差异，
        保证 GCN 输入的关键点坐标处于统一坐标系。

        与 normalize_face 的区别：
        - normalize_face 使用面宽 (point0-point16) 做尺度，受张嘴/下巴运动影响大
        - 本方法使用 IOD (两眼中心距) 做尺度，更稳定
        - 本方法在插值修复后立刻执行，保证后续运动打分和特征提取均基于归一化坐标

        Args:
            landmarks: (T, 68, 2) 插值修复后的关键点坐标
        Returns:
            (T, 68, 2) 几何归一化后的关键点
        """
        normalized = landmarks.copy()
        T_len = normalized.shape[0]

        for t in range(T_len):
            frame = normalized[t]  # (68, 2)

            # 跳过全零帧（padding 或缺失帧）
            if np.abs(frame).sum() < eps:
                continue

            # 1. 中心化：以鼻尖 (landmark 30) 为原点
            nose_tip = frame[30:31, :]  # (1, 2)
            frame = frame - nose_tip

            # 2. 尺度归一化：除以两眼中心距离 (IOD)
            #    IOD 比面宽更稳定（不受张嘴 / 下巴运动影响）
            left_eye_center = frame[36:42, :].mean(axis=0)   # 左眼 6 点均值
            right_eye_center = frame[42:48, :].mean(axis=0)  # 右眼 6 点均值
            iod = np.linalg.norm(right_eye_center - left_eye_center)

            if iod > eps:
                frame = frame / iod

            normalized[t] = frame

        return normalized

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

    @staticmethod
    def _maybe_swap_by_dim(v, a, expected_vid_d=136, expected_aud_d=25):
        """
        DVLOG 维度纠错：检测 video/audio 是否被意外对调。
        
        期望: video=(T, 136), audio=(T, 25)
        若检测到 video.shape[1]==25 且 audio.shape[1]==136，则自动交换。
        
        Args:
            v: 视频特征 ndarray
            a: 音频特征 ndarray
            expected_vid_d: 期望视频特征维度
            expected_aud_d: 期望音频特征维度
        Returns:
            (video, audio, swapped: bool)
        """
        if v.ndim == 2 and a.ndim == 2:
            if v.shape[1] == expected_aud_d and a.shape[1] == expected_vid_d:
                return a.copy(), v.copy(), True
        return v, a, False

    @staticmethod
    def _is_allzero_or_constant(x, eps=1e-8, zero_frac_thr=0.999, std_thr=1e-6):
        """
        鲁棒判定特征是否"无效"（全零 / 近全零 / 近常数）。
        
        判定条件（满足其一即返回 True）:
        1. 数组为空
        2. 绝对值 < eps 的比例 ≥ zero_frac_thr（几乎全零）
        3. 整体 std < std_thr（近常数，无信息量）
        
        Args:
            x: ndarray, 通常 shape (T, D)
            eps: 零值判定阈值
            zero_frac_thr: 零值占比阈值
            std_thr: 标准差阈值（低于此值视为常数）
        Returns:
            bool — True 表示该模态特征无效/缺失
        """
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 0:
            return True
        zero_frac = float((np.abs(x) < eps).mean())
        if zero_frac >= zero_frac_thr:
            return True
        if float(np.std(x)) < std_thr:
            return True
        return False

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
        batch: list of (video, audio, face_regions, actual_len, label, v_missing)
    
    Returns:
        video_batch: (B, T, D_video)
        audio_batch: (B, T, D_audio)
        face_regions_batch: dict of {region: (B, T, N, C)}
        actual_lens: (B,)
        labels: (B,)
        v_missing_batch: (B,) bool — True 表示该样本视觉模态缺失
    """
    video_list, audio_list, face_list, len_list, label_list, v_missing_list = zip(*batch)
    
    # 视频和音频直接堆叠
    video_batch = torch.stack(video_list)
    audio_batch = torch.stack(audio_list)
    actual_lens = torch.stack(len_list)
    labels = torch.stack(label_list)
    v_missing_batch = torch.stack(v_missing_list)  # (B,) bool
    
    # 面部区域需要特殊处理
    region_keys = face_list[0].keys()
    face_regions_batch = {}
    
    for key in region_keys:
        # 收集所有样本的当前区域
        regions = [face[key] for face in face_list]
        # 堆叠成 (B, T, N, C)
        face_regions_batch[key] = torch.stack([torch.from_numpy(r) for r in regions])
    
    return video_batch, audio_batch, face_regions_batch, actual_lens, labels, v_missing_batch
