import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 分区方案默认值（同时作为 "legacy6" 单独使用 DepressionFaceDataset 时的默认）
# 训练脚本通过构造参数覆盖；扁平可改此值做快速单文件测试
REGION_PARTITION_SCHEME = "legacy6"   # "legacy6" | "symptom7"

def interpolate_missing_landmarks(landmarks):
    """
    针对 0 值或 NaN 进行线性插值修复。
    输入 shape: (T, features)
    """
    # 1. 将全 0 视为 NaN (设 0 是无效值)
    landmarks = landmarks.astype(float)
    # 如果某一行全是 0，标记为 NaN (或者根据具体业务逻辑，如果坐标是 0 则为 NaN)
    landmarks[landmarks == 0] = np.nan
    
    # 2. 使用 Pandas 进行线性插值
    df = pd.DataFrame(landmarks)
    # limit_direction='both' 确保开头和结尾的 NaN 也能被填充
    df = df.interpolate(method='linear', limit_direction='both', axis=0)
    
    # 3. 如果整列都是 NaN (该特征在整个视频都丢失)，则补 0
    df = df.fillna(0)
class DepressionFaceDataset(Dataset):
    def __init__(self, data_list, labels, mode='dvlog', T_target=None, interpolate_invalid=True, augment=False, stride=1, adaptive_sampling=True,
                 region_partition_scheme=None):
        """
        新增参数: 
            augment (bool) - 是否开启数据增强
            adaptive_sampling (bool) - 是否启用自适应采样
            region_partition_scheme (str|None) - 分区方案（None 表示使用模块层 REGION_PARTITION_SCHEME 默认值）
        """
        self.data_list = list(data_list)
        self.labels = list(labels)
        self.mode = mode
        self.T_target = int(T_target) if T_target is not None else None
        self.interpolate_invalid = interpolate_invalid
        self.augment = augment  # 保存增强标志
        self.stride = stride
        self.adaptive_sampling = adaptive_sampling  # 保存自适应采样标志
        # 分区方案：None 时回落模块层全局常量
        self.region_partition_scheme = region_partition_scheme if region_partition_scheme is not None else REGION_PARTITION_SCHEME

    def __len__(self):
        return len(self.data_list)

    def _dvlog_seq136_to_pts68x2(self, seq136):
        """
        DVLOG 解码：
          输入  (T,136) : [x0..x67, y0..y67]
          输出  (T,68,2)
        """
        arr = np.asarray(seq136)
        if arr.ndim != 2 or arr.shape[1] != 136:
            raise ValueError(f"[DVLOG decode] expect (T,136), got {arr.shape}")
        x = arr[:, :68]
        y = arr[:, 68:]
        return np.stack([x, y], axis=-1).astype(np.float32)

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        label = int(self.labels[idx])

        # 1. 加载并强制转换为 3D 形状 (Time, 68, 2)
        raw = np.load(file_path)
        if raw.ndim == 2:
            # 处理 D-vlog 的 (T, 136) 情况（split 解码，不可直接 reshape）
            raw = self._dvlog_seq136_to_pts68x2(raw)
        
        # 2. 插值修复 (防止全 0 或 NaN 破坏特征计算)
        # 将 (T, 68, 2) 展平为 (T, 136) 进行逐列(特征维度)插值
        T_len, N_pts, C_coords = raw.shape
        flat_data = raw.reshape(T_len, -1)
        
        # 将 0 视为缺失值
        flat_data[flat_data == 0] = np.nan
        
        for i in range(flat_data.shape[1]):
            col = flat_data[:, i]
            mask = np.isnan(col)
            if np.any(mask) and not np.all(mask):
                # 只对有数据的列进行插值，全 NaN 的列后面用 fillna(0) 处理
                valid_idxs = np.where(~mask)[0]
                nan_idxs = np.where(mask)[0]
                flat_data[mask, i] = np.interp(nan_idxs, valid_idxs, col[~mask])
        
        # 填充无法插值的全空列，并还原回 3D
        raw = np.nan_to_num(flat_data).reshape(T_len, 68, 2)

        # 3. 自适应采样 (关键帧选择) - 新增智能采样策略
        if self.T_target is not None and T_len > self.T_target:
            if self.adaptive_sampling:
                # === 自适应采样：根据运动强度动态调整采样 ===
                raw = self.adaptive_frame_sampling(raw, self.T_target)
            else:
                # === 原始固定步长采样 ===
                # 计算帧间运动强度：(T-1, 68, 2)
                diff = np.diff(raw, axis=0)
                # 计算每一帧 68 个点的位移范数之和: (T-1,)
                motion_scores = np.linalg.norm(diff, axis=(1, 2))
                # 补齐首帧得分
                motion_scores = np.concatenate(([0], motion_scores))
                
                # 选取分数最高的 T_target 个索引
                top_indices = np.argsort(motion_scores)[-self.T_target:]
                # 必须排序，确保关键帧是按时间顺序排列的！
                top_indices = np.sort(top_indices)
                raw = raw[top_indices]

        # 4. 特征提取 (此时 raw 已经是干净的 3D 数组)
        # 传入 normalize_face 进一步处理归一化、速度、距离等特征
        landmarks = self.normalize_face(raw)
        
        # 5. 后续处理 (Pad/Crop 和 Region Partition)
        actual_len = min(len(landmarks), self.T_target)
        landmarks = self.pad_or_crop(landmarks, self.T_target)
        regions = self.partition_regions(landmarks, scheme=self.region_partition_scheme)

        return regions, label, actual_len

    def apply_augmentation(self, landmarks):
        """
        【优化】增强策略：以空间增强为主，时序增强降低概率
        避免过度破坏微表情的连续性
        """
        # === 空间增强（保留，微表情识别的核心）===
        # 1. 随机旋转（幅度减小）
        angle = np.radians(np.random.uniform(-5, 5))  # 从±10度减小到±5度
        cos_val, sin_val = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
        coords = landmarks[:, :, :2]
        landmarks[:, :, :2] = np.dot(coords, rot_mat)

        # 2. 随机缩放（保留）
        scale = np.random.uniform(0.97, 1.03)  # 缩小范围避免失真
        landmarks = landmarks * scale
        
        # 3. 随机高斯噪声（保留）
        noise = np.random.normal(0, 0.0005, landmarks.shape)  # 降低噪声强度
        landmarks = landmarks + noise
        
        # === 时序增强（大幅降低概率，仅用于LMVD）===
        # 【关键修改】时序失真概率从30%降到15%
        if np.random.rand() < 0.15:
            landmarks = self.temporal_distortion(landmarks)
        
        # 【关键修改】随机删除帧概率从20%降到10%
        if np.random.rand() < 0.10:
            landmarks = self.random_frame_drop(landmarks)
        
        return landmarks.astype(np.float32)
    
    def temporal_distortion(self, landmarks, max_shift=5):
        """
        时序失真增强：局部时间平移 + 时间缩放
        
        Args:
            landmarks: (T, N, C) 时序特征
            max_shift: 最大时间偏移量
        
        Returns:
            失真后的时序数据
        """
        T = landmarks.shape[0]
        if T < 10:  # 序列太短则跳过
            return landmarks
        
        # 策略1: 局部时间平移（概率从50%提高到60%，因为这个相对温和）
        if np.random.rand() < 0.6:
            # 随机选择一个时间段进行平移
            segment_len = np.random.randint(T // 4, T // 2)
            start_idx = np.random.randint(0, max(1, T - segment_len))
            shift = np.random.randint(-max_shift, max_shift + 1)
            
            # 创建新序列并进行平移
            new_landmarks = landmarks.copy()
            if shift > 0:
                # 向后平移
                new_landmarks[start_idx + shift:start_idx + segment_len + shift] = \
                    landmarks[start_idx:start_idx + segment_len]
            elif shift < 0:
                # 向前平移
                new_landmarks[max(0, start_idx + shift):start_idx + segment_len + shift] = \
                    landmarks[start_idx:start_idx + segment_len]
            landmarks = new_landmarks
        
        # 策略2: 时间缩放（概率降低到40%，且缩放范围更保守）
        else:
            # 【关键】缩放范围从0.8-1.2缩小到0.9-1.1，保护微表情
            scale_factor = np.random.choice([0.9, 0.95, 1.05, 1.1])  # 90%-110%速度
            
            # 计算新的帧数
            new_T = int(T * scale_factor)
            if new_T < 5:  # 避免过度压缩
                return landmarks
            
            # 使用线性插值进行时间重采样
            old_indices = np.linspace(0, T - 1, T)
            new_indices = np.linspace(0, T - 1, new_T)
            
            # 对每个关键点和特征维度进行插值
            resampled = np.zeros((new_T, landmarks.shape[1], landmarks.shape[2]), dtype=np.float32)
            for n in range(landmarks.shape[1]):
                for c in range(landmarks.shape[2]):
                    resampled[:, n, c] = np.interp(new_indices, old_indices, landmarks[:, n, c])
            
            # 裁剪或填充回原始长度
            if new_T > T:
                landmarks = resampled[:T]  # 裁剪
            else:
                # 填充（重复最后一帧）
                pad = np.repeat(resampled[-1:], T - new_T, axis=0)
                landmarks = np.concatenate([resampled, pad], axis=0)
        
        return landmarks
    
    def random_frame_drop(self, landmarks, drop_ratio=0.10):
        """
        【优化】随机删除不重要的帧 - 降低删除比例保护连续性
        
        Args:
            landmarks: (T, N, C) 时序特征
            drop_ratio: 删除比例（从15%降到10%）
        
        Returns:
            删除帧后的序列
        """
        T = landmarks.shape[0]
        if T < 10:  # 序列太短则跳过
            return landmarks
        
        # 计算帧间运动强度 (与关键帧采样类似)
        diff = np.diff(landmarks, axis=0)
        motion_scores = np.linalg.norm(diff, axis=(1, 2))
        motion_scores = np.concatenate(([0], motion_scores))  # 补齐首帧
        
        # 计算要保留的帧数
        keep_num = int(T * (1 - drop_ratio))
        
        # 保留运动强度最高的帧 (关键帧)
        keep_indices = np.argsort(motion_scores)[-keep_num:]
        keep_indices = np.sort(keep_indices)  # 保持时间顺序
        
        # 删除低运动帧
        landmarks = landmarks[keep_indices]
        
        # 填充回原始长度 (使用线性插值)
        if len(keep_indices) < T:
            old_indices = keep_indices
            new_indices = np.linspace(0, T - 1, T).astype(int)
            
            resampled = np.zeros((T, landmarks.shape[1], landmarks.shape[2]), dtype=np.float32)
            for n in range(landmarks.shape[1]):
                for c in range(landmarks.shape[2]):
                    resampled[:, n, c] = np.interp(new_indices, old_indices, landmarks[:, n, c])
            landmarks = resampled
        
        return landmarks
    
    def adaptive_frame_sampling(self, landmarks, target_T):
        """
        自适应采样：根据运动强度动态调整采样步长
        
        核心思想：
        - 运动剧烈的时段（关键帧）→ 密集采样（小步长）
        - 运动平缓的时段（冗余帧）→ 稀疏采样（大步长）
        
        Args:
            landmarks: (T, 68, 2) 原始序列
            target_T: 目标帧数
        
        Returns:
            (target_T, 68, 2) 自适应采样后的序列
        """
        T_original = landmarks.shape[0]
        
        # 1. 计算每一帧的运动强度
        diff = np.diff(landmarks, axis=0)
        motion_scores = np.linalg.norm(diff, axis=(1, 2))  # (T-1,)
        motion_scores = np.concatenate(([0], motion_scores))  # 补齐首帧
        
        # 2. 平滑运动曲线（避免噪声影响）
        window_size = min(5, T_original // 10)
        if window_size > 1:
            motion_scores = np.convolve(motion_scores, np.ones(window_size)/window_size, mode='same')
        
        # 3. 归一化运动分数到 [0, 1]
        min_score = motion_scores.min()
        max_score = motion_scores.max()
        if max_score - min_score > 1e-6:
            normalized_scores = (motion_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(motion_scores) * 0.5
        
        # 4. 动态计算每一帧的采样权重
        # 运动强度越大，权重越高（被采样的概率越大）
        # 使用指数函数增强差异性
        sampling_weights = np.exp(normalized_scores * 2)  # 指数放大
        sampling_weights = sampling_weights / sampling_weights.sum()  # 归一化为概率分布
        
        # 5. 基于权重进行采样
        # 策略：混合确定性采样(70%) + 随机加权采样(30%)
        
        # 5.1 确定性采样：保证关键帧被选中
        n_deterministic = int(target_T * 0.7)
        top_indices = np.argsort(motion_scores)[-n_deterministic:]
        
        # 5.2 加权随机采样：增加多样性
        n_random = target_T - n_deterministic
        remaining_indices = np.setdiff1d(np.arange(T_original), top_indices)
        if len(remaining_indices) > 0 and n_random > 0:
            remaining_weights = sampling_weights[remaining_indices]
            remaining_weights = remaining_weights / remaining_weights.sum()
            random_indices = np.random.choice(
                remaining_indices, 
                size=min(n_random, len(remaining_indices)),
                replace=False,
                p=remaining_weights
            )
        else:
            random_indices = np.array([], dtype=int)
        
        # 5.3 合并并排序
        selected_indices = np.concatenate([top_indices, random_indices])
        selected_indices = np.sort(selected_indices)[:target_T]  # 保持时间顺序
        
        # 6. 如果采样数不足，用均匀采样补足
        if len(selected_indices) < target_T:
            uniform_indices = np.linspace(0, T_original - 1, target_T).astype(int)
            return landmarks[uniform_indices]
        
        return landmarks[selected_indices]

    def parse_dvlog_planar(self, raw):
        # 1. 确保 raw 形状正确 (T, 68, 2)
        if raw.ndim == 2:
            raw = self._dvlog_seq136_to_pts68x2(raw)
            
        T, N, C = raw.shape
        # 2. 展平为 (T, 136) 方便插值处理
        arr = raw.reshape(T, -1) 
        
        # 3. 处理全 0 或 NaN
        arr[arr == 0] = np.nan
        
        # 4. 执行插值
        for i in range(arr.shape[1]):  # 遍历 136 个特征维度
            mask = np.isnan(arr[:, i])
            # 如果全是 NaN，没法插值，补 0
            if np.all(mask):
                arr[:, i] = 0
            # 如果有部分 NaN，且有有效值，执行插值
            elif np.any(mask):
                # 找到有效值的索引和对应的值
                valid_idx = np.where(~mask)[0]
                nan_idx = np.where(mask)[0]
                arr[mask, i] = np.interp(nan_idx, valid_idx, arr[~mask, i])
        
        # 5. 还原形状回 (T, 68, 2)
        arr = arr.reshape(T, 68, 2)
        valid_ratio = np.count_nonzero(~np.isnan(arr)) / arr.size
        if valid_ratio < 0.5:  # 若有效比值低于 50%，直接丢弃样本
            raise ValueError("Invalid sample with too many NaNs or zero values")
        return arr

    def parse_lmvd_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            raise ValueError(f"Error reading CSV {csv_path}: {e}")
        df.columns = df.columns.str.strip()
        if 'success' in df.columns:
            if self.interpolate_invalid:
                x_cols = [f'x_{i}' for i in range(68)]
                y_cols = [f'y_{i}' for i in range(68)]
                existing_cols = set(df.columns)
                missing = [c for c in x_cols + y_cols if c not in existing_cols]
                if missing:
                    filename = os.path.basename(csv_path)
                    raise ValueError(f"Missing landmark columns in csv [{filename}]: {missing[:5]}...")
                for col in x_cols + y_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                valid_mask = (df['success'] == 1)
                df.loc[~valid_mask, x_cols + y_cols] = np.nan
                df[x_cols + y_cols] = df[x_cols + y_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
            else:
                df = df[df['success'] == 1]
        x_cols = [f'x_{i}' for i in range(68)]
        y_cols = [f'y_{i}' for i in range(68)]
        xs = df[x_cols].values.astype(np.float32)
        ys = df[y_cols].values.astype(np.float32)
        pts = np.stack([xs, ys], axis=2)
        return pts

    def normalize_face(self, landmarks):
        # 1. 基础几何归一化 (解决相机远近/位置问题)
        nose = landmarks[:, 30:31, :]
        centered = landmarks - nose
        left_corner = landmarks[:, 0, :]   
        right_corner = landmarks[:, 16, :] 
        width = np.linalg.norm(right_corner - left_corner, axis=1, keepdims=True).reshape(-1, 1, 1)
        safe_width = np.where(width < 1e-3, 1.0, width)
        normalized = centered / safe_width # (T, 68, 2)

        # 2. 计算速度特征 (保留动态信息)
        window = 3 
        velocity = np.zeros_like(normalized)
        velocity[window:] = normalized[window:] - normalized[:-window]
        magnitude = np.linalg.norm(velocity, axis=-1, keepdims=True)

        # 3. 几何距离特征 (捕捉关键区域间距)
        ref_idx = [36, 45, 30, 62] # 左右眼、鼻、唇
        ref_pts = normalized[:, ref_idx, :]
        dist_feats = []
        for i in range(len(ref_idx)):
            d = np.linalg.norm(normalized - ref_pts[:, i:i+1, :], axis=-1, keepdims=True)
            dist_feats.append(d)
        dist_feat = np.concatenate(dist_feats, axis=-1) # (T, 68, 4)

        # 4. 对称性特征 (新增：衡量左右脸动作差异)
        # 定义对称点对 (左, 右)
        left_side = [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41]
        right_side = [26, 25, 24, 23, 22, 45, 44, 43, 42, 47, 46]
        asym_val = np.abs(normalized[:, left_side, :] - normalized[:, right_side, :])
        asym_feat = np.zeros((normalized.shape[0], 68, 1))
        # 简单取均值并广播到所有点作为全局对称性指标
        asym_feat[:] = np.mean(asym_val, axis=(1, 2), keepdims=True)

        # 5. 拼接特征: 坐标(2) + 速度(2) + 速率(1) + 距离(4) + 对称性(1) = 10维
        feat = np.concatenate([normalized, velocity, magnitude, dist_feat, asym_feat], axis=-1)

        # 核心修改：固定缩放。确保数值在 Transformer 友好范围，同时保留样本间强度差异
        feat = feat * 1.0  
        feat = np.clip(feat, -5.0, 5.0)
        return feat.astype(np.float32)

    def pad_or_crop(self, arr, target_T):
        t = arr.shape[0]
        if t >= target_T: return arr[:target_T]
        
        # 使用全 0 填充，而不是重复最后一帧
        pad = np.zeros((target_T - t, arr.shape[1], arr.shape[2]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def partition_regions(self, landmarks, scheme="legacy6"):
        if scheme == "symptom7":
            regions = {
                'ljaw':           landmarks[:, list(range(0, 9)), :].astype(np.float32),
                'rjaw':           landmarks[:, list(range(8, 17)), :].astype(np.float32),
                'leye':           landmarks[:, list(range(36, 42)), :].astype(np.float32),
                'reye':           landmarks[:, list(range(42, 48)), :].astype(np.float32),
                'brow_glabella':  landmarks[:, list(range(17, 27)) + list(range(27, 31)), :].astype(np.float32),
                'nose_lower':     landmarks[:, list(range(31, 36)), :].astype(np.float32),
                'mouth':          landmarks[:, list(range(48, 68)), :].astype(np.float32),
            }
            return regions

        # 默认：legacy6
        if scheme != "legacy6":
            raise ValueError(f"Unknown partition scheme: {scheme}")
        mouth_idx = list(range(48, 68))
        nose_idx = list(range(27, 36))
        # 确保顺序正确：先眉毛后眼睛
        leye_idx = list(range(17, 22)) + list(range(36, 42))
        reye_idx = list(range(22, 27)) + list(range(42, 48))
        ljaw_idx = list(range(0, 9))
        rjaw_idx = list(range(8, 17))

        regions = {
            'ljaw':  landmarks[:, ljaw_idx, :].astype(np.float32),
            'rjaw':  landmarks[:, rjaw_idx, :].astype(np.float32),
            'leye':  landmarks[:, leye_idx, :].astype(np.float32),
            'reye':  landmarks[:, reye_idx, :].astype(np.float32),
            'nose':  landmarks[:, nose_idx, :].astype(np.float32),
            'mouth': landmarks[:, mouth_idx, :].astype(np.float32),
        }
        return regions