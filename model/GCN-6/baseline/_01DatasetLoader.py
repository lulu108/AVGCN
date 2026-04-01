import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class DepressionFaceDataset(Dataset):
    """
    统一的 Dataset：支持 dvlog (.npy planar) 与 lmvd (OpenFace csv) 两种模式。
    输出 regions 字典，6 个区域，每个 value 形状为 (T, N, 2) (numpy float32)。
    """
    def __init__(self, data_list, labels, mode='dvlog', T_target=None, interpolate_invalid=True):
        """
        :param data_list: list of file paths (完整路径)
        :param labels: list/array of labels (0/1)
        :param mode: 'dvlog' 或 'lmvd'
        :param T_target: 若不为 None，则对每个样本 pad/crop 到 T_target
        :param interpolate_invalid: 对于 CSV invalid frames 是否做上一帧填充
        """
        self.data_list = list(data_list)
        self.labels = list(labels)
        self.mode = mode
        self.T_target = int(T_target) if T_target is not None else None
        self.interpolate_invalid = interpolate_invalid

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        label = int(self.labels[idx])

        if self.mode == 'dvlog':
            raw = np.load(file_path)  # (T, 136) planar: first68 x, last68 y
            landmarks = self.parse_dvlog_planar(raw)  # (T, 68, 2)
        elif self.mode == 'lmvd':
            landmarks = self.parse_lmvd_csv(file_path)  # (T, 68, 2)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # 归一化（鼻尖对齐 + 脸宽归一）
        landmarks = self.normalize_face(landmarks)  # (T,68,2)

        # pad / crop 到 T_target（若指定）
        if self.T_target is not None:
            landmarks = self.pad_or_crop(landmarks, self.T_target)

        # 切分为 6 个区域
        regions = self.partition_regions(landmarks)  # dict of (T, N_i, 2)

        # 返回 numpy (not torch) -> DataLoader will collate into tensors
        return regions, label

    def parse_dvlog_planar(self, arr):
        # arr shape (T,136): first 68 are x, next 68 are y
        if arr.ndim != 2 or arr.shape[1] != 136:
            raise ValueError("Expected dvlog planar shape (T,136)")
        xs = arr[:, :68]
        ys = arr[:, 68:136]
        pts = np.stack([xs, ys], axis=2)  # (T, 68, 2) where last dim is (x,y)
        return pts.astype(np.float32)

    def parse_lmvd_csv(self, csv_path):
        # Read CSV
        try:
            # 【修改点 1】加上 low_memory=False，解决 DtypeWarning
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            raise ValueError(f"Error reading CSV {csv_path}: {e}")

        # 去除列名空格
        df.columns = df.columns.str.strip()

        # Check for 'success' column
        if 'success' in df.columns:
            if self.interpolate_invalid:
                x_cols = [f'x_{i}' for i in range(68)]
                y_cols = [f'y_{i}' for i in range(68)]
                
                # Check missing
                existing_cols = set(df.columns)
                missing = [c for c in x_cols + y_cols if c not in existing_cols]
                if missing:
                    filename = os.path.basename(csv_path)
                    raise ValueError(f"Missing landmark columns in csv [{filename}]: {missing[:5]}...")

                # 【修改点 2】强制将坐标列转换为数字，无法转换的变成 NaN
                # 这能彻底清洗掉 CSV 中的乱码或字符串
                for col in x_cols + y_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                valid_mask = (df['success'] == 1)
                
                # Set invalid frames to NaN
                df.loc[~valid_mask, x_cols + y_cols] = np.nan
                
                # Forward fill then backward fill then fill 0
                df[x_cols + y_cols] = df[x_cols + y_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
            else:
                df = df[df['success'] == 1]

        # Final extraction
        x_cols = [f'x_{i}' for i in range(68)]
        y_cols = [f'y_{i}' for i in range(68)]
        
        missing = [c for c in x_cols + y_cols if c not in df.columns]
        if missing:
             filename = os.path.basename(csv_path)
             raise ValueError(f"Missing landmark columns in csv [{filename}]: {missing[:5]}...")

        # 再次确保取出来的值是 float32
        xs = df[x_cols].values.astype(np.float32)
        ys = df[y_cols].values.astype(np.float32)
        
        pts = np.stack([xs, ys], axis=2)  # (T,68,2) where last is (x,y)
        return pts

    def normalize_face(self, landmarks):
        # landmarks: (T,68,2)
        T = landmarks.shape[0]
        # nose tip index 30 (standard 68-point)
        nose = landmarks[:, 30:31, :]  # (T,1,2)
        centered = landmarks - nose  # broadcast
        # face width: distance between point 0 and 16
        left = landmarks[:, 0, :]  # (T,2)
        right = landmarks[:, 16, :]
        width = np.linalg.norm(right - left, axis=1, keepdims=True)  # (T,1)
        width = width.reshape(T, 1, 1)  # (T,1,1)
        return centered / (width + 1e-6)

    def pad_or_crop(self, arr, target_T):
        # arr: (T,68,2)
        t = arr.shape[0]
        if t == target_T:
            return arr
        if t > target_T:
            return arr[:target_T]
        # pad with last valid frame (repeat last)
        pad_count = target_T - t
        pad = np.repeat(arr[-1][np.newaxis, ...], pad_count, axis=0)
        return np.concatenate([arr, pad], axis=0)

    def partition_regions(self, landmarks):
        """
        Split 68 points into 6 anatomical regions.
        Returns dict with keys:
        'ljaw','rjaw','leye','reye','nose','mouth'
        Each value shape: (T, N_i, 2)
        NOTE: index sets chosen reasonably; can be adjusted.
        """
        # Landmark indices (0-based)
        mouth_idx = list(range(48, 68))       # 20
        nose_idx = list(range(27, 36))        # 9
        leye_idx = list(range(17, 22)) + list(range(36, 42))# 6
        reye_idx = list(range(22, 27))+list(range(42, 48)) # 6
        # split jaw: left(0..8) right(9..16)
        ljaw_idx = list(range(0, 9))          # 9
        rjaw_idx = list(range(9, 17))         # 8

        regions = {
            'ljaw': landmarks[:, ljaw_idx, :].astype(np.float32),
            'rjaw': landmarks[:, rjaw_idx, :].astype(np.float32),
            'leye': landmarks[:, leye_idx, :].astype(np.float32),
            'reye': landmarks[:, reye_idx, :].astype(np.float32),
            'nose': landmarks[:, nose_idx, :].astype(np.float32),
            'mouth': landmarks[:, mouth_idx, :].astype(np.float32)
        }
        return regions