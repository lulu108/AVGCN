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
                 uniform_ratio=0.5, dvlog_aug_noise_std=0.005, dvlog_audio_mask_ratio=0.03,
                 segment_len=0, use_two_clip_train=False, two_clip_mode='center_random',
                 use_three_clip_train=False, three_clip_mode='center_lr',
                 quality_candidate_count=5,
                 region_partition_scheme='legacy6'):
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
            uniform_ratio: 关键帧采样中"均匀采样"的比例（默认 0.5）
            dvlog_aug_noise_std: DVLOG 训练噪声标准差（默认 0.005）
            dvlog_audio_mask_ratio: DVLOG 训练时 audio 时间掩码比例（默认 0.03）
            segment_len: DVLOG 训练期固定切片长度（0=关闭）
            region_partition_scheme: landmark 分区方案 ("legacy6" | "symptom7")
        """
        self.file_list = list(file_list)
        self.video_feature_path = video_feature_path
        self.audio_feature_path = audio_feature_path
        self.face_landmark_path = face_landmark_path
        self.label_path = label_path
        self.T_target = int(T_target)
        self.mode = mode
        self.dataset = dataset.upper()  # 统一大写
        self.uniform_ratio = float(uniform_ratio)
        self.dvlog_aug_noise_std = float(dvlog_aug_noise_std)
        self.dvlog_audio_mask_ratio = float(dvlog_audio_mask_ratio)
        self.segment_len = int(segment_len)
        self.use_two_clip_train = bool(use_two_clip_train)
        self.two_clip_mode = str(two_clip_mode)
        self.use_three_clip_train = bool(use_three_clip_train)
        self.three_clip_mode = str(three_clip_mode)
        self.quality_candidate_count = int(quality_candidate_count)
        self.region_partition_scheme = str(region_partition_scheme)
        self._segment_debug_printed = False
        
        # 【优化】在 __init__ 中一次性缓存所有标签，避免每次 __getitem__ 都读 CSV
        self._label_cache = {}
        if self.dataset == 'DVLOG':
            _ldf = pd.read_csv(label_path)
            _ldf['index'] = _ldf['index'].astype(str)
            _ldf['label'] = _ldf['label'].astype(str).str.strip().str.lower()
            for _, row in _ldf.iterrows():
                self._label_cache[row['index']] = 1 if row['label'] == 'depression' else 0
            print(f"[MultiModalDataLoader] DVLOG 标签已缓存: {len(self._label_cache)} 条")
            print(f"[MultiModalDataLoader] DVLOG aug config: noise_std={self.dvlog_aug_noise_std}, "
                f"audio_mask_ratio={self.dvlog_audio_mask_ratio}, uniform_ratio={self.uniform_ratio}")
            if self.mode == 'train' and self.segment_len > 0:
                print(f"[MultiModalDataLoader] DVLOG train fixed segment_len={self.segment_len}, T_target={self.T_target}")
        else:
            # LMVD: 预读所有 _Depression.csv
            for fn in self.file_list:
                fr = os.path.splitext(fn)[0]
                lf = os.path.join(label_path, fr + '_Depression.csv')
                if os.path.exists(lf):
                    self._label_cache[fr] = int(pd.read_csv(lf, header=None).iloc[0, 0])
            print(f"[MultiModalDataLoader] LMVD 标签已缓存: {len(self._label_cache)} 条")
        
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

        # ===== 1.1 对齐视频/音频时间长度（DVLOG 个别样本存在 T mismatch） =====
        # 统一截断到 min(T_v, T_a)，避免后续 segment slicing 采样落在错位区间。
        if self.dataset == 'DVLOG':
            t_v, t_a = input_video.shape[0], input_audio.shape[0]
            t_min = min(t_v, t_a)
            if t_v != t_a:
                input_video = input_video[:t_min]
                input_audio = input_audio[:t_min]

        orig_T = input_video.shape[0]  # pad/crop 前的真实帧数（对齐后）

        # 处理 NaN（先于 copy，确保 raw 也是干净数值）
        input_video = np.nan_to_num(input_video, nan=0.0, posinf=1.0, neginf=-1.0)
        input_audio = np.nan_to_num(input_audio, nan=0.0, posinf=1.0, neginf=-1.0)

        # ── 在 pad/crop 之前保留原始长度副本，供 face 分支使用 ──────────────────
        # 关键：input_video_raw 保持 (orig_T, 136)，不做 pad/crop；
        # _process_face_landmarks_array 会自己完成采样/截断/pad 到 T_target。
        input_video_raw = input_video.copy()   # shape: (orig_T, 136)

        # ===== [Three-Clip] 训练时返回三段（center + left/right 或 center + quality + jitter），提前 return =====
        # 仅 DVLOG + train + use_three_clip_train + segment_len > 0 时生效
        if self.use_three_clip_train and self.dataset == 'DVLOG' and self.mode == 'train' and self.segment_len > 0:
            if self.use_two_clip_train:
                raise RuntimeError("use_three_clip_train and use_two_clip_train cannot both be True.")
            _label = self._label_cache.get(file_root)
            if _label is None:
                raise FileNotFoundError(f"Label not found for '{file_root}' in cache.")
            L = self.segment_len
            max_start = max(0, orig_T - L)
            center_start = max_start // 2

            if orig_T > L:
                if self.three_clip_mode == 'center_lr':
                    delta = max(1, L // 2)
                    left_start = max(0, center_start - delta)
                    right_start = min(max_start, center_start + delta)
                    starts = [center_start, left_start, right_start]
                elif self.three_clip_mode == 'center_quality_jitter':
                    jitter = min(64, max_start // 4)
                    low = max(0, center_start - jitter)
                    high = min(max_start, center_start + jitter)
                    jitter_start = int(np.random.randint(low, high + 1)) if high > low else center_start
                    quality_start = self._pick_quality_start(
                        input_video_raw, input_audio, orig_T, L, center_start, max_start
                    )
                    starts = [center_start, quality_start, jitter_start]
                else:
                    raise ValueError(f"Unknown three_clip_mode: {self.three_clip_mode}")

                v1, a1, raw1, t1 = self._slice_segment_by_start(
                    input_video, input_audio, input_video_raw, orig_T, starts[0], L
                )
                v2, a2, raw2, t2 = self._slice_segment_by_start(
                    input_video, input_audio, input_video_raw, orig_T, starts[1], L
                )
                v3, a3, raw3, t3 = self._slice_segment_by_start(
                    input_video, input_audio, input_video_raw, orig_T, starts[2], L
                )
            else:
                v1, a1, raw1, t1 = input_video.copy(), input_audio.copy(), input_video_raw.copy(), orig_T
                v2, a2, raw2, t2 = input_video.copy(), input_audio.copy(), input_video_raw.copy(), orig_T
                v3, a3, raw3, t3 = input_video.copy(), input_audio.copy(), input_video_raw.copy(), orig_T

            if not self._segment_debug_printed:
                print(f"[SEG-GETITEM][ThreeClip] mode={self.three_clip_mode}, center={center_start}, "
                      f"seg_len={L}, orig_T={orig_T}")
                self._segment_debug_printed = True

            clip1 = self._process_single_clip(v1, a1, raw1, t1, _label)
            clip2 = self._process_single_clip(v2, a2, raw2, t2, _label)
            clip3 = self._process_single_clip(v3, a3, raw3, t3, _label)
            return clip1, clip2, clip3

        # ===== [Two-Clip] 训练时返回两段（中心段 + 随机/jitter 段），提前 return =====
        # 仅 DVLOG + train + use_two_clip_train + segment_len > 0 时生效
        if self.use_two_clip_train and self.dataset == 'DVLOG' and self.mode == 'train' and self.segment_len > 0:
            _label = self._label_cache.get(file_root)
            if _label is None:
                raise FileNotFoundError(f"Label not found for '{file_root}' in cache.")
            L = self.segment_len
            max_start = max(0, orig_T - L)
            center_start = max_start // 2
            if orig_T > L:
                if self.two_clip_mode == 'center_random':
                    b_start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
                elif self.two_clip_mode == 'center_jitter':
                    _jitter = min(64, max_start // 4)
                    _low  = max(0, center_start - _jitter)
                    _high = min(max_start, center_start + _jitter)
                    b_start = int(np.random.randint(_low, _high + 1)) if _high > _low else center_start
                elif self.two_clip_mode == 'center_quality':
                    b_start = self._pick_quality_start(input_video_raw, input_audio, orig_T, L, center_start, max_start)
                else:
                    raise ValueError(f"Unknown two_clip_mode: {self.two_clip_mode}")
                v1, a1, raw1, t1 = self._slice_segment_by_start(
                    input_video, input_audio, input_video_raw, orig_T, center_start, L
                )
                v2, a2, raw2, t2 = self._slice_segment_by_start(
                    input_video, input_audio, input_video_raw, orig_T, b_start, L
                )
            else:
                # 序列不足 L：两段均使用完整序列（由 _process_single_clip 内 pad 到 T_target）
                v1, a1, raw1, t1 = input_video.copy(), input_audio.copy(), input_video_raw.copy(), orig_T
                v2, a2, raw2, t2 = input_video.copy(), input_audio.copy(), input_video_raw.copy(), orig_T
                b_start = 0
            if not self._segment_debug_printed:
                print(f"[SEG-GETITEM][TwoClip] center={center_start}, clip_b={b_start}, "
                      f"seg_len={L}, orig_T={orig_T}, mode={self.two_clip_mode}")
                self._segment_debug_printed = True
            clip1 = self._process_single_clip(v1, a1, raw1, t1, _label)
            clip2 = self._process_single_clip(v2, a2, raw2, t2, _label)
            return clip1, clip2

        # ===== 1.2 DVLOG 训练固定长度切片（优先打掉 length/pad confound） =====
        # 仅在 train 生效：orig_T > L 随机裁一段；orig_T <= L 保留原序列，后续 pad 到 T_target。
        if self.dataset == 'DVLOG' and self.mode == 'train' and self.segment_len > 0:
            L = self.segment_len
            if orig_T > L:
                max_start = orig_T - L

                # 以中心段为基准，只在小范围内做随机抖动
                # 目的：消除"随机起点差异过大"引起的 seed 方差，同时保留轻量增强
                center_start = max_start // 2
                jitter = min(64, max_start // 4)   # 最大抖动幅度：64帧 or max_start/4 取小
                low  = max(0, center_start - jitter)
                high = min(max_start, center_start + jitter)

                start = np.random.randint(low, high + 1) if high > low else center_start
                end = start + L
                input_video = input_video[start:end]
                input_audio = input_audio[start:end]
                input_video_raw = input_video_raw[start:end]
                orig_T = L
                if not self._segment_debug_printed:
                    print(f"[SEG-GETITEM] apply center-jitter segment: start={start}, end={end}, "
                          f"seg_len={L}, center={center_start}, jitter={jitter}")
                    self._segment_debug_printed = True

        # Pad/Crop 到目标长度（只针对 ViT 输入的 video / audio）
        input_video = self._pad_or_crop(input_video, self.T_target)
        input_audio = self._pad_or_crop(input_audio, self.T_target)
        # input_video_raw 故意不做 pad/crop，保留 orig_T

        # DVLOG 数值归一化（避免数值过大导致 NaN）
        if self.dataset == 'DVLOG':
            input_video = self._normalize_features(input_video)
            input_audio = self._normalize_features(input_audio)
        
        # ===== 2. 加载/提取面部关键点并处理 =====
        if self.dataset == 'DVLOG':
            # D-Vlog: 面部关键点已包含在 visual 文件中（136维 = 68点×2坐标）
            # 使用 pad/crop 之前的原始序列（orig_T 帧），让 _process_face_landmarks_array
            # 自己完成插值 → 关键帧采样（当 orig_T > T_target）→ pad/crop 到 T_target。
            # 这样 face 分支的 validity / 采样逻辑才与真实时序对齐。
            # 关键：DVLOG 的 136 维是 [x0..x67, y0..y67]，必须按 split 方式解码，不能直接 reshape。
            face_landmarks = self._dvlog_seq136_to_pts68x2(input_video_raw)   # (orig_T, 68, 2)
            face_regions   = self._process_face_landmarks_array(face_landmarks)
            # actual_len：真实有效帧数（超长时截断到 T_target）
            actual_len = min(orig_T, self.T_target)

            # ===== 2.1 质量指标（quality scores） =====
            # q_v: 视频质量（硬阈值版）
            #   - v_zero_frac 高（接近全零）视为坏模态
            v_zero_frac = float(np.mean(np.isclose(input_video_raw, 0.0)))
            q_v = 1.0 if v_zero_frac < 0.95 else 0.0

            # q_g: 面部质量（使用关键点有效率均值）
            #   raw face (T,68,2) -> 每帧有效点比例 -> 再对时间做平均
            _pt_valid = np.any(face_landmarks != 0, axis=-1)      # (T,68)
            _valid_ratio_t = _pt_valid.mean(axis=1).astype(np.float32)  # (T,)
            q_g = float(_valid_ratio_t.mean()) if _valid_ratio_t.size > 0 else 0.0

            # pad 后质量统计（用于样本级降权/skip 规则）
            _region_any = face_regions.get('mouth', next(iter(face_regions.values())))  # (T,N,C)
            if _region_any.shape[-1] >= 11:
                _valid_pad = _region_any[:, 0, -1].astype(np.float32)                    # (T_target,)
                # 仅统计有效帧区间 [:actual_len]，避免把“短”误判为“强缺失”
                _eff = _valid_pad[:max(1, int(actual_len))]
                face_valid_eff_mean = float(_eff.mean())
                face_low02_eff = float((_eff < 0.2).mean())
            else:
                face_valid_eff_mean = q_g
                face_low02_eff = 0.0
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
                stride=1,
                region_partition_scheme=self.region_partition_scheme
            )
            face_regions, _, actual_len = face_dataset[0]
            face_regions = self._ensure_validity_channel(face_regions)
            # LMVD 默认质量全开（后续可扩展为真实质量估计）
            q_v, q_g = 1.0, 1.0
            face_valid_eff_mean = 1.0
            face_low02_eff = 0.0
        
        # ===== 3. 读取标签（从 __init__ 缓存中读取，避免重复 I/O）=====
        label = self._label_cache.get(file_root)
        if label is None:
            raise FileNotFoundError(f"Label not found for '{file_root}' in cache. "
                                    f"Dataset={self.dataset}, cached keys (first 10): "
                                    f"{list(self._label_cache.keys())[:10]}...")
        
        # ===== 2.2 强缺失判定 & 样本权重 =====
        # 仅基于“有效帧内”的 face 质量判定强缺失，避免“短样本=缺失”误判：
        #   - face_valid_eff_mean < 0.05
        #   - 或 face_low02_eff > 0.9
        hard_missing = (
            (face_valid_eff_mean < 0.05) or
            (face_low02_eff > 0.9)
        )
        sample_w = 0.2 if hard_missing else 1.0

        # ===== 4. 转为 Tensor =====
        input_video = torch.from_numpy(input_video).float()
        input_audio = torch.from_numpy(input_audio).float()
        actual_len_tensor = torch.tensor(actual_len, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        quality_tensor = torch.tensor([q_v, q_g], dtype=torch.float32)  # [q_v, q_g]
        sample_weight_tensor = torch.tensor(sample_w, dtype=torch.float32)
        
        # ===== 5. 训练模式数据增强 =====
        if self.mode == 'train':
            if self.dataset == 'DVLOG':
                # DVLOG 可控弱增强：默认保持历史配置（0.005 / 0.03）
                # 也可在外部传入 0.0 / 0.0 关闭增强做稳定性对照
                if self.dvlog_aug_noise_std > 0:
                    noise_video = torch.randn_like(input_video) * self.dvlog_aug_noise_std
                    input_video += noise_video
                    noise_audio = torch.randn_like(input_audio) * self.dvlog_aug_noise_std
                    input_audio += noise_audio

                if self.dvlog_audio_mask_ratio > 0:
                    T_aug = input_audio.shape[0]
                    mask_len = max(1, int(T_aug * self.dvlog_audio_mask_ratio))
                    start = np.random.randint(0, max(1, T_aug - mask_len))
                    input_audio[start:start + mask_len, :] = 0   # 只掩 audio，不动 video
            else:
                # LMVD 数据集：保留原增强强度（数据量更大，扛得住）
                noise_video = torch.randn_like(input_video) * 0.02
                input_video += noise_video
                noise_audio = torch.randn_like(input_audio) * 0.02
                input_audio += noise_audio

                T_aug = input_video.shape[0]
                mask_len = int(T_aug * 0.1)
                start = np.random.randint(0, T_aug - mask_len)
                input_video[start:start + mask_len, :] = 0
                input_audio[start:start + mask_len, :] = 0
        
        return input_video, input_audio, face_regions, actual_len_tensor, label_tensor, quality_tensor, sample_weight_tensor

    def _ensure_validity_channel(self, face_regions):
        """
        兼容 LMVD 旧版 10 维 landmarks：补一个全 1 的 validity 通道。
        face_regions: dict of {region: (T, N, C)}
        """
        if not isinstance(face_regions, dict):
            return face_regions
        updated = {}
        for key, arr in face_regions.items():
            if not isinstance(arr, np.ndarray) or arr.ndim != 3:
                updated[key] = arr
                continue
            if arr.shape[-1] == 10:
                ones = np.ones((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
                updated[key] = np.concatenate([arr, ones], axis=-1)
            else:
                updated[key] = arr
        return updated
    
    def _process_face_landmarks_array(self, raw_landmarks):
        """
        直接处理面部关键点数组，复用 DepressionFaceDataset 的处理逻辑（插值、归一化、区域划分）
        raw_landmarks: (T, 68, 2)
        Returns:
            face_regions: dict
        """
        from _01DatasetLoader import DepressionFaceDataset

        raw = raw_landmarks
        # 在插值前统计人脸有效度：每帧有效关键点比例（0~1）
        raw0 = raw_landmarks
        pt_valid = np.any(raw0 != 0, axis=-1)            # (T, 68) bool
        valid_ratio = pt_valid.mean(axis=1).astype(np.float32)  # (T,)

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

        # 2) 关键帧采样（D-Vlog）：　50% 均匀 + 50% top-motion」混合采样
        #    纯 top-motion 会系统性偏向“大动作帧”，丢失低幅度抑郁线索（长时间低活跃模式）。
        #    均匀保底确保时间轴全程覆盖，motion 补强保留动态信息较强的帧。
        if self.T_target is not None and T_len > self.T_target:
            n_total = self.T_target
            n_u = int(round(n_total * self.uniform_ratio))
            n_u = max(1, min(n_total - 1, n_u))              # 保证 1 <= n_u <= n_total-1
            n_m = n_total - n_u                              # top-motion 采样帧数

            # ---- 均匀采样 ----
            uniform_indices = np.linspace(0, T_len - 1, n_u, dtype=int)

            # ---- top-motion 采样 ----
            diff = np.diff(raw, axis=0)
            motion_scores = np.linalg.norm(diff, axis=(1, 2))
            motion_scores = np.concatenate(([0], motion_scores))
            motion_indices = np.argsort(motion_scores)[-n_m:]

            # ---- 合并去重 + 时序排序 ----
            merged = np.union1d(uniform_indices, motion_indices)
            # union1d 可能比 n_total 少（重叠去除后），按均匀步长补充
            if len(merged) < n_total:
                extra = np.linspace(0, T_len - 1, n_total, dtype=int)
                merged = np.union1d(merged, extra)[:n_total]
            # 超出则按 motion score 降序截取（保留最具多样性的帧）
            if len(merged) > n_total:
                keep_scores = motion_scores[merged]
                merged = merged[np.argsort(keep_scores)[-n_total:]]
            merged = np.sort(merged)                        # 恢复时间顺序
            raw = raw[merged]
            valid_ratio = valid_ratio[merged]

        # 3) 归一化与区域划分
        helper_ds = DepressionFaceDataset([], [], mode='dvlog', T_target=self.T_target,
                                          augment=(self.mode == 'train'), adaptive_sampling=False, stride=1,
                                          region_partition_scheme=self.region_partition_scheme)
        landmarks = helper_ds.normalize_face(raw)
        landmarks = helper_ds.pad_or_crop(landmarks, self.T_target)
        # 将 validity 作为额外 1 维特征拼接： (T, 68, 10) -> (T, 68, 11)
        # valid_ratio 是 1D (T,)，用自身的 _pad_or_crop（支持 2D）而非 helper_ds.pad_or_crop（3D）
        valid_ratio_pad = self._pad_or_crop(valid_ratio.reshape(-1, 1), self.T_target).astype(np.float32)
        valid_feat = np.repeat(valid_ratio_pad[:, None, :], landmarks.shape[1], axis=1)  # (T, 68, 1)
        landmarks = np.concatenate([landmarks, valid_feat], axis=-1)
        regions = helper_ds.partition_regions(landmarks, scheme=self.region_partition_scheme)

        return regions

    def _score_segment_quality(self, input_video_raw, input_audio, start, L):
        """
        给候选窗口打分：validity 优先，motion 次之，audio 作为补充。
        返回一个标量分数，越大越好。
        """
        end = min(start + L, input_video_raw.shape[0])
        if end <= start:
            return -1e9

        # Face validity: 帧内有效点比例的均值
        seg_raw = input_video_raw[start:end]
        pts = self._dvlog_seq136_to_pts68x2(seg_raw)
        pt_valid = np.any(pts != 0, axis=-1)  # (T, 68)
        valid_ratio = float(pt_valid.mean())

        # Motion energy: 关键点差分位移均值
        if pts.shape[0] > 1:
            diff = np.diff(pts, axis=0)
            motion = float(np.linalg.norm(diff, axis=(1, 2)).mean())
        else:
            motion = 0.0

        # Audio energy: 作为补充信号
        if input_audio is not None and end <= input_audio.shape[0]:
            audio_seg = input_audio[start:end]
            audio_energy = float(np.mean(np.abs(audio_seg)))
        else:
            audio_energy = 0.0

        # 加权：validity 为主，motion 次之，audio 轻量加权
        return 0.7 * valid_ratio + 0.25 * motion + 0.05 * audio_energy

    def _pick_quality_start(self, input_video_raw, input_audio, orig_T, L, center_start, max_start):
        """从若干候选窗口中挑选质量最高的一段。"""
        n = max(3, self.quality_candidate_count)
        # 候选 start: center + 均匀采样 + 一个随机
        uniform_starts = np.linspace(0, max_start, num=n, dtype=int).tolist() if max_start > 0 else [0]
        rand_start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
        candidates = list(dict.fromkeys([center_start] + uniform_starts + [rand_start]))

        best_s = center_start
        best_score = -1e9
        for s in candidates:
            score = self._score_segment_quality(input_video_raw, input_audio, s, L)
            if score > best_score:
                best_score = score
                best_s = int(s)
        return best_s

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

    def _slice_segment_by_start(self, input_video, input_audio, input_video_raw, orig_T, start, L):
        """按起点裁剪三路数组为长度 L 的段（返回深拷贝，避免两段之间的 alias）。"""
        end  = min(start + L, orig_T)
        return (
            input_video[start:end].copy(),
            input_audio[start:end].copy(),
            input_video_raw[start:end].copy(),
            end - start,
        )

    def _dvlog_seq136_to_pts68x2(self, seq136):
        """
        DVLOG landmark 解码：
          输入  (T,136) : [x0..x67, y0..y67]
          输出  (T,68,2): [...,0]=x, [...,1]=y
        """
        arr = np.asarray(seq136)
        if arr.ndim != 2 or arr.shape[1] != 136:
            raise ValueError(f"[DVLOG decode] expect (T,136), got {arr.shape}")
        x = arr[:, :68]
        y = arr[:, 68:]
        return np.stack([x, y], axis=-1).astype(np.float32)

    def _process_single_clip(self, input_video, input_audio, input_video_raw, orig_T, label):
        """
        从已裁剪的原始数组构建完整单样本 7-tuple，供 Two-Clip Training 使用。
        复用 __getitem__ 的 pad/normalize/face/quality/aug 流程，仅支持 DVLOG。

        Returns:
            (video_t, audio_t, face_regions, actual_len_t, label_t, quality_t, sw_t)
        """
        # 1. Pad/Crop + Normalize
        v_pad = self._pad_or_crop(input_video, self.T_target)
        a_pad = self._pad_or_crop(input_audio, self.T_target)
        v_pad = self._normalize_features(v_pad)
        a_pad = self._normalize_features(a_pad)

        # 2. Face 处理（使用裁剪前原始坐标）
        face_landmarks = self._dvlog_seq136_to_pts68x2(input_video_raw)
        face_regions   = self._process_face_landmarks_array(face_landmarks)
        actual_len     = min(orig_T, self.T_target)

        # 3. Quality 估计
        v_zero_frac = float(np.mean(np.isclose(input_video_raw, 0.0)))
        q_v         = 1.0 if v_zero_frac < 0.95 else 0.0
        _pt_valid   = np.any(face_landmarks != 0, axis=-1)          # (T', 68)
        q_g         = float(_pt_valid.mean(axis=1).mean()) if _pt_valid.size > 0 else 0.0
        _region_any = face_regions.get('mouth', next(iter(face_regions.values())))
        if _region_any.shape[-1] >= 11:
            _valid_pad          = _region_any[:, 0, -1].astype(np.float32)
            _eff                = _valid_pad[:max(1, int(actual_len))]
            face_valid_eff_mean = float(_eff.mean())
            face_low02_eff      = float((_eff < 0.2).mean())
        else:
            face_valid_eff_mean = q_g
            face_low02_eff      = 0.0

        # 4. Sample weight
        hard_missing = (face_valid_eff_mean < 0.05) or (face_low02_eff > 0.9)
        sample_w     = 0.2 if hard_missing else 1.0

        # 5. Tensor 转换
        v_t     = torch.from_numpy(v_pad).float()
        a_t     = torch.from_numpy(a_pad).float()
        len_t   = torch.tensor(actual_len, dtype=torch.long)
        label_t = torch.tensor(label,      dtype=torch.long)
        q_t     = torch.tensor([q_v, q_g], dtype=torch.float32)
        sw_t    = torch.tensor(sample_w,   dtype=torch.float32)

        # 6. 数据增强（仅 train）
        if self.mode == 'train':
            if self.dvlog_aug_noise_std > 0:
                v_t = v_t + torch.randn_like(v_t) * self.dvlog_aug_noise_std
                a_t = a_t + torch.randn_like(a_t) * self.dvlog_aug_noise_std
            if self.dvlog_audio_mask_ratio > 0:
                T_aug    = a_t.shape[0]
                mask_len = max(1, int(T_aug * self.dvlog_audio_mask_ratio))
                s_aug    = np.random.randint(0, max(1, T_aug - mask_len))
                a_t[s_aug:s_aug + mask_len, :] = 0

        return v_t, a_t, face_regions, len_t, label_t, q_t, sw_t


def _collate_single_multimodal(batch):
    """单 clip batch collate（原 collate_fn_multimodal 核心逻辑）。"""
    video_list, audio_list, face_list, len_list, label_list, quality_list, sw_list = zip(*batch)

    video_batch   = torch.stack(video_list)
    audio_batch   = torch.stack(audio_list)
    actual_lens   = torch.stack(len_list)
    labels        = torch.stack(label_list)
    quality       = torch.stack(quality_list)
    sample_weight = torch.stack(sw_list)

    region_keys        = face_list[0].keys()
    face_regions_batch = {}
    for key in region_keys:
        regions = [face[key] for face in face_list]
        face_regions_batch[key] = torch.stack(
            [torch.from_numpy(r) if not torch.is_tensor(r) else r for r in regions]
        )

    return video_batch, audio_batch, face_regions_batch, actual_lens, labels, quality, sample_weight


def collate_fn_multimodal(batch):
    """
    自定义 collate 函数，兼容两种返回格式：
      - 单 clip：每个样本为 7-tuple
      - Two-Clip：每个样本为 (clip1_7tuple, clip2_7tuple)

    Returns:
        单 clip : (video, audio, face_regions, actual_lens, labels, quality, sample_weight)
        双 clip : 由两个上述 tuple 组成的 tuple
    """
    # Three-clip case: batch[0] = (clip1, clip2, clip3)
    if isinstance(batch[0], tuple) and len(batch[0]) == 3 and isinstance(batch[0][0], tuple):
        clip1_list = [x[0] for x in batch]
        clip2_list = [x[1] for x in batch]
        clip3_list = [x[2] for x in batch]
        return (
            _collate_single_multimodal(clip1_list),
            _collate_single_multimodal(clip2_list),
            _collate_single_multimodal(clip3_list),
        )

    # Two-clip case: batch[0] = (clip1_tuple, clip2_tuple)
    if isinstance(batch[0], tuple) and len(batch[0]) == 2 and isinstance(batch[0][0], tuple):
        clip1_list = [x[0] for x in batch]
        clip2_list = [x[1] for x in batch]
        return _collate_single_multimodal(clip1_list), _collate_single_multimodal(clip2_list)

    # Single-clip case
    return _collate_single_multimodal(batch)
