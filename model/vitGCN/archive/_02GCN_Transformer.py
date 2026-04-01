import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv
from torch.nn import MultiheadAttention

# ── 低人脸有效度阈值 ────────────────────────────────────────────────────────────
# valid_ratio < FACE_VALID_THRESH 的帧视为无效（即该帧人脸追踪质量过差）。
# 含义：该帧中 68 个关键点里检测到的比例 < 20% 时认为噪声过高。
# LMVD 数据集的 validity 通道全为 1.0，此阈值对其无影响。
# 调节范围：0.0（关闭此 mask）~ 0.5（严格过滤）。
FACE_VALID_THRESH = 0.2

# ── 分区方案定义（两种预设，可通过 AnatomicalGCN(region_scheme=...) 切换） ──────────
# legacy6：经典 6 区（眉+眼混合、鼻整段）
_LEGACY6_REGION_NAMES   = ['ljaw', 'rjaw', 'leye', 'reye', 'nose', 'mouth']
_LEGACY6_REGION_INDICES = {
    'ljaw':  list(range(0, 9)),
    'rjaw':  list(range(8, 17)),
    'leye':  list(range(17, 22)) + list(range(36, 42)),
    'reye':  list(range(22, 27)) + list(range(42, 48)),
    'nose':  list(range(27, 36)),
    'mouth': list(range(48, 68)),
}
_LEGACY6_PRIORS = {'ljaw': 0.30, 'rjaw': 0.30, 'leye': 0.80, 'reye': 0.80, 'nose': 0.40, 'mouth': 1.00}

# symptom7：症状导向 7 区（纯眼、眉+鼻梁桥接、下鼻、嘴闭环）
_SYMPTOM7_REGION_NAMES   = ['ljaw', 'rjaw', 'leye', 'reye', 'brow_glabella', 'nose_lower', 'mouth']
_SYMPTOM7_REGION_INDICES = {
    'ljaw':          list(range(0, 9)),
    'rjaw':          list(range(8, 17)),
    'leye':          list(range(36, 42)),
    'reye':          list(range(42, 48)),
    'brow_glabella': list(range(17, 27)) + list(range(27, 31)),  # 左右眉 + 鼻梓上段
    'nose_lower':    list(range(31, 36)),
    'mouth':         list(range(48, 68)),
}
_SYMPTOM7_PRIORS = {
    'ljaw': 0.25, 'rjaw': 0.25, 'leye': 0.85, 'reye': 0.85,
    'brow_glabella': 1.00, 'nose_lower': 0.20, 'mouth': 1.10,
}

# 统一查计表
_REGION_NAMES_MAP   = {'legacy6': _LEGACY6_REGION_NAMES,   'symptom7': _SYMPTOM7_REGION_NAMES}
_REGION_INDICES_MAP = {'legacy6': _LEGACY6_REGION_INDICES, 'symptom7': _SYMPTOM7_REGION_INDICES}
_REGION_PRIORS      = {'legacy6': _LEGACY6_PRIORS,          'symptom7': _SYMPTOM7_PRIORS}
# ────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]

class AnatomicalGCN(nn.Module):
    """
    输入: region_dict, 每个 value 是 tensor (B, T, N, 2)
    输出: logits (B, num_classes)
    处理流程：
      - 对每个 region 单独做 GCN（节点特征为 (x,y)）
      - 每帧对节点做 mean pooling -> 每帧每 region 得到向量
      - 将 6 region 向量拼成 seq_len=6 的 token，送 transformer encoder
      - transformer 输出后，按时间聚合 -> 分类
    """
    def __init__(self,
                 out_dim=32,
                 nhead=4, 
                 num_classes=2,
                 device='cpu',
                 face_valid_thresh=0.2,
                 use_global_branch=True,
                 global_alpha_init=0.2,
                 region_fusion_mode='cross_attn',
                 gcn_temporal_mode='transformer',
                 region_mlp_dropout=0.1,
                 tcn_kernel_size=3,
                 region_scheme="legacy6"):
        """Args:
            face_valid_thresh: 人脸关键点有效度阈值（对照实验用）
              T1=0.1 宽松，减少随机性；T2=0.3 严格，减少噪声。
              默认 0.2，对应模块级 FACE_VALID_THRESH 的原始値。
            use_global_branch: 是否开启全局坐标投影支路。
              True  → 开启，全局特征经 global_alpha 门控后与区域特征拼接。
              False → 关闭，拼接位填零向量（fused_dim 不变）。
            global_alpha_init: global_alpha 初始値，建议：
              DVLOG=0.2（保守，tanh(0.2)≈0.197）、LMVD=0.5（tanh(0.5)≈0.462）。
        """
        super().__init__()
        self.out_dim = out_dim
        self.device = device
        self.face_valid_thresh = face_valid_thresh  # A4: 实例级阈值，可从外部控制
        self.use_global_branch = use_global_branch
        self.region_fusion_mode = region_fusion_mode
        self.gcn_temporal_mode = gcn_temporal_mode

        if self.region_fusion_mode not in {'cross_attn', 'concat_linear', 'mlp'}:
            raise ValueError(
                f"Unknown region_fusion_mode: {self.region_fusion_mode!r}. "
                "Valid: ['cross_attn', 'concat_linear', 'mlp']"
            )
        if self.gcn_temporal_mode not in {'transformer', 'meanmax', 'tcn'}:
            raise ValueError(
                f"Unknown gcn_temporal_mode: {self.gcn_temporal_mode!r}. "
                "Valid: ['transformer', 'meanmax', 'tcn']"
            )
        # 可学习标量门：有效权重 = tanh(global_alpha) ∈ (-1, 1)
        # 注意：当 alpha 训练中变负时，权重也会变负（可视为“反向校正”语义）。
        # 若需严格的 (0,1) 强弱门控，可把这里改为 sigmoid。
        # 当前选择 tanh：允许负权重，训练更灵活。
        self.global_alpha = nn.Parameter(torch.tensor(float(global_alpha_init)))
        # 按 region_scheme 确定区域名称列表和节点索引映射
        self.region_scheme = region_scheme
        if region_scheme not in _REGION_NAMES_MAP:
            raise ValueError(f"Unknown region_scheme: {region_scheme!r}. Valid: {list(_REGION_NAMES_MAP)}")
        self.region_names   = _REGION_NAMES_MAP[region_scheme]
        self.region_indices = _REGION_INDICES_MAP[region_scheme]
        # 1. 区域特征维度现在是 out_dim * 2 (即 64)
        region_feat_dim = out_dim * 2
        self.region_feat_dim = region_feat_dim
        self.num_regions = len(self.region_names)
        
        # [已简化] 区域内自注意力对单 token (B*T,1,D) 做 QKV 无实质建模意义，
        # 只增参数和残差扰动，故注释掉。若需回滚可恢复以下代码。
        # self.intra_region_attention = MultiheadAttention(
        #     embed_dim=region_feat_dim,
        #     num_heads=2,
        #     batch_first=True,
        #     dropout=0.1
        # )
        
        # (修改) 定义跨区域全局注意力层 - 用于建模不同区域间的交互
        self.cross_attention = MultiheadAttention(
            embed_dim=region_feat_dim, 
            num_heads=nhead, 
            batch_first=True,
            dropout=0.1
        )
        region_concat_dim = region_feat_dim * self.num_regions
        self.region_concat_linear = nn.Sequential(
            nn.Linear(region_concat_dim, region_feat_dim),
            nn.LayerNorm(region_feat_dim),
            nn.ReLU()
        )
        self.region_concat_mlp = nn.Sequential(
            nn.Linear(region_concat_dim, region_concat_dim),
            nn.GELU(),
            nn.Dropout(region_mlp_dropout),
            nn.Linear(region_concat_dim, region_feat_dim)
        )
        self.region_concat_mlp_norm = nn.LayerNorm(region_feat_dim)
        # GCN 层：按 region_names 动态构建，支持 legacy6 / symptom7 两种方案
        # 输入维度=11（坐月2+速度2+速琛1+距离4+对称1+validity1），输出=out_dim
        self.gcn1 = nn.ModuleDict({name: GCNConv(11, out_dim)      for name in self.region_names})
        self.gcn2 = nn.ModuleDict({name: GCNConv(out_dim, out_dim) for name in self.region_names})

        # Phase A: 区域先验 + 可学习标量（正权重）
        # region token 顺序与 self.region_names 严格一致，随 region_scheme 自动切换
        _priors    = _REGION_PRIORS[region_scheme]
        prior_init = torch.tensor([_priors[n] for n in self.region_names], dtype=torch.float32)
        num_regions = len(self.region_names)
        # 用 softplus 参数化正权重：w = softplus(logits) > 0
        # inverse_softplus(x) = log(exp(x)-1)
        self.region_logits = nn.Parameter(torch.log(torch.expm1(prior_init)).view(1, num_regions, 1))
        # Phase B: sample-wise gate（每个样本每个区域一个 [0,1] 门控）
        # 最终权重 = prior_weight * sample_gate
        self.use_sample_region_gate = True
        gate_hidden = max(8, region_feat_dim // 2)
        self.region_sample_gate = nn.Sequential(
            nn.Linear(region_feat_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid()
        )

        # 全局动态支路（motion/global dynamics）
        # 输入改为每帧 4 维动态描述子：
        # [centroid_vel, centroid_acc, global_energy, mouth_eye_coord]
        self.global_branch = nn.Sequential(
            nn.Linear(4, region_feat_dim),
            nn.LayerNorm(region_feat_dim),   # 原 BatchNorm1d：B*T 维统计，LayerNorm 更稳
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 空间 Transformer 输入维度: 每个区 Mean+Max 后的维度是 out_dim * 2
        spatial_in_dim = out_dim * 2
        
        # 注意：nhead 必须能整除 transformer_input_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_in_dim, 
            nhead=nhead, 
            dropout=0.5,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # --- 3. 时序建模层（简化版 + 增强注意力池化）---
        # 【优化1】去除冗余的region_interaction层
        # 【优化2】仅保留1层Transformer用于时序建模（避免过拟合）
        self.fused_dim = region_feat_dim * 2  # 区域特征(64) + 全局支路(64)
        self.pos_encoder = PositionalEncoding(self.fused_dim)
        
        encoder_layer_temp = nn.TransformerEncoderLayer(
            d_model=self.fused_dim, nhead=4, dropout=0.3, batch_first=True  # 降低dropout防止欠拟合
        )
        # 【关键改动】从2层减少到1层，降低过拟合风险，加快收敛
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer_temp, num_layers=1)

        # 轻量 temporal 备选：mean+max 投影 / shallow TCN
        self.meanmax_proj = nn.Sequential(
            nn.Linear(self.fused_dim * 2, self.fused_dim),
            nn.LayerNorm(self.fused_dim),
            nn.ReLU()
        )
        tcn_pad = max(int(tcn_kernel_size) // 2, 0)
        self.temporal_tcn = nn.Sequential(
            nn.Conv1d(self.fused_dim, self.fused_dim, kernel_size=tcn_kernel_size, padding=tcn_pad),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(self.fused_dim, self.fused_dim, kernel_size=tcn_kernel_size, padding=tcn_pad)
        )

        # 【核心创新】使用MultiheadAttention进行自适应帧选择（Attention Pooling）
        # 这个机制可以自动识别视频中的关键帧（如微表情出现的时刻）
        self.attention_pooling = MultiheadAttention(
            embed_dim=self.fused_dim,  # 输入特征维度
            num_heads=4,               # 4个注意力头，平衡性能和复杂度
            dropout=0.2,
            batch_first=True
        )
        # 用于计算最终注意力分数的投影层
        self.attn_projection = nn.Linear(self.fused_dim, 1)
        # 【简化】分类器从temporal_dim改为fused_dim（因为去掉了RNN）
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, out_dim), 
            nn.LayerNorm(out_dim),           # 原 BatchNorm1d：B=1 时统计失效，LayerNorm 无此问题
            nn.ReLU(),
            nn.Dropout(0.4),  # 降低dropout
            nn.Linear(out_dim, num_classes)
        )
        # Precompute canonical edge_index patterns for each region (list of edges)
        # self.region_indices 已在 __init__ 开头按 region_scheme 确定，此处直接使用
        # 因为拼接了 Mean 和 Max 池化，所以维度是 out_dim * 2
        self.region_norms = nn.ModuleDict({
            k: nn.LayerNorm(out_dim * 2) for k in self.region_names
        })
        
        # [已简化] 与 intra_region_attention 配套，一并注释掉
        # self.intra_attn_norm = nn.LayerNorm(region_feat_dim)
        # (新增) 跨区域注意力的LayerNorm
        self.cross_attn_norm = nn.LayerNorm(region_feat_dim)
        # (新增) 特征融合层的LayerNorm
        self.fused_norm = nn.LayerNorm(self.fused_dim)
        # create local adjacency (sequence adjacency) for each region
        self.region_edge_index = self._init_region_edges()

        temp_edge_dict = self._init_region_edges()
        
        # 2.将 Tensor 注册为 buffer，这样 model.to(cuda) 时它们会自动移动
        for k, v in temp_edge_dict.items():
            # 命名为 edge_index_mouth, edge_index_nose 等
            self.register_buffer(f'edge_index_{k}', v)
        
        # 【关键删除】移除RNN层 - Transformer已足够处理时序依赖
        # 【关键删除】移除time_attention - 使用统一的attention_weights

    def _init_region_edges(self):
        """
        为每个区域构建拓扑连接。
          所有区域：线性链 + 二阶辺
          symptom7 专属拓扑：
            leye / reye  : 闭环
            mouth        : 外唇闭环、内唇闭环、上下唇纵向连接
            brow_glabella: 左右眉链 + 鼻梓桥接（眉中段 → 鼻梓top）
            nose_lower   : 额外跨阶连（鼻孔左-右）
        """
        _closed_regions = {'leye', 'reye', 'mouth'} if self.region_scheme == 'symptom7' else set()
        edge_dict = {}
        for k, idxs in self.region_indices.items():
            n = len(idxs)
            edges = []
            # ── 线性链 + 二阶邻居 ────────────────────────────────────────
            for i in range(n - 1):
                edges.append((i, i + 1));  edges.append((i + 1, i))
            for i in range(n - 2):
                edges.append((i, i + 2));  edges.append((i + 2, i))
            # ── symptom7 专属边 ─────────────────────────────────────────────
            if k in _closed_regions and n >= 3:
                if k == 'mouth':
                    # mouth 20节点: local 0-11=外唇(48-59), 12-19=内唇(60-67)
                    edges.append((11, 0));   edges.append((0, 11))   # 外唇闭环
                    if n > 19:
                        edges.append((19, 12));  edges.append((12, 19))  # 内唇闭环
                    # 上下唇纵向：外唇角 ↔ 内唇对应点
                    for (out_i, in_i) in [(0, 12), (3, 14), (6, 16), (9, 18)]:
                        if out_i < n and in_i < n:
                            edges.append((out_i, in_i));  edges.append((in_i, out_i))
                else:  # leye / reye简单闭环
                    edges.append((n - 1, 0));  edges.append((0, n - 1))
            if k == 'brow_glabella' and self.region_scheme == 'symptom7':
                # brow_glabella 14节点:
                #   local 0-4 = 左眉(17-21), 5-9 = 右眉(22-26), 10-13 = 鼻梓(27-30)
                # 桥接：眉中段 → 鼻梓顶点(local 10)
                for src in [2, 4, 5, 7]:  # face 19, 21, 22, 24 → 鼻梓27
                    if src < n and 10 < n:
                        edges.append((src, 10));  edges.append((10, src))
            if k == 'nose_lower' and self.region_scheme == 'symptom7':
                # nose_lower 5节点(31-35)：额外跨阶串联鼻孔左右
                if n >= 5:
                    edges.append((0, 4));  edges.append((4, 0))
            edge_dict[k] = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_dict
    
    def _batch_edge_index(self, base_edge_index, n_nodes, batch_frames):
        """
        Given base_edge_index (2,E) for a graph of n_nodes,
        replicate across batch_frames graphs by offsetting node indices.
        Returns big_edge_index on `device`.
        """
        # 这一行必须保留，它会自动从输入的 tensor 获取设备信息
        device = base_edge_index.device 
        
        E = base_edge_index.shape[1]
        row0 = base_edge_index[0].unsqueeze(0).repeat(batch_frames, 1)
        row1 = base_edge_index[1].unsqueeze(0).repeat(batch_frames, 1)
        
        offsets = (torch.arange(batch_frames, device=device, dtype=torch.long) * n_nodes).unsqueeze(1)
        
        big0 = (row0 + offsets).reshape(-1)
        big1 = (row1 + offsets).reshape(-1)
        big_edge = torch.stack([big0, big1], dim=0)
        return big_edge

    def _build_global_dynamic_features(self, region_dict):
        """
        构造每帧全局动态特征 (B, T, 4):
          0) centroid velocity
          1) centroid acceleration
          2) global motion energy
          3) mouth-eye coordination
        """
        # 统一拼接为全脸点集 (B, T, N_total, C)
        all_pts = torch.cat(
            [region_dict[k] for k in self.region_names],
            dim=2
        )
        xy = all_pts[..., :2]
        B, T, _, _ = xy.shape

        # (a) 质心速度
        centroid = xy.mean(dim=2)  # (B, T, 2)
        d_centroid = centroid[:, 1:] - centroid[:, :-1]  # (B, T-1, 2)
        centroid_vel = torch.norm(d_centroid, dim=-1)  # (B, T-1)
        z = torch.zeros(B, 1, device=xy.device, dtype=xy.dtype)
        centroid_vel = torch.cat([z, centroid_vel], dim=1)  # (B, T)

        # (b) 质心加速度（速度一阶差分绝对值）
        d_vel = centroid_vel[:, 1:] - centroid_vel[:, :-1]  # (B, T-1)
        centroid_acc = torch.cat([z, d_vel.abs()], dim=1)  # (B, T)

        # (c) 全脸动态能量
        d_all = xy[:, 1:] - xy[:, :-1]  # (B, T-1, 69, 2)
        global_energy = torch.norm(d_all, dim=-1).mean(dim=-1)  # (B, T-1)
        global_energy = torch.cat([z, global_energy], dim=1)  # (B, T)

        # (d) 嘴-眼协同
        mouth_xy = region_dict['mouth'][..., :2]
        leye_xy = region_dict['leye'][..., :2]
        reye_xy = region_dict['reye'][..., :2]
        eye_xy = torch.cat([leye_xy, reye_xy], dim=2)  # (B, T, N_eye, 2)

        d_mouth = mouth_xy[:, 1:] - mouth_xy[:, :-1]
        d_eye = eye_xy[:, 1:] - eye_xy[:, :-1]
        mouth_energy = torch.norm(d_mouth, dim=-1).mean(dim=-1)  # (B, T-1)
        eye_energy = torch.norm(d_eye, dim=-1).mean(dim=-1)  # (B, T-1)
        mouth_eye_coord = torch.cat([z, mouth_energy * eye_energy], dim=1)  # (B, T)

        return torch.stack([centroid_vel, centroid_acc, global_energy, mouth_eye_coord], dim=-1)

    def _region_forward(self, x_region, gcn1, gcn2, edge_base, region_name):
        """
        x_region: (B, T, N, C)
        流程: GCN × 2 → mean/max pooling → LayerNorm
        [简化] 原区域内自注意力对已 pool 的单 token 做 QKV 无实质建模意义，已移除。
        """
        B, T, N, C = x_region.shape
        BT = B * T
        x_flat = x_region.reshape(BT * N, C)

        base = edge_base.to(x_region.device)
        big_edge = self._batch_edge_index(base, N, BT)

        # 1. GCN 特征提取（两层）
        h = F.relu(gcn1(x_flat, big_edge))
        h = F.relu(gcn2(h, big_edge))
        h = h.view(BT, N, -1)  # (B*T, N, out_dim)

        # 2. Mean+Max 池化 → LayerNorm（去掉了区域内自注意力）
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1)[0]], dim=-1)  # (B*T, region_feat_dim)
        pooled = self.region_norms[region_name](pooled)

        return pooled.view(B, T, -1)

    def _forward_backbone_features(self, region_dict, actual_lens):
        """
        GCN 骨干特征提取：从区域 GCN 到跨区域注意力 → 全局支路 → 位置编码 → 时序 Transformer → Attention Pooling。
        返回各阶段中间特征，供内部 forward 和外部融合模型共用。

        Returns:
            temporal_out : (B, T, fused_dim)  — temporal_transformer 输出
            video_repr   : (B, fused_dim)     — attention pooling 后的全局表示
            time_mask    : (B, T)             — 有效帧掩码
        """
        curr_device = next(self.parameters()).device
        region_dict = {k: v.to(curr_device) for k, v in region_dict.items()}
        actual_lens = actual_lens.to(curr_device)
        B, T = region_dict['mouth'].shape[0], region_dict['mouth'].shape[1]

        # 1. 各区域支路（动态按 self.region_names 遍历，支持 legacy6 / symptom7）
        num_regions = len(self.region_names)
        regions = [
            self._region_forward(
                region_dict[name], self.gcn1[name], self.gcn2[name],
                getattr(self, f'edge_index_{name}'), name
            )
            for name in self.region_names
        ]
        region_tokens = torch.stack(regions, dim=2)  # (B, T, num_regions, D)
        # Phase A: 固定先验 + 可学习标量（正权重）
        region_prior = F.softplus(self.region_logits).unsqueeze(1).to(region_tokens.dtype)  # (1,1,6,1)
        # Phase B: sample-wise gate（按样本自适应区域可靠性）
        if self.use_sample_region_gate:
            # 用每个区域在时间维的均值表征估计样本级区域 gate
            region_desc = region_tokens.mean(dim=1)  # (B, 6, D)
            sample_gate = self.region_sample_gate(region_desc).unsqueeze(1)  # (B, 1, 6, 1)
            final_region_weight = region_prior * sample_gate
        else:
            final_region_weight = region_prior
        all_regions = region_tokens * final_region_weight
        regions = [all_regions[:, :, i, :] for i in range(all_regions.size(2))]

        # 2. 跨区域融合（可切换：cross_attn | concat_linear | mlp）
        if self.region_fusion_mode == 'cross_attn':
            cross_attention_results = []
            for i, query_region in enumerate(regions):
                kv = all_regions.clone()
                kv[:, :, i, :] = 0
                q       = query_region.view(B * T, 1, -1)
                kv_flat = kv.view(B * T, all_regions.size(2), -1)
                attn_out, _ = self.cross_attention(q, kv_flat, kv_flat)
                attn_out = attn_out.view(B, T, -1)
                enhanced_region = self.cross_attn_norm(query_region + attn_out)
                cross_attention_results.append(enhanced_region)
            cross_attention_output = torch.stack(cross_attention_results, dim=2).mean(dim=2)  # (B, T, D)
        elif self.region_fusion_mode == 'concat_linear':
            region_concat = all_regions.reshape(B, T, -1)
            cross_attention_output = self.region_concat_linear(region_concat)
        else:  # 'mlp'
            region_concat = all_regions.reshape(B, T, -1)
            cross_attention_output = self.region_concat_mlp_norm(self.region_concat_mlp(region_concat))

        # 3. 全局动态支路（可通过 use_global_branch 开关控制）
        global_dyn = self._build_global_dynamic_features(region_dict)  # (B, T, 4)
        if self.use_global_branch:
            # tanh(global_alpha) 将权重约束在 (-1, 1)；正值为加权，负值为反向校正
            # 初始 alpha=0.2 → tanh≈0.197（保守），alpha=0.5 → tanh≈0.462（适中）
            global_feat = self.global_branch(global_dyn.view(B * T, 4)).view(B, T, -1)
            global_feat = torch.tanh(self.global_alpha) * global_feat
        else:
            # 关闭全局支路：用零向量占位，保持 fused_dim 不变
            global_feat = torch.zeros_like(cross_attention_output)

        # 4. 特征融合
        per_frame = torch.cat([cross_attention_output, global_feat], dim=-1)
        per_frame = self.fused_norm(per_frame)

        # 5. 时序 Transformer
        # ── 基础时间 mask（actual_lens）──────────────────────────────────────
        time_mask = (
            torch.arange(T, device=curr_device).expand(B, T)
            < actual_lens.unsqueeze(1)
        )  # (B, T) bool

        # ── 低人脸有效度帧 mask：valid_ratio < FACE_VALID_THRESH 的帧当无效 ──
        # valid_ratio 在预处理时被均匀写入每个 region 的最后一维（validity 通道）。
        #   LMVD：validity = 1.0（全部有效）→ face_qual_mask 全 True，行为不变。
        #   DVLOG：validity = 真实人脸追踪质量 → 低质量帧被过滤掉。
        # 安全守卫：若某样本在 actual_len 内所有帧都低于阈值（极端情况），
        #   则回退到纯 actual_len mask，防止 time_mask 全 False 导致 softmax 崩溃。
        _any_region = region_dict.get('mouth', next(iter(region_dict.values())))  # (B,T,N,C)
        if _any_region.shape[-1] >= 11:
            # validity 通道是最后一维，且在预处理时已均匀复制到同帧所有关键点
            face_valid_ratio = _any_region[:, :, 0, -1]           # (B, T) float [0,1]
            face_qual_mask   = (face_valid_ratio >= self.face_valid_thresh)  # (B, T) bool
            compound_mask    = time_mask & face_qual_mask

            # 安全守卫：全低质量样本回退
            all_invalid = ~compound_mask.any(dim=1)                # (B,)
            if all_invalid.any():
                compound_mask[all_invalid] = time_mask[all_invalid]
            time_mask = compound_mask

        if self.gcn_temporal_mode == 'transformer':
            per_frame_pos = self.pos_encoder(per_frame)
            temporal_out = self.temporal_transformer(per_frame_pos, src_key_padding_mask=(~time_mask))
            video_repr = self._attention_pool_temporal(temporal_out, time_mask)
        elif self.gcn_temporal_mode == 'meanmax':
            temporal_out = per_frame
            video_repr = self._masked_meanmax_project(temporal_out, time_mask)
        else:  # 'tcn'
            tcn_in = per_frame.transpose(1, 2)  # (B, D, T)
            tcn_out = self.temporal_tcn(tcn_in).transpose(1, 2)  # (B, T, D)
            temporal_out = per_frame + tcn_out
            video_repr = self._masked_meanmax_project(temporal_out, time_mask)

        return temporal_out, video_repr, time_mask

    def _attention_pool_temporal(self, temporal_out, time_mask):
        valid_mask_expanded = time_mask.unsqueeze(-1).float()
        global_context = (temporal_out * valid_mask_expanded).sum(dim=1) / valid_mask_expanded.sum(dim=1).clamp(min=1.0)
        global_context = global_context.unsqueeze(1)

        _, attn_weights = self.attention_pooling(
            global_context, temporal_out, temporal_out,
            key_padding_mask=(~time_mask)
        )
        frame_importance = attn_weights.squeeze(1)
        attn_logits = self.attn_projection(temporal_out).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~time_mask, -1e9)
        combined_scores = F.softmax(attn_logits + frame_importance * 10, dim=1).unsqueeze(-1)
        return torch.sum(temporal_out * combined_scores, dim=1)

    def _masked_meanmax_project(self, temporal_out, time_mask):
        valid_mask = time_mask.unsqueeze(-1).float()
        mean_feat = (temporal_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        max_feat = temporal_out.masked_fill(~time_mask.unsqueeze(-1), -1e9).amax(dim=1)
        pooled = torch.cat([mean_feat, max_feat], dim=-1)
        return self.meanmax_proj(pooled)

    def forward(self, region_dict, actual_lens, return_dict=False):
        """
        Args:
            region_dict : dict of {region: (B, T, N, C)}
            actual_lens : (B,)
            return_dict : 若为 True，返回包含中间特征的字典；否则只返回 logits（兼容原有调用方式）
        Returns (return_dict=False): logits (B, num_classes)
        Returns (return_dict=True):  dict with keys:
            'logits_gcn'   : (B, num_classes)
            'temporal_out' : (B, T, fused_dim)
            'gcn_repr'     : (B, fused_dim)
            'time_mask'    : (B, T)
        """
        temporal_out, video_repr, time_mask = self._forward_backbone_features(region_dict, actual_lens)
        logits = self.classifier(video_repr)

        if not return_dict:
            return logits

        return {
            'logits_gcn':   logits,
            'temporal_out': temporal_out,
            'gcn_repr':     video_repr,
            'time_mask':    time_mask,
        }