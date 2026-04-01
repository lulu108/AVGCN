import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv
from torch.nn import MultiheadAttention

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
                 dropout=0.2,
                 spatial_dropout=None,
                 temporal_dropout=None,
                 classifier_dropout=None):
        super().__init__()
        self.out_dim = out_dim
        self.device = device
        
        # --- dropout 配置：None 时 fallback 到通用 dropout ---
        self.dropout = dropout
        self.spatial_dropout = spatial_dropout if spatial_dropout is not None else dropout
        self.temporal_dropout = temporal_dropout if temporal_dropout is not None else dropout
        self.classifier_dropout = classifier_dropout if classifier_dropout is not None else dropout
        
        # 1. 区域特征维度现在是 out_dim * 2 (即 64)
        region_feat_dim = out_dim * 2
        
        # (新增) 定义区域内自注意力层 - 用于建模区域内部关键点的依赖关系
        self.intra_region_attention = MultiheadAttention(
            embed_dim=region_feat_dim, 
            num_heads=2,  # 减少头数以降低复杂度
            batch_first=True,
            dropout=self.dropout
        )
        
        # (修改) 定义跨区域全局注意力层 - 用于建模不同区域间的交互
        self.cross_attention = MultiheadAttention(
            embed_dim=region_feat_dim, 
            num_heads=nhead, 
            batch_first=True,
            dropout=self.dropout
        )
        # 定义 GCN 层
        self.gcn_mouth = GCNConv(10, out_dim)
        self.gcn_mouth_2 = GCNConv(out_dim, out_dim)

        self.gcn_nose = GCNConv(10, out_dim)
        self.gcn_nose_2 = GCNConv(out_dim, out_dim)

        self.gcn_leye = GCNConv(10, out_dim)
        self.gcn_leye_2 = GCNConv(out_dim, out_dim)

        self.gcn_reye = GCNConv(10, out_dim)
        self.gcn_reye_2 = GCNConv(out_dim, out_dim)

        self.gcn_ljaw = GCNConv(10, out_dim)
        self.gcn_ljaw_2 = GCNConv(out_dim, out_dim)

        self.gcn_rjaw = GCNConv(10, out_dim)
        self.gcn_rjaw_2 = GCNConv(out_dim, out_dim)

        self.region_weights = nn.Parameter(torch.ones(1, 6, 1))

        #增加全局坐标投影支路 (解决跨区域联动)
        # 69个点 * 10个特征
        self.global_branch = nn.Sequential(
            nn.Linear(69 * 10, region_feat_dim), 
            nn.BatchNorm1d(region_feat_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        # 空间 Transformer 输入维度: 每个区 Mean+Max 后的维度是 out_dim * 2
        spatial_in_dim = out_dim * 2
        
        # 注意：nhead 必须能整除 transformer_input_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_in_dim, 
            nhead=nhead, 
            dropout=self.spatial_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # --- 3. 时序建模层（简化版 + 增强注意力池化）---
        # 【优化1】去除冗余的region_interaction层
        # 【优化2】仅保留1层Transformer用于时序建模（避免过拟合）
        self.fused_dim = region_feat_dim * 2  # 区域特征(64) + 全局支路(64)
        self.pos_encoder = PositionalEncoding(self.fused_dim)
        
        encoder_layer_temp = nn.TransformerEncoderLayer(
            d_model=self.fused_dim, nhead=4, dropout=self.temporal_dropout, batch_first=True
        )
        # 【关键改动】从2层减少到1层，降低过拟合风险，加快收敛
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer_temp, num_layers=1)

        # 【核心创新】使用MultiheadAttention进行自适应帧选择（Attention Pooling）
        # 这个机制可以自动识别视频中的关键帧（如微表情出现的时刻）
        self.attention_pooling = MultiheadAttention(
            embed_dim=self.fused_dim,  # 输入特征维度
            num_heads=4,               # 4个注意力头，平衡性能和复杂度
            dropout=self.dropout,
            batch_first=True
        )
        # 用于计算最终注意力分数的投影层
        self.attn_projection = nn.Linear(self.fused_dim, 1)
        # 【简化】分类器从temporal_dim改为fused_dim（因为去掉了RNN）
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, out_dim), 
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(self.classifier_dropout),
            nn.Linear(out_dim, num_classes)
        )
        # Precompute canonical edge_index patterns for each region (list of edges)
        self.region_indices = {
            'mouth': list(range(48, 68)),
            'nose': list(range(27, 36)),
            'leye': list(range(17, 22)) + list(range(36, 42)),
            'reye': list(range(22, 27))+list(range(42, 48)),
            'ljaw': list(range(0, 9)),
            'rjaw': list(range(8, 17))
        }
        # 因为拼接了 Mean 和 Max 池化，所以维度是 out_dim * 2
        self.region_norms = nn.ModuleDict({
            k: nn.LayerNorm(out_dim * 2) for k in self.region_indices.keys()
        })
        
        # (新增) 区域内自注意力的LayerNorm
        self.intra_attn_norm = nn.LayerNorm(region_feat_dim)
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
        # 逻辑：为每个区域构建拓扑连接
        edge_dict = {}
        for k, idxs in self.region_indices.items():
            n = len(idxs)
            edges = []
            for i in range(n - 1):
                edges.append((i, i + 1))
                edges.append((i + 1, i))
            # 建立二阶连接增加感受野
            for i in range(n - 2):
                edges.append((i, i + 2))
                edges.append((i + 2, i))
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

    def _region_forward(self, x_region, gcn1, gcn2, edge_base, region_name):
        """
        x_region: (B, T, N, C)
        改进流程: GCN → 区域内Self-Attention → Pooling
        """
        B, T, N, C = x_region.shape
        BT = B * T
        x_flat = x_region.reshape(BT * N, C) 

        base = edge_base.to(x_region.device)
        big_edge = self._batch_edge_index(base, N, BT)

        # 1. GCN特征提取
        h = F.relu(gcn1(x_flat, big_edge))
        h = F.relu(gcn2(h, big_edge))  
        h = h.view(BT, N, -1)  # (B*T, N, out_dim)
        
        # 2. (新增) 区域内自注意力 - 建模区域内关键点的相互依赖
        # 先pooling得到初步特征
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1)[0]], dim=-1)  # (B*T, region_feat_dim)
        pooled = self.region_norms[region_name](pooled)
        
        # 将pooled特征reshape用于自注意力 (B*T, 1, D)
        pooled_reshaped = pooled.unsqueeze(1)  # (B*T, 1, region_feat_dim)
        
        # Self-Attention: Query/Key/Value都来自同一区域
        # 这里使用pooled特征作为query,用GCN节点特征扩展维度后作为key/value
        # 为简化,我们对pooled特征本身做self-attention (可视为时序上的自注意力)
        attn_out, _ = self.intra_region_attention(
            pooled_reshaped, pooled_reshaped, pooled_reshaped
        )  # (B*T, 1, region_feat_dim)
        
        # 残差连接 + LayerNorm
        pooled = self.intra_attn_norm(pooled + attn_out.squeeze(1))
        
        return pooled.view(B, T, -1)

    def forward(self, region_dict, actual_lens):
        # 动态获取当前模型层所在的设备（取第一个参数的设备即可）
        curr_device = next(self.parameters()).device 
        
        # 确保输入数据与模型设备一致
        region_dict = {k: v.to(curr_device) for k, v in region_dict.items()} 
        
        # 这一行也可以根据需要保留或删除，确保 actual_lens 也在同一设备
        actual_lens = actual_lens.to(curr_device)
        B, T = region_dict['mouth'].shape[0], region_dict['mouth'].shape[1]
        
        # 1. 各区域支路
        # 使用 getattr 动态获取已经位于 GPU 的 buffer
        ljaw = self._region_forward(region_dict['ljaw'], self.gcn_ljaw, self.gcn_ljaw_2, 
                                   getattr(self, 'edge_index_ljaw'), 'ljaw')
        rjaw = self._region_forward(region_dict['rjaw'], self.gcn_rjaw, self.gcn_rjaw_2, 
                                   getattr(self, 'edge_index_rjaw'), 'rjaw')
        leye = self._region_forward(region_dict['leye'], self.gcn_leye, self.gcn_leye_2, 
                                   getattr(self, 'edge_index_leye'), 'leye')
        reye = self._region_forward(region_dict['reye'], self.gcn_reye, self.gcn_reye_2, 
                                   getattr(self, 'edge_index_reye'), 'reye')
        nose = self._region_forward(region_dict['nose'], self.gcn_nose, self.gcn_nose_2, 
                                   getattr(self, 'edge_index_nose'), 'nose')
        mouth = self._region_forward(region_dict['mouth'], self.gcn_mouth, self.gcn_mouth_2, 
                                    getattr(self, 'edge_index_mouth'), 'mouth')
        
        regions = [ljaw, rjaw, leye, reye, nose, mouth]  # 6 个区域
        all_regions = torch.stack(regions, dim=2)  # 初始化 Key/Value 全区域堆叠 (B, T, 6, D)

        # === 步骤2: 跨区域全局注意力 (Cross-Attention) ===
        cross_attention_results = []
        for i, query_region in enumerate(regions):
            # A. 构造当前 Query 的背景 KV (剔除自身)
            kv = all_regions.clone()
            kv[:, :, i, :] = 0  # 将当前区域置零
            
            # B. 展平时间维度: (B, T) -> (B*T)
            q = query_region.view(B * T, 1, -1)        # (B*T, 1, D)
            kv_flat = kv.view(B * T, 6, -1)            # (B*T, 6, D)
            
            # C. 跨区域注意力 - 让当前区域关注其他区域的上下文
            attn_out, _ = self.cross_attention(q, kv_flat, kv_flat)  # (B*T, 1, D)
            attn_out = attn_out.view(B, T, -1)  # 还原为 (B, T, D)
            
            # D. (新增) 残差连接 + LayerNorm
            enhanced_region = self.cross_attn_norm(query_region + attn_out)
            cross_attention_results.append(enhanced_region)

        # 最终融合所有区域的增强特征 (平均池化)
        cross_attention_output = torch.stack(cross_attention_results, dim=2).mean(dim=2)  # (B, T, D)
        
            # 3. 全局支路
        all_pts = torch.cat([region_dict[k] for k in ['ljaw', 'rjaw', 'leye', 'reye', 'nose', 'mouth']], dim=2)
        global_feat = self.global_branch(all_pts.view(B*T, -1)).view(B, T, -1)

        # 4. 特征融合 + 位置编码
        per_frame = torch.cat([cross_attention_output, global_feat], dim=-1)
        per_frame = self.fused_norm(per_frame)
        per_frame = self.pos_encoder(per_frame)
        
        # 5. 时序建模 - 仅使用Transformer（去掉RNN）
        time_mask = (torch.arange(T, device=curr_device).expand(B, T) < actual_lens.unsqueeze(1))
        temporal_out = self.temporal_transformer(
            per_frame, 
            src_key_padding_mask=(~time_mask)  # 掩码无效帧
        )
        
        # 6. 【核心创新】自适应Attention Pooling - 自动识别关键帧
        # 原理：通过MultiheadAttention计算每帧与全局的相关性，突出微表情等关键时刻
        
        # 6.1 计算全局上下文向量（所有有效帧的平均）
        valid_mask_expanded = time_mask.unsqueeze(-1).float()  # (B, T, 1)
        global_context = (temporal_out * valid_mask_expanded).sum(dim=1) / valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, D)
        global_context = global_context.unsqueeze(1)  # (B, 1, D) 作为Query
        
        # 6.2 使用MultiheadAttention计算每帧的重要性
        # Query: 全局上下文, Key/Value: 所有帧的特征
        attn_out, attn_weights = self.attention_pooling(
            global_context,           # Query: (B, 1, D) - "我想找什么样的帧?"
            temporal_out,             # Key:   (B, T, D) - "每帧的特征是什么?"
            temporal_out,             # Value: (B, T, D) - "具体的帧内容"
            key_padding_mask=(~time_mask)  # 忽略padding帧
        )  # 输出: (B, 1, D), 注意力权重: (B, 1, T)
        
        # 6.3 从注意力权重中提取每帧的重要性分数
        frame_importance = attn_weights.squeeze(1)  # (B, T) - 每帧的重要性分数
        
        # 6.4 通过投影层进一步细化分数（可选增强）
        attn_logits = self.attn_projection(temporal_out).squeeze(-1)  # (B, T)
        attn_logits = attn_logits.masked_fill(~time_mask, -1e9)
        
        # 6.5 结合两种注意力机制（MultiheadAttention权重 + 投影层权重）
        combined_scores = F.softmax(attn_logits + frame_importance * 10, dim=1).unsqueeze(-1)  # (B, T, 1)
        
        # 7. 加权池化得到视频级表示
        video_repr = torch.sum(temporal_out * combined_scores, dim=1)  # (B, fused_dim)
        
        # 8. 分类输出
        return self.classifier(video_repr)
        