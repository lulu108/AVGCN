import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from torch_geometric.nn import GCNConv
except Exception as e:
    raise ImportError("torch_geometric is required for this file. Install torch_geometric.") from e

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
                 out_dim=64,
                 nhead=4,
                 transformer_layers=2,
                 num_classes=2,
                 device='cpu'):
        super().__init__()
        self.out_dim = out_dim
        self.device = device

        # GCN per-region: simple 2-layer MPNN (GCNConv)
        # 输入 feat_dim = 2 (x,y)
        self.gcn_mouth = GCNConv(2, out_dim)
        self.gcn_mouth_2 = GCNConv(out_dim, out_dim)

        self.gcn_nose = GCNConv(2, out_dim)
        self.gcn_nose_2 = GCNConv(out_dim, out_dim)

        self.gcn_leye = GCNConv(2, out_dim)
        self.gcn_leye_2 = GCNConv(out_dim, out_dim)

        self.gcn_reye = GCNConv(2, out_dim)
        self.gcn_reye_2 = GCNConv(out_dim, out_dim)

        self.gcn_ljaw = GCNConv(2, out_dim)
        self.gcn_ljaw_2 = GCNConv(out_dim, out_dim)

        self.gcn_rjaw = GCNConv(2, out_dim)
        self.gcn_rjaw_2 = GCNConv(out_dim, out_dim)

        # Transformer Encoder (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim // 2, num_classes)
        )

        # Precompute canonical edge_index patterns for each region (list of edges)
        self.region_indices = {
            'mouth': list(range(48, 68)),
            'nose': list(range(27, 36)),
            'leye': list(range(17, 22)) + list(range(36, 42)),
            'reye': list(range(22, 27))+list(range(42, 48)),
            'ljaw': list(range(0, 9)),
            'rjaw': list(range(9, 17))
        }
        # create local adjacency (sequence adjacency) for each region
        # create local adjacency for each region
        self.region_edge_index = {}
        for k, idxs in self.region_indices.items():
            n = len(idxs) # 获取该区域的节点数量，例如 mouth=20
            
            #使用相对索引 0 到 n-1，而不是全局索引 i
            edges = []
            # 建立邻居连接 (Chain structure: 0-1, 1-2...)
            for i in range(n - 1):
                edges.append((i, i + 1))
                edges.append((i + 1, i))
            # 建立二阶连接 (Capture structure: 0-2, 1-3...)
            for i in range(n - 2):
                edges.append((i, i + 2))
                edges.append((i + 2, i))
            
            if len(edges) == 0:
                edges = [(0, 0)]
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.region_edge_index[k] = edge_index

    def _batch_edge_index(self, base_edge_index, n_nodes, batch_frames): # <--- 删除了 device
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

    def _region_forward(self, x_region, gcn1, gcn2, edge_base):
        """
        x_region: (B, T, N, 2) tensor on device
        returns: pooled (B, T, out_dim)
        """
        B, T, N, C = x_region.shape
        BT = B * T
        # flatten frames: (BT, N, 2) -> (BT*N, 2)
        x_flat = x_region.reshape(BT * N, C)  # node features for big graph

        # build big edge_index for BT graphs
        base = edge_base.to(x_region.device)
        big_edge = self._batch_edge_index(base, N, BT)  # (2, E_total)

        # run GCNConv on big graph
        h = F.relu(gcn1(x_flat, big_edge))
        h = F.relu(gcn2(h, big_edge))  # (BT*N, out_dim)

        # reshape back -> (BT, N, out_dim)
        h = h.view(BT, N, -1)
        # pool nodes per frame
        pooled = h.mean(dim=1)  # (BT, out_dim)
        # reshape to (B, T, out_dim)
        pooled = pooled.view(B, T, -1)
        return pooled

    def forward(self, region_dict):
        """
        region_dict values expected to be tensors (B, T, N, 2) on same device as model
        """
        device = next(self.parameters()).device
        # Ensure tensors on same device
        for k in region_dict:
            if isinstance(region_dict[k], torch.Tensor):
                region_dict[k] = region_dict[k].to(device)
            else:
                region_dict[k] = torch.tensor(region_dict[k], dtype=torch.float32, device=device)

        # region-wise forward: each returns (B, T, out_dim)
        mouth_feat = self._region_forward(region_dict['mouth'], self.gcn_mouth, self.gcn_mouth_2, self.region_edge_index['mouth'])
        nose_feat  = self._region_forward(region_dict['nose'],  self.gcn_nose,  self.gcn_nose_2,  self.region_edge_index['nose'])
        leye_feat  = self._region_forward(region_dict['leye'],  self.gcn_leye,  self.gcn_leye_2,  self.region_edge_index['leye'])
        reye_feat  = self._region_forward(region_dict['reye'],  self.gcn_reye,  self.gcn_reye_2,  self.region_edge_index['reye'])
        ljaw_feat  = self._region_forward(region_dict['ljaw'],  self.gcn_ljaw,  self.gcn_ljaw_2,  self.region_edge_index['ljaw'])
        rjaw_feat  = self._region_forward(region_dict['rjaw'],  self.gcn_rjaw,  self.gcn_rjaw_2,  self.region_edge_index['rjaw'])

        # stack tokens: for each frame, 6 tokens -> first create (B*T, 6, D)
        B, T, D = mouth_feat.shape
        BT = B * T
        # reshape each (B, T, D) -> (BT, D)
        def to_BT(x): return x.reshape(BT, D)
        tokens = torch.stack([to_BT(ljaw_feat), to_BT(rjaw_feat), to_BT(leye_feat),
                              to_BT(reye_feat), to_BT(nose_feat), to_BT(mouth_feat)], dim=1)  # (BT, 6, D)

        tokens = tokens.permute(1, 0, 2)
        # Transformer expects (BT, seq_len=6, D) with batch_first=True
        fused = self.transformer(tokens)  # (BT, 6, D)
        fused = fused.permute(1, 0, 2)   # 转回来 (Seq, Batch, Dim) -> (Batch, Seq, Dim)
        # pool over token dimension -> per-frame vector
        per_frame = fused.mean(dim=1)  # (BT, D)
        # reshape to (B, T, D)
        per_frame = per_frame.view(B, T, D)

        # temporal aggregation: simple mean pooling over time
        video_repr = per_frame.mean(dim=1)  # (B, D)

        logits = self.classifier(video_repr)  # (B, num_classes)
        return logits