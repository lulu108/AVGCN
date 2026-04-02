## 重构状态（阶段 2 已完成）

### 1) 训练入口分层

通用训练实现位于 [model/vitGCN/vit_gcn_config_train.py](model/vitGCN/vit_gcn_config_train.py)，运行入口已收敛为：

- [model/vitGCN/DVLOG_AVGCN/train_dvlog.py](model/vitGCN/DVLOG_AVGCN/train_dvlog.py)
- [model/vitGCN/LMVD_AVGCN/train_lmvd.py](model/vitGCN/LMVD_AVGCN/train_lmvd.py)

### 2) DataLoader 分层

新增数据集专用加载入口（第二步重构）：

- [model/vitGCN/data/dvlog_loader.py](model/vitGCN/data/dvlog_loader.py)
- [model/vitGCN/data/lmvd_loader.py](model/vitGCN/data/lmvd_loader.py)
- [model/vitGCN/data/face_region_dataset.py](model/vitGCN/data/face_region_dataset.py)

训练脚本已改为使用这些入口（而非直接 import 原始 loader 文件）。

### 3) AV Backbone 现状

- [model/vitGCN/av_backbone_common.py](model/vitGCN/av_backbone_common.py)（主实现）

数据集区分逻辑已在 [model/vitGCN/Vit_gcnmodel.py](model/vitGCN/Vit_gcnmodel.py) 中直接传参，不再依赖超薄封装文件。

### 4) Fusion 拆分（按数据集）

- [model/vitGCN/fusion_common.py](model/vitGCN/fusion_common.py)
  - 仅放通用融合积木：`LateFusionHead` / `ConcatFusionHead` / `IAResidualFusionHead` / `IntermediateCrossFusion` / `AFIFusionHead` 等
  - 仅放通用张量工具：`masked_mean` / `safe_time_pool_temporal` / `mask_safe_attention_pool`

- [model/vitGCN/vit_gcn_fusion_dvlog.py](model/vitGCN/vit_gcn_fusion_dvlog.py)
  - 仅放 DVLOG 组装默认值与入口（`DVLOGViTGCNFusion`）

- [model/vitGCN/vit_gcn_fusion_lmvd.py](model/vitGCN/vit_gcn_fusion_lmvd.py)
  - 仅放 LMVD 组装默认值与入口（`LMVDViTGCNFusion`）

- [model/vitGCN/Vit_gcnmodel.py](model/vitGCN/Vit_gcnmodel.py)
  - `ViT_GCN_Fusion` 现在仅作为路由壳，按 `dataset` 转发到 DVLOG/LMVD 专用实现

## 运行命令

请在仓库根目录 `D:/Paper/code/LMVD_new` 下执行。

说明：本次 fusion 拆分后，运行命令**不变**。

### LMVD

```bash
python model/vitGCN/LMVD_AVGCN/train_lmvd.py
```

### DVLOG（统一入口 + 开关）

`--exp` 可选：`fusion` / `av_only_new` / `av_only_legacy` / `gcn_only`

1. Fusion（AV+GCN）

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp fusion
```

2. AV-only（new）

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp av_only_new | tee dvlogAVonlyNew.log
```

3. AV-only（legacy）

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp av_only_legacy | tee dvlogAVonlyLegacy.log
```

4. GCN-only

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp gcn_only | tee dvlogGCNonluNew.log
```

## Delta 汇总

DVLOG 汇总逻辑在 [model/vitGCN/vit_gcn_config_train.py](model/vitGCN/vit_gcn_config_train.py)：

- AV+GCN - AV-only
- AV-only - GCN-only
- New AV-only - Old AV-only

结果会写入 `logs/dvlog_mode_metrics_registry.json`。
