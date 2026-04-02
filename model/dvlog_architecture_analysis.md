# DVLOG 当前架构解析（AV + GCN）与 Large-Scale-Multimodal-Depression-Detection-main 对比

本文基于仓库 `model/vitGCN` 相关实现与 `vit_gcn_config_train.py` 的 DVLOG 默认配置，整理当前 DVLOG 训练/推理路径中的真实结构。代码引用以文件路径标注，便于回查。

## 1. 整体数据流（DVLOG）

1. **输入三路信息**：视频特征、音频特征、面部关键点（分区后的 region 字典）。对应数据加载与预处理在 `model/vitGCN/kfoldLoader_multimodal.py` 与 `model/vitGCN/data/*_loader.py` 路径。  
2. **AV 主干编码**：视频与音频分别做特征序列编码（或走 legacy AV backbone），再做跨模态交互得到 AV 表示。实现于 `model/vitGCN/av_backbone_common.py` 与 `model/vitGCN/Vit_gcnmodel.py`。  
3. **GCN 分支**：对 face landmark 的每个 region 建图，GCN 编码后进行 region 交互、global dynamics、时间建模，得到 GCN 表示。实现于 `model/vitGCN/_02GCN_Transformer.py`。  
4. **AV+GCN 融合**：按配置选择 fusion 模式（默认 DVLOG 走 `concat`），输出最终分类。实现于 `model/vitGCN/vit_gcn_fusion_lmvd.py` 与 `model/vitGCN/vit_gcn_fusion_dvlog.py`。  

## 2. DVLOG 的默认配置（当前仓库）

DVLOG 训练使用 `vit_gcn_config_train.py` 的默认设定：

- **序列长度与维度**：`T=870`，视频维度 `D_VIDEO=136`，音频维度 `D_AUDIO=25`。见 `model/vitGCN/vit_gcn_config_train.py`。  
- **Fusion 默认**：DVLOG 默认 `FUSION_MODE='concat'`。见 `model/vitGCN/vit_gcn_config_train.py`。  
- **Feature-Sequence Encoder**：DVLOG 默认开启 `USE_FEATURE_SEQUENCE_ENCODER=True`（并可用 `USE_LEGACY_AV_BACKBONE` 切换）。见 `model/vitGCN/vit_gcn_config_train.py` 与 `model/vitGCN/av_backbone_common.py`。  
- **Temporal Conv Stem**：DVLOG 默认开启，且默认作用在 audio（`TEMPORAL_CONV_STEM_FOR='audio'`）。见 `model/vitGCN/vit_gcn_config_train.py`。  
- **Temporal Attention Pool**：DVLOG 默认开启（`USE_TEMPORAL_ATTN_POOL=True`）。见 `model/vitGCN/vit_gcn_config_train.py` 与 `model/vitGCN/vit_gcn_fusion_lmvd.py`。  
- **Audio TCN Encoder**：DVLOG 默认关闭，LMVD 默认打开。见 `model/vitGCN/vit_gcn_config_train.py`。  

> 重要：`DVLOGViTGCNFusion` 会覆写一些默认参数（例如 `video_dim=136`、`audio_dim=25`、`fusion_mode='concat'`、`face_valid_thresh=0.1`、`use_av_cross_attn=False`）。见 `model/vitGCN/vit_gcn_fusion_dvlog.py`。若训练脚本显式传参，仍可覆盖这些默认值。

## 3. AV 主干结构（当前 DVLOG）

### 3.1 特征序列编码路线（默认启用）

当 `USE_FEATURE_SEQUENCE_ENCODER=True` 时，`AVBackboneCore` 使用 **“单模态先编码、再 AV 融合”** 的路线：

1. **输入投影与局部时序建模**  
   - `FeatureSequenceEncoder`：`Conv1d + GroupNorm + GELU` 的 stem，然后 `ResidualTemporalConvBlock` 进行局部时序卷积。  
   - 代码：`model/vitGCN/av_backbone_common.py`。  
2. **Patch / 下采样**  
   - `Conv1d(kernel=patch_size, stride=patch_size)` 将长序列变为 token 序列。  
3. **全局时序 Transformer**  
   - `TemporalTransformerStack` 多层 Transformer 处理 token。  
4. **Attention Pool**  
   - `AttnPool1D` 根据 mask 做注意力汇聚得到 `repr`。  
5. **AV 融合**  
   - `PostEncoderAVFusion`：可选 cross-attn + gated token fusion。  

### 3.2 Legacy AV 路线（可选）

若 `USE_LEGACY_AV_BACKBONE=True`，则走旧版路线：

- `proj_video` / `proj_audio` → `CrossAttentionFusion`（可选）→ patch embedding + CLS + pos embedding → Transformer 堆叠 → `CLS` 作为 AV 表示。  
- 代码：`model/vitGCN/av_backbone_common.py` 中 `_encode_feature_sequences_legacy`。  

### 3.3 DVLOG 里 AV 相关细节

- DVLOG 默认 `use_av_cross_attn` 在 `DVLOGViTGCNFusion` 里设为 False，但训练配置可强制开启。  
  代码：`model/vitGCN/vit_gcn_fusion_dvlog.py` 与 `model/vitGCN/vit_gcn_config_train.py`。  
- Temporal conv stem 只在 DVLOG 开启（默认作用于 audio）。  
  代码：`model/vitGCN/vit_gcn_config_train.py` 与 `model/vitGCN/av_backbone_common.py`。  

## 4. GCN 分支结构（AnatomicalGCN）

GCN 分支目标是对 **面部关键点的区域图结构** 进行时序建模，核心实现位于 `model/vitGCN/_02GCN_Transformer.py`。主要步骤如下：

1. **Region 划分**  
   - 支持 `legacy6` 与 `symptom7` 两种分区。  
   - 代码：`_LEGACY6_REGION_INDICES` / `_SYMPTOM7_REGION_INDICES`。  
2. **每个 region 的图卷积**  
   - `GCNConv(11 → out_dim → out_dim)`，对每个 region 单独建图与编码。  
   - 关键点特征维度包含 `(x,y)` 等统计，后续进行 `mean+max` pooling 得到 region token。  
3. **Region 权重与 gate**  
   - 先验 `region_logits` + sample gate（`region_sample_gate`）得到最终 region 权重。  
4. **Region 交互**  
   - `cross_attn` / `concat_linear` / `mlp` 三种融合方式。  
5. **Global dynamics 分支（可选）**  
   - 构造 `(centroid velocity, centroid acc, global motion energy, mouth-eye coord)`，再线性投影并用 `global_alpha` 门控。  
6. **时间建模**  
   - `transformer` / `meanmax` / `meanmaxdiff` / `tcn` 四种模式；默认 `transformer`。  
   - Attention pooling 得到 GCN 表示。  
7. **Face quality mask**  
   - 使用 `face_valid_thresh` 过滤低质量帧，并与 `actual_lens` 结合生成 `time_mask`。  

这条分支输出 `gcn_repr` 和 `temporal_out`，并在融合时与 AV 分支对齐。见 `model/vitGCN/_02GCN_Transformer.py` 与 `model/vitGCN/vit_gcn_fusion_lmvd.py`。

## 5. AV + GCN 融合（DVLOG 默认 concat）

融合逻辑统一在 `LMVDViTGCNFusion` 中实现，DVLOG 通过 `DVLOGViTGCNFusion` 继承并覆写参数。关键点：

1. **对齐 token**  
   - GCN 的 `temporal_out` 通过 `gcn_proj` 映射到 `dim`，再用 `safe_time_pool_temporal` 对齐到 AV token 长度。  
2. **融合模式**  
   - `late`：AV 表示与 GCN 表示做门控融合。  
   - `it_cross` / `it_bi_gate` / `afi`：基于 token 的跨注意力/门控融合。  
   - `concat`：同时保留 token 融合和 summary 融合，最终以可学习 `blend` 组合。  
3. **DVLOG 默认**  
   - DVLOG 默认是 `concat`（见配置和 `DVLOGViTGCNFusion`）。  
4. **输出**  
   - 返回 `logits`、`logits_gcn`、`av_tokens`、`gcn_repr` 等用于训练和分析。  

见 `model/vitGCN/vit_gcn_fusion_lmvd.py` 与 `model/vitGCN/fusion_common.py`。

## 6. 与 Large-Scale-Multimodal-Depression-Detection-main 的主要差异

### 6.1 架构与分支差异

- **本仓库 DVLOG**：显式包含 **Face landmark → GCN** 分支，强调面部结构图建模与 region 交互。  
  - 关键实现：`model/vitGCN/_02GCN_Transformer.py`。  
- **Large-Scale-Multimodal-Depression-Detection-main**：不使用 GCN 或 face landmark 图结构；主要是 **Audio Transformer + Visual Transformer** 的双分支结构。  
  - 关键实现：`Large-Scale-Multimodal-Depression-Detection-main/models/MultiModalDepDet.py`。  

### 6.2 AV Backbone 与预训练策略

- **本仓库 DVLOG**：  
  - AV Backbone 可选“legacy ViT + CrossAttn”或“FeatureSequenceEncoder + PostEncoderAVFusion”。  
  - 默认更轻量、可控，强调时序 token 的 mask-aware 处理。  
  - 关键实现：`model/vitGCN/av_backbone_common.py`。  
- **Large-Scale**：  
  - Audio 分支使用 AST/ViT（可选 AudioSet 预训练），Visual 分支基于 VisualMAE；预训练权重较重。  
  - 关键实现：`models/Generate_Audio_Model.py` 与 `models/Generate_Visual_Model.py`。  

### 6.3 融合方式差异

- **本仓库 DVLOG**：  
  - 融合发生在 **AV token 与 GCN token** 之间，支持 `late`、`it_cross`、`it_bi_gate`、`concat`、`afi` 等。  
  - `concat` 模式有可学习 blend。  
  - 关键实现：`model/vitGCN/fusion_common.py` 与 `model/vitGCN/vit_gcn_fusion_lmvd.py`。  
- **Large-Scale**：  
  - 主要在 AV 之间做融合：`lt`（late transformer）、`it`（intermediate transformer）、`ia`（attention-based）、`MT`（MutualTransformer）等。  
  - 关键实现：`models/MultiModalDepDet.py` 与 `models/mutualtransformer.py`。  

### 6.4 时序处理与 mask 逻辑

- **本仓库 DVLOG**：  
  - 对 token/帧使用 `actual_lens` 与 `face_valid_thresh` 生成 mask，避免无效帧影响。  
  - GCN 分支会结合 face 质量构造 `time_mask`。  
- **Large-Scale**：  
  - 主要是固定下采样后序列，常用 mean pooling，mask 逻辑较少体现。  

### 6.5 输入维度与数据组织

- **本仓库 DVLOG**：  
  - 明确分离 `video_features`、`audio_features`、`face_regions` 三个输入源。  
  - 默认 DVLOG 维度：`video_dim=136`、`audio_dim=25`、`T=870`。  
  - 关键实现：`model/vitGCN/vit_gcn_config_train.py` 与 `model/vitGCN/vit_gcn_fusion_dvlog.py`。  
- **Large-Scale**：  
  - `MultiModalDepDet` 在 `feature_extractor` 中将输入拼接的 `x` 切分为 video/audio，再进入各自 backbone。  
  - 关键实现：`models/MultiModalDepDet.py`。  

## 7. 小结（DVLOG 当前实现特征）

DVLOG 版本的核心特点是 **“AV 特征序列编码 + Face Landmark GCN + 多策略融合”**。  
与 Large-Scale 的 MMFformer 风格相比，本仓库更强调 **面部结构图建模、可解释的 region 权重** 和 **mask-aware 的时序稳健性**，而 Large-Scale 更偏向 **重预训练 backbone + AV 融合** 的路线。

如果你希望我进一步补充：  
1. 结合当前训练脚本实际运行参数（日志）给出“本次实验”的精确结构  
2. 对 DVLOG 的某一种融合模式（如 `concat` 或 `afi`）写更细的张量流示意图  
3. 生成可视化架构图（流程图或 Mermaid）  
我可以继续完善。
