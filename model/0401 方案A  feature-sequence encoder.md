## 一、方案 A 的目标到底是什么

方案 A 的目标不是“换个 fusion 头”，而是：

> **把当前 AV 主干从轻量 feature projector，升级成真正的 feature-sequence encoder。**

也就是让：

- 视频 136 维序列先变成更强的视频 token / video repr
- 音频 25 维序列先变成更强的音频 token / audio repr
- 然后再做 AV 交互，再接 GCN 或分类头

这和你当前的 `ViT` 路线不同。你现在的 `ViT.forward()` 是：

1. 视频 `Conv1d + GroupNorm + ReLU`
2. 音频可选浅层 `ShallowAudioTCNEncoder`，再 `Conv1d + GroupNorm + ReLU`
3. 可选 `CrossAttentionFusion`
4. 把“融合后视频”和“原始音频投影”拼接
5. 再 `PatchEmbedding + CLS + 位置编码 + Transformer + classifier`。

这个结构的问题不是不能用，而是：

- 单模态先没学稳
- 交互比较早
- patch embedding 前的 token 语义不够强
- 所以后面做 fusion，本质上还是在“处理弱 token”

---

## 二、方案 A 的具体改法：视频 3 个 stage

### Stage 1：Video Stem，先把 136 维序列变成稳定的局部时序表示

这一层替换你当前的 `self.proj_video`。现在它只有一层 `Conv1d -> GroupNorm -> ReLU`，太轻了。

建议改成：

- `Linear / 1x1 Conv1d` 先投影到 `D_v`，比如 192
- 接 2 到 3 个 **Residual Local Temporal Block**
- 每个 block 用：
    - `depthwise temporal Conv1d`
    - `pointwise Conv1d`
    - `LayerNorm 或 GroupNorm`
    - `GELU`
    - 小 `Dropout`
    - 残差连接

### 这一步的原理

它的作用不是做全局建模，而是先把相邻时间步的局部动态学出来。  
你现在的视频 token 太“毛坯”，就是因为当前只有一层轻投影，短程动态没有被充分编码。这个 Stage 1 先解决“局部运动模式”问题。

### 它对应的含义

你可以把它理解成：

> “先把视频特征序列从原始 feature 序列，变成带局部动态语义的 token 序列。”

---

### Stage 2：Video Global Encoder，再做长程时间建模

这一层对应的是“更强的 temporal encoder / pretrained-like transformer encoder”。

建议你不要再像现在这样，直接把前面轻投影后的结果 patch 化后扔给共享 Transformer。更合理的是：

- Stage 1 输出的序列先做时间下采样或 patch 化
- 然后进入 **Video-only Transformer Stack**
- 用 4 到 6 层 pre-norm Transformer 编码长程依赖

### 这一步的原理

Stage 1 解决局部模式，Stage 2 解决：

- 表情变化的长程节律
- 视频内部的稀疏关键时刻
- 全局上下文依赖

你现在的问题之一就是全局 Transformer 吃到的 token 还不够强，所以它更像在“补表示”。这里改完后，全局 Transformer 吃的是已经做过局部动态建模的 token。

### 它对应的含义

你可以把它理解成：

> “让视频分支先形成真正独立的视频时序编码器，而不是 AV 拼接前的一层投影。”

---

### Stage 3：Video Summary Head，输出稳定的视频级表示

这里我建议保留你现在比较有价值的思路：

- `temporal attention pooling`
- 或者 `CLS + attention pooling residual`

也就是说，视频分支最终输出两样东西：

- `video_tokens`
- `video_repr`

### 这一步的原理

后面不管你做：

- `AV-only`
- `AV + GCN`
- `late / concat`

都需要一个稳定的视频级摘要。  
如果 Stage 3 稳定，后面的 fusion 才不是在“重新造 summary”。

---

## 三、方案 A 的具体改法：音频 3 个 stage

### Stage 1：Audio Stem，把 25 维音频序列变成更强的局部节律表示

这一层替换你当前的：

- `audio_tcn_encoder`
- `proj_audio`

因为现在音频分支更像一个辅助投影器，而不是独立强编码器。你那份总结里也明确提到，当前音频分支不是“独立 AST 风格编码器”，而更像共同组成 ViT token 的一部分。

建议改成：

- `Linear / 1x1 Conv1d` 投影到 `D_a=128 或 192`
- 2 到 3 个 **Dilated Temporal Conv Block / TCN Block**
- 每层：
    - dilation 逐步增大
    - `LayerNorm / GroupNorm`
    - `GELU`
    - 小 dropout
    - 残差

### 原理

音频现在的局部节律特征提取得太浅。  
这一步是先把：

- 音高/能量变化
- 语音节律
- 小范围情绪声学波动

先编码出来。

---

### Stage 2：Audio Global Encoder，再做长程时序建模

这一层对应：

- 2 到 4 层小 Transformer
- 或者 Conformer-lite

最后输出：

- `audio_tokens`
- `audio_repr`

### 原理

这一步让音频也先“自己说清楚”，而不是一开始就去配合视频。  
这是你前面总结里提到的核心原则：先让每个模态自己形成稳定表示，再交互。

---

### Stage 3：Audio Summary Head

音频也做 attention pooling，输出 `audio_repr`。

### 原理

这样后面你不管做：

- AV-only
- AV + GCN
- late / concat

音频都是真正独立的强摘要，而不是轻量拼接项。

---

## 四、方案 A 里哪些地方保留，哪些地方删掉

### 保留的部分

#### 1. 数据加载逻辑先保留

`MultiModalDataLoader` 当前已经负责：

- 读取 DVLOG 的 `*_visual.npy / *_acoustic.npy`
- 对齐视频和音频时间长度
- pad/crop
- 返回 face region 相关结构

这些先不用动。

LMVD 的 `MyDataLoader` 也可以保留，用于单模态或纯 AV-only 对照。

#### 2. GCN 分支整体先保留

`_02GCN_Transformer.py` 先不动。  
因为方案 A 的第一阶段，重点是验证：

- AV 主干强了以后，AV-only 会不会涨
- GCN 在强 AV 主干下还有没有真实增量

现在 GCN 分支本来就是独立三段式：

- 区域 GCN
- 区域融合
- 时间建模

你先别把变量再扩散。

#### 3. 训练主循环和评估协议先保留

`Vit_gcnfold.py` 和 `vit_gcn_config_train.py` 当前已经支持：

- `MODEL_MODE = 'fusion' | 'av_only' | 'gcn_only'`
- 同一套训练/评估入口切换模式

这对你做信息增量关系对照非常有用。

---

### 需要删掉或替换的部分

#### 1. `ViT` 里当前“过早 AV 融合”的逻辑

现在 `ViT.forward()` 里是：

- video 轻投影
- audio 轻投影
- 可选 AV cross-attn
- 再拼接 patch embedding

这条路要改。

#### 2. 当前 `self.proj_video / self.audio_tcn_encoder / self.proj_audio / self.fusion`

这些都应该被新的三阶段 encoder 替换掉。  
不是全删不留，而是**语义上被新 backbone 取代**。

---

## 五、在 `Vit_gcnmodel.py` 里，方案 A 应该具体替换哪几块

### 第一个改动点：`ViT.__init__()`

你现在这里定义了：

- `self.proj_video`
- `self.audio_tcn_encoder`
- `self.proj_audio`
- `self.fusion`
- `self.embedding`
- `self.transformers`
- `self.classifier`。

### 方案 A 要改成

我建议你新增这些模块：

- `self.video_stem`
- `self.video_local_encoder`
- `self.video_global_encoder`
- `self.video_pool`
- `self.audio_stem`
- `self.audio_local_encoder`
- `self.audio_global_encoder`
- `self.audio_pool`
- `self.av_post_encoder_fusion`  
    这一步是“单模态先编码完，再交互”，不是一开始就交互

这样结构会清晰很多。

---

### 第二个改动点：`ViT.forward()`

你现在的 `forward()` 是一个“先投影再拼接”的路径。

### 方案 A 的新顺序应该是

1. `video = video_stem(video)`
2. `video = video_local_encoder(video)`
3. `video_tokens = video_global_encoder(video)`
4. `video_repr = video_pool(video_tokens)`
5. `audio = audio_stem(audio)`
6. `audio = audio_local_encoder(audio)`
7. `audio_tokens = audio_global_encoder(audio)`
8. `audio_repr = audio_pool(audio_tokens)`
9. 然后再做 AV interaction  
    比如：
    - 简单 concat + linear
    - 或轻量 cross-attn
    - 或 gated sum
10. 输出：

- `av_repr`
- 以及可选的 `video_repr / audio_repr`

### 这一步的含义

它把“先单模态编码，再交互”真正落实到了 forward 顺序上。

---

### 第三个改动点：`ViT_GCN_Fusion.forward()`

这部分不需要大改，但要适配新的 `ViT` 返回形式。

因为当前 fusion 分支默认假设 AV backbone 会返回某种 `vit_repr / audio_repr` 结构，后面再和 GCN 做 late/concat。

### 你要改成

让新的 AV backbone 明确返回：

- `video_tokens`
- `audio_tokens`
- `video_repr`
- `audio_repr`
- `av_repr`

然后 `ViT_GCN_Fusion` 再决定：

- 当前模式是 `av_only`
- 还是 `fusion`
- 用哪个 fusion head

这样接口语义更清楚。

---

## 六、方案 A 每个修改背后的代码原理与含义

我直接按“新增模块”来讲。

### 1. `ResidualTemporalConvBlock`

#### 原理

用卷积残差块先提取短程时序模式。

#### 含义

相当于“局部动态提纯器”，让 token 在进全局 Transformer 前先有更稳定的局部语义。

---

### 2. `TemporalTransformerStack`

#### 原理

在局部编码后的 token 上，建长程依赖。

#### 含义

相当于“单模态全局语义建模器”，不再让全局 Transformer 去吃毛坯特征。

---

### 3. `AttnPool1D`

#### 原理

对单模态 token 做注意力池化，得到更稳的 `repr`。

#### 含义

让每个模态先自己学一个稳定的 summary，再进入后融合。

---

### 4. `PostEncoderAVFusion`

#### 原理

只在单模态编码器完成后，再做交互。

#### 含义

它不是“早期混合”，而是“后期互补”，更符合你现在需要的顺序。

---

## 七、先确认 AV-only / GCN-only / AV+GCN 的信息增量关系，这一步怎么做

这一步非常重要，而且你现在的代码已经基本支持它了。

### 你要跑的 3 组

#### 1. AV-only

在配置里设：

- `MODEL_MODE = 'av_only'`

当前训练入口会在这种模式下直接构建 `ViT(...)`，不用 fusion 模型。

#### 2. GCN-only

设：

- `MODEL_MODE = 'gcn_only'`

当前框架保留这条模式，就是为了让你做这种对照。

#### 3. AV + GCN

设：

- `MODEL_MODE = 'fusion'`
- `FUSION_MODE = 'concat'` 或当前 best baseline

---

### 如何判断信息增量关系

你主要看三个差值：

#### `Δ(AV+GCN - AV-only)`

表示：

- GCN 对 AV 主干有没有真实增量

#### `Δ(AV-only - GCN-only)`

表示：

- 主判别信息是不是主要来自 AV 主干

#### `Δ(New AV-only - Old AV-only)`

表示：

- 你的强 AV backbone 本身有没有真的解决主矛盾

### 这一步的意义

这不是为了做最终结果，而是为了先确定：

> 你后面到底应该继续投资 AV 主干，还是继续折腾 GCN / fusion

---

## 八、DVLOG 和 LMVD 是否应该拆成不同文件

### 我的结论

**应该拆，但不是“全部复制两套”，而是“数据集特定部分拆开，共享部分保留公共模块”。**

原因很简单：你现在很多文件已经出现了大量 `if DATASET_SELECT == "DVLOG" else ...`，包括：

- 路径
- 维度
- batch size
- hyperparameter
- temporal stem
- shallow audio tcn
- MIL
- sliding eval
- region scheme
- aux gcn
- early stop 等等。

这会导致两个问题：

1. 文件越来越长，实验含义越来越难看清
2. DVLOG 和 LMVD 的实验逻辑本来就不同，却被硬塞进同一入口里

---

### 但我不建议怎么拆

我**不建议**你这样拆：

- 复制两份完整 `Vit_gcnmodel.py`
- 复制两份完整 `_02GCN_Transformer.py`
- 复制两份完整 runtime utils

因为那样后面会非常难维护。

---

### 我建议的拆法

#### 保留公共模块

这些应继续共用：

- `vit_gcn_runtime_utils.py`
- `_02GCN_Transformer.py`
- 一些基础 block，如 `Attention1d / Transformer / pooling / losses`
- 公共 fusion head

#### 拆开的部分

最值得拆开的，是：

### 1. 训练入口

- `train_dvlog.py`
- `train_lmvd.py`

因为两个数据集：

- split 策略不同
- 训练策略不同
- 增强策略不同
- 滑窗 / MIL / aux 使用也不同。

### 2. 配置文件

- `config_dvlog.py`
- `config_lmvd.py`

这比当前一个超长 `vit_gcn_config_train.py` 更清楚。

### 3. AV backbone

这是最值得拆开的模型部分。

因为 LMVD 和 DVLOG 的输入维度、时序长度、最优结构偏好都不一样：

- DVLOG：136 / 25，短、噪声大
- LMVD：171 / 128，设定不同。

所以建议：

- `av_backbone_dvlog.py`
- `av_backbone_lmvd.py`

共享公共小模块，但 backbone 本身分开。

### 4. 数据加载器

这个也建议拆 wrapper：

- `dvlog_loader.py`
- `lmvd_loader.py`

因为你现在 `MultiModalDataLoader` 已经把两者都塞进一个类里，里面到处判断 `self.dataset == 'DVLOG'`。

---

## 九、我建议你的新文件组织

可以这样：

- `models/common_blocks.py`
- `models/gcn_face.py`
- `models/av_backbone_dvlog.py`
- `models/av_backbone_lmvd.py`
- `models/fusion_heads.py`
- `models/vit_gcn_fusion.py`
- `configs/config_dvlog.py`
- `configs/config_lmvd.py`
- `train/train_dvlog.py`
- `train/train_lmvd.py`
- `data/dvlog_loader.py`
- `data/lmvd_loader.py`

### 这样拆的好处

- 公共逻辑不重复
- 数据集差异清楚
- 你以后做方案 A / B 时，不会一直被 `if DATASET_SELECT` 干扰

---

## 十、我对你下一步最直接的建议

### 先做这两步

#### 第一步

先用当前代码里的 `MODEL_MODE` 跑出：

- `AV-only`
- `GCN-only`
- `AV+GCN`

看信息增量关系。

#### 第二步

开始方案 A，但先只改 **DVLOG 的 AV backbone**，别同时改 LMVD。

因为你当前主要卡的是 DVLOG，DVLOG 的问题和 LMVD 不完全一样。  
这也是为什么我支持你把两者训练和 backbone 文件拆开。

---

## 十一、一句话总结

### 方案 A 如何具体修改

就是：

> 把当前 `ViT` 里的“轻投影 + 早交互 + 共享小 Transformer”  
> 改成“视频三阶段编码器 + 音频三阶段编码器 + 后交互”。

### 为什么这样改

因为你现在的瓶颈更像是：

- 主干 token 太弱
- 单模态自己还没学清楚
- 后面 fusion 只能做弱表示上的后处理。

### 是否应该把 DVLOG 和 LMVD 拆开

**应该拆开训练入口、配置和 AV backbone；共享基础模块和 GCN。**  
这样最清楚，也最不容易把两个数据集的实验逻辑继续混在一起。