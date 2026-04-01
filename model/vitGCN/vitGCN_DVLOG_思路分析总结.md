# vitGCN 在 DVLOG 上的结构思路与问题分析

本文档基于当前工作区里的 `model/vitGCN/` 代码，以及 `Large-Scale-Multimodal-Depression-Detection-main` 项目中对 D-Vlog 的处理方式整理而成，目的是回答下面几个问题：

1. 当前 `vitGCN` 模型整体到底在做什么。
2. 音频、视频、GCN 面部关键点三条分支分别是怎么处理的。
3. 为什么前面做了 IA-style / concat 路线修改以后，效果依然基本没有明显提升。
4. 参考项目 `Large-Scale-Multimodal-Depression-Detection-main` 是怎么处理 DVLOG 的。
5. 参考项目是不是“依次做单模态编码器训练”。

---

## 1. 当前 vitGCN 的总体思路

当前 `vitGCN` 不是一个“单纯的音视频融合模型”，也不是一个“单纯的人脸关键点 GCN 模型”，而是一个三分支结构：

1. 视频分支：输入视频特征序列。
2. 音频分支：输入音频特征序列。
3. 面部关键点分支：从面部 68 点构建区域图，再做 GCN 和时间建模。

最终目标是：

1. 先把视频和音频编码到一个共享的 ViT 时序表征里。
2. 再把人脸关键点通过 GCN 编码成另一条结构先验表征。
3. 最后在 late / intermediate / concat / afi 等不同 fusion 模式下，把 ViT 分支和 GCN 分支融合起来做二分类。

从建模意图上看，这个模型的核心假设是：

1. 视频和音频能提供“多模态行为信号”。
2. 面部关键点能提供“局部结构、微表情、几何动态”。
3. 两者结合后，应该比纯音视频更稳。

这个思路本身没有问题，但它也带来一个现实困难：

1. 你现在优化的不是一个简单的双模态模型，而是一个“音频 + 视频 + 人脸图结构 + 多级 mask + 质量门控 + 多种 fusion”的复合系统。
2. 这种系统里，最后的 fusion 头往往不是主矛盾，真正决定上限的是前面各分支输入质量和编码器质量。

---

## 2. 当前 vitGCN 中音频、视频、GCN 是怎么做的

## 2.1 数据输入层

当前 DVLOG 的数据加载在 `kfoldLoader_multimodal.py` 里完成，整体流程大致如下：

1. 从 `{index}/{index}_visual.npy` 读取视频特征。
2. 从 `{index}/{index}_acoustic.npy` 读取音频特征。
3. 如果视频长度和音频长度不一致，先截断到二者最短长度。
4. 保留原始视频序列 `input_video_raw`，专门给人脸关键点分支使用。
5. 对视频、音频做 pad/crop 到固定长度 `T_target`。
6. DVLOG 下对视频和音频做轻量归一化。
7. 从视频的 136 维中解码出 68 点坐标，构建 face regions。
8. 计算质量指标 `q_v` 和 `q_g`。
9. 返回：
   `video, audio, face_regions, actual_lens, label, quality, sample_weight`

也就是说，当前 DVLOG 里的人脸关键点并不是来自单独的人脸关键点文件，而是直接从视频特征的 136 维里反解出来的。

这点非常关键，因为它意味着：

1. 你的人脸分支不是一个独立高质量来源。
2. 它和视频分支本质上共享同源信息。
3. 如果 136 维本身的人脸定位质量一般，那么 GCN 分支天然上限就会比较低。

---

## 2.2 视频与音频分支

当前 `ViT` 分支在 `Vit_gcnmodel.py` 里，流程可以概括为：

1. 视频输入 `(B, T, video_dim)` 转成 `(B, video_dim, T)`。
2. 音频输入 `(B, T, audio_dim)` 转成 `(B, audio_dim, T)`。
3. 视频用 `Conv1d + GroupNorm + ReLU` 做线性投影，得到 `D_PROJECTION = dim/2` 维。
4. 音频先可选过一个浅层 `ShallowAudioTCNEncoder`，再用 `Conv1d + GroupNorm + ReLU` 投影到同样维度。
5. 如果打开 `use_av_cross_attn`，就让视频 token 去 cross-attend 音频 token。
6. 把“融合后视频”和“原始音频投影”拼接成 `(B, 256, T)`。
7. 再做 patch embedding、CLS token、位置编码和一串轻量 Transformer。

这里有几个要点：

1. 当前视频和音频编码器是从头训练的小型投影 + 小型 Transformer。
2. 没有像参考项目那样直接使用强预训练的 AST 或 VisualMAE 主干。
3. 所以你现在的 ViT 分支，本质上更像“轻量时序 Transformer”，不是“大模型特征提取器”。

换句话说，当前 vitGCN 的 AV 主干，偏向“小数据集上自己学一套”，而不是“站在强预训练表示上微调”。

---

## 2.3 音频分支具体在学什么

音频分支的作用并不是单独做一个很强的音频识别模型，而是：

1. 先通过 `Conv1d` 把低维音频特征投影到共享子空间。
2. 可选地用浅层 TCN 提前建一些局部时间上下文。
3. 参与和视频的 AV cross-attn。
4. 最终一起进入 ViT patch token。

因此当前音频分支的定位更像：

1. 一个轻量辅助模态。
2. 一个和视频共同组成 ViT token 的组成部分。

它不是参考项目里那种“独立 AST 音频编码器先抽强时频语义，再进入融合头”的路线。

---

## 2.4 视频分支具体在学什么

视频分支的作用是：

1. 用 136 维视频特征序列作为主时序输入。
2. 先做轻量投影。
3. 可选地跟音频做一次 AV cross-attn。
4. 再进入 ViT 进行 patch 化和时序 Transformer 编码。

这条路的特点是：

1. 它吃的是“已经抽好的视频特征”，不是原始帧。
2. 它也不是像参考项目那样用 VisualMAE 预训练网络从视觉 token 里抽更高级的语义。
3. 因此它更依赖你当前这套特征本身的质量。

---

## 2.5 GCN 面部关键点分支

GCN 分支在 `_02GCN_Transformer.py`，整体是“区域 GCN + 跨区域融合 + 时间建模”的三段式。

### 第一步：从 68 点构造每帧区域特征

你先把 68 点分成若干区域，目前支持两种方案：

1. `legacy6`
2. `symptom7`

然后对每个点构造几何特征：

1. 归一化坐标
2. 速度
3. 速度模长
4. 到几个参考点的距离
5. 左右对称性
6. validity 通道

最后单个点的特征维度是 11 维。

然后对每个区域分别做两层 GCN，把区域内部的点关系编码成区域表示。

### 第二步：区域之间再融合

区域级特征形成 `region_tokens` 后，再做跨区域融合。
当前你在 DVLOG 配置里用的是：

1. `REGION_FUSION_MODE = 'concat_linear'`

也就是说，并不是最复杂的跨区域 attention，而是把所有区域拼起来后过一层线性映射。

这条路更稳，但表达能力比 cross-attn 弱。

### 第三步：时间建模

区域融合结果再和一个全局动态分支拼接，得到每一帧的人脸结构表征 `per_frame`。

然后根据配置做时间聚合。
当前 DVLOG 用的是：

1. `GCN_TEMPORAL_MODE = 'meanmax'`

这意味着：

1. DVLOG 上 GCN 分支不是走时间 Transformer。
2. 而是走更稳、更便宜的 mean + max 投影。

最后输出：

1. `temporal_out`
2. `gcn_repr`
3. `time_mask`
4. `logits_gcn`

---

## 2.6 三条分支最后是怎么对齐的

在 `ViT_GCN_Fusion.forward()` 里，当前逻辑是：

1. AV 分支先得到 `vit_transformed`。
2. 去掉 CLS 得到 `vit_features`。
3. GCN 分支得到 `temporal_out`，再投影成 `gcn_features`。
4. 再把 `gcn_features` 按 patch 数量池化成 `gcn_pooled`。
5. 使用 `q_v`、`q_g` 做 token 级质量门控。
6. 使用 `patch_valid` 和 `gcn_out['time_mask']` 合成最终 `time_mask`。
7. 对两个分支做 masked mean，得到 `vit_repr` 和 `gcn_repr`。
8. 最后进入不同的 fusion mode。

如果是 `concat` 路线，现在已经是：

1. token 分支：`fusion_head(vit_features, gcn_pooled, time_mask)`
2. summary 分支：`concat_head(vit_repr, gcn_repr)`
3. 最终：可学习 `blend`

所以你最近改的那部分，本质上只发生在“很后面的融合层”。

---

## 3. 为什么做了 IA-style / concat 修改以后，还是基本没有提升

这是现在最核心的问题。

结论先说：

当前没有明显提升，大概率不是因为你这次改动“写错了”，而是因为你改动的位置太靠后，而当前 DVLOG 的主要瓶颈更可能在前面的表征质量、分支可靠性和任务设定上。

下面分开解释。

### 3.1 你改的是融合头，不是主干表示

你这次 IA-style 修改，主要动的是：

1. `concat` 路线里的 token fusion 头
2. summary/token 的最终混合方式

但没有改：

1. 视频特征本身质量
2. 音频特征本身质量
3. 人脸关键点来源质量
4. GCN 区域构图质量
5. GCN 时间建模方式

如果真正的瓶颈在“输入和主干弱”，那么后面把 `BiGate` 换成 `IAResidual`，往往只能带来很小的增益。

也就是说：

1. 你现在是在“弱骨干 + 复杂后融合”上调头。
2. 但参考项目更接近“强骨干 + 相对简单融合”。

这是两条路线最大的差异。

### 3.2 当前 AV 编码器是从头学的小模型，不是强预训练编码器

这是我认为最关键的一点。

参考项目里：

1. 音频分支是 AST 风格的 `AudioTransformerModel`
2. 视觉分支是 `GenerateVisualModel`，底层来自 VisualMAE/ViT 预训练

也就是说，参考项目的音频和视频编码器不是从零开始学。

而你当前 vitGCN 里：

1. 视频分支是 `Conv1d + GroupNorm + 小型 Transformer`
2. 音频分支是 `Conv1d + 可选浅层 TCN + 小型 Transformer`
3. 整体更偏轻量、从头训练

这会导致一个结果：

1. 你后面的 fusion 再怎么精细，也是在比较弱的 token 表示上做。
2. 如果底层 token 已经不够判别，fusion 头很难凭空救回来。

所以“改 fusion 没明显提升”是完全可能的。

### 3.3 DVLOG 的 GCN 分支噪声可能比你想象得更大

当前 GCN 的输入来源不是独立的人脸关键点检测结果文件，而是从 DVLOG 的 136 维视频特征里拆出来的 68 点。

这带来几个问题：

1. 这 68 点不是一个专门为人脸图建模优化过的高质量输入。
2. 它与视频分支高度同源，额外信息增量可能有限。
3. 如果关键点质量一般，那么 GCN 分支不仅不能显著补充信息，还可能引入噪声。

你代码里其实已经意识到了这个问题，所以加了很多保护：

1. `q_g` 质量门控
2. `time_mask`
3. `face_valid_thresh`
4. 低质量帧过滤

但副作用是：

1. GCN 分支有效信号可能被进一步稀释。
2. 最终 `gcn_pooled` 经常只是“弱而保守”的结构先验，而不是强判别信号。

于是最后现象就会是：

1. GCN 分支不会明显拉高上限。
2. 但也不一定完全坏，只是贡献很有限。

### 3.4 当前 DVLOG 配置本身就在走“稳而保守”的 GCN 方案

你当前配置里，DVLOG 其实不是最强 GCN 配置，而是偏保守配置：

1. `REGION_FUSION_MODE = 'concat_linear'`
2. `GCN_TEMPORAL_MODE = 'meanmax'`
3. `USE_GLOBAL_BRANCH = False`
4. `DVLOG_AUX_ZERO = True`

这意味着：

1. 区域间交互偏弱
2. 时间建模偏弱
3. 全局动态支路关闭
4. GCN 辅助监督关闭

从稳定性角度，这样做是合理的。
但从“让 GCN 真正学成一个强分支”的角度，它又明显偏保守。

所以现在出现“fusion 结构改了，但增益不明显”，很可能是因为：

1. 你的 GCN 分支根本还没有强到足以让 fusion 结构发挥差异。
2. 最终学习器大概率还是更依赖 AV 分支。

### 3.5 当前配置下，IA 改动本身可能根本没启用

这一点要特别提醒。

以当前仓库里的配置文件为准，你现在是：

1. `USE_IA_RECALIBRATION = False`
2. `IA_RECAL_MODE = 'off'`

如果你是按当前文件直接训练，那么前面那套 IA-style 修改根本不会生效，模型会回退到旧逻辑。

所以如果你觉得“我都改了怎么还是没提升”，要先区分两种情况：

1. 你之前某次实验确实开过 IA，但是没提升。
2. 你当前工作区配置其实已经把 IA 关了，所以当前这份配置跑出来本来就不会体现改动收益。

这点一定要和实验日志对应着确认。

### 3.6 现在 token 对齐方式本身也会削弱 GCN 的细粒度优势

GCN 分支先在帧级得到 `temporal_out`，然后你再做：

1. `gcn_proj`
2. `_mask_aware_pool_temporal()`
3. 对齐到 ViT 的 patch 数量

这一步会让 GCN 的帧级细节被平均进 patch bin 里。

因此即使 GCN 有一些很短暂的局部异常动态：

1. 也可能在 patch pooling 时被抹平。
2. 最后只能以比较平滑的结构先验形式参与融合。

而你最近改的 IA-style，本质上又是建立在 `gcn_pooled` 上的。

所以这个 IA 分支拿到的“指导信号”本来就不是特别尖锐。

### 3.7 当前训练目标更像“先求稳”，不是“先冲高”

你现在整套 DVLOG 配置明显带有“先降低波动”的思路：

1. `LOSS_MODE='focal_no_pw'`
2. `USE_SEGMENT_MIL=False`
3. `USE_SLIDING_SEGMENT_EVAL=True`
4. `SD_RATE=0.0`
5. dropout 偏低

这会让训练更稳，但也意味着：

1. 模型更难从复杂交互里榨取额外增益。
2. 很多结构改动最后只会表现为“波动变小”，而不是“均值显著变高”。

所以如果你看到结果是：

1. 指标均值差不多
2. 但 seed 方差略有变小

这其实已经说明 IA-style 改动可能在起作用了，只是作用是“稳态改善”，不是“峰值提升”。

---

## 4. 参考项目 Large-Scale-Multimodal-Depression-Detection-main 是怎么处理 DVLOG 的

参考项目的 DVLOG 思路比你当前 vitGCN 更简单，也更“预训练驱动”。

### 4.1 数据处理方式

在 `datasets_process/dvlog.py` 里，它直接做的是：

1. 读 `visual.npy`
2. 读 `acoustic.npy`
3. 如果长度不同，截断到最短长度
4. 在特征维拼接成 `(T, 161)`，其中 `136 + 25 = 161`
5. collate 时只做 pad，不额外拆出 face graph

也就是说，参考项目对 DVLOG 的主输入只有两条：

1. 视频特征
2. 音频特征

它没有你现在这条“面部关键点 GCN”旁路。

### 4.2 编码器设计

参考项目的核心是 `MultiModalDepDet`。

它会把输入拆成：

1. 前 136 维给视频
2. 后 25 维给音频

然后：

1. 音频先过 `conv_audio`
2. 视频先过 `conv_video`
3. 音频进入 `AudioTransformerModel`
4. 视频进入 `GenerateVisualModel`

这里最重要的是：

1. 音频编码器是 AST 风格的预训练音频 Transformer
2. 视频编码器是 VisualMAE/ViT 风格的预训练视觉 Transformer

所以它不是把融合写得特别复杂，而是先把单模态编码器做强。

### 4.3 融合方式

参考项目支持：

1. `audio`
2. `video`
3. `lt`
4. `it`
5. `ia`
6. `MT`
7. 一些 add/multi/concat/tensor ablation

其中：

1. `audio` 和 `video` 是单模态模式
2. `lt` 是 late transformer fusion
3. `it` 是 intermediate transformer fusion
4. `ia` 是 intermediate attention fusion

这说明它会做单模态对照实验，但不是说它默认训练流程是“先训练单模态，再接着训练双模态”。

---

## 5. 参考项目是不是依次进行单模态编码器训练

默认不是。

更准确地说，参考项目的默认逻辑是：

1. 先直接加载预训练好的音频编码器和视频编码器。
2. 然后根据 `fusion` 参数，选择当前要跑的模式。
3. 直接端到端训练当前模式。

它支持 `fusion='audio'` 或 `fusion='video'`，但这更像：

1. 单模态 ablation baseline
2. 对照实验

而不是：

1. 先训练 audio-only
2. 再训练 video-only
3. 再把两个 checkpoint 拼起来继续训练

代码里并没有这样一个“三阶段串联训练”的默认流程。

所以对于你的问题，回答是：

1. 参考项目会做单模态模式训练作为对照。
2. 但默认不是依次单模态预训练再融合。
3. 它真正依赖的是“强预训练编码器”，而不是“分三阶段训练策略”。

---

## 6. 你当前 vitGCN 与参考项目的本质差异

把两套系统并列看，会更清楚为什么你现在改 fusion 效果不大。

### 当前 vitGCN

1. AV 主干偏轻量，多数部分从头训练。
2. 增加了 GCN face graph 分支。
3. 有质量门控、时序 mask、region fusion、temporal pooling 等大量机制。
4. 结构更复杂，变量更多。
5. 优势是可解释性更强，也更适合做细粒度 ablation。
6. 劣势是每一层都有机会成为瓶颈。

### 参考项目 MMFformer / MultiModalDepDet

1. 没有 face GCN 分支。
2. 主要依赖音频和视频两条主模态。
3. 单模态编码器更强，且带预训练。
4. 融合结构虽然也有多种模式，但整体复杂度低于你现在这条三分支图模型。
5. 更接近“先把基础表征做强，再比较融合方式”。

所以如果你直接问：

“为什么我只改 concat 路线最后一段，结果还是不涨？”

一个非常现实的回答就是：

1. 参考项目的增益很多来自更强的单模态编码器。
2. 你当前改的只是最后的融合头。
3. 两者作用层级根本不在同一层面。

---

## 7. 我对“为什么当前改动没提升”的最终判断

如果只保留最核心的判断，我会总结成下面五点。

### 第一，当前主瓶颈可能不在 fusion head

如果底层 AV token 和 GCN token 本身不够强，后面 fusion 再怎么换，提升都有限。

### 第二，GCN 分支在 DVLOG 上可能是“弱增益甚至噪声增益”

因为它来自同源 136 维特征反解的人脸点，而不是独立高质量 landmark 流。

### 第三，当前 DVLOG 配置把 GCN 设得太保守

`concat_linear + meanmax + no global branch + aux off` 更像“稳住”，不像“冲上限”。

### 第四，你最近改的是后融合，不是强主干

而参考项目真正强的地方，是音频 AST 和视频 VisualMAE 这类强预训练编码器。

### 第五，当前仓库配置里 IA 还是关闭的

如果按当前配置直接跑，最近那套 IA-style 修改根本不会生效。

---

## 8. 如果你接下来要继续做实验，我更建议的顺序

如果目的是回答“为什么不涨”，我建议不要继续优先改 fusion 头，而是按下面顺序排查。

### 8.1 先确认当前实验到底有没有真的启用 IA

先看：

1. `USE_IA_RECALIBRATION`
2. `IA_RECAL_MODE`
3. 日志里有没有 `[ConcatFusion] use_ia=...`

如果当前配置仍然是 off，那么先别讨论结构收益。

### 8.2 先做单模态和双模态基线对照

建议至少有：

1. AV only
2. GCN only
3. AV + GCN late
4. AV + GCN concat old
5. AV + GCN concat IA

如果 `GCN only` 本身很弱，而 `AV + GCN` 也不比 `AV only` 强多少，那就说明问题不在最后 fusion，而在 GCN 分支信息增量不足。

### 8.3 优先验证“强主干”而不是“更复杂 fusion”

真正更接近参考项目的方向不是继续加复杂交互，而是：

1. 强化 AV 编码器
2. 让视频和音频先有更好的单模态表示

### 8.4 再决定 GCN 是“辅助分支”还是“主增益分支”

如果最终发现 GCN 增益一直很弱，那更合理的定位可能是：

1. GCN 用作辅助正则或置信修正
2. 而不是指望它成为主融合增益来源

---

## 9. 一句话总结

当前 `vitGCN` 的思路是：

1. 用轻量 AV ViT 学多模态时序表示；
2. 用 face landmark GCN 学结构几何先验；
3. 再把两者在 late / intermediate / concat 等层级做融合。

而最近改动后仍然没有明显提升，最可能的原因不是“最后那段 concat 没改好”，而是：

1. 当前真正的瓶颈更可能在主干表示不够强；
2. DVLOG 上的 face GCN 分支信息增量有限且噪声不小；
3. 你改的是后融合头，但参考项目真正强的是预训练音频/视频编码器。

参考项目默认也不是“依次先训单模态再训融合”，而是：

1. 直接加载强预训练单模态编码器；
2. 再按 `audio / video / lt / it / ia` 等模式做端到端训练。
