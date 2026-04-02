# AV-only 拆分对照实验说明

## 1. 为什么要拆成 video-only / audio-only

当前 AV-only 提升不明显时，至少有 3 种可能：

1. 视频分支本身还没学好。
2. 音频分支引入了噪声，抵消了收益。
3. 融合顺序或交互方式仍不匹配当前数据分布。

把 AV-only 拆成 `video_only` 和 `audio_only` 后，你可以直接判断：

- 若 `video_only ≈ av_only`，说明音频边际贡献有限。
- 若 `audio_only` 明显低，而 `av_only` 也没超过 `video_only`，应优先增强视频主干。
- 若 `av_only` 显著高于两者，说明双模态互补有效，后续重点应放在融合策略。

## 2. 模式定义

在统一入口里，当前支持：

- `fusion`: AV + GCN 完整融合。
- `av_only`: 仅 AV 主干（不走 GCN）。
- `video_only`: 仅视频输入（音频输入置零）。
- `audio_only`: 仅音频输入（视频输入置零）。
- `gcn_only`: 仅 GCN 分支。

说明：

- `video_only/audio_only` 是在 ViT-only 路径上做输入屏蔽，不改动主干结构。
- 这样可保证对照公平：参数规模和训练流程尽量一致，仅改变模态可见性。

## 3. 运行命令

在仓库根目录执行。

### 3.0 先改什么文件（A / V / AV）

你有两种方式切换实验：

1. 直接改命令行 `--exp`（推荐，最省事）。
2. 固定改配置文件里的默认实验，再直接运行训练脚本。

#### 方式 A：命令行直接切换（推荐）

- DVLOG: 改命令参数 `--exp` 即可，不需要改文件。
- LMVD: 改命令参数 `--exp` 即可，不需要改文件。

#### 方式 B：修改默认实验，再运行脚本

1. DVLOG：修改 `model/vitGCN/DVLOG_AVGCN/config_dvlog.py`

- 改 `FILE_EXPERIMENT` 为：
  - `audio_only` 跑 A-only
  - `video_only` 跑 V-only
  - `av_only_new` 跑新版 AV-only
  - `fusion` 跑 AV+GCN

2. LMVD：修改 `model/vitGCN/LMVD_AVGCN/config_lmvd.py`

- 改 `FILE_EXPERIMENT` 为：
  - `audio_only` 跑 A-only
  - `video_only` 跑 V-only
  - `av_only` 跑 AV-only
  - `fusion` 跑 AV+GCN

3. 如果你要改单模态 clean path 或强编码器开关（DVLOG）：修改 `model/vitGCN/DVLOG_AVGCN/config_dvlog.py` 的这些键：

- `SINGLE_MODALITY_CLEAN_PATH`
- `USE_STRONG_AUDIO_ENCODER`
- `USE_STRONG_VIDEO_ENCODER`
- `AUDIO_FIXED_LEN`
- `VIDEO_FIXED_LEN`
- `VIDEO_USE_DELTA`

### 3.1 DVLOG

#### 3.1.1 一条命令直接跑

- A-only（音频）

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp audio_only
```

- V-only（视频）

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp video_only
```

- AV-only（不走 GCN）

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp av_only_new
```

- AV+GCN（完整融合）

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp fusion
```

#### 3.1.2 先改配置再跑

1. 打开 `model/vitGCN/DVLOG_AVGCN/config_dvlog.py`。
2. 修改 `FILE_EXPERIMENT`。
3. 运行：

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py
```

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp fusion
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp av_only_new
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp video_only
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp audio_only
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp gcn_only
```

可选（旧版 AV-only 主干）：

```bash
python model/vitGCN/DVLOG_AVGCN/train_dvlog.py --exp av_only_legacy
```

### 3.2 LMVD

#### 3.2.1 一条命令直接跑

- A-only（音频）

```bash
python model/vitGCN/LMVD_AVGCN/train_lmvd.py --exp audio_only
```

- V-only（视频）

```bash
python model/vitGCN/LMVD_AVGCN/train_lmvd.py --exp video_only
```

- AV-only（不走 GCN）

```bash
python model/vitGCN/LMVD_AVGCN/train_lmvd.py --exp av_only
```

- AV+GCN（完整融合）

```bash
python model/vitGCN/LMVD_AVGCN/train_lmvd.py --exp fusion
```

#### 3.2.2 先改配置再跑

1. 打开 `model/vitGCN/LMVD_AVGCN/config_lmvd.py`。
2. 修改 `FILE_EXPERIMENT`。
3. 运行：

```bash
python model/vitGCN/LMVD_AVGCN/train_lmvd.py
```

```bash
python model/vitGCN/LMVD_AVGCN/train_lmvd.py --exp fusion
python model/vitGCN/LMVD_AVGCN/train_lmvd.py --exp av_only
python model/vitGCN/LMVD_AVGCN/train_lmvd.py --exp video_only
python model/vitGCN/LMVD_AVGCN/train_lmvd.py --exp audio_only
python model/vitGCN/LMVD_AVGCN/train_lmvd.py --exp gcn_only
```

## 4. 建议的最小对照矩阵

至少跑以下 4 组（同随机种子/同 fold 设置）：

1. `fusion`
2. `av_only` (DVLOG 推荐 `av_only_new`)
3. `video_only`
4. `audio_only`

优先观察：`F1`、`Recall(Depression)`、阈值稳定性（dev sweep 后 test 表现）。

## 5. 结果解读模板

你可以按如下方式快速归因：

- 情况 A：`video_only ≈ av_only` 且两者都明显优于 `audio_only`
  结论：当前音频贡献弱，优先提升视频主干或清理音频噪声。

- 情况 B：`av_only` 明显高于 `video_only`
  结论：音频有增益，后续可继续优化 AV 交互而非完全偏向视频。

- 情况 C：`audio_only` 尚可但 `av_only` 没明显提升
  结论：存在融合瓶颈，建议优先检查 cross-attn/gate 与时序对齐策略。
