## 核心文件归属

#### 通用训练主入口（两套数据都依赖）

vit_gcn_config_train.py
Vit_gcnmodel.py

AV backbone（已拆分）
av_backbone_common.py
av_backbone_dvlog.py
av_backbone_lmvd.py

#### DVLOG 训练相关

config_dvlog.py
train_dvlog.py
config_dvlog_av_only.py
train_dvlog_av_only.py
config_dvlog_gcn_only.py
train_dvlog_gcn_only.py
config_dvlog_av_only_legacy.py
train_dvlog_av_only_legacy.py

#### LMVD 训练相关

config_lmvd.py
train_lmvd.py

三个增量统计现在的位置

增量 registry 与打印逻辑在
vit_gcn_config_train.py
汇总项为
AV+GCN - AV-only
AV-only - GCN-only
New AV-only - Old AV-only
运行后会在 logs 目录更新 dvlog_mode_metrics_registry.json（自动累积各模式均值结果）
