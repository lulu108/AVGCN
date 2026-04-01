import numpy as np
from pathlib import Path

from symptom7_plot_utils import plot_partition_comparison, plot_partition_landmarks

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "quality_audit_out"

def dvlog_seq136_to_pts68x2(seq136):
    """
    seq136: (T,136)
    return: (T,68,2)
    """
    x = seq136[:, :68]
    y = seq136[:, 68:]
    return np.stack([x, y], axis=-1).astype(np.float32)

# ===== DVLOG 文件路径（固定样本） =====
dvlog_file = ROOT / "data/dvlog-dataset/dvlog-dataset/0_visual.npy"

raw = np.load(dvlog_file)   # (T,136)
print("DVLOG raw shape:", raw.shape, "dtype:", raw.dtype)
pts_seq = dvlog_seq136_to_pts68x2(raw)
print("Decoded pts_seq shape:", pts_seq.shape)

# 取第一帧非零帧
frame_idx = 0
for i in range(len(pts_seq)):
    if np.linalg.norm(pts_seq[i]) > 1e-6:
        frame_idx = i
        break

pts = pts_seq[frame_idx]   # (68,2)
plot_partition_landmarks(
    pts,
    title=f"DVLOG legacy6 | frame={frame_idx}",
    save_path=str(OUT_DIR / "dvlog_legacy6.png"),
    scheme="legacy6",
)
plot_partition_landmarks(
    pts,
    title=f"DVLOG symptom7 | frame={frame_idx}",
    save_path=str(OUT_DIR / "dvlog_symptom7.png"),
    scheme="symptom7",
)
plot_partition_comparison(
    pts,
    dataset_name="DVLOG",
    frame_idx=frame_idx,
    save_path=str(OUT_DIR / "dvlog_legacy6_vs_symptom7.png"),
)
print("Saved:", OUT_DIR / "dvlog_legacy6.png")
print("Saved:", OUT_DIR / "dvlog_symptom7.png")
print("Saved:", OUT_DIR / "dvlog_legacy6_vs_symptom7.png")