import numpy as np
from pathlib import Path

from symptom7_plot_utils import plot_partition_comparison, plot_partition_landmarks

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "quality_audit_out"

# ===== LMVD landmarks 文件路径 =====
lmvd_file = ROOT / "data/LMVD_Feature/Video_landmarks_npy/001.npy"

raw = np.load(lmvd_file)
print("LMVD raw shape:", raw.shape)
print("LMVD dtype:", raw.dtype)

if raw.ndim == 3 and raw.shape[1] == 68 and raw.shape[2] == 2:
    pts_seq = raw.astype(np.float32)

elif raw.ndim == 2 and raw.shape[1] == 136:
    # 这里只是临时尝试 split 模式，你需要画图确认
    x = raw[:, :68]
    y = raw[:, 68:]
    pts_seq = np.stack([x, y], axis=-1).astype(np.float32)

else:
    raise ValueError(f"Unsupported LMVD landmark shape: {raw.shape}")

frame_idx = 0
for i in range(len(pts_seq)):
    if np.linalg.norm(pts_seq[i]) > 1e-6:
        frame_idx = i
        break

pts = pts_seq[frame_idx]
plot_partition_landmarks(
    pts,
    title=f"LMVD legacy6 | frame={frame_idx}",
    save_path=str(OUT_DIR / "lmvd_legacy6.png"),
    scheme="legacy6",
)
plot_partition_landmarks(
    pts,
    title=f"LMVD symptom7 | frame={frame_idx}",
    save_path=str(OUT_DIR / "lmvd_symptom7.png"),
    scheme="symptom7",
)
plot_partition_comparison(
    pts,
    dataset_name="LMVD",
    frame_idx=frame_idx,
    save_path=str(OUT_DIR / "lmvd_legacy6_vs_symptom7.png"),
)
print("Saved:", OUT_DIR / "lmvd_legacy6.png")
print("Saved:", OUT_DIR / "lmvd_symptom7.png")
print("Saved:", OUT_DIR / "lmvd_legacy6_vs_symptom7.png")