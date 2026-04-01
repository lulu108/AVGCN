"""
Verify DVLOG 136-dim visual feature coordinate encoding:
  Mode A - Interleaved: [x0,y0, x1,y1, ..., x67,y67]  -> reshape(68,2)
  Mode B - Split:       [x0..x67, y0..y67]             -> stack([vec[:68], vec[68:]], -1)

Correct result: standard 68-point face topology visible (jaw/brow/eye/nose/mouth)
Wrong result:   points scattered randomly, no face shape
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Find a sample with actual non-zero data ───────────────────────────────────
CANDIDATES = [
    r"data/dvlog-dataset/dvlog-dataset/0_visual.npy",
]

data = None
used_file = None
for path in CANDIDATES:
    d = np.load(path)
    if np.any(d != 0):
        data = d
        used_file = path
        break
    else:
        print(f"[SKIP] {path} is all-zero")

if data is None:
    raise RuntimeError("All candidate files are all-zero! Cannot verify encoding.")

print(f"[INFO] Using: {used_file}  shape={data.shape}")

# Find first frame where L2-norm > threshold (more robust than exact==0)
THRESH = 1e-4
frame_idx = 0
for i in range(data.shape[0]):
    if np.linalg.norm(data[i]) > THRESH:
        frame_idx = i
        break
print(f"[INFO] First non-zero frame index: {frame_idx}")

vec = data[frame_idx].astype(np.float32)             # (136,)
print(f"[INFO] vec min={vec.min():.4f}  max={vec.max():.4f}  mean={vec.mean():.4f}  norm={np.linalg.norm(vec):.4f}")

# ── 两种解码方式 ───────────────────────────────────────────────────────────────
pts_interleave = vec.reshape(68, 2)                        # 方式 A：交错
pts_split      = np.stack([vec[:68], vec[68:]], axis=-1)   # 方式 B：分离

# ── 68-point face region colors (English labels to avoid font issues) ─────────
REGION_COLORS = {
    "jaw"    : (range(0,  17), "skyblue",   "Jaw 0-16"),
    "l_brow" : (range(17, 22), "orange",    "L.Brow 17-21"),
    "r_brow" : (range(22, 27), "darkorange","R.Brow 22-26"),
    "nose"   : (range(27, 36), "green",     "Nose 27-35"),
    "l_eye"  : (range(36, 42), "red",       "L.Eye 36-41"),
    "r_eye"  : (range(42, 48), "darkred",   "R.Eye 42-47"),
    "mouth"  : (range(48, 68), "purple",    "Mouth 48-67"),
}

def plot_pts(pts, title, ax):
    """将 68 点按分区着色并标注编号，y 轴翻转（图像坐标→屏幕坐标）"""
    legend_handles = []
    for region, (indices, color, label) in REGION_COLORS.items():
        idx = list(indices)
        xs  = pts[idx, 0]
        ys  = -pts[idx, 1]   # 翻转 y 使图像显示方向正确
        sc  = ax.scatter(xs, ys, s=40, c=color, label=label, zorder=3)
        for i in idx:
            ax.text(pts[i, 0], -pts[i, 1], str(i),
                    fontsize=6, ha='center', va='bottom', color=color)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
sample_name = os.path.basename(os.path.dirname(used_file))
fig.suptitle(f"{sample_name}_visual.npy  frame={frame_idx}  --  136-dim decode comparison", fontsize=13)

plot_pts(pts_interleave, "Mode A: Interleaved  reshape(68,2)\n[x0,y0, x1,y1, ..., x67,y67]", axes[0])
plot_pts(pts_split,      "Mode B: Split  stack([vec[:68], vec[68:]], -1)\n[x0..x67, y0..y67]", axes[1])

plt.tight_layout()
save_path = "dvlog_decode_verify.png"
plt.savefig(save_path, dpi=150)
print(f"[INFO] 已保存: {save_path}")
plt.show()

# ── 简单数值诊断（辅助判断哪种"看起来更像脸"）────────────────────────────────
def face_sanity(pts, name):
    """
    Rough topology check:
      1. Jaw points (0-16) should be roughly monotone in x (left to right)
      2. Mouth center y (48-67) > Eye center y (36-47) in image coords (y increases downward)
    """
    jaw_x_sorted  = np.all(np.diff(pts[0:17, 0]) >= -5)
    mouth_y_mean  = pts[48:68, 1].mean()
    eye_y_mean    = pts[36:48, 1].mean()
    mouth_below   = mouth_y_mean > eye_y_mean
    print(f"[SANITY {name}] jaw_x_monotone={jaw_x_sorted}  "
          f"mouth_y({mouth_y_mean:.4f}) > eye_y({eye_y_mean:.4f}) -> mouth_below_eye={mouth_below}")

face_sanity(pts_interleave, "ModeA-interleaved")
face_sanity(pts_split,      "ModeB-split")
