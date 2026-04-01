import os
import numpy as np
import pandas as pd
from pathlib import Path

DVLOG_ROOT = r"data/dvlog-dataset/dvlog-dataset"
LABEL_CSV  = r"data/dvlog-dataset/dvlog-dataset/labels.csv"

PATCH_SIZE = 15
PCTL = [50, 75, 80, 85, 90, 95, 99]

def round_up_to_multiple(x: int, m: int) -> int:
    return int(((x + m - 1) // m) * m)

def parse_dvlog_label(val):
    if isinstance(val, str):
        return 1 if val.strip().lower() == 'depression' else 0
    return int(val)

def load_T_for_index(root: str, idx: str) -> int:
    # DVLOG: {index}/{index}_visual.npy
    visual_path = Path(root) / idx / f"{idx}_visual.npy"
    if not visual_path.exists():
        raise FileNotFoundError(f"Missing: {visual_path}")
    arr = np.load(str(visual_path), mmap_mode="r")  # 省内存
    return int(arr.shape[0])

def recommend_T(actual_lens, patch_size=15):
    p80 = int(np.percentile(actual_lens, 80))
    p90 = int(np.percentile(actual_lens, 90))
    T_p80 = round_up_to_multiple(p80, patch_size)
    T_p90 = round_up_to_multiple(p90, patch_size)
    min_T = patch_size * 30  # 至少 30 patches
    return {
        "p80": p80, "p90": p90,
        "T_by_p80": max(T_p80, min_T),
        "T_by_p90": max(T_p90, min_T),
        "min_T": min_T
    }

def main():
    df = pd.read_csv(LABEL_CSV)
    fold = df["fold"].astype(str).str.strip().str.lower()
    trainval = df[fold.isin(["train", "valid"])].copy()

    idxs = [str(x) for x in trainval["index"].values]
    labels = np.array([parse_dvlog_label(v) for v in trainval["label"].values], dtype=np.int32)

    lens = []
    for idx in idxs:
        lens.append(load_T_for_index(DVLOG_ROOT, idx))
    lens = np.array(lens, dtype=np.int32)

    print(f"[DVLOG] N(train+valid)={len(lens)}")
    print("[DVLOG] percentiles:", {p: int(np.percentile(lens, p)) for p in PCTL})
    print("[DVLOG] min/max:", int(lens.min()), int(lens.max()))
    print("[DVLOG] mean/std:", float(lens.mean()), float(lens.std()))

    for thr in [150, 200, 300, 450, 600, 750, 915]:
        print(f"[DVLOG] ratio(T < {thr})={(lens < thr).mean():.3f}")

    for c in [0, 1]:
        lc = lens[labels == c]
        print(f"\n[DVLOG] class={c}, N={len(lc)}")
        print("  percentiles:", {p: int(np.percentile(lc, p)) for p in [50, 80, 90, 95]})
        print("  mean/std:", float(lc.mean()), float(lc.std()))

    rec = recommend_T(lens, PATCH_SIZE)
    print("\n[Recommend]")
    print(f"  P80={rec['p80']} -> T_target={rec['T_by_p80']} (x{PATCH_SIZE})")
    print(f"  P90={rec['p90']} -> T_target={rec['T_by_p90']} (more conservative)")
    print(f"  min_T={rec['min_T']} (>=30 patches)")

if __name__ == "__main__":
    main()
