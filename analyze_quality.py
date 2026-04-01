import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch

from kfoldLoader_multimodal import MultiModalDataLoader


def make_splits_dvlog(labels_csv):
    """DVLOG: labels.csv 需要包含 fold/index/label 三列"""
    df = pd.read_csv(labels_csv)
    for col in ["fold", "index", "label"]:
        if col not in df.columns:
            raise ValueError(f"DVLOG labels.csv missing column: {col}")

    fold = df["fold"].astype(str).str.strip().str.lower()
    train = df[fold == "train"]["index"].astype(str).tolist()
    dev   = df[fold.isin(["valid", "val", "dev"])]["index"].astype(str).tolist()
    test  = df[fold == "test"]["index"].astype(str).tolist()
    return train, dev, test


def make_list_lmvd(video_path):
    """
    LMVD: 没有 labels.csv。最简单是扫 video_path 下所有 .npy 作为样本列表。
    MultiModalDataLoader(LMVD) 期望 file_list 形如 ['001.npy', '002.npy', ...]
    """
    files = sorted([os.path.basename(x) for x in glob.glob(os.path.join(video_path, "*.npy"))])
    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files found under: {video_path}")
    return files


def collect_quality(file_list, split_name, dataset, video_path, audio_path, face_path, label_path, t_target):
    ds = MultiModalDataLoader(
        file_list, video_path, audio_path, face_path, label_path,
        T_target=t_target, mode="test", dataset=dataset,
        dvlog_aug_noise_std=0.0, dvlog_audio_mask_ratio=0.0
    )

    rows = []
    for i in range(len(ds)):
        # ===== 1) 读 raw visual 的 orig_T（这一步是关键：衡量 padding）=====
        idx_str = str(file_list[i])
        # DVLOG 文件路径规则见 loader：{root}/{idx}/{idx}_visual.npy 
        visual_file = os.path.join(video_path, idx_str, f"{idx_str}_visual.npy")
        v = np.load(visual_file)
        orig_T = int(v.shape[0])
        eff_T = min(orig_T, t_target)
        pad_frac = float(max(0, t_target - eff_T) / t_target)

        # ===== 2) 走 loader 得到 face_regions，并从 validity 通道读“pad 后有效度”=====
        item = ds[i]
        _, _, face_regions, actual_len, label, quality = item[:6]
        y = int(label.item()) if torch.is_tensor(label) else int(label)

        # 按 GCN 的逻辑：取任一 region 的最后一维 validity（shape[-1] >= 11）:contentReference[oaicite:8]{index=8}
        any_region = face_regions.get("mouth", next(iter(face_regions.values())))
        # any_region: (T, N, C)
        if any_region.shape[-1] >= 11:
            face_valid_ratio = any_region[:, 0, -1].astype(np.float32)  # (T,)
            face_valid_mean = float(face_valid_ratio.mean())
            face_valid_low02 = float(np.mean(face_valid_ratio < 0.2))
        else:
            face_valid_mean = 1.0
            face_valid_low02 = 0.0

        rows.append({
            "split": split_name,
            "idx": idx_str,
            "y": y,
            "orig_T": orig_T,
            "pad_frac": pad_frac,
            "actual_len": int(actual_len.item()) if torch.is_tensor(actual_len) else int(actual_len),
            "face_valid_pad_mean": face_valid_mean,
            "face_valid_pad_low02": face_valid_low02,
        })
    return pd.DataFrame(rows)

def summarize(df):
    def frac(x):
        return float(np.mean(x))

    # 兼容两套列名：
    # 旧版: q_v, q_g, hard_missing
    # 当前: pad_frac, face_valid_pad_mean, face_valid_pad_low02
    qv_col = "q_v" if "q_v" in df.columns else ("pad_frac" if "pad_frac" in df.columns else None)
    qg_col = "q_g" if "q_g" in df.columns else ("face_valid_pad_mean" if "face_valid_pad_mean" in df.columns else None)
    hm_col = "hard_missing" if "hard_missing" in df.columns else ("face_valid_pad_low02" if "face_valid_pad_low02" in df.columns else None)

    if qv_col is None or qg_col is None or hm_col is None:
        missing = [c for c, v in [("q_v/pad_frac", qv_col), ("q_g/face_valid_pad_mean", qg_col), ("hard_missing/face_valid_pad_low02", hm_col)] if v is None]
        raise KeyError(f"Required quality columns not found: {missing}; existing columns={list(df.columns)}")

    g = df.groupby(["split", "y"], dropna=False)
    out = g.agg(
        n=("y", "size"),
        qv_mean=(qv_col, "mean"),
        qv_zero=(qv_col, lambda s: frac(s < 0.5)),
        qg_mean=(qg_col, "mean"),
        qg_p10=(qg_col, lambda s: float(s.quantile(0.10))),
        qg_p05=(qg_col, lambda s: float(s.quantile(0.05))),
        qg_low02=(qg_col, lambda s: frac(s < 0.2)),
        hard_missing_rate=(hm_col, "mean"),
    ).reset_index()

    # 记录本次实际使用的列名，便于审计
    out["qv_col"] = qv_col
    out["qg_col"] = qg_col
    out["hard_missing_col"] = hm_col
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["DVLOG", "LMVD"])
    ap.add_argument("--label_csv", default=None, help="DVLOG only: path to labels.csv")
    ap.add_argument("--label_dir", default=None, help="LMVD only: label directory (contains *_Depression.csv)")
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--audio_path", required=True)
    ap.add_argument("--face_path",  required=True)
    ap.add_argument("--t_target", type=int, default=596)
    ap.add_argument("--out_dir", default="quality_audit_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "DVLOG":
        if args.label_csv is None:
            raise ValueError("DVLOG requires --label_csv")
        if not os.path.exists(args.label_csv):
            raise FileNotFoundError(f"label_csv not found: {args.label_csv}")
        X_train, X_dev, X_test = make_splits_dvlog(args.label_csv)
        label_path = args.label_csv  # MultiModalDataLoader DVLOG 用 label_path=labels.csv :contentReference[oaicite:4]{index=4}

    else:  # LMVD
        if args.label_dir is None:
            raise ValueError("LMVD requires --label_dir (folder label/label)")
        if not os.path.isdir(args.label_dir):
            raise FileNotFoundError(f"label_dir not found: {args.label_dir}")
        files = make_list_lmvd(args.video_path)
        # LMVD 没 fold 列：先做全量审计（split=all）
        X_train, X_dev, X_test = files, [], []
        label_path = args.label_dir  # MultiModalDataLoader LMVD 用目录缓存 *_Depression.csv :contentReference[oaicite:5]{index=5}

    df_parts = []
    df_parts.append(collect_quality(X_train, "train" if args.dataset=="DVLOG" else "all",
                                    args.dataset, args.video_path, args.audio_path, args.face_path, label_path, args.t_target))
    if len(X_dev) > 0:
        df_parts.append(collect_quality(X_dev, "dev", args.dataset, args.video_path, args.audio_path, args.face_path, label_path, args.t_target))
    if len(X_test) > 0:
        df_parts.append(collect_quality(X_test, "test", args.dataset, args.video_path, args.audio_path, args.face_path, label_path, args.t_target))

    df_all = pd.concat(df_parts, ignore_index=True)

    df_all.to_csv(os.path.join(args.out_dir, "quality_detail.csv"), index=False)
    summary = summarize(df_all)
    summary.to_csv(os.path.join(args.out_dir, "quality_summary.csv"), index=False)

    # worst 排序同样做列名兼容
    qv_col = "q_v" if "q_v" in df_all.columns else ("pad_frac" if "pad_frac" in df_all.columns else None)
    qg_col = "q_g" if "q_g" in df_all.columns else ("face_valid_pad_mean" if "face_valid_pad_mean" in df_all.columns else None)
    if qv_col is not None and qg_col is not None:
        worst = df_all.sort_values([qg_col, qv_col], ascending=True).head(50)
    elif qg_col is not None:
        worst = df_all.sort_values([qg_col], ascending=True).head(50)
    elif qv_col is not None:
        worst = df_all.sort_values([qv_col], ascending=True).head(50)
    else:
        worst = df_all.head(50)
    worst.to_csv(os.path.join(args.out_dir, "worst_50_by_qg.csv"), index=False)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    print("\n==== quality_summary.csv ====")
    print(summary)
    print(f"\nSaved to: {args.out_dir}/quality_summary.csv, quality_detail.csv, worst_50_by_qg.csv")


if __name__ == "__main__":
    main()