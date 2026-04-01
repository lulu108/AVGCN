#!/usr/bin/env python3
"""
Minimal .npy health-check for DVLOG offenders.

Usage:
  python npy_healthcheck.py --in_csv dvlog_offender_spotcheck_20.csv --out_csv npy_health_report.csv

The input CSV must contain columns: id, v_path, a_path (optional: l_path).
Paths can be absolute or relative; you can pass --base_dir to prefix relative paths.
"""
import argparse, os
from typing import Optional
import numpy as np
import pandas as pd

def safe_load(path: str):
    try:
        arr = np.load(path, allow_pickle=False)
        return arr, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def stats(arr: np.ndarray):
    # Flatten numeric arrays only
    a = arr
    if not np.issubdtype(a.dtype, np.number):
        return {"dtype": str(a.dtype), "shape": tuple(a.shape), "numeric": False}
    x = a.astype(np.float32, copy=False).reshape(-1)
    finite = np.isfinite(x)
    x_f = x[finite]
    n = x.size
    n_f = x_f.size
    z_frac = float(np.mean(x == 0.0)) if n > 0 else 0.0
    if n_f == 0:
        return {
            "dtype": str(a.dtype), "shape": tuple(a.shape), "numeric": True,
            "n": int(n), "finite_frac": float(n_f / max(n,1)), "zero_frac": z_frac,
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "l2": np.nan,
        }
    return {
        "dtype": str(a.dtype), "shape": tuple(a.shape), "numeric": True,
        "n": int(n), "finite_frac": float(n_f / max(n,1)), "zero_frac": z_frac,
        "mean": float(x_f.mean()), "std": float(x_f.std()),
        "min": float(x_f.min()), "max": float(x_f.max()),
        "l2": float(np.linalg.norm(x_f)),
    }

def add_prefix_if_needed(p: str, base_dir: Optional[str]):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    p = str(p)
    if p == "" or p.lower() == "nan":
        return None
    if base_dir and not os.path.isabs(p):
        return os.path.join(base_dir, p)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--base_dir", default=None, help="Prefix for relative paths")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    need_cols = [c for c in ["id","v_path","a_path","l_path","label"] if c in df.columns]
    df = df[need_cols].copy()

    rows = []
    for _, r in df.iterrows():
        rid = r.get("id", None)
        label = r.get("label", None)

        v_path = add_prefix_if_needed(r.get("v_path", None), args.base_dir)
        a_path = add_prefix_if_needed(r.get("a_path", None), args.base_dir)
        l_path = add_prefix_if_needed(r.get("l_path", None), args.base_dir)

        rec = {"id": rid, "label": label, "v_path": v_path, "a_path": a_path, "l_path": l_path}

        for tag, path in [("v", v_path), ("a", a_path), ("l", l_path)]:
            if path is None:
                rec[f"{tag}_load_err"] = ""
                rec[f"{tag}_exists"] = False
                continue
            rec[f"{tag}_exists"] = os.path.exists(path)
            if not rec[f"{tag}_exists"]:
                rec[f"{tag}_load_err"] = "FileNotFound"
                continue
            arr, err = safe_load(path)
            rec[f"{tag}_load_err"] = err
            if arr is None:
                continue
            st = stats(arr)
            for k, v in st.items():
                rec[f"{tag}_{k}"] = v

            # Sequence length guess (first dim) for common (T, D) features
            if hasattr(arr, "shape") and len(arr.shape) >= 1:
                rec[f"{tag}_T"] = int(arr.shape[0])

        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} ({len(out)} rows)")

if __name__ == "__main__":
    main()
