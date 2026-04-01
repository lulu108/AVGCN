import argparse
import os
import math
import numpy as np
import pandas as pd


def safe_load_npy(path: str):
    """Return (arr, err_msg)."""
    if path is None or (isinstance(path, float) and math.isnan(path)) or str(path).strip() == "":
        return None, "EMPTY_PATH"
    path = str(path)
    if not os.path.exists(path):
        return None, "NOT_EXISTS"
    try:
        arr = np.load(path, allow_pickle=False)
        return arr, ""
    except Exception as e:
        return None, f"LOAD_ERR:{type(e).__name__}:{e}"


def array_stats(x: np.ndarray, name: str):
    """
    Stats designed for DVLOG offenders:
      - numeric sanity: finite_frac, zero_frac
      - scale: mean/std/min/max/l2
      - dynamics: mean_abs_diff (detect repeated/constant sequence)
      - head/tail energy: l2 of first/last 20 frames
    """
    out = {}

    if x is None:
        # fill NA-ish
        for k in [
            "exists", "load_err", "dtype", "shape", "T", "D",
            "n", "finite_frac", "zero_frac",
            "mean", "std", "min", "max", "l2",
            "mean_abs_diff", "head20_l2", "tail20_l2"
        ]:
            out[f"{name}_{k}"] = None
        return out

    out[f"{name}_exists"] = True
    out[f"{name}_load_err"] = ""
    out[f"{name}_dtype"] = str(x.dtype)
    out[f"{name}_shape"] = str(tuple(x.shape))

    # flatten numeric
    x_num = x.astype(np.float32, copy=False) if np.issubdtype(x.dtype, np.number) else None
    if x_num is None:
        out[f"{name}_n"] = None
        out[f"{name}_finite_frac"] = None
        out[f"{name}_zero_frac"] = None
        out[f"{name}_mean"] = None
        out[f"{name}_std"] = None
        out[f"{name}_min"] = None
        out[f"{name}_max"] = None
        out[f"{name}_l2"] = None
        out[f"{name}_T"] = x.shape[0] if len(x.shape) >= 1 else None
        out[f"{name}_D"] = x.shape[1] if len(x.shape) >= 2 else None
        out[f"{name}_mean_abs_diff"] = None
        out[f"{name}_head20_l2"] = None
        out[f"{name}_tail20_l2"] = None
        return out

    out[f"{name}_T"] = int(x.shape[0]) if len(x.shape) >= 1 else None
    out[f"{name}_D"] = int(x.shape[1]) if len(x.shape) >= 2 else None

    flat = x_num.reshape(-1)
    out[f"{name}_n"] = int(flat.size)

    finite = np.isfinite(flat)
    finite_frac = float(finite.mean()) if flat.size > 0 else 0.0
    out[f"{name}_finite_frac"] = finite_frac

    if finite_frac > 0:
        flat_f = flat[finite]
        out[f"{name}_mean"] = float(flat_f.mean())
        out[f"{name}_std"] = float(flat_f.std())
        out[f"{name}_min"] = float(flat_f.min())
        out[f"{name}_max"] = float(flat_f.max())
        out[f"{name}_l2"] = float(np.linalg.norm(flat_f))
        out[f"{name}_zero_frac"] = float((flat_f == 0).mean())
    else:
        out[f"{name}_mean"] = None
        out[f"{name}_std"] = None
        out[f"{name}_min"] = None
        out[f"{name}_max"] = None
        out[f"{name}_l2"] = None
        out[f"{name}_zero_frac"] = None

    # dynamics: mean absolute frame-to-frame difference (works for (T,D) arrays)
    mean_abs_diff = None
    head20_l2 = None
    tail20_l2 = None
    if len(x.shape) >= 2 and x.shape[0] >= 2 and np.issubdtype(x.dtype, np.number):
        x2 = x_num
        # If there are NaNs, mask them by finite
        # simple: replace non-finite with 0 so diff doesn't blow up
        x2 = np.where(np.isfinite(x2), x2, 0.0).astype(np.float32)

        diff = np.abs(x2[1:] - x2[:-1])
        mean_abs_diff = float(diff.mean())

        k = min(20, x2.shape[0])
        head20_l2 = float(np.linalg.norm(x2[:k].reshape(-1)))
        tail20_l2 = float(np.linalg.norm(x2[-k:].reshape(-1)))

    out[f"{name}_mean_abs_diff"] = mean_abs_diff
    out[f"{name}_head20_l2"] = head20_l2
    out[f"{name}_tail20_l2"] = tail20_l2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offender_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="offender_npy_spotcheck.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.offender_csv)

    # expected columns: id, label, v_path, a_path (your file has them)
    needed = ["id", "label", "v_path", "a_path"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in offender_csv. Got columns: {list(df.columns)}")

    rows = []
    for _, r in df.iterrows():
        sid = int(r["id"])
        label = int(r["label"])
        v_path = str(r["v_path"])
        a_path = str(r["a_path"])

        v, v_err = safe_load_npy(v_path)
        a, a_err = safe_load_npy(a_path)

        rec = {
            "id": sid,
            "label": label,
            "v_path": v_path,
            "a_path": a_path,
            "v_load_err": v_err if v_err else "",
            "a_load_err": a_err if a_err else "",
        }

        # stats
        if v is None:
            rec["v_exists"] = False
        if a is None:
            rec["a_exists"] = False

        rec.update(array_stats(v, "v"))
        rec.update(array_stats(a, "a"))

        # alignment / mismatch
        vT = rec.get("v_T")
        aT = rec.get("a_T")
        rec["T_match"] = (vT is not None and aT is not None and int(vT) == int(aT))
        rec["T_diff"] = (int(vT) - int(aT)) if (vT is not None and aT is not None) else None

        # quick flags
        # missing visual: full-zero OR almost full-zero
        v_zero = rec.get("v_zero_frac")
        v_l2 = rec.get("v_l2")
        rec["v_allzero"] = (v_l2 == 0.0) if v_l2 is not None else None
        rec["v_almost_allzero"] = (v_zero is not None and v_zero > 0.999)

        # repeated/constant: mean_abs_diff tiny (tune threshold after you see distribution)
        v_mad = rec.get("v_mean_abs_diff")
        rec["v_constant_like"] = (v_mad is not None and v_mad < 1e-4)

        # audio scale outlier heuristics
        a_max = rec.get("a_max")
        a_std = rec.get("a_std")
        rec["a_scale_suspect"] = (a_max is not None and a_max > 2000) or (a_std is not None and a_std > 1000)

        rows.append(rec)

    out = pd.DataFrame(rows)

    # Sort: first show the worst potential data issues
    out = out.sort_values(
        by=["v_allzero", "v_almost_allzero", "v_constant_like", "a_scale_suspect", "id"],
        ascending=[False, False, False, False, True]
    )

    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote: {args.out_csv}  (rows={len(out)})")


if __name__ == "__main__":
    main()