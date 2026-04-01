import os
import math
import argparse
import numpy as np
import pandas as pd


def load_npy_safe(path):
    if not isinstance(path, str) or len(path) == 0:
        return None, "EMPTY_PATH"
    if not os.path.exists(path):
        return None, "NOT_EXISTS"
    try:
        arr = np.load(path)
        return arr, ""
    except Exception as e:
        return None, f"LOAD_ERR:{repr(e)}"


def stats_2d(arr):
    # 期望 (T, D)，如果不是就尽量展平到 (T, -1)
    out = {}
    if arr is None:
        out.update(dict(
            exists=False, dtype="", shape="", T="", D="", n="",
            finite_frac="", zero_frac="", mean="", std="", min="", max="", l2="",
            mean_abs_diff="", head20_l2="", tail20_l2="",
            allzero=False, almost_allzero=False, constant_like=False
        ))
        return out

    out["exists"] = True
    out["dtype"] = str(arr.dtype)
    out["shape"] = str(arr.shape)

    x = arr
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    T, D = x.shape[0], x.shape[1]
    out["T"] = int(T)
    out["D"] = int(D)
    out["n"] = int(x.size)

    finite_mask = np.isfinite(x)
    finite_frac = float(finite_mask.mean()) if x.size > 0 else 0.0
    out["finite_frac"] = finite_frac

    # 用 finite 部分做统计，避免 NaN/Inf 污染
    xf = x[finite_mask]
    if xf.size == 0:
        out.update(dict(
            zero_frac=1.0, mean=math.nan, std=math.nan, min=math.nan, max=math.nan, l2=math.nan,
            mean_abs_diff=math.nan, head20_l2=math.nan, tail20_l2=math.nan,
            allzero=False, almost_allzero=False, constant_like=False
        ))
        return out

    zero_frac = float((xf == 0).mean())
    out["zero_frac"] = zero_frac

    out["mean"] = float(xf.mean())
    out["std"] = float(xf.std())
    out["min"] = float(xf.min())
    out["max"] = float(xf.max())
    out["l2"] = float(np.linalg.norm(xf))

    # “变化量”诊断：是否像重复帧/常量
    if T >= 2:
        diff = np.abs(x[1:] - x[:-1])
        out["mean_abs_diff"] = float(np.nanmean(diff))
    else:
        out["mean_abs_diff"] = 0.0

    head = x[: min(20, T)]
    tail = x[max(0, T - 20):]
    out["head20_l2"] = float(np.linalg.norm(head[np.isfinite(head)])) if head.size else 0.0
    out["tail20_l2"] = float(np.linalg.norm(tail[np.isfinite(tail)])) if tail.size else 0.0

    # flags
    out["allzero"] = (out["l2"] == 0.0) or (zero_frac >= 0.999999)
    out["almost_allzero"] = (zero_frac >= 0.999)  # 99.9% 以上为 0
    out["constant_like"] = (out["std"] == 0.0) or (out["mean_abs_diff"] == 0.0)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True, help="offender_aggregated.csv path")
    ap.add_argument("--out_csv", type=str, default="dvlog_offenders_spotcheck.csv")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--v_path_col", type=str, default="v_path")
    ap.add_argument("--a_path_col", type=str, default="a_path")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    rows = []
    for _, r in df.iterrows():
        sid = int(r[args.id_col])
        label = int(r[args.label_col]) if args.label_col in df.columns else -1
        v_path = str(r.get(args.v_path_col, ""))
        a_path = str(r.get(args.a_path_col, ""))

        v, v_err = load_npy_safe(v_path)
        a, a_err = load_npy_safe(a_path)

        vs = stats_2d(v)
        as_ = stats_2d(a)

        T_match = (vs["T"] != "" and as_["T"] != "" and vs["T"] == as_["T"])
        T_diff = (abs(vs["T"] - as_["T"]) if (isinstance(vs["T"], int) and isinstance(as_["T"], int)) else "")

        # 音频尺度是否可疑（这里给个非常粗的启发式：均值过大/方差过大）
        a_scale_suspect = False
        if isinstance(as_.get("mean", None), float) and isinstance(as_.get("std", None), float):
            if abs(as_["mean"]) > 50 or as_["std"] > 200:  # 你的 DVLOG a_mean~320, a_std~680 会直接触发
                a_scale_suspect = True

        rows.append({
            "id": sid, "label": label,
            "v_path": v_path, "a_path": a_path,
            "v_load_err": v_err, "a_load_err": a_err,

            "v_exists": vs["exists"], "v_dtype": vs["dtype"], "v_shape": vs["shape"],
            "v_T": vs["T"], "v_D": vs["D"], "v_n": vs["n"],
            "v_finite_frac": vs["finite_frac"], "v_mean": vs["mean"], "v_std": vs["std"],
            "v_min": vs["min"], "v_max": vs["max"], "v_l2": vs["l2"], "v_zero_frac": vs["zero_frac"],
            "v_mean_abs_diff": vs["mean_abs_diff"], "v_head20_l2": vs["head20_l2"], "v_tail20_l2": vs["tail20_l2"],
            "v_allzero": vs["allzero"], "v_almost_allzero": vs["almost_allzero"], "v_constant_like": vs["constant_like"],

            "a_exists": as_["exists"], "a_dtype": as_["dtype"], "a_shape": as_["shape"],
            "a_T": as_["T"], "a_D": as_["D"], "a_n": as_["n"],
            "a_finite_frac": as_["finite_frac"], "a_mean": as_["mean"], "a_std": as_["std"],
            "a_min": as_["min"], "a_max": as_["max"], "a_l2": as_["l2"], "a_zero_frac": as_["zero_frac"],
            "a_mean_abs_diff": as_["mean_abs_diff"], "a_head20_l2": as_["head20_l2"], "a_tail20_l2": as_["tail20_l2"],

            "T_match": T_match, "T_diff": T_diff,
            "a_scale_suspect": a_scale_suspect,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote: {args.out_csv}  rows={len(out)}")


if __name__ == "__main__":
    main()