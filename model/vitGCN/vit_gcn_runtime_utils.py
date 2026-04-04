import math
import os
import random
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.optim.lr_scheduler import LambdaLR

class LabelSmoothingCrossEntropy(nn.Module):
    """Weighted label smoothing cross entropy."""

    def __init__(self, smoothing=0.05, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, x, target, sample_weight=None):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss

        if self.weight is not None:
            class_w = self.weight[target]
            loss = loss * class_w

        if sample_weight is not None:
            sw = sample_weight.float()
            denom = sw.sum().clamp(min=1e-6)
            return (loss * sw).sum() / denom

        return loss.mean()


class FocalBCELoss(nn.Module):
    """Focal BCE for 2-class logits."""

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        if pos_weight is not None:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.tensor(float(pos_weight))
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits, target, sample_weight=None):
        logit_bin = logits[:, 1] - logits[:, 0]
        target_f = target.float()

        bce_raw = F.binary_cross_entropy_with_logits(
            logit_bin, target_f, pos_weight=self.pos_weight, reduction="none"
        )

        with torch.no_grad():
            p = torch.sigmoid(logit_bin)
            p_t = p * target_f + (1.0 - p) * (1.0 - target_f)
            focal_w = (1.0 - p_t).pow(self.gamma)

        loss_vec = focal_w * bce_raw

        if sample_weight is None:
            return loss_vec.mean()

        sw = sample_weight.float()
        denom = sw.sum().clamp(min=1e-6)
        return (loss_vec * sw).sum() / denom


class PlainBCELoss(nn.Module):
    """Plain BCEWithLogits for 2-class logits."""

    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.tensor(float(pos_weight))
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits, target, sample_weight=None):
        logit_bin = logits[:, 1] - logits[:, 0]
        target_f = target.float()

        loss_vec = F.binary_cross_entropy_with_logits(
            logit_bin,
            target_f,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if sample_weight is None:
            return loss_vec.mean()

        sw = sample_weight.float()
        denom = sw.sum().clamp(min=1e-6)
        return (loss_vec * sw).sum() / denom


class ModelEMA:
    """Exponential moving average for model parameters."""

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model, backup):
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    @contextmanager
    def average_parameters(self, model):
        backup = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.apply_shadow(model)
        try:
            yield
        finally:
            self.restore(model, backup)


class EarlyStopTracker:
    """EMA-smoothed early stopping helper."""

    def __init__(self, monitor="macro_f1", alpha=0.3, patience=20):
        self.monitor = monitor
        self.alpha = alpha
        self.patience = patience
        self.counter = 0
        if monitor == "macro_f1":
            self.best = float("-inf")
            self.ema = float("-inf")
        else:
            self.best = float("inf")
            self.ema = float("inf")

    def update(self, value):
        if self.monitor == "macro_f1":
            self.ema = value if self.ema == float("-inf") else self.alpha * value + (1.0 - self.alpha) * self.ema
            improved = self.ema > self.best
        else:
            self.ema = value if self.ema == float("inf") else self.alpha * value + (1.0 - self.alpha) * self.ema
            improved = self.ema < self.best

        if improved:
            self.best = self.ema
            self.counter = 0
        else:
            self.counter += 1

        return improved, self.counter >= self.patience


def set_all_seeds(seed, strict_deterministic_algos=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict_deterministic_algos:
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    print(
        f"[set_all_seeds] seed={seed}, cudnn.deterministic=True, "
        f"strict_deterministic_algos={strict_deterministic_algos}"
    )


def compute_class_weights(train_set, num_classes=2, sqrt_inverse=True):
    file_list = train_set.file_list
    cache = train_set._label_cache

    counts = torch.zeros(num_classes)
    for fn in file_list:
        file_root = os.path.splitext(fn)[0]
        lbl = cache.get(file_root)
        if lbl is not None:
            counts[int(lbl)] += 1

    n_total = counts.sum().clamp(min=1)
    if sqrt_inverse:
        raw_w = torch.sqrt(n_total / (num_classes * counts.clamp(min=1)))
    else:
        raw_w = n_total / (num_classes * counts.clamp(min=1))

    raw_w = raw_w / raw_w.mean()
    print(f"[class_weight] counts={counts.tolist()}  weights={raw_w.tolist()}")
    return raw_w


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        phase = min((current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps), 1)
        return 0.5 * (math.cos(phase * math.pi) + 1)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def plot_confusion_matrix(y_true, y_pred, labels_name, savename, title=None, thresh=0.6):
    cm = confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.colorbar()

    if title is not None:
        plt.title(title)

    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, ["Normal", "Depression"])
    plt.yticks(num_local, ["Normal", "Depression"], rotation=90, va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if cm[i][j] * 100 > 0:
                plt.text(
                    j,
                    i,
                    format(cm[i][j] * 100, "0.2f") + "%",
                    ha="center",
                    va="center",
                    color="white" if cm[i][j] > thresh else "black",
                )

    plt.savefig(savename, format="png")
    plt.clf()


def compute_per_class_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return cm, p, r, f1, pm, rm, f1m


def threshold_sweep_macro_f1(y_true, y_prob1, start=0.1, end=0.9, steps=81):
    best_thr, best_f1m = 0.5, -1.0
    for thr in np.linspace(start, end, steps):
        pred = (y_prob1 >= thr).astype(int)
        _, _, f1m, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
        if f1m > best_f1m:
            best_f1m = float(f1m)
            best_thr = float(thr)
    return best_thr, best_f1m


def assert_fusion_batch_schema(batch, stage):
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"[{stage}] fusion batch type must be tuple/list, got {type(batch)}")
    if len(batch) != 7:
        raise RuntimeError(
            f"[{stage}] expected 7 items (video,audio,face_regions,actual_lens,label,quality,sample_weight), got {len(batch)}"
        )


def move_fusion_batch_to_device(batch, device):
    v, a, fr, al, lbl, q, sw = batch
    v = v.to(device)
    a = a.to(device)
    fr = {k: val.to(device) for k, val in fr.items()}
    al = al.to(device)
    lbl = lbl.to(device)
    q = q.to(device)
    sw = sw.to(device)
    return v, a, fr, al, lbl, q, sw


def compute_segment_symptom_score(face_regions_seg, actual_lens_seg, region_scheme):
    any_region = next(iter(face_regions_seg.values()))
    b_loc, l_loc = any_region.shape[0], any_region.shape[1]
    if l_loc <= 1:
        return torch.ones(b_loc, device=any_region.device, dtype=any_region.dtype)

    device = any_region.device
    dtype = any_region.dtype
    t_idx = torch.arange(l_loc, device=device).unsqueeze(0)
    mask = (t_idx < actual_lens_seg.unsqueeze(1)).to(dtype)

    def _masked_mean(x, m, dims):
        denom = m.sum(dim=dims, keepdim=False).clamp(min=1.0)
        return (x * m).sum(dim=dims, keepdim=False) / denom

    def _masked_var(x, m, dims):
        mu = _masked_mean(x, m, dims)
        mu_keep = mu
        for _ in dims:
            mu_keep = mu_keep.unsqueeze(-1)
        denom = m.sum(dim=dims, keepdim=False).clamp(min=1.0)
        return ((x - mu_keep) ** 2 * m).sum(dim=dims, keepdim=False) / denom

    def _speed_stats(x):
        d = x[:, 1:, :, :2] - x[:, :-1, :, :2]
        sp = torch.norm(d, dim=-1)
        m = mask[:, 1:].unsqueeze(-1)
        mean = _masked_mean(sp, m, dims=(1, 2))
        var = _masked_var(sp, m, dims=(1, 2))
        sp_masked = sp.masked_fill(m == 0, float("-inf"))
        peak = sp_masked.amax(dim=(1, 2))
        peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
        return mean, var, peak

    def _get_region_list(scheme, kind):
        if scheme == "symptom7":
            if kind == "global":
                return ["ljaw", "rjaw", "leye", "reye", "brow_glabella", "nose_lower", "mouth"]
            if kind == "head":
                return ["brow_glabella", "nose_lower"]
        if kind == "global":
            return ["ljaw", "rjaw", "leye", "reye", "nose", "mouth"]
        if kind == "head":
            return ["nose"]
        return []

    m_mean, m_var, _ = _speed_stats(face_regions_seg["mouth"])
    mouth_activity = 0.7 * m_mean + 0.3 * m_var

    le_mean, le_var, le_peak = _speed_stats(face_regions_seg["leye"])
    re_mean, re_var, re_peak = _speed_stats(face_regions_seg["reye"])
    left_eye = 0.6 * le_mean + 0.2 * le_var + 0.2 * le_peak
    right_eye = 0.6 * re_mean + 0.2 * re_var + 0.2 * re_peak
    eye_activity = 0.5 * (left_eye + right_eye)

    head_keys = [k for k in _get_region_list(region_scheme, "head") if k in face_regions_seg]
    if len(head_keys) == 0:
        head_keys = list(face_regions_seg.keys())
    head_pts = torch.cat([face_regions_seg[k] for k in head_keys], dim=2)[..., :2]
    head_centroid = head_pts.mean(dim=2)
    d_centroid = head_centroid[:, 1:, :] - head_centroid[:, :-1, :]
    head_sp = torch.norm(d_centroid, dim=-1)
    head_motion = _masked_mean(head_sp, mask[:, 1:], dims=(1,))

    global_keys = [k for k in _get_region_list(region_scheme, "global") if k in face_regions_seg]
    if len(global_keys) == 0:
        global_keys = list(face_regions_seg.keys())
    all_pts = torch.cat([face_regions_seg[k] for k in global_keys], dim=2)[..., :2]
    d_all = all_pts[:, 1:, :, :] - all_pts[:, :-1, :, :]
    g_sp = torch.norm(d_all, dim=-1)
    g_m = mask[:, 1:].unsqueeze(-1)
    global_energy = _masked_mean(g_sp, g_m, dims=(1, 2))

    score = 0.40 * mouth_activity + 0.30 * eye_activity + 0.20 * head_motion + 0.10 * global_energy
    return score.clamp(min=1e-6)


def count(string):
    return sum(1 for char in string if char.isdigit())


def read_lmvd_label_value(label_csv_path):
    """Robustly parse LMVD single-label CSV values under multiple encodings."""
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "gbk", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            df = pd.read_csv(label_csv_path, header=None, encoding=enc)
            if df.shape[0] > 0 and df.shape[1] > 0:
                val = pd.to_numeric(df.iloc[0, 0], errors="coerce")
                if not pd.isna(val):
                    return int(val)
        except Exception as exc:
            last_err = exc

        try:
            df = pd.read_csv(label_csv_path, encoding=enc)
            if len(df.columns) > 0:
                col0 = pd.to_numeric(str(df.columns[0]).strip(), errors="coerce")
                if not pd.isna(col0):
                    return int(col0)
        except Exception as exc:
            last_err = exc

    raise RuntimeError(
        f"Failed to parse LMVD label file: {label_csv_path}. "
        f"Tried encodings={encodings}. Last error: {last_err}"
    )
