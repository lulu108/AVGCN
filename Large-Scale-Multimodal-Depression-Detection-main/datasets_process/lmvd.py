#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""
#----------------------------------------------------------------
# imports
#----------------------------------------------------------------
from pathlib import Path
from typing import Union, Optional
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random


def _first_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _first_existing_dir(candidates):
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return candidates[0]


def _resolve_lmvd_root(root: Path) -> Path:
    """Allow both {root} and {root}/lmvd-dataset layouts."""
    candidates = [root, root / "lmvd-dataset"]
    for candidate in candidates:
        if (candidate / "lmvd_labels.csv").exists():
            return candidate
    return root


def _to_2d_feature(x: np.ndarray) -> np.ndarray:
    """Flatten non-time dimensions so feature shape becomes [T, F]."""
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)


def normalize_data(x):
    """
    Normalize the input data array x to zero mean and unit variance.
    Args:
        x (np.ndarray)  : Input data.
    Returns:
        np.ndarray      : Normalized data.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.clip(std, a_min=1e-8, a_max=None)  # Avoid division by zero
    return (x - mean) / std


class LMVD(data.Dataset):
    def __init__(
        self, root: Union[str, Path], fold: str="train",
        gender: str="both", transform=None, target_transform=None, aug=False
    ):
        raw_root = root if isinstance(root, Path) else Path(root)
        self.root = _resolve_lmvd_root(raw_root)
        self.fold = fold
        self.gender = gender
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug

        self.features = []
        self.labels = []

        labels_path = self.root / "lmvd_labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"LMVD labels file not found: {labels_path}. "
                "Please create lmvd_labels.csv first."
            )

        audio_dir = _first_existing_dir(
            [
                self.root / "audio",
                self.root / "Audio_feature",
                self.root / "LMVD_Feature" / "Audio_feature",
            ]
        )
        visual_dir = _first_existing_dir(
            [
                self.root / "visual",
                self.root / "Video_landmarks_npy",
                self.root / "LMVD_Feature" / "Video_landmarks_npy",
            ]
        )

        with open(labels_path, "r") as f:
            for line in f:
                sample = line.strip().split(",")
                if self.is_sample(sample):
                    s_id = sample[0]
                    if s_id == "index":
                        continue

                    s_label = int(sample[1] == "depression")  # depression = 1, normal = 0

                    v_feature_path = _first_existing_path(
                        [
                            visual_dir / f"{s_id}_visual.npy",
                            visual_dir / f"{s_id}.npy",
                        ]
                    )
                    a_feature_path = _first_existing_path(
                        [
                            audio_dir / f"{s_id}.npy",
                        ]
                    )

                    if not v_feature_path.exists() or not a_feature_path.exists():
                        continue

                    self.labels.append(s_label)
                    v_feature = np.load(v_feature_path)
                    a_feature = np.load(a_feature_path)

                    v_feature = _to_2d_feature(v_feature)
                    a_feature = _to_2d_feature(a_feature)

                    # concat visual and acoustic features along the 2nd axis
                    T_v, T_a = v_feature.shape[0], a_feature.shape[0]
                    if T_v == T_a:
                        feature = np.concatenate(
                            (v_feature, a_feature), axis=1
                        ).astype(np.float32)
                    else:
                        T = min(T_v, T_a)
                        feature = np.concatenate(
                            (v_feature[:T], a_feature[:T]), axis=1
                        ).astype(np.float32)
                    # feature = normalize_data(feature)  # Normalize feature
                    self.features.append(feature)

                    if self.aug and self.fold == 'train':
                        t_length = feature.shape[0]
                        for i in range(5):
                            f_length = int(random.random() * t_length)
                            if f_length < 400:
                                continue
                            t_start = random.randint(1, t_length - f_length)
                            self.labels.append(s_label)
                            self.features.append(feature[t_start:t_start + f_length, :])

    def is_sample(self, sample) -> bool:
        if len(sample) < 3:
            return False
        fold = sample[2]
        return fold == self.fold

    def __getitem__(self, i: int):
        feature = self.features[i]
        label = self.labels[i]
        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return feature, label

    def __len__(self):
        return len(self.labels)


def _collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(
        [torch.from_numpy(f) for f in features], batch_first=True
    )
    padding_mask = (padded_features.sum(dim=-1) != 0).long()
    labels = torch.tensor(labels)
    return padded_features, labels, padding_mask


def get_lmvd_dataloader(
    root: Union[str, Path], fold: str="train", batch_size: int=8,
    gender: str="both",
    transform=None, target_transform=None, aug=True
):
    """Get dataloader for LMVD dataset.

    Args:
        root (Union[str, Path]): path to the LMVD dataset. Should be something
            like `*/LMVD-dataset`.
        fold (str, optional): train / valid / test. Defaults to "train".
        batch_size (int, optional): Defaults to 8.
        gender (str, optional): m / f / both. Defaults to both.
        transform (optional): Defaults to None.
        target_transform (optional): Defaults to None.

    Returns:
        the dataloader.
    """
    dataset = LMVD(root, fold, gender, transform, target_transform, aug)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=_collate_fn,
        shuffle=(fold == "train"),
    )
    return dataloader


if __name__ == '__main__':
    train_loader = get_lmvd_dataloader(
        "../data/lmvd-dataset", "train"
    )
    print(f"train_loader: {len(train_loader.dataset)} samples")
    valid_loader = get_lmvd_dataloader(
        "../data/lmvd-dataset", "valid"
    )
    print(f"valid_loader: {len(valid_loader.dataset)} samples")
    test_loader = get_lmvd_dataloader(
        "../data/lmvd-dataset", "test"
    )
    print(f"test_loader: {len(test_loader.dataset)} samples")

    b1 = next(iter(train_loader))[0]
    print(f"A train_loader batch: shape={b1.shape}, dtype={b1.dtype}")
    b2 = next(iter(valid_loader))[0]
    print(f"A valid_loader batch: shape={b2.shape}, dtype={b2.dtype}")
    b3 = next(iter(test_loader))[0]
    print(f"A test_loader  batch: shape={b3.shape}, dtype={b3.dtype}")

    '''
    ALL: 1934, Positive: 928, Negative: 1006
        -- train_loader: 1934 samples
    ALL: 181, Positive: 90, Negative: 91
        -- valid_loader: 181 samples
    ALL: 367, Positive: 183, Negative: 184
        -- test_loader: 367 samples

        A train_loader batch: shape=torch.Size([8, 591, 264]), dtype=torch.float32
        A valid_loader batch: shape=torch.Size([8, 491, 264]), dtype=torch.float32
        A test_loader  batch: shape=torch.Size([8, 828, 264]), dtype=torch.float32
    '''
