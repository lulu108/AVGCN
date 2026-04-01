import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


LEGACY6_REGIONS: Dict[str, List[int]] = {
    "ljaw": list(range(0, 9)),
    "rjaw": list(range(8, 17)),
    "leye": list(range(17, 22)) + list(range(36, 42)),
    "reye": list(range(22, 27)) + list(range(42, 48)),
    "nose": list(range(27, 36)),
    "mouth": list(range(48, 68)),
}

LEGACY6_COLORS = {
    "ljaw": "#4C78A8",
    "rjaw": "#72B7B2",
    "leye": "#F58518",
    "reye": "#E45756",
    "nose": "#54A24B",
    "mouth": "#FF9DA6",
}

SYMPTOM7_REGIONS: Dict[str, List[int]] = {
    "ljaw": list(range(0, 9)),
    "rjaw": list(range(8, 17)),
    "leye": list(range(36, 42)),
    "reye": list(range(42, 48)),
    "brow_glabella": list(range(17, 27)) + list(range(27, 31)),
    "nose_lower": list(range(31, 36)),
    "mouth": list(range(48, 68)),
}

REGION_COLORS = {
    "ljaw": "#4C78A8",
    "rjaw": "#72B7B2",
    "leye": "#F58518",
    "reye": "#E45756",
    "brow_glabella": "#54A24B",
    "nose_lower": "#B279A2",
    "mouth": "#FF9DA6",
}

REGION_MAPS = {
    "legacy6": LEGACY6_REGIONS,
    "symptom7": SYMPTOM7_REGIONS,
}

COLOR_MAPS = {
    "legacy6": LEGACY6_COLORS,
    "symptom7": REGION_COLORS,
}


def _connect(ax, pts: np.ndarray, idx: List[int], color: str) -> None:
    xy = pts[idx]
    ax.plot(xy[:, 0], -xy[:, 1], color=color, linewidth=1.2, alpha=0.9)


def _plot_partition_on_axis(
    ax,
    pts68x2: np.ndarray,
    scheme: str,
    title: str,
    show_index: bool = True,
) -> None:
    pts = np.asarray(pts68x2, dtype=np.float32)
    if pts.shape != (68, 2):
        raise ValueError(f"Expected (68,2), got {pts.shape}")
    if scheme not in REGION_MAPS:
        raise ValueError(f"Unknown scheme: {scheme}")

    region_map = REGION_MAPS[scheme]
    color_map = COLOR_MAPS[scheme]

    for region, idx in region_map.items():
        color = color_map[region]
        region_pts = pts[idx]
        ax.scatter(region_pts[:, 0], -region_pts[:, 1], s=26, c=color, label=region, alpha=0.95)
        _connect(ax, pts, idx, color)

        if show_index:
            for i in idx:
                ax.text(pts[i, 0], -pts[i, 1], str(i), fontsize=6, color=color)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def plot_partition_landmarks(
    pts68x2: np.ndarray,
    title: str,
    save_path: str,
    scheme: str,
    show_index: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_partition_on_axis(ax, pts68x2, scheme=scheme, title=title, show_index=show_index)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_partition_comparison(
    pts68x2: np.ndarray,
    dataset_name: str,
    frame_idx: int,
    save_path: str,
    show_index: bool = True,
) -> None:
    pts = np.asarray(pts68x2, dtype=np.float32)
    if pts.shape != (68, 2):
        raise ValueError(f"Expected (68,2), got {pts.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    _plot_partition_on_axis(
        axes[0], pts, scheme="legacy6",
        title=f"{dataset_name} legacy6 | frame={frame_idx}",
        show_index=show_index,
    )
    _plot_partition_on_axis(
        axes[1], pts, scheme="symptom7",
        title=f"{dataset_name} symptom7 | frame={frame_idx}",
        show_index=show_index,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_symptom7_landmarks(
    pts68x2: np.ndarray,
    title: str,
    save_path: str,
    show_index: bool = True,
) -> None:
    """Backward-compatible wrapper for symptom7-only plotting."""
    plot_partition_landmarks(
        pts68x2,
        title=title,
        save_path=save_path,
        scheme="symptom7",
        show_index=show_index,
    )
