"""Unified DVLOG profile for vitGCN training entrypoints.

Use one switch to select among 4 experiments:
    - fusion
    - av_only_new
    - av_only_legacy
    - video_only
    - audio_only
    - gcn_only

Priority:
    1) Environment variable DVLOG_EXPERIMENT (optional override)
    2) FILE_EXPERIMENT below (edit in this file)
"""

import os


# Main switch: edit this value directly in file for daily use.
FILE_EXPERIMENT = "av_only_new"  # 'fusion' | 'av_only_new' | 'av_only_legacy' | 'video_only' | 'audio_only' | 'gcn_only'

# Optional override (e.g., train_dvlog.py --exp ... sets env var).
EXPERIMENT = os.getenv("DVLOG_EXPERIMENT", FILE_EXPERIMENT).strip().lower()

_BASE = {
    "DATASET_SELECT": "DVLOG",
    "FUSION_MODE": "concat",
    "USE_LEGACY_AV_BACKBONE": False,
    "USE_FEATURE_SEQUENCE_ENCODER": True,
    "FS_VIDEO_LOCAL_BLOCKS": 2,
    "FS_AUDIO_LOCAL_BLOCKS": 3,
    "FS_DILATED_AUDIO": True,
    "FS_FUSION_DROPOUT": 0.05,
    "SINGLE_MODALITY_CLEAN_PATH": True,
    "USE_STRONG_AUDIO_ENCODER": True,
    "USE_STRONG_VIDEO_ENCODER": True,
    "AUDIO_FIXED_LEN": 128,
    "VIDEO_FIXED_LEN": 128,
    "VIDEO_USE_DELTA": True,
    "AUDIO_STRONG_STEM_KERNEL": 3,
    "VIDEO_STRONG_STEM_KERNEL": 5,
    "USE_SEGMENT_MIL": False,
    "USE_SLIDING_SEGMENT_EVAL": True,
}

_MODE_OVERRIDES = {
    "fusion": {
        "MODEL_MODE": "fusion",
    },
    "av_only_new": {
        "MODEL_MODE": "av_only",
        "USE_LEGACY_AV_BACKBONE": False,
        "USE_FEATURE_SEQUENCE_ENCODER": False,
    },
    "av_only_legacy": {
        "MODEL_MODE": "av_only",
        "USE_LEGACY_AV_BACKBONE": True,
        "USE_FEATURE_SEQUENCE_ENCODER": False,
    },
    "video_only": {
        "MODEL_MODE": "video_only",
        "USE_LEGACY_AV_BACKBONE": False,
        "USE_FEATURE_SEQUENCE_ENCODER": True,
        "SINGLE_MODALITY_CLEAN_PATH": True,
        "USE_STRONG_AUDIO_ENCODER": False,
        "USE_STRONG_VIDEO_ENCODER": True,
        "VIDEO_USE_DELTA": True,
    },
    "audio_only": {
        "MODEL_MODE": "audio_only",
        "USE_LEGACY_AV_BACKBONE": False,
        "USE_FEATURE_SEQUENCE_ENCODER": True,
        "SINGLE_MODALITY_CLEAN_PATH": True,
        "USE_STRONG_AUDIO_ENCODER": True,
        "USE_STRONG_VIDEO_ENCODER": False,
    },
    "gcn_only": {
        "MODEL_MODE": "gcn_only",
    },
}

if EXPERIMENT not in _MODE_OVERRIDES:
    valid = ", ".join(sorted(_MODE_OVERRIDES.keys()))
    raise ValueError(f"Invalid DVLOG_EXPERIMENT={EXPERIMENT!r}. Valid values: {valid}")

OVERRIDES = dict(_BASE)
OVERRIDES.update(_MODE_OVERRIDES[EXPERIMENT])


