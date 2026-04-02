"""Unified DVLOG profile for vitGCN training entrypoints.

Use one switch to select among 4 experiments:
    - fusion
    - av_only_new
    - av_only_legacy
    - gcn_only

Priority:
    1) Environment variable DVLOG_EXPERIMENT (optional override)
    2) FILE_EXPERIMENT below (edit in this file)
"""

import os


# Main switch: edit this value directly in file for daily use.
FILE_EXPERIMENT = "fusion"  #'av_only_new' 'av_only_legacy' 'gcn_only'  'fusion'

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
        "USE_FEATURE_SEQUENCE_ENCODER": True,
    },
    "av_only_legacy": {
        "MODEL_MODE": "av_only",
        "USE_LEGACY_AV_BACKBONE": True,
        "USE_FEATURE_SEQUENCE_ENCODER": False,
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


