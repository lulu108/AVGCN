"""LMVD profile for vitGCN training entrypoints."""

import os


FILE_EXPERIMENT = "fusion"  # 'fusion' | 'av_only' | 'video_only' | 'audio_only' | 'gcn_only'
EXPERIMENT = os.getenv("LMVD_EXPERIMENT", FILE_EXPERIMENT).strip().lower()

_MODE_OVERRIDES = {
    "fusion": {
        "MODEL_MODE": "fusion",
    },
    "av_only": {
        "MODEL_MODE": "av_only",
    },
    "video_only": {
        "MODEL_MODE": "video_only",
    },
    "audio_only": {
        "MODEL_MODE": "audio_only",
    },
    "gcn_only": {
        "MODEL_MODE": "gcn_only",
    },
}

if EXPERIMENT not in _MODE_OVERRIDES:
    valid = ", ".join(sorted(_MODE_OVERRIDES.keys()))
    raise ValueError(f"Invalid LMVD_EXPERIMENT={EXPERIMENT!r}. Valid values: {valid}")

OVERRIDES = {
    "DATASET_SELECT": "LMVD",
    "FUSION_MODE": "late",
    "USE_LEGACY_AV_BACKBONE": False,
    "USE_FEATURE_SEQUENCE_ENCODER": False,
    "USE_SEGMENT_MIL": False,
    "USE_SLIDING_SEGMENT_EVAL": False,
}

OVERRIDES.update(_MODE_OVERRIDES[EXPERIMENT])

