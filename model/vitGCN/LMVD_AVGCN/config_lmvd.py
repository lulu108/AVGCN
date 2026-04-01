"""LMVD profile for vitGCN training entrypoints."""

OVERRIDES = {
    "DATASET_SELECT": "LMVD",
    "MODEL_MODE": "fusion",
    "FUSION_MODE": "late",
    "USE_LEGACY_AV_BACKBONE": False,
    "USE_FEATURE_SEQUENCE_ENCODER": False,
    "USE_SEGMENT_MIL": False,
    "USE_SLIDING_SEGMENT_EVAL": False,
}

