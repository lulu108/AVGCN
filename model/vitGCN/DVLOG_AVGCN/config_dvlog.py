"""DVLOG profile for vitGCN training entrypoints."""

OVERRIDES = {
    "DATASET_SELECT": "DVLOG",
    "MODEL_MODE": "fusion",
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

