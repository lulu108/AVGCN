"""DVLOG AV-only legacy profile for vitGCN training entrypoints."""

OVERRIDES = {
    "DATASET_SELECT": "DVLOG",
    "MODEL_MODE": "av_only",
    "USE_LEGACY_AV_BACKBONE": True,
    "USE_FEATURE_SEQUENCE_ENCODER": False,
    "FS_VIDEO_LOCAL_BLOCKS": 2,
    "FS_AUDIO_LOCAL_BLOCKS": 3,
    "FS_DILATED_AUDIO": True,
    "FS_FUSION_DROPOUT": 0.05,
    "USE_SEGMENT_MIL": False,
    "USE_SLIDING_SEGMENT_EVAL": True,
}
