"""DVLOG-only multimodal data loader entrypoint."""

from kfoldLoader_multimodal import MultiModalDataLoader, collate_fn_multimodal


class DVLOGMultiModalDataLoader(MultiModalDataLoader):
    """Force dataset='DVLOG' to avoid passing dataset switches around training code."""

    def __init__(self, *args, **kwargs):
        kwargs["dataset"] = "DVLOG"
        super().__init__(*args, **kwargs)


__all__ = ["DVLOGMultiModalDataLoader", "collate_fn_multimodal"]
