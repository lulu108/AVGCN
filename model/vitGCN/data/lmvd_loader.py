"""LMVD-specific data loader entrypoints."""

from kfoldLoader import MyDataLoader
from kfoldLoader_multimodal import MultiModalDataLoader, collate_fn_multimodal


class LMVDMultiModalDataLoader(MultiModalDataLoader):
    """Force dataset='LMVD' for multimodal LMVD runs."""

    def __init__(self, *args, **kwargs):
        kwargs["dataset"] = "LMVD"
        super().__init__(*args, **kwargs)


LMVDBasicDataLoader = MyDataLoader


__all__ = ["LMVDMultiModalDataLoader", "LMVDBasicDataLoader", "collate_fn_multimodal"]
