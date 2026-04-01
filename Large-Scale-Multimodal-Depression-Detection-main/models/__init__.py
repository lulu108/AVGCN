from .MultiModalDepDet import MultiModalDepDet

__all__ = ["MultiModalDepDet", "DepMamba"]


def __getattr__(name):
    if name == "DepMamba":
        from .DepMamba import DepMamba
        return DepMamba
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
