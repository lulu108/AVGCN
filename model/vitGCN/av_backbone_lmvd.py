from av_backbone_common import AVBackboneCore


class LMVDAVBackbone(AVBackboneCore):
    def __init__(self, *args, **kwargs):
        kwargs['dataset'] = 'LMVD'
        super().__init__(*args, **kwargs)
