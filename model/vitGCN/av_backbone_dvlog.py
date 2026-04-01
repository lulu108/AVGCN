from av_backbone_common import AVBackboneCore


class DVLOGAVBackbone(AVBackboneCore):
    def __init__(self, *args, **kwargs):
        kwargs['dataset'] = 'DVLOG'
        super().__init__(*args, **kwargs)
