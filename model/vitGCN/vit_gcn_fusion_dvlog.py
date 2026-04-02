from vit_gcn_fusion_lmvd import LMVDViTGCNFusion


class DVLOGViTGCNFusion(LMVDViTGCNFusion):
    """DVLOG-specific assembly with DVLOG-oriented defaults and switches."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('dataset', 'DVLOG')
        kwargs.setdefault('video_dim', 136)
        kwargs.setdefault('audio_dim', 25)
        kwargs.setdefault('fusion_mode', 'concat')
        kwargs.setdefault('face_valid_thresh', 0.1)
        kwargs.setdefault('global_alpha_init', 0.2)
        kwargs.setdefault('use_av_cross_attn', False)
        super().__init__(*args, **kwargs)
