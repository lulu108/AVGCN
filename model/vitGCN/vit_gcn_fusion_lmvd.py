import math

import torch
from torch import nn

from _02GCN_Transformer import AnatomicalGCN
from fusion_common import (
    AFIFusionHead,
    ConcatFusionHead,
    IAResidualFusionHead,
    IntermediateBiGateFusion,
    IntermediateCrossFusion,
    LateFusionHead,
    masked_mean,
    safe_time_pool_temporal,
)
from Vit_gcnmodel import ViT


class LMVDViTGCNFusion(nn.Module):
    """LMVD-specific assembly for ViT+GCN fusion using shared fusion heads."""

    def __init__(
        self,
        spectra_size=915,
        patch_size=15,
        dim=256,
        depth=8,
        heads=8,
        dim_mlp=1024,
        sd=0.0,
        gcn_out_dim=32,
        gcn_nhead=4,
        use_global_branch=True,
        global_alpha_init=0.5,
        num_classes=2,
        dropout=0.45,
        channel=256,
        video_dim=171,
        audio_dim=128,
        dataset='LMVD',
        fusion_mode='it_cross',
        use_av_cross_attn=True,
        av_cross_heads=None,
        av_cross_alpha_init=0.1,
        use_temporal_conv_stem=False,
        temporal_conv_stem_kernel=3,
        temporal_conv_stem_for='both',
        temporal_conv_stem_dvlog_only=False,
        use_shallow_audio_tcn_encoder=False,
        audio_tcn_kernel_size=3,
        audio_tcn_dropout=0.05,
        audio_tcn_encoder_dvlog_only=True,
        use_temporal_attn_pool=False,
        temporal_attn_pool_dvlog_only=True,
        temporal_attn_pool_alpha_init=0.2,
        use_legacy_av_backbone=False,
        use_feature_sequence_encoder=False,
        fs_video_local_blocks=2,
        fs_audio_local_blocks=3,
        fs_video_global_depth=None,
        fs_audio_global_depth=None,
        fs_dilated_audio=True,
        fs_fusion_dropout=None,
        face_valid_thresh=0.2,
        region_scheme='legacy6',
        region_fusion_mode='cross_attn',
        gcn_temporal_mode='transformer',
        region_mlp_dropout=0.1,
        tcn_kernel_size=3,
        modality_mode='fusion',
        use_ia_recalibration=False,
        ia_recal_bidirectional=False,
        ia_recal_mode='g2v',
        ia_hidden_ratio=0.5,
        ia_dropout=0.05,
        ia_alpha_init=0.0,
        concat_blend_init=0.3,
        single_modality_clean_path=True,
        use_strong_audio_encoder=False,
        use_strong_video_encoder=False,
        audio_fixed_len=128,
        video_fixed_len=128,
        video_use_delta=True,
    ):
        super().__init__()
        self.dim = dim
        self.dataset = str(dataset).upper()
        self.fusion_mode = fusion_mode
        self.modality_mode = str(modality_mode).lower()
        if self.modality_mode not in {'fusion', 'gcn_only'}:
            raise ValueError("modality_mode for ViT_GCN_Fusion must be 'fusion' or 'gcn_only'")

        self.use_ia_recalibration = bool(use_ia_recalibration)
        ia_mode = str(ia_recal_mode).lower()
        if ia_mode == 'off':
            ia_mode = 'bi' if ia_recal_bidirectional else 'g2v'
        self.ia_recal_mode = ia_mode

        self.patch_size = patch_size
        self.face_valid_thresh = face_valid_thresh
        self.region_scheme = region_scheme
        if use_av_cross_attn is None:
            use_av_cross_attn = (self.dataset != 'DVLOG')

        self.vit_branch = ViT(
            spectra_size=spectra_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_mlp=dim_mlp,
            channel=channel,
            dim_head=dim // heads,
            dropout=dropout,
            sd=sd,
            video_dim=video_dim,
            audio_dim=audio_dim,
            dataset=self.dataset,
            use_av_cross_attn=use_av_cross_attn,
            av_cross_heads=av_cross_heads,
            av_cross_alpha_init=av_cross_alpha_init,
            use_temporal_conv_stem=use_temporal_conv_stem,
            temporal_conv_stem_kernel=temporal_conv_stem_kernel,
            temporal_conv_stem_for=temporal_conv_stem_for,
            temporal_conv_stem_dvlog_only=temporal_conv_stem_dvlog_only,
            use_shallow_audio_tcn_encoder=use_shallow_audio_tcn_encoder,
            audio_tcn_kernel_size=audio_tcn_kernel_size,
            audio_tcn_dropout=audio_tcn_dropout,
            audio_tcn_encoder_dvlog_only=audio_tcn_encoder_dvlog_only,
            use_legacy_av_backbone=use_legacy_av_backbone,
            use_feature_sequence_encoder=use_feature_sequence_encoder,
            fs_video_local_blocks=fs_video_local_blocks,
            fs_audio_local_blocks=fs_audio_local_blocks,
            fs_video_global_depth=fs_video_global_depth,
            fs_audio_global_depth=fs_audio_global_depth,
            fs_dilated_audio=fs_dilated_audio,
            fs_fusion_dropout=fs_fusion_dropout,
            modality_mode=modality_mode,
            single_modality_clean_path=single_modality_clean_path,
            use_strong_audio_encoder=use_strong_audio_encoder,
            use_strong_video_encoder=use_strong_video_encoder,
            audio_fixed_len=audio_fixed_len,
            video_fixed_len=video_fixed_len,
            video_use_delta=video_use_delta,
        )
        self.vit_branch.classifier = nn.Identity()

        self.use_temporal_attn_pool = use_temporal_attn_pool
        self.temporal_attn_pool_effective = (
            self.use_temporal_attn_pool
            and (not temporal_attn_pool_dvlog_only or self.dataset == 'DVLOG')
        )
        if self.temporal_attn_pool_effective:
            self.temporal_attn_fc = nn.Linear(dim, 1)
            self.temporal_attn_alpha = nn.Parameter(torch.tensor(float(temporal_attn_pool_alpha_init)))
        else:
            self.temporal_attn_fc = None
            self.temporal_attn_alpha = None

        self.gcn_branch = AnatomicalGCN(
            out_dim=gcn_out_dim,
            nhead=gcn_nhead,
            num_classes=num_classes,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            face_valid_thresh=face_valid_thresh,
            use_global_branch=use_global_branch,
            global_alpha_init=global_alpha_init,
            region_scheme=region_scheme,
            region_fusion_mode=region_fusion_mode,
            gcn_temporal_mode=gcn_temporal_mode,
            region_mlp_dropout=region_mlp_dropout,
            tcn_kernel_size=tcn_kernel_size,
        )
        gcn_feat_dim = gcn_out_dim * 4

        self.gcn_proj = nn.Sequential(
            nn.Linear(gcn_feat_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gcn_only_proj = nn.Sequential(
            nn.LayerNorm(gcn_feat_dim),
            nn.Linear(gcn_feat_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.concat_head = None
        self.concat_blend_logit = None

        if fusion_mode == 'late':
            self.fusion_head = LateFusionHead(dim=dim, gcn_feat_dim=gcn_feat_dim, dropout=dropout)
            self.recon_v = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.recon_g = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, gcn_feat_dim),
                nn.ReLU(),
                nn.Linear(gcn_feat_dim, gcn_feat_dim),
            )
        elif fusion_mode == 'it_bi_gate':
            self.fusion_head = IntermediateBiGateFusion(dim=dim, heads=heads, dropout=dropout)
            self.recon_v = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.recon_g = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        elif fusion_mode == 'concat':
            if self.use_ia_recalibration:
                self.fusion_head = IAResidualFusionHead(
                    dim=dim,
                    hidden_ratio=ia_hidden_ratio,
                    dropout=ia_dropout,
                    alpha_init=ia_alpha_init,
                    mode=self.ia_recal_mode,
                )
            else:
                self.fusion_head = IntermediateBiGateFusion(dim=dim, heads=heads, dropout=dropout)
            self.concat_head = ConcatFusionHead(dim=dim, dropout=dropout)
            blend_init = float(min(max(concat_blend_init, 1e-4), 1.0 - 1e-4))
            blend_init_t = torch.tensor(blend_init, dtype=torch.float32)
            self.concat_blend_logit = nn.Parameter(torch.log(blend_init_t / (1.0 - blend_init_t)))
            self.recon_v = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.recon_g = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        elif fusion_mode == 'afi':
            self.fusion_head = AFIFusionHead(dim=dim, heads=heads, dropout=dropout, n_blocks=2)
            self.recon_v = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.recon_g = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        else:
            self.fusion_head = IntermediateCrossFusion(dim=dim, heads=heads, dropout=dropout)
            self.recon_v = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.recon_g = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def _temporal_attention_pool(self, temporal_tokens, patch_valid):
        scores = self.temporal_attn_fc(temporal_tokens).squeeze(-1)
        scores = scores.masked_fill(~patch_valid, -1e9)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.sum(temporal_tokens * attn.unsqueeze(-1), dim=1)
        return pooled, attn

    def forward(self, video_features, audio_features, face_regions, actual_lens, quality=None):
        bsz = video_features.size(0)
        device = video_features.device

        if quality is None:
            q_v = torch.ones(bsz, device=device, dtype=video_features.dtype)
            q_g = torch.ones(bsz, device=device, dtype=video_features.dtype)
        elif isinstance(quality, dict):
            q_v = quality.get('q_v', torch.ones(bsz, device=device, dtype=video_features.dtype))
            q_g = quality.get('q_g', torch.ones(bsz, device=device, dtype=video_features.dtype))
            q_v = q_v.to(device=device, dtype=video_features.dtype)
            q_g = q_g.to(device=device, dtype=video_features.dtype)
        else:
            q_v = quality[:, 0].to(device=device, dtype=video_features.dtype)
            q_g = quality[:, 1].to(device=device, dtype=video_features.dtype)

        if self.modality_mode == 'gcn_only':
            gcn_out = self.gcn_branch(face_regions, actual_lens, return_dict=True)
            gcn_repr = gcn_out['gcn_repr'] * q_g.unsqueeze(-1)
            video_repr = self.gcn_only_proj(gcn_repr)
            logits_fusion = self.classifier(video_repr)
            token_steps = max(1, int(math.ceil(float(video_features.size(1)) / float(self.patch_size))))
            token_mask = torch.ones(gcn_repr.size(0), token_steps, dtype=torch.bool, device=device)
            zero_tokens = torch.zeros(gcn_repr.size(0), token_steps, self.dim, device=device, dtype=video_features.dtype)
            return {
                'logits': logits_fusion,
                'logits_gcn': gcn_out['logits_gcn'],
                'recon_v': None,
                'recon_g': None,
                'video_tokens': zero_tokens,
                'audio_tokens': zero_tokens,
                'av_tokens': zero_tokens,
                'vit_repr': torch.zeros(gcn_repr.size(0), self.dim, device=device, dtype=video_features.dtype),
                'gcn_repr': gcn_repr,
                'audio_repr': torch.zeros(gcn_repr.size(0), self.dim, device=device, dtype=video_features.dtype),
                'av_repr': video_repr,
                'token_mask': token_mask,
            }

        av_out = self.vit_branch(video_features, audio_features, actual_lens=actual_lens, return_dict=True)
        vit_tokens = av_out['av_tokens']
        vit_repr = av_out['av_repr']
        audio_repr = av_out['audio_repr']
        patch_valid = av_out['token_mask']
        target_t = vit_tokens.size(1)

        if self.fusion_mode == 'late':
            if self.temporal_attn_pool_effective:
                attn_pool, _ = self._temporal_attention_pool(vit_tokens, patch_valid)
                alpha = torch.clamp(torch.tanh(self.temporal_attn_alpha), 0.0, 1.0)
                vit_repr = (1.0 - alpha) * vit_repr + alpha * attn_pool

            gcn_out = self.gcn_branch(face_regions, actual_lens, return_dict=True)
            gcn_repr = gcn_out['gcn_repr']
            vit_repr = vit_repr * q_v.unsqueeze(-1)
            gcn_repr = gcn_repr * q_g.unsqueeze(-1)
            video_repr = self.fusion_head(vit_repr, gcn_repr)
            recon_v = self.recon_v(video_repr)
            recon_g = self.recon_g(video_repr)
        else:
            gcn_out = self.gcn_branch(face_regions, actual_lens, return_dict=True)
            gcn_temporal = gcn_out['temporal_out']
            gcn_tokens = self.gcn_proj(gcn_temporal)
            gcn_tokens = safe_time_pool_temporal(gcn_tokens, target_t, actual_lens)

            vit_tokens = vit_tokens * q_v[:, None, None]
            gcn_tokens = gcn_tokens * q_g[:, None, None]

            frame_mask = gcn_out.get('time_mask', None)
            if frame_mask is None:
                pooled_face_mask = patch_valid
            else:
                pooled_mask = safe_time_pool_temporal(frame_mask.float().unsqueeze(-1), target_t, actual_lens).squeeze(-1)
                pooled_face_mask = pooled_mask > 0.0

            time_mask = patch_valid & pooled_face_mask
            all_invalid = ~time_mask.any(dim=1)
            if all_invalid.any():
                time_mask[all_invalid] = patch_valid[all_invalid]

            valid_tok = time_mask.float().unsqueeze(-1)
            vit_tokens = vit_tokens * valid_tok
            gcn_tokens = gcn_tokens * valid_tok

            vit_repr = masked_mean(vit_tokens, time_mask)
            gcn_repr = masked_mean(gcn_tokens, time_mask)

            if self.fusion_mode == 'concat':
                token_repr = self.fusion_head(vit_tokens, gcn_tokens, time_mask)
                sum_repr = self.concat_head(vit_repr, gcn_repr)
                blend = torch.sigmoid(self.concat_blend_logit)
                video_repr = (1.0 - blend) * sum_repr + blend * token_repr
            else:
                video_repr = self.fusion_head(vit_tokens, gcn_tokens, time_mask)

            recon_v = self.recon_v(video_repr)
            recon_g = self.recon_g(video_repr)

        logits_fusion = self.classifier(video_repr)
        return {
            'logits': logits_fusion,
            'logits_gcn': gcn_out['logits_gcn'],
            'recon_v': recon_v,
            'recon_g': recon_g,
            'video_tokens': av_out['video_tokens'],
            'audio_tokens': av_out['audio_tokens'],
            'av_tokens': av_out['av_tokens'],
            'vit_repr': vit_repr,
            'gcn_repr': gcn_repr,
            'audio_repr': audio_repr,
            'av_repr': av_out['av_repr'],
            'token_mask': av_out['token_mask'],
        }
