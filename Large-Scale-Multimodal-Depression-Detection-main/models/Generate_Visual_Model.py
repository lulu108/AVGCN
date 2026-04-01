'''
-*- coding: utf-8 -*-
@Author     :   Md Rezwanul Haque
@Source     :   https://github.com/katerynaCh/MMA-DFER/blob/main/AudioMAE/audio_models_vit.py
@Adapted    :   https://github.com/katerynaCh/MMA-DFER/blob/main/models/Generate_Model.py
            :   https://github.com/katerynaCh/MMA-DFER/blob/main/AudioMAE/audio_models_vit.py
@Paper      :   https://arxiv.org/abs/2404.09010
@Description:   This is for Visual Feature Extraction!
'''
import argparse
import torch
from typing import Tuple
import torch.nn as nn
import math
import torch.nn.functional as F
from .dfer.Temporal_Model import *
from .dfer.VisualMAE import visual_models_vit

def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
        gs_old = None,
) -> torch.Tensor:
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    if gs_old is None:
        gs_old = (int(math.sqrt(len(posemb_grid))), int(math.sqrt(len(posemb_grid))))
    if gs_new is None or not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old[0], gs_old[1], -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb

# Define a simple linear patch embedding for visual-only processing.
class VisualPatchEmbed(nn.Module):
    """Simple patch embedding for sequence visual data."""
    def __init__(self, in_dim=128, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
    def forward(self, x):
        # x: [B, seq_length, in_dim]
        return self.proj(x)  # outputs [B, seq_length, embed_dim]

class GenerateVisualModel(nn.Module):
    def __init__(self, temporal_layers=6, number_class=10):
        super().__init__()
        self.temporal_layers = temporal_layers
        self.number_class = number_class

        # The temporal transformer for downstream processing.
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=self.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        # Define downsampling layers: audio
        self.visual_downsample = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(128)  # Downsample to the desired length
        )
        
        self.our_classifier = nn.Linear(512, self.number_class)
        self.vision_proj = nn.Linear(768, 512)

        # For the visual branch, update the expected dimensions.
        # Now expecting an visual input of shape [batch, seq_length, feature] where:
        #   seq_length = 128 and feature dimension = 128.
        self.visual_seq_len = 128   # Updated sequence length.
        self.visual_in_dim = 128    # Updated input feature dimension.
        self.embed_dim = 768

        # Build the visual model branch.
        self._build_visual_model()

        # For visual-only, replace the patch embedding with a linear projection.
        self.visual_model.patch_embed = VisualPatchEmbed(in_dim=self.visual_in_dim, embed_dim=self.embed_dim)
        # Adjust positional embeddings to match sequence length (plus one for CLS token).
        num_tokens = self.visual_seq_len + 1
        self.visual_model.pos_embed = nn.Parameter(torch.randn(1, num_tokens, self.embed_dim))

    def _build_visual_model(self, model_name='vit_base_patch16', drop_path_rate=0.1, global_pool=False,
                           mask_2d=True, use_custom_patch=False, ckpt_path='../pretrained_models/visualmae_pretrained.pth'):
        
        # Create the visual model with the desired sequence length.
        self.visual_model = visual_models_vit.__dict__[model_name](
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
            mask_2d=mask_2d,
            use_custom_patch=use_custom_patch,
            n_seq=self.visual_seq_len, 
            n_progr=3)
        
        # Override pos_embed to match our desired shape: [1, visual_seq_len + 1, embed_dim]
        self.visual_model.pos_embed = nn.Parameter(torch.randn(1, self.visual_seq_len + 1, self.embed_dim))
        
        # Allow argparse.Namespace for safe unpickling.
        torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        ckpt = ckpt['model']
        
        # Resize pos_embed from the checkpoint to our target shape.
        orig_pos_embed = ckpt['pos_embed']
        # print("Original visual pos_embed shape:", orig_pos_embed.shape, 
        #       "Target visual_model pos_embed shape:", self.visual_model.pos_embed.shape)
        ckpt['pos_embed'] = resize_pos_embed(
            orig_pos_embed, 
            self.visual_model.pos_embed, 
            gs_old=(1024//16, 128//16), 
            gs_new=(self.visual_seq_len, 1)
        )
        
        # Remove patch_embed weights from the checkpoint since we replace it with a linear projection.
        ckpt.pop('patch_embed.proj.weight', None)
        ckpt.pop('patch_embed.proj.bias', None)
        
        msg = self.visual_model.load_state_dict(ckpt, strict=False)
        # print('visual checkpoint loading: ', msg)

    def forward_visual(self, visual):
        visual = self.visual_downsample(visual.transpose(1, 2))
        """
        Process only visual data.
        Expects visual input of shape [batch, seq_length, in_dim], e.g. [8, 128, 128].
        """
        # Use our new patch embedding: project from in_dim -> embed_dim.
        x = self.visual_model.patch_embed(visual)  # [B, seq_length, embed_dim]
        B = x.shape[0]
        # Add a CLS token.
        cls_tokens = self.visual_model.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + seq_length, embed_dim]
        # Add positional embeddings.
        x = x + self.visual_model.pos_embed[:, :x.size(1)]
        x = self.visual_model.pos_drop(x)
        # Forward through each transformer block.
        for blk in self.visual_model.blocks:
            x = blk(x)
        x = self.visual_model.norm(x)
        return x 

        # print(f"x shape: {x.shape}")
        # # Use the CLS token for classification.
        # visual_features = x[:, 0]
        # # Project visual_features from 768 to 512 using vision_proj.
        # visual_features = self.vision_proj(visual_features)
        # print(f"visual_features: {visual_features.shape}")
        # output = self.our_classifier(visual_features)
        # return output

    def forward(self, image, visual):
        # For visual-only inference, simply call forward_visual.
        return self.forward_visual(visual)

# Example usage for visual-only.
if __name__ == '__main__':

    temporal_layers = 6
    number_class = 10
    
    model = GenerateVisualModel(temporal_layers, number_class)
    # model.eval()
    
    # Simulate visual input with shape [batch, seq_length, in_dim] -> e.g. [8, 128, 128]
    input_tdim = 1320
    test_input = torch.rand([16, input_tdim, 256])
    
    # with torch.no_grad():

    outputs = model.forward_visual(test_input)
    print("visual-only output shape:", outputs.shape)
