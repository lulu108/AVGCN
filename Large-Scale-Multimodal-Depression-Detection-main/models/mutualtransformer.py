# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
Source Paper: https://arxiv.org/abs/2401.14185
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------

import torch
import torch.nn as nn

class MutualTransformer(nn.Module):
    def __init__(self, a_d=145, v_d=128, d=256, f_d=512, l=6):
        """
        Initializes the MutualTransformer class.

        Args:
            d (int)                 :   Dimensionality of the model, used as the `d_model` parameter in the Transformer layers.
            f_d (int)               :   Fused Dimension from the model
            l (int)                 :   Number of layers in the Transformer encoder.
            num_iterations (int)    :   Number of cross-attention iterations between audio and visual features.
        """
        super().__init__()

        # Linear layer to adjust audio dimension
        self.a_linear = nn.Linear(a_d, 144) # 144 is divisible by 4

        # Encoders for audio features
        self.a_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=144, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l,
        )

        # Encoders for visual features
        self.v_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=v_d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l,
        )

        # Projection layers to match d dimension
        self.a_projection = nn.Linear(144, d)
        self.v_projection = nn.Linear(128, d)

        # Projection layers for cross-attention
        ## Audio
        self.qa_transform = nn.Linear(d, d)
        self.ka_transform = nn.Linear(d, d)
        self.va_transform = nn.Linear(d, d)
        ## Video
        self.qv_transform = nn.Linear(d, d)
        self.kv_transform = nn.Linear(d, d)
        self.vv_transform = nn.Linear(d, d)

        ## Fused: Audio + Video
        self.qf_transform = nn.Linear(f_d, d)
        self.kf_transform = nn.Linear(f_d, d)
        self.vf_transform = nn.Linear(f_d, d)

        # Cross-attention layers: audio x video
        self.cross_av = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )

        # Cross-attention layers: video x audio
        self.cross_va = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )

    def forward(self, a, v):
        """
        Forward pass for the MutualTransformer model.

        Args:
            v (torch.Tensor)        :   Input tensor representing visual features, shape [batch_size, seq_length, d].
            a (torch.Tensor)        :   Input tensor representing audio features, shape [batch_size, seq_length, d].

        Returns:
            torch.Tensor            :   Output tensor after mutual cross-attention and fusion, shape [batch_size, seq_length, 2*d].
        """
        # print(f"a shape : {a.shape}")
        # print(f"v shape : {v.shape}")
        # Encode audio and visual features
        a = self.a_linear(a) # for reducing 145 to 144 that is devisible by 4
        a_encoded = self.a_encoder(a)
        v_encoded = self.v_encoder(v)

        # print(f"a_encoded : {a_encoded.shape}")
        # print(f"v_encoded : {v_encoded.shape}")

        # Project encoded features to d dimension
        a_encoded = self.a_projection(a_encoded)
        v_encoded = self.v_projection(v_encoded)
        # print(f"a_encoded : {a_encoded.shape}")
        # print(f"v_encoded : {v_encoded.shape}")
        

        # MT-1: Audio (q), Video (v, k)
        q_a = self.qa_transform(a_encoded)
        k_a = self.ka_transform(v_encoded)
        v_a = self.va_transform(v_encoded)
        # Cross Attention    
        fav = self.cross_av(q_a, k_a, v_a)[0]

        # MT-2: Video (q), Audio (v, k)
        q_v = self.qv_transform(v_encoded)
        k_v = self.kv_transform(a_encoded)
        v_v = self.vv_transform(a_encoded)
        # Cross Attention
        fva = self.cross_va(q_v, k_v, v_v)[0]

        # T-3: Audio + Video Features
        fused_a_v = torch.cat([v_encoded, a_encoded], dim=2)
        # print(f"fused_a_v: {fused_a_v.shape}")
        q_f = self.qf_transform(fused_a_v)
        k_f = self.kf_transform(fused_a_v)
        v_f = self.vf_transform(fused_a_v)
        # Cross Attention
        f_a_v = self.cross_va(q_f, k_f, v_f)[0]

        # Concatenate and encode fused features: M-1, M-2 & T-1
        fused_features = torch.cat([fav, fva, f_a_v], dim=2)
        
        return fused_features
    

if __name__ == '__main__':

    # Example Usage
    batch_size = 16
    seq_length = 512
    a_d = 145  # Audio feature dimension
    v_d = 128  # Visual feature dimension

    # Create dummy input tensors
    audio_input = torch.randn(batch_size, seq_length, a_d)
    visual_input = torch.randn(batch_size, seq_length, v_d)

    # Instantiate the MutualTransformer model
    model = MutualTransformer()

    # Forward pass
    output_features = model(audio_input, visual_input)

    # Print the shape of the output
    print(f"Output features shape: {output_features.shape}")