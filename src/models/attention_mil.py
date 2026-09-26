"""
Attention Multiple Instance Learning (Attention MIL) for BraTS 2021 Task 2.
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Learns slice-level importance weights dynamically to form patient embeddings.
"""
from typing import Tuple
import torch
import torch.nn as nn
from .cnn_lstm import SliceFeatureExtractor

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning mechanism.
    a = softmax(w^T (tanh(V h) ⊙ sigm(U h)))
    """
    def __init__(self, in_channels: int = 4, embed_dim: int = 256, attn_dim: int = 128):
        super().__init__()
        self.slice_cnn = SliceFeatureExtractor(in_channels, embed_dim)
        
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(attn_dim, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, S, H, W]
        B, C, S, H, W = x.shape
        x_slices = x.permute(0, 2, 1, 3, 4).reshape(B * S, C, H, W)
        H_feats = self.slice_cnn(x_slices).view(B, S, -1) # [B, S, embed_dim]
        
        # Calculate gated attention weights
        A_V = self.attention_V(H_feats) # [B, S, attn_dim]
        A_U = self.attention_U(H_feats) # [B, S, attn_dim]
        A = self.attention_w(A_V * A_U) # [B, S, 1]
        A = torch.softmax(A, dim=1) # [B, S, 1] (Slice attention weights)
        
        # Patient representation: sum(A_i * H_i)
        M = torch.sum(A * H_feats, dim=1) # [B, embed_dim]
        logits = self.classifier(M).squeeze(-1) # [B]
        
        return logits, A.squeeze(-1) # returns logits and slice attention maps
