"""
Modality-Specific Multi-Branch Network (BraTS 2021 3rd Place Winning Strategy).
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Trains 4 independent modality branches (T1, T1ce, T2, FLAIR) and fuses their patient embeddings.
"""
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from .cnn_lstm import SliceFeatureExtractor

class SingleModalityBranch(nn.Module):
    """Processes one specific MRI sequence with dedicated weights."""
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.encoder = SliceFeatureExtractor(in_channels=1, embed_dim=embed_dim)
        self.patient_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x_modality: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_modality: [B, 1, S, H, W]
        B, _, S, H, W = x_modality.shape
        slices = x_modality.squeeze(1).reshape(B * S, 1, H, W)
        feats = self.encoder(slices).view(B, S, -1) # [B, S, embed_dim]
        # Pool across slices: [B, embed_dim]
        patient_emb = feats.mean(dim=1)
        logit = self.head(patient_emb).squeeze(-1)
        return logit, patient_emb

class ModalitySpecificEnsemble(nn.Module):
    """
    Ensemble fusing independent predictions from T1, T1ce, T2, and FLAIR branches.
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.t1_branch = SingleModalityBranch(embed_dim)
        self.t1ce_branch = SingleModalityBranch(embed_dim)
        self.t2_branch = SingleModalityBranch(embed_dim)
        self.flair_branch = SingleModalityBranch(embed_dim)
        
        # Meta-fusion classifier on combined embeddings
        self.meta_head = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, 4, S, H, W] (0=T1, 1=T1ce, 2=T2, 3=FLAIR)
        l_t1, emb_t1 = self.t1_branch(x[:, 0:1])
        l_t1ce, emb_t1ce = self.t1ce_branch(x[:, 1:2])
        l_t2, emb_t2 = self.t2_branch(x[:, 2:3])
        l_flair, emb_flair = self.flair_branch(x[:, 3:4])
        
        combined_emb = torch.cat([emb_t1, emb_t1ce, emb_t2, emb_flair], dim=-1)
        meta_logit = self.meta_head(combined_emb).squeeze(-1)
        
        # Branch-averaged logit
        avg_logit = (l_t1 + l_t1ce + l_t2 + l_flair) / 4.0
        
        return {
            "meta_logit": meta_logit,
            "avg_logit": avg_logit,
            "t1_logit": l_t1,
            "t1ce_logit": l_t1ce,
            "t2_logit": l_t2,
            "flair_logit": l_flair
        }
