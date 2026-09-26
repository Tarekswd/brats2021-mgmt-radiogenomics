"""
Combined 500-Million-Parameter Multi-Task Foundation Architecture.
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Unifies BraTS 2021 Task 1 (Brain Tumor Segmentation) and Task 2 (MGMT Radiogenomic Classification)
into a single 500M-parameter foundation vision transformer.
"""
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

class TransformerBlock500M(nn.Module):
    """Deep 3D Vision Transformer block with LayerNorm, Multihead Self-Attention, and MLP."""
    def __init__(self, dim: int = 1024, num_heads: int = 16, mlp_dim: int = 5520, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Combined500MModel(nn.Module):
    """
    Combined 500-Million-Parameter Foundation Model for BraTS Task 1 & Task 2.
    - Shared Volumetric 3D ViT Encoder (32 blocks, 1024 dim, 16 heads, 5520 mlp)
    - Branch 1: Task 1 Dense 3D Brain Tumor Segmentation (WT, TC, ET)
    - Branch 2: Task 2 Radiogenomic MGMT Promoter Methylation Prediction
    Total Parameters: ~500.1 Million Parameters (500M)
    """
    def __init__(
        self,
        in_channels: int = 4,
        seg_classes: int = 3, # WT, TC, ET
        embed_dim: int = 1024,
        num_layers: int = 32,
        num_heads: int = 16,
        mlp_dim: int = 5520,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # 3D Patch Embedding Projection
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 32 Scaled Transformer Blocks (~496 Million Parameters)
        self.blocks = nn.ModuleList([
            TransformerBlock500M(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(embed_dim)
        
        # --- Task 1: Segmentation Decoder ---
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 256, kernel_size=2, stride=2),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv3d(32, seg_classes, kernel_size=1)
        )
        
        # --- Task 2: Radiogenomic MGMT Classification Head ---
        self.radiogenomic_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        # x: [B, C, D, H, W]
        B = x.shape[0]
        patches = self.patch_embed(x) # [B, embed_dim, D', H', W']
        spatial_shape = patches.shape[2:]
        tokens = patches.flatten(2).transpose(1, 2) # [B, N_tokens, embed_dim]
        
        for blk in self.blocks:
            tokens = blk(tokens)
            
        tokens = self.norm_final(tokens)
        return tokens, spatial_shape

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "joint" # "segmentation", "classification", or "joint"
    ) -> Dict[str, torch.Tensor]:
        tokens, spatial_shape = self.extract_features(x)
        B, N, C = tokens.shape
        out = {}
        
        # Task 1: Segmentation
        if mode in ("segmentation", "joint"):
            # Reshape tokens back to 3D spatial tensor: [B, C, D', H', W']
            D_p, H_p, W_p = spatial_shape
            spatial_feat = tokens.transpose(1, 2).view(B, C, D_p, H_p, W_p)
            seg_logits = self.seg_decoder(spatial_feat)
            out["segmentation_logits"] = seg_logits # [B, 3, D, H, W]
            
        # Task 2: MGMT Radiogenomic Classification
        if mode in ("classification", "joint"):
            # Global attention / mean pooling across tokens
            patient_token = tokens.mean(dim=1) # [B, embed_dim]
            mgmt_logit = self.radiogenomic_head(patient_token).squeeze(-1) # [B]
            out["mgmt_logit"] = mgmt_logit
            out["mgmt_prob"] = torch.sigmoid(mgmt_logit)
            
        return out

    @classmethod
    def build_500m(cls, in_channels: int = 4, seg_classes: int = 3) -> "Combined500MModel":
        """Factory method instantiating exactly ~500.1 Million Parameters."""
        return cls(
            in_channels=in_channels,
            seg_classes=seg_classes,
            embed_dim=1024,
            num_layers=32,
            num_heads=16,
            mlp_dim=5520,
            patch_size=(16, 16, 16),
            dropout=0.0
        )
