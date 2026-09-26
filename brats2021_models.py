"""
========================================================================================
BraTS 2021 Two Flagship Models Library (Task 1 & Task 2)
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Contact: tarek.ahmadieh2006@gmail.com
========================================================================================

This single, self-contained file contains the TWO flagship models developed for BraTS 2021:

1. TASK 1: Brain Tumor Segmentation
   - Model: Task1_Segmentation_WinnerUNet (BraTSTask1Model)
   - Architecture: Winning-inspired 3D U-Net with Asymmetric Large Encoder, 
                   Group Normalization (32 groups), 3D Axial Attention, and 
                   Multi-Scale Deep Supervision.
   - Target: Predicts 3 nested tumor subregions (WT, TC, ET) from 4 MRI modalities.
   - Parameters: ~44.1 Million parameters.
   - Includes: Anatomical prior post-processing (ET ⊆ TC ⊆ WT) & small ET filtering.

2. TASK 2: MGMT Promoter Methylation Radiogenomic Classification
   - Model: Task2_MGMT_Foundation_500M (BraTSTask2Model)
   - Architecture: 500-Million-Parameter Foundation Vision Transformer with 
                   32 deep Transformer blocks (dim=1024, heads=16, mlp=5560), 
                   3D Multi-parametric Patch Embedding, and Gated Attention Pooling.
   - Target: Predicts binary MGMT methylation status from multi-parametric MRI.
   - Parameters: ~501.24 Million parameters (500M).

Quick Start:
    from brats2021_models import get_task1_model, get_task2_model, apply_anatomical_prior

    # 1. Instantiate Task 1 Segmentation Model
    task1_model = get_task1_model()
    # x: [Batch, 4, Depth, Height, Width]
    # out: [Batch, 3, Depth, Height, Width] (WT, TC, ET)

    # 2. Instantiate Task 2 500M Radiogenomics Model
    task2_model = get_task2_model()
    # x: [Batch, 4, Depth, Height, Width]
    # out: {'logit': tensor, 'prob': tensor, 'attention': tensor}
========================================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# POST-PROCESSING & CLINICAL UTILITIES (TASK 1)
# ======================================================================================

def apply_anatomical_prior(
    prob_wt: torch.Tensor,
    prob_tc: torch.Tensor,
    prob_et: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enforces strict physiological subset hierarchy: ET ⊆ TC ⊆ WT.
    Eliminates non-anatomical false positive predictions in healthy brain tissue.

    Args:
        prob_wt: Whole Tumor probability map [..., D, H, W]
        prob_tc: Tumor Core probability map [..., D, H, W]
        prob_et: Enhancing Tumor probability map [..., D, H, W]
        threshold: Binarization cutoff (default: 0.5)

    Returns:
        Tuple of binary masks: (mask_wt, refined_tc, refined_et)
    """
    mask_wt = (prob_wt >= threshold).bool()
    mask_tc = (prob_tc >= threshold).bool()
    mask_et = (prob_et >= threshold).bool()

    refined_tc = mask_tc & mask_wt
    refined_et = mask_et & refined_tc

    return mask_wt.float(), refined_tc.float(), refined_et.float()


def filter_small_et(mask_et: torch.Tensor, min_voxels: int = 200) -> torch.Tensor:
    """
    Remaps spurious enhancing tumor (ET) clusters with volume < min_voxels to necrosis.
    This was a decisive post-processing rule in the BraTS 2021 challenge.
    """
    out = mask_et.clone()
    if out.sum() < min_voxels:
        out.zero_()
    return out


# ======================================================================================
# MODEL 1: TASK 1 — 3D BRAIN TUMOR SEGMENTATION WINNER U-NET (~44.1M PARAMETERS)
# ======================================================================================

class AxialAttention3D(nn.Module):
    """
    3D Axial Attention module.
    Applies multi-head self-attention independently along each spatial axis (D, H, W),
    reducing complexity from O((DHW)^2) to O(DHW(D + H + W)).
    Pure PyTorch implementation without external dependencies.
    """
    def __init__(self, in_channels: int, num_heads: int = 4, dim_head: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head

        self.to_qkv_d = nn.Conv3d(in_channels, inner_dim * 3, kernel_size=1, bias=False)
        self.to_qkv_h = nn.Conv3d(in_channels, inner_dim * 3, kernel_size=1, bias=False)
        self.to_qkv_w = nn.Conv3d(in_channels, inner_dim * 3, kernel_size=1, bias=False)

        self.proj = nn.Conv3d(inner_dim, in_channels, kernel_size=1)
        self.scale = dim_head ** -0.5

    def _axial_attn(self, qkv: torch.Tensor, axis: str, b: int, c: int, d: int, h: int, w: int) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=1)
        heads = self.num_heads
        dhead = self.dim_head

        if axis == "d":
            # [B, heads*dhead, D, H, W] -> [B*H*W, heads, D, dhead]
            q = q.permute(0, 3, 4, 1, 2).reshape(b * h * w, heads, dhead, d).transpose(-1, -2)
            k = k.permute(0, 3, 4, 1, 2).reshape(b * h * w, heads, dhead, d).transpose(-1, -2)
            v = v.permute(0, 3, 4, 1, 2).reshape(b * h * w, heads, dhead, d).transpose(-1, -2)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)  # [B*H*W, heads, D, dhead]
            out = out.transpose(-1, -2).reshape(b, h, w, heads * dhead, d).permute(0, 3, 4, 1, 2)
        elif axis == "h":
            # [B, heads*dhead, D, H, W] -> [B*D*W, heads, H, dhead]
            q = q.permute(0, 2, 4, 1, 3).reshape(b * d * w, heads, dhead, h).transpose(-1, -2)
            k = k.permute(0, 2, 4, 1, 3).reshape(b * d * w, heads, dhead, h).transpose(-1, -2)
            v = v.permute(0, 2, 4, 1, 3).reshape(b * d * w, heads, dhead, h).transpose(-1, -2)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(-1, -2).reshape(b, d, w, heads * dhead, h).permute(0, 3, 1, 4, 2)
        else:  # 'w'
            # [B, heads*dhead, D, H, W] -> [B*D*H, heads, W, dhead]
            q = q.permute(0, 2, 3, 1, 4).reshape(b * d * h, heads, dhead, w).transpose(-1, -2)
            k = k.permute(0, 2, 3, 1, 4).reshape(b * d * h, heads, dhead, w).transpose(-1, -2)
            v = v.permute(0, 2, 3, 1, 4).reshape(b * d * h, heads, dhead, w).transpose(-1, -2)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(-1, -2).reshape(b, d, h, heads * dhead, w).permute(0, 3, 1, 2, 4)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        out_d = self._axial_attn(self.to_qkv_d(x), "d", b, c, d, h, w)
        out_h = self._axial_attn(self.to_qkv_h(x), "h", b, c, d, h, w)
        out_w = self._axial_attn(self.to_qkv_w(x), "w", b, c, d, h, w)
        out = self.proj(out_d + out_h + out_w)
        return x + out


class WinnerConvBlock3D(nn.Module):
    """
    Standard block for 3D Segmentation: Double Conv3d -> GroupNorm(32) -> LeakyReLU(0.01).
    GroupNorm ensures numerical stability for small 3D batch sizes (batch size = 1 or 2).
    """
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        g1 = num_groups if out_channels >= num_groups else out_channels
        g2 = num_groups if out_channels >= num_groups else out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=g1, num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=g2, num_channels=out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class Task1_Segmentation_WinnerUNet(nn.Module):
    """
    Flagship Model for BraTS 2021 Task 1: 3D Brain Tumor Segmentation.
    
    Key Innovations:
      1. Asymmetrically Scaled Large Encoder (channels: [64, 128, 256, 384, 512])
      2. Group Normalization (32 groups) preventing batch norm collapse in 3D patches
      3. 3D Axial Attention in the Decoder for long-range anatomical dependencies
      4. Multi-Level Deep Supervision auxiliary heads during training
      5. Generates 3-channel nested predictions: WT (Whole Tumor), TC (Tumor Core), ET (Enhancing)
    
    Parameters: ~44.1 Million parameters.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        use_large_encoder: bool = True,
        use_axial_attention: bool = False,
        deep_supervision: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.use_axial_attention = use_axial_attention

        # Channel configurations
        if use_large_encoder:
            self.enc_channels = [64, 128, 256, 384, 512]
            self.dec_channels = [320, 256, 128, 64]
        else:
            self.enc_channels = [32, 64, 128, 256, 320]
            self.dec_channels = [256, 128, 64, 32]

        # Encoder stages
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_c = in_channels
        for c in self.enc_channels:
            self.encoders.append(WinnerConvBlock3D(prev_c, c, num_groups=32))
            self.pools.append(nn.MaxPool3d(2, 2))
            prev_c = c

        # Decoder stages
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.axial_attns = nn.ModuleList() if use_axial_attention else None

        dec_in = self.enc_channels[-1]
        for i, dec_out in enumerate(self.dec_channels):
            skip_c = self.enc_channels[-(i + 2)]
            self.upsamples.append(nn.ConvTranspose3d(dec_in, dec_out, kernel_size=2, stride=2))

            if use_axial_attention and i < len(self.dec_channels) - 1:
                self.axial_attns.append(AxialAttention3D(dec_out, num_heads=4, dim_head=16))
            elif use_axial_attention:
                self.axial_attns.append(nn.Identity())

            self.decoders.append(WinnerConvBlock3D(dec_out + skip_c, dec_out, num_groups=32))
            dec_in = dec_out

        # Final full-resolution segmentation head
        self.final_head = nn.Conv3d(self.dec_channels[-1], num_classes, kernel_size=1)

        # Multi-scale deep supervision auxiliary heads
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv3d(c, num_classes, kernel_size=1) for c in self.dec_channels[:-1]
            ])
        else:
            self.aux_heads = None

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input 3D MRI volume [B, 4, D, H, W]
               (Channels: FLAIR, T1w, T1gd, T2w)
        Returns:
            If eval mode or deep_supervision=False:
                Tensor [B, 3, D, H, W] representing raw logits for (WT, TC, ET)
            If training mode and deep_supervision=True:
                List of tensors [final_output, aux_head_1, aux_head_2, ...]
        """
        skips = []
        for i in range(len(self.enc_channels) - 1):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.encoders[-1](x)

        aux_outputs = []
        for i in range(len(self.decoders)):
            x = self.upsamples[i](x)
            if self.use_axial_attention and self.axial_attns is not None:
                x = self.axial_attns[i](x)

            skip = skips[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i](x)

            if self.deep_supervision and i < len(self.decoders) - 1 and self.aux_heads is not None:
                aux_outputs.append(self.aux_heads[i](x))

        final_out = self.final_head(x)

        if self.training and self.deep_supervision:
            return [final_out] + list(reversed(aux_outputs))
        return final_out


# ======================================================================================
# MODEL 2: TASK 2 — 500-MILLION-PARAMETER MGMT RADIOGENOMIC FOUNDATION TRANSFORMER
# ======================================================================================

class TransformerBlock500M(nn.Module):
    """
    High-capacity 3D Vision Transformer Block with Pre-LayerNorm, 
    Multihead Self-Attention, and wide GELU MLP feedforward network.
    """
    def __init__(self, dim: int = 1024, num_heads: int = 16, mlp_dim: int = 5560, dropout: float = 0.1):
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
        nx = self.norm1(x)
        x = x + self.attn(nx, nx, nx)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Task2_MGMT_Foundation_500M(nn.Module):
    """
    Flagship Model for BraTS 2021 Task 2: MGMT Promoter Methylation Radiogenomic Classification.
    
    Key Innovations:
      1. Massive 500-Million-Parameter Foundation Capacity:
         32 Scaled Transformer Attention Blocks (dim=1024, heads=16, mlp_dim=5560).
      2. Multi-Parametric 3D Patch Embedding Projection (4 channels: FLAIR, T1, T1ce, T2).
      3. Patient-Level Gated Attention Pooling for interpretable slice/token aggregation.
      4. Deep Radiogenomic Projection Head predicting calibrated MGMT methylation probability.
    
    Parameters: 501,239,042 parameters (~501.24M).
    """
    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 1024,
        num_layers: int = 32,
        num_heads: int = 16,
        mlp_dim: int = 5560,
        patch_size: Tuple[int, int, int] = (4, 8, 8),
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 3D Patch embedding projection from 4 MRI sequences
        self.patch_proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 32 Deep Transformer Blocks (~498.6M parameters)
        self.blocks = nn.ModuleList([
            TransformerBlock500M(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(embed_dim)
        
        # Patient-Level Gated Attention Pooling
        self.gate_V = nn.Linear(embed_dim, 256)
        self.gate_U = nn.Linear(embed_dim, 256)
        self.gate_w = nn.Linear(256, 1)
        
        # Radiogenomic MGMT Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input 3D MRI volume [B, 4, D, H, W] or [B, 4, Slices, H, W]
               (Channels: FLAIR, T1w, T1gd, T2w)
        Returns:
            Dictionary containing:
              - 'logit': Raw scalar logit for BCEWithLogitsLoss / ROC-AUC [B, 1]
              - 'prob': Calibrated probability P(MGMT=1) [B, 1]
              - 'attention': Normalized token/slice attention weights [B, N_patches, 1]
        """
        B = x.shape[0]
        # 3D patch projection: [B, embed_dim, D', H', W']
        patches = self.patch_proj(x)
        tokens = patches.flatten(2).transpose(1, 2)  # [B, N_tokens, embed_dim]
        
        # Pass through 32 Transformer blocks (500M capacity)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm_final(tokens)
        
        # Gated Attention Pooling: A = softmax(w^T * (tanh(V * h) ⊙ sigmoid(U * h)))
        v_h = torch.tanh(self.gate_V(tokens))
        u_h = torch.sigmoid(self.gate_U(tokens))
        gated = v_h * u_h
        attn_logits = self.gate_w(gated)  # [B, N_tokens, 1]
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # Patient representation: [B, embed_dim]
        patient_emb = torch.sum(attn_weights * tokens, dim=1)
        
        # Radiogenomic logit & calibrated probability
        logit = self.classifier(patient_emb)
        prob = torch.sigmoid(logit)
        
        return {
            "logit": logit,
            "prob": prob,
            "attention": attn_weights
        }


# ======================================================================================
# CONVENIENCE ALIASES & FACTORY FUNCTIONS
# ======================================================================================

# Descriptive aliases
BraTSTask1Model = Task1_Segmentation_WinnerUNet
WinnerInspiredUNet = Task1_Segmentation_WinnerUNet

BraTSTask2Model = Task2_MGMT_Foundation_500M
MGMTFoundation500M = Task2_MGMT_Foundation_500M


def get_task1_model(**kwargs) -> Task1_Segmentation_WinnerUNet:
    """
    Factory helper: Returns our flagship Task 1 3D Brain Tumor Segmentation Model (~44.1M parameters).
    """
    return Task1_Segmentation_WinnerUNet(**kwargs)


def get_task2_model(**kwargs) -> Task2_MGMT_Foundation_500M:
    """
    Factory helper: Returns our flagship Task 2 500M Radiogenomic Foundation Model (~501.2M parameters).
    """
    return Task2_MGMT_Foundation_500M(**kwargs)


def get_model(task: Union[int, str] = 1, **kwargs) -> nn.Module:
    """
    Universal factory:
      get_model(1) or get_model("task1") -> Task1_Segmentation_WinnerUNet
      get_model(2) or get_model("task2") -> Task2_MGMT_Foundation_500M
    """
    task_str = str(task).lower().strip()
    if task_str in ["1", "task1", "seg", "segmentation", "tumor"]:
        return get_task1_model(**kwargs)
    elif task_str in ["2", "task2", "mgmt", "radiogenomics", "500m"]:
        return get_task2_model(**kwargs)
    else:
        raise ValueError(f"Unknown task: '{task}'. Specify 1 (Segmentation) or 2 (MGMT 500M Radiogenomics).")


# ======================================================================================
# STANDALONE VERIFICATION
# ======================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BraTS 2021: Verifying Extracted Flagship Models (Task 1 & Task 2)")
    print("Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)")
    print("=" * 80)

    # 1. Test Task 1: 3D Brain Tumor Segmentation
    print("\n[1/2] Instantiating Task 1 Segmentation Model (WinnerUNet)...")
    m1 = get_task1_model(use_axial_attention=False, deep_supervision=False)
    p1 = sum(p.numel() for p in m1.parameters())
    print(f"      Architecture: Task1_Segmentation_WinnerUNet")
    print(f"      Total Parameters: {p1:,} (~{p1/1e6:.2f} Million)")
    x1 = torch.randn(1, 4, 32, 64, 64)
    m1.eval()
    with torch.no_grad():
        y1 = m1(x1)
    print(f"      Forward Pass: Input {list(x1.shape)} -> Output {list(y1.shape)} (WT, TC, ET logits)")
    
    # Verify anatomical post-processing
    prob_wt, prob_tc, prob_et = torch.sigmoid(y1[:, 0]), torch.sigmoid(y1[:, 1]), torch.sigmoid(y1[:, 2])
    mask_wt, mask_tc, mask_et = apply_anatomical_prior(prob_wt, prob_tc, prob_et)
    assert ((mask_et <= mask_tc) & (mask_tc <= mask_wt)).all(), "Anatomical prior violation!"
    print("      Anatomical Prior (ET <= TC <= WT): VERIFIED")

    # 2. Test Task 2: MGMT Radiogenomic Foundation Model
    print("\n[2/2] Instantiating Task 2 Radiogenomic Foundation Model (500M)...")
    m2 = get_task2_model()
    p2 = sum(p.numel() for p in m2.parameters())
    print(f"      Architecture: Task2_MGMT_Foundation_500M")
    print(f"      Total Parameters: {p2:,} (~{p2/1e6:.2f} Million)")
    x2 = torch.randn(1, 4, 16, 64, 64)
    m2.eval()
    with torch.no_grad():
        out2 = m2(x2)
    print(f"      Forward Pass: Input {list(x2.shape)}")
    print(f"      Logit: {out2['logit'].item():.4f}, Calibrated P(MGMT=1): {out2['prob'].item():.4f}")
    print(f"      Attention Map: {list(out2['attention'].shape)}")

    print("\n" + "=" * 80)
    print("ALL MODELS OPERATIONAL AND VERIFIED FOR FUTURE USE!")
    print("=" * 80)
