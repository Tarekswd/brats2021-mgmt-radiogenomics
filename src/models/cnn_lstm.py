"""
CNN-LSTM Architecture for BraTS 2021 Task 2 (2nd Place Winning Strategy).
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Extracts slice-level embeddings across ordered MRI sequences and aggregates them via BiLSTM.
"""
import torch
import torch.nn as nn

class SliceFeatureExtractor(nn.Module):
    """2D Convolutional feature extractor per slice."""
    def __init__(self, in_channels: int = 4, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # /2
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # /4
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # /8
            
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CNNLSTMClassifier(nn.Module):
    """
    Combines 2D CNN slice feature extractor with a Bidirectional LSTM
    to model inter-slice anatomical progression along the z-axis.
    """
    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.slice_cnn = SliceFeatureExtractor(in_channels, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, C, S, H, W]
        B, C, S, H, W = x.shape
        # Permute and fold slices into batch dimension: [B*S, C, H, W]
        x_slices = x.permute(0, 2, 1, 3, 4).reshape(B * S, C, H, W)
        feats = self.slice_cnn(x_slices) # [B*S, embed_dim]
        # Reshape back to sequence: [B, S, embed_dim]
        seq_feats = feats.view(B, S, -1)
        # Sequence aggregation through BiLSTM
        lstm_out, (hn, cn) = self.lstm(seq_feats) # [B, S, hidden_dim*2]
        # Attention or final hidden state pooling
        patient_emb = lstm_out[:, -1, :] # [B, hidden_dim*2]
        logits = self.head(patient_emb).squeeze(-1) # [B]
        return logits
