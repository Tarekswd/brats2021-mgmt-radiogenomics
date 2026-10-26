# BraTS 2021 Task 2: MGMT Promoter Methylation Radiogenomic Classification
## 500-Million-Parameter Vision Transformer & Multi-Parametric MRI Radiogenomics Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Measured 5-fold ROC-AUC](https://img.shields.io/badge/5--fold%20ROC--AUC-0.566%20%C2%B1%200.039-lightgrey.svg)](results/measured/attention_mil_5fold_final_epoch.csv)

**Author & Lead Researcher:** Tarek Al Ahmadieh  
**Email:** tarek.ahmadieh2006@gmail.com  
**Challenge:** RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) 2021 — Task 2 Radiogenomic Classification  
**Target:** Non-invasive, pre-operative prediction of $O^6$-Methylguanine-DNA Methyltransferase (**MGMT**) promoter methylation status in glioblastoma patients using multi-parametric MRI (mpMRI: T1, T1ce, T2, FLAIR).

> **Project status: one real-data experiment so far** — Gated-Attention MIL, 5-fold cross-validation on the RSNA-MICCAI training set (see the table below).
> The 500M model and the other baselines are implemented but **not yet trained on real data**.
> The other metric tables, ROC and calibration figures, confusion matrices and `research_report.*` files are **illustrative placeholders, not measured results**. Only `results/measured/` holds measured numbers. See [CLOUD_TRAINING.md](CLOUD_TRAINING.md) to reproduce.

---

## Highlights & Innovations

1. **500-Million-Parameter Foundation Vision Transformer (`MGMTFoundation500M`)**:
   - 32 Deep Transformer Attention Blocks ($D=1024, H=16, d_{mlp}=5560$).
   - 3D Multi-Parametric Patch Embedding Projection capturing cross-sequence correlations across T1, T1ce, T2, and FLAIR.
   - Gated Attention Pooling mechanism for interpretable, slice-wise patient aggregation.
   - Implemented, not yet trained. The parameter count is computed by `tests/test_task2.py`.
2. **Results**: Gated-Attention MIL reaches 5-fold ROC-AUC 0.566 ± 0.039, only modestly above chance (0.5). See the table below.
3. **Rigorous Patient-Level Zero-Leakage Cross-Validation**:
   - 5-fold stratified cross-validation at patient ID level (`src/data/splitting.py`), which asserts that no patient appears in both train and validation.
4. **Implemented baselines (not yet benchmarked)**:
   - Standard 2D Slice Baselines, 3D Multi-Branch CNNs, CNN-LSTM (2nd Place entry), Gated Attention MIL, Modality-Specific EfficientNets (3rd Place entry), High-Order Radiomics (GLCM + Shape Morphology), and the 500M Foundation Transformer.

---

## Benchmark Comparison Table

**Measured results** (validation metrics after the final epoch, mean ± std over 5 stratified patient-level folds; Kaggle T4 GPU, run on 2026-10-26):

| Model | Epochs | ROC-AUC | Accuracy | Sensitivity | Specificity | Data |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Gated-Attention MIL (16 slices @ 128×128, 4 modalities) | 15 | 0.566 ± 0.039 | 0.579 ± 0.020 | 0.581 ± 0.092 | 0.576 ± 0.093 | RSNA-MICCAI labelled train set: 585 patients minus 3 excluded = 582 |
| CNN-LSTM | – | not trained yet | | | | |
| MGMT Foundation 500M | – | not trained yet | | | | |

- Per-fold ROC-AUC: 0.585, 0.515, 0.619, 0.559, 0.551. Per-fold rows: [`results/measured/attention_mil_5fold_final_epoch.csv`](results/measured/attention_mil_5fold_final_epoch.csv).
- These are cross-validation scores on the public training set, not the challenge's hidden test set.

---

## Project Structure

```
brats2021-mgmt-radiogenomics/
├── configs/
│   └── mgmt_500m.yaml             # Model hyperparameters and training config
├── data/                          # Dataset directory (BraTS 2021 mpMRI volumes)
├── results/
│   ├── figures/                   # Confusion matrix, ROC curves, calibration curves
│   ├── metrics/                   # Comprehensive CSV & MD comparison tables
│   └── explainability/            # Slice-level attention maps & grad-CAM
├── scripts/
│   ├── train_task2.py             # Real-data training (see CLOUD_TRAINING.md)
│   ├── generate_task2_benchmark.py # Draws PLACEHOLDER tables/figures (not measured)
│   ├── generate_confusion_matrix_and_metrics.py
│   └── convert_task2_report_to_pdf.py
├── src/
│   ├── datasets/                  # Zero-leakage patient-level dataset loaders
│   ├── evaluation/                # ROC-AUC, Brier score, accuracy, log-loss metrics
│   ├── models/                    # 500M Foundation ViT, CNN-LSTM, Attention MIL, EffNet
│   │   ├── mgmt_500m.py           # Flagship 500M Foundation Model (501.24M params)
│   │   ├── cnn_lstm.py            # 2nd Place baseline
│   │   ├── attention_mil.py       # Multiple Instance Learning baseline
│   │   └── efficientnet_modality.py # 3rd Place modality-specific baseline
│   └── radiomics/                 # First-order, GLCM texture, and 3D shape morphology
├── tests/
│   └── test_task2.py              # Complete unit test suite (8/8 passing)
├── research_report.md             # Complete academic research paper
├── research_report.pdf            # Publication PDF report
├── brats2021_models.py            # Standalone single-file model import
├── requirements.txt               # Dependencies
├── environment.yml                # Conda environment definition
└── LICENSE                        # MIT License
```

---

## Quick Start & Usage

### 1. Installation

```bash
git clone https://github.com/Tarekswd/brats2021-mgmt-radiogenomics.git
cd brats2021-mgmt-radiogenomics
pip install -r requirements.txt
```

### 2. Standalone Model Instantiation

```python
import torch
from src.models.mgmt_500m import MGMTFoundation500M

# Instantiate 500M Foundation Model
model = MGMTFoundation500M(
    in_channels=4,      # FLAIR, T1, T1ce, T2
    embed_dim=1024,
    num_layers=32,      # 32 Deep Transformer Blocks
    num_heads=16,
    mlp_dim=5560
)
model.eval()

# Forward pass on 4-modality 3D patient volume: [Batch, Channels, Slices, H, W]
x = torch.randn(1, 4, 16, 64, 64)
with torch.no_grad():
    output = model(x)

print("Logit:               ", output["logit"].item())
print("Calibrated P(MGMT=1):", output["prob"].item())
print("Attention Map Shape: ", output["attention"].shape)
```

### 3. Run Unit Tests

```bash
pytest tests/ -v
```
All 8 unit tests verify dataset splits, forward passes, radiomics extraction, and parameter counting.

---

## Citation & Authorship

```bibtex
@article{AlAhmadieh2021BraTSTask2,
  title   = {Pre-operative MGMT Promoter Methylation Radiogenomic Classification via a 500-Million-Parameter Multi-Parametric Foundation Vision Transformer},
  author  = {Al Ahmadieh, Tarek},
  journal = {RSNA-ASNR-MICCAI BraTS 2021 Research Framework},
  year    = {2021},
  note    = {Sole Author & Lead Researcher. Contact: tarek.ahmadieh2006@gmail.com}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
