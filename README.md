# BraTS 2021 Task 2: MGMT Promoter Methylation Radiogenomic Classification
## 500-Million-Parameter Vision Transformer & Multi-Parametric MRI Radiogenomics Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status: Not yet trained on real data](https://img.shields.io/badge/Results-pending%20real--data%20training-lightgrey.svg)](CLOUD_TRAINING.md)

**Author & Lead Researcher:** Tarek Al Ahmadieh  
**Email:** tarek.ahmadieh2006@gmail.com  
**Challenge:** RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) 2021 — Task 2 Radiogenomic Classification  
**Target:** Non-invasive, pre-operative prediction of $O^6$-Methylguanine-DNA Methyltransferase (**MGMT**) promoter methylation status in glioblastoma patients using multi-parametric MRI (mpMRI: T1, T1ce, T2, FLAIR).

> **Project status: no model has been trained or evaluated on the real BraTS 2021 / RSNA-MICCAI MGMT data yet.**
> Every run so far used synthetic noise volumes with random labels, only to check that the code runs.
> All metric tables, ROC and calibration figures, confusion matrices and `research_report.*` files in this repository are **illustrative placeholders, not measured results**.
> Real numbers will be added after the cloud GPU runs described in [CLOUD_TRAINING.md](CLOUD_TRAINING.md).

---

## Highlights & Innovations

1. **500-Million-Parameter Foundation Vision Transformer (`MGMTFoundation500M`)**:
   - 32 Deep Transformer Attention Blocks ($D=1024, H=16, d_{mlp}=5560$).
   - 3D Multi-Parametric Patch Embedding Projection capturing cross-sequence correlations across T1, T1ce, T2, and FLAIR.
   - Gated Attention Pooling mechanism for interpretable, slice-wise patient aggregation.
   - Implemented, not yet trained. The parameter count is computed by `tests/test_task2.py`.
2. **Results**: pending. No validation metrics exist yet; see the status note above.
3. **Rigorous Patient-Level Zero-Leakage Cross-Validation**:
   - 5-fold stratified cross-validation at patient ID level (`src/data/splitting.py`), which asserts that no patient appears in both train and validation.
4. **Implemented baselines (not yet benchmarked)**:
   - Standard 2D Slice Baselines, 3D Multi-Branch CNNs, CNN-LSTM (2nd Place entry), Gated Attention MIL, Modality-Specific EfficientNets (3rd Place entry), High-Order Radiomics (GLCM + Shape Morphology), and the 500M Foundation Transformer.

---

## Benchmark Comparison Table

**No measured results yet.** After the real-data runs, this table will list the mean ± std validation ROC-AUC across the 5 patient-level folds, taken from `experiments/<run>/fold*/training_log.csv`.

| Model | ROC-AUC (5-fold mean ± std) | Accuracy | Data |
|:---|:---:|:---:|:---|
| Gated-Attention MIL | pending | pending | RSNA-MICCAI labelled train set, minus 3 excluded IDs |
| CNN-LSTM | pending | pending | same folds |

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
