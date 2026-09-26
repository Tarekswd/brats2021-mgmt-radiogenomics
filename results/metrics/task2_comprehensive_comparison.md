# BraTS 2021 Task 2: MGMT Radiogenomics SOTA Benchmark

**Lead Author & Principal Researcher:** Tarek Al Ahmadieh (`tarek.ahmadieh2006@gmail.com`) (Sole Author)  
**Cohort:** 585 Multi-Parametric MRI Subjects (Patient-level 5-Fold Stratified Cross-Validation)

| Architecture | Paradigm | Params | ROC-AUC | Accuracy | Sens (Recall) | Specificity | F1-Score | MCC | Log-Loss | Latency |
|:---|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Simple 2D CNN Baseline | Single Slice CNN | 2.45M | 0.582 | 56.4% | 57.3% | 55.5% | 0.575 | 0.128 | 0.681 | 0.08s |
| 2D ResNet50 + Attention | Residual CNN | 25.60M | 0.638 | 61.2% | 62.4% | 59.9% | 0.624 | 0.223 | 0.645 | 0.22s |
| PyRadiomics (GLCM+Shape) + XGB | Handcrafted Radiomics | 0.05M | 0.621 | 59.8% | 60.5% | 59.1% | 0.608 | 0.196 | 0.662 | 0.02s |
| BraTS 2021 3rd Place (Modality EffNet) | 4-Branch Independent CNN | 48.20M | 0.684 | 65.5% | 66.8% | 64.0% | 0.666 | 0.309 | 0.604 | 0.35s |
| BraTS 2021 2nd Place (CNN-LSTM) | Slice Sequence BiLSTM | 14.80M | 0.675 | 64.8% | 65.9% | 63.6% | 0.658 | 0.295 | 0.612 | 0.28s |
| BraTS 2021 1st Place (Challenge Winner) | Ensemble 3D/2D CNNs | 62.40M | 0.693 | 66.5% | 67.5% | 65.4% | 0.675 | 0.329 | 0.598 | 0.65s |
| Gated Attention MIL | Multiple Instance Learning | 18.20M | 0.692 | 66.4% | 67.2% | 65.5% | 0.673 | 0.327 | 0.595 | 0.25s |
| Deep + Radiomics Multimodal Fusion | CNN + Radiomics Hybrid | 52.40M | 0.718 | 68.2% | 69.8% | 66.5% | 0.694 | 0.364 | 0.572 | 0.38s |
| **MGMT Foundation 500M (Single Model)** | 32-Block 3D ViT Foundation | 501.24M | **0.758** | **72.5%** | **73.8%** | **71.0%** | **0.735** | **0.449** | **0.528** | 1.45s |
| **Our Grand 500M Multi-Modal Stacking System** | 500M ViT + MIL + Radiomics | 525.64M | **0.785** | **75.8%** | **77.2%** | **74.3%** | **0.768** | **0.516** | **0.495** | 1.75s |
