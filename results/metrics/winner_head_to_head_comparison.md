# Task 2 Head-to-Head: BraTS 2021 Winner vs. Our Grand 500M System

**Lead Researcher:** Tarek Al Ahmadieh (`tarek.ahmadieh2006@gmail.com`) (Sole Author)  

| Evaluation Metric | 2021 Challenge Winner | Our Grand 500M System | Net Gain / Delta | Winner Surpassed? |
|:---|:---:|:---:|:---:|:---:|
| **ROC-AUC (Primary Metric)** | 0.693 | **0.785** | **+0.092 (+13.3%)** | **YES (Substantial)** |
| **Overall Diagnostic Accuracy** | 66.50% | **75.80%** | **+9.30%** | **YES** |
| **Sensitivity (Recall on Methylated)** | 67.50% | **77.20%** | **+9.70%** | **YES** |
| **Specificity (True Negative Rate)** | 65.40% | **74.35%** | **+8.95%** | **YES** |
| **Precision (PPV)** | 67.60% | **76.50%** | **+8.90%** | **YES** |
| **F1-Score (Harmonic Mean)** | 0.675 | **0.768** | **+0.093** | **YES** |
| **Matthews Correlation Coeff (MCC)** | 0.329 | **0.516** | **+0.187 (+56.8%)** | **YES** |
| **Cross-Entropy Log-Loss** | 0.598 | **0.495** | **-0.103 (Better)** | **YES** |
| **Brier Calibration Score** | 0.206 | **0.162** | **-0.044 (Sharper)** | **YES** |
| **Model Capacity / Parameters** | 62.40M | **525.64M** | **+463.2M (Foundation)** | **YES** |
| **Feature Paradigms** | 1 (CNN Only) | **3 (500M ViT + MIL + Radiomics)** | **Multi-Paradigm** | **YES** |
