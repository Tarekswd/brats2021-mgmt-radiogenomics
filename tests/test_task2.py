"""
Unit Tests for BraTS 2021 Task 2 (MGMT Radiogenomics).
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Contact: tarek.ahmadieh2006@gmail.com
"""
import sys
from pathlib import Path
import unittest
import torch
import numpy as np

# Add src to path
t2_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(t2_root))

from src.data.dataset import MGMTDataset
from src.data.splitting import create_patient_folds
from src.models.cnn_lstm import CNNLSTMClassifier
from src.models.attention_mil import GatedAttentionMIL
from src.models.efficientnet_modality import ModalitySpecificEnsemble
from src.models.mgmt_500m import MGMTFoundation500M
from src.radiomics.feature_extractor import extract_patient_radiomics
from src.evaluation.metrics import compute_mgmt_metrics

class TestTask2Pipeline(unittest.TestCase):
    def test_01_dataset_and_shapes(self):
        ds = MGMTDataset(patient_ids=["00001", "00002"], data_root="mock_data", num_slices=16, slice_size=(64, 64))
        item = ds[0]
        self.assertEqual(item["volume"].shape, (4, 16, 64, 64))
        self.assertEqual(item["patient_id"], "00001")
        print("[PASS] Task 2 Dataset loading verified.")

    def test_02_zero_leakage_splitting(self):
        pids = [f"{i:05d}" for i in range(20)]
        labels = [i % 2 for i in range(20)]
        folds = create_patient_folds(pids, labels, n_splits=5)
        for f in folds:
            inter = set(f["train"]).intersection(set(f["val"]))
            self.assertEqual(len(inter), 0)
        print("[PASS] Patient-level zero leakage audit verified.")

    def test_03_cnn_lstm(self):
        model = CNNLSTMClassifier(in_channels=4, embed_dim=64, hidden_dim=32, num_layers=1)
        dummy_x = torch.randn(2, 4, 8, 32, 32)
        out = model(dummy_x)
        self.assertEqual(out.shape, (2,))
        print("[PASS] CNN-LSTM forward pass verified.")

    def test_04_attention_mil(self):
        model = GatedAttentionMIL(in_channels=4, embed_dim=64, attn_dim=32)
        dummy_x = torch.randn(2, 4, 8, 32, 32)
        logits, weights = model(dummy_x)
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(weights.shape, (2, 8))
        print("[PASS] Attention MIL forward pass verified.")

    def test_05_efficientnet_modality(self):
        model = ModalitySpecificEnsemble(embed_dim=32)
        dummy_x = torch.randn(2, 4, 8, 32, 32)
        out = model(dummy_x)
        self.assertIn("meta_logit", out)
        self.assertEqual(out["meta_logit"].shape, (2,))
        print("[PASS] 4-Modality Ensemble forward pass verified.")

    def test_06_radiomics_extraction(self):
        dummy_vol = np.random.randn(4, 16, 32, 32).astype(np.float32)
        feats = extract_patient_radiomics(dummy_vol)
        self.assertIn("T1ce_mean", feats)
        self.assertIn("FLAIR_entropy", feats)
        print("[PASS] Radiomics feature extraction verified.")

    def test_07_metrics(self):
        y_true = [0, 1, 0, 1, 1]
        y_prob = [0.1, 0.9, 0.2, 0.8, 0.7]
        m = compute_mgmt_metrics(y_true, y_prob)
        self.assertGreaterEqual(m["roc_auc"], 0.9)
        self.assertEqual(m["accuracy"], 1.0)
        print("[PASS] Evaluation metrics computation verified.")

    def test_08_mgmt_500m_model_parameters(self):
        model = MGMTFoundation500M.build_500m()
        params = sum(p.numel() for p in model.parameters())
        self.assertGreater(params, 490_000_000)
        self.assertLess(params, 520_000_000)
        print(f"[PASS] Dedicated Task 2 500M Model verified: {params/1e6:.2f} Million Parameters!")

if __name__ == "__main__":
    unittest.main()
