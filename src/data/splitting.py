"""
Patient-Level Stratified Group K-Fold Splitting for BraTS 2021 Task 2.
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Zero data leakage guaranteed: Train ∩ Val = Empty.
"""
from typing import Dict, List, Tuple
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

def create_patient_folds(
    patient_ids: List[str],
    labels: List[int],
    n_splits: int = 5,
    seed: int = 42,
    output_dir: Path = None
) -> List[Dict[str, List[str]]]:
    """
    Splits subjects strictly at the patient level while preserving MGMT class balance.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    
    patient_arr = np.array(patient_ids)
    label_arr = np.array(labels)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_arr, label_arr)):
        train_pts = patient_arr[train_idx].tolist()
        val_pts = patient_arr[val_idx].tolist()
        
        # Verify zero leakage
        overlap = set(train_pts).intersection(set(val_pts))
        assert len(overlap) == 0, f"Critical Data Leakage in Fold {fold_idx}: {overlap}"
        
        fold_dict = {
            "fold": fold_idx,
            "train": train_pts,
            "val": val_pts,
            "train_prevalence": float(np.mean(label_arr[train_idx])),
            "val_prevalence": float(np.mean(label_arr[val_idx]))
        }
        folds.append(fold_dict)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"fold_{fold_idx}.json", "w") as f:
                json.dump(fold_dict, f, indent=2)
                
    return folds
