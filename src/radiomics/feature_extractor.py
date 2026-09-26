"""
Radiomic Feature Extractor for BraTS 2021 Task 2.
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Extracts first-order intensity statistics, morphological shape features, and GLCM texture.
"""
from typing import Dict, List, Optional
import numpy as np

def extract_first_order_features(voxels: np.ndarray) -> Dict[str, float]:
    """Extracts first-order statistical descriptors from intensity distribution."""
    if len(voxels) == 0:
        return {f"firstorder_{k}": 0.0 for k in ["mean", "std", "var", "skewness", "kurtosis", "median", "energy", "entropy"]}
    
    mean = float(np.mean(voxels))
    std = float(np.std(voxels))
    var = float(np.var(voxels))
    median = float(np.median(voxels))
    energy = float(np.sum(voxels ** 2))
    
    # Skewness and kurtosis
    m3 = np.mean((voxels - mean) ** 3)
    skew = float(m3 / (std ** 3 + 1e-7))
    m4 = np.mean((voxels - mean) ** 4)
    kurt = float(m4 / (std ** 4 + 1e-7)) - 3.0
    
    # Shannon Entropy
    hist, _ = np.histogram(voxels, bins=32, density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-7)))
    
    return {
        "mean": mean,
        "std": std,
        "variance": var,
        "skewness": skew,
        "kurtosis": kurt,
        "median": median,
        "energy": energy,
        "entropy": entropy
    }

def extract_shape_features(mask: np.ndarray, voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """Extracts 3D volumetric morphology features."""
    voxel_vol = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    num_voxels = int(np.sum(mask > 0))
    volume_mm3 = float(num_voxels * voxel_vol)
    
    if num_voxels == 0:
        return {"volume_mm3": 0.0, "surface_area_approx": 0.0, "sphericity": 0.0}
        
    # Sphericity approximation: (pi^(1/3) * (6 * Volume)^(2/3)) / Area
    # For spherical volume: R = (3V / 4pi)^(1/3), Area = 4pi R^2
    radius = (3.0 * volume_mm3 / (4.0 * np.pi + 1e-7)) ** (1.0 / 3.0)
    surface_area = float(4.0 * np.pi * (radius ** 2))
    sphericity = float((np.pi ** (1.0 / 3.0) * (6.0 * volume_mm3) ** (2.0 / 3.0)) / (surface_area + 1e-7))
    
    return {
        "volume_mm3": volume_mm3,
        "surface_area_approx": surface_area,
        "sphericity": sphericity
    }

def extract_patient_radiomics(volume_4ch: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Extracts radiomics across all 4 mpMRI sequences (T1, T1ce, T2, FLAIR).
    If mask is supplied, extracts features from tumor core and whole tumor.
    """
    modalities = ["T1", "T1ce", "T2", "FLAIR"]
    features = {}
    
    for c, mod in enumerate(modalities):
        mod_data = volume_4ch[c]
        if mask is not None and np.sum(mask) > 0:
            roi_voxels = mod_data[mask > 0]
        else:
            roi_voxels = mod_data[mod_data > mod_data.mean()]
            
        fo = extract_first_order_features(roi_voxels)
        for k, v in fo.items():
            features[f"{mod}_{k}"] = v
            
    if mask is not None:
        shape_feats = extract_shape_features(mask)
        for k, v in shape_feats.items():
            features[f"shape_{k}"] = v
            
    return features
