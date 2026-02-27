#!/usr/bin/env python3
"""
ML-Based Cell Nucleus Detection System
Streamlined implementation with multi-model support
"""

import numpy as np
import json
import pickle
from pathlib import Path
from skimage import io, measure, filters, feature
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from config import *

print("=" * 80)
print("ML-Based Cell Nucleus Detection System")
print("=" * 80)

create_output_dirs()

# ============================================================================
# STEP 1: Load and Preprocess Data
# ============================================================================

print("\n[1/5] Loading and preprocessing data...")

dapi_file = DATA_CONFIG['dapi_channel_file']
if not Path(dapi_file).exists():
    print(f"Error: DAPI file not found: {dapi_file}")
    print("Please update DATA_CONFIG['dapi_channel_file'] in config.py")
    exit(1)

dapi_original = io.imread(dapi_file)
print(f"  Loaded DAPI image: {dapi_original.shape}")

if PREPROCESSING_CONFIG['apply_enhancement']:
    method = PREPROCESSING_CONFIG['method']
    print(f"  Applying {method} enhancement...")
    
    if method == 'manual':
        intensity_min = PREPROCESSING_CONFIG['intensity_min']
        intensity_max = PREPROCESSING_CONFIG['intensity_max']
        dapi_clipped = np.clip(dapi_original, intensity_min, intensity_max)
        dapi_signal = ((dapi_clipped - intensity_min) / (intensity_max - intensity_min) * 255).astype(np.uint8)
        print(f"    Clipped to [{intensity_min}, {intensity_max}] → [0, 255]")
    elif method == 'adaptive':
        threshold = filters.threshold_otsu(dapi_original)
        dapi_signal = np.clip(dapi_original, 0, threshold * 2)
        dapi_signal = (dapi_signal / dapi_signal.max() * 255).astype(np.uint8)
        print(f"    Adaptive threshold: {threshold:.2f}")
    else:
        dapi_signal = dapi_original
else:
    dapi_signal = dapi_original

print(f"  Preprocessed range: [{dapi_signal.min()}, {dapi_signal.max()}]")

# ============================================================================
# STEP 2: Extract Annotations
# ============================================================================

print("\n[2/5] Extracting annotations...")

nucleus_annot_file = DATA_CONFIG['annotation_nucleus']
if not Path(nucleus_annot_file).exists():
    print(f"Error: Nucleus annotation file not found: {nucleus_annot_file}")
    exit(1)

nucleus_annot = io.imread(nucleus_annot_file)
print(f"  Loaded nucleus annotations: {nucleus_annot.shape}")

if nucleus_annot.ndim == 3 and nucleus_annot.shape[2] >= 3:
    r, g, b = nucleus_annot[:,:,0], nucleus_annot[:,:,1], nucleus_annot[:,:,2]
    red_mask = (r > ANNOTATION_CONFIG['nucleus_color_threshold']['r_min']) & \
               (g < ANNOTATION_CONFIG['nucleus_color_threshold']['g_max']) & \
               (b < ANNOTATION_CONFIG['nucleus_color_threshold']['b_max'])
    
    labeled_nuclei = measure.label(red_mask)
    nucleus_regions = measure.regionprops(labeled_nuclei)
    
    scale_y = dapi_signal.shape[1] / nucleus_annot.shape[0]
    scale_x = dapi_signal.shape[2] / nucleus_annot.shape[1]
    
    nucleus_centers = []
    for region in nucleus_regions:
        if region.area >= ANNOTATION_CONFIG['min_nucleus_area']:
            y, x = region.centroid
            nucleus_centers.append((int(y * scale_y), int(x * scale_x)))
    
    print(f"  Extracted {len(nucleus_centers)} nucleus annotations")
else:
    print("  Error: Invalid nucleus annotation format")
    exit(1)

# ============================================================================
# STEP 3: Generate Candidates using LoG
# ============================================================================

print("\n[3/5] Generating candidate nuclei...")

pixel_size = DATA_CONFIG['pixel_size_um']
nucleus_diameter_min, nucleus_diameter_max = SIZE_LEARNING_CONFIG['nucleus_diameter_range_um']
diameter_min_px = nucleus_diameter_min / pixel_size
diameter_max_px = nucleus_diameter_max / pixel_size

sigma_min = (diameter_min_px / 2) / np.sqrt(2)
sigma_max = (diameter_max_px / 2) / np.sqrt(2)

print(f"  Size range: {nucleus_diameter_min}-{nucleus_diameter_max} µm")
print(f"  Sigma range: {sigma_min:.2f}-{sigma_max:.2f}")

all_candidates = []
for z in range(dapi_signal.shape[0]):
    blobs = feature.blob_log(dapi_signal[z], min_sigma=sigma_min, max_sigma=sigma_max, 
                             threshold=DETECTION_CONFIG['nucleus_log_threshold'])
    for blob in blobs:
        y, x, sigma = blob
        diameter = sigma * np.sqrt(2) * 2
        all_candidates.append({
            'z': z, 'y': int(y), 'x': int(x),
            'diameter_px': diameter,
            'diameter_um': diameter * pixel_size,
        })

print(f"  Found {len(all_candidates)} candidates")

# Extract features
print("  Extracting features...")

def extract_features(candidate, dapi_signal):
    z, y, x = candidate['z'], candidate['y'], candidate['x']
    radius = int(candidate['diameter_px'] / 2)
    
    z_min = max(0, z - 2)
    z_max = min(dapi_signal.shape[0], z + 3)
    y_min = max(0, y - radius)
    y_max = min(dapi_signal.shape[1], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(dapi_signal.shape[2], x + radius + 1)
    
    region_3d = dapi_signal[z_min:z_max, y_min:y_max, x_min:x_max]
    
    return {
        'diameter_px': candidate['diameter_px'],
        'mean_intensity': np.mean(region_3d),
        'max_intensity': np.max(region_3d),
        'std_intensity': np.std(region_3d),
        'center_intensity': dapi_signal[z, y, x],
        'intensity_q25': np.percentile(region_3d, 25),
        'intensity_q50': np.percentile(region_3d, 50),
        'intensity_q75': np.percentile(region_3d, 75),
        'z_position': z / dapi_signal.shape[0],
        'y_position': y / dapi_signal.shape[1],
        'x_position': x / dapi_signal.shape[2],
    }

candidate_features = [extract_features(c, dapi_signal) for c in all_candidates]
print(f"  Extracted {len(candidate_features[0])} features")

# ============================================================================
# STEP 4: Create Training Data
# ============================================================================

print("\n[4/5] Creating training data...")

X_train = []
y_train = []

for i, candidate in enumerate(all_candidates):
    features = candidate_features[i]
    feature_vector = list(features.values())
    
    is_positive = False
    for ny, nx in nucleus_centers:
        if abs(candidate['z'] - 20) <= 5:
            dist = np.sqrt((candidate['y'] - ny)**2 + (candidate['x'] - nx)**2)
            if dist < diameter_max_px:
                is_positive = True
                break
    
    X_train.append(feature_vector)
    y_train.append(1 if is_positive else 0)

X_train = np.array(X_train)
y_train = np.array(y_train)

n_positive = np.sum(y_train == 1)
n_negative = np.sum(y_train == 0)

print(f"  Training samples: {len(y_train)}")
print(f"    Positive: {n_positive}, Negative: {n_negative}")

# ============================================================================
# STEP 5: Train and Compare Models
# ============================================================================

print("\n[5/5] Training and comparing models...")

models = {}
results = []

model_definitions = {
    'random_forest': RandomForestClassifier(**MODEL_CONFIG['hyperparameters']['random_forest']),
    'gradient_boosting': GradientBoostingClassifier(**MODEL_CONFIG['hyperparameters']['gradient_boosting']),
    'svm': SVC(**MODEL_CONFIG['hyperparameters']['svm'], probability=True),
    'logistic_regression': LogisticRegression(**MODEL_CONFIG['hyperparameters']['logistic_regression']),
    'knn': KNeighborsClassifier(**MODEL_CONFIG['hyperparameters']['knn']),
    'mlp': MLPClassifier(**MODEL_CONFIG['hyperparameters']['mlp']),
}

if HAS_XGBOOST:
    model_definitions['xgboost'] = xgb.XGBClassifier(**MODEL_CONFIG['hyperparameters']['xgboost'])

for model_name in MODEL_CONFIG['models_to_train']:
    if model_name not in model_definitions:
        continue
    
    print(f"\n  Training {model_name}...")
    model = model_definitions[model_name]
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    cv_scores = cross_val_score(model, X_train, y_train, 
                                cv=min(MODEL_CONFIG['cross_validation_folds'], n_positive),
                                scoring='f1')
    
    y_pred = model.predict(X_train)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_train, y_pred),
        'Precision': precision_score(y_train, y_pred, zero_division=0),
        'Recall': recall_score(y_train, y_pred, zero_division=0),
        'F1 Score': f1_score(y_train, y_pred, zero_division=0),
        'CV F1': f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
        'Train Time (s)': f"{train_time:.2f}",
    })
    
    models[model_name] = model

print("\n  Model Comparison:")
print("  " + "=" * 100)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print("  " + "=" * 100)

best_model_name = df_results.loc[df_results['F1 Score'].idxmax(), 'Model']
best_model = models[best_model_name]
print(f"\n  ✓ Best model: {best_model_name}")

model_path = Path('models') / 'best_nucleus_detector.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"  ✓ Saved to {model_path}")

# Detect nuclei
y_pred_proba = best_model.predict_proba(X_train)[:, 1]
threshold = DETECTION_CONFIG['nucleus_classification_threshold']

detected_nuclei = []
for i, (candidate, prob) in enumerate(zip(all_candidates, y_pred_proba)):
    if prob > threshold:
        nucleus = candidate.copy()
        nucleus['probability'] = float(prob)
        nucleus['id'] = len(detected_nuclei) + 1
        detected_nuclei.append(nucleus)

print(f"\n  Detected {len(detected_nuclei)} nuclei (threshold: {threshold})")

results_file = Path('results') / 'detected_nuclei.json'
with open(results_file, 'w') as f:
    json.dump(detected_nuclei, f, indent=2)

df_nuclei = pd.DataFrame(detected_nuclei)
df_nuclei.to_csv('results/detected_nuclei.csv', index=False)

print("\n" + "=" * 80)
print("✓ Pipeline completed successfully!")
print("=" * 80)
print(f"\nResults:")
print(f"  - Detected nuclei: {len(detected_nuclei)}")
print(f"  - Best model: {best_model_name}")
print(f"  - Output: results/detected_nuclei.json")
print(f"\nVisualize: python3 visualize_results.py")
