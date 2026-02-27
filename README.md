# ML-Based Cell Nucleus and Cell Wall Detection System

A flexible machine learning-based system for detecting cell nuclei and cell walls in 3D microscopy images (DAPI + CFW channels).

## Features

- ✨ **Flexible Signal Enhancement**: Optional intensity adjustment with manual, adaptive, or no enhancement
- 📊 **Size Learning**: Automatically learns size distributions from user annotations
- 🤖 **Multiple ML Models**: Supports 7 different machine learning models with automatic comparison
- 🎯 **High Accuracy**: Ensemble learning and model selection for optimal performance
- 📈 **Comprehensive Evaluation**: Cross-validation, feature importance, and performance metrics
- 🎨 **Rich Visualization**: 2D slices and 3D rendering of detection results

## Supported Models

1. **Random Forest** - Fast and stable
2. **XGBoost** - High accuracy
3. **SVM** - Good for small samples
4. **Logistic Regression** - Simple and fast
5. **Gradient Boosting** - High accuracy
6. **K-Nearest Neighbors** - No training required
7. **Multi-Layer Perceptron** - Deep learning approach

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-cell-detection.git
cd ml-cell-detection

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

- **Input**: CZI file or extracted DAPI channel (TIFF)
- **Annotations**: 
  - Nucleus annotations (50+ samples, red circles/points)
  - Cell wall annotations (50+ cells, magenta outlines)

### 2. Configure Parameters

Edit `config.py` to set your data paths and parameters:

```python
DATA_CONFIG = {
    'czi_file': 'data/your_image.czi',
    'annotation_nucleus': 'data/nucleus_annotations.png',
    'annotation_wall': 'data/wall_annotations.png',
}

PREPROCESSING_CONFIG = {
    'apply_enhancement': True,
    'method': 'manual',  # 'manual', 'adaptive', or 'none'
    'intensity_min': 8,
    'intensity_max': 53,
}

MODEL_CONFIG = {
    'models_to_train': [
        'random_forest',
        'xgboost',
        'svm',
    ],
    'auto_select_best': True,
}
```

### 3. Run the Pipeline

```bash
# Option 1: Run complete pipeline
bash run_pipeline.sh

# Option 2: Run step by step
python 01_preprocess.py
python 02_extract_annotations.py
python 03_learn_sizes.py
python 04_extract_features.py
python 05_train_models.py
python 06_select_best_model.py
python 07_detect_nucleus.py
python 08_detect_wall.py
python 09_associate.py
python 10_visualize.py
python 11_export.py
```

## Project Structure

```
ml_cell_detection/
├── config.py                      # Configuration file (edit this!)
├── 01_preprocess.py               # Data preprocessing
├── 02_extract_annotations.py     # Extract user annotations
├── 03_learn_sizes.py              # Learn size distributions
├── 04_extract_features.py         # Feature extraction
├── 05_train_models.py             # Train all models and compare
├── 06_select_best_model.py        # Select best model
├── 07_detect_nucleus.py           # Detect nuclei
├── 08_detect_wall.py              # Detect cell walls
├── 09_associate.py                # Associate nuclei with cells
├── 10_visualize.py                # Visualize results
├── 11_export.py                   # Export results
├── run_pipeline.sh                # Run complete pipeline
├── utils/                         # Utility modules
│   ├── io_utils.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── model_comparison.py
│   └── visualization.py
├── models/                        # Saved models
├── results/                       # Output results
├── examples/                      # Example data and notebooks
└── docs/                          # Documentation

```

## Configuration Options

### Signal Enhancement

```python
PREPROCESSING_CONFIG = {
    'apply_enhancement': True,      # Enable/disable enhancement
    'method': 'manual',             # 'manual', 'adaptive', or 'none'
    'intensity_min': 8,             # Min intensity (manual mode)
    'intensity_max': 53,            # Max intensity (manual mode)
}
```

### Size Learning

```python
SIZE_LEARNING_CONFIG = {
    'learn_from_annotations': True,           # Learn from annotations
    'nucleus_diameter_tolerance': 0.2,        # ±20% tolerance
    'cell_size_tolerance': 0.3,               # ±30% tolerance
    'use_size_filter': True,                  # Apply size filtering
}
```

### Model Selection

```python
MODEL_CONFIG = {
    'models_to_train': [
        'random_forest',
        'xgboost',
        'svm',
    ],
    'use_ensemble': True,                     # Use ensemble learning
    'ensemble_method': 'voting',              # 'voting' or 'stacking'
    'auto_select_best': True,                 # Auto-select best model
    'selection_metric': 'f1',                 # Selection metric
}
```

## Output

### JSON Export
```json
{
  "nuclei": [
    {
      "id": 1,
      "centroid": [256, 256, 20],
      "diameter_um": 6.5,
      "volume_um3": 143.8,
      "intensity_mean": 75.2,
      "cell_id": 1
    }
  ],
  "cells": [
    {
      "id": 1,
      "area_um2": 250.5,
      "perimeter_um": 65.3,
      "nucleus_count": 1
    }
  ]
}
```

### CSV Export
- `nuclei.csv`: Nucleus properties
- `cells.csv`: Cell properties
- `associations.csv`: Nucleus-cell associations

### TIFF Export
- `nucleus_mask.tif`: Binary mask of detected nuclei
- `wall_mask.tif`: Binary mask of detected cell walls
- `overlay.tif`: Overlay visualization

## Model Comparison

The system automatically trains all selected models and outputs a comparison table:

```
Model                  Accuracy  Precision  Recall    F1 Score  ROC AUC   Train Time  Predict Time
---------------------------------------------------------------------------------------------------
XGBoost                0.92      0.90       0.94      0.92      0.96      2.3s        0.1s
Random Forest          0.90      0.88       0.92      0.90      0.94      1.8s        0.2s
Gradient Boosting      0.91      0.89       0.93      0.91      0.95      3.1s        0.1s
SVM                    0.88      0.86       0.90      0.88      0.92      4.5s        0.3s
Logistic Regression    0.85      0.83       0.87      0.85      0.89      0.5s        0.05s
KNN                    0.87      0.85       0.89      0.87      0.91      0.1s        0.4s
MLP                    0.89      0.87       0.91      0.89      0.93      5.2s        0.1s
---------------------------------------------------------------------------------------------------
Recommended: XGBoost (F1 Score: 0.92)
```

## Examples

See `examples/` directory for:
- Sample annotation images
- Jupyter notebooks with tutorials
- Example configuration files

## Troubleshooting

### Issue: Not enough annotations detected

**Solution**: Adjust color thresholds in `config.py`:

```python
ANNOTATION_CONFIG = {
    'nucleus_color_threshold': {
        'r_min': 80,  # Lower threshold
        'g_max': 100,  # Raise threshold
        'b_max': 100,
    },
}
```

### Issue: Poor detection accuracy

**Solutions**:
1. Increase annotation samples (100+ recommended)
2. Try different models
3. Enable ensemble learning
4. Adjust size learning tolerance
5. Fine-tune preprocessing parameters

### Issue: Too many false positives

**Solutions**:
1. Increase classification threshold
2. Enable size filtering
3. Apply non-maximum suppression
4. Use stricter feature selection

## Citation

If you use this system in your research, please cite:

```bibtex
@software{ml_cell_detection,
  title={ML-Based Cell Nucleus and Cell Wall Detection System},
  author={Zhihe Cai},
  year={2024},
  url={https://github.com/IceBear321/ml-cell-detection}
}
```

## License

MIT License - see LICENSE file for details

---

**Happy Cell Detecting! 🔬🤖**
