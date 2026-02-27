# Usage Guide

## Quick Start

### 1. Prepare Your Data

Place your files in the `data/` directory:
```
data/
├── channel_2_cfw_dapi.tif       # DAPI channel (required)
├── nucleus_annotations.png       # Nucleus annotations (required)
└── wall_annotations.png          # Cell wall annotations (optional)
```

### 2. Edit Configuration

Edit `config.py` to match your data:

```python
DATA_CONFIG = {
    'dapi_channel_file': 'data/channel_2_cfw_dapi.tif',
    'annotation_nucleus': 'data/nucleus_annotations.png',
    'pixel_size_um': 0.4402,  # Your pixel size
}
```

### 3. Run Pipeline

```bash
python3 train_and_detect.py
```

Or use the shell script:
```bash
./run_pipeline.sh
```

### 4. Visualize Results

```bash
python3 visualize_results.py
```

## Configuration Options

### Signal Enhancement

Control how DAPI signal is preprocessed:

```python
PREPROCESSING_CONFIG = {
    'apply_enhancement': True,    # Enable/disable
    'method': 'manual',           # 'manual', 'adaptive', 'none'
    'intensity_min': 8,           # For manual method
    'intensity_max': 53,
}
```

### Model Selection

Choose which models to train:

```python
MODEL_CONFIG = {
    'models_to_train': [
        'random_forest',
        'xgboost',
        'svm',
    ],
    'auto_select_best': True,
}
```

### Detection Threshold

Adjust sensitivity:

```python
DETECTION_CONFIG = {
    'nucleus_classification_threshold': 0.15,  # Lower = more detections
}
```

## Output Files

- `results/detected_nuclei.json` - Full detection results
- `results/detected_nuclei.csv` - Tabular format
- `models/best_nucleus_detector.pkl` - Trained model
- `results/detection_visualization.png` - Visualization

## Troubleshooting

### Too few detections

- Lower `nucleus_classification_threshold` (e.g., 0.10)
- Check annotation quality
- Adjust size range in config

### Too many false positives

- Raise `nucleus_classification_threshold` (e.g., 0.25)
- Enable size filtering
- Add more negative samples

### Poor model performance

- Increase annotation samples (50+ recommended)
- Try different models
- Check preprocessing settings

## Advanced Usage

### Custom Feature Extraction

Edit `extract_features()` function in `train_and_detect.py`

### Model Hyperparameter Tuning

Edit `MODEL_CONFIG['hyperparameters']` in `config.py`

### Ensemble Learning

```python
MODEL_CONFIG = {
    'use_ensemble': True,
    'ensemble_method': 'voting',  # or 'stacking'
}
```
