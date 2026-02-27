"""
Configuration file for ML-based cell nucleus and cell wall detection system.

All parameters can be adjusted here without modifying the code.
"""

import os

# ==============================================================================
# Data Configuration
# ==============================================================================

DATA_CONFIG = {
    # Input data paths
    'czi_file': 'data/stack_oil_2.czi',  # Path to CZI file
    'dapi_channel_file': 'data/channel_2_cfw_dapi.tif',  # Path to DAPI channel (if already extracted)
    
    # Annotation data paths
    'annotation_nucleus': 'data/nucleus_annotations.png',  # Nucleus annotations (red circles/points)
    'annotation_wall': 'data/wall_annotations.png',  # Cell wall annotations (magenta outlines)
    
    # Output directory
    'output_dir': 'results/',
    
    # Image metadata
    'pixel_size_um': 0.4402,  # Pixel size in micrometers (225.36 / 512)
    'image_size': (512, 512),  # Image size in pixels (Y, X)
}

# ==============================================================================
# Preprocessing Configuration
# ==============================================================================

PREPROCESSING_CONFIG = {
    # Signal enhancement options
    'apply_enhancement': True,  # Whether to apply signal enhancement
    'method': 'manual',  # 'manual', 'adaptive', or 'none'
    
    # Manual enhancement parameters (used when method='manual')
    'intensity_min': 8,  # Minimum intensity threshold
    'intensity_max': 53,  # Maximum intensity threshold
    
    # Adaptive enhancement parameters (used when method='adaptive')
    'adaptive_method': 'otsu',  # 'otsu', 'li', or 'yen'
    
    # Z-stack processing
    'z_range': None,  # (z_min, z_max) or None for all slices
}

# ==============================================================================
# Annotation Extraction Configuration
# ==============================================================================

ANNOTATION_CONFIG = {
    # Nucleus annotation extraction
    'nucleus_color_threshold': {
        'r_min': 100,  # Red channel minimum
        'g_max': 80,   # Green channel maximum
        'b_max': 80,   # Blue channel maximum
    },
    
    # Cell wall annotation extraction
    'wall_color_threshold': {
        'r_min': 80,   # Red channel minimum
        'g_max': 80,   # Green channel maximum
        'b_min': 80,   # Blue channel minimum
    },
    
    # Minimum area for valid regions (in pixels)
    'min_nucleus_area': 10,
    'min_wall_area': 50,
    
    # Expected sample counts
    'min_nucleus_samples': 50,  # Minimum number of nucleus annotations
    'min_wall_samples': 50,     # Minimum number of cell wall annotations
}

# ==============================================================================
# Size Learning Configuration
# ==============================================================================

SIZE_LEARNING_CONFIG = {
    # Whether to learn size distribution from annotations
    'learn_from_annotations': True,
    
    # Tolerance for size filtering (as a fraction of mean)
    'nucleus_diameter_tolerance': 0.2,  # ±20%
    'cell_size_tolerance': 0.3,          # ±30%
    
    # Whether to use learned sizes for filtering candidates
    'use_size_filter': True,
    
    # Manual size constraints (used if learn_from_annotations=False)
    'nucleus_diameter_range_um': (5.0, 8.0),  # (min, max) in micrometers
    'cell_area_range_um2': (100.0, 400.0),    # (min, max) in square micrometers
}

# ==============================================================================
# Feature Extraction Configuration
# ==============================================================================

FEATURE_CONFIG = {
    # Geometric features
    'extract_geometric': True,
    
    # Intensity features
    'extract_intensity': True,
    
    # Texture features
    'extract_texture': True,
    'texture_lbp_radius': 3,
    'texture_lbp_points': 24,
    
    # Position features
    'extract_position': True,
    
    # Size deviation features (relative to learned sizes)
    'extract_size_deviation': True,
}

# ==============================================================================
# Model Configuration
# ==============================================================================

MODEL_CONFIG = {
    # Models to train and compare
    'models_to_train': [
        'random_forest',
        'xgboost',
        'svm',
        'logistic_regression',
        'gradient_boosting',
        'knn',
        'mlp',
    ],
    
    # Ensemble learning
    'use_ensemble': True,
    'ensemble_method': 'voting',  # 'voting' or 'stacking'
    
    # Model selection
    'auto_select_best': True,  # Automatically select best model based on F1 score
    'selection_metric': 'f1',  # 'accuracy', 'precision', 'recall', 'f1', or 'roc_auc'
    
    # Cross-validation
    'cross_validation_folds': 5,
    'random_state': 42,
    
    # Hyperparameters for each model
    'hyperparameters': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'random_state': 42,
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
        },
        'svm': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42,
        },
        'logistic_regression': {
            'C': 1.0,
            'penalty': 'l2',
            'random_state': 42,
            'max_iter': 1000,
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42,
        },
        'knn': {
            'n_neighbors': 5,
            'weights': 'distance',
        },
        'mlp': {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'learning_rate_init': 0.001,
            'random_state': 42,
            'max_iter': 1000,
        },
    },
}

# ==============================================================================
# Detection Configuration
# ==============================================================================

DETECTION_CONFIG = {
    # Nucleus detection
    'nucleus_detection_method': 'log',  # 'log', 'watershed', or 'local_max'
    'nucleus_log_threshold': 0.01,
    'nucleus_classification_threshold': 0.15,  # Probability threshold for classification
    
    # Cell wall detection
    'wall_detection_method': 'threshold',  # 'threshold', 'edge', or 'watershed'
    'wall_threshold_percentile': 85,
    'wall_classification_threshold': 0.15,
    
    # Post-processing
    'nms_threshold': 0.3,  # Non-maximum suppression threshold (IoU)
    'apply_nms': True,
    
    # Size filtering (uses learned sizes if available)
    'apply_size_filter': True,
}

# ==============================================================================
# Association Configuration
# ==============================================================================

ASSOCIATION_CONFIG = {
    # Nucleus-cell association method
    'method': 'spatial',  # 'spatial' or 'intensity'
    
    # Maximum distance for association (in micrometers)
    'max_distance_um': 10.0,
    
    # Whether to enforce one-to-one matching
    'one_to_one': True,
    
    # Handle orphan nuclei (nuclei without cells)
    'keep_orphan_nuclei': True,
}

# ==============================================================================
# Visualization Configuration
# ==============================================================================

VISUALIZATION_CONFIG = {
    # 3D visualization
    'generate_3d': True,
    'nucleus_color': 'green',
    'wall_color': 'blue',
    'rna_color': 'red',
    
    # 2D slice visualization
    'generate_2d_slices': True,
    'slices_to_visualize': [10, 15, 20, 25, 30],  # Z slices to visualize
    
    # Figure settings
    'figure_dpi': 150,
    'figure_size': (20, 10),
}

# ==============================================================================
# Export Configuration
# ==============================================================================

EXPORT_CONFIG = {
    # Export formats
    'export_json': True,
    'export_csv': True,
    'export_tiff': True,
    
    # JSON export options
    'json_indent': 2,
    
    # TIFF export options
    'save_masks': True,  # Save nucleus and wall masks
    'save_overlay': True,  # Save overlay image
}

# ==============================================================================
# Logging Configuration
# ==============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'results/pipeline.log',
}

# ==============================================================================
# Helper Functions
# ==============================================================================

def create_output_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(DATA_CONFIG['output_dir'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('results/exports', exist_ok=True)

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Check if files exist
    if not os.path.exists(DATA_CONFIG['czi_file']) and not os.path.exists(DATA_CONFIG['dapi_channel_file']):
        errors.append("Neither CZI file nor DAPI channel file exists")
    
    # Check preprocessing method
    if PREPROCESSING_CONFIG['method'] not in ['manual', 'adaptive', 'none']:
        errors.append(f"Invalid preprocessing method: {PREPROCESSING_CONFIG['method']}")
    
    # Check model names
    valid_models = ['random_forest', 'xgboost', 'svm', 'logistic_regression', 
                    'gradient_boosting', 'knn', 'mlp']
    for model in MODEL_CONFIG['models_to_train']:
        if model not in valid_models:
            errors.append(f"Invalid model name: {model}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

if __name__ == '__main__':
    # Validate configuration
    try:
        validate_config()
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    
    # Create output directories
    create_output_dirs()
    print("✓ Output directories created")
