"""Example configuration for different scenarios."""

# Scenario 1: Plant cells with weak DAPI signal
PLANT_WEAK_DAPI = {
    'PREPROCESSING_CONFIG': {
        'apply_enhancement': True,
        'method': 'manual',
        'intensity_min': 8,
        'intensity_max': 53,
    },
    'SIZE_LEARNING_CONFIG': {
        'nucleus_diameter_range_um': (5.0, 8.0),
    },
    'DETECTION_CONFIG': {
        'nucleus_classification_threshold': 0.15,
    },
}

# Scenario 2: Animal cells with strong DAPI signal
ANIMAL_STRONG_DAPI = {
    'PREPROCESSING_CONFIG': {
        'apply_enhancement': False,
        'method': 'none',
    },
    'SIZE_LEARNING_CONFIG': {
        'nucleus_diameter_range_um': (8.0, 15.0),
    },
    'DETECTION_CONFIG': {
        'nucleus_classification_threshold': 0.20,
    },
}

# Scenario 3: High-resolution imaging
HIGH_RESOLUTION = {
    'DATA_CONFIG': {
        'pixel_size_um': 0.2,  # Smaller pixels
    },
    'SIZE_LEARNING_CONFIG': {
        'nucleus_diameter_range_um': (5.0, 8.0),
    },
    'DETECTION_CONFIG': {
        'nucleus_log_threshold': 0.005,  # More sensitive
    },
}
