#!/bin/bash
# Complete ML-based cell detection pipeline

echo "========================================="
echo "ML Cell Detection Pipeline"
echo "========================================="

# Check if data files exist
if [ ! -f "data/channel_2_cfw_dapi.tif" ]; then
    echo "Error: DAPI channel file not found!"
    echo "Please place your data in the data/ directory"
    exit 1
fi

# Run the complete pipeline
python3 train_and_detect.py

echo ""
echo "Pipeline completed!"
