#!/usr/bin/env python3
"""Visualize detection results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path

print("Loading results...")
with open('results/detected_nuclei.json') as f:
    nuclei = json.load(f)

print(f"Found {len(nuclei)} detected nuclei")

# Load DAPI image
dapi = io.imread('data/channel_2_cfw_dapi.tif')

# Visualize Z=20 slice
z = 20
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Original image
axes[0].imshow(dapi[z], cmap='gray')
axes[0].set_title(f'DAPI Signal (Z={z})')
axes[0].axis('off')

# With detections
axes[1].imshow(dapi[z], cmap='gray')
for nucleus in nuclei:
    if abs(nucleus['z'] - z) <= 2:
        y, x = nucleus['y'], nucleus['x']
        radius = nucleus['diameter_px'] / 2
        circle = plt.Circle((x, y), radius, color='lime', fill=False, linewidth=2)
        axes[1].add_patch(circle)
        axes[1].plot(x, y, 'r+', markersize=10)

axes[1].set_title(f'Detected Nuclei (Z={z}, n={len([n for n in nuclei if abs(n["z"]-z)<=2])})')
axes[1].axis('off')

plt.tight_layout()
output_file = 'results/detection_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {output_file}")
plt.close()

print("Done!")
