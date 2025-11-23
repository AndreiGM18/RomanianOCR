#!/usr/bin/env python3
"""Visualize RecAug effects on training images."""

import sys
sys.path.insert(0, 'PaddleOCR')

from ppocr.data import create_operators
import cv2
import numpy as np
from pathlib import Path

# Read a training sample
with open('data/document_pages/train_lines/train.txt', 'r', encoding='utf-8') as f:
    line = f.readline().strip()
    img_path, label = line.split('\t')

img_path = Path('data/document_pages/train_lines') / img_path
img = cv2.imread(str(img_path))

print(f"Image shape: {img.shape}")
cv2.imwrite('aug_sample_0_original.jpg', img)

# Create RecAug operator (without DecodeImage since we already have numpy array)
from ppocr.data.imaug.rec_img_aug import RecAug

aug_op = RecAug()

# Apply augmentation 5 times to see variations
for i in range(5):
    data = {'image': img.copy(), 'label': label}

    # Apply augmentation
    try:
        data = aug_op(data)
        if data:
            aug_img = data['image']
            print(f"Aug {i+1}: shape {aug_img.shape}")
            cv2.imwrite(f'aug_sample_{i+1}.jpg', aug_img)
    except Exception as e:
        print(f"Aug {i+1}: FAILED - {e}")

print("\nSaved aug_sample_1.jpg through aug_sample_5.jpg")
print("Check if text is still readable in these images")
