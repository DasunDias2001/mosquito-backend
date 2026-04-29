"""
Fixed Data Splitting Script
Ensures no overlap between train/val/test
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# Configuration
RAW_DATA_PATH = r"C:\MosquitoProj\Dataset\raw"
OUTPUT_PATH = r"C:\MosquitoProj\mosquito-classification\data\processed"
CLASSES = ['aegypti', 'albopictus']
RANDOM_SEED = 42

# Set random seed
random.seed(RANDOM_SEED)

print("🔄 Starting fixed data split...")

# Clean output directory
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

# Create directories
for split in ['train', 'val', 'test']:
    for class_name in CLASSES:
        os.makedirs(os.path.join(OUTPUT_PATH, split, class_name), exist_ok=True)

# Process each class
for class_name in CLASSES:
    print(f"\n📁 Processing {class_name}...")
    
    # Get all images
    source_dir = os.path.join(RAW_DATA_PATH, class_name)
    all_images = list(Path(source_dir).glob('*.jpg'))
    
    if len(all_images) == 0:
        print(f"   ❌ ERROR: No images found in {source_dir}")
        continue
    
    print(f"   Found {len(all_images)} images")
    
    # First split: 70% train, 30% temp
    train_files, temp_files = train_test_split(
        all_images, 
        test_size=0.30, 
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Second split: Split temp into val (50%) and test (50%)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"   Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Copy files
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        dest_dir = os.path.join(OUTPUT_PATH, split_name, class_name)
        for src_file in files:
            dest_file = os.path.join(dest_dir, src_file.name)
            shutil.copy2(src_file, dest_file)
    
    # Verify no overlaps
    train_names = set([f.name for f in train_files])
    val_names = set([f.name for f in val_files])
    test_names = set([f.name for f in test_files])
    
    overlap_train_val = train_names & val_names
    overlap_train_test = train_names & test_names
    overlap_val_test = val_names & test_names
    
    if overlap_train_val:
        print(f"   ⚠️ WARNING: {len(overlap_train_val)} overlaps between train and val!")
    if overlap_train_test:
        print(f"   ⚠️ WARNING: {len(overlap_train_test)} overlaps between train and test!")
    if overlap_val_test:
        print(f"   ⚠️ WARNING: {len(overlap_val_test)} overlaps between val and test!")
    
    if not (overlap_train_val or overlap_train_test or overlap_val_test):
        print(f"   ✅ No overlaps - Clean split!")

print("\n✅ Data split completed!")
print("\nStatistics:")
print("="*60)

for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()}:")
    total = 0
    for class_name in CLASSES:
        split_dir = os.path.join(OUTPUT_PATH, split, class_name)
        count = len(list(Path(split_dir).glob('*.jpg')))
        print(f"  {class_name}: {count} images")
        total += count
    print(f"  Total: {total} images")

print("\n" + "="*60)