
import os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm


def validate_dataset(dataset_path="dataset"):
    """
    Validate dataset structure and provide statistics
    
    Args:
        dataset_path: Path to dataset directory
    """
    print("\n" + "="*80)
    print("DATASET VALIDATION")
    print("="*80 + "\n")
    
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        print(f"❌ ERROR: Dataset directory not found: {dataset_path}")
        print("\nExpected structure:")
        print(f"  {dataset_path}/")
        print(f"    ├── Aedes aegypti/")
        print(f"    │   ├── abdomen/")
        print(f"    │   ├── full body/")
        print(f"    │   ├── head/")
        print(f"    │   └── siphon/")
        print(f"    └── Culex quinquefasciatus/")
        print(f"        ├── abdomen/")
        print(f"        ├── full body/")
        print(f"        ├── head/")
        print(f"        └── siphon/")
        return
    
    # Expected classes
    expected_classes = ["Aedes aegypti", "Culex quinquefasciatus"]
    
    stats = {
        'total_images': 0,
        'classes': {},
        'corrupted_images': [],
        'image_sizes': [],
        'file_formats': defaultdict(int)
    }
    
    print("Scanning dataset...\n")
    
    for class_name in expected_classes:
        class_path = dataset_dir / class_name
        
        if not class_path.exists():
            print(f"⚠️  WARNING: Class directory not found: {class_name}")
            stats['classes'][class_name] = {
                'count': 0,
                'valid': 0,
                'corrupted': 0,
                'body_parts': {}
            }
            continue
        
        print(f"Checking {class_name}...")
        
        # Check for body part subfolders
        subfolders = [f for f in class_path.iterdir() if f.is_dir()]
        
        class_stats = {
            'count': 0,
            'valid': 0,
            'corrupted': 0,
            'body_parts': {}
        }
        
        if subfolders:
            print(f"  Found {len(subfolders)} subfolders:")
            for subfolder in subfolders:
                print(f"    - {subfolder.name}")
            
            # Process each subfolder
            for subfolder in subfolders:
                subfolder_name = subfolder.name
                
                # Get all image files
                image_files = list(subfolder.glob("*.jpg")) + \
                             list(subfolder.glob("*.jpeg")) + \
                             list(subfolder.glob("*.png")) + \
                             list(subfolder.glob("*.JPG")) + \
                             list(subfolder.glob("*.JPEG")) + \
                             list(subfolder.glob("*.PNG"))
                
                subfolder_stats = {
                    'count': len(image_files),
                    'valid': 0,
                    'corrupted': 0
                }
                
                print(f"\n  Validating {subfolder_name}/ ({len(image_files)} images)...")
                
                for img_path in tqdm(image_files, desc=f"    Processing", ncols=70):
                    try:
                        img = cv2.imread(str(img_path))
                        
                        if img is None:
                            subfolder_stats['corrupted'] += 1
                            class_stats['corrupted'] += 1
                            stats['corrupted_images'].append(str(img_path))
                        else:
                            subfolder_stats['valid'] += 1
                            class_stats['valid'] += 1
                            height, width = img.shape[:2]
                            stats['image_sizes'].append((width, height))
                        
                        ext = img_path.suffix.lower()
                        stats['file_formats'][ext] += 1
                        
                    except Exception as e:
                        subfolder_stats['corrupted'] += 1
                        class_stats['corrupted'] += 1
                        stats['corrupted_images'].append(str(img_path))
                
                class_stats['body_parts'][subfolder_name] = subfolder_stats
                class_stats['count'] += subfolder_stats['count']
        else:
            # No subfolders, process directly
            image_files = list(class_path.glob("*.jpg")) + \
                         list(class_path.glob("*.jpeg")) + \
                         list(class_path.glob("*.png")) + \
                         list(class_path.glob("*.JPG")) + \
                         list(class_path.glob("*.JPEG")) + \
                         list(class_path.glob("*.PNG"))
            
            class_stats['count'] = len(image_files)
            
            print(f"  No subfolders found, validating {len(image_files)} images...")
            
            for img_path in tqdm(image_files, desc="    Processing", ncols=70):
                try:
                    img = cv2.imread(str(img_path))
                    
                    if img is None:
                        class_stats['corrupted'] += 1
                        stats['corrupted_images'].append(str(img_path))
                    else:
                        class_stats['valid'] += 1
                        height, width = img.shape[:2]
                        stats['image_sizes'].append((width, height))
                    
                    ext = img_path.suffix.lower()
                    stats['file_formats'][ext] += 1
                    
                except Exception as e:
                    class_stats['corrupted'] += 1
                    stats['corrupted_images'].append(str(img_path))
        
        stats['classes'][class_name] = class_stats
        stats['total_images'] += class_stats['count']
        
        print(f"  ✓ Total: {class_stats['count']} images ({class_stats['valid']} valid, {class_stats['corrupted']} corrupted)\n")
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80 + "\n")
    
    print(f"Total Images: {stats['total_images']}")
    print(f"Valid Images: {sum(c['valid'] for c in stats['classes'].values())}")
    print(f"Corrupted Images: {sum(c['corrupted'] for c in stats['classes'].values())}")
    
    print("\nClass Distribution:")
    for class_name, class_stats in stats['classes'].items():
        if class_stats['count'] > 0:
            percentage = (class_stats['valid'] / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
            print(f"\n  {class_name}:")
            print(f"    - Total: {class_stats['count']}")
            print(f"    - Valid: {class_stats['valid']} ({percentage:.1f}%)")
            if class_stats['corrupted'] > 0:
                print(f"    - Corrupted: {class_stats['corrupted']}")
            
            # Body parts breakdown
            if class_stats.get('body_parts'):
                print(f"    - Body parts:")
                for part_name, part_stats in class_stats['body_parts'].items():
                    print(f"      • {part_name}: {part_stats['valid']} valid, {part_stats['corrupted']} corrupted")
    
    # File format distribution
    if stats['file_formats']:
        print("\nFile Formats:")
        for fmt, count in sorted(stats['file_formats'].items()):
            print(f"  {fmt}: {count}")
    
    # Image size statistics
    if stats['image_sizes']:
        widths = [w for w, h in stats['image_sizes']]
        heights = [h for w, h in stats['image_sizes']]
        
        print("\nImage Size Statistics:")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.0f}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.0f}")
    
    # Corrupted images
    if stats['corrupted_images']:
        print(f"\n⚠️  Corrupted Images ({len(stats['corrupted_images'])}):")
        for img_path in stats['corrupted_images'][:10]:
            print(f"    - {img_path}")
        if len(stats['corrupted_images']) > 10:
            print(f"    ... and {len(stats['corrupted_images']) - 10} more")
    
    # Warnings and recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")
    
    total_valid = sum(c['valid'] for c in stats['classes'].values())
    
    if total_valid == 0:
        print("❌ No valid images found! Please check your dataset.")
    elif total_valid < 100:
        print("⚠️  Very small dataset (<100 images). Consider:")
        print("   - Collecting more data")
        print("   - Using aggressive data augmentation")
        print("   - Using a smaller model architecture")
    elif total_valid < 500:
        print("⚠️  Small dataset (<500 images). Consider:")
        print("   - Enabling data augmentation")
        print("   - Using transfer learning (pre-trained models)")
    else:
        print("✓ Dataset size is adequate for training")
    
    # Check class balance
    class_counts = [c['valid'] for c in stats['classes'].values() if c['valid'] > 0]
    if class_counts:
        imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')
        
        if imbalance_ratio > 3:
            print("\n⚠️  Class imbalance detected (ratio: {:.1f}:1)".format(imbalance_ratio))
            print("   Consider:")
            print("   - Collecting more data for minority class")
            print("   - Using class weights during training")
            print("   - Using data augmentation on minority class")
        else:
            print("\n✓ Classes are reasonably balanced")
    
    if stats['corrupted_images']:
        print(f"\n⚠️  Found {len(stats['corrupted_images'])} corrupted images")
        print("   - These will be skipped during training")
        print("   - Consider removing or re-downloading them")
    
    print("\n" + "="*80)
    print("✓ Dataset validation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate mosquito larvae dataset")
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset',
        help='Path to dataset directory (default: dataset)'
    )
    
    args = parser.parse_args()
    validate_dataset(args.dataset)
