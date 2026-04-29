"""
Data Preparation Module
Handles dataset splitting, copying, and organization
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class DataPreparation:
    """Handles data preparation and splitting"""
    
    def __init__(self):
        self.config = Config
        self.config.create_directories()
        
    def count_images(self, directory):
        """Count total images in a directory"""
        count = 0
        for ext in self.config.IMAGE_EXTENSIONS:
            count += len(list(Path(directory).rglob(f'*{ext}')))
        return count
    
    def get_image_files(self, directory):
        """Get all image files from a directory"""
        image_files = []
        for ext in self.config.IMAGE_EXTENSIONS:
            image_files.extend(list(Path(directory).glob(f'*{ext}')))
        return image_files
    
    def verify_source_data(self):
        """Verify that source data exists"""
        print("\n📋 Verifying source data...")
        
        if not os.path.exists(self.config.RAW_DATA_PATH):
            print(f"❌ Error: Raw data path not found: {self.config.RAW_DATA_PATH}")
            return False
        
        for class_name in self.config.CLASSES:
            class_path = os.path.join(self.config.RAW_DATA_PATH, class_name)
            if not os.path.exists(class_path):
                print(f"❌ Error: Class folder not found: {class_path}")
                return False
            
            count = self.count_images(class_path)
            print(f"  ✓ {class_name}: {count} images found")
            
            if count == 0:
                print(f"❌ Error: No images found in {class_path}")
                return False
        
        return True
    
    def clean_processed_data(self):
        """Clean existing processed data"""
        if os.path.exists(self.config.PROCESSED_DATA_DIR):
            print("\n🧹 Cleaning existing processed data...")
            shutil.rmtree(self.config.PROCESSED_DATA_DIR)
            os.makedirs(self.config.PROCESSED_DATA_DIR)
            
            # Recreate split directories
            for split_dir in [self.config.TRAIN_DIR, self.config.VAL_DIR, self.config.TEST_DIR]:
                os.makedirs(split_dir, exist_ok=True)
                for class_name in self.config.CLASSES:
                    os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
            
            print("  ✓ Cleaned successfully")
    
    def split_and_copy_data(self, clean_first=True):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            clean_first: Whether to clean existing processed data first
        """
        # Set random seed for reproducibility
        random.seed(self.config.RANDOM_SEED)
        
        # Verify source data
        if not self.verify_source_data():
            print("\n❌ Data verification failed. Please check your data paths.")
            return False
        
        # Clean existing data if requested
        if clean_first:
            self.clean_processed_data()
        
        print("\n🔄 Splitting and copying data...")
        print(f"Split ratios - Train: {self.config.TRAIN_SPLIT}, Val: {self.config.VAL_SPLIT}, Test: {self.config.TEST_SPLIT}")
        
        # Statistics
        stats = {
            'train': {class_name: 0 for class_name in self.config.CLASSES},
            'val': {class_name: 0 for class_name in self.config.CLASSES},
            'test': {class_name: 0 for class_name in self.config.CLASSES}
        }
        
        # Process each class
        for class_name in self.config.CLASSES:
            print(f"\n  Processing class: {class_name}")
            
            # Get source path
            source_path = os.path.join(self.config.RAW_DATA_PATH, class_name)
            
            # Get all image files
            image_files = self.get_image_files(source_path)
            total_images = len(image_files)
            
            print(f"    Total images: {total_images}")
            
            # Shuffle images
            random.shuffle(image_files)
            
            # Calculate split indices
            train_end = int(total_images * self.config.TRAIN_SPLIT)
            val_end = train_end + int(total_images * self.config.VAL_SPLIT)
            
            # Split files
            train_files = image_files[:train_end]
            val_files = image_files[train_end:val_end]
            test_files = image_files[val_end:]
            
            # Copy files to respective directories
            splits = {
                'train': (train_files, self.config.TRAIN_DIR),
                'val': (val_files, self.config.VAL_DIR),
                'test': (test_files, self.config.TEST_DIR)
            }
            
            for split_name, (files, split_dir) in splits.items():
                dest_dir = os.path.join(split_dir, class_name)
                
                print(f"    Copying {len(files)} images to {split_name}...")
                
                for file_path in tqdm(files, desc=f"    {split_name}", leave=False):
                    dest_path = os.path.join(dest_dir, file_path.name)
                    shutil.copy2(file_path, dest_path)
                    stats[split_name][class_name] += 1
        
        # Print statistics
        self.print_split_statistics(stats)
        
        print("\n✅ Data preparation completed successfully!")
        return True
    
    def print_split_statistics(self, stats):
        """Print dataset split statistics"""
        print("\n" + "="*60)
        print("DATASET SPLIT STATISTICS")
        print("="*60)
        
        for split_name in ['train', 'val', 'test']:
            print(f"\n{split_name.upper()}:")
            total = 0
            for class_name in self.config.CLASSES:
                count = stats[split_name][class_name]
                total += count
                print(f"  {class_name}: {count} images")
            print(f"  Total: {total} images")
        
        # Grand total
        grand_total = sum(sum(stats[split].values()) for split in stats)
        print(f"\nGRAND TOTAL: {grand_total} images")
        print("="*60)
    
    def verify_split(self):
        """Verify the data split was successful"""
        print("\n🔍 Verifying data split...")
        
        all_good = True
        for split_name, split_dir in [
            ('train', self.config.TRAIN_DIR),
            ('val', self.config.VAL_DIR),
            ('test', self.config.TEST_DIR)
        ]:
            print(f"\n  {split_name.upper()}:")
            for class_name in self.config.CLASSES:
                class_dir = os.path.join(split_dir, class_name)
                count = self.count_images(class_dir)
                print(f"    {class_name}: {count} images")
                
                if count == 0:
                    print(f"    ⚠️  Warning: No images found!")
                    all_good = False
        
        if all_good:
            print("\n✅ Verification passed!")
        else:
            print("\n⚠️  Verification found issues!")
        
        return all_good


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("MOSQUITO CLASSIFICATION - DATA PREPARATION")
    print("="*60)
    
    # Initialize data preparation
    data_prep = DataPreparation()
    
    # Split and copy data
    success = data_prep.split_and_copy_data(clean_first=True)
    
    if success:
        # Verify the split
        data_prep.verify_split()
    else:
        print("\n❌ Data preparation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())