"""
Data Loader Module for Mosquito Larvae Classification
Handles data loading with body part subfolders, preprocessing, and augmentation
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm


class LarvaeDataLoader:
    """Data loader for mosquito larvae images with body part subfolders"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize data loader with configuration
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.aug_config = self.config['augmentation']
        self.image_size = tuple(self.dataset_config['image_size'])
        self.classes = self.dataset_config['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all images from dataset directory, including body part subfolders
        
        Returns:
            Tuple of (images, labels, file_paths)
        """
        images = []
        labels = []
        file_paths = []
        
        base_path = Path(self.dataset_config['base_path'])
        
        print("Loading dataset with body part subfolders...")
        print(f"Base path: {base_path}")
        
        for class_name in self.classes:
            class_path = base_path / class_name
            
            if not class_path.exists():
                print(f"Warning: Class directory {class_path} does not exist!")
                continue
            
            print(f"\n{'='*60}")
            print(f"Loading class: {class_name}")
            print(f"{'='*60}")
            
            # Check if there are body part subfolders
            subfolders = [f for f in class_path.iterdir() if f.is_dir()]
            
            if subfolders:
                # Has subfolders (abdomen, full body, head, siphon)
                print(f"Found {len(subfolders)} body part subfolders:")
                for subfolder in subfolders:
                    print(f"  - {subfolder.name}")
                
                # Load images from each subfolder
                for subfolder in subfolders:
                    subfolder_name = subfolder.name
                    print(f"\n  Loading from: {subfolder_name}/")
                    
                    image_files = self._get_image_files(subfolder)
                    
                    if len(image_files) == 0:
                        print(f"    No images found in {subfolder_name}")
                        continue
                    
                    print(f"    Found {len(image_files)} images")
                    
                    for img_path in tqdm(image_files, desc=f"    Processing {subfolder_name}", ncols=80):
                        success = self._load_single_image(
                            img_path, class_name, images, labels, file_paths
                        )
            else:
                # No subfolders, load directly from class folder
                print(f"No subfolders found, loading directly from {class_name}")
                image_files = self._get_image_files(class_path)
                
                print(f"Found {len(image_files)} images")
                
                for img_path in tqdm(image_files, desc=f"  Processing {class_name}", ncols=80):
                    success = self._load_single_image(
                        img_path, class_name, images, labels, file_paths
                    )
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"\n{'='*60}")
        print(f"DATASET LOADED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total images: {len(images)}")
        if len(images) > 0:
            print(f"Image shape: {images[0].shape}")
        
        # Print class distribution
        print(f"\nClass distribution:")
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            count = np.sum(labels == class_idx)
            percentage = (count / len(labels) * 100) if len(labels) > 0 else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        return images, labels, file_paths
    
    def _get_image_files(self, folder_path: Path) -> List[Path]:
        """
        Get all image files from a folder
        
        Args:
            folder_path: Path to folder
            
        Returns:
            List of image file paths
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
        
        return sorted(image_files)
    
    def _load_single_image(self, img_path: Path, class_name: str,
                          images: List, labels: List, file_paths: List) -> bool:
        """
        Load a single image and add to lists
        
        Args:
            img_path: Path to image
            class_name: Class name
            images: List to append image to
            labels: List to append label to
            file_paths: List to append file path to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read and preprocess image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"      Warning: Could not read {img_path.name}")
                return False
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.image_size)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(self.class_to_idx[class_name])
            file_paths.append(str(img_path))
            
            return True
            
        except Exception as e:
            print(f"      Error loading {img_path.name}: {e}")
            return False
    
    def split_dataset(self, images: np.ndarray, labels: np.ndarray, 
                     file_paths: List[str]) -> Dict:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            images: Array of images
            labels: Array of labels
            file_paths: List of file paths
            
        Returns:
            Dictionary containing train, validation, and test splits
        """
        val_split = self.dataset_config['validation_split']
        test_split = self.dataset_config['test_split']
        seed = self.dataset_config['seed']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test, paths_temp, paths_test = train_test_split(
            images, labels, file_paths,
            test_size=test_split,
            random_state=seed,
            stratify=labels
        )
        
        # Second split: separate train and validation
        val_ratio = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
            X_temp, y_temp, paths_temp,
            test_size=val_ratio,
            random_state=seed,
            stratify=y_temp
        )
        
        # Convert labels to categorical
        num_classes = len(self.classes)
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        print("\n" + "="*60)
        print("DATASET SPLIT")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Print class distribution for each split
        print("\nClass distribution by split:")
        for split_name, split_labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            print(f"\n{split_name}:")
            for class_name in self.classes:
                class_idx = self.class_to_idx[class_name]
                count = np.sum(split_labels == class_idx)
                total = len(split_labels)
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return {
            'X_train': X_train, 'y_train': y_train_cat,
            'X_val': X_val, 'y_val': y_val_cat,
            'X_test': X_test, 'y_test': y_test_cat,
            'paths_train': paths_train,
            'paths_val': paths_val,
            'paths_test': paths_test,
            'y_train_raw': y_train,
            'y_val_raw': y_val,
            'y_test_raw': y_test
        }
    
    def create_data_generators(self):
        """
        Create data generators for training and validation
        
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        if self.aug_config['enabled']:
            train_datagen = ImageDataGenerator(
                rotation_range=self.aug_config['rotation_range'],
                width_shift_range=self.aug_config['width_shift_range'],
                height_shift_range=self.aug_config['height_shift_range'],
                shear_range=self.aug_config['shear_range'],
                zoom_range=self.aug_config['zoom_range'],
                horizontal_flip=self.aug_config['horizontal_flip'],
                vertical_flip=self.aug_config['vertical_flip'],
                brightness_range=self.aug_config['brightness_range'],
                fill_mode=self.aug_config['fill_mode']
            )
        else:
            train_datagen = ImageDataGenerator()
        
        # Validation generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        return train_datagen, val_datagen
    
    def visualize_samples(self, data_dict: Dict, num_samples: int = 10, 
                         save_path: str = None):
        """
        Visualize random samples from each class
        
        Args:
            data_dict: Dictionary containing dataset splits
            num_samples: Number of samples to visualize per class
            save_path: Path to save visualization
        """
        X_train = data_dict['X_train']
        y_train = data_dict['y_train_raw']
        
        fig, axes = plt.subplots(len(self.classes), num_samples, 
                                figsize=(20, 4 * len(self.classes)))
        
        for cls_idx, class_name in enumerate(self.classes):
            # Get indices for this class
            class_indices = np.where(y_train == cls_idx)[0]
            
            # Randomly select samples
            if len(class_indices) < num_samples:
                selected_indices = class_indices
            else:
                selected_indices = np.random.choice(class_indices, num_samples, replace=False)
            
            for i, idx in enumerate(selected_indices):
                ax = axes[cls_idx, i] if len(self.classes) > 1 else axes[i]
                ax.imshow(X_train[idx])
                ax.axis('off')
                if i == 0:
                    ax.set_title(f"{class_name}\n(Sample {i+1})", fontsize=10, fontweight='bold')
                else:
                    ax.set_title(f"Sample {i+1}", fontsize=10)
        
        plt.suptitle("Sample Images from Dataset (All Body Parts Combined)", 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.close()
        
        return fig


if __name__ == "__main__":
    # Test data loader
    loader = LarvaeDataLoader()
    images, labels, paths = loader.load_dataset()
    
    if len(images) > 0:
        data_dict = loader.split_dataset(images, labels, paths)
        
        # Visualize samples
        os.makedirs("results", exist_ok=True)
        loader.visualize_samples(data_dict, save_path="results/sample_images.png")
    else:
        print("\n❌ No images loaded. Please check your dataset structure!")