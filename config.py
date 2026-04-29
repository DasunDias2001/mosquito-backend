"""
Configuration file for Mosquito Species Classification
All paths, hyperparameters, and settings
"""

import os

class Config:
    """Configuration class for the project"""
    
    # ==================== PATHS ====================
    # Raw data paths (your laptop paths)
    RAW_DATA_PATH = r"C:\MosquitoProj\Dataset\raw"
    AEGYPTI_PATH = os.path.join(RAW_DATA_PATH, "aegypti")
    ALBOPICTUS_PATH = os.path.join(RAW_DATA_PATH, "albopictus")
    
    # Project directories
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Split directories
    TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
    VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
    TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")
    
    # Model directories
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    SAVED_MODELS_DIR = os.path.join(MODELS_DIR, "saved_models")
    CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
    
    # Results directories
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    EVALUATION_DIR = os.path.join(RESULTS_DIR, "evaluation")
    PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
    
    # Logs directory
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
    TENSORBOARD_DIR = os.path.join(LOGS_DIR, "tensorboard")
    
    # ==================== DATA PARAMETERS ====================
    # Class names
    CLASSES = ['aegypti', 'albopictus']
    NUM_CLASSES = 2
    
    # Data split ratios (must sum to 1.0)
    TRAIN_SPLIT = 0.70  # 70% for training
    VAL_SPLIT = 0.15    # 15% for validation
    TEST_SPLIT = 0.15   # 15% for testing
    
    # Image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    IMG_CHANNELS = 3
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    # Supported image formats
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # ==================== MODEL PARAMETERS ====================
    # Model architecture options: 'efficientnet', 'resnet', 'mobilenet', 'custom_cnn'
    MODEL_ARCHITECTURE = 'efficientnet'
    
    # Transfer learning
    USE_PRETRAINED = True
    FREEZE_BASE_MODEL = False  # Set to True to freeze pretrained layers initially
    
    # ==================== TRAINING PARAMETERS ====================
    BATCH_SIZE = 32  # Reduce to 16 or 8 if out of memory
    EPOCHS = 50
    INITIAL_LEARNING_RATE = 0.001
    
    # Optimizer: 'adam', 'sgd', 'rmsprop'
    OPTIMIZER = 'adam'
    
    # Loss function: 'categorical_crossentropy', 'sparse_categorical_crossentropy'
    LOSS_FUNCTION = 'categorical_crossentropy'
    
    # Metrics
    METRICS = ['accuracy', 'AUC', 'Precision', 'Recall']
    
    # ==================== DATA AUGMENTATION ====================
    # Training data augmentation
    AUGMENTATION_TRAIN = {
        'rotation_range': 40,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'fill_mode': 'nearest',
        'brightness_range': [0.8, 1.2]
    }
    
    # Validation/Test data (only rescaling)
    AUGMENTATION_VAL_TEST = {}
    
    # ==================== CALLBACKS ====================
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MONITOR = 'val_loss'
    EARLY_STOPPING_MODE = 'min'
    
    # Model checkpoint
    CHECKPOINT_MONITOR = 'val_accuracy'
    CHECKPOINT_MODE = 'max'
    CHECKPOINT_SAVE_BEST_ONLY = True
    
    # Learning rate reduction
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_MIN_LR = 1e-7
    
    # ==================== EVALUATION PARAMETERS ====================
    # Confidence threshold for predictions
    CONFIDENCE_THRESHOLD = 0.5
    
    # ==================== RANDOM SEED ====================
    RANDOM_SEED = 42
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.TRAIN_DIR,
            cls.VAL_DIR,
            cls.TEST_DIR,
            cls.MODELS_DIR,
            cls.SAVED_MODELS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.RESULTS_DIR,
            cls.PLOTS_DIR,
            cls.EVALUATION_DIR,
            cls.PREDICTIONS_DIR,
            cls.LOGS_DIR,
            cls.TENSORBOARD_DIR
        ]
        
        # Create class subdirectories for train, val, test
        for split_dir in [cls.TRAIN_DIR, cls.VAL_DIR, cls.TEST_DIR]:
            for class_name in cls.CLASSES:
                directories.append(os.path.join(split_dir, class_name))
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"✓ All directories created successfully!")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("MOSQUITO CLASSIFICATION - CONFIGURATION")
        print("="*60)
        print(f"\n📁 Data Paths:")
        print(f"  Raw Data: {cls.RAW_DATA_PATH}")
        print(f"  Processed Data: {cls.PROCESSED_DATA_DIR}")
        
        print(f"\n🖼️  Image Settings:")
        print(f"  Size: {cls.IMG_SIZE}")
        print(f"  Classes: {cls.CLASSES}")
        
        print(f"\n📊 Data Split:")
        print(f"  Train: {cls.TRAIN_SPLIT*100}%")
        print(f"  Validation: {cls.VAL_SPLIT*100}%")
        print(f"  Test: {cls.TEST_SPLIT*100}%")
        
        print(f"\n🧠 Model Settings:")
        print(f"  Architecture: {cls.MODEL_ARCHITECTURE}")
        print(f"  Use Pretrained: {cls.USE_PRETRAINED}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Learning Rate: {cls.INITIAL_LEARNING_RATE}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test configuration
    Config.create_directories()
    Config.print_config()