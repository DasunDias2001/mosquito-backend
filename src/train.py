"""
Training Module
Trains the mosquito classification model
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
# Keras imports - avoid Pylance warnings
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
ModelCheckpoint = keras.callbacks.ModelCheckpoint
EarlyStopping = keras.callbacks.EarlyStopping
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
TensorBoard = keras.callbacks.TensorBoard

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.model_builder import ModelBuilder


class ModelTrainer:
    """Handles model training"""
    
    def __init__(self):
        self.config = Config
        self.config.create_directories()
        self.model_builder = ModelBuilder()
        self.model = None
        self.history = None
        
    def create_data_generators(self):
        """Create data generators for training and validation"""
        print("\n📊 Creating data generators...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **self.config.AUGMENTATION_TRAIN
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_DIR,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=self.config.RANDOM_SEED
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.config.VAL_DIR,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        print(f"✓ Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator
    
    def create_callbacks(self):
        """Create training callbacks"""
        print("\n⚙️  Setting up callbacks...")
        
        # Timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model checkpoint - save best model
        checkpoint_path = os.path.join(
            self.config.SAVED_MODELS_DIR, 
            f'best_model_{timestamp}.keras'
        )
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.config.CHECKPOINT_MONITOR,
            mode=self.config.CHECKPOINT_MODE,
            save_best_only=self.config.CHECKPOINT_SAVE_BEST_ONLY,
            verbose=1
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.EARLY_STOPPING_MONITOR,
            patience=self.config.EARLY_STOPPING_PATIENCE,
            mode=self.config.EARLY_STOPPING_MODE,
            verbose=1,
            restore_best_weights=True
        )
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.REDUCE_LR_FACTOR,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=self.config.REDUCE_LR_MIN_LR,
            verbose=1
        )
        
        # TensorBoard
        tensorboard_dir = os.path.join(
            self.config.TENSORBOARD_DIR,
            f'run_{timestamp}'
        )
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True
        )
        
        callbacks = [model_checkpoint, early_stopping, reduce_lr, tensorboard]
        
        print(f"✓ Model will be saved to: {checkpoint_path}")
        print(f"✓ TensorBoard logs: {tensorboard_dir}")
        
        return callbacks, checkpoint_path
    
    def train(self):
        """Train the model"""
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING")
        print("="*60)
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        
        # Build and compile model
        print(f"\n🏗️  Building model: {self.config.MODEL_ARCHITECTURE}")
        self.model = self.model_builder.build_model()
        self.model = self.model_builder.compile_model(self.model)
        
        # Print model summary
        self.model_builder.print_model_summary(self.model)
        
        # Create callbacks
        callbacks, checkpoint_path = self.create_callbacks()
        
        # Train the model
        print("\n🚀 Starting training...")
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.INITIAL_LEARNING_RATE}")
        print(f"\n⏳ Training on CPU - This may take 2-5 hours...")
        print("You can close this window - training will continue in background")
        print("="*60 + "\n")
        
        start_time = datetime.now()
        
        self.history = self.model.fit(
            train_gen,
            epochs=self.config.EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED!")
        print("="*60)
        print(f"Training time: {training_time}")
        print(f"Best model saved to: {checkpoint_path}")
        print("="*60 + "\n")
        
        # Save training history
        self.save_training_history(checkpoint_path)
        
        # Plot training history
        self.plot_training_history()
        
        return self.model, self.history, checkpoint_path
    
    def save_training_history(self, model_path):
        """Save training history to JSON"""
        history_dict = {key: [float(val) for val in values] 
                       for key, values in self.history.history.items()}
        
        history_path = model_path.replace('.keras', '_history.json')
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"✓ Training history saved to: {history_path}")
    
    def plot_training_history(self):
        """Plot and save training history"""
        print("\n📈 Creating training plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision if available
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall if available
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.PLOTS_DIR, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training plots saved to: {plot_path}")
        
        plt.close()


def main():
    """Main training function"""
    # Set random seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    tf.random.set_seed(Config.RANDOM_SEED)
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Train model
    model, history, model_path = trainer.train()
    
    print("\n✅ Training completed successfully!")
    print(f"📦 Model saved at: {model_path}")
    print("\n💡 Next steps:")
    print("  1. Run evaluation: python run_pipeline.py --mode evaluate")
    print("  2. Make predictions: python run_pipeline.py --mode predict --image path/to/image.jpg")
    
    return 0


if __name__ == "__main__":
    exit(main())