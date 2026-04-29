"""
Trainer Module for Mosquito Larvae Classification
Handles model training, callbacks, and evaluation
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)


class LarvaeTrainer:
    """Training pipeline for mosquito larvae classification"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.output_config = self.config['output']
        self.dataset_config = self.config['dataset']
        
        # Create output directories
        self.model_dir = Path(self.output_config['model_dir'])
        self.logs_dir = Path(self.output_config['logs_dir'])
        self.results_dir = Path(self.output_config['results_dir'])
        
        self.model_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / self.timestamp
        self.run_dir.mkdir(exist_ok=True)
        
        self.classes = self.dataset_config['classes']
        
    def create_callbacks(self) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model Checkpoint
        if self.training_config['checkpoint']['enabled']:
            checkpoint_path = self.model_dir / f"best_model_{self.timestamp}.h5"
            checkpoint = ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=self.training_config['checkpoint']['monitor'],
                mode=self.training_config['checkpoint']['mode'],
                save_best_only=self.training_config['checkpoint']['save_best_only'],
                verbose=1
            )
            callbacks.append(checkpoint)
            print(f"✓ Model checkpoint: {checkpoint_path}")
        
        # Early Stopping
        if self.training_config['early_stopping']['enabled']:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping']['patience'],
                restore_best_weights=self.training_config['early_stopping']['restore_best_weights'],
                verbose=1
            )
            callbacks.append(early_stop)
            print(f"✓ Early stopping: patience={self.training_config['early_stopping']['patience']}")
        
        # Learning Rate Scheduler
        if self.training_config['lr_schedule']['enabled']:
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['lr_schedule']['factor'],
                patience=self.training_config['lr_schedule']['patience'],
                min_lr=self.training_config['lr_schedule']['min_lr'],
                verbose=1
            )
            callbacks.append(lr_scheduler)
            print(f"✓ LR scheduler: factor={self.training_config['lr_schedule']['factor']}")
        
        # TensorBoard
        tensorboard_dir = self.logs_dir / self.timestamp
        tensorboard = TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        print(f"✓ TensorBoard: {tensorboard_dir}")
        
        # CSV Logger
        csv_path = self.run_dir / "training_log.csv"
        csv_logger = CSVLogger(str(csv_path))
        callbacks.append(csv_logger)
        print(f"✓ CSV Logger: {csv_path}")
        
        return callbacks
    
    def train(self, model: keras.Model, data_dict: Dict, 
             train_datagen=None, val_datagen=None) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            model: Keras model to train
            data_dict: Dictionary containing train/val data
            train_datagen: Training data generator (optional)
            val_datagen: Validation data generator (optional)
            
        Returns:
            Training history
        """
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['epochs']
        
        callbacks = self.create_callbacks()
        
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.training_config['learning_rate']}")
        print("="*80 + "\n")
        
        if train_datagen is not None:
            # Use data generator
            train_generator = train_datagen.flow(
                X_train, y_train,
                batch_size=batch_size,
                shuffle=True
            )
            
            val_generator = val_datagen.flow(
                X_val, y_val,
                batch_size=batch_size,
                shuffle=False
            )
            
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Direct training
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def evaluate(self, model: keras.Model, data_dict: Dict) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            model: Trained Keras model
            data_dict: Dictionary containing test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        y_test_raw = data_dict['y_test_raw']
        
        print("\n" + "="*80)
        print("EVALUATING MODEL")
        print("="*80)
        
        # Predict
        y_pred_proba = model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(
            X_test, y_test, verbose=0
        )
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_auc': float(test_auc),
            'predictions': y_pred.tolist(),
            'predictions_proba': y_pred_proba.tolist(),
            'true_labels': y_test_raw.tolist()
        }
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall: {test_rec:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        
        # Classification report
        report = classification_report(
            y_test_raw, y_pred,
            target_names=self.classes,
            digits=4
        )
        print("\nClassification Report:")
        print(report)
        
        # Save results
        results_path = self.run_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        report_path = self.run_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Results saved to {self.run_dir}")
        
        return results
    
    def plot_training_history(self, history: keras.callbacks.History):
        """
        Plot training history
        
        Args:
            history: Training history object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Train')
            axes[1, 0].plot(history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Train')
            axes[1, 1].plot(history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.run_dir / "training_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training history plot saved to {save_path}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = self.run_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {save_path}")
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """
        Plot ROC curves for each class
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred_proba: Predicted probabilities
        """
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.run_dir / "roc_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to {save_path}")


if __name__ == "__main__":
    print("Trainer module - Use train.py for full training pipeline")