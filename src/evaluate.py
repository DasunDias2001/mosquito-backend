"""
Evaluation Module
Evaluates trained model on test dataset
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class ModelEvaluator:
    """Evaluates trained model"""
    
    def __init__(self, model_path=None):
        self.config = Config
        self.config.create_directories()
        self.model_path = model_path or self.get_latest_model()
        self.model = None
        
    def get_latest_model(self):
        """Get the most recent trained model"""
        model_files = [f for f in os.listdir(self.config.SAVED_MODELS_DIR) 
                      if f.endswith('.keras')]
        
        if not model_files:
            raise FileNotFoundError(
                f"No model files found in {self.config.SAVED_MODELS_DIR}\n"
                "Please train a model first: python run_pipeline.py --mode train"
            )
        
        # Sort by modification time
        model_files.sort(
            key=lambda x: os.path.getmtime(
                os.path.join(self.config.SAVED_MODELS_DIR, x)
            ),
            reverse=True
        )
        
        latest_model = os.path.join(self.config.SAVED_MODELS_DIR, model_files[0])
        return latest_model
    
    def load_model(self):
        """Load trained model"""
        print(f"\n📦 Loading model from: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("✓ Model loaded successfully")
        return self.model
    
    def create_test_generator(self):
        """Create test data generator"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            self.config.TEST_DIR,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False  # Important for evaluation
        )
        
        print(f"\n📊 Test set: {test_generator.samples} images")
        print(f"Classes: {test_generator.class_indices}")
        
        return test_generator
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Load model
        self.load_model()
        
        # Create test generator
        test_gen = self.create_test_generator()
        
        # Get predictions
        print("\n🔮 Making predictions on test set...")
        predictions = self.model.predict(test_gen, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes
        
        # Get class names
        class_names = list(test_gen.class_indices.keys())
        
        # Calculate metrics
        print("\n📈 Calculating metrics...")
        results = self.calculate_metrics(y_true, y_pred, predictions, class_names)
        
        # Print results
        self.print_results(results)
        
        # Save results
        self.save_results(results)
        
        # Create visualizations
        self.plot_confusion_matrix(y_true, y_pred, class_names)
        self.plot_prediction_distribution(predictions, class_names)
        
        print("\n✅ Evaluation completed!")
        return results
    
    def calculate_metrics(self, y_true, y_pred, predictions, class_names):
        """Calculate all evaluation metrics"""
        results = {}
        
        # Overall metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted')
        results['recall'] = recall_score(y_true, y_pred, average='weighted')
        results['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names, 
            output_dict=True
        )
        results['classification_report'] = report
        
        # Per-class accuracy
        results['per_class_accuracy'] = {}
        for i, class_name in enumerate(class_names):
            class_mask = y_true == i
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
                results['per_class_accuracy'][class_name] = float(class_acc)
        
        # Confidence statistics
        max_confidences = np.max(predictions, axis=1)
        results['confidence_stats'] = {
            'mean': float(np.mean(max_confidences)),
            'std': float(np.std(max_confidences)),
            'min': float(np.min(max_confidences)),
            'max': float(np.max(max_confidences))
        }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        
        print(f"\n📋 Per-Class Accuracy:")
        for class_name, acc in results['per_class_accuracy'].items():
            print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")
        
        print(f"\n🎯 Prediction Confidence:")
        stats = results['confidence_stats']
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        
        print("\n" + "="*60)
    
    def save_results(self, results):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            self.config.EVALUATION_DIR,
            f'evaluation_results_{timestamp}.json'
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n💾 Results saved to: {results_path}")
        
        # Also save as text report
        report_path = results_path.replace('.json', '.txt')
        with open(report_path, 'w') as f:
            f.write("MOSQUITO CLASSIFICATION - EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: {os.path.basename(self.model_path)}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall:    {results['recall']:.4f}\n")
            f.write(f"F1-Score:  {results['f1_score']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-"*60 + "\n")
            for class_name, metrics in results['classification_report'].items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"\n{class_name.upper()}:\n")
                    if isinstance(metrics, dict):
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                        f.write(f"  F1-Score:  {metrics['f1-score']:.4f}\n")
                        f.write(f"  Support:   {int(metrics['support'])}\n")
        
        print(f"💾 Text report saved to: {report_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.PLOTS_DIR, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {plot_path}")
        plt.close()
    
    def plot_prediction_distribution(self, predictions, class_names):
        """Plot prediction confidence distribution"""
        max_confidences = np.max(predictions, axis=1)
        pred_classes = np.argmax(predictions, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Overall confidence distribution
        axes[0].hist(max_confidences, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(max_confidences), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(max_confidences):.3f}')
        axes[0].set_title('Prediction Confidence Distribution', fontsize=14)
        axes[0].set_xlabel('Confidence', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Per-class confidence
        for i, class_name in enumerate(class_names):
            class_confidences = max_confidences[pred_classes == i]
            axes[1].hist(class_confidences, bins=30, alpha=0.6, label=class_name)
        
        axes[1].set_title('Confidence Distribution by Class', fontsize=14)
        axes[1].set_xlabel('Confidence', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.PLOTS_DIR, 'prediction_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Prediction distribution saved to: {plot_path}")
        plt.close()


def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate()
    
    print("\n💡 Evaluation complete! Check the results folder for detailed metrics and plots.")
    print(f"   Results: {Config.EVALUATION_DIR}")
    print(f"   Plots: {Config.PLOTS_DIR}")
    
    return 0


if __name__ == "__main__":
    exit(main())