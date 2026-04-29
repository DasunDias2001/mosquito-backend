"""
Prediction Module
Predicts mosquito species from new images
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class MosquitoPredictor:
    """Predicts mosquito species from images"""
    
    def __init__(self, model_path=None):
        self.config = Config
        self.model_path = model_path or self.get_latest_model()
        self.model = None
        self.class_names = self.config.CLASSES
        
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
        if self.model is None:
            print(f"📦 Loading model from: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("✓ Model loaded successfully\n")
        return self.model
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(self.config.IMG_SIZE)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    def predict(self, image_path, save_result=True, show_plot=False):
        """
        Predict mosquito species from image
        
        Args:
            image_path: Path to the image
            save_result: Whether to save prediction result
            show_plot: Whether to display the result plot
        
        Returns:
            dict: Prediction results
        """
        # Load model
        self.load_model()
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"🔮 Predicting species for: {os.path.basename(image_path)}")
        
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Get probabilities for all classes
        probabilities = {
            class_name: float(predictions[0][i]) 
            for i, class_name in enumerate(self.class_names)
        }
        
        # Prepare result
        result = {
            'image_path': image_path,
            'predicted_species': predicted_class,
            'confidence': float(confidence),
            'probabilities': probabilities,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print results
        self.print_prediction(result)
        
        # Visualize results
        if save_result or show_plot:
            self.visualize_prediction(original_img, result, save_result, show_plot)
        
        return result
    
    def print_prediction(self, result):
        """Print prediction results"""
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"\n🦟 Predicted Species: {result['predicted_species'].upper()}")
        print(f"🎯 Confidence: {result['confidence']*100:.2f}%")
        print(f"\n📊 Probabilities:")
        for species, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {species:12s}: {prob*100:5.2f}% {bar}")
        print("="*60 + "\n")
    
    def visualize_prediction(self, image, result, save=True, show=False):
        """Visualize prediction results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot image
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title(f'Input Image\n{os.path.basename(result["image_path"])}', 
                         fontsize=12, pad=10)
        
        # Plot probabilities
        species = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        colors = ['#2ecc71' if s == result['predicted_species'] else '#3498db' 
                 for s in species]
        
        bars = axes[1].barh(species, probs, color=colors)
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_title(
            f'Prediction: {result["predicted_species"].upper()}\n'
            f'Confidence: {result["confidence"]*100:.2f}%',
            fontsize=12, pad=10, weight='bold'
        )
        axes[1].set_xlim(0, 1)
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            axes[1].text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{prob*100:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'prediction_{timestamp}.png'
            save_path = os.path.join(self.config.PREDICTIONS_DIR, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Prediction visualization saved to: {save_path}")
        
        # Show plot
        if show:
            plt.show()
        else:
            plt.close()
    
    def predict_batch(self, image_folder, save_results=True):
        """
        Predict species for all images in a folder
        
        Args:
            image_folder: Path to folder containing images
            save_results: Whether to save results
        
        Returns:
            list: List of prediction results
        """
        # Load model
        self.load_model()
        
        # Get all image files
        image_files = []
        for ext in self.config.IMAGE_EXTENSIONS:
            image_files.extend([
                os.path.join(image_folder, f) 
                for f in os.listdir(image_folder) 
                if f.lower().endswith(ext)
            ])
        
        if not image_files:
            print(f"⚠️  No images found in {image_folder}")
            return []
        
        print(f"\n🔮 Predicting species for {len(image_files)} images...")
        print("="*60)
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            try:
                result = self.predict(image_path, save_result=save_results, show_plot=False)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {image_path}: {str(e)}")
        
        # Print summary
        self.print_batch_summary(results)
        
        return results
    
    def print_batch_summary(self, results):
        """Print summary of batch predictions"""
        if not results:
            return
        
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        print(f"\nTotal images processed: {len(results)}")
        
        # Count predictions per class
        class_counts = {class_name: 0 for class_name in self.class_names}
        avg_confidences = {class_name: [] for class_name in self.class_names}
        
        for result in results:
            predicted = result['predicted_species']
            class_counts[predicted] += 1
            avg_confidences[predicted].append(result['confidence'])
        
        print("\nPredictions by species:")
        for species, count in class_counts.items():
            pct = (count / len(results)) * 100
            avg_conf = np.mean(avg_confidences[species]) if avg_confidences[species] else 0
            print(f"  {species:12s}: {count:3d} ({pct:5.1f}%) - "
                  f"Avg confidence: {avg_conf*100:.1f}%")
        
        # Overall average confidence
        all_confidences = [r['confidence'] for r in results]
        print(f"\nOverall average confidence: {np.mean(all_confidences)*100:.2f}%")
        print("="*60 + "\n")


def main():
    """Main prediction function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict mosquito species from images')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--model', type=str, help='Path to model file (optional)')
    parser.add_argument('--show', action='store_true', help='Show prediction plot')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = MosquitoPredictor(model_path=args.model)
    
    if args.image:
        # Predict single image
        predictor.predict(args.image, save_result=True, show_plot=args.show)
    elif args.folder:
        # Predict batch of images
        predictor.predict_batch(args.folder, save_results=True)
    else:
        print("❌ Please provide either --image or --folder argument")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())