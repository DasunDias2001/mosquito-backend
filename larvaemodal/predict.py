"""
Inference Script for Mosquito Larvae Classification
Load trained model and make predictions on new images
"""

import os
import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import json

import tensorflow as tf
from tensorflow import keras


class LarvaeInference:
    """Inference engine for mosquito larvae classification"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model (.h5 or SavedModel)
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.image_size = tuple(self.config['dataset']['image_size'])
        self.classes = self.config['dataset']['classes']
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def predict_single(self, image_path: str) -> Dict:
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        img = self.preprocess_image(image_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Get predicted class
        predicted_idx = np.argmax(predictions)
        predicted_class = self.classes[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Create result dictionary
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.classes, predictions)
            }
        }
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        for img_path in image_paths:
            try:
                result = self.predict_single(img_path)
                results.append(result)
                print(f"✓ {Path(img_path).name}: {result['predicted_class']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, directory_path: str, 
                         output_path: str = None) -> List[Dict]:
        """
        Predict all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            output_path: Optional path to save results JSON
            
        Returns:
            List of prediction results
        """
        dir_path = Path(directory_path)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(dir_path.glob(f"*{ext}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"No images found in {directory_path}")
            return []
        
        # Predict
        results = self.predict_batch(image_paths)
        
        # Save results if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to {output_path}")
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """
        Print prediction summary
        
        Args:
            results: List of prediction results
        """
        # Count predictions per class
        class_counts = {cls: 0 for cls in self.classes}
        successful = 0
        
        for result in results:
            if 'predicted_class' in result:
                class_counts[result['predicted_class']] += 1
                successful += 1
        
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Total images: {len(results)}")
        print(f"Successfully processed: {successful}")
        print(f"Errors: {len(results) - successful}")
        print("\nPredictions by class:")
        for class_name, count in class_counts.items():
            percentage = (count / successful * 100) if successful > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print("="*60)


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Mosquito Larvae Classification Inference'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.h5 or SavedModel directory)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for prediction'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        help='Path to directory containing images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save prediction results (JSON)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = LarvaeInference(args.model, args.config)
    
    # Run predictions
    if args.image:
        # Single image prediction
        result = inference.predict_single(args.image)
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Image: {result['image_path']}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nProbabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
        print("="*60)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Result saved to {args.output}")
    
    elif args.directory:
        # Directory prediction
        results = inference.predict_directory(args.directory, args.output)
    
    else:
        print("Error: Please provide either --image or --directory")
        parser.print_help()


if __name__ == "__main__":
    main()