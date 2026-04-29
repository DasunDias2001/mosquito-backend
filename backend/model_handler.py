import os
import numpy as np
import cv2
from tensorflow import keras
from pathlib import Path

class MosquitoClassifier:
    def __init__(self, model_path):
        """
        Initialize the classifier with the trained model.
        
        Args:
            model_path: Path to the .keras model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['aegypti', 'albopictus']
        self.img_size = (224, 224)  # Match your training size
        
        self.load_model()
    
    def load_model(self):
        """Load the trained Keras model."""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print(" Model loaded successfully!")
        except Exception as e:
            print(f" Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess the uploaded image for prediction.
        
        Args:
            image_path: Path to the uploaded image
            
        Returns:
            Preprocessed image array ready for model input
        """
        # Read image
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Resize to model input size
        img = cv2.resize(img, self.img_size)
        
        # Normalize pixel values (0-255 → 0-1)
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path):
        """
        Make prediction on the uploaded image.
        
        Args:
            image_path: Path to the uploaded image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Get prediction
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            # Get probabilities for both classes
            probabilities = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
            
            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "message": f"Predicted as {predicted_class.upper()} with {confidence*100:.2f}% confidence"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Prediction failed"
            }


# Global model instance (loaded once at startup)
MODEL_PATH = Path(__file__).parent.parent / "models" / "saved_models" / "best_model_20260101_214956.keras"
classifier = None

def get_classifier():
    """Get or initialize the global classifier instance."""
    global classifier
    if classifier is None:
        classifier = MosquitoClassifier(MODEL_PATH)
    return classifier