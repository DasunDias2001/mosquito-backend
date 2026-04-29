"""
larvae_classifier.py
--------------------
Handles model loading and inference for mosquito larvae classification.
Classifies: Aedes aegypti vs Culex quinquefasciatus
Model: larvaemodal/models/best_model_20260309_232238.h5
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from PIL import Image
import io

CLASS_NAMES = ['Aedes aegypti', 'Culex quinquefasciatus']

MODEL_PATH = Path(__file__).parent.parent / "larvaemodal" / "models" / "best_model_20260309_232238.h5"

_classifier = None


class Cast(keras.layers.Layer):
    """Custom Cast layer — handles models saved with TF Cast operations."""
    def __init__(self, dtype_str='float32', **kwargs):
        super().__init__(**kwargs)
        self.dtype_str = dtype_str

    def call(self, inputs):
        return tf.cast(inputs, self.dtype_str)

    def get_config(self):
        config = super().get_config()
        config.update({"dtype_str": self.dtype_str})
        return config


class LarvaeClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.class_names = CLASS_NAMES
        self.img_size = (224, 224)
        self._load_model()

    def _load_model(self):
        print(f"[LarvaeClassifier] Loading model: {self.model_path}")

        # Attempt 1: custom object scope with Cast layer
        try:
            with keras.utils.custom_object_scope({'Cast': Cast}):
                self.model = keras.models.load_model(
                    str(self.model_path), compile=False
                )
            print("[LarvaeClassifier] Model loaded successfully.")
            return
        except Exception as e:
            print(f"[LarvaeClassifier] Attempt 1 failed: {e}")

        # Attempt 2: tf.keras with custom_objects dict
        try:
            self.model = tf.keras.models.load_model(
                str(self.model_path),
                custom_objects={'Cast': Cast},
                compile=False
            )
            print("[LarvaeClassifier] Model loaded successfully (attempt 2).")
            return
        except Exception as e:
            print(f"[LarvaeClassifier] Attempt 2 failed: {e}")

        # Attempt 3: plain load, no compile
        self.model = keras.models.load_model(
            str(self.model_path), compile=False
        )
        print("[LarvaeClassifier] Model loaded successfully (attempt 3).")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(self.img_size, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        processed = self.preprocess_image(image_bytes)
        preds = self.model.predict(processed, verbose=0)[0]

        # Support sigmoid (binary) or softmax output
        if len(preds) == 1:
            prob_culex = float(preds[0])
            probs = np.array([1.0 - prob_culex, prob_culex])
        else:
            probs = preds

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        return {
            "success": True,
            "predicted_class": self.class_names[class_idx],
            "confidence": confidence,
            "probabilities": {
                self.class_names[i]: round(float(probs[i]), 6)
                for i in range(len(self.class_names))
            },
            "message": f"Predicted as {self.class_names[class_idx].upper()} with {confidence * 100:.2f}% confidence"
        }


def get_larvae_classifier() -> LarvaeClassifier:
    """Get or initialize the global larvae classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = LarvaeClassifier()
    return _classifier