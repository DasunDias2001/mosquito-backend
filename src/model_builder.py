"""
Model Builder Module
Defines different CNN architectures for mosquito classification
"""

import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
EfficientNetB0 = keras.applications.EfficientNetB0
EfficientNetB3 = keras.applications.EfficientNetB3
ResNet50V2 = keras.applications.ResNet50V20
MobileNetV2 = keras.applications.MobileNetV2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class ModelBuilder:
    """Build different CNN architectures for mosquito classification"""
    
    def __init__(self):
        self.config = Config
        
    def build_custom_cnn(self):
        """
        Build a custom CNN from scratch
        Good for: Understanding baseline performance
        """
        print("\n  Building Custom CNN...")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.config.INPUT_SHAPE),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        return model
    
    def build_efficientnet(self, variant='B0'):
        """
        Build EfficientNet model with transfer learning
        Good for: Best accuracy with reasonable speed
        
        Args:
            variant: 'B0', 'B1', 'B2', 'B3' (B3 is more accurate but slower)
        """
        print(f"\n  Building EfficientNet-{variant}...")
        
        # Select EfficientNet variant
        if variant == 'B0':
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet' if self.config.USE_PRETRAINED else None,
                input_shape=self.config.INPUT_SHAPE
            )
        elif variant == 'B3':
            base_model = EfficientNetB3(
                include_top=False,
                weights='imagenet' if self.config.USE_PRETRAINED else None,
                input_shape=self.config.INPUT_SHAPE
            )
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
        
        # Freeze base model if specified
        base_model.trainable = not self.config.FREEZE_BASE_MODEL
        
        # Build the complete model
        inputs = keras.Input(shape=self.config.INPUT_SHAPE)
        
        # Data augmentation (will only apply during training)
        x = inputs
        
        # Preprocessing
        x = keras.applications.efficientnet.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=not self.config.FREEZE_BASE_MODEL)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model
    
    def build_resnet(self):
        """
        Build ResNet50V2 model with transfer learning
        Good for: Strong performance, well-established architecture
        """
        print("\n🏗️  Building ResNet50V2...")
        
        base_model = ResNet50V2(
            include_top=False,
            weights='imagenet' if self.config.USE_PRETRAINED else None,
            input_shape=self.config.INPUT_SHAPE
        )
        
        # Freeze base model if specified
        base_model.trainable = not self.config.FREEZE_BASE_MODEL
        
        # Build the complete model
        inputs = keras.Input(shape=self.config.INPUT_SHAPE)
        
        x = inputs
        
        # Preprocessing
        x = keras.applications.resnet_v2.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=not self.config.FREEZE_BASE_MODEL)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model
    
    def build_mobilenet(self):
        """
        Build MobileNetV2 model with transfer learning
        Good for: Fastest inference, lower accuracy
        """
        print("\n🏗️  Building MobileNetV2...")
        
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet' if self.config.USE_PRETRAINED else None,
            input_shape=self.config.INPUT_SHAPE
        )
        
        # Freeze base model if specified
        base_model.trainable = not self.config.FREEZE_BASE_MODEL
        
        # Build the complete model
        inputs = keras.Input(shape=self.config.INPUT_SHAPE)
        
        x = inputs
        
        # Preprocessing
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=not self.config.FREEZE_BASE_MODEL)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model
    
    def build_model(self, architecture=None):
        """
        Build model based on configuration or specified architecture
        
        Args:
            architecture: Model architecture name (overrides config if provided)
        """
        arch = architecture or self.config.MODEL_ARCHITECTURE
        
        model_builders = {
            'custom_cnn': self.build_custom_cnn,
            'efficientnet': self.build_efficientnet,
            'efficientnet_b3': lambda: self.build_efficientnet('B3'),
            'resnet': self.build_resnet,
            'mobilenet': self.build_mobilenet
        }
        
        if arch not in model_builders:
            raise ValueError(f"Unknown architecture: {arch}. Choose from {list(model_builders.keys())}")
        
        model = model_builders[arch]()
        
        print(f"✓ Model built: {arch}")
        return model
    
    def compile_model(self, model):
        """
        Compile the model with optimizer, loss, and metrics
        
        Args:
            model: Keras model to compile
        """
        # Select optimizer
        if self.config.OPTIMIZER.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.config.INITIAL_LEARNING_RATE)
        elif self.config.OPTIMIZER.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=self.config.INITIAL_LEARNING_RATE,
                momentum=0.9,
                nesterov=True
            )
        elif self.config.OPTIMIZER.lower() == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.config.INITIAL_LEARNING_RATE)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=self.config.LOSS_FUNCTION,
            metrics=self.config.METRICS
        )
        
        print(f"✓ Model compiled with {self.config.OPTIMIZER} optimizer")
        return model
    
    def print_model_summary(self, model):
        """Print model architecture summary"""
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*60)
        model.summary()
        print("="*60)
        
        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print("="*60 + "\n")


def main():
    """Test model builder"""
    builder = ModelBuilder()
    
    # Test EfficientNet (default)
    print("\nTesting EfficientNet...")
    model = builder.build_model('efficientnet')
    builder.compile_model(model)
    builder.print_model_summary(model)
    
    # Clear session to free memory
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()