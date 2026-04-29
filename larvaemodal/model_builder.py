

import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB3,
    ResNet50, MobileNetV2, InceptionV3
)
from typing import Tuple


class ModelBuilder:
    """Build and configure neural network models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize model builder with configuration
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        
    def build_model(self) -> Model:
        """
        Build model based on configuration
        
        Returns:
            Compiled Keras model
        """
        architecture = self.model_config['architecture'].lower()
        
        if architecture == "efficientnet_b0":
            return self._build_efficientnet_b0()
        elif architecture == "efficientnet_b3":
            return self._build_efficientnet_b3()
        elif architecture == "resnet50":
            return self._build_resnet50()
        elif architecture == "mobilenet_v2":
            return self._build_mobilenet_v2()
        elif architecture == "inception_v3":
            return self._build_inception_v3()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _build_efficientnet_b0(self) -> Model:
        """Build EfficientNetB0-based model"""
        input_shape = tuple(self.model_config['input_shape'])
        num_classes = self.model_config['num_classes']
        dropout_rate = self.model_config['dropout_rate']
        
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet' if self.model_config['pretrained'] else None,
            input_shape=input_shape
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='EfficientNetB0_Larvae')
        
        print(f"✓ EfficientNetB0 model created")
        return model
    
    def _build_efficientnet_b3(self) -> Model:
        """Build EfficientNetB3-based model"""
        input_shape = tuple(self.model_config['input_shape'])
        num_classes = self.model_config['num_classes']
        dropout_rate = self.model_config['dropout_rate']
        
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet' if self.model_config['pretrained'] else None,
            input_shape=input_shape
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='EfficientNetB3_Larvae')
        
        print(f"✓ EfficientNetB3 model created")
        return model
    
    def _build_resnet50(self) -> Model:
        """Build ResNet50-based model"""
        input_shape = tuple(self.model_config['input_shape'])
        num_classes = self.model_config['num_classes']
        dropout_rate = self.model_config['dropout_rate']
        
        base_model = ResNet50(
            include_top=False,
            weights='imagenet' if self.model_config['pretrained'] else None,
            input_shape=input_shape
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='ResNet50_Larvae')
        
        print(f"✓ ResNet50 model created")
        return model
    
    def _build_mobilenet_v2(self) -> Model:
        """Build MobileNetV2-based model"""
        input_shape = tuple(self.model_config['input_shape'])
        num_classes = self.model_config['num_classes']
        dropout_rate = self.model_config['dropout_rate']
        
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet' if self.model_config['pretrained'] else None,
            input_shape=input_shape
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='MobileNetV2_Larvae')
        
        print(f"✓ MobileNetV2 model created")
        return model
    
    def _build_inception_v3(self) -> Model:
        """Build InceptionV3-based model"""
        input_shape = tuple(self.model_config['input_shape'])
        num_classes = self.model_config['num_classes']
        dropout_rate = self.model_config['dropout_rate']
        
        base_model = InceptionV3(
            include_top=False,
            weights='imagenet' if self.model_config['pretrained'] else None,
            input_shape=input_shape
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='InceptionV3_Larvae')
        
        print(f"✓ InceptionV3 model created")
        return model
    
    def compile_model(self, model: Model) -> Model:
        """
        Compile model with optimizer and loss
        
        Args:
            model: Keras model to compile
            
        Returns:
            Compiled model
        """
        optimizer_name = self.training_config['optimizer'].lower()
        learning_rate = self.training_config['learning_rate']
        
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        model.compile(
            optimizer=optimizer,
            loss=self.training_config['loss'],
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"✓ Model compiled with {optimizer_name} optimizer")
        return model
    
    def unfreeze_model(self, model: Model, from_layer: int = None) -> Model:
        """
        Unfreeze base model layers for fine-tuning
        
        Args:
            model: Keras model
            from_layer: Layer index to start unfreezing from
            
        Returns:
            Model with unfrozen layers
        """
        if from_layer is None:
            from_layer = self.model_config.get('fine_tune_from_layer', 100)
        
        base_model = model.layers[1]
        
        base_model.trainable = True
        for layer in base_model.layers[:from_layer]:
            layer.trainable = False
        
        trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)
        print(f"✓ Fine-tuning enabled: {trainable_layers}/{len(base_model.layers)} base layers trainable")
        
        return model
    
    def get_model_summary(self, model: Model) -> str:
        """
        Get model summary as string
        
        Args:
            model: Keras model
            
        Returns:
            Model summary string
        """
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


if __name__ == "__main__":
    builder = ModelBuilder()
    model = builder.build_model()
    model = builder.compile_model(model)
    print("\nModel Summary:")
    print(builder.get_model_summary(model))
