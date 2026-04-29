
import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from data_loader import LarvaeDataLoader
from model_builder import ModelBuilder
from trainer import LarvaeTrainer


def setup_gpu():
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU(s) available: {len(gpus)}")
            print(f"  {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPU detected - using CPU")
    
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✓ Mixed precision enabled (float16)")
    except:
        print("⚠ Mixed precision not available")


def main(args):
    """Main training function"""
    
    print("\n" + "="*80)
    print("MOSQUITO LARVAE CLASSIFICATION - TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    if not args.no_gpu:
        setup_gpu()
    
    config_path = args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ Configuration loaded from {config_path}")
    print(f"  Architecture: {config['model']['architecture']}")
    print(f"  Image size: {config['dataset']['image_size']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['epochs']}")
    
    print("\n" + "-"*80)
    print("STEP 1: Loading Dataset")
    print("-"*80)
    
    data_loader = LarvaeDataLoader(config_path)
    images, labels, file_paths = data_loader.load_dataset()
    
    if len(images) == 0:
        print("\n❌ ERROR: No images found in dataset!")
        print("Please ensure dataset is in the correct location:")
        print(f"  {config['dataset']['base_path']}/")
        print(f"    ├── {config['dataset']['classes'][0]}/")
        print(f"    └── {config['dataset']['classes'][1]}/")
        return
    
    data_dict = data_loader.split_dataset(images, labels, file_paths)
    
    trainer = LarvaeTrainer(config_path)
    data_loader.visualize_samples(data_dict, 
                                  save_path=trainer.run_dir / "dataset_samples.png")
    
    print("\n" + "-"*80)
    print("STEP 2: Creating Data Augmentation")
    print("-"*80)
    
    train_datagen, val_datagen = data_loader.create_data_generators()
    print("✓ Data generators created")
    
    print("\n" + "-"*80)
    print("STEP 3: Building Model")
    print("-"*80)
    
    model_builder = ModelBuilder(config_path)
    model = model_builder.build_model()
    model = model_builder.compile_model(model)
    
    # Save model summary with UTF-8 encoding
    summary_path = trainer.run_dir / "model_summary.txt"
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(model_builder.get_model_summary(model))
        print(f"✓ Model summary saved to {summary_path}")
    except Exception as e:
        print(f"⚠ Could not save model summary: {e}")
    
    print("\n" + "-"*80)
    print("STEP 4: Training Model (Phase 1 - Transfer Learning)")
    print("-"*80)
    
    history = trainer.train(model, data_dict, train_datagen, val_datagen)
    
    trainer.plot_training_history(history)
    
    if args.fine_tune:
        print("\n" + "-"*80)
        print("STEP 5: Fine-Tuning Model (Phase 2 - Unfreezing Layers)")
        print("-"*80)
        
        model = model_builder.unfreeze_model(model)
        
        config['training']['learning_rate'] = config['training']['learning_rate'] / 10
        model = model_builder.compile_model(model)
        
        config['training']['epochs'] = args.fine_tune_epochs
        trainer_ft = LarvaeTrainer(config_path)
        
        history_ft = trainer_ft.train(model, data_dict, train_datagen, val_datagen)
        trainer_ft.plot_training_history(history_ft)
        
        trainer = trainer_ft
    
    print("\n" + "-"*80)
    print("STEP 6: Evaluating Model")
    print("-"*80)
    
    results = trainer.evaluate(model, data_dict)
    
    trainer.plot_confusion_matrix(
        data_dict['y_test_raw'],
        results['predictions']
    )
    
    trainer.plot_roc_curves(
        data_dict['y_test'],
        results['predictions_proba']
    )
    
    print("\n" + "-"*80)
    print("STEP 7: Saving Final Model")
    print("-"*80)
    
    final_model_path = trainer.model_dir / f"final_model_{trainer.timestamp}.h5"
    model.save(final_model_path)
    print(f"✓ Final model saved to {final_model_path}")
    
    saved_model_path = trainer.model_dir / f"saved_model_{trainer.timestamp}"
    model.save(saved_model_path, save_format='tf')
    print(f"✓ SavedModel format saved to {saved_model_path}")
    
    config_save_path = trainer.run_dir / "config.yaml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Configuration saved to {config_save_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults directory: {trainer.run_dir}")
    print(f"Model path: {final_model_path}")
    print(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Precision: {results['test_precision']:.4f}")
    print(f"Test Recall: {results['test_recall']:.4f}")
    print(f"Test AUC: {results['test_auc']:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Mosquito Larvae Classification Model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Enable fine-tuning after initial training'
    )
    
    parser.add_argument(
        '--fine-tune-epochs',
        type=int,
        default=50,
        help='Number of epochs for fine-tuning (default: 50)'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
