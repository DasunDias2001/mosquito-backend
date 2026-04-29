"""
Main Pipeline Controller
Runs the complete mosquito classification pipeline
"""

import os
import sys
import argparse
from datetime import datetime

# Import modules
from config import Config
from src.data_preparation import DataPreparation
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import MosquitoPredictor


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print(" " * 15 + "MOSQUITO SPECIES CLASSIFICATION")
    print(" " * 10 + "Aedes aegypti vs Aedes albopictus")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def run_data_preparation():
    """Run data preparation step"""
    print("\n📋 STEP 1: DATA PREPARATION")
    print("-" * 70)
    
    data_prep = DataPreparation()
    success = data_prep.split_and_copy_data(clean_first=True)
    
    if success:
        data_prep.verify_split()
        print("\n✅ Data preparation completed successfully!")
        return True
    else:
        print("\n❌ Data preparation failed!")
        return False


def run_training():
    """Run model training step"""
    print("\n🧠 STEP 2: MODEL TRAINING")
    print("-" * 70)
    
    trainer = ModelTrainer()
    model, history, model_path = trainer.train()
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved at: {model_path}")
    return True


def run_evaluation():
    """Run model evaluation step"""
    print("\n📊 STEP 3: MODEL EVALUATION")
    print("-" * 70)
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate()
    
    print("\n✅ Evaluation completed successfully!")
    print(f"Results saved in: {Config.EVALUATION_DIR}")
    return True


def run_prediction(image_path=None, folder_path=None, show_plot=False):
    """Run prediction step"""
    print("\n🔮 STEP 4: PREDICTION")
    print("-" * 70)
    
    predictor = MosquitoPredictor()
    
    if image_path:
        predictor.predict(image_path, save_result=True, show_plot=show_plot)
    elif folder_path:
        predictor.predict_batch(folder_path, save_results=True)
    else:
        print("❌ Please provide either --image or --folder argument")
        return False
    
    print("\n✅ Prediction completed successfully!")
    return True


def run_full_pipeline():
    """Run the complete pipeline"""
    print_banner()
    
    print("🚀 RUNNING COMPLETE PIPELINE")
    print("="*70)
    print("This will:")
    print("  1. Prepare and split your dataset")
    print("  2. Train the model (this may take 2-5 hours on CPU)")
    print("  3. Evaluate the model on test set")
    print("\n⏳ Estimated time: 2-6 hours (CPU) or 30-90 minutes (GPU)")
    print("="*70)
    
    # Confirm
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Pipeline cancelled.")
        return False
    
    # Step 1: Data Preparation
    if not run_data_preparation():
        return False
    
    # Step 2: Training
    if not run_training():
        return False
    
    # Step 3: Evaluation
    if not run_evaluation():
        return False
    
    print("\n" + "="*70)
    print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Check these folders for results:")
    print(f"  Models: {Config.SAVED_MODELS_DIR}")
    print(f"  Plots: {Config.PLOTS_DIR}")
    print(f"  Evaluation: {Config.EVALUATION_DIR}")
    print("\n💡 Next step: Use the model to predict new images:")
    print("  python run_pipeline.py --mode predict --image path/to/mosquito.jpg")
    print("="*70 + "\n")
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Mosquito Species Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --mode full
  
  # Run individual steps
  python run_pipeline.py --mode prepare
  python run_pipeline.py --mode train
  python run_pipeline.py --mode evaluate
  
  # Predict single image
  python run_pipeline.py --mode predict --image path/to/mosquito.jpg
  
  # Predict batch of images
  python run_pipeline.py --mode predict --folder path/to/images/
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['full', 'prepare', 'train', 'evaluate', 'predict'],
        help='Pipeline mode to run'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file (for predict mode)'
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        help='Path to folder containing images (for predict mode)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show prediction plot (for predict mode)'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    Config.create_directories()
    
    # Run selected mode
    try:
        if args.mode == 'full':
            success = run_full_pipeline()
        
        elif args.mode == 'prepare':
            print_banner()
            success = run_data_preparation()
        
        elif args.mode == 'train':
            print_banner()
            success = run_training()
        
        elif args.mode == 'evaluate':
            print_banner()
            success = run_evaluation()
        
        elif args.mode == 'predict':
            print_banner()
            success = run_prediction(
                image_path=args.image,
                folder_path=args.folder,
                show_plot=args.show
            )
        
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())