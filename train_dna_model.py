#!/usr/bin/env python3
"""
DNA Model Training Script
Command-line interface for training DNA prediction models
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

from dna_model_trainer import DNAModelTrainer
from config import (
    TRAINING_CONFIG, MODELS_DIR, LOGS_DIR,
    INITIAL_MODEL_VERSION
)


def setup_logging(verbose: bool = False):
    """Configure logging for training script"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Configure logging
    log_file = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train DNA prediction models for driver performance analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_dna_model.py
  
  # Train with custom data directory and epochs
  python train_dna_model.py --data-dir ./racing_data --epochs 150
  
  # Train with validation and report generation
  python train_dna_model.py --validate --report --save
  
  # Train with custom output directory and version
  python train_dna_model.py --output-dir ./my_models --version 2.0.0
  
  # Train with all options
  python train_dna_model.py --data-dir ./data --output-dir ./models \\
      --epochs 200 --batch-size 64 --learning-rate 0.0005 \\
      --validate --report --save --verbose
        """
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory containing track data folders (default: current directory)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Directory to save trained models (default: {MODELS_DIR})'
    )
    
    # Model version
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help=f'Model version string (default: {INITIAL_MODEL_VERSION})'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of training epochs (default: {TRAINING_CONFIG["epochs"]})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Training batch size (default: {TRAINING_CONFIG["batch_size"]})'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help=f'Learning rate for optimizer (default: {TRAINING_CONFIG["learning_rate"]})'
    )
    
    # Cross-validation
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=None,
        help=f'Number of cross-validation folds (default: {TRAINING_CONFIG["cv_folds"]})'
    )
    
    # Actions
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation on test set after training'
    )
    
    parser.add_argument(
        '--cross-validate',
        action='store_true',
        help='Perform k-fold cross-validation after training'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate HTML validation report'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save model artifacts to disk'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    
    return parser.parse_args()


def update_training_config(args):
    """Update training configuration with command-line arguments"""
    if args.epochs is not None:
        TRAINING_CONFIG['epochs'] = args.epochs
    
    if args.batch_size is not None:
        TRAINING_CONFIG['batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        TRAINING_CONFIG['learning_rate'] = args.learning_rate
    
    if args.cv_folds is not None:
        TRAINING_CONFIG['cv_folds'] = args.cv_folds


def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("=" * 70)
    logger.info("🧬 DNA Model Training Pipeline")
    logger.info("=" * 70)
    
    # Update configuration
    update_training_config(args)
    
    # Log configuration
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir or MODELS_DIR}")
    logger.info(f"Model version: {args.version or INITIAL_MODEL_VERSION}")
    logger.info(f"Training epochs: {TRAINING_CONFIG['epochs']}")
    logger.info(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    logger.info(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    
    try:
        # Initialize trainer
        logger.info("\n📦 Initializing trainer...")
        trainer = DNAModelTrainer(
            data_dir=args.data_dir,
            models_dir=args.output_dir
        )
        
        # Disable CUDA if requested
        if args.no_cuda:
            import torch
            trainer.device = torch.device('cpu')
            logger.info("   CUDA disabled, using CPU")
        
        # Step 1: Load data
        logger.info("\n📂 Loading track data...")
        trainer.load_all_track_data()
        logger.info(f"   ✅ Loaded {len(trainer.raw_data)} total rows")
        
        # Step 2: Prepare training data
        logger.info("\n🔧 Preparing training data...")
        training_data = trainer.prepare_training_data()
        logger.info(f"   ✅ Prepared {len(training_data)} training samples")
        
        # Step 3: Split data
        logger.info("\n✂️  Splitting data...")
        trainer.split_data(training_data)
        logger.info(f"   ✅ Train: {len(trainer.X_train)}, "
                   f"Val: {len(trainer.X_val)}, "
                   f"Test: {len(trainer.X_test)}")
        
        # Step 4: Build models
        logger.info("\n🏗️  Building models...")
        trainer.build_models()
        logger.info("   ✅ Models built successfully")
        
        # Step 5: Train models
        logger.info(f"\n🚀 Training models for {TRAINING_CONFIG['epochs']} epochs...")
        logger.info("   (This may take several minutes...)")
        training_history = trainer.train_models()
        logger.info("   ✅ Training complete!")
        
        # Step 6: Validate (if requested or if saving)
        metrics = None
        if args.validate or args.save or args.report:
            logger.info("\n📊 Validating models on test set...")
            metrics = trainer.validate_models()
            
            logger.info("\n   Validation Results:")
            logger.info(f"   DNA Regression - MAE: {metrics['dna_mae']:.4f}, R²: {metrics['dna_r2']:.4f}")
            logger.info(f"   Archetype Classification:")
            logger.info(f"      Accuracy:  {metrics['archetype_accuracy']:.4f}")
            logger.info(f"      Precision: {metrics['archetype_precision']:.4f}")
            logger.info(f"      Recall:    {metrics['archetype_recall']:.4f}")
            logger.info(f"      F1 Score:  {metrics['archetype_f1']:.4f}")
            logger.info(f"   Overall Reliability: {metrics['overall_reliability']:.4f}")
            
            # Check if meets threshold
            min_threshold = TRAINING_CONFIG['min_reliability_score']
            if metrics['overall_reliability'] >= min_threshold:
                logger.info(f"   ✅ Model meets reliability threshold (≥{min_threshold})")
            else:
                logger.warning(f"   ⚠️  Model does not meet reliability threshold (≥{min_threshold})")
                logger.warning("   Consider retraining with different hyperparameters")
        
        # Step 7: Cross-validation (if requested)
        if args.cross_validate:
            logger.info(f"\n🔄 Performing {TRAINING_CONFIG['cv_folds']}-fold cross-validation...")
            logger.info("   (This may take several minutes...)")
            cv_results = trainer.cross_validate()
            
            logger.info("\n   Cross-Validation Results:")
            for metric, values in cv_results.items():
                import numpy as np
                mean_val = np.mean(values)
                std_val = np.std(values)
                logger.info(f"      {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Step 8: Save model artifacts (if requested)
        model_dir = None
        if args.save:
            logger.info("\n💾 Saving model artifacts...")
            version = args.version or INITIAL_MODEL_VERSION
            model_dir = trainer.save_model_artifacts(version=version)
            logger.info(f"   ✅ Model saved to: {model_dir}")
        
        # Step 9: Generate validation report (if requested)
        if args.report:
            logger.info("\n📄 Generating validation report...")
            if model_dir:
                report_path = Path(model_dir) / "validation_report.html"
            else:
                report_path = MODELS_DIR / "validation_report.html"
            
            report_path = trainer.generate_validation_report(str(report_path))
            logger.info(f"   ✅ Report saved to: {report_path}")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ Training pipeline completed successfully!")
        logger.info("=" * 70)
        
        if metrics:
            logger.info(f"\nFinal Reliability Score: {metrics['overall_reliability']:.2%}")
        
        if model_dir:
            logger.info(f"Model Location: {model_dir}")
        
        logger.info("\nNext steps:")
        if not args.save:
            logger.info("  - Run with --save to save the trained model")
        if not args.report:
            logger.info("  - Run with --report to generate a validation report")
        if not args.cross_validate:
            logger.info("  - Run with --cross-validate to verify model consistency")
        if args.save:
            logger.info("  - Use the trained model in your application")
            logger.info("  - Run evaluate_model.py to test on new data")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
