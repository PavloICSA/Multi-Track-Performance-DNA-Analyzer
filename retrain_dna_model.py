#!/usr/bin/env python3
"""
DNA Model Retraining Script
Retrain models with new data while preserving previous versions
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
import logging

from dna_model_trainer import DNAModelTrainer
from dna_model_registry import DNAModelRegistry
from config import MODELS_DIR, get_model_dir, TRAINING_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DNAModelRetrainer:
    """
    Handles model retraining with version management and backup
    """
    
    def __init__(self, data_dir: str = ".", models_dir: Path = None):
        """
        Initialize retrainer
        
        Args:
            data_dir: Directory containing track data
            models_dir: Directory for model storage
        """
        self.data_dir = Path(data_dir)
        self.models_dir = models_dir if models_dir else MODELS_DIR
        self.registry = DNAModelRegistry()
        
        logger.info(f"Initialized DNAModelRetrainer")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Models directory: {self.models_dir}")
    
    def get_next_version(self, current_version: str, bump: str = "minor") -> str:
        """
        Calculate next version number
        
        Args:
            current_version: Current version string (e.g., "1.0.0")
            bump: Version component to bump ("major", "minor", or "patch")
            
        Returns:
            Next version string
        """
        try:
            major, minor, patch = map(int, current_version.split('.'))
        except ValueError:
            logger.warning(f"Invalid version format: {current_version}, using 1.0.0")
            major, minor, patch = 1, 0, 0
        
        if bump == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump == "minor":
            minor += 1
            patch = 0
        elif bump == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid bump type: {bump}. Use 'major', 'minor', or 'patch'")
        
        return f"{major}.{minor}.{patch}"
    
    def backup_model(self, version: str) -> Path:
        """
        Create backup of existing model
        
        Args:
            version: Version to backup
            
        Returns:
            Path to backup directory
        """
        model_dir = get_model_dir(version)
        
        if not model_dir.exists():
            logger.warning(f"Model {version} not found, skipping backup")
            return None
        
        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.models_dir / f"backup_{version}_{timestamp}"
        
        logger.info(f"Creating backup: {backup_dir}")
        shutil.copytree(model_dir, backup_dir)
        
        logger.info(f"✅ Backup created: {backup_dir}")
        
        return backup_dir
    
    def retrain_from_scratch(
        self,
        new_version: str = None,
        epochs: int = None,
        backup_current: bool = True
    ) -> str:
        """
        Retrain model from scratch with new data
        
        Args:
            new_version: Version for new model (auto-increments if None)
            epochs: Number of training epochs (uses config default if None)
            backup_current: Whether to backup current production model
            
        Returns:
            Path to new model directory
        """
        logger.info("=" * 60)
        logger.info("Starting model retraining from scratch")
        logger.info("=" * 60)
        
        # Determine new version
        if new_version is None:
            # Get current production model version
            prod_model = self.registry.get_production_model()
            if prod_model:
                current_version = prod_model["version"]
                new_version = self.get_next_version(current_version, bump="minor")
                logger.info(f"Auto-incrementing version: {current_version} → {new_version}")
            else:
                # No production model, start with 1.0.0
                new_version = "1.0.0"
                logger.info(f"No production model found, starting with version {new_version}")
        
        # Backup current production model if requested
        if backup_current:
            prod_model = self.registry.get_production_model()
            if prod_model:
                self.backup_model(prod_model["version"])
        
        # Initialize trainer
        logger.info(f"\n📦 Initializing trainer for version {new_version}")
        trainer = DNAModelTrainer(data_dir=str(self.data_dir), models_dir=str(self.models_dir))
        
        # Load data
        logger.info("\n📂 Loading training data...")
        trainer.load_all_track_data()
        
        # Prepare data
        training_data = trainer.prepare_training_data()
        trainer.split_data(training_data)
        
        # Build models
        trainer.build_models()
        
        # Train models
        logger.info(f"\n🚀 Training models...")
        trainer.train_models(epochs=epochs)
        
        # Validate models
        logger.info(f"\n📊 Validating models...")
        metrics = trainer.validate_models()
        
        # Check if model meets threshold
        min_threshold = TRAINING_CONFIG['min_reliability_score']
        if metrics['overall_reliability'] < min_threshold:
            logger.error(f"❌ Model does not meet reliability threshold!")
            logger.error(f"   Required: {min_threshold:.2%}, Achieved: {metrics['overall_reliability']:.2%}")
            logger.error(f"   Model will be saved but NOT promoted to production")
        
        # Perform cross-validation
        logger.info(f"\n🔄 Performing cross-validation...")
        cv_results = trainer.cross_validate()
        
        # Save model artifacts
        logger.info(f"\n💾 Saving model artifacts...")
        model_dir = trainer.save_model_artifacts(version=new_version)
        
        # Generate validation report
        logger.info(f"\n📄 Generating validation report...")
        report_path = trainer.generate_validation_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Retraining complete!")
        logger.info("=" * 60)
        logger.info(f"Model version: {new_version}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Validation report: {report_path}")
        logger.info(f"Overall reliability: {metrics['overall_reliability']:.2%}")
        
        return model_dir
    
    def retrain_incremental(
        self,
        base_version: str,
        new_version: str = None,
        epochs: int = None
    ) -> str:
        """
        Retrain model incrementally from existing model (fine-tuning)
        
        Args:
            base_version: Version of model to start from
            new_version: Version for new model (auto-increments if None)
            epochs: Number of training epochs (uses config default if None)
            
        Returns:
            Path to new model directory
        """
        logger.info("=" * 60)
        logger.info("Starting incremental model retraining (fine-tuning)")
        logger.info("=" * 60)
        
        # Check if base model exists
        base_model = self.registry.get_model_by_version(base_version)
        if not base_model:
            raise ValueError(f"Base model version {base_version} not found in registry")
        
        base_model_dir = Path(base_model["model_dir"])
        if not base_model_dir.exists():
            raise ValueError(f"Base model directory not found: {base_model_dir}")
        
        logger.info(f"Base model: {base_version}")
        logger.info(f"Base model directory: {base_model_dir}")
        
        # Determine new version
        if new_version is None:
            new_version = self.get_next_version(base_version, bump="patch")
            logger.info(f"Auto-incrementing version: {base_version} → {new_version}")
        
        # Create backup of base model
        self.backup_model(base_version)
        
        # Initialize trainer
        logger.info(f"\n📦 Initializing trainer for version {new_version}")
        trainer = DNAModelTrainer(data_dir=str(self.data_dir), models_dir=str(self.models_dir))
        
        # Load data
        logger.info("\n📂 Loading training data...")
        trainer.load_all_track_data()
        
        # Prepare data
        training_data = trainer.prepare_training_data()
        trainer.split_data(training_data)
        
        # Build models
        trainer.build_models()
        
        # Load weights from base model
        logger.info(f"\n📥 Loading weights from base model {base_version}...")
        import torch
        
        dna_model_path = base_model_dir / "dna_regression_model.pth"
        archetype_model_path = base_model_dir / "archetype_classifier.pth"
        
        if dna_model_path.exists():
            trainer.dna_model.load_state_dict(torch.load(dna_model_path, map_location=trainer.device))
            logger.info("   ✅ Loaded DNA regression model weights")
        else:
            logger.warning(f"   ⚠️  DNA model weights not found, training from scratch")
        
        if archetype_model_path.exists():
            trainer.archetype_model.load_state_dict(torch.load(archetype_model_path, map_location=trainer.device))
            logger.info("   ✅ Loaded archetype classifier weights")
        else:
            logger.warning(f"   ⚠️  Archetype model weights not found, training from scratch")
        
        # Fine-tune models with fewer epochs
        fine_tune_epochs = epochs if epochs else max(20, TRAINING_CONFIG['epochs'] // 5)
        logger.info(f"\n🚀 Fine-tuning models for {fine_tune_epochs} epochs...")
        trainer.train_models(epochs=fine_tune_epochs)
        
        # Validate models
        logger.info(f"\n📊 Validating models...")
        metrics = trainer.validate_models()
        
        # Check if model meets threshold
        min_threshold = TRAINING_CONFIG['min_reliability_score']
        if metrics['overall_reliability'] < min_threshold:
            logger.error(f"❌ Model does not meet reliability threshold!")
            logger.error(f"   Required: {min_threshold:.2%}, Achieved: {metrics['overall_reliability']:.2%}")
        
        # Save model artifacts
        logger.info(f"\n💾 Saving model artifacts...")
        model_dir = trainer.save_model_artifacts(version=new_version)
        
        # Generate validation report
        logger.info(f"\n📄 Generating validation report...")
        report_path = trainer.generate_validation_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Incremental retraining complete!")
        logger.info("=" * 60)
        logger.info(f"Base version: {base_version}")
        logger.info(f"New version: {new_version}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Validation report: {report_path}")
        logger.info(f"Overall reliability: {metrics['overall_reliability']:.2%}")
        
        return model_dir


def main():
    """Main retraining script"""
    parser = argparse.ArgumentParser(
        description="Retrain DNA model with new data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retrain from scratch with auto-versioning
  python retrain_dna_model.py --mode scratch
  
  # Retrain from scratch with specific version
  python retrain_dna_model.py --mode scratch --version 2.0.0
  
  # Incremental retraining (fine-tuning) from version 1.0.0
  python retrain_dna_model.py --mode incremental --base-version 1.0.0
  
  # Retrain with custom epochs
  python retrain_dna_model.py --mode scratch --epochs 50
  
  # Retrain without backing up current model
  python retrain_dna_model.py --mode scratch --no-backup
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["scratch", "incremental"],
        default="scratch",
        help="Retraining mode: 'scratch' (full retrain) or 'incremental' (fine-tuning)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing track data (default: current directory)"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version for new model (auto-increments if not specified)"
    )
    
    parser.add_argument(
        "--base-version",
        type=str,
        default=None,
        help="Base model version for incremental retraining (required for incremental mode)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (uses config default if not specified)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up current production model"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "incremental" and not args.base_version:
        parser.error("--base-version is required for incremental retraining mode")
    
    try:
        # Initialize retrainer
        retrainer = DNAModelRetrainer(data_dir=args.data_dir)
        
        # Perform retraining
        if args.mode == "scratch":
            model_dir = retrainer.retrain_from_scratch(
                new_version=args.version,
                epochs=args.epochs,
                backup_current=not args.no_backup
            )
        else:  # incremental
            model_dir = retrainer.retrain_incremental(
                base_version=args.base_version,
                new_version=args.version,
                epochs=args.epochs
            )
        
        logger.info(f"\n🎉 Success! New model saved to: {model_dir}")
        logger.info(f"\nTo promote this model to production, run:")
        logger.info(f"  python promote_model.py --version {Path(model_dir).name.replace('dna_model_', '')}")
        
    except Exception as e:
        logger.error(f"\n❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
