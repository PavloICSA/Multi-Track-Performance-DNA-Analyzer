#!/usr/bin/env python3
"""
DNA Model Promotion Script
Compare and promote models to production
"""

import argparse
import os
from pathlib import Path
import logging

from dna_model_registry import DNAModelRegistry
from config import MODELS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DNAModelPromoter:
    """
    Handles model promotion to production with comparison and validation
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize promoter
        
        Args:
            models_dir: Directory for model storage
        """
        self.models_dir = models_dir if models_dir else MODELS_DIR
        self.registry = DNAModelRegistry()
        
        logger.info(f"Initialized DNAModelPromoter")
        logger.info(f"Models directory: {self.models_dir}")
    
    def compare_with_production(self, candidate_version: str) -> dict:
        """
        Compare candidate model with current production model
        
        Args:
            candidate_version: Version of candidate model
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("=" * 60)
        logger.info("Comparing candidate model with production model")
        logger.info("=" * 60)
        
        # Get candidate model
        candidate = self.registry.get_model_by_version(candidate_version)
        if not candidate:
            raise ValueError(f"Candidate model {candidate_version} not found in registry")
        
        logger.info(f"\n📦 Candidate Model: {candidate_version}")
        logger.info(f"   Registered: {candidate.get('registered_at', 'unknown')}")
        
        # Get production model
        production = self.registry.get_production_model()
        
        if not production:
            logger.info(f"\n⚠️  No production model currently set")
            logger.info(f"   Candidate will be compared against minimum thresholds only")
            
            # Check if candidate meets minimum requirements
            metrics = candidate.get("metrics", {})
            reliability = metrics.get("overall_reliability", 0)
            
            from config import TRAINING_CONFIG
            min_threshold = TRAINING_CONFIG['min_reliability_score']
            
            if reliability >= min_threshold:
                logger.info(f"\n✅ Candidate meets minimum threshold:")
                logger.info(f"   Reliability: {reliability:.2%} (≥ {min_threshold:.2%})")
                return {
                    "candidate_version": candidate_version,
                    "production_version": None,
                    "comparison": None,
                    "recommendation": "promote",
                    "reason": f"No production model set and candidate meets threshold ({reliability:.2%} ≥ {min_threshold:.2%})"
                }
            else:
                logger.info(f"\n❌ Candidate does NOT meet minimum threshold:")
                logger.info(f"   Reliability: {reliability:.2%} (< {min_threshold:.2%})")
                return {
                    "candidate_version": candidate_version,
                    "production_version": None,
                    "comparison": None,
                    "recommendation": "reject",
                    "reason": f"Candidate does not meet threshold ({reliability:.2%} < {min_threshold:.2%})"
                }
        
        production_version = production["version"]
        logger.info(f"\n🌟 Production Model: {production_version}")
        logger.info(f"   Registered: {production.get('registered_at', 'unknown')}")
        
        # Compare models
        logger.info(f"\n📊 Comparing metrics...")
        comparison = self.registry.compare_models(
            candidate_version,
            production_version,
            metrics=["overall_reliability", "archetype_accuracy", "dna_r2", "dna_mae"]
        )
        
        # Display comparison
        logger.info(f"\n{'Metric':<25} {'Candidate':<15} {'Production':<15} {'Difference':<15} {'Winner':<10}")
        logger.info("-" * 80)
        
        for metric, values in comparison["metrics"].items():
            val1 = values["version1"]
            val2 = values["version2"]
            diff = values["difference"]
            winner = values["winner"]
            
            if val1 is not None and val2 is not None:
                # Format based on metric type
                if metric == "dna_mae":
                    # Lower is better for MAE
                    val1_str = f"{val1:.4f}"
                    val2_str = f"{val2:.4f}"
                    diff_str = f"{diff:+.4f}"
                    winner_display = "🏆 " + winner if winner != "tie" else "tie"
                else:
                    # Higher is better for other metrics
                    val1_str = f"{val1:.4f}"
                    val2_str = f"{val2:.4f}"
                    diff_str = f"{diff:+.4f}"
                    winner_display = "🏆 " + winner if winner != "tie" else "tie"
                
                logger.info(f"{metric:<25} {val1_str:<15} {val2_str:<15} {diff_str:<15} {winner_display:<10}")
            else:
                logger.info(f"{metric:<25} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
        
        # Determine recommendation
        overall_winner = comparison["winner"]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Overall Winner: {overall_winner}")
        logger.info(f"{'='*80}")
        
        if overall_winner == candidate_version:
            recommendation = "promote"
            reason = f"Candidate outperforms production model on majority of metrics"
        elif overall_winner == production_version:
            recommendation = "keep_production"
            reason = f"Production model still performs better on majority of metrics"
        else:
            # Tie - use overall_reliability as tiebreaker
            cand_reliability = candidate.get("metrics", {}).get("overall_reliability", 0)
            prod_reliability = production.get("metrics", {}).get("overall_reliability", 0)
            
            if cand_reliability > prod_reliability:
                recommendation = "promote"
                reason = f"Tie on metrics, but candidate has higher reliability ({cand_reliability:.2%} vs {prod_reliability:.2%})"
            else:
                recommendation = "keep_production"
                reason = f"Tie on metrics, production has equal or higher reliability"
        
        return {
            "candidate_version": candidate_version,
            "production_version": production_version,
            "comparison": comparison,
            "recommendation": recommendation,
            "reason": reason
        }
    
    def promote_model(self, version: str, force: bool = False) -> bool:
        """
        Promote a model to production
        
        Args:
            version: Version to promote
            force: Force promotion without comparison
            
        Returns:
            True if promotion successful
        """
        logger.info("=" * 60)
        logger.info(f"Promoting model {version} to production")
        logger.info("=" * 60)
        
        # Check if model exists
        model = self.registry.get_model_by_version(version)
        if not model:
            logger.error(f"❌ Model {version} not found in registry")
            return False
        
        # Check if model directory exists
        model_dir = Path(model["model_dir"])
        if not model_dir.exists():
            logger.error(f"❌ Model directory not found: {model_dir}")
            return False
        
        # Compare with production if not forcing
        if not force:
            comparison_result = self.compare_with_production(version)
            
            if comparison_result["recommendation"] != "promote":
                logger.warning(f"\n⚠️  Comparison recommends: {comparison_result['recommendation']}")
                logger.warning(f"   Reason: {comparison_result['reason']}")
                logger.warning(f"\n   Use --force to promote anyway")
                return False
            
            logger.info(f"\n✅ Comparison recommends promotion")
            logger.info(f"   Reason: {comparison_result['reason']}")
        else:
            logger.warning(f"\n⚠️  Forcing promotion without comparison")
        
        # Set as production in registry
        logger.info(f"\n📝 Updating registry...")
        success = self.registry.set_production_model(version)
        
        if not success:
            logger.error(f"❌ Failed to update registry")
            return False
        
        # Update 'latest' symlink
        logger.info(f"\n🔗 Updating 'latest' symlink...")
        latest_link = self.models_dir / "latest"
        
        # Remove existing symlink/directory
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_symlink():
                latest_link.unlink()
            else:
                logger.warning(f"   'latest' is a directory, not a symlink. Removing...")
                import shutil
                shutil.rmtree(latest_link)
        
        # Create new symlink
        try:
            # Use relative path for symlink
            os.symlink(model_dir.name, latest_link, target_is_directory=True)
            logger.info(f"   ✅ Updated 'latest' → {model_dir.name}")
        except (OSError, NotImplementedError) as e:
            logger.warning(f"   ⚠️  Could not create symlink: {e}")
            logger.warning(f"   This may require admin rights on Windows")
            logger.warning(f"   You can manually update the 'latest' directory")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✅ Model {version} promoted to production!")
        logger.info("=" * 60)
        
        return True
    
    def list_models(self):
        """List all models with their status"""
        logger.info("=" * 60)
        logger.info("Registered Models")
        logger.info("=" * 60)
        
        models = self.registry.list_models(sort_by="registered_at")
        
        if not models:
            logger.info("\nNo models registered yet")
            return
        
        production_version = None
        prod_model = self.registry.get_production_model()
        if prod_model:
            production_version = prod_model["version"]
        
        logger.info(f"\n{'Version':<15} {'Status':<15} {'Reliability':<15} {'Registered':<25}")
        logger.info("-" * 70)
        
        for model in models:
            version = model["version"]
            is_prod = model.get("is_production", False)
            status = "🌟 PRODUCTION" if is_prod else "   Available"
            reliability = model.get("metrics", {}).get("overall_reliability", 0)
            registered = model.get("registered_at", "unknown")[:19]  # Trim to datetime
            
            logger.info(f"{version:<15} {status:<15} {reliability:<15.2%} {registered:<25}")
        
        logger.info("")


def main():
    """Main promotion script"""
    parser = argparse.ArgumentParser(
        description="Promote DNA model to production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all models
  python promote_model.py --list
  
  # Compare candidate with production
  python promote_model.py --compare --version 1.1.0
  
  # Promote model (with automatic comparison)
  python promote_model.py --promote --version 1.1.0
  
  # Force promotion without comparison
  python promote_model.py --promote --version 1.1.0 --force
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered models"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare candidate model with production"
    )
    
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote model to production"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        help="Model version to compare or promote"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force promotion without comparison"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.list, args.compare, args.promote]):
        parser.error("Must specify one of: --list, --compare, or --promote")
    
    if (args.compare or args.promote) and not args.version:
        parser.error("--version is required for --compare and --promote")
    
    try:
        promoter = DNAModelPromoter()
        
        if args.list:
            promoter.list_models()
        
        elif args.compare:
            result = promoter.compare_with_production(args.version)
            logger.info(f"\n📋 Recommendation: {result['recommendation'].upper()}")
            logger.info(f"   {result['reason']}")
        
        elif args.promote:
            success = promoter.promote_model(args.version, force=args.force)
            if not success:
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
