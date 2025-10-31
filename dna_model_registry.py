#!/usr/bin/env python3
"""
DNA Model Registry
Manages model versions, metadata, and selection
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from config import MODEL_REGISTRY_PATH, MODELS_DIR, get_model_dir

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DNAModelRegistry:
    """
    Manages model versions, metadata, and selection
    Tracks all trained models with their performance metrics
    """
    
    def __init__(self, registry_path: Path = None):
        """
        Initialize model registry
        
        Args:
            registry_path: Path to registry JSON file (uses default if None)
        """
        self.registry_path = registry_path if registry_path else MODEL_REGISTRY_PATH
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or create new one
        self.registry = self._load_registry()
        
        logger.info(f"Initialized DNAModelRegistry with {len(self.registry['models'])} models")
    
    def _load_registry(self) -> Dict[str, Any]:
        """
        Load registry from disk or create new one
        
        Returns:
            Registry dictionary
        """
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
                logger.info(f"Loaded registry from {self.registry_path}")
                return registry
            except json.JSONDecodeError as e:
                logger.error(f"Corrupted registry file: {e}")
                logger.info("Creating new registry")
        
        # Create new registry
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "models": []
        }
    
    def _save_registry(self):
        """Save registry to disk"""
        self.registry["last_updated"] = datetime.now().isoformat()
        
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        logger.info(f"Saved registry to {self.registry_path}")
    
    def register_model(
        self,
        version: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any] = None,
        is_production: bool = False
    ) -> Dict[str, Any]:
        """
        Register a new model with metadata
        
        Args:
            version: Model version string (e.g., "1.0.0")
            metrics: Dictionary with validation metrics
            metadata: Additional metadata (optional)
            is_production: Whether this is the production model
            
        Returns:
            Model entry dictionary
        """
        logger.info(f"Registering model version {version}")
        
        # Check if version already exists
        existing_model = self.get_model_by_version(version)
        if existing_model:
            logger.warning(f"Model version {version} already exists. Updating entry.")
            self.remove_model(version)
        
        # Create model entry
        model_entry = {
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "model_dir": str(get_model_dir(version)),
            "metrics": metrics,
            "is_production": is_production,
            "metadata": metadata or {}
        }
        
        # Add to registry
        self.registry["models"].append(model_entry)
        
        # If this is production, unmark other models
        if is_production:
            for model in self.registry["models"]:
                if model["version"] != version:
                    model["is_production"] = False
        
        # Save registry
        self._save_registry()
        
        logger.info(f"✅ Registered model {version} (production={is_production})")
        
        return model_entry
    
    def get_model_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get model entry by version
        
        Args:
            version: Model version string
            
        Returns:
            Model entry dictionary or None if not found
        """
        for model in self.registry["models"]:
            if model["version"] == version:
                return model
        return None
    
    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the current production model
        
        Returns:
            Production model entry or None if no production model set
        """
        for model in self.registry["models"]:
            if model.get("is_production", False):
                return model
        return None
    
    def get_best_model(self, metric: str = "overall_reliability") -> Optional[Dict[str, Any]]:
        """
        Get the best performing model based on a metric
        
        Args:
            metric: Metric name to compare (default: "overall_reliability")
            
        Returns:
            Best model entry or None if no models exist
        """
        if not self.registry["models"]:
            return None
        
        # Filter models that have the specified metric
        models_with_metric = [
            model for model in self.registry["models"]
            if metric in model.get("metrics", {})
        ]
        
        if not models_with_metric:
            logger.warning(f"No models found with metric '{metric}'")
            return None
        
        # Find model with highest metric value
        best_model = max(
            models_with_metric,
            key=lambda m: m["metrics"][metric]
        )
        
        return best_model
    
    def list_models(self, sort_by: str = "registered_at") -> List[Dict[str, Any]]:
        """
        List all registered models
        
        Args:
            sort_by: Field to sort by (default: "registered_at")
            
        Returns:
            List of model entries sorted by specified field
        """
        models = self.registry["models"].copy()
        
        # Sort models
        if sort_by == "registered_at":
            models.sort(key=lambda m: m.get("registered_at", ""), reverse=True)
        elif sort_by == "version":
            models.sort(key=lambda m: m.get("version", ""))
        elif sort_by in ["overall_reliability", "archetype_accuracy"]:
            models.sort(
                key=lambda m: m.get("metrics", {}).get(sort_by, 0),
                reverse=True
            )
        
        return models
    
    def compare_models(
        self,
        version1: str,
        version2: str,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two models by their metrics
        
        Args:
            version1: First model version
            version2: Second model version
            metrics: List of metrics to compare (uses all if None)
            
        Returns:
            Dictionary with comparison results
        """
        model1 = self.get_model_by_version(version1)
        model2 = self.get_model_by_version(version2)
        
        if not model1:
            raise ValueError(f"Model version {version1} not found")
        if not model2:
            raise ValueError(f"Model version {version2} not found")
        
        # Get metrics to compare
        if metrics is None:
            metrics = list(model1.get("metrics", {}).keys())
        
        # Compare metrics
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics": {},
            "winner": None
        }
        
        wins_v1 = 0
        wins_v2 = 0
        
        for metric in metrics:
            val1 = model1.get("metrics", {}).get(metric)
            val2 = model2.get("metrics", {}).get(metric)
            
            if val1 is None or val2 is None:
                comparison["metrics"][metric] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": None,
                    "winner": None
                }
                continue
            
            diff = val2 - val1
            winner = version2 if diff > 0 else version1 if diff < 0 else "tie"
            
            if winner == version1:
                wins_v1 += 1
            elif winner == version2:
                wins_v2 += 1
            
            comparison["metrics"][metric] = {
                "version1": val1,
                "version2": val2,
                "difference": diff,
                "percent_change": (diff / val1 * 100) if val1 != 0 else None,
                "winner": winner
            }
        
        # Determine overall winner
        if wins_v1 > wins_v2:
            comparison["winner"] = version1
        elif wins_v2 > wins_v1:
            comparison["winner"] = version2
        else:
            comparison["winner"] = "tie"
        
        return comparison
    
    def remove_model(self, version: str) -> bool:
        """
        Remove a model from the registry
        
        Args:
            version: Model version to remove
            
        Returns:
            True if model was removed, False if not found
        """
        initial_count = len(self.registry["models"])
        self.registry["models"] = [
            m for m in self.registry["models"]
            if m["version"] != version
        ]
        
        removed = len(self.registry["models"]) < initial_count
        
        if removed:
            self._save_registry()
            logger.info(f"Removed model {version} from registry")
        else:
            logger.warning(f"Model {version} not found in registry")
        
        return removed
    
    def set_production_model(self, version: str) -> bool:
        """
        Set a model as the production model
        
        Args:
            version: Model version to promote to production
            
        Returns:
            True if successful, False if model not found
        """
        model = self.get_model_by_version(version)
        
        if not model:
            logger.error(f"Model version {version} not found")
            return False
        
        # Unmark all models as production
        for m in self.registry["models"]:
            m["is_production"] = False
        
        # Mark specified model as production
        model["is_production"] = True
        
        self._save_registry()
        
        logger.info(f"✅ Set model {version} as production")
        
        return True
    
    def get_model_count(self) -> int:
        """Get total number of registered models"""
        return len(self.registry["models"])
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get registry information
        
        Returns:
            Dictionary with registry metadata
        """
        production_model = self.get_production_model()
        best_model = self.get_best_model()
        
        return {
            "total_models": len(self.registry["models"]),
            "created_at": self.registry.get("created_at"),
            "last_updated": self.registry.get("last_updated"),
            "production_model": production_model.get("version") if production_model else None,
            "best_model": best_model.get("version") if best_model else None,
            "registry_path": str(self.registry_path)
        }
    
    def export_registry(self, output_path: Path) -> str:
        """
        Export registry to a file
        
        Args:
            output_path: Path to export file
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        logger.info(f"Exported registry to {output_path}")
        
        return str(output_path)


def main():
    """Example usage of DNAModelRegistry"""
    print("🗂️  DNA Model Registry")
    print("=" * 50)
    
    # Initialize registry
    registry = DNAModelRegistry()
    
    # Display registry info
    info = registry.get_registry_info()
    print(f"\n📊 Registry Information:")
    print(f"   Total Models: {info['total_models']}")
    print(f"   Production Model: {info['production_model']}")
    print(f"   Best Model: {info['best_model']}")
    print(f"   Registry Path: {info['registry_path']}")
    
    # List all models
    if info['total_models'] > 0:
        print(f"\n📋 Registered Models:")
        models = registry.list_models(sort_by="registered_at")
        for model in models:
            prod_marker = "🌟" if model.get("is_production") else "  "
            reliability = model.get("metrics", {}).get("overall_reliability", 0)
            print(f"   {prod_marker} {model['version']} - Reliability: {reliability:.2%}")
    
    # Get production model
    prod_model = registry.get_production_model()
    if prod_model:
        print(f"\n🌟 Production Model: {prod_model['version']}")
        print(f"   Metrics:")
        for metric, value in prod_model.get("metrics", {}).items():
            print(f"      {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
