"""
Configuration management for ML model integration
Defines paths, hyperparameters, and model settings
"""

from pathlib import Path
from typing import Dict, Any

# Directory paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
TRAINING_ARTIFACTS_DIR = BASE_DIR / "training_artifacts"
DATA_DIR = BASE_DIR

# Track directories
TRACKS = ['barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']

# Model paths
LATEST_MODEL_DIR = MODELS_DIR / "latest"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# Model artifact filenames
DNA_REGRESSION_MODEL_FILE = "dna_regression_model.pth"
ARCHETYPE_CLASSIFIER_FILE = "archetype_classifier.pth"
FEATURE_SCALER_FILE = "feature_scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
METADATA_FILE = "metadata.json"
VALIDATION_REPORT_FILE = "validation_report.html"

# Training hyperparameters
TRAINING_CONFIG: Dict[str, Any] = {
    # Data splitting
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15,
    "random_state": 42,
    "stratify": True,
    
    # Model architecture - DNA Regression
    "dna_hidden_layers": [128, 64, 32],
    "dna_dropout_rate": 0.2,
    "dna_activation": "relu",
    
    # Model architecture - Archetype Classifier
    "archetype_hidden_layers": [64, 32],
    "archetype_dropout_rate": 0.3,
    "archetype_activation": "relu",
    
    # Training parameters
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "weight_decay": 1e-5,
    
    # Early stopping
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.001,
    
    # Learning rate scheduling
    "lr_scheduler": "ReduceLROnPlateau",
    "lr_patience": 10,
    "lr_factor": 0.5,
    
    # Cross-validation
    "cv_folds": 5,
    
    # Validation thresholds
    "min_reliability_score": 0.95,
    "max_cross_val_std": 0.02,
}

# Feature names
PERFORMANCE_FEATURES = [
    'avg_lap_time', 'std_lap_time', 'min_lap_time', 'lap_count',
    'avg_s1', 'std_s1', 'min_s1',
    'avg_s2', 'std_s2', 'min_s2',
    'avg_s3', 'std_s3', 'min_s3',
    'avg_speed', 'max_speed'
]

DNA_FEATURES = [
    'speed_vs_consistency_ratio',
    'track_adaptability',
    'consistency_index',
    'performance_variance',
    'speed_consistency'
]

ARCHETYPE_CLASSES = [
    'Speed Demon',
    'Consistency Master',
    'Track Specialist',
    'Balanced Racer'
]

# Track type classifications
TRACK_TYPES = {
    'barber': 'technical',
    'COTA': 'mixed',
    'Road America': 'high_speed',
    'Sebring': 'technical',
    'Sonoma': 'technical',
    'VIR': 'mixed'
}

# Required CSV columns for user data
REQUIRED_COLUMNS = ['NUMBER', 'LAP_TIME', 'S1', 'S2', 'S3', 'KPH', 'track']

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "dna_analyzer.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "standard",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["file", "console"],
    },
}

# Performance targets
PERFORMANCE_TARGETS = {
    "model_loading_time_seconds": 5,
    "inference_time_100_drivers_seconds": 10,
    "inference_time_1000_drivers_seconds": 60,
    "max_memory_usage_gb": 2,
    "concurrent_requests": 10,
}

# Model versioning
MODEL_VERSION_FORMAT = "v{major}.{minor}.{patch}"
INITIAL_MODEL_VERSION = "1.0.0"

def get_model_dir(version: str) -> Path:
    """Get the directory path for a specific model version"""
    return MODELS_DIR / f"dna_model_{version}"

def ensure_directories():
    """Create all required directories if they don't exist"""
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    TRAINING_ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # Create .gitkeep files to preserve empty directories in git
    for directory in [MODELS_DIR, LOGS_DIR, TRAINING_ARTIFACTS_DIR]:
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

# Initialize directories on import
ensure_directories()
