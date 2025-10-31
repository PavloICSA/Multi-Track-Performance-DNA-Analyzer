#!/usr/bin/env python3
"""
DNA Model Inference
Loads pre-trained models and generates predictions for new user data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

from dna_feature_engineering import DNAFeatureEngineering
from dna_model_trainer import DNARegressionModel, ArchetypeClassifier
from dna_logging import get_logger, DNALogger, ErrorContext
from dna_exceptions import (
    ModelLoadingError, DataValidationError, PredictionError,
    ModelIntegrityError, MissingColumnsError, InvalidDataFormatError,
    EmptyDataError, format_error_message
)
from config import (
    MODELS_DIR, DNA_REGRESSION_MODEL_FILE, ARCHETYPE_CLASSIFIER_FILE,
    FEATURE_SCALER_FILE, LABEL_ENCODER_FILE, METADATA_FILE,
    PERFORMANCE_FEATURES, DNA_FEATURES, ARCHETYPE_CLASSES,
    REQUIRED_COLUMNS
)

# Set up logger
logger = get_logger(__name__)


class DNAModelInference:
    """
    Loads pre-trained models and generates predictions for new user data
    """
    
    def __init__(self, model_dir: Union[str, Path] = None, fallback_on_error: bool = False):
        """
        Initialize inference system
        
        Args:
            model_dir: Path to model artifacts directory (uses 'models/latest' if None)
            fallback_on_error: If True, don't raise errors on model loading failure
            
        Raises:
            ModelLoadingError: If model loading fails and fallback_on_error is False
        """
        if model_dir is None:
            model_dir = MODELS_DIR / "latest"
        
        self.model_dir = Path(model_dir)
        self.feature_engineering = DNAFeatureEngineering()
        self.fallback_on_error = fallback_on_error
        
        # Model components
        self.dna_model = None
        self.archetype_model = None
        self.feature_scaler = None
        self.label_encoder = None
        self.metadata = None
        self.model_loaded = False
        
        # Device configuration (only for PyTorch models)
        self.device = None
        self.model_type = None  # Will be set after loading metadata
        
        # Load models on initialization
        logger.info(f"Initializing DNAModelInference with model directory: {self.model_dir}")
        
        try:
            self.load_model_artifacts()
            self.verify_model_integrity()
            self.model_loaded = True
            DNALogger.log_info(logger, 'I002', f"Model loaded successfully from {self.model_dir}")
        except Exception as e:
            if fallback_on_error:
                DNALogger.log_warning(logger, 'W002', f"Model loading failed, fallback mode enabled: {e}")
                logger.warning("Inference system initialized in fallback mode")
            else:
                raise
    
    def load_model_artifacts(self):
        """
        Load all model artifacts from disk
        
        Raises:
            ModelLoadingError: If model directory or required files don't exist or are corrupted
        """
        logger.info("Loading model artifacts...")
        
        # Check if model directory exists
        if not self.model_dir.exists():
            error_msg = f"Model directory not found: {self.model_dir}"
            DNALogger.log_error(logger, 'E001', error_msg)
            raise ModelLoadingError(
                error_msg,
                model_path=str(self.model_dir),
                fallback_available=self.fallback_on_error
            )
        
        # Load metadata first to get model configuration
        metadata_path = self.model_dir / METADATA_FILE
        if not metadata_path.exists():
            error_msg = f"Metadata file not found: {metadata_path}"
            DNALogger.log_error(logger, 'E002', error_msg)
            raise ModelLoadingError(
                error_msg,
                model_path=str(metadata_path),
                fallback_available=self.fallback_on_error
            )
        
        try:
            with ErrorContext(logger, "Loading metadata", 'E003'):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"   ✅ Loaded metadata (version {self.metadata.get('version', 'unknown')})")
        except json.JSONDecodeError as e:
            error_msg = f"Corrupted metadata file: {e}"
            DNALogger.log_error(logger, 'E003', error_msg, exception=e)
            raise ModelLoadingError(
                error_msg,
                model_path=str(metadata_path),
                fallback_available=self.fallback_on_error
            )
        except Exception as e:
            if not isinstance(e, ModelLoadingError):
                raise ModelLoadingError(
                    f"Failed to load metadata: {e}",
                    model_path=str(metadata_path),
                    fallback_available=self.fallback_on_error
                )
            raise
        
        # Validate version compatibility
        self._validate_version_compatibility()
        
        # Load feature scaler
        scaler_path = self.model_dir / FEATURE_SCALER_FILE
        if not scaler_path.exists():
            error_msg = f"Feature scaler not found: {scaler_path}"
            DNALogger.log_error(logger, 'E004', error_msg)
            raise ModelLoadingError(error_msg, model_path=str(scaler_path), fallback_available=self.fallback_on_error)
        
        try:
            with ErrorContext(logger, "Loading feature scaler", 'E005'):
                import joblib
                self.feature_scaler = joblib.load(scaler_path)
                logger.info(f"   ✅ Loaded feature scaler")
        except Exception as e:
            if not isinstance(e, ModelLoadingError):
                error_msg = f"Failed to load feature scaler: {e}"
                DNALogger.log_error(logger, 'E005', error_msg, exception=e)
                raise ModelLoadingError(error_msg, model_path=str(scaler_path), fallback_available=self.fallback_on_error)
            raise
        
        # Load label encoder
        encoder_path = self.model_dir / LABEL_ENCODER_FILE
        if not encoder_path.exists():
            error_msg = f"Label encoder not found: {encoder_path}"
            DNALogger.log_error(logger, 'E006', error_msg)
            raise ModelLoadingError(error_msg, model_path=str(encoder_path), fallback_available=self.fallback_on_error)
        
        try:
            with ErrorContext(logger, "Loading label encoder", 'E007'):
                import joblib
                self.label_encoder = joblib.load(encoder_path)
                logger.info(f"   ✅ Loaded label encoder")
        except Exception as e:
            if not isinstance(e, ModelLoadingError):
                error_msg = f"Failed to load label encoder: {e}"
                DNALogger.log_error(logger, 'E007', error_msg, exception=e)
                raise ModelLoadingError(error_msg, model_path=str(encoder_path), fallback_available=self.fallback_on_error)
            raise
        
        # Check model type from metadata
        self.model_type = self.metadata.get('model_type', 'neural_network')
        framework = self.metadata.get('framework', 'pytorch')
        
        # Load DNA regression model
        if self.model_type == 'random_forest' or framework == 'sklearn':
            # Load sklearn model
            dna_model_path = self.model_dir / 'dna_regression_model.pkl'
            if not dna_model_path.exists():
                error_msg = f"DNA regression model not found: {dna_model_path}"
                DNALogger.log_error(logger, 'E008', error_msg)
                raise ModelLoadingError(error_msg, model_path=str(dna_model_path), fallback_available=self.fallback_on_error)
            
            try:
                with ErrorContext(logger, "Loading DNA regression model (sklearn)", 'E009'):
                    import joblib
                    self.dna_model = joblib.load(dna_model_path)
                    logger.info(f"   ✅ Loaded DNA regression model (Random Forest)")
            except Exception as e:
                if not isinstance(e, ModelLoadingError):
                    error_msg = f"Failed to load DNA regression model: {e}"
                    DNALogger.log_error(logger, 'E009', error_msg, exception=e)
                    raise ModelLoadingError(error_msg, model_path=str(dna_model_path), fallback_available=self.fallback_on_error)
                raise
        else:
            # Load PyTorch model
            dna_model_path = self.model_dir / DNA_REGRESSION_MODEL_FILE
            if not dna_model_path.exists():
                error_msg = f"DNA regression model not found: {dna_model_path}"
                DNALogger.log_error(logger, 'E008', error_msg)
                raise ModelLoadingError(error_msg, model_path=str(dna_model_path), fallback_available=self.fallback_on_error)
            
            try:
                with ErrorContext(logger, "Loading DNA regression model (PyTorch)", 'E009'):
                    # Get input dimension from metadata
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    input_dim = len(self.metadata.get('feature_names', PERFORMANCE_FEATURES))
                    self.dna_model = DNARegressionModel(input_dim=input_dim, output_dim=5)
                    self.dna_model.load_state_dict(torch.load(dna_model_path, map_location=self.device))
                    self.dna_model.to(self.device)
                    self.dna_model.eval()
                    logger.info(f"   ✅ Loaded DNA regression model (Neural Network)")
            except Exception as e:
                if not isinstance(e, ModelLoadingError):
                    error_msg = f"Failed to load DNA regression model: {e}"
                    DNALogger.log_error(logger, 'E009', error_msg, exception=e)
                    raise ModelLoadingError(error_msg, model_path=str(dna_model_path), fallback_available=self.fallback_on_error)
                raise
            raise
        
        # Load archetype classifier
        if self.model_type == 'random_forest' or framework == 'sklearn':
            # Load sklearn model
            archetype_model_path = self.model_dir / 'archetype_classifier.pkl'
            if not archetype_model_path.exists():
                error_msg = f"Archetype classifier not found: {archetype_model_path}"
                DNALogger.log_error(logger, 'E010', error_msg)
                raise ModelLoadingError(error_msg, model_path=str(archetype_model_path), fallback_available=self.fallback_on_error)
            
            try:
                with ErrorContext(logger, "Loading archetype classifier (sklearn)", 'E011'):
                    import joblib
                    self.archetype_model = joblib.load(archetype_model_path)
                    logger.info(f"   ✅ Loaded archetype classifier (Random Forest)")
            except Exception as e:
                if not isinstance(e, ModelLoadingError):
                    error_msg = f"Failed to load archetype classifier: {e}"
                    DNALogger.log_error(logger, 'E011', error_msg, exception=e)
                    raise ModelLoadingError(error_msg, model_path=str(archetype_model_path), fallback_available=self.fallback_on_error)
                raise
        else:
            # Load PyTorch model
            archetype_model_path = self.model_dir / ARCHETYPE_CLASSIFIER_FILE
            if not archetype_model_path.exists():
                error_msg = f"Archetype classifier not found: {archetype_model_path}"
                DNALogger.log_error(logger, 'E010', error_msg)
                raise ModelLoadingError(error_msg, model_path=str(archetype_model_path), fallback_available=self.fallback_on_error)
            
            try:
                with ErrorContext(logger, "Loading archetype classifier (PyTorch)", 'E011'):
                    self.archetype_model = ArchetypeClassifier(input_dim=5, output_dim=4)
                    self.archetype_model.load_state_dict(torch.load(archetype_model_path, map_location=self.device))
                    self.archetype_model.to(self.device)
                    self.archetype_model.eval()
                    logger.info(f"   ✅ Loaded archetype classifier (Neural Network)")
            except Exception as e:
                if not isinstance(e, ModelLoadingError):
                    error_msg = f"Failed to load archetype classifier: {e}"
                    DNALogger.log_error(logger, 'E011', error_msg, exception=e)
                    raise ModelLoadingError(error_msg, model_path=str(archetype_model_path), fallback_available=self.fallback_on_error)
                raise
        
        logger.info("✅ All model artifacts loaded successfully")
    
    def _validate_version_compatibility(self):
        """
        Validate that model version is compatible with current code
        
        Raises:
            ModelLoadingError: If version is incompatible
        """
        # Check for required metadata fields
        required_fields = ['version', 'feature_names', 'target_names', 'archetype_classes']
        missing_fields = [field for field in required_fields if field not in self.metadata]
        
        if missing_fields:
            error_msg = f"Metadata missing required fields: {missing_fields}"
            DNALogger.log_error(logger, 'E012', error_msg)
            raise ModelLoadingError(
                error_msg,
                model_path=str(self.model_dir / METADATA_FILE),
                fallback_available=self.fallback_on_error
            )
        
        # Validate feature names match expected
        expected_features = set(PERFORMANCE_FEATURES)
        model_features = set(self.metadata['feature_names'])
        
        if expected_features != model_features:
            warning_msg = (
                f"Feature mismatch - Expected: {expected_features}, "
                f"Model has: {model_features}"
            )
            DNALogger.log_warning(logger, 'W001', warning_msg)
            # Don't raise error, just warn - model might still work
        
        logger.info(f"   ✅ Version compatibility validated")
    
    def verify_model_integrity(self) -> bool:
        """
        Verify model loaded correctly by performing test inference
        
        Returns:
            True if model passes integrity checks
            
        Raises:
            ModelIntegrityError: If model fails integrity checks
        """
        logger.info("Verifying model integrity...")
        
        try:
            with ErrorContext(logger, "Model integrity verification", 'E018'):
                # Create synthetic test data
                input_dim = len(self.metadata.get('feature_names', PERFORMANCE_FEATURES))
                synthetic_data = np.random.randn(5, input_dim).astype(np.float32)
                
                # Scale the data
                synthetic_data_scaled = self.feature_scaler.transform(synthetic_data)
                
                # Test DNA regression model
                if self.model_type == 'random_forest':
                    dna_predictions = self.dna_model.predict(synthetic_data_scaled)
                else:
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(synthetic_data_scaled).to(self.device)
                        dna_predictions = self.dna_model(X_tensor).cpu().numpy()
                
                # Check output shape
                expected_shape = (5, 5)  # 5 samples, 5 DNA features
                if dna_predictions.shape != expected_shape:
                    error_msg = f"DNA model output shape mismatch: expected {expected_shape}, got {dna_predictions.shape}"
                    DNALogger.log_error(logger, 'E013', error_msg)
                    raise ModelIntegrityError(error_msg, check_type="output_shape")
                
                # Check for NaN or infinite values
                if np.isnan(dna_predictions).any():
                    error_msg = "DNA model produced NaN values"
                    DNALogger.log_error(logger, 'E014', error_msg)
                    raise ModelIntegrityError(error_msg, check_type="nan_values")
                
                if np.isinf(dna_predictions).any():
                    error_msg = "DNA model produced infinite values"
                    DNALogger.log_error(logger, 'E015', error_msg)
                    raise ModelIntegrityError(error_msg, check_type="infinite_values")
                
                logger.info(f"   ✅ DNA model integrity verified")
                
                # Test archetype classifier
                if self.model_type == 'random_forest':
                    archetype_predictions = self.archetype_model.predict(dna_predictions)
                else:
                    with torch.no_grad():
                        dna_tensor = torch.FloatTensor(dna_predictions).to(self.device)
                        archetype_outputs = self.archetype_model(dna_tensor)
                        _, archetype_predictions = torch.max(archetype_outputs, 1)
                        archetype_predictions = archetype_predictions.cpu().numpy()
                
                # Check output shape
                expected_shape = (5,)  # 5 samples
                if archetype_predictions.shape != expected_shape:
                    error_msg = f"Archetype model output shape mismatch: expected {expected_shape}, got {archetype_predictions.shape}"
                    DNALogger.log_error(logger, 'E016', error_msg)
                    raise ModelIntegrityError(error_msg, check_type="output_shape")
                
                # Check value ranges (should be 0-3 for 4 classes)
                if archetype_predictions.min() < 0 or archetype_predictions.max() >= 4:
                    error_msg = f"Archetype predictions out of range: {archetype_predictions}"
                    DNALogger.log_error(logger, 'E017', error_msg)
                    raise ModelIntegrityError(error_msg, check_type="value_range")
                
                logger.info(f"   ✅ Archetype model integrity verified")
                logger.info("✅ Model integrity verification passed")
                
                return True
                
        except ModelIntegrityError:
            raise
        except Exception as e:
            error_msg = f"Model integrity verification failed: {e}"
            DNALogger.log_error(logger, 'E018', error_msg, exception=e)
            raise ModelIntegrityError(error_msg, check_type="unexpected_error")
    
    def validate_user_data(self, df: pd.DataFrame, allow_partial: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate user CSV data format
        
        Args:
            df: DataFrame with user data
            allow_partial: If True, allow partial data with warnings
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        warnings = []
        
        # Check if DataFrame is empty
        if df.empty:
            error_msg = "DataFrame is empty"
            DNALogger.log_error(logger, 'E051', error_msg)
            errors.append(error_msg)
            return False, errors
        
        # Check for required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            DNALogger.log_error(logger, 'E052', error_msg)
            errors.append(error_msg)
        
        # Check data types and value ranges
        if 'NUMBER' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['NUMBER']):
                try:
                    df['NUMBER'] = pd.to_numeric(df['NUMBER'], errors='coerce')
                    if df['NUMBER'].isna().all():
                        error_msg = "Column 'NUMBER' contains no valid numeric values"
                        DNALogger.log_error(logger, 'E056', error_msg)
                        errors.append(error_msg)
                    elif df['NUMBER'].isna().any():
                        warning_msg = f"Column 'NUMBER' has {df['NUMBER'].isna().sum()} invalid values"
                        DNALogger.log_warning(logger, 'W008', warning_msg)
                        warnings.append(warning_msg)
                except Exception as e:
                    error_msg = "Column 'NUMBER' must contain numeric driver IDs"
                    DNALogger.log_error(logger, 'E056', error_msg)
                    errors.append(error_msg)
        
        # Check time columns (can be string or numeric)
        time_columns = ['LAP_TIME', 'S1', 'S2', 'S3']
        for col in time_columns:
            if col in df.columns:
                # Try to convert a sample to check validity
                try:
                    sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample_value is not None:
                        self.feature_engineering.convert_time_to_seconds(sample_value)
                except Exception as e:
                    error_msg = f"Column '{col}' contains invalid time format: {e}"
                    DNALogger.log_error(logger, 'E054', error_msg)
                    errors.append(error_msg)
        
        # Check speed column
        if 'KPH' in df.columns:
            try:
                kph_numeric = pd.to_numeric(df['KPH'], errors='coerce')
                valid_speeds = kph_numeric[(kph_numeric >= 50) & (kph_numeric <= 400)]
                if len(valid_speeds) == 0:
                    error_msg = "Column 'KPH' contains no valid speed values (expected 50-400 km/h)"
                    DNALogger.log_error(logger, 'E055', error_msg)
                    errors.append(error_msg)
                elif len(valid_speeds) < len(df) * 0.5:
                    warning_msg = f"Column 'KPH' has too many invalid values ({len(df) - len(valid_speeds)} out of {len(df)})"
                    if allow_partial:
                        DNALogger.log_warning(logger, 'W008', warning_msg)
                        warnings.append(warning_msg)
                    else:
                        DNALogger.log_error(logger, 'E055', warning_msg)
                        errors.append(warning_msg)
            except Exception as e:
                error_msg = "Column 'KPH' must contain numeric speed values"
                DNALogger.log_error(logger, 'E055', error_msg)
                errors.append(error_msg)
        
        # Check track column
        if 'track' in df.columns:
            if df['track'].isna().all():
                error_msg = "Column 'track' contains no valid track names"
                DNALogger.log_error(logger, 'E057', error_msg)
                errors.append(error_msg)
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            DNALogger.log_warning(logger, 'W008', f"User data validation failed with {len(errors)} errors")
            logger.warning(f"Validation errors: {errors}")
        else:
            DNALogger.log_info(logger, 'I008', "User data validation passed")
            if warnings:
                logger.warning(f"Validation warnings: {warnings}")
        
        return is_valid, errors
    
    def predict_driver_dna(self, driver_features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict DNA signatures for drivers
        
        Args:
            driver_features: DataFrame with driver performance features
            
        Returns:
            DataFrame with DNA signature values
        """
        logger.info(f"Predicting DNA signatures for {len(driver_features)} drivers...")
        
        # Extract feature columns in correct order
        feature_cols = self.metadata.get('feature_names', PERFORMANCE_FEATURES)
        X = driver_features[feature_cols].values
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Run inference
        if self.model_type == 'random_forest':
            # sklearn model
            dna_predictions = self.dna_model.predict(X_scaled)
        else:
            # PyTorch model
            self.dna_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                dna_predictions = self.dna_model(X_tensor).cpu().numpy()
        
        # Create DataFrame with predictions
        dna_feature_names = self.metadata.get('target_names', DNA_FEATURES)
        dna_df = pd.DataFrame(dna_predictions, columns=dna_feature_names)
        dna_df['driver_id'] = driver_features['driver_id'].values
        
        # Reorder columns to put driver_id first
        cols = ['driver_id'] + dna_feature_names
        dna_df = dna_df[cols]
        
        logger.info(f"   ✅ DNA predictions complete")
        
        return dna_df
    
    def predict_archetypes(self, dna_features: pd.DataFrame) -> pd.Series:
        """
        Predict driver archetypes
        
        Args:
            dna_features: DataFrame with DNA signature features
            
        Returns:
            Series with archetype labels
        """
        logger.info(f"Predicting archetypes for {len(dna_features)} drivers...")
        
        # Extract DNA features in correct order
        dna_feature_names = self.metadata.get('target_names', DNA_FEATURES)
        X_dna = dna_features[dna_feature_names].values
        
        # Run inference
        if self.model_type == 'random_forest':
            # sklearn model
            archetype_predictions = self.archetype_model.predict(X_dna)
        else:
            # PyTorch model
            self.archetype_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_dna).to(self.device)
                archetype_outputs = self.archetype_model(X_tensor)
                _, archetype_predictions = torch.max(archetype_outputs, 1)
                archetype_predictions = archetype_predictions.cpu().numpy()
        
        # Decode predictions
        archetype_labels = self.label_encoder.inverse_transform(archetype_predictions)
        
        logger.info(f"   ✅ Archetype predictions complete")
        
        return pd.Series(archetype_labels, index=dna_features.index)
    
    def create_driver_profiles(self, user_data: pd.DataFrame, allow_partial: bool = False) -> Dict[int, Dict]:
        """
        Create complete driver profiles matching original format
        
        Args:
            user_data: Raw user data with lap times, sectors, speeds
            allow_partial: If True, attempt to create profiles even with some data issues
            
        Returns:
            Dictionary matching PerformanceDNAAnalyzer.driver_profiles structure
            
        Raises:
            DataValidationError: If user data validation fails
            PredictionError: If prediction fails for some or all drivers
        """
        logger.info("Creating driver profiles...")
        
        # Validate user data
        is_valid, errors = self.validate_user_data(user_data, allow_partial=allow_partial)
        if not is_valid:
            error_msg = f"User data validation failed: {errors}"
            DNALogger.log_error(logger, 'E019', error_msg)
            raise DataValidationError(error_msg, validation_errors=errors, allow_partial=allow_partial)
        
        # Extract driver features using feature engineering
        try:
            with ErrorContext(logger, "Extracting driver features", 'E028'):
                driver_features = self.feature_engineering.extract_driver_features(user_data)
                
                if driver_features.empty:
                    error_msg = "No valid driver features could be extracted from user data"
                    DNALogger.log_error(logger, 'E020', error_msg)
                    raise PredictionError(error_msg)
                
                logger.info(f"Extracted features for {len(driver_features)} driver-track combinations")
        except PredictionError:
            raise
        except Exception as e:
            error_msg = f"Feature extraction failed: {e}"
            DNALogger.log_error(logger, 'E028', error_msg, exception=e)
            raise PredictionError(error_msg)
        
        # Aggregate features per driver (across all tracks)
        try:
            driver_agg_features = driver_features.groupby('driver_id').agg({
                'avg_lap_time': 'mean',
                'std_lap_time': 'mean',
                'min_lap_time': 'min',
                'lap_count': 'sum',
                'avg_s1': 'mean',
                'std_s1': 'mean',
                'min_s1': 'min',
                'avg_s2': 'mean',
                'std_s2': 'mean',
                'min_s2': 'min',
                'avg_s3': 'mean',
                'std_s3': 'mean',
                'min_s3': 'min',
                'avg_speed': 'mean',
                'max_speed': 'max'
            }).reset_index()
        except Exception as e:
            error_msg = f"Failed to aggregate driver features: {e}"
            DNALogger.log_error(logger, 'E041', error_msg, exception=e)
            raise PredictionError(error_msg)
        
        # Predict DNA signatures with error handling
        failed_drivers = []
        partial_results = {}
        
        try:
            with ErrorContext(logger, "Predicting DNA signatures", 'E043'):
                dna_predictions = self.predict_driver_dna(driver_agg_features)
        except Exception as e:
            error_msg = f"DNA prediction failed: {e}"
            DNALogger.log_error(logger, 'E043', error_msg, exception=e)
            if not allow_partial:
                raise PredictionError(error_msg)
            else:
                logger.warning("Attempting partial prediction recovery...")
                dna_predictions = None
        
        # Predict archetypes with error handling
        try:
            if dna_predictions is not None:
                with ErrorContext(logger, "Predicting archetypes", 'E044'):
                    archetype_predictions = self.predict_archetypes(dna_predictions)
            else:
                archetype_predictions = None
        except Exception as e:
            error_msg = f"Archetype prediction failed: {e}"
            DNALogger.log_error(logger, 'E044', error_msg, exception=e)
            if not allow_partial:
                raise PredictionError(error_msg)
            else:
                logger.warning("Continuing with partial results...")
                archetype_predictions = None
        
        # Build driver profiles in original format
        driver_profiles = {}
        
        for driver_id in driver_features['driver_id'].unique():
            try:
                driver_data = driver_features[driver_features['driver_id'] == driver_id]
                
                # Get DNA and archetype predictions if available
                if dna_predictions is not None and archetype_predictions is not None:
                    driver_dna = dna_predictions[dna_predictions['driver_id'] == driver_id].iloc[0]
                    driver_archetype = archetype_predictions[dna_predictions['driver_id'] == driver_id].iloc[0]
                else:
                    # Use placeholder values if predictions failed
                    driver_dna = None
                    driver_archetype = "Unknown"
                    failed_drivers.append(int(driver_id))
                
                # Get tracks raced
                tracks_raced = driver_data['track'].unique().tolist()
                
                # Build performance metrics per track
                performance_metrics = {}
                for track in tracks_raced:
                    track_data = driver_data[driver_data['track'] == track].iloc[0]
                    
                    performance_metrics[track] = {
                        'avg_lap_time': float(track_data['avg_lap_time']),
                        'consistency': float(1.0 / (track_data['std_lap_time'] + 0.001)),
                        'best_lap': float(track_data['min_lap_time']),
                        'sector_balance': {
                            'S1_avg': float(track_data['avg_s1']),
                            'S2_avg': float(track_data['avg_s2']),
                            'S3_avg': float(track_data['avg_s3'])
                        },
                        'speed_profile': {
                            'avg_speed': float(track_data['avg_speed']),
                            'top_speed': float(track_data['max_speed'])
                        }
                    }
                
                # Build DNA signature
                if driver_dna is not None:
                    dna_signature = {
                        'speed_vs_consistency_ratio': float(driver_dna['speed_vs_consistency_ratio']),
                        'track_adaptability': float(driver_dna['track_adaptability']),
                        'consistency_index': float(driver_dna['consistency_index']),
                        'performance_variance': float(driver_dna['performance_variance']),
                        'speed_consistency': float(driver_dna['speed_consistency']),
                        'archetype': str(driver_archetype),
                        'track_specialization': {
                            'technical': 1.0,  # Placeholder - would need track type info
                            'high_speed': 1.0,
                            'mixed': 1.0
                        }
                    }
                else:
                    # Placeholder DNA signature
                    dna_signature = {
                        'speed_vs_consistency_ratio': 0.0,
                        'track_adaptability': 0.0,
                        'consistency_index': 0.0,
                        'performance_variance': 0.0,
                        'speed_consistency': 0.0,
                        'archetype': str(driver_archetype),
                        'track_specialization': {
                            'technical': 1.0,
                            'high_speed': 1.0,
                            'mixed': 1.0
                        }
                    }
                
                # Create driver profile
                driver_profiles[int(driver_id)] = {
                    'tracks_raced': tracks_raced,
                    'total_races': len(tracks_raced),
                    'performance_metrics': performance_metrics,
                    'dna_signature': dna_signature
                }
                
                if driver_dna is not None:
                    partial_results[int(driver_id)] = driver_profiles[int(driver_id)]
                
            except Exception as e:
                error_msg = f"Failed to create profile for driver {driver_id}: {e}"
                logger.error(error_msg)
                failed_drivers.append(int(driver_id))
                if not allow_partial:
                    DNALogger.log_error(logger, 'E045', error_msg, exception=e)
                    raise PredictionError(error_msg, driver_ids=[int(driver_id)])
        
        # Log results
        if failed_drivers:
            warning_msg = f"Failed to create complete profiles for {len(failed_drivers)} drivers: {failed_drivers}"
            DNALogger.log_warning(logger, 'W003', warning_msg)
            logger.warning(warning_msg)
        
        if driver_profiles:
            DNALogger.log_info(logger, 'I007', f"Created profiles for {len(driver_profiles)} drivers")
            logger.info(f"✅ Created profiles for {len(driver_profiles)} drivers")
            if failed_drivers and allow_partial:
                logger.info(f"   ⚠️  {len(failed_drivers)} drivers have incomplete DNA predictions")
        else:
            error_msg = "Failed to create any driver profiles"
            DNALogger.log_error(logger, 'E045', error_msg)
            raise PredictionError(error_msg, driver_ids=failed_drivers)
        
        return driver_profiles
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata (version, metrics, training date)
        
        Returns:
            Dictionary with model information
        """
        if self.metadata is None:
            return {
                'error': 'Model not loaded',
                'status': 'not_loaded'
            }
        
        return {
            'version': self.metadata.get('version', 'unknown'),
            'training_date': self.metadata.get('training_date', 'unknown'),
            'model_type': self.metadata.get('model_type', 'unknown'),
            'framework': self.metadata.get('framework', 'unknown'),
            'validation_metrics': self.metadata.get('validation_metrics', {}),
            'feature_names': self.metadata.get('feature_names', []),
            'archetype_classes': self.metadata.get('archetype_classes', []),
            'status': 'loaded'
        }


def main():
    """Example usage of DNAModelInference"""
    print("🧬 DNA Model Inference")
    print("=" * 50)
    
    try:
        # Initialize inference system
        inference = DNAModelInference()
        
        # Display model info
        model_info = inference.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"   Version: {model_info['version']}")
        print(f"   Training Date: {model_info['training_date']}")
        print(f"   Reliability: {model_info['validation_metrics'].get('overall_reliability', 'N/A'):.2%}")
        
        # Example: Load and process user data
        print(f"\n📂 Loading sample user data...")
        sample_data = inference.feature_engineering.load_and_process_csv(
            'barber/23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV',
            track_name='barber'
        )
        print(f"   Loaded {len(sample_data)} rows")
        
        # Create driver profiles
        print(f"\n🔮 Generating driver profiles...")
        driver_profiles = inference.create_driver_profiles(sample_data)
        print(f"   ✅ Created profiles for {len(driver_profiles)} drivers")
        
        # Display sample profile
        if driver_profiles:
            sample_driver_id = list(driver_profiles.keys())[0]
            sample_profile = driver_profiles[sample_driver_id]
            print(f"\n👤 Sample Profile (Driver {sample_driver_id}):")
            print(f"   Archetype: {sample_profile['dna_signature']['archetype']}")
            print(f"   Tracks Raced: {sample_profile['tracks_raced']}")
            print(f"   DNA Signature:")
            for key, value in sample_profile['dna_signature'].items():
                if key not in ['archetype', 'track_specialization']:
                    print(f"      {key}: {value:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
