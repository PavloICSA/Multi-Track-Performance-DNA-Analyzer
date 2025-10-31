#!/usr/bin/env python3
"""
DNA Model Trainer
Trains and validates ML models for driver DNA prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from dna_feature_engineering import DNAFeatureEngineering
from dna_model_registry import DNAModelRegistry
from dna_logging import get_logger, DNALogger, ErrorContext
from dna_exceptions import (
    InsufficientDataError, DataQualityError, ValidationFailedError,
    TrainingError, format_error_message
)
from config import (
    TRACKS, MODELS_DIR, TRAINING_CONFIG, DNA_FEATURES,
    ARCHETYPE_CLASSES, get_model_dir, INITIAL_MODEL_VERSION,
    DNA_REGRESSION_MODEL_FILE, ARCHETYPE_CLASSIFIER_FILE,
    FEATURE_SCALER_FILE, LABEL_ENCODER_FILE, METADATA_FILE,
    VALIDATION_REPORT_FILE, PERFORMANCE_FEATURES
)

# Set up logger
logger = get_logger(__name__)


class DNARegressionModel(nn.Module):
    """Neural network for DNA signature prediction"""
    
    def __init__(self, input_dim: int, output_dim: int = 5):
        super(DNARegressionModel, self).__init__()
        
        hidden_layers = TRAINING_CONFIG['dna_hidden_layers']
        dropout_rate = TRAINING_CONFIG['dna_dropout_rate']
        
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = nn.Linear(hidden_layers[2], output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ArchetypeClassifier(nn.Module):
    """Neural network for archetype classification"""
    
    def __init__(self, input_dim: int = 5, output_dim: int = 4):
        super(ArchetypeClassifier, self).__init__()
        
        hidden_layers = TRAINING_CONFIG['archetype_hidden_layers']
        dropout_rate = TRAINING_CONFIG['archetype_dropout_rate']
        
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DNAModelTrainer:
    """
    Trains and validates ML models for driver DNA prediction
    """
    
    def __init__(self, data_dir: str = ".", models_dir: str = None):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing track data folders
            models_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)
        
        self.feature_engineering = DNAFeatureEngineering()
        
        # Data containers
        self.raw_data = None
        self.driver_features = None
        self.dna_features = None
        self.archetype_labels = None
        
        # Train/val/test splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_dna_train = None
        self.y_dna_val = None
        self.y_dna_test = None
        self.y_archetype_train = None
        self.y_archetype_val = None
        self.y_archetype_test = None
        
        # Models and scalers
        self.dna_model = None
        self.archetype_model = None
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Training history
        self.training_history = {
            'dna_train_loss': [],
            'dna_val_loss': [],
            'archetype_train_loss': [],
            'archetype_val_loss': [],
            'archetype_train_acc': [],
            'archetype_val_acc': []
        }
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
    
    def load_all_track_data(self) -> pd.DataFrame:
        """
        Load data from all track directories
        
        Returns:
            Combined DataFrame with all sector data
            
        Raises:
            DataQualityError: If no valid track data found or data quality issues detected
        """
        logger.info("Loading data from all tracks...")
        print("🔄 Loading data from all tracks...")
        
        all_sector_data = []
        loading_errors = []
        data_quality_issues = []
        
        for track in TRACKS:
            logger.debug(f"Loading track: {track}")
            print(f"   📍 Loading {track}...")
            track_path = self.data_dir / track
            
            if not track_path.exists():
                warning_msg = f"Track directory not found: {track_path}"
                logger.warning(f"[W008] {warning_msg}")
                print(f"      ⚠️  {warning_msg}")
                loading_errors.append(warning_msg)
                continue
            
            # Handle different directory structures
            if track == 'barber':
                race_paths = [track_path]
            else:
                race_paths = [track_path / 'Race 1', track_path / 'Race 2']
            
            for race_path in race_paths:
                if not race_path.exists():
                    continue
                
                # Load sector analysis files
                sector_files = list(race_path.glob('*AnalysisEnduranceWithSections*.CSV'))
                
                if not sector_files:
                    warning_msg = f"No sector files found in {race_path}"
                    logger.warning(f"[W008] {warning_msg}")
                
                for file in sector_files:
                    try:
                        with ErrorContext(logger, f"Loading {file.name}", 'E027'):
                            df = self.feature_engineering.load_and_process_csv(file, track_name=track)
                            
                            # Check data quality
                            if len(df) < 10:
                                data_quality_issues.append(f"{file.name}: Only {len(df)} rows (minimum 10 recommended)")
                            
                            all_sector_data.append(df)
                            logger.info(f"Loaded {len(df)} rows from {file.name}")
                            print(f"      ✅ Loaded {len(df)} rows from {file.name}")
                    except Exception as e:
                        error_msg = f"Error loading {file.name}: {e}"
                        DNALogger.log_error(logger, 'E027', error_msg, exception=e)
                        print(f"      ⚠️  {error_msg}")
                        loading_errors.append(error_msg)
        
        # Check if we have any data
        if not all_sector_data:
            error_msg = "No track data found! Check data directory structure."
            issues = loading_errors + data_quality_issues
            DNALogger.log_error(logger, 'E022', error_msg)
            raise DataQualityError(error_msg, issues=issues)
        
        # Combine all data
        try:
            self.raw_data = pd.concat(all_sector_data, ignore_index=True)
            logger.info(f"Loaded {len(self.raw_data)} total rows from {len(all_sector_data)} files")
            print(f"✅ Loaded {len(self.raw_data)} total rows from {len(all_sector_data)} files")
            
            # Log data quality warnings if any
            if data_quality_issues:
                logger.warning(f"[W008] Data quality issues detected: {len(data_quality_issues)} files with low row counts")
                print(f"⚠️  Data quality warnings: {len(data_quality_issues)} files with low row counts")
            
            return self.raw_data
            
        except Exception as e:
            error_msg = f"Failed to combine track data: {e}"
            DNALogger.log_error(logger, 'E022', error_msg, exception=e)
            raise DataQualityError(error_msg, issues=loading_errors)

    
    def prepare_training_data(self) -> pd.DataFrame:
        """
        Apply feature engineering and split data into train/val/test sets
        
        Returns:
            Combined training DataFrame
            
        Raises:
            InsufficientDataError: If not enough data for training
            DataQualityError: If data quality issues prevent training
        """
        logger.info("Preparing training data...")
        print("🔧 Preparing training data...")
        
        if self.raw_data is None:
            error_msg = "No data loaded. Call load_all_track_data() first."
            DNALogger.log_error(logger, 'E022', error_msg)
            raise DataQualityError(error_msg)
        
        # Apply feature engineering pipeline with error handling
        try:
            with ErrorContext(logger, "Extracting driver features", 'E028'):
                print("   Extracting driver features...")
                self.driver_features = self.feature_engineering.extract_driver_features(self.raw_data)
                logger.info(f"Extracted features for {len(self.driver_features)} driver-track combinations")
                print(f"   ✅ Extracted features for {len(self.driver_features)} driver-track combinations")
                
                if self.driver_features.empty:
                    raise DataQualityError("No valid driver features could be extracted from raw data")
        except Exception as e:
            if not isinstance(e, DataQualityError):
                DNALogger.log_error(logger, 'E028', f"Feature extraction failed: {e}", exception=e)
                raise DataQualityError(f"Feature extraction failed: {e}")
            raise
        
        try:
            with ErrorContext(logger, "Calculating DNA signatures", 'E029'):
                print("   Calculating DNA signatures...")
                self.dna_features = self.feature_engineering.calculate_dna_features(self.driver_features)
                logger.info(f"Calculated DNA for {len(self.dna_features)} drivers")
                print(f"   ✅ Calculated DNA for {len(self.dna_features)} drivers")
                
                if self.dna_features.empty:
                    raise DataQualityError(
                        "No DNA features calculated. Drivers may need data from multiple tracks.",
                        issues=["Each driver needs at least 2 tracks for DNA calculation"]
                    )
        except Exception as e:
            if not isinstance(e, DataQualityError):
                DNALogger.log_error(logger, 'E029', f"DNA calculation failed: {e}", exception=e)
                raise DataQualityError(f"DNA calculation failed: {e}")
            raise
        
        try:
            with ErrorContext(logger, "Creating archetype labels", 'E030'):
                print("   Creating archetype labels...")
                self.archetype_labels = self.feature_engineering.create_archetype_labels(self.dna_features)
                logger.info(f"Assigned archetypes to {len(self.archetype_labels)} drivers")
                print(f"   ✅ Assigned archetypes to {len(self.archetype_labels)} drivers")
        except Exception as e:
            DNALogger.log_error(logger, 'E030', f"Archetype labeling failed: {e}", exception=e)
            raise DataQualityError(f"Archetype labeling failed: {e}")
        
        # Merge driver features with DNA features and labels
        try:
            # Aggregate driver features across tracks for each driver
            driver_agg_features = self.driver_features.groupby('driver_id').agg({
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
            
            # Merge with DNA features
            training_data = driver_agg_features.merge(
                self.dna_features,
                on='driver_id',
                how='inner'
            )
            
            # Add archetype labels
            training_data['archetype'] = self.archetype_labels.values
            
        except Exception as e:
            error_msg = f"Failed to merge training data: {e}"
            DNALogger.log_error(logger, 'E022', error_msg, exception=e)
            raise DataQualityError(error_msg)
        
        # Check minimum data requirement
        min_samples = 30
        if len(training_data) < min_samples:
            error_msg = (
                f"Insufficient data for training. Found {len(training_data)} samples, "
                f"need at least {min_samples} driver-track combinations."
            )
            DNALogger.log_error(logger, 'E021', error_msg)
            raise InsufficientDataError(
                error_msg,
                sample_count=len(training_data),
                required_count=min_samples
            )
        
        # Check archetype distribution
        archetype_counts = training_data['archetype'].value_counts()
        min_per_archetype = 3
        underrepresented = archetype_counts[archetype_counts < min_per_archetype]
        
        if not underrepresented.empty:
            warning_msg = f"Some archetypes have few samples: {underrepresented.to_dict()}"
            DNALogger.log_warning(logger, 'W008', warning_msg)
            print(f"   ⚠️  {warning_msg}")
        
        logger.info(f"Prepared {len(training_data)} training samples")
        logger.info(f"Archetype distribution: {archetype_counts.to_dict()}")
        print(f"✅ Prepared {len(training_data)} training samples")
        print(f"   Archetype distribution: {training_data['archetype'].value_counts().to_dict()}")
        
        return training_data
    
    def split_data(self, training_data: pd.DataFrame):
        """
        Split data into train/validation/test sets with stratification
        
        Args:
            training_data: Combined DataFrame with features and labels
        """
        print("✂️  Splitting data into train/val/test sets...")
        
        # Prepare feature matrix (X) and targets (y)
        feature_cols = [
            'avg_lap_time', 'std_lap_time', 'min_lap_time', 'lap_count',
            'avg_s1', 'std_s1', 'min_s1',
            'avg_s2', 'std_s2', 'min_s2',
            'avg_s3', 'std_s3', 'min_s3',
            'avg_speed', 'max_speed'
        ]
        
        X = training_data[feature_cols].values
        y_dna = training_data[DNA_FEATURES].values
        y_archetype = training_data['archetype'].values
        
        # Encode archetype labels
        y_archetype_encoded = self.label_encoder.fit_transform(y_archetype)
        
        # First split: train+val vs test (85% vs 15%)
        train_size = TRAINING_CONFIG['train_size']
        val_size = TRAINING_CONFIG['val_size']
        test_size = TRAINING_CONFIG['test_size']
        
        # Stratified split to maintain archetype distribution
        X_temp, X_test, y_dna_temp, y_dna_test, y_arch_temp, y_arch_test = train_test_split(
            X, y_dna, y_archetype_encoded,
            test_size=test_size,
            random_state=TRAINING_CONFIG['random_state'],
            stratify=y_archetype_encoded
        )
        
        # Second split: train vs val
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_dna_train, y_dna_val, y_arch_train, y_arch_val = train_test_split(
            X_temp, y_dna_temp, y_arch_temp,
            test_size=val_ratio,
            random_state=TRAINING_CONFIG['random_state'],
            stratify=y_arch_temp
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Store splits
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_dna_train = y_dna_train
        self.y_dna_val = y_dna_val
        self.y_dna_test = y_dna_test
        self.y_archetype_train = y_arch_train
        self.y_archetype_val = y_arch_val
        self.y_archetype_test = y_arch_test
        
        print(f"✅ Data split complete:")
        print(f"   Train: {len(X_train)} samples ({train_size*100:.0f}%)")
        print(f"   Val:   {len(X_val)} samples ({val_size*100:.0f}%)")
        print(f"   Test:  {len(X_test)} samples ({test_size*100:.0f}%)")
    
    def build_models(self):
        """Build DNA regression and archetype classification models"""
        print("🏗️  Building neural network models...")
        
        input_dim = self.X_train.shape[1]
        
        # DNA Regression Model
        self.dna_model = DNARegressionModel(input_dim=input_dim, output_dim=5)
        self.dna_model.to(self.device)
        print(f"   ✅ DNA Regression Model: {input_dim} → 128 → 64 → 32 → 5")
        
        # Archetype Classifier
        self.archetype_model = ArchetypeClassifier(input_dim=5, output_dim=4)
        self.archetype_model.to(self.device)
        print(f"   ✅ Archetype Classifier: 5 → 64 → 32 → 4")
    
    def train_models(self, epochs: int = None) -> Dict[str, Any]:
        """
        Train both DNA regression and archetype classification models
        
        Args:
            epochs: Number of training epochs (uses config default if None)
            
        Returns:
            Dictionary with training history and metrics
        """
        if epochs is None:
            epochs = TRAINING_CONFIG['epochs']
        
        print(f"🚀 Training models for {epochs} epochs...")
        
        # Build models if not already built
        if self.dna_model is None or self.archetype_model is None:
            self.build_models()
        
        # Train DNA regression model
        print("\n📊 Training DNA Regression Model...")
        self._train_dna_model(epochs)
        
        # Train archetype classifier
        print("\n🎯 Training Archetype Classifier...")
        self._train_archetype_model(epochs)
        
        print("✅ Training complete!")
        
        return self.training_history
    
    def _train_dna_model(self, epochs: int):
        """Train the DNA regression model"""
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_dna_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.FloatTensor(self.y_dna_val)
        )
        
        batch_size = TRAINING_CONFIG['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.dna_model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=TRAINING_CONFIG['lr_patience'],
            factor=TRAINING_CONFIG['lr_factor']
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = TRAINING_CONFIG['early_stopping_patience']
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.dna_model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.dna_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.dna_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.dna_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Store history
            self.training_history['dna_train_loss'].append(train_loss)
            self.training_history['dna_val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - TRAINING_CONFIG['early_stopping_min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    def _train_archetype_model(self, epochs: int):
        """Train the archetype classification model"""
        # Use predicted DNA features as input for archetype classifier
        self.dna_model.eval()
        
        with torch.no_grad():
            X_train_dna = self.dna_model(torch.FloatTensor(self.X_train).to(self.device))
            X_val_dna = self.dna_model(torch.FloatTensor(self.X_val).to(self.device))
            X_train_dna = X_train_dna.cpu().numpy()
            X_val_dna = X_val_dna.cpu().numpy()
        
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_dna),
            torch.LongTensor(self.y_archetype_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_dna),
            torch.LongTensor(self.y_archetype_val)
        )
        
        batch_size = TRAINING_CONFIG['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.archetype_model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=TRAINING_CONFIG['lr_patience'],
            factor=TRAINING_CONFIG['lr_factor']
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = TRAINING_CONFIG['early_stopping_patience']
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.archetype_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.archetype_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            self.archetype_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.archetype_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Store history
            self.training_history['archetype_train_loss'].append(train_loss)
            self.training_history['archetype_val_loss'].append(val_loss)
            self.training_history['archetype_train_acc'].append(train_acc)
            self.training_history['archetype_val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - TRAINING_CONFIG['early_stopping_min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    
    def validate_models(self) -> Dict[str, float]:
        """
        Validate models on test set and calculate metrics
        
        Returns:
            Dictionary with validation metrics
            
        Raises:
            ValidationFailedError: If model fails to meet reliability threshold
        """
        logger.info("Validating models on test set...")
        print("📊 Validating models on test set...")
        
        try:
            with ErrorContext(logger, "Model validation", 'E040'):
                # Prepare test data
                X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
                
                # DNA Regression validation
                self.dna_model.eval()
                with torch.no_grad():
                    y_dna_pred = self.dna_model(X_test_tensor).cpu().numpy()
                
                # Check for NaN or infinite values in predictions
                if np.isnan(y_dna_pred).any():
                    raise ValidationFailedError("DNA model produced NaN predictions")
                if np.isinf(y_dna_pred).any():
                    raise ValidationFailedError("DNA model produced infinite predictions")
                
                # Calculate DNA regression metrics
                dna_mae = mean_absolute_error(self.y_dna_test, y_dna_pred)
                dna_r2 = r2_score(self.y_dna_test, y_dna_pred)
                
                logger.info(f"DNA Regression - MAE: {dna_mae:.4f}, R²: {dna_r2:.4f}")
                print(f"   DNA Regression - MAE: {dna_mae:.4f}, R²: {dna_r2:.4f}")
                
                # Archetype Classification validation
                # Use predicted DNA features as input
                y_dna_pred_tensor = torch.FloatTensor(y_dna_pred).to(self.device)
                
                self.archetype_model.eval()
                with torch.no_grad():
                    archetype_outputs = self.archetype_model(y_dna_pred_tensor)
                    _, y_archetype_pred = torch.max(archetype_outputs, 1)
                    y_archetype_pred = y_archetype_pred.cpu().numpy()
                
                # Calculate archetype classification metrics
                archetype_accuracy = accuracy_score(self.y_archetype_test, y_archetype_pred)
                archetype_precision = precision_score(
                    self.y_archetype_test, y_archetype_pred, average='weighted', zero_division=0
                )
                archetype_recall = recall_score(
                    self.y_archetype_test, y_archetype_pred, average='weighted', zero_division=0
                )
                archetype_f1 = f1_score(
                    self.y_archetype_test, y_archetype_pred, average='weighted', zero_division=0
                )
                
                logger.info(f"Archetype Classification - Accuracy: {archetype_accuracy:.4f}, "
                           f"Precision: {archetype_precision:.4f}, Recall: {archetype_recall:.4f}, "
                           f"F1: {archetype_f1:.4f}")
                print(f"   Archetype Classification:")
                print(f"      Accuracy:  {archetype_accuracy:.4f}")
                print(f"      Precision: {archetype_precision:.4f}")
                print(f"      Recall:    {archetype_recall:.4f}")
                print(f"      F1 Score:  {archetype_f1:.4f}")
                
                # Calculate overall reliability score (weighted average)
                # Weights: accuracy (40%), precision (30%), recall (20%), F1 (10%)
                overall_reliability = (
                    0.4 * archetype_accuracy +
                    0.3 * archetype_precision +
                    0.2 * archetype_recall +
                    0.1 * archetype_f1
                )
                
                logger.info(f"Overall Reliability Score: {overall_reliability:.4f}")
                print(f"\n   Overall Reliability Score: {overall_reliability:.4f}")
                
                # Return metrics
                metrics = {
                    'dna_mae': float(dna_mae),
                    'dna_r2': float(dna_r2),
                    'archetype_accuracy': float(archetype_accuracy),
                    'archetype_precision': float(archetype_precision),
                    'archetype_recall': float(archetype_recall),
                    'archetype_f1': float(archetype_f1),
                    'overall_reliability': float(overall_reliability)
                }
                
                # Check if meets threshold
                min_threshold = TRAINING_CONFIG['min_reliability_score']
                if overall_reliability >= min_threshold:
                    DNALogger.log_info(logger, 'I005', f"Model meets reliability threshold (≥{min_threshold})")
                    print(f"   ✅ Model meets reliability threshold (≥{min_threshold})")
                else:
                    error_msg = f"Model does not meet reliability threshold (≥{min_threshold})"
                    DNALogger.log_error(logger, 'E032', error_msg)
                    print(f"   ❌ {error_msg}")
                    
                    # Provide detailed failure report
                    failure_details = {
                        'overall_reliability': overall_reliability,
                        'threshold': min_threshold,
                        'gap': min_threshold - overall_reliability,
                        'metrics': metrics
                    }
                    
                    raise ValidationFailedError(
                        error_msg,
                        metrics=failure_details,
                        threshold=min_threshold
                    )
                
                return metrics
                
        except ValidationFailedError:
            raise
        except Exception as e:
            error_msg = f"Validation failed with unexpected error: {e}"
            DNALogger.log_error(logger, 'E040', error_msg, exception=e)
            raise ValidationFailedError(error_msg)
    
    def cross_validate(self, n_folds: int = None) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation
        
        Args:
            n_folds: Number of folds (uses config default if None)
            
        Returns:
            Dictionary with metrics for each fold
        """
        if n_folds is None:
            n_folds = TRAINING_CONFIG['cv_folds']
        
        print(f"🔄 Performing {n_folds}-fold cross-validation...")
        
        # Prepare full dataset (combine train + val + test)
        X_full = np.vstack([self.X_train, self.X_val, self.X_test])
        y_dna_full = np.vstack([self.y_dna_train, self.y_dna_val, self.y_dna_test])
        y_archetype_full = np.concatenate([
            self.y_archetype_train, 
            self.y_archetype_val, 
            self.y_archetype_test
        ])
        
        # Initialize cross-validation
        skf = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=TRAINING_CONFIG['random_state']
        )
        
        # Store metrics for each fold
        cv_results = {
            'dna_mae': [],
            'dna_r2': [],
            'archetype_accuracy': [],
            'archetype_precision': [],
            'archetype_recall': [],
            'archetype_f1': [],
            'overall_reliability': []
        }
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_archetype_full), 1):
            print(f"\n   Fold {fold}/{n_folds}")
            
            # Split data for this fold
            X_train_fold = X_full[train_idx]
            X_val_fold = X_full[val_idx]
            y_dna_train_fold = y_dna_full[train_idx]
            y_dna_val_fold = y_dna_full[val_idx]
            y_arch_train_fold = y_archetype_full[train_idx]
            y_arch_val_fold = y_archetype_full[val_idx]
            
            # Build fresh models for this fold
            input_dim = X_train_fold.shape[1]
            dna_model_fold = DNARegressionModel(input_dim=input_dim, output_dim=5).to(self.device)
            archetype_model_fold = ArchetypeClassifier(input_dim=5, output_dim=4).to(self.device)
            
            # Train DNA regression model
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_fold),
                torch.FloatTensor(y_dna_train_fold)
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=TRAINING_CONFIG['batch_size'], 
                shuffle=True
            )
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                dna_model_fold.parameters(),
                lr=TRAINING_CONFIG['learning_rate'],
                weight_decay=TRAINING_CONFIG['weight_decay']
            )
            
            # Quick training (fewer epochs for CV)
            cv_epochs = min(30, TRAINING_CONFIG['epochs'] // 2)
            dna_model_fold.train()
            
            for epoch in range(cv_epochs):
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = dna_model_fold(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate DNA model
            dna_model_fold.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_fold).to(self.device)
                y_dna_pred = dna_model_fold(X_val_tensor).cpu().numpy()
            
            dna_mae = mean_absolute_error(y_dna_val_fold, y_dna_pred)
            dna_r2 = r2_score(y_dna_val_fold, y_dna_pred)
            
            # Train archetype classifier using predicted DNA features
            with torch.no_grad():
                X_train_dna = dna_model_fold(torch.FloatTensor(X_train_fold).to(self.device)).cpu().numpy()
            
            train_dataset_arch = TensorDataset(
                torch.FloatTensor(X_train_dna),
                torch.LongTensor(y_arch_train_fold)
            )
            train_loader_arch = DataLoader(
                train_dataset_arch,
                batch_size=TRAINING_CONFIG['batch_size'],
                shuffle=True
            )
            
            criterion_arch = nn.CrossEntropyLoss()
            optimizer_arch = optim.Adam(
                archetype_model_fold.parameters(),
                lr=TRAINING_CONFIG['learning_rate'],
                weight_decay=TRAINING_CONFIG['weight_decay']
            )
            
            archetype_model_fold.train()
            for epoch in range(cv_epochs):
                for X_batch, y_batch in train_loader_arch:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer_arch.zero_grad()
                    outputs = archetype_model_fold(X_batch)
                    loss = criterion_arch(outputs, y_batch)
                    loss.backward()
                    optimizer_arch.step()
            
            # Evaluate archetype classifier
            archetype_model_fold.eval()
            with torch.no_grad():
                y_dna_pred_tensor = torch.FloatTensor(y_dna_pred).to(self.device)
                archetype_outputs = archetype_model_fold(y_dna_pred_tensor)
                _, y_arch_pred = torch.max(archetype_outputs, 1)
                y_arch_pred = y_arch_pred.cpu().numpy()
            
            arch_accuracy = accuracy_score(y_arch_val_fold, y_arch_pred)
            arch_precision = precision_score(
                y_arch_val_fold, y_arch_pred, average='weighted', zero_division=0
            )
            arch_recall = recall_score(
                y_arch_val_fold, y_arch_pred, average='weighted', zero_division=0
            )
            arch_f1 = f1_score(
                y_arch_val_fold, y_arch_pred, average='weighted', zero_division=0
            )
            
            # Calculate overall reliability for this fold
            overall_reliability = (
                0.4 * arch_accuracy +
                0.3 * arch_precision +
                0.2 * arch_recall +
                0.1 * arch_f1
            )
            
            # Store results
            cv_results['dna_mae'].append(float(dna_mae))
            cv_results['dna_r2'].append(float(dna_r2))
            cv_results['archetype_accuracy'].append(float(arch_accuracy))
            cv_results['archetype_precision'].append(float(arch_precision))
            cv_results['archetype_recall'].append(float(arch_recall))
            cv_results['archetype_f1'].append(float(arch_f1))
            cv_results['overall_reliability'].append(float(overall_reliability))
            
            print(f"      DNA MAE: {dna_mae:.4f}, R²: {dna_r2:.4f}")
            print(f"      Archetype Acc: {arch_accuracy:.4f}, Reliability: {overall_reliability:.4f}")
        
        # Calculate statistics across folds
        print(f"\n   Cross-Validation Results:")
        for metric, values in cv_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"      {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Check consistency
        reliability_std = np.std(cv_results['overall_reliability'])
        max_std = TRAINING_CONFIG['max_cross_val_std']
        
        if reliability_std < max_std:
            print(f"   ✅ Cross-validation consistent (std={reliability_std:.4f} < {max_std})")
        else:
            print(f"   ⚠️  Cross-validation shows high variance (std={reliability_std:.4f} ≥ {max_std})")
        
        return cv_results
    
    def save_model_artifacts(self, version: str = None) -> str:
        """
        Save model artifacts to disk
        
        Args:
            version: Model version string (uses default if None)
            
        Returns:
            Path to saved model directory
        """
        if version is None:
            version = INITIAL_MODEL_VERSION
        
        print(f"💾 Saving model artifacts (version {version})...")
        
        # Create model directory
        model_dir = get_model_dir(version)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch models
        dna_model_path = model_dir / DNA_REGRESSION_MODEL_FILE
        archetype_model_path = model_dir / ARCHETYPE_CLASSIFIER_FILE
        
        torch.save(self.dna_model.state_dict(), dna_model_path)
        print(f"   ✅ Saved DNA regression model: {dna_model_path}")
        
        torch.save(self.archetype_model.state_dict(), archetype_model_path)
        print(f"   ✅ Saved archetype classifier: {archetype_model_path}")
        
        # Save StandardScaler
        scaler_path = model_dir / FEATURE_SCALER_FILE
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        print(f"   ✅ Saved feature scaler: {scaler_path}")
        
        # Save LabelEncoder
        encoder_path = model_dir / LABEL_ENCODER_FILE
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"   ✅ Saved label encoder: {encoder_path}")
        
        # Get validation metrics
        metrics = self.validate_models()
        
        # Create metadata
        metadata = {
            "version": version,
            "training_date": datetime.now().isoformat(),
            "model_type": "neural_network",
            "framework": "pytorch",
            "framework_version": torch.__version__,
            "feature_names": PERFORMANCE_FEATURES,
            "target_names": DNA_FEATURES,
            "archetype_classes": ARCHETYPE_CLASSES,
            "validation_metrics": metrics,
            "training_config": {
                "epochs": TRAINING_CONFIG['epochs'],
                "batch_size": TRAINING_CONFIG['batch_size'],
                "learning_rate": TRAINING_CONFIG['learning_rate'],
                "optimizer": TRAINING_CONFIG['optimizer'],
                "train_size": TRAINING_CONFIG['train_size'],
                "val_size": TRAINING_CONFIG['val_size'],
                "test_size": TRAINING_CONFIG['test_size'],
                "dna_hidden_layers": TRAINING_CONFIG['dna_hidden_layers'],
                "dna_dropout_rate": TRAINING_CONFIG['dna_dropout_rate'],
                "archetype_hidden_layers": TRAINING_CONFIG['archetype_hidden_layers'],
                "archetype_dropout_rate": TRAINING_CONFIG['archetype_dropout_rate']
            },
            "data_statistics": {
                "total_samples": len(self.X_train) + len(self.X_val) + len(self.X_test),
                "train_samples": len(self.X_train),
                "val_samples": len(self.X_val),
                "test_samples": len(self.X_test),
                "tracks": TRACKS
            }
        }
        
        # Save metadata
        metadata_path = model_dir / METADATA_FILE
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Saved metadata: {metadata_path}")
        
        # Update model registry
        self._update_model_registry(version, metrics)
        
        # Create/update 'latest' symlink
        latest_link = self.models_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        # Create relative symlink
        try:
            import os
            os.symlink(model_dir.name, latest_link, target_is_directory=True)
            print(f"   ✅ Updated 'latest' symlink")
        except (OSError, NotImplementedError):
            # Symlinks may not be supported on Windows without admin rights
            print(f"   ⚠️  Could not create symlink (may require admin rights on Windows)")
        
        print(f"✅ Model artifacts saved to: {model_dir}")
        
        return str(model_dir)
    
    def _update_model_registry(self, version: str, metrics: Dict[str, float]):
        """Update the model registry with new model information"""
        registry = DNAModelRegistry()
        
        # Register the model with metadata
        metadata = {
            "training_date": datetime.now().isoformat(),
            "framework": "pytorch",
            "framework_version": torch.__version__
        }
        
        registry.register_model(
            version=version,
            metrics=metrics,
            metadata=metadata,
            is_production=False  # Don't auto-promote to production
        )
        
        print(f"   ✅ Updated model registry")
    
    def generate_validation_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive validation report with visualizations
        
        Args:
            output_path: Path to save HTML report (uses model dir if None)
            
        Returns:
            Path to generated report
        """
        print("📄 Generating validation report...")
        
        # Determine output path
        if output_path is None:
            # Use latest model directory
            latest_dir = self.models_dir / "latest"
            if latest_dir.exists():
                output_path = latest_dir / VALIDATION_REPORT_FILE
            else:
                output_path = self.models_dir / VALIDATION_REPORT_FILE
        else:
            output_path = Path(output_path)
        
        # Get predictions for test set
        X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        
        self.dna_model.eval()
        with torch.no_grad():
            y_dna_pred = self.dna_model(X_test_tensor).cpu().numpy()
            y_dna_pred_tensor = torch.FloatTensor(y_dna_pred).to(self.device)
            
            archetype_outputs = self.archetype_model(y_dna_pred_tensor)
            _, y_archetype_pred = torch.max(archetype_outputs, 1)
            y_archetype_pred = y_archetype_pred.cpu().numpy()
        
        # Calculate metrics
        metrics = self.validate_models()
        
        # Get confusion matrix
        cm = confusion_matrix(self.y_archetype_test, y_archetype_pred)
        archetype_names = self.label_encoder.classes_
        
        # Calculate per-archetype metrics
        per_archetype_metrics = {}
        for i, archetype in enumerate(archetype_names):
            mask = self.y_archetype_test == i
            if mask.sum() > 0:
                pred_for_archetype = y_archetype_pred[mask]
                accuracy = (pred_for_archetype == i).sum() / mask.sum()
                per_archetype_metrics[archetype] = {
                    'accuracy': float(accuracy),
                    'count': int(mask.sum())
                }
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DNA Model Validation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric-card.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .confusion-matrix {{
            margin: 20px 0;
        }}
        .confusion-matrix table {{
            margin: 0 auto;
        }}
        .confusion-matrix td, .confusion-matrix th {{
            text-align: center;
            min-width: 80px;
        }}
        .cm-cell {{
            font-weight: bold;
        }}
        .cm-diagonal {{
            background-color: #d4edda;
        }}
        .cm-off-diagonal {{
            background-color: #f8d7da;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 30px;
            text-align: center;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .status-pass {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-fail {{
            background-color: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 DNA Model Validation Report</h1>
        
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Model Version:</strong> {INITIAL_MODEL_VERSION}</p>
        <p><strong>Overall Reliability:</strong> 
            <span class="status-badge {'status-pass' if metrics['overall_reliability'] >= TRAINING_CONFIG['min_reliability_score'] else 'status-fail'}">
                {metrics['overall_reliability']:.2%}
            </span>
        </p>
        
        <h2>📊 Overall Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card {'success' if metrics['overall_reliability'] >= TRAINING_CONFIG['min_reliability_score'] else 'warning'}">
                <div class="metric-label">Overall Reliability</div>
                <div class="metric-value">{metrics['overall_reliability']:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Archetype Accuracy</div>
                <div class="metric-value">{metrics['archetype_accuracy']:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">DNA R² Score</div>
                <div class="metric-value">{metrics['dna_r2']:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">DNA MAE</div>
                <div class="metric-value">{metrics['dna_mae']:.3f}</div>
            </div>
        </div>
        
        <h2>🎯 Classification Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{metrics['archetype_accuracy']:.4f}</td>
            </tr>
            <tr>
                <td>Precision (Weighted)</td>
                <td>{metrics['archetype_precision']:.4f}</td>
            </tr>
            <tr>
                <td>Recall (Weighted)</td>
                <td>{metrics['archetype_recall']:.4f}</td>
            </tr>
            <tr>
                <td>F1 Score (Weighted)</td>
                <td>{metrics['archetype_f1']:.4f}</td>
            </tr>
        </table>
        
        <h2>🔍 Confusion Matrix</h2>
        <div class="confusion-matrix">
            <table>
                <tr>
                    <th></th>
                    {''.join(f'<th>{name}</th>' for name in archetype_names)}
                </tr>
"""
        
        # Add confusion matrix rows
        for i, true_label in enumerate(archetype_names):
            html_content += f"                <tr><th>{true_label}</th>"
            for j, pred_label in enumerate(archetype_names):
                cell_class = 'cm-diagonal' if i == j else 'cm-off-diagonal'
                html_content += f'<td class="cm-cell {cell_class}">{cm[i][j]}</td>'
            html_content += "</tr>\n"
        
        html_content += """
            </table>
        </div>
        
        <h2>📈 Performance by Archetype</h2>
        <table>
            <tr>
                <th>Archetype</th>
                <th>Test Samples</th>
                <th>Accuracy</th>
            </tr>
"""
        
        # Add per-archetype metrics
        for archetype, arch_metrics in per_archetype_metrics.items():
            html_content += f"""
            <tr>
                <td>{archetype}</td>
                <td>{arch_metrics['count']}</td>
                <td>{arch_metrics['accuracy']:.2%}</td>
            </tr>
"""
        
        html_content += f"""
        </table>
        
        <h2>🔧 Training Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Epochs</td>
                <td>{TRAINING_CONFIG['epochs']}</td>
            </tr>
            <tr>
                <td>Batch Size</td>
                <td>{TRAINING_CONFIG['batch_size']}</td>
            </tr>
            <tr>
                <td>Learning Rate</td>
                <td>{TRAINING_CONFIG['learning_rate']}</td>
            </tr>
            <tr>
                <td>Train/Val/Test Split</td>
                <td>{TRAINING_CONFIG['train_size']:.0%} / {TRAINING_CONFIG['val_size']:.0%} / {TRAINING_CONFIG['test_size']:.0%}</td>
            </tr>
            <tr>
                <td>DNA Hidden Layers</td>
                <td>{TRAINING_CONFIG['dna_hidden_layers']}</td>
            </tr>
            <tr>
                <td>Archetype Hidden Layers</td>
                <td>{TRAINING_CONFIG['archetype_hidden_layers']}</td>
            </tr>
        </table>
        
        <h2>📊 Dataset Statistics</h2>
        <table>
            <tr>
                <th>Split</th>
                <th>Samples</th>
            </tr>
            <tr>
                <td>Training</td>
                <td>{len(self.X_train)}</td>
            </tr>
            <tr>
                <td>Validation</td>
                <td>{len(self.X_val)}</td>
            </tr>
            <tr>
                <td>Test</td>
                <td>{len(self.X_test)}</td>
            </tr>
            <tr>
                <td><strong>Total</strong></td>
                <td><strong>{len(self.X_train) + len(self.X_val) + len(self.X_test)}</strong></td>
            </tr>
        </table>
        
        <div class="timestamp">
            Report generated by DNA Model Trainer on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"   ✅ Validation report saved: {output_path}")
        
        return str(output_path)


def main():
    """Example usage of DNAModelTrainer"""
    print("🧬 DNA Model Trainer")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = DNAModelTrainer(data_dir=".")
        
        # Load all track data
        trainer.load_all_track_data()
        
        # Prepare training data
        training_data = trainer.prepare_training_data()
        
        # Split data
        trainer.split_data(training_data)
        
        # Build models
        trainer.build_models()
        
        # Train models
        trainer.train_models(epochs=50)
        
        # Validate models
        metrics = trainer.validate_models()
        
        # Perform cross-validation
        cv_results = trainer.cross_validate(n_folds=5)
        
        # Save model artifacts
        model_dir = trainer.save_model_artifacts(version="1.0.0")
        
        # Generate validation report
        report_path = trainer.generate_validation_report()
        
        print("\n✅ Training pipeline complete!")
        print(f"Final metrics: {metrics}")
        print(f"Model saved to: {model_dir}")
        print(f"Validation report: {report_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

