# Design Document

## Overview

This design document outlines the technical architecture for transforming the Multi-Track Performance DNA Analyzer from a direct data processing system into a machine learning model-based inference system. The redesign introduces three major components: a Training System that learns from historical track data, a Model Management System that handles versioning and deployment, and an Inference System that replaces the current data processing pipeline. The solution maintains backward compatibility while eliminating the dependency on raw dataset folders.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Phase (One-time)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Raw Track Data    →    Feature Engineering    →    Model        │
│  (barber, COTA,         (Extract DNA metrics)       Training     │
│   Road America,                                                   │
│   Sebring, etc.)                                                  │
│                                                                   │
│                              ↓                                    │
│                                                                   │
│                    Model Validation (≥95%)                        │
│                                                                   │
│                              ↓                                    │
│                                                                   │
│                    Save Model Artifacts                           │
│                    (models/dna_model_v1/)                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Inference Phase (Production)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User CSV Data  →  Feature Engineering  →  Model Inference       │
│                    (Same pipeline)          (Load pre-trained)   │
│                                                                   │
│                              ↓                                    │
│                                                                   │
│                    DNA Profiles & Archetypes                      │
│                                                                   │
│                              ↓                                    │
│                                                                   │
│                    Streamlit Dashboard                            │
│                    (Existing UI)                                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Architecture

The system consists of four main components:

1. **DNAModelTrainer**: Trains and validates ML models from historical data
2. **DNAFeatureEngineering**: Shared feature extraction pipeline
3. **DNAModelInference**: Loads models and generates predictions
4. **DNAAnalyzerApp**: Updated application using inference instead of raw data

## Components and Interfaces

### 1. DNAFeatureEngineering

**Purpose**: Provides a consistent feature extraction pipeline used by both training and inference systems.

**Interface**:
```python
class DNAFeatureEngineering:
    def __init__(self):
        """Initialize feature engineering pipeline"""
        
    def load_and_process_csv(self, file_path: str, track_name: str) -> pd.DataFrame:
        """Load CSV with proper delimiter detection and column cleaning"""
        
    def convert_time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS.mmm format to seconds"""
        
    def extract_driver_features(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract per-driver features from sector data
        
        Returns DataFrame with columns:
        - driver_id
        - track
        - avg_lap_time, std_lap_time, min_lap_time, lap_count
        - avg_s1, std_s1, min_s1
        - avg_s2, std_s2, min_s2
        - avg_s3, std_s3, min_s3
        - avg_speed, max_speed
        """
        
    def calculate_dna_features(self, driver_track_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate DNA signature features for each driver
        
        Returns DataFrame with columns:
        - driver_id
        - speed_vs_consistency_ratio
        - track_adaptability
        - consistency_index
        - performance_variance
        - speed_consistency
        - technical_track_performance
        - high_speed_track_performance
        - mixed_track_performance
        - sector_balance_score
        """
        
    def create_archetype_labels(self, dna_features: pd.DataFrame) -> pd.Series:
        """
        Create archetype labels based on DNA features
        
        Returns: Series with values in ['Speed Demon', 'Consistency Master', 
                                        'Track Specialist', 'Balanced Racer']
        """
```

**Key Design Decisions**:
- Shared between training and inference to ensure consistency
- Handles all data cleaning, time conversion, and feature extraction
- Encapsulates the domain knowledge from the original PerformanceDNAAnalyzer
- Stateless design for easy testing and reuse

### 2. DNAModelTrainer

**Purpose**: Trains, validates, and saves ML models that predict driver DNA signatures and archetypes.

**Interface**:
```python
class DNAModelTrainer:
    def __init__(self, data_dir: str = ".", models_dir: str = "models"):
        """Initialize trainer with data and model directories"""
        
    def load_all_track_data(self) -> pd.DataFrame:
        """Load data from all track folders and combine"""
        
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare train/val/test splits
        
        Returns: (train_df, val_df, test_df)
        """
        
    def build_dna_regression_model(self, input_dim: int) -> nn.Module:
        """
        Build neural network for DNA signature prediction
        
        Architecture:
        - Input: driver performance features (lap times, speeds, sectors)
        - Hidden layers: [128, 64, 32]
        - Output: DNA signature values (5 continuous values)
        """
        
    def build_archetype_classifier(self, input_dim: int) -> nn.Module:
        """
        Build neural network for archetype classification
        
        Architecture:
        - Input: DNA signature features
        - Hidden layers: [64, 32]
        - Output: 4 classes (archetypes)
        """
        
    def train_models(self, epochs: int = 100) -> Dict[str, Any]:
        """
        Train both regression and classification models
        
        Returns: Dictionary with training history and metrics
        """
        
    def validate_models(self) -> Dict[str, float]:
        """
        Validate models on test set
        
        Returns: Dictionary with metrics:
        - dna_mae: Mean Absolute Error for DNA predictions
        - dna_r2: R² score for DNA predictions
        - archetype_accuracy: Classification accuracy
        - archetype_precision: Weighted precision
        - archetype_recall: Weighted recall
        - archetype_f1: Weighted F1 score
        - overall_reliability: Weighted average (target ≥0.95)
        """
        
    def cross_validate(self, n_folds: int = 5) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation"""
        
    def save_model_artifacts(self, version: str) -> str:
        """
        Save model artifacts to disk
        
        Saves:
        - dna_regression_model.pth
        - archetype_classifier.pth
        - feature_scaler.pkl
        - label_encoder.pkl
        - metadata.json (version, metrics, feature names, etc.)
        
        Returns: Path to saved model directory
        """
        
    def generate_validation_report(self, output_path: str):
        """Generate comprehensive validation report with visualizations"""
```

**Model Architecture Details**:

**DNA Regression Model**:
```
Input Features (15):
- avg_lap_time, std_lap_time, min_lap_time
- avg_s1, std_s1, avg_s2, std_s2, avg_s3, std_s3
- avg_speed, max_speed
- lap_count
- track_type_encoded (3 categories)

Hidden Layers:
- Dense(128) + ReLU + Dropout(0.2)
- Dense(64) + ReLU + Dropout(0.2)
- Dense(32) + ReLU

Output (5):
- speed_vs_consistency_ratio
- track_adaptability
- consistency_index
- performance_variance
- speed_consistency
```

**Archetype Classifier**:
```
Input Features (5):
- DNA signature values from regression model

Hidden Layers:
- Dense(64) + ReLU + Dropout(0.3)
- Dense(32) + ReLU

Output (4):
- Softmax over 4 archetype classes
```

**Key Design Decisions**:
- Two-stage approach: predict DNA signatures first, then classify archetypes
- Neural networks chosen for flexibility and ability to capture non-linear relationships
- Dropout layers prevent overfitting on limited training data
- StandardScaler for feature normalization
- Stratified sampling ensures balanced representation of all archetypes

### 3. DNAModelInference

**Purpose**: Loads pre-trained models and generates predictions for new user data.

**Interface**:
```python
class DNAModelInference:
    def __init__(self, model_dir: str = "models/latest"):
        """
        Initialize inference system
        
        Loads:
        - Pre-trained models
        - Feature scaler
        - Label encoder
        - Metadata
        """
        
    def load_model_artifacts(self):
        """Load all model artifacts from disk"""
        
    def verify_model_integrity(self) -> bool:
        """
        Verify model loaded correctly
        
        Performs test inference with synthetic data
        """
        
    def validate_user_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate user CSV data format
        
        Returns: (is_valid, list_of_errors)
        """
        
    def predict_driver_dna(self, driver_features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict DNA signatures for drivers
        
        Input: Driver performance features
        Output: DataFrame with DNA signature values
        """
        
    def predict_archetypes(self, dna_features: pd.DataFrame) -> pd.Series:
        """
        Predict driver archetypes
        
        Input: DNA signature features
        Output: Series with archetype labels
        """
        
    def create_driver_profiles(self, user_data: pd.DataFrame) -> Dict[int, Dict]:
        """
        Create complete driver profiles matching original format
        
        Returns: Dictionary matching PerformanceDNAAnalyzer.driver_profiles structure
        """
        
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (version, metrics, training date)"""
```

**Key Design Decisions**:
- Maintains same output format as original PerformanceDNAAnalyzer
- Validates data before processing to provide clear error messages
- Lazy loading of models for faster application startup
- Thread-safe for concurrent requests
- Caches loaded models in memory

### 4. Updated Application Components

**PerformanceDNAAnalyzer (Refactored)**:
```python
class PerformanceDNAAnalyzer:
    def __init__(self, use_pretrained: bool = True, model_dir: str = "models/latest"):
        """
        Initialize analyzer
        
        Args:
            use_pretrained: If True, use ML model; if False, use original logic
            model_dir: Path to model artifacts
        """
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            self.inference_engine = DNAModelInference(model_dir)
            self.feature_engineering = DNAFeatureEngineering()
        else:
            # Original implementation for backward compatibility
            self._init_original_implementation()
    
    def load_track_data(self, data_source: Union[str, pd.DataFrame]):
        """
        Load track data from directory or DataFrame
        
        Supports both original folder structure and user-provided CSV
        """
        
    def analyze_sector_performance(self):
        """Extract features using feature engineering pipeline"""
        
    def create_driver_dna_profiles(self):
        """
        Generate DNA profiles
        
        Uses inference engine if use_pretrained=True,
        otherwise uses original calculation logic
        """
```

**Key Design Decisions**:
- Maintains backward compatibility with `use_pretrained` flag
- Same API interface as original implementation
- Allows gradual migration and A/B testing
- Supports both folder-based and DataFrame inputs

## Data Models

### Training Data Schema

**driver_features.csv** (Intermediate training data):
```
driver_id: int
track: str
avg_lap_time: float
std_lap_time: float
min_lap_time: float
lap_count: int
avg_s1: float
std_s1: float
min_s1: float
avg_s2: float
std_s2: float
min_s2: float
avg_s3: float
std_s3: float
min_s3: float
avg_speed: float
max_speed: float
track_type: str (technical/high_speed/mixed)
```

**dna_labels.csv** (Training labels):
```
driver_id: int
speed_vs_consistency_ratio: float
track_adaptability: float
consistency_index: float
performance_variance: float
speed_consistency: float
archetype: str (Speed Demon/Consistency Master/Track Specialist/Balanced Racer)
```

### Model Artifacts Structure

```
models/
├── dna_model_v1/
│   ├── dna_regression_model.pth          # PyTorch model weights
│   ├── archetype_classifier.pth          # PyTorch model weights
│   ├── feature_scaler.pkl                # StandardScaler object
│   ├── label_encoder.pkl                 # LabelEncoder for archetypes
│   ├── metadata.json                     # Model metadata
│   └── validation_report.html            # Validation results
├── dna_model_v2/
│   └── ...
├── latest -> dna_model_v1/               # Symlink to production model
└── model_registry.json                   # Registry of all models
```

**metadata.json**:
```json
{
  "version": "1.0.0",
  "training_date": "2025-10-24T10:30:00Z",
  "model_type": "neural_network",
  "framework": "pytorch",
  "framework_version": "2.0.0",
  "feature_names": ["avg_lap_time", "std_lap_time", ...],
  "target_names": ["speed_vs_consistency_ratio", ...],
  "archetype_classes": ["Speed Demon", "Consistency Master", ...],
  "validation_metrics": {
    "dna_mae": 0.023,
    "dna_r2": 0.967,
    "archetype_accuracy": 0.973,
    "archetype_precision": 0.968,
    "archetype_recall": 0.971,
    "archetype_f1": 0.969,
    "overall_reliability": 0.971
  },
  "cross_validation_scores": {
    "fold_1": 0.968,
    "fold_2": 0.972,
    "fold_3": 0.975,
    "fold_4": 0.969,
    "fold_5": 0.971
  },
  "training_config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15
  },
  "data_statistics": {
    "total_drivers": 38,
    "total_samples": 155,
    "tracks": ["barber", "COTA", "Road America", "Sebring", "Sonoma", "VIR"]
  }
}
```

### User Input Data Schema

Users provide CSV files matching the original format:
```
NUMBER: int (driver ID)
LAP_TIME: str or float (MM:SS.mmm or seconds)
S1: str or float (sector 1 time)
S2: str or float (sector 2 time)
S3: str or float (sector 3 time)
KPH: float (speed in km/h)
track: str (track name)
```

### Output Data Schema

The inference system produces driver profiles matching the original format:
```python
{
    driver_id: int,
    'tracks_raced': List[str],
    'total_races': int,
    'performance_metrics': {
        track_name: {
            'avg_lap_time': float,
            'consistency': float,
            'best_lap': float,
            'sector_balance': {
                'S1_avg': float,
                'S2_avg': float,
                'S3_avg': float
            },
            'speed_profile': {
                'avg_speed': float,
                'top_speed': float
            }
        }
    },
    'dna_signature': {
        'speed_vs_consistency_ratio': float,
        'track_adaptability': float,
        'consistency_index': float,
        'performance_variance': float,
        'speed_consistency': float,
        'track_specialization': Dict[str, float]
    }
}
```

## Error Handling

### Training Phase Errors

1. **Insufficient Data Error**:
   - Trigger: Less than 30 driver-track combinations
   - Action: Log error, suggest collecting more data
   - Recovery: None (training cannot proceed)

2. **Validation Failure Error**:
   - Trigger: Overall reliability < 95%
   - Action: Log detailed metrics, save failed model for analysis
   - Recovery: Adjust hyperparameters and retrain

3. **Data Quality Error**:
   - Trigger: Missing values > 10%, invalid time formats
   - Action: Log problematic files, attempt cleaning
   - Recovery: Skip problematic files or impute values

### Inference Phase Errors

1. **Model Loading Error**:
   - Trigger: Missing model files, corrupted artifacts
   - Action: Log error with file paths, prevent app startup
   - Recovery: Fallback to original implementation if available

2. **Data Validation Error**:
   - Trigger: Missing required columns, wrong data types
   - Action: Return detailed validation report to user
   - Recovery: User corrects data and resubmits

3. **Prediction Error**:
   - Trigger: Unexpected input values, NaN in features
   - Action: Log input data, return partial results if possible
   - Recovery: Skip problematic drivers, warn user

4. **Version Mismatch Error**:
   - Trigger: User data schema doesn't match model expectations
   - Action: Log schema differences, suggest data format
   - Recovery: Attempt automatic schema mapping if possible

### Error Logging Strategy

All errors logged to `logs/dna_analyzer.log` with format:
```
[TIMESTAMP] [LEVEL] [COMPONENT] [ERROR_CODE] Message
```

Example:
```
[2025-10-24 10:30:15] [ERROR] [DNAModelInference] [E001] Model file not found: models/latest/dna_regression_model.pth
[2025-10-24 10:30:15] [INFO] [DNAModelInference] [I001] Falling back to original implementation
```

## Testing Strategy

### Unit Tests

1. **DNAFeatureEngineering Tests**:
   - Test time conversion with various formats
   - Test feature extraction with edge cases (single lap, missing sectors)
   - Test DNA calculation with boundary values
   - Test archetype labeling logic

2. **DNAModelTrainer Tests**:
   - Test data loading from multiple folders
   - Test train/val/test splitting maintains proportions
   - Test model architecture builds correctly
   - Test validation metrics calculation
   - Test model saving and loading

3. **DNAModelInference Tests**:
   - Test model loading with valid/invalid paths
   - Test data validation with various error conditions
   - Test prediction with synthetic data
   - Test output format matches original

### Integration Tests

1. **End-to-End Training Pipeline**:
   - Load real track data → train models → validate → save artifacts
   - Verify all artifacts created correctly
   - Verify metadata accuracy

2. **End-to-End Inference Pipeline**:
   - Load model → process user CSV → generate profiles → display in UI
   - Verify output matches original implementation (within 5%)
   - Verify UI renders correctly

3. **Backward Compatibility Tests**:
   - Run original and new implementation on same data
   - Compare driver profiles, DNA signatures, archetypes
   - Verify ≥95% agreement

### Performance Tests

1. **Training Performance**:
   - Measure training time with full dataset
   - Target: < 10 minutes on standard hardware

2. **Inference Performance**:
   - Measure model loading time (target: < 5 seconds)
   - Measure prediction time for 100 drivers (target: < 10 seconds)
   - Measure memory usage (target: < 2GB)

3. **Concurrent Request Tests**:
   - Simulate 10 concurrent users
   - Verify no degradation in response time
   - Verify no memory leaks

### Validation Tests

1. **Model Accuracy Tests**:
   - Verify DNA prediction MAE < 0.05
   - Verify archetype accuracy ≥ 95%
   - Verify cross-validation consistency (std < 0.02)

2. **Robustness Tests**:
   - Test with noisy data (outliers, missing values)
   - Test with new tracks not in training data
   - Test with extreme performance values

## Deployment Strategy

### Phase 1: Model Training (Week 1)

1. Implement DNAFeatureEngineering class
2. Implement DNAModelTrainer class
3. Train initial model on existing data
4. Validate model meets ≥95% threshold
5. Generate validation report
6. Save model artifacts

### Phase 2: Inference System (Week 2)

1. Implement DNAModelInference class
2. Create model loading and validation logic
3. Implement user data processing pipeline
4. Unit test all components
5. Integration test with sample data

### Phase 3: Application Integration (Week 3)

1. Refactor PerformanceDNAAnalyzer to support both modes
2. Update Streamlit app to use inference system
3. Add model info display in UI
4. Add user data upload functionality
5. Test backward compatibility

### Phase 4: Testing and Validation (Week 4)

1. Run comprehensive test suite
2. Perform A/B testing (original vs. model-based)
3. Validate performance metrics
4. Generate documentation
5. Create migration guide

### Phase 5: Deployment (Week 5)

1. Package model artifacts
2. Update application configuration
3. Deploy to production
4. Monitor performance and errors
5. Collect user feedback

## Best Practices and Recommendations

### Model Development

1. **Version Control**: Use Git LFS for model artifacts
2. **Experiment Tracking**: Use MLflow or Weights & Biases to track experiments
3. **Hyperparameter Tuning**: Use Optuna for automated hyperparameter optimization
4. **Model Monitoring**: Log prediction distributions to detect drift

### Data Management

1. **Data Versioning**: Use DVC to version training datasets
2. **Feature Store**: Consider implementing a feature store for consistency
3. **Data Validation**: Use Great Expectations for data quality checks
4. **Synthetic Data**: Generate synthetic data for testing edge cases

### Production Deployment

1. **Model Serving**: Consider using TorchServe for production deployment
2. **API Gateway**: Wrap inference in FastAPI for RESTful access
3. **Caching**: Cache model predictions for identical inputs
4. **Monitoring**: Set up Prometheus/Grafana for metrics
5. **A/B Testing**: Gradually roll out model-based system

### Maintenance

1. **Retraining Schedule**: Retrain monthly or when new data available
2. **Performance Monitoring**: Track prediction accuracy over time
3. **Model Registry**: Maintain registry of all model versions
4. **Rollback Plan**: Keep previous model version for quick rollback
5. **Documentation**: Update model cards with each new version

### Security

1. **Input Validation**: Sanitize all user inputs
2. **Model Protection**: Encrypt model artifacts at rest
3. **Access Control**: Restrict model directory access
4. **Audit Logging**: Log all model predictions for audit trail
5. **Data Privacy**: Ensure no PII in training data or logs
