#!/usr/bin/env python3
"""
DNA Analyzer Custom Exceptions
Defines custom exception classes for different error scenarios
"""


class DNAAnalyzerError(Exception):
    """Base exception for DNA Analyzer errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """
        Initialize exception
        
        Args:
            message: Error message
            error_code: Optional error code (e.g., 'E001')
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
        full_message = f"[{error_code}] {message}" if error_code else message
        super().__init__(full_message)


# Training Phase Exceptions

class InsufficientDataError(DNAAnalyzerError):
    """Raised when there is insufficient data for training"""
    
    def __init__(self, message: str, sample_count: int = None, required_count: int = 30):
        details = {
            'sample_count': sample_count,
            'required_count': required_count,
            'suggestion': f'Need at least {required_count} driver-track combinations for training'
        }
        super().__init__(message, error_code='E021', details=details)


class DataQualityError(DNAAnalyzerError):
    """Raised when data quality issues are detected"""
    
    def __init__(self, message: str, issues: list = None):
        details = {
            'issues': issues or [],
            'suggestion': 'Clean data by removing outliers, handling missing values, or fixing invalid formats'
        }
        super().__init__(message, error_code='E022', details=details)


class ValidationFailedError(DNAAnalyzerError):
    """Raised when model validation fails to meet threshold"""
    
    def __init__(self, message: str, metrics: dict = None, threshold: float = 0.95):
        details = {
            'metrics': metrics or {},
            'threshold': threshold,
            'suggestion': 'Try adjusting hyperparameters, collecting more data, or improving data quality'
        }
        super().__init__(message, error_code='E032', details=details)


class TrainingError(DNAAnalyzerError):
    """Raised when model training fails"""
    
    def __init__(self, message: str, epoch: int = None):
        details = {
            'epoch': epoch,
            'suggestion': 'Check training data, reduce learning rate, or adjust model architecture'
        }
        super().__init__(message, error_code='E031', details=details)


# Inference Phase Exceptions

class ModelLoadingError(DNAAnalyzerError):
    """Raised when model loading fails"""
    
    def __init__(self, message: str, model_path: str = None, fallback_available: bool = False):
        details = {
            'model_path': model_path,
            'fallback_available': fallback_available,
            'suggestion': 'Check model directory exists and contains all required files' if not fallback_available 
                         else 'Falling back to original implementation'
        }
        super().__init__(message, error_code='E001', details=details)


class DataValidationError(DNAAnalyzerError):
    """Raised when user data validation fails"""
    
    def __init__(self, message: str, validation_errors: list = None, allow_partial: bool = False):
        details = {
            'validation_errors': validation_errors or [],
            'allow_partial': allow_partial,
            'suggestion': 'Fix data format issues: ' + ', '.join(validation_errors or []) if validation_errors 
                         else 'Check data format matches required schema'
        }
        super().__init__(message, error_code='E019', details=details)


class PredictionError(DNAAnalyzerError):
    """Raised when prediction fails"""
    
    def __init__(self, message: str, driver_ids: list = None, partial_results: dict = None):
        details = {
            'failed_driver_ids': driver_ids or [],
            'partial_results_available': partial_results is not None,
            'partial_results': partial_results,
            'suggestion': 'Check input data quality for failed drivers'
        }
        super().__init__(message, error_code='E041', details=details)


class ModelIntegrityError(DNAAnalyzerError):
    """Raised when model integrity check fails"""
    
    def __init__(self, message: str, check_type: str = None):
        details = {
            'check_type': check_type,
            'suggestion': 'Model may be corrupted. Try retraining or using a different model version'
        }
        super().__init__(message, error_code='E018', details=details)


# Data Validation Exceptions

class MissingColumnsError(DNAAnalyzerError):
    """Raised when required columns are missing"""
    
    def __init__(self, message: str, missing_columns: list = None, available_columns: list = None):
        details = {
            'missing_columns': missing_columns or [],
            'available_columns': available_columns or [],
            'suggestion': f'Add missing columns: {", ".join(missing_columns or [])}'
        }
        super().__init__(message, error_code='E052', details=details)


class InvalidDataFormatError(DNAAnalyzerError):
    """Raised when data format is invalid"""
    
    def __init__(self, message: str, column: str = None, expected_format: str = None):
        details = {
            'column': column,
            'expected_format': expected_format,
            'suggestion': f'Column "{column}" should be in format: {expected_format}' if column and expected_format 
                         else 'Check data format matches expected schema'
        }
        super().__init__(message, error_code='E046', details=details)


class EmptyDataError(DNAAnalyzerError):
    """Raised when DataFrame is empty"""
    
    def __init__(self, message: str = "DataFrame is empty"):
        details = {
            'suggestion': 'Provide non-empty data for analysis'
        }
        super().__init__(message, error_code='E051', details=details)


# Feature Engineering Exceptions

class FeatureExtractionError(DNAAnalyzerError):
    """Raised when feature extraction fails"""
    
    def __init__(self, message: str, stage: str = None):
        details = {
            'stage': stage,
            'suggestion': 'Check input data format and values'
        }
        super().__init__(message, error_code='E028', details=details)


class DNACalculationError(DNAAnalyzerError):
    """Raised when DNA calculation fails"""
    
    def __init__(self, message: str, driver_id: int = None):
        details = {
            'driver_id': driver_id,
            'suggestion': 'Driver may need data from multiple tracks for DNA calculation'
        }
        super().__init__(message, error_code='E029', details=details)


def format_error_message(error: DNAAnalyzerError) -> str:
    """
    Format error message with details and suggestions
    
    Args:
        error: DNAAnalyzerError instance
        
    Returns:
        Formatted error message string
    """
    lines = [str(error)]
    
    if error.details:
        lines.append("\nDetails:")
        for key, value in error.details.items():
            if key != 'suggestion' and value is not None:
                lines.append(f"  - {key}: {value}")
        
        if 'suggestion' in error.details:
            lines.append(f"\nSuggestion: {error.details['suggestion']}")
    
    return '\n'.join(lines)


if __name__ == "__main__":
    # Test custom exceptions
    print("🧪 Testing DNA Analyzer Exceptions")
    print("=" * 50)
    
    # Test InsufficientDataError
    try:
        raise InsufficientDataError(
            "Not enough training data",
            sample_count=15,
            required_count=30
        )
    except InsufficientDataError as e:
        print(f"\n{format_error_message(e)}")
    
    # Test DataValidationError
    try:
        raise DataValidationError(
            "User data validation failed",
            validation_errors=['Missing column: LAP_TIME', 'Invalid speed values'],
            allow_partial=True
        )
    except DataValidationError as e:
        print(f"\n{format_error_message(e)}")
    
    # Test PredictionError with partial results
    try:
        raise PredictionError(
            "Prediction failed for some drivers",
            driver_ids=[101, 102],
            partial_results={'driver_103': {'archetype': 'Speed Demon'}}
        )
    except PredictionError as e:
        print(f"\n{format_error_message(e)}")
    
    print("\n✅ Exception testing complete")
