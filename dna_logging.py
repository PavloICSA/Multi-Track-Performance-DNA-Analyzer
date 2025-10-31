#!/usr/bin/env python3
"""
DNA Logging Infrastructure
Centralized logging setup with structured format, error codes, and log rotation
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime
import sys

from config import LOGS_DIR

# Error code definitions
ERROR_CODES = {
    # Model Loading Errors (E001-E020)
    'E001': 'Model directory not found',
    'E002': 'Metadata file not found',
    'E003': 'Corrupted metadata file',
    'E004': 'Feature scaler not found',
    'E005': 'Failed to load feature scaler',
    'E006': 'Label encoder not found',
    'E007': 'Failed to load label encoder',
    'E008': 'DNA regression model not found',
    'E009': 'Failed to load DNA regression model',
    'E010': 'Archetype classifier not found',
    'E011': 'Failed to load archetype classifier',
    'E012': 'Metadata missing required fields',
    'E013': 'DNA model output shape mismatch',
    'E014': 'DNA model produced NaN values',
    'E015': 'DNA model produced infinite values',
    'E016': 'Archetype model output shape mismatch',
    'E017': 'Archetype predictions out of range',
    'E018': 'Model integrity verification failed',
    'E019': 'User data validation failed',
    'E020': 'No valid driver features extracted',
    
    # Training Errors (E021-E040)
    'E021': 'Insufficient training data',
    'E022': 'Data quality issues detected',
    'E023': 'Missing required columns in training data',
    'E024': 'Invalid time format in training data',
    'E025': 'Invalid speed values in training data',
    'E026': 'Track data directory not found',
    'E027': 'Failed to load track data file',
    'E028': 'Feature extraction failed',
    'E029': 'DNA calculation failed',
    'E030': 'Archetype labeling failed',
    'E031': 'Model training failed',
    'E032': 'Validation failed - reliability below threshold',
    'E033': 'Cross-validation failed',
    'E034': 'Model saving failed',
    'E035': 'Failed to update model registry',
    'E036': 'Training data split failed',
    'E037': 'Model architecture build failed',
    'E038': 'Optimizer initialization failed',
    'E039': 'Training epoch failed',
    'E040': 'Validation metrics calculation failed',
    
    # Inference Errors (E041-E060)
    'E041': 'Prediction failed',
    'E042': 'Feature scaling failed',
    'E043': 'DNA prediction failed',
    'E044': 'Archetype prediction failed',
    'E045': 'Driver profile creation failed',
    'E046': 'Invalid input data format',
    'E047': 'Missing required features',
    'E048': 'Feature value out of range',
    'E049': 'Partial prediction failure',
    'E050': 'Model version incompatibility',
    
    # Data Validation Errors (E051-E070)
    'E051': 'Empty DataFrame',
    'E052': 'Missing required columns',
    'E053': 'Invalid data types',
    'E054': 'Invalid time format',
    'E055': 'Invalid speed values',
    'E056': 'Invalid driver ID',
    'E057': 'Invalid track name',
    'E058': 'Too many missing values',
    'E059': 'Data quality below threshold',
    'E060': 'Outlier detection failed',
}

# Warning code definitions
WARNING_CODES = {
    'W001': 'Feature mismatch between model and data',
    'W002': 'Using fallback implementation',
    'W003': 'Partial data available',
    'W004': 'Performance degradation detected',
    'W005': 'High memory usage',
    'W006': 'Slow inference time',
    'W007': 'Model version mismatch',
    'W008': 'Data quality warning',
    'W009': 'Missing optional features',
    'W010': 'Symlink creation failed',
}

# Info code definitions
INFO_CODES = {
    'I001': 'Falling back to original implementation',
    'I002': 'Model loaded successfully',
    'I003': 'Training started',
    'I004': 'Training completed',
    'I005': 'Validation passed',
    'I006': 'Model saved successfully',
    'I007': 'Inference completed',
    'I008': 'Data validation passed',
    'I009': 'Feature engineering completed',
    'I010': 'Cross-validation completed',
}


class DNALogger:
    """
    Centralized logger for DNA Analyzer with structured logging,
    error codes, and log rotation
    """
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup_logging(cls, log_level: str = 'INFO', log_to_console: bool = True):
        """
        Set up logging infrastructure with file rotation and structured format
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_console: Whether to also log to console
        """
        if cls._initialized:
            return
        
        # Ensure logs directory exists
        LOGS_DIR.mkdir(exist_ok=True)
        
        # Create log file path
        log_file = LOGS_DIR / "dna_analyzer.log"
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        root_logger.handlers = []
        
        # Create formatter with structured format
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler (optional)
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        cls._initialized = True
        
        # Log initialization
        root_logger.info(f"Logging initialized - Log file: {log_file}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger for a specific module
        
        Args:
            name: Logger name (typically __name__ of the module)
            
        Returns:
            Logger instance
        """
        if not cls._initialized:
            cls.setup_logging()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @staticmethod
    def log_error(logger: logging.Logger, error_code: str, message: str, 
                  exception: Optional[Exception] = None):
        """
        Log an error with error code
        
        Args:
            logger: Logger instance
            error_code: Error code (e.g., 'E001')
            message: Error message
            exception: Optional exception object
        """
        error_desc = ERROR_CODES.get(error_code, 'Unknown error')
        full_message = f"[{error_code}] {error_desc}: {message}"
        
        if exception:
            logger.error(full_message, exc_info=True)
        else:
            logger.error(full_message)
    
    @staticmethod
    def log_warning(logger: logging.Logger, warning_code: str, message: str):
        """
        Log a warning with warning code
        
        Args:
            logger: Logger instance
            warning_code: Warning code (e.g., 'W001')
            message: Warning message
        """
        warning_desc = WARNING_CODES.get(warning_code, 'Unknown warning')
        full_message = f"[{warning_code}] {warning_desc}: {message}"
        logger.warning(full_message)
    
    @staticmethod
    def log_info(logger: logging.Logger, info_code: str, message: str):
        """
        Log an info message with info code
        
        Args:
            logger: Logger instance
            info_code: Info code (e.g., 'I001')
            message: Info message
        """
        info_desc = INFO_CODES.get(info_code, 'Info')
        full_message = f"[{info_code}] {info_desc}: {message}"
        logger.info(full_message)


class ErrorContext:
    """
    Context manager for error handling with automatic logging
    """
    
    def __init__(self, logger: logging.Logger, operation: str, 
                 error_code: str, cleanup_func=None):
        """
        Initialize error context
        
        Args:
            logger: Logger instance
            operation: Description of operation being performed
            error_code: Error code to use if operation fails
            cleanup_func: Optional cleanup function to call on error
        """
        self.logger = logger
        self.operation = operation
        self.error_code = error_code
        self.cleanup_func = cleanup_func
    
    def __enter__(self):
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            DNALogger.log_error(
                self.logger,
                self.error_code,
                f"{self.operation} failed: {exc_val}",
                exception=exc_val
            )
            
            if self.cleanup_func:
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    self.logger.error(f"Cleanup failed: {cleanup_error}")
            
            # Don't suppress the exception
            return False
        else:
            self.logger.debug(f"Completed: {self.operation}")
            return True


def setup_logging(log_level: str = 'INFO', log_to_console: bool = True):
    """
    Convenience function to set up logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to also log to console
    """
    DNALogger.setup_logging(log_level, log_to_console)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        Logger instance
    """
    return DNALogger.get_logger(name)


# Initialize logging on module import
DNALogger.setup_logging()


if __name__ == "__main__":
    # Test logging infrastructure
    print("🧪 Testing DNA Logging Infrastructure")
    print("=" * 50)
    
    # Get a test logger
    logger = get_logger(__name__)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test structured logging with codes
    DNALogger.log_info(logger, 'I002', "Model loaded successfully")
    DNALogger.log_warning(logger, 'W001', "Feature mismatch detected")
    DNALogger.log_error(logger, 'E001', "Model directory not found")
    
    # Test error context
    try:
        with ErrorContext(logger, "Test operation", 'E031'):
            raise ValueError("Test error")
    except ValueError:
        pass
    
    print(f"\n✅ Logging test complete. Check {LOGS_DIR / 'dna_analyzer.log'}")
