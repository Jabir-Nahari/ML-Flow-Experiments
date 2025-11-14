"""
Comprehensive Error Handling and Validation for Student C Pipeline
Provides robust error handling, retry mechanisms, and validation for ML pipeline components.
"""

import os
import sys
import logging
import time
import psutil
import functools
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import boto3
from botocore.exceptions import NoCredentialsError, ClientError, EndpointConnectionError
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error_handling.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Custom Exception Classes
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class DataIngestionError(PipelineError):
    """Exception raised for data ingestion failures."""
    pass

class DataValidationError(PipelineError):
    """Exception raised for data validation failures."""
    pass

class ModelTrainingError(PipelineError):
    """Exception raised for model training failures."""
    pass

class ModelValidationError(PipelineError):
    """Exception raised for model validation failures."""
    pass

class MLflowError(PipelineError):
    """Exception raised for MLflow operation failures."""
    pass

class S3ConnectivityError(PipelineError):
    """Exception raised for S3 connectivity issues."""
    pass

class ResourceLimitError(PipelineError):
    """Exception raised for resource limit violations."""
    pass

class NetworkError(PipelineError):
    """Exception raised for network-related failures."""
    pass

# Retry Decorator
def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0,
                    exceptions: tuple = (Exception,)):
    """
    Decorator to retry function calls on specified exceptions.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
                        raise e

            raise last_exception
        return wrapper
    return decorator

# Resource Monitoring
class ResourceMonitor:
    """Monitor system resources during pipeline execution."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.start_cpu = None
        self.peak_memory = 0
        self.peak_cpu = 0

    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent(interval=None)
        self.peak_memory = self.start_memory
        self.peak_cpu = self.start_cpu
        logger.info(".2f")

    def update_metrics(self):
        """Update current resource metrics."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_cpu = self.process.cpu_percent(interval=None)

        self.peak_memory = max(self.peak_memory, current_memory)
        self.peak_cpu = max(self.peak_cpu, current_cpu)

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage statistics."""
        self.update_metrics()
        return {
            'current_memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'peak_memory_mb': self.peak_memory,
            'current_cpu_percent': self.process.cpu_percent(interval=None),
            'peak_cpu_percent': self.peak_cpu
        }

    def check_resource_limits(self, memory_limit_mb: float = 2048, cpu_limit_percent: float = 90):
        """Check if resource usage exceeds limits."""
        usage = self.get_resource_usage()

        if usage['current_memory_mb'] > memory_limit_mb:
            raise ResourceLimitError(f"Memory usage ({usage['current_memory_mb']:.2f} MB) exceeds limit ({memory_limit_mb} MB)")

        if usage['current_cpu_percent'] > cpu_limit_percent:
            raise ResourceLimitError(f"CPU usage ({usage['current_cpu_percent']:.2f}%) exceeds limit ({cpu_limit_percent}%)")

# Data Validation
class DataValidator:
    """Validate data quality and integrity."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None,
                         min_rows: int = 1, max_missing_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Validate DataFrame quality.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            max_missing_ratio: Maximum allowed missing value ratio

        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        try:
            # Basic structure validation
            if df.empty:
                validation_results['errors'].append("DataFrame is empty")
                validation_results['is_valid'] = False
                return validation_results

            if len(df) < min_rows:
                validation_results['errors'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
                validation_results['is_valid'] = False

            # Column validation
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    validation_results['errors'].append(f"Missing required columns: {missing_columns}")
                    validation_results['is_valid'] = False

            # Missing value validation
            missing_ratios = df.isnull().sum() / len(df)
            high_missing_cols = missing_ratios[missing_ratios > max_missing_ratio]

            if not high_missing_cols.empty:
                validation_results['errors'].append(f"Columns with high missing ratios: {dict(high_missing_cols)}")
                validation_results['is_valid'] = False

            # Data type validation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                validation_results['warnings'].append("No numeric columns found")

            # Statistical validation
            validation_results['stats'] = {
                'shape': df.shape,
                'dtypes': df.dtypes.to_dict(),
                'missing_ratios': missing_ratios.to_dict(),
                'numeric_stats': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
            }

            # Check for infinite values
            if df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any():
                validation_results['warnings'].append("Infinite values detected in numeric columns")

            # Check for duplicate rows
            duplicate_ratio = df.duplicated().sum() / len(df)
            if duplicate_ratio > 0.1:
                validation_results['warnings'].append(f"High duplicate ratio: {duplicate_ratio:.2%}")

        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            validation_results['is_valid'] = False

        return validation_results

    @staticmethod
    def validate_features(X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Validate feature data for ML training.

        Args:
            X: Feature DataFrame
            y: Target series (optional)

        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'feature_stats': {}
        }

        try:
            # Feature validation
            if X.empty:
                validation_results['errors'].append("Feature matrix is empty")
                validation_results['is_valid'] = False
                return validation_results

            # Check for constant features
            constant_features = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_features.append(col)

            if constant_features:
                validation_results['errors'].append(f"Constant features detected: {constant_features}")
                validation_results['is_valid'] = False

            # Check for high correlation (multicollinearity warning)
            if len(X.columns) > 1:
                numeric_X = X.select_dtypes(include=[np.number])
                if len(numeric_X.columns) > 1:
                    corr_matrix = numeric_X.corr()
                    high_corr = (corr_matrix.abs() > 0.95) & (corr_matrix != 1.0)
                    correlated_pairs = []
                    for i in range(len(high_corr.columns)):
                        for j in range(i+1, len(high_corr.columns)):
                            if high_corr.iloc[i, j]:
                                correlated_pairs.append((high_corr.columns[i], high_corr.columns[j]))

                    if correlated_pairs:
                        validation_results['warnings'].append(f"Highly correlated features: {correlated_pairs}")

            # Target validation
            if y is not None:
                if len(y) != len(X):
                    validation_results['errors'].append(f"Feature matrix ({len(X)} rows) and target ({len(y)} rows) have different lengths")
                    validation_results['is_valid'] = False

                if y.nunique() < 2:
                    validation_results['errors'].append("Target has less than 2 unique classes")
                    validation_results['is_valid'] = False

            validation_results['feature_stats'] = {
                'n_features': X.shape[1],
                'n_samples': X.shape[0],
                'feature_types': X.dtypes.value_counts().to_dict(),
                'target_classes': y.nunique() if y is not None else None
            }

        except Exception as e:
            validation_results['errors'].append(f"Feature validation failed: {str(e)}")
            validation_results['is_valid'] = False

        return validation_results

# Model Validation
class ModelValidator:
    """Validate trained models."""

    @staticmethod
    def validate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                      accuracy_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Validate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            accuracy_threshold: Minimum acceptable accuracy

        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }

        try:
            # Generate predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted')
            }

            validation_results['metrics'] = metrics

            # Check accuracy threshold
            if metrics['accuracy'] < accuracy_threshold:
                validation_results['errors'].append(f"Model accuracy ({metrics['accuracy']:.4f}) below threshold ({accuracy_threshold})")
                validation_results['is_valid'] = False

            # Check for overfitting (if training metrics available)
            # This would be enhanced with training metrics comparison

            # Check prediction distribution
            pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
            pred_distribution = dict(zip(pred_classes, pred_counts))

            if len(pred_classes) == 1:
                validation_results['warnings'].append("Model predicts only one class")

            validation_results['prediction_stats'] = {
                'n_predictions': len(y_pred),
                'unique_predictions': len(pred_classes),
                'prediction_distribution': pred_distribution
            }

        except Exception as e:
            validation_results['errors'].append(f"Model validation failed: {str(e)}")
            validation_results['is_valid'] = False

        return validation_results

# Network and S3 Utilities
class NetworkUtils:
    """Utilities for network operations and S3 connectivity."""

    @staticmethod
    @retry_on_failure(max_retries=3, delay=1.0, exceptions=(ClientError, EndpointConnectionError))
    def test_s3_connectivity(bucket_name: str, region: str = 'us-east-1') -> bool:
        """
        Test S3 connectivity and permissions.

        Args:
            bucket_name: S3 bucket name
            region: AWS region

        Returns:
            True if connectivity is successful
        """
        try:
            s3_client = boto3.client('s3', region_name=region)
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 connectivity test passed for bucket: {bucket_name}")
            return True
        except NoCredentialsError:
            raise S3ConnectivityError("AWS credentials not found or invalid")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise S3ConnectivityError(f"S3 bucket '{bucket_name}' does not exist")
            elif e.response['Error']['Code'] == '403':
                raise S3ConnectivityError(f"Access denied to S3 bucket '{bucket_name}'")
            else:
                raise S3ConnectivityError(f"S3 connectivity error: {e}")
        except Exception as e:
            raise S3ConnectivityError(f"Unexpected S3 error: {e}")

    @staticmethod
    @retry_on_failure(max_retries=3, delay=2.0, exceptions=(RequestException, Timeout, ConnectionError))
    def test_network_connectivity(url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Test network connectivity to a URL.

        Args:
            url: URL to test
            timeout: Request timeout in seconds

        Returns:
            Connectivity test results
        """
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            response_time = time.time() - start_time

            result = {
                'success': True,
                'status_code': response.status_code,
                'response_time': response_time,
                'error': None
            }

            if response.status_code != 200:
                result['success'] = False
                result['error'] = f"HTTP {response.status_code}"

            logger.info(f"Network connectivity test for {url}: {result}")
            return result

        except Timeout:
            raise NetworkError(f"Network timeout connecting to {url}")
        except ConnectionError:
            raise NetworkError(f"Network connection error for {url}")
        except RequestException as e:
            raise NetworkError(f"Network request failed for {url}: {e}")

# MLflow Integration
class MLflowErrorLogger:
    """Log errors and validation results to MLflow."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id

    def log_error(self, error_type: str, error_message: str, error_details: Optional[Dict[str, Any]] = None):
        """Log error to MLflow."""
        try:
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_param(f"error_{error_type}_occurred", True)
                mlflow.log_param(f"error_{error_type}_message", error_message)

                if error_details:
                    for key, value in error_details.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"error_{error_type}_{key}", value)
                        else:
                            mlflow.log_param(f"error_{error_type}_{key}", str(value))

                logger.info(f"Error logged to MLflow: {error_type} - {error_message}")

        except Exception as e:
            logger.error(f"Failed to log error to MLflow: {e}")

    def log_validation_results(self, validation_type: str, results: Dict[str, Any]):
        """Log validation results to MLflow."""
        try:
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_param(f"validation_{validation_type}_passed", results.get('is_valid', False))

                # Log errors and warnings
                if 'errors' in results and results['errors']:
                    mlflow.log_param(f"validation_{validation_type}_errors", '; '.join(results['errors']))

                if 'warnings' in results and results['warnings']:
                    mlflow.log_param(f"validation_{validation_type}_warnings", '; '.join(results['warnings']))

                # Log stats
                if 'stats' in results:
                    self._log_nested_dict(f"validation_{validation_type}_stats", results['stats'])

                if 'metrics' in results:
                    for key, value in results['metrics'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"validation_{validation_type}_{key}", value)

                logger.info(f"Validation results logged to MLflow: {validation_type}")

        except Exception as e:
            logger.error(f"Failed to log validation results to MLflow: {e}")

    def log_resource_usage(self, resource_usage: Dict[str, float]):
        """Log resource usage metrics to MLflow."""
        try:
            with mlflow.start_run(run_id=self.run_id):
                for key, value in resource_usage.items():
                    mlflow.log_metric(f"resource_{key}", value)

                logger.info("Resource usage logged to MLflow")

        except Exception as e:
            logger.error(f"Failed to log resource usage to MLflow: {e}")

    def _log_nested_dict(self, prefix: str, data: Dict[str, Any], max_depth: int = 2):
        """Recursively log nested dictionary to MLflow."""
        if max_depth <= 0:
            return

        for key, value in data.items():
            param_key = f"{prefix}_{key}"
            if isinstance(value, dict):
                self._log_nested_dict(param_key, value, max_depth - 1)
            elif isinstance(value, (int, float)):
                mlflow.log_metric(param_key, value)
            else:
                mlflow.log_param(param_key, str(value))

# Error Handling Context Manager
class ErrorHandler:
    """Context manager for comprehensive error handling."""

    def __init__(self, operation_name: str, mlflow_logger: Optional[MLflowErrorLogger] = None,
                 resource_monitor: Optional[ResourceMonitor] = None):
        self.operation_name = operation_name
        self.mlflow_logger = mlflow_logger
        self.resource_monitor = resource_monitor
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting operation: {self.operation_name}")
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time

        if exc_type is not None:
            error_message = f"{exc_type.__name__}: {exc_val}"
            logger.error(f"Operation '{self.operation_name}' failed after {duration:.2f}s: {error_message}")

            if self.mlflow_logger:
                error_details = {
                    'operation': self.operation_name,
                    'duration_seconds': duration,
                    'error_type': exc_type.__name__
                }
                self.mlflow_logger.log_error('operation_failure', error_message, error_details)

            # Log resource usage at failure
            if self.resource_monitor:
                resource_usage = self.resource_monitor.get_resource_usage()
                resource_usage['operation_duration'] = duration
                if self.mlflow_logger:
                    self.mlflow_logger.log_resource_usage(resource_usage)

            return False  # Re-raise the exception

        else:
            logger.info(f"Operation '{self.operation_name}' completed successfully in {duration:.2f}s")

            # Log successful resource usage
            if self.resource_monitor:
                resource_usage = self.resource_monitor.get_resource_usage()
                resource_usage['operation_duration'] = duration
                if self.mlflow_logger:
                    self.mlflow_logger.log_resource_usage(resource_usage)

            return True

# Global instances for easy access
resource_monitor = ResourceMonitor()
data_validator = DataValidator()
model_validator = ModelValidator()
network_utils = NetworkUtils()

def initialize_error_handling(run_id: Optional[str] = None) -> MLflowErrorLogger:
    """Initialize error handling with MLflow logging."""
    return MLflowErrorLogger(run_id)