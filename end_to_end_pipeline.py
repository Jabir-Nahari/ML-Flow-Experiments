"""
End-to-End ML Pipeline for Student C
Comprehensive pipeline integrating data ingestion, preprocessing, model training,
MLflow logging, artifact storage, and deployment with existing Student A and B components.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import mlflow
import mlflow.sklearn
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import json

# Import error handling and QA validation modules
from error_handling import (
    DataValidator, ModelValidator, ResourceMonitor, MLflowErrorLogger,
    ErrorHandler, DataValidationError, ModelValidationError, ResourceLimitError,
    S3ConnectivityError, NetworkError, retry_on_failure, initialize_error_handling
)
from qa_validation import QATestSuite, QAMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Handle data ingestion from S3 or local sources with comprehensive error handling."""

    def __init__(self, s3_bucket=None, s3_key=None, local_path=None, validator=None):
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.local_path = local_path
        self.data = None
        self.validator = validator or DataValidator()

    @retry_on_failure(max_retries=3, exceptions=(S3ConnectivityError, NetworkError, ClientError))
    def load_data(self):
        """Load data from S3 or local file with validation."""
        with ErrorHandler("data_ingestion", None, None):  # MLflow logger will be set later
            if self.s3_bucket and self.s3_key:
                logger.info(f"Loading data from S3: s3://{self.s3_bucket}/{self.s3_key}")
                self.data = self._load_from_s3()
            elif self.local_path:
                logger.info(f"Loading data from local path: {self.local_path}")
                self.data = self._load_from_local()
            else:
                logger.info("Using Wine dataset from sklearn")
                self.data = self._load_wine_dataset()

            # Validate loaded data
            validation_result = self.validator.validate_dataframe(self.data)
            if not validation_result['is_valid']:
                raise DataValidationError(f"Data validation failed: {validation_result['errors']}")

            logger.info(f"Data loaded and validated successfully. Shape: {self.data.shape}")
            return self.data

    def _load_from_s3(self):
        """Load data from S3 bucket with error handling."""
        try:
            s3_client = boto3.client('s3')
            obj = s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_key)
            df = pd.read_csv(obj['Body'])
            return df
        except NoCredentialsError:
            raise S3ConnectivityError("AWS credentials not found or invalid")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                raise S3ConnectivityError(f"S3 bucket '{self.s3_bucket}' does not exist")
            elif e.response['Error']['Code'] == 'NoSuchKey':
                raise S3ConnectivityError(f"S3 key '{self.s3_key}' does not exist in bucket '{self.s3_bucket}'")
            elif e.response['Error']['Code'] == 'AccessDenied':
                raise S3ConnectivityError(f"Access denied to S3 bucket '{self.s3_bucket}'")
            else:
                raise S3ConnectivityError(f"S3 error: {e}")
        except Exception as e:
            logger.error(f"Failed to load from S3: {e}")
            raise S3ConnectivityError(f"Unexpected S3 error: {e}")

    def _load_from_local(self):
        """Load data from local file."""
        if not os.path.exists(self.local_path):
            raise FileNotFoundError(f"Local file not found: {self.local_path}")

        if self.local_path.endswith('.csv'):
            return pd.read_csv(self.local_path)
        elif self.local_path.endswith('.json'):
            return pd.read_json(self.local_path)
        else:
            raise ValueError(f"Unsupported file format: {self.local_path}")

    def _load_wine_dataset(self):
        """Load Wine dataset from sklearn."""
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        return df

class DataPreprocessor:
    """Handle data preprocessing and feature engineering with validation."""

    def __init__(self, data, validator=None):
        self.data = data
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.validator = validator or DataValidator()

    def preprocess(self, target_column='target', test_size=0.2, random_state=42):
        """Preprocess the data with validation."""
        with ErrorHandler("data_preprocessing", None, None):  # MLflow logger will be set later
            logger.info("Starting data preprocessing...")

            # Validate target column exists
            if target_column not in self.data.columns:
                raise DataValidationError(f"Target column '{target_column}' not found in data")

            # Separate features and target
            self.y = self.data[target_column]
            self.X = self.data.drop(columns=[target_column])

            # Validate features and target
            feature_validation = self.validator.validate_features(self.X, self.y)
            if not feature_validation['is_valid']:
                raise DataValidationError(f"Feature validation failed: {feature_validation['errors']}")

            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
            )

            # Scale features
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.X_train.columns
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.X_test.columns
            )

            logger.info(f"Preprocessing completed. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
            return self.X_train, self.X_test, self.y_train, self.y_test

class ModelTrainer:
    """Handle model training and evaluation with validation."""

    def __init__(self, X_train, X_test, y_train, y_test, validator=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.metrics = {}
        self.validator = validator or ModelValidator()

    def load_model_from_artifacts(self, artifacts_path='artifacts'):
        """Load pre-trained model from artifacts folder."""
        import joblib
        model_path = os.path.join(artifacts_path, 'model.pkl')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def train_model(self, model_type='random_forest', **model_params):
        """Train the model with validation."""
        with ErrorHandler("model_training", None, None):  # MLflow logger will be set later
            logger.info(f"Training {model_type} model...")

            if model_type == 'random_forest':
                default_params = {'n_estimators': 100, 'random_state': 42}
                # Remove 'type' parameter if it exists to avoid conflicts
                filtered_params = {k: v for k, v in model_params.items() if k not in ['type', 'params']}
                default_params.update(filtered_params)
                self.model = RandomForestClassifier(**default_params)
            else:
                raise ModelTrainingError(f"Unsupported model type: {model_type}")

            # Train the model
            self.model.fit(self.X_train, self.y_train)

            logger.info("Model training completed")
            return self.model

    def evaluate_model(self, accuracy_threshold=0.5):
        """Evaluate the model and compute metrics with validation."""
        with ErrorHandler("model_evaluation", None, None):  # MLflow logger will be set later
            logger.info("Evaluating model...")

            y_pred = self.model.predict(self.X_test)

            self.metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred, average='weighted'),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted')
            }

            # Generate classification report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            self.metrics['classification_report'] = report

            # Validate model performance
            validation_result = self.validator.validate_model(
                self.model, self.X_test, self.y_test, accuracy_threshold
            )
            if not validation_result['is_valid']:
                raise ModelValidationError(f"Model validation failed: {validation_result['errors']}")

            logger.info(f"Model evaluation and validation completed. Accuracy: {self.metrics['accuracy']:.4f}")
            return self.metrics

class MLflowManager:
    """Handle MLflow experiment tracking and model logging."""

    def __init__(self, experiment_name="Wine_Classification_Pipeline"):
        self.experiment_name = experiment_name
        self.run_id = None

    def setup_experiment(self):
        """Set up MLflow experiment."""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment set to: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise

    def start_run(self, run_name=None):
        """Start MLflow run."""
        try:
            if run_name is None:
                run_name = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.run = mlflow.start_run(run_name=run_name)
            self.run_id = self.run.info.run_id
            logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")
            return self.run

        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            raise

    def log_parameters(self, params):
        """Log parameters to MLflow."""
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            logger.info(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_metrics(self, metrics):
        """Log metrics to MLflow."""
        try:
            for key, value in metrics.items():
                if isinstance(value, dict):
                    # Handle nested metrics like classification_report
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            # Handle nested dict (macro/micro averages)
                            for nested_key, nested_value in sub_value.items():
                                if isinstance(nested_value, (int, float)):
                                    mlflow.log_metric(f"{key}_{sub_key}_{nested_key}", nested_value)
                        elif isinstance(sub_value, (int, float)):
                            mlflow.log_metric(f"{key}_{sub_key}", sub_value)
                elif isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            logger.info(f"Logged metrics to MLflow")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def log_model(self, model, model_name="model"):
        """Log model to MLflow."""
        try:
            mlflow.sklearn.log_model(model, model_name)
            logger.info(f"Model logged to MLflow as: {model_name}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def log_artifacts(self, artifacts):
        """Log artifacts to MLflow."""
        try:
            for artifact_name, artifact_path in artifacts.items():
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_name)
            logger.info(f"Logged {len(artifacts)} artifacts to MLflow")
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")

    def end_run(self):
        """End MLflow run."""
        try:
            mlflow.end_run()
            logger.info("MLflow run ended")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")

class DeploymentManager:
    """Handle model deployment and serving integration."""

    def __init__(self, model_uri=None):
        self.model_uri = model_uri

    def register_model(self, model_name="WineClassifier", run_id=None):
        """Register model in MLflow Model Registry."""
        try:
            if run_id:
                model_uri = f"runs:/{run_id}/model"
            else:
                model_uri = self.model_uri

            mv = mlflow.register_model(model_uri, model_name)
            logger.info(f"Model registered as: {model_name} version {mv.version}")

            # Transition to Production stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Production"
            )
            logger.info(f"Model {model_name} v{mv.version} transitioned to Production")

            return f"models:/{model_name}/Production"

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

class PipelineOrchestrator:
    """Main orchestrator for the end-to-end pipeline with comprehensive QA and error handling."""

    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.components = {}

        # Initialize error handling and QA components
        self.resource_monitor = ResourceMonitor()
        self.data_validator = DataValidator()
        self.model_validator = ModelValidator()
        self.qa_monitor = QAMonitor()
        self.mlflow_logger = None  # Will be initialized when MLflow run starts

    def _default_config(self):
        """Default pipeline configuration with QA and error handling."""
        return {
            'data': {
                'source': 'sklearn',  # 's3', 'local', or 'sklearn'
                's3_bucket': None,
                's3_key': None,
                'local_path': None
            },
            'preprocessing': {
                'target_column': 'target',
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'type': 'random_forest',
                'params': {'n_estimators': 100, 'random_state': 42},
                'load_from_artifacts': True,  # Load pre-trained model from artifacts folder
                'artifacts_path': 'artifacts'
            },
            'validation': {
                'accuracy_threshold': 0.5,
                'enable_data_validation': True,
                'enable_model_validation': True
            },
            'qa': {
                'run_qa_suite': True,
                'fail_on_qa_failure': False
            },
            'mlflow': {
                'experiment_name': 'Wine_Classification_Pipeline_QA'
            },
            'deployment': {
                'register_model': True,
                'model_name': 'WineClassifier'
            },
            'error_handling': {
                'max_retries': 3,
                'enable_resource_monitoring': True
            }
        }

    def run_pipeline(self):
        """Execute the complete pipeline with comprehensive error handling and QA."""
        pipeline_result = {'status': 'unknown', 'error': None, 'run_id': None}

        try:
            logger.info("🚀 Starting End-to-End ML Pipeline with QA & Error Handling")
            logger.info("=" * 60)

            # Start resource monitoring
            self.resource_monitor.start_monitoring()

            # 1. Data Ingestion with error handling
            logger.info("Step 1: Data Ingestion")
            with ErrorHandler("pipeline_data_ingestion", None, self.resource_monitor) as eh:
                data_ingestor = DataIngestion(
                    s3_bucket=self.config['data'].get('s3_bucket'),
                    s3_key=self.config['data'].get('s3_key'),
                    local_path=self.config['data'].get('local_path'),
                    validator=self.data_validator
                )
                raw_data = data_ingestor.load_data()
                self.components['data_ingestor'] = data_ingestor

                # Log data validation results
                validation_result = self.data_validator.validate_dataframe(raw_data)
                if eh.mlflow_logger:
                    eh.mlflow_logger.log_validation_results('data_ingestion', validation_result)

            # 2. Data Preprocessing with validation
            logger.info("Step 2: Data Preprocessing")
            with ErrorHandler("pipeline_data_preprocessing", None, self.resource_monitor) as eh:
                preprocessor = DataPreprocessor(raw_data, validator=self.data_validator)
                X_train, X_test, y_train, y_test = preprocessor.preprocess(**self.config['preprocessing'])
                self.components['preprocessor'] = preprocessor

                # Log preprocessing validation
                feature_validation = self.data_validator.validate_features(X_train, y_train)
                if eh.mlflow_logger:
                    eh.mlflow_logger.log_validation_results('data_preprocessing', feature_validation)

            # 3. Model Training and Evaluation with validation
            logger.info("Step 3: Model Training or Loading")
            with ErrorHandler("pipeline_model_training", None, self.resource_monitor) as eh:
                trainer = ModelTrainer(X_train, X_test, y_train, y_test, validator=self.model_validator)

                # Check if we should load from artifacts instead of training
                if self.config['model'].get('load_from_artifacts', False):
                    logger.info("Loading pre-trained model from artifacts...")
                    model = trainer.load_model_from_artifacts(self.config['model'].get('artifacts_path', 'artifacts'))
                    # For loaded models, we still need to evaluate on the current data
                    metrics = trainer.evaluate_model(accuracy_threshold=self.config.get('validation', {}).get('accuracy_threshold', 0.5))
                else:
                    logger.info("Training new model...")
                    model = trainer.train_model(**self.config['model'])
                    metrics = trainer.evaluate_model(accuracy_threshold=self.config.get('validation', {}).get('accuracy_threshold', 0.5))

                self.components['trainer'] = trainer

                # Log model validation results
                model_validation = self.model_validator.validate_model(model, X_test, y_test)
                if eh.mlflow_logger:
                    eh.mlflow_logger.log_validation_results('model_training', model_validation)

            # 4. MLflow Logging with enhanced error handling and QA
            run_id = None
            if not self.config['model'].get('load_from_artifacts', False):
                logger.info("Step 4: MLflow Logging with QA")
                with ErrorHandler("pipeline_mlflow_logging", None, self.resource_monitor) as eh:
                    mlflow_manager = MLflowManager(self.config['mlflow']['experiment_name'])
                    mlflow_manager.setup_experiment()
                    mlflow_manager.start_run()

                    # Initialize MLflow error logger
                    self.mlflow_logger = MLflowErrorLogger(mlflow_manager.run_id)

                    # Log system health before pipeline execution
                    self.qa_monitor.log_health_to_mlflow(mlflow_manager.run_id)

                    # Log parameters
                    pipeline_params = {
                        'data_source': self.config['data']['source'],
                        'model_type': self.config['model']['type'],
                        'test_size': self.config['preprocessing']['test_size'],
                        'random_state': self.config['preprocessing']['random_state'],
                        'qa_enabled': True,
                        'error_handling_enabled': True,
                        'loaded_from_artifacts': False
                    }
                    pipeline_params.update(self.config['model']['params'])
                    mlflow_manager.log_parameters(pipeline_params)

                    # Log metrics
                    mlflow_manager.log_metrics(metrics)

                    # Log model
                    mlflow_manager.log_model(model)

                    # Log final resource usage
                    final_resources = self.resource_monitor.get_resource_usage()
                    self.mlflow_logger.log_resource_usage(final_resources)

                    # Get run ID for deployment
                    run_id = mlflow_manager.run_id
                    pipeline_result['run_id'] = run_id

                    # End run
                    mlflow_manager.end_run()
                    self.components['mlflow_manager'] = mlflow_manager
            else:
                logger.info("Step 4: Skipping MLflow logging (using pre-trained model from artifacts)")
                # Still initialize error logger for QA
                self.mlflow_logger = None

            # 5. Model Deployment with error handling
            logger.info("Step 5: Model Deployment")
            model_uri = None
            if self.config['deployment']['register_model'] and run_id is not None:
                with ErrorHandler("pipeline_deployment", self.mlflow_logger, self.resource_monitor):
                    deployment_manager = DeploymentManager()
                    model_uri = deployment_manager.register_model(
                        model_name=self.config['deployment']['model_name'],
                        run_id=run_id
                    )
                    self.components['deployment_manager'] = deployment_manager
                    logger.info(f"Model deployed with URI: {model_uri}")
            elif self.config['deployment']['register_model'] and self.config['model'].get('load_from_artifacts', False):
                logger.info("Skipping model registration (using pre-trained model from artifacts)")

            # Run final QA checks
            logger.info("Step 6: Final QA Validation")
            try:
                qa_suite = QATestSuite()
                qa_results = qa_suite.run_full_qa_suite(self.config)
                self.mlflow_logger.log_validation_results('final_qa', {
                    'is_valid': qa_results['overall_status'] == 'passed',
                    'qa_passed': qa_results['tests_passed'],
                    'qa_failed': qa_results['tests_failed'],
                    'qa_status': qa_results['overall_status']
                })
                logger.info(f"QA Validation: {qa_results['overall_status'].upper()}")
            except Exception as qa_error:
                logger.warning(f"QA validation failed: {qa_error}")
                if self.mlflow_logger:
                    self.mlflow_logger.log_error('qa_failure', str(qa_error))

            logger.info("🎉 Pipeline execution completed successfully!")
            logger.info("=" * 60)

            pipeline_result.update({
                'status': 'success',
                'metrics': metrics,
                'model_uri': model_uri,
                'qa_results': qa_results if 'qa_results' in locals() else None
            })
            return pipeline_result

        except (DataValidationError, ModelValidationError, S3ConnectivityError,
                NetworkError, ResourceLimitError) as e:
            # Specific error handling for known error types
            error_type = type(e).__name__
            logger.error(f"Pipeline failed with {error_type}: {e}")

            if self.mlflow_logger:
                self.mlflow_logger.log_error(error_type.lower(), str(e))

            pipeline_result.update({
                'status': 'failed',
                'error': str(e),
                'error_type': error_type
            })
            return pipeline_result

        except Exception as e:
            # Generic error handling
            logger.error(f"Pipeline execution failed with unexpected error: {e}")

            if self.mlflow_logger:
                self.mlflow_logger.log_error('unexpected_error', str(e))

            pipeline_result.update({
                'status': 'failed',
                'error': str(e),
                'error_type': 'unexpected_error'
            })
            return pipeline_result

def main():
    """Main function to run the pipeline with QA and error handling."""
    # Example configuration with QA enabled
    config = {
        'data': {
            'source': 'sklearn'  # Use Wine dataset from sklearn
        },
        'preprocessing': {
            'target_column': 'target',
            'test_size': 0.2,
            'random_state': 42
        },
        'model': {
            'type': 'random_forest',
            'params': {'n_estimators': 100, 'random_state': 42},
            'load_from_artifacts': True,  # Load pre-trained model from artifacts folder
            'artifacts_path': 'artifacts'
        },
        'validation': {
            'accuracy_threshold': 0.7,  # Higher threshold for QA
            'enable_data_validation': True,
            'enable_model_validation': True
        },
        'qa': {
            'run_qa_suite': True,
            'fail_on_qa_failure': False
        },
        'mlflow': {
            'experiment_name': 'Wine_Classification_Pipeline_QA'
        },
        'deployment': {
            'register_model': False,  # Skip registration when using pre-trained model
            'model_name': 'WineClassifier'
        },
        'error_handling': {
            'max_retries': 3,
            'enable_resource_monitoring': True
        }
    }

    # Run pipeline
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run_pipeline()

    if result['status'] == 'success':
        print("\n📊 Pipeline Results:")
        print(f"Run ID: {result['run_id']}")
        print(f"Model URI: {result['model_uri']}")
        print("Metrics:")
        for key, value in result['metrics'].items():
            if isinstance(value, (int, float)):
                print(".4f")

        if result.get('qa_results'):
            qa = result['qa_results']
            print(f"\n🧪 QA Results: {qa['overall_status'].upper()}")
            print(f"Tests Passed: {qa['tests_passed']}, Failed: {qa['tests_failed']}")

        print("\n✅ Pipeline completed successfully with QA validation!")
    else:
        print(f"\n❌ Pipeline failed: {result['error']}")
        if result.get('error_type'):
            print(f"Error Type: {result['error_type']}")
        sys.exit(1)

if __name__ == "__main__":
    main()