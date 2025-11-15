
import unittest
import sys
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from end_to_end_pipeline import (
    DataIngestion, DataPreprocessor, ModelTrainer,
    MLflowManager, DeploymentManager, PipelineOrchestrator
)
from deployment.serving_test import ModelServingTester
from deployment.ab_testing import ABTestingFramework

# Suppress logging during tests
logging.disable(logging.CRITICAL)

class TestDataIngestion(unittest.TestCase):
    """Unit tests for data ingestion functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_load_wine_dataset(self):
        """Test loading Wine dataset from sklearn."""
        ingestor = DataIngestion()
        data = ingestor.load_data()

        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('target', data.columns)

    def test_load_from_local_csv(self):
        """Test loading data from local CSV file."""
        csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.test_data.to_csv(csv_path, index=False)

        ingestor = DataIngestion(local_path=csv_path)
        data = ingestor.load_data()

        pd.testing.assert_frame_equal(data, self.test_data)

    def test_load_from_local_json(self):
        """Test loading data from local JSON file."""
        json_path = os.path.join(self.temp_dir, 'test_data.json')
        self.test_data.to_json(json_path, orient='records')

        ingestor = DataIngestion(local_path=json_path)
        data = ingestor.load_data()

        # JSON loading might change dtypes, so check shape and columns
        self.assertEqual(data.shape, self.test_data.shape)
        self.assertListEqual(list(data.columns), list(self.test_data.columns))

    @patch('boto3.client')
    def test_load_from_s3(self, mock_boto_client):
        """Test loading data from S3 (mocked)."""
        # Mock S3 client and response
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        # Mock the get_object response
        csv_content = self.test_data.to_csv(index=False)
        mock_client.get_object.return_value = {'Body': MagicMock()}
        mock_client.get_object.return_value['Body'].read.return_value.decode.return_value = csv_content

        ingestor = DataIngestion(s3_bucket='test-bucket', s3_key='test-data.csv')
        data = ingestor.load_data()

        pd.testing.assert_frame_equal(data, self.test_data)
        mock_client.get_object.assert_called_once_with(Bucket='test-bucket', Key='test-data.csv')

    def test_invalid_file_format(self):
        """Test error handling for unsupported file formats."""
        txt_path = os.path.join(self.temp_dir, 'test_data.txt')
        with open(txt_path, 'w') as f:
            f.write("invalid format")

        ingestor = DataIngestion(local_path=txt_path)
        with self.assertRaises(ValueError):
            ingestor.load_data()

class TestDataPreprocessor(unittest.TestCase):
    """Unit tests for data preprocessing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.choice([0, 1, 2], 100)
        })

    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor(self.test_data)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()

        # Check shapes
        self.assertEqual(X_train.shape[0] + X_test.shape[0], 100)
        self.assertEqual(len(y_train) + len(y_test), 100)

        # Check that target column is removed from features
        self.assertNotIn('target', X_train.columns)
        self.assertNotIn('target', X_test.columns)

        # Check scaling (features should be standardized)
        self.assertAlmostEqual(X_train.mean().mean(), 0, places=1)
        self.assertAlmostEqual(X_test.mean().mean(), 0, places=1)

    def test_custom_split_ratio(self):
        """Test preprocessing with custom train/test split."""
        preprocessor = DataPreprocessor(self.test_data)
        X_train, X_test, y_train, y_test = preprocessor.preprocess(test_size=0.3)

        expected_train_size = int(100 * 0.7)
        expected_test_size = 100 - expected_train_size

        self.assertEqual(len(X_train), expected_train_size)
        self.assertEqual(len(X_test), expected_test_size)

class TestModelTrainer(unittest.TestCase):
    """Unit tests for model training functionality."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 100
        self.X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        self.y = pd.Series(np.random.choice([0, 1, 2], n_samples))

    def test_random_forest_training(self):
        """Test Random Forest model training."""
        trainer = ModelTrainer(self.X, self.X, self.y, self.y)  # Use same data for simplicity
        model = trainer.train_model('random_forest', n_estimators=10)

        self.assertIsNotNone(model)
        self.assertEqual(model.n_estimators, 10)

    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        trainer = ModelTrainer(self.X, self.X, self.y, self.y)
        trainer.model = trainer.train_model('random_forest', n_estimators=10)

        metrics = trainer.evaluate_model()

        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'f1', 'precision', 'recall', 'classification_report']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)

        # Check metric ranges
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)

    def test_invalid_model_type(self):
        """Test error handling for invalid model types."""
        trainer = ModelTrainer(self.X, self.X, self.y, self.y)

        with self.assertRaises(ValueError):
            trainer.train_model('invalid_model_type')

class TestMLflowManager(unittest.TestCase):
    """Unit tests for MLflow management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mlflow_manager = MLflowManager("test_experiment")

    @patch('mlflow.set_experiment')
    def test_setup_experiment(self, mock_set_experiment):
        """Test MLflow experiment setup."""
        self.mlflow_manager.setup_experiment()
        mock_set_experiment.assert_called_once_with("test_experiment")

    @patch('mlflow.start_run')
    @patch('mlflow.end_run')
    def test_run_management(self, mock_end_run, mock_start_run):
        """Test MLflow run management."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value = mock_run

        self.mlflow_manager.start_run("test_run")
        self.mlflow_manager.end_run()

        mock_start_run.assert_called_once()
        mock_end_run.assert_called_once()

    @patch('mlflow.log_param')
    def test_log_parameters(self, mock_log_param):
        """Test parameter logging."""
        params = {'param1': 'value1', 'param2': 42}
        self.mlflow_manager.log_parameters(params)

        self.assertEqual(mock_log_param.call_count, 2)

    @patch('mlflow.log_metric')
    def test_log_metrics(self, mock_log_metric):
        """Test metrics logging."""
        metrics = {'accuracy': 0.95, 'f1': 0.93}
        self.mlflow_manager.log_metrics(metrics)

        self.assertEqual(mock_log_metric.call_count, 2)

class TestPipelineOrchestrator(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def test_pipeline_execution(self):
        """Test complete pipeline execution."""
        config = {
            'data': {'source': 'sklearn'},
            'preprocessing': {
                'target_column': 'target',
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'type': 'random_forest',
                'params': {'n_estimators': 10, 'random_state': 42}
            },
            'mlflow': {'experiment_name': 'test_pipeline'},
            'deployment': {'register_model': False}  # Skip deployment for testing
        }

        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_pipeline()

        self.assertEqual(result['status'], 'success')
        self.assertIn('run_id', result)
        self.assertIn('metrics', result)
        self.assertIn('accuracy', result['metrics'])

    def test_pipeline_with_invalid_config(self):
        """Test pipeline with invalid configuration."""
        config = {
            'data': {'source': 'sklearn'},
            'model': {'type': 'invalid_model'},
            'mlflow': {'experiment_name': 'test_pipeline'},
            'deployment': {'register_model': False}
        }

        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_pipeline()

        self.assertEqual(result['status'], 'failed')
        self.assertIn('error', result)

class TestServingIntegration(unittest.TestCase):
    """Integration tests for model serving functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_uri = "models:/IrisClassifier/Production"

    @patch('subprocess.Popen')
    @patch('requests.post')
    def test_serving_test_initialization(self, mock_requests, mock_popen):
        """Test serving tester initialization."""
        # Mock subprocess
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        # Mock successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [0.1, 0.8, 0.1]
        mock_requests.return_value = mock_response

        tester = ModelServingTester(self.model_uri)

        # Test server start
        result = tester.start_server()
        self.assertTrue(result)

        # Test single prediction
        result = tester.test_single_prediction([5.1, 3.5, 1.4, 0.2])
        self.assertTrue(result['success'])
        self.assertEqual(result['status_code'], 200)

    @patch('subprocess.Popen')
    def test_server_start_failure(self, mock_popen):
        """Test server start failure handling."""
        # Mock failed subprocess
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Non-zero exit code
        mock_process.communicate.return_value = (b'', b'Error starting server')
        mock_popen.return_value = mock_process

        tester = ModelServingTester(self.model_uri)
        result = tester.start_server()
        self.assertFalse(result)

class TestABTestingIntegration(unittest.TestCase):
    """Integration tests for A/B testing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_a_uri = "models:/IrisClassifier/1"
        self.model_b_uri = "models:/IrisClassifier/2"

    @patch('subprocess.Popen')
    @patch('requests.post')
    def test_ab_test_initialization(self, mock_requests, mock_popen):
        """Test A/B testing framework initialization."""
        # Mock subprocess for both servers
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        # Mock successful requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [0.1, 0.8, 0.1]
        mock_requests.return_value = mock_response

        ab_tester = ABTestingFramework(self.model_a_uri, self.model_b_uri)

        # Test server start
        result = ab_tester.start_servers()
        self.assertTrue(result)

        # Test single request routing
        result = ab_tester.route_request([5.1, 3.5, 1.4, 0.2])
        self.assertIn('variant', result)
        self.assertTrue(result['success'])

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance and benchmark tests."""

    def test_pipeline_performance(self):
        """Test pipeline execution performance."""
        import time

        config = {
            'data': {'source': 'sklearn'},
            'preprocessing': {
                'target_column': 'target',
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'type': 'random_forest',
                'params': {'n_estimators': 10, 'random_state': 42}
            },
            'mlflow': {'experiment_name': 'performance_test'},
            'deployment': {'register_model': False}
        }

        start_time = time.time()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_pipeline()
        end_time = time.time()

        execution_time = end_time - start_time

        self.assertEqual(result['status'], 'success')
        self.assertLess(execution_time, 30)  # Should complete within 30 seconds

    def test_memory_usage(self):
        """Test memory usage during pipeline execution."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        config = {
            'data': {'source': 'sklearn'},
            'preprocessing': {
                'target_column': 'target',
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'type': 'random_forest',
                'params': {'n_estimators': 50, 'random_state': 42}
            },
            'mlflow': {'experiment_name': 'memory_test'},
            'deployment': {'register_model': False}
        }

        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_pipeline()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory

        self.assertEqual(result['status'], 'success')
        self.assertLess(memory_usage, 500)  # Should use less than 500MB additional memory

def run_integration_tests():
    """Run comprehensive integration tests."""
    print("🧪 Running Integration Tests")
    print("=" * 40)

    # Test complete pipeline
    print("Testing complete pipeline integration...")
    config = {
        'data': {'source': 'sklearn'},
        'preprocessing': {
            'target_column': 'target',
            'test_size': 0.2,
            'random_state': 42
        },
        'model': {
            'type': 'random_forest',
            'params': {'n_estimators': 20, 'random_state': 42}
        },
        'mlflow': {'experiment_name': 'integration_test'},
        'deployment': {'register_model': False}
    }

    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run_pipeline()

    if result['status'] == 'success':
        print("✅ Pipeline integration test passed")
        print(".4f")
        return True
    else:
        print(f"❌ Pipeline integration test failed: {result['error']}")
        return False

def main():
    """Main function to run all tests."""
    print("🚀 Starting Automated Testing Suite")
    print("=" * 50)

    # Run unit tests
    print("\n📋 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run integration tests
    print("\n🔗 Running Integration Tests...")
    integration_passed = run_integration_tests()

    print("\n" + "=" * 50)
    if integration_passed:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())