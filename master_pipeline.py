"""
Master Pipeline Orchestrator for Student C
Complete workflow integration from data ingestion to deployment with
comprehensive error handling, logging, and testing.
"""

import os
import sys
import logging
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from end_to_end_pipeline import PipelineOrchestrator
from deployment.serving_test import ModelServingTester
from deployment.ab_testing import ABTestingFramework
from deployment.s3_setup import setup_s3_artifact_store, test_s3_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MasterPipelineOrchestrator:
    """Master orchestrator for the complete ML workflow."""

    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.pipeline_results = {}
        self.setup_complete = False

    def _load_config(self, config_path=None):
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            config = self._default_config()
            logger.info("Using default configuration")

        return config

    def _default_config(self):
        """Default master pipeline configuration."""
        return {
            'environment': {
                'setup_s3': True,
                's3_bucket': 'mlflow-artifacts-bucket',
                's3_region': 'us-east-1',
                'mlflow_tracking_uri': 'sqlite:///mlruns/mlflow.db'
            },
            'pipeline': {
                'data_source': 'sklearn',  # 'sklearn', 's3', 'local'
                's3_bucket': None,
                's3_key': None,
                'local_path': None,
                'experiment_name': 'Wine_Classification_Master_Pipeline',
                'model_name': 'WineClassifier',
                'register_model': True
            },
            'testing': {
                'run_unit_tests': True,
                'run_integration_tests': True,
                'run_serving_tests': True,
                'run_ab_tests': False,  # Requires multiple model versions
                'performance_test': True
            },
            'deployment': {
                'start_serving': True,
                'serving_port': 5001,
                'test_duration_minutes': 1
            }
        }

    def setup_environment(self):
        """Set up the ML environment (S3, MLflow, etc.)."""
        try:
            logger.info("🔧 Setting up ML environment...")
            start_time = time.time()

            # Configure MLflow tracking URI
            import mlflow
            mlflow.set_tracking_uri(self.config['environment']['mlflow_tracking_uri'])
            logger.info(f"MLflow tracking URI set to: {self.config['environment']['mlflow_tracking_uri']}")

            # Set up S3 artifact storage if configured
            if self.config['environment']['setup_s3']:
                logger.info("Setting up S3 artifact storage...")
                setup_s3_artifact_store(
                    bucket_name=self.config['environment']['s3_bucket'],
                    region=self.config['environment']['s3_region']
                )

                # Test S3 connection
                if test_s3_connection(
                    bucket_name=self.config['environment']['s3_bucket'],
                    region=self.config['environment']['s3_region']
                ):
                    logger.info("✅ S3 setup completed successfully")
                else:
                    logger.warning("⚠️  S3 setup failed, falling back to local storage")

            setup_time = time.time() - start_time
            logger.info(".2f")
            self.setup_complete = True
            return True

        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False

    def run_pipeline(self):
        """Execute the main ML pipeline."""
        try:
            logger.info("🔬 Running ML Pipeline...")
            start_time = time.time()

            # Configure pipeline
            pipeline_config = {
                'data': {
                    'source': self.config['pipeline']['data_source'],
                    's3_bucket': self.config['pipeline']['s3_bucket'],
                    's3_key': self.config['pipeline']['s3_key'],
                    'local_path': self.config['pipeline']['local_path']
                },
                'preprocessing': {
                    'target_column': 'target',
                    'test_size': 0.2,
                    'random_state': 42
                },
                'model': {
                    'type': 'random_forest',
                    'params': {'n_estimators': 100, 'random_state': 42}
                },
                'mlflow': {
                    'experiment_name': self.config['pipeline']['experiment_name']
                },
                'deployment': {
                    'register_model': self.config['pipeline']['register_model'],
                    'model_name': self.config['pipeline']['model_name']
                }
            }

            # Execute pipeline
            orchestrator = PipelineOrchestrator(pipeline_config)
            result = orchestrator.run_pipeline()

            pipeline_time = time.time() - start_time
            logger.info(".2f")

            if result['status'] == 'success':
                logger.info("✅ Pipeline execution completed successfully")
                self.pipeline_results = result
                return True
            else:
                logger.error(f"❌ Pipeline execution failed: {result['error']}")
                return False

        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            return False

    def run_tests(self):
        """Execute comprehensive testing suite."""
        try:
            logger.info("🧪 Running Testing Suite...")
            start_time = time.time()

            test_results = {
                'unit_tests': False,
                'integration_tests': False,
                'serving_tests': False,
                'ab_tests': False,
                'performance_tests': False
            }

            # Run unit tests
            if self.config['testing']['run_unit_tests']:
                logger.info("Running unit tests...")
                test_results['unit_tests'] = self._run_unit_tests()

            # Run integration tests
            if self.config['testing']['run_integration_tests']:
                logger.info("Running integration tests...")
                test_results['integration_tests'] = self._run_integration_tests()

            # Run serving tests
            if self.config['testing']['run_serving_tests'] and self.pipeline_results.get('model_uri'):
                logger.info("Running serving performance tests...")
                test_results['serving_tests'] = self._run_serving_tests()

            # Run A/B tests (if multiple model versions available)
            if self.config['testing']['run_ab_tests']:
                logger.info("Running A/B tests...")
                test_results['ab_tests'] = self._run_ab_tests()

            # Run performance tests
            if self.config['testing']['performance_test']:
                logger.info("Running performance tests...")
                test_results['performance_tests'] = self._run_performance_tests()

            testing_time = time.time() - start_time
            logger.info(".2f")

            # Summarize test results
            passed_tests = sum(test_results.values())
            total_tests = len(test_results)
            logger.info(f"Test Results: {passed_tests}/{total_tests} test suites passed")

            return passed_tests == total_tests

        except Exception as e:
            logger.error(f"Testing execution error: {e}")
            return False

    def _run_unit_tests(self):
        """Run unit tests using subprocess."""
        try:
            # Run the automated_tests.py script
            result = subprocess.run([
                sys.executable, 'automated_tests.py', '--unit-only'
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info("✅ Unit tests passed")
                return True
            else:
                logger.error(f"❌ Unit tests failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Unit tests timed out")
            return False
        except Exception as e:
            logger.error(f"Error running unit tests: {e}")
            return False

    def _run_integration_tests(self):
        """Run integration tests."""
        try:
            # Import and run integration test function
            from automated_tests import run_integration_tests
            return run_integration_tests()

        except Exception as e:
            logger.error(f"Error running integration tests: {e}")
            return False

    def _run_serving_tests(self):
        """Run model serving performance tests."""
        try:
            model_uri = self.pipeline_results.get('model_uri', f"models:/{self.config['pipeline']['model_name']}/Production")

            tester = ModelServingTester(model_uri, port=self.config['deployment']['serving_port'])

            # Start server
            if not tester.start_server():
                logger.error("Failed to start model server for testing")
                return False

            try:
                # Run performance test
                metrics, results = tester.run_performance_test(
                    n_requests=50,
                    n_concurrent=5
                )

                # Log metrics to MLflow
                tester.log_metrics_to_mlflow(metrics, "Serving Performance Test")

                logger.info("✅ Serving tests completed successfully")
                logger.info(".2f")
                return True

            finally:
                tester.stop_server()

        except Exception as e:
            logger.error(f"Error running serving tests: {e}")
            return False

    def _run_ab_tests(self):
        """Run A/B testing (requires multiple model versions)."""
        try:
            # This would require multiple model versions to be available
            # For now, skip if not enough versions
            import mlflow
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            versions = client.search_model_versions(f"name='{self.config['pipeline']['model_name']}'")

            if len(versions) < 2:
                logger.info("⚠️  Not enough model versions for A/B testing (need at least 2)")
                return False

            # Use latest two versions
            model_a_uri = f"models:/{self.config['pipeline']['model_name']}/{versions[0].version}"
            model_b_uri = f"models:/{self.config['pipeline']['model_name']}/{versions[1].version}"

            ab_tester = ABTestingFramework(model_a_uri, model_b_uri)

            if not ab_tester.start_servers():
                logger.error("Failed to start A/B testing servers")
                return False

            try:
                # Run A/B test
                results = ab_tester.run_ab_test(
                    n_requests=100,
                    n_concurrent=5,
                    test_duration_minutes=1
                )

                # Analyze results
                analysis = ab_tester.analyze_results(results)
                ab_tester.log_results_to_mlflow(analysis)

                logger.info("✅ A/B tests completed successfully")
                return True

            finally:
                ab_tester.stop_servers()

        except Exception as e:
            logger.error(f"Error running A/B tests: {e}")
            return False

    def _run_performance_tests(self):
        """Run performance benchmark tests."""
        try:
            # Run benchmark script
            result = subprocess.run([
                sys.executable, 'benchmark_mlflow.py'
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info("✅ Performance tests completed")
                return True
            else:
                logger.error(f"❌ Performance tests failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Performance tests timed out")
            return False
        except Exception as e:
            logger.error(f"Error running performance tests: {e}")
            return False

    def deploy_and_serve(self):
        """Deploy model and start serving."""
        try:
            logger.info("🚀 Starting Model Deployment and Serving...")
            start_time = time.time()

            if not self.config['deployment']['start_serving']:
                logger.info("Model serving disabled in configuration")
                return True

            model_uri = self.pipeline_results.get('model_uri', f"models:/{self.config['pipeline']['model_name']}/Production")

            # Start model server
            logger.info(f"Starting model server on port {self.config['deployment']['serving_port']}...")
            tester = ModelServingTester(model_uri, port=self.config['deployment']['serving_port'])

            if tester.start_server():
                logger.info("✅ Model server started successfully")

                # Test a few predictions
                logger.info("Testing model predictions...")
                test_data = [13.5, 2.3, 2.3, 17.0, 105.0, 2.1, 2.8, 0.27, 0.22, 5.5, 0.48, 0.63, 3.1]  # Sample wine data
                result = tester.test_single_prediction(test_data)

                if result['success']:
                    logger.info("✅ Model prediction test successful")
                    logger.info(f"Prediction result: {result['prediction']}")
                else:
                    logger.warning(f"⚠️  Model prediction test failed: {result['error']}")

                deployment_time = time.time() - start_time
                logger.info(".2f")

                # Keep server running for the specified duration
                if self.config['deployment']['test_duration_minutes'] > 0:
                    logger.info(f"Model server will run for {self.config['deployment']['test_duration_minutes']} minutes...")
                    logger.info(f"Server URL: http://127.0.0.1:{self.config['deployment']['serving_port']}/invocations")

                    # In a real deployment, you would keep the server running
                    # For this demo, we'll stop it after testing
                    time.sleep(5)  # Brief pause to show it's working
                    tester.stop_server()
                    logger.info("Model server stopped (demo mode)")

                return True
            else:
                logger.error("❌ Failed to start model server")
                return False

        except Exception as e:
            logger.error(f"Deployment error: {e}")
            return False

    def generate_report(self):
        """Generate comprehensive execution report."""
        try:
            logger.info("📊 Generating execution report...")

            report = {
                'execution_timestamp': datetime.now().isoformat(),
                'configuration': self.config,
                'pipeline_results': self.pipeline_results,
                'environment_setup': self.setup_complete,
                'status': 'completed' if self.setup_complete and self.pipeline_results.get('status') == 'success' else 'failed'
            }

            # Save report to file
            report_path = f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"✅ Execution report saved to: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None

    def run_complete_workflow(self):
        """Execute the complete workflow from setup to deployment."""
        logger.info("🎯 Starting Complete ML Workflow")
        logger.info("=" * 60)

        workflow_start = time.time()
        success = True

        try:
            # Step 1: Environment Setup
            logger.info("\n📋 Step 1: Environment Setup")
            if not self.setup_environment():
                success = False
                raise Exception("Environment setup failed")

            # Step 2: Pipeline Execution
            logger.info("\n🔬 Step 2: Pipeline Execution")
            if not self.run_pipeline():
                success = False
                raise Exception("Pipeline execution failed")

            # Step 3: Testing
            logger.info("\n🧪 Step 3: Testing Suite")
            if not self.run_tests():
                logger.warning("⚠️  Some tests failed, but continuing with deployment")

            # Step 4: Deployment and Serving
            logger.info("\n🚀 Step 4: Deployment and Serving")
            if not self.deploy_and_serve():
                success = False
                raise Exception("Deployment failed")

            # Step 5: Generate Report
            logger.info("\n📊 Step 5: Generate Report")
            report_path = self.generate_report()

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            success = False

        finally:
            workflow_time = time.time() - workflow_start
            logger.info("\n" + "=" * 60)
            if success:
                logger.info("🎉 Complete workflow executed successfully!")
                logger.info(".2f")
                if 'report_path' in locals():
                    logger.info(f"📄 Detailed report: {report_path}")
            else:
                logger.error("❌ Workflow execution failed!")
                logger.info(".2f")

            return success

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Master ML Pipeline Orchestrator')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--skip-tests', action='store_true', help='Skip testing phase')
    parser.add_argument('--skip-serving', action='store_true', help='Skip model serving')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize orchestrator
    orchestrator = MasterPipelineOrchestrator(args.config)

    # Override config based on arguments
    if args.skip_tests:
        orchestrator.config['testing'] = {k: False for k in orchestrator.config['testing']}

    if args.skip_serving:
        orchestrator.config['deployment']['start_serving'] = False

    # Run complete workflow
    success = orchestrator.run_complete_workflow()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()