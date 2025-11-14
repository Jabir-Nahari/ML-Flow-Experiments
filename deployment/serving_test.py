"""
Live Model Serving Testing
This script starts the MLflow model server locally and tests it with API calls,
logging serving performance metrics (response time, throughput).
"""

import os
import time
import requests
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlflow
import subprocess
import signal
import sys
from datetime import datetime

class ModelServingTester:
    def __init__(self, model_uri="models:/IrisClassifier/Production", port=5001):
        self.model_uri = model_uri
        self.port = port
        self.server_process = None
        self.base_url = f"http://127.0.0.1:{port}"

    def start_server(self):
        """Start the MLflow model server in the background."""
        print(f"Starting MLflow model server on port {self.port}...")

        # Command to start the server
        cmd = [
            "mlflow", "models", "serve",
            "-m", self.model_uri,
            "-p", str(self.port),
            "--host", "127.0.0.1",
            "--no-conda"
        ]

        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )

            # Wait for server to start
            time.sleep(5)

            # Check if server is running
            if self.server_process.poll() is None:
                print(f"✓ MLflow model server started successfully on port {self.port}")
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                print(f"✗ Failed to start server. STDOUT: {stdout.decode()}, STDERR: {stderr.decode()}")
                return False

        except Exception as e:
            print(f"✗ Error starting server: {e}")
            return False

    def stop_server(self):
        """Stop the MLflow model server."""
        if self.server_process:
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
                print("✓ MLflow model server stopped")
            except Exception as e:
                print(f"⚠️  Error stopping server: {e}")
                try:
                    self.server_process.kill()
                except:
                    pass

    def test_single_prediction(self, sample_data):
        """Test a single prediction and measure response time."""
        payload = {
            "dataframe_split": {
                "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
                "data": [sample_data]
            }
        }

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/invocations",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response_time": response_time,
                    "prediction": result,
                    "status_code": response.status_code
                }
            else:
                return {
                    "success": False,
                    "response_time": response_time,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code
                }

        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "response_time": response_time,
                "error": str(e),
                "status_code": None
            }

    def load_test_data(self, n_samples=100):
        """Generate or load test data for performance testing."""
        # Load iris dataset for testing
        from sklearn.datasets import load_iris
        iris = load_iris()

        # Sample random data points
        np.random.seed(42)
        indices = np.random.choice(len(iris.data), n_samples, replace=True)

        return iris.data[indices].tolist()

    def run_performance_test(self, n_requests=50, n_concurrent=5):
        """Run performance testing with multiple concurrent requests."""
        print(f"Running performance test with {n_requests} requests, {n_concurrent} concurrent...")

        test_data = self.load_test_data(n_requests)
        results = []

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            # Submit all requests
            future_to_data = {
                executor.submit(self.test_single_prediction, data): data
                for data in test_data
            }

            # Collect results as they complete
            for future in as_completed(future_to_data):
                result = future.result()
                results.append(result)

        total_time = time.time() - start_time

        # Calculate metrics
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]

        metrics = {
            "total_requests": len(results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(results) * 100,
            "total_time_seconds": total_time,
            "requests_per_second": len(results) / total_time,
            "avg_response_time": np.mean([r["response_time"] for r in results]),
            "min_response_time": np.min([r["response_time"] for r in results]),
            "max_response_time": np.max([r["response_time"] for r in results]),
            "p95_response_time": np.percentile([r["response_time"] for r in results], 95),
            "p99_response_time": np.percentile([r["response_time"] for r in results], 99)
        }

        return metrics, results

    def log_metrics_to_mlflow(self, metrics, experiment_name="Model Serving Performance"):
        """Log performance metrics to MLflow."""
        try:
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"serving_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Log parameters
                mlflow.log_param("model_uri", self.model_uri)
                mlflow.log_param("port", self.port)

                print(f"✓ Performance metrics logged to MLflow experiment: {experiment_name}")

        except Exception as e:
            print(f"⚠️  Error logging to MLflow: {e}")

def main():
    """Main function to run serving tests."""
    print("🚀 Starting Model Serving Performance Testing")
    print("=" * 50)

    tester = ModelServingTester()

    try:
        # Start the server
        if not tester.start_server():
            print("✗ Failed to start model server. Exiting.")
            return

        # Wait a bit for server to be ready
        time.sleep(2)

        # Test single prediction first
        print("\nTesting single prediction...")
        sample_data = [5.1, 3.5, 1.4, 0.2]  # Sample iris data
        result = tester.test_single_prediction(sample_data)

        if result["success"]:
            print(".3f")
        else:
            print(f"✗ Single prediction failed: {result['error']}")
            return

        # Run performance test
        print("\nRunning performance tests...")
        metrics, results = tester.run_performance_test(n_requests=100, n_concurrent=10)

        # Display results
        print("\n📊 Performance Test Results:")
        print("-" * 30)
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Success Rate: {metrics['success_rate']:.1f}%")
        print(f"Requests/Second: {metrics['requests_per_second']:.2f}")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")

        # Log to MLflow
        tester.log_metrics_to_mlflow(metrics)

        print("\n🎉 Serving performance testing completed!")

    except Exception as e:
        print(f"✗ Error during testing: {e}")
    finally:
        # Always stop the server
        tester.stop_server()

if __name__ == "__main__":
    main()