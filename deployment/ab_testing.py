"""
A/B Testing Framework for MLflow Models
This script creates a simple A/B testing framework that compares two model versions
by serving both models and routing requests between them, logging comparison results.
"""

import os
import time
import random
import requests
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import subprocess
import signal

class ABTestingFramework:
    def __init__(self, model_a_uri, model_b_uri, port_a=5001, port_b=5002):
        self.model_a_uri = model_a_uri
        self.model_b_uri = model_b_uri
        self.port_a = port_a
        self.port_b = port_b
        self.base_url_a = f"http://127.0.0.1:{port_a}"
        self.base_url_b = f"http://127.0.0.1:{port_b}"
        self.server_a = None
        self.server_b = None
        self.results = []

    def start_servers(self):
        """Start both model servers."""
        print("Starting A/B testing servers...")

        # Start server A
        self.server_a = ModelServer(self.model_a_uri, self.port_a, "Model_A")
        if not self.server_a.start():
            print("✗ Failed to start Model A server")
            return False

        # Start server B
        self.server_b = ModelServer(self.model_b_uri, self.port_b, "Model_B")
        if not self.server_b.start():
            print("✗ Failed to start Model B server")
            self.server_a.stop()
            return False

        print("✓ Both A/B testing servers started successfully")
        return True

    def stop_servers(self):
        """Stop both model servers."""
        if self.server_a:
            self.server_a.stop()
        if self.server_b:
            self.server_b.stop()
        print("✓ A/B testing servers stopped")

    def route_request(self, data, variant=None):
        """Route a request to either model A or B."""
        if variant is None:
            variant = random.choice(['A', 'B'])

        if variant == 'A':
            return self._make_request(self.base_url_a, data, 'A')
        else:
            return self._make_request(self.base_url_b, data, 'B')

    def _make_request(self, base_url, data, variant):
        """Make a request to a specific model server."""
        payload = {
            "dataframe_split": {
                "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
                "data": [data]
            }
        }

        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/invocations",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response_time = time.time() - start_time

            result = {
                "variant": variant,
                "success": response.status_code == 200,
                "response_time": response_time,
                "input_data": data,
                "timestamp": datetime.now()
            }

            if response.status_code == 200:
                result["prediction"] = response.json()
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text}"

            return result

        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return {
                "variant": variant,
                "success": False,
                "response_time": response_time,
                "input_data": data,
                "error": str(e),
                "timestamp": datetime.now()
            }

    def run_ab_test(self, n_requests=100, n_concurrent=5, test_duration_minutes=5):
        """Run A/B test for a specified duration or number of requests."""
        print(f"Running A/B test with {n_requests} requests, {n_concurrent} concurrent threads...")

        # Generate test data
        test_data = self._generate_test_data(n_requests)

        results = []
        start_time = time.time()
        end_time = start_time + (test_duration_minutes * 60)

        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            futures = []

            for data in test_data:
                if time.time() > end_time:
                    break

                future = executor.submit(self.route_request, data)
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                self.results.append(result)

        return results

    def _generate_test_data(self, n_samples):
        """Generate test data for A/B testing."""
        from sklearn.datasets import load_iris
        iris = load_iris()

        # Sample random data points
        np.random.seed(42)
        indices = np.random.choice(len(iris.data), n_samples, replace=True)

        return iris.data[indices].tolist()

    def analyze_results(self, results=None):
        """Analyze A/B test results and return comparison metrics."""
        if results is None:
            results = self.results

        if not results:
            return {"error": "No results to analyze"}

        # Separate results by variant
        results_a = [r for r in results if r['variant'] == 'A']
        results_b = [r for r in results if r['variant'] == 'B']

        # Calculate metrics for each variant
        def calculate_metrics(res_list, variant_name):
            if not res_list:
                return {"count": 0}

            successful = [r for r in res_list if r['success']]
            response_times = [r['response_time'] for r in res_list]

            return {
                "variant": variant_name,
                "count": len(res_list),
                "success_count": len(successful),
                "success_rate": len(successful) / len(res_list) * 100,
                "avg_response_time": np.mean(response_times),
                "min_response_time": np.min(response_times),
                "max_response_time": np.max(response_times),
                "p95_response_time": np.percentile(response_times, 95)
            }

        metrics_a = calculate_metrics(results_a, "A")
        metrics_b = calculate_metrics(results_b, "B")

        # Calculate statistical significance (simple comparison)
        if metrics_a['count'] > 0 and metrics_b['count'] > 0:
            # Response time comparison
            response_times_a = [r['response_time'] for r in results_a]
            response_times_b = [r['response_time'] for r in results_b]

            # Simple t-test approximation for response times
            mean_diff = metrics_a['avg_response_time'] - metrics_b['avg_response_time']
            pooled_std = np.sqrt((np.var(response_times_a) + np.var(response_times_b)) / 2)
            t_stat = mean_diff / (pooled_std / np.sqrt(min(len(response_times_a), len(response_times_b))))

            analysis = {
                "model_a_metrics": metrics_a,
                "model_b_metrics": metrics_b,
                "comparison": {
                    "response_time_difference": mean_diff,
                    "response_time_t_statistic": t_stat,
                    "success_rate_difference": metrics_a['success_rate'] - metrics_b['success_rate'],
                    "recommendation": "A" if metrics_a['avg_response_time'] < metrics_b['avg_response_time'] else "B"
                },
                "total_requests": len(results),
                "test_duration_seconds": (results[-1]['timestamp'] - results[0]['timestamp']).total_seconds()
            }
        else:
            analysis = {
                "model_a_metrics": metrics_a,
                "model_b_metrics": metrics_b,
                "comparison": {"error": "Insufficient data for comparison"},
                "total_requests": len(results)
            }

        return analysis

    def log_results_to_mlflow(self, analysis, experiment_name="A-B Testing Results"):
        """Log A/B test results to MLflow."""
        try:
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log metrics for model A
                if 'model_a_metrics' in analysis:
                    for key, value in analysis['model_a_metrics'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"model_a_{key}", value)

                # Log metrics for model B
                if 'model_b_metrics' in analysis:
                    for key, value in analysis['model_b_metrics'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"model_b_{key}", value)

                # Log comparison metrics
                if 'comparison' in analysis:
                    for key, value in analysis['comparison'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"comparison_{key}", value)

                # Log parameters
                mlflow.log_param("model_a_uri", self.model_a_uri)
                mlflow.log_param("model_b_uri", self.model_b_uri)
                mlflow.log_param("total_requests", analysis.get('total_requests', 0))

                print(f"✓ A/B test results logged to MLflow experiment: {experiment_name}")

        except Exception as e:
            print(f"⚠️  Error logging to MLflow: {e}")

class ModelServer:
    """Helper class to manage individual model servers."""
    def __init__(self, model_uri, port, name):
        self.model_uri = model_uri
        self.port = port
        self.name = name
        self.process = None

    def start(self):
        """Start the model server."""
        print(f"Starting {self.name} server on port {self.port}...")

        cmd = [
            "mlflow", "models", "serve",
            "-m", self.model_uri,
            "-p", str(self.port),
            "--host", "127.0.0.1",
            "--no-conda"
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            # Wait for server to start
            time.sleep(5)

            if self.process.poll() is None:
                print(f"✓ {self.name} server started on port {self.port}")
                return True
            else:
                stdout, stderr = self.process.communicate()
                print(f"✗ Failed to start {self.name} server: {stderr.decode()}")
                return False

        except Exception as e:
            print(f"✗ Error starting {self.name} server: {e}")
            return False

    def stop(self):
        """Stop the model server."""
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
                print(f"✓ {self.name} server stopped")
            except Exception as e:
                print(f"⚠️  Error stopping {self.name} server: {e}")
                try:
                    self.process.kill()
                except:
                    pass

def main():
    """Main function to run A/B testing."""
    print("🚀 Starting A/B Testing Framework")
    print("=" * 50)

    # Example model URIs - replace with your actual model versions
    MODEL_A_URI = "models:/IrisClassifier/1"  # Version 1
    MODEL_B_URI = "models:/IrisClassifier/2"  # Version 2

    # You can get model versions from MLflow registry
    client = MlflowClient()
    try:
        # Try to get the latest two versions
        versions = client.search_model_versions("name='IrisClassifier'")
        if len(versions) >= 2:
            MODEL_A_URI = f"models:/IrisClassifier/{versions[0].version}"
            MODEL_B_URI = f"models:/IrisClassifier/{versions[1].version}"
            print(f"Using latest versions: A={versions[0].version}, B={versions[1].version}")
        else:
            print("⚠️  Not enough model versions found. Using default URIs.")
    except Exception as e:
        print(f"⚠️  Error getting model versions: {e}. Using default URIs.")

    ab_tester = ABTestingFramework(MODEL_A_URI, MODEL_B_URI)

    try:
        # Start servers
        if not ab_tester.start_servers():
            print("✗ Failed to start A/B testing servers")
            return

        # Wait for servers to be ready
        time.sleep(3)

        # Run A/B test
        print("\nRunning A/B test...")
        results = ab_tester.run_ab_test(n_requests=200, n_concurrent=10, test_duration_minutes=2)

        # Analyze results
        analysis = ab_tester.analyze_results(results)

        # Display results
        print("\n📊 A/B Testing Results:")
        print("-" * 40)

        if 'model_a_metrics' in analysis:
            a_metrics = analysis['model_a_metrics']
            print("Model A:")
            print(f"  Requests: {a_metrics['count']}")
            print(f"  Success Rate: {a_metrics['success_rate']:.1f}%")
            print(".3f")

        if 'model_b_metrics' in analysis:
            b_metrics = analysis['model_b_metrics']
            print("Model B:")
            print(f"  Requests: {b_metrics['count']}")
            print(f"  Success Rate: {b_metrics['success_rate']:.1f}%")
            print(".3f")

        if 'comparison' in analysis and 'recommendation' in analysis['comparison']:
            comp = analysis['comparison']
            print("Comparison:")
            print(".3f")
            print(f"  Recommended Model: {comp['recommendation']}")

        # Log to MLflow
        ab_tester.log_results_to_mlflow(analysis)

        print("\n🎉 A/B testing completed!")

    except Exception as e:
        print(f"✗ Error during A/B testing: {e}")
    finally:
        ab_tester.stop_servers()

if __name__ == "__main__":
    main()