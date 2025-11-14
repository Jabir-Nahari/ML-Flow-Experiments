"""
Student C - Integrated Demo Coordination
Scripts for coordinating with Students A and B for shared model registry
and combined demonstrations across the team.
"""

import os
import sys
import time
import requests
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Optional
import json

# Import our components
from end_to_end_pipeline import PipelineOrchestrator
from deployment.serving_test import ModelServingTester
from deployment.ab_testing import ABTestingFramework
from error_handling import MLflowErrorLogger
import mlflow
from mlflow.tracking import MlflowClient

class TeamCoordinator:
    """Coordinates demonstrations across Students A, B, and C."""

    def __init__(self, shared_registry_uri: str = "sqlite:///shared_mlflow.db"):
        self.shared_registry_uri = shared_registry_uri
        self.team_members = {
            'A': {'name': 'Student A', 'role': 'Data Engineer', 'port': 5001},
            'B': {'name': 'Student B', 'role': 'MLOps Engineer', 'port': 5002},
            'C': {'name': 'Student C', 'role': 'ML Pipeline Specialist', 'port': 5003}
        }
        self.active_servers = {}
        self.coordination_log = []

    def log_action(self, action: str, details: str = ""):
        """Log coordination actions."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {action}: {details}"
        self.coordination_log.append(log_entry)
        print(f"📋 {log_entry}")

    def setup_shared_registry(self):
        """Set up shared MLflow registry for team collaboration."""
        self.log_action("Setting up shared MLflow registry")

        # Configure shared tracking URI
        mlflow.set_tracking_uri(self.shared_registry_uri)
        self.log_action("MLflow tracking URI configured", self.shared_registry_uri)

        # Create team experiment
        try:
            experiment = mlflow.get_experiment_by_name("Team_Collaboration")
            if experiment is None:
                experiment_id = mlflow.create_experiment("Team_Collaboration")
                self.log_action("Created team experiment", f"ID: {experiment_id}")
            else:
                self.log_action("Using existing team experiment", f"ID: {experiment.id}")
        except Exception as e:
            self.log_action("Error setting up team experiment", str(e))

    def register_team_models(self):
        """Register models from all team members in shared registry."""
        self.log_action("Registering team models in shared registry")

        client = MlflowClient()

        # Register Student C's model
        try:
            self.log_action("Registering Student C's Wine Classifier")

            # Run a quick pipeline to generate a model
            config = {
                'data': {'source': 'sklearn'},
                'preprocessing': {'target_column': 'target', 'test_size': 0.2, 'random_state': 42},
                'model': {'type': 'random_forest', 'params': {'n_estimators': 50, 'random_state': 42}},
                'mlflow': {'experiment_name': 'Team_Collaboration'},
                'deployment': {'register_model': True, 'model_name': 'Team_WineClassifier_C'}
            }

            orchestrator = PipelineOrchestrator(config)
            result = orchestrator.run_pipeline()

            if result['status'] == 'success':
                self.log_action("Student C model registered successfully", result['model_uri'])
            else:
                self.log_action("Student C model registration failed", result['error'])

        except Exception as e:
            self.log_action("Error registering Student C model", str(e))

        # Note: In a real scenario, Students A and B would register their models here
        # For demo purposes, we'll simulate their model registrations
        self._simulate_team_model_registration(client)

    def _simulate_team_model_registration(self, client):
        """Simulate model registrations from Students A and B."""
        # Simulate Student A's model (data preprocessing focus)
        try:
            with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("Team_Collaboration").experiment_id,
                                 run_name="Student_A_Data_Preprocessing_Model") as run:
                mlflow.log_param("student", "A")
                mlflow.log_param("specialization", "Data_Preprocessing")
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_metric("accuracy", 0.89)
                mlflow.log_metric("f1_score", 0.87)

            # Register the model
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, "Team_WineClassifier_A")
            self.log_action("Student A model registered", f"Version {mv.version}")

        except Exception as e:
            self.log_action("Error simulating Student A model", str(e))

        # Simulate Student B's model (deployment focus)
        try:
            with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("Team_Collaboration").experiment_id,
                                 run_name="Student_B_Deployment_Model") as run:
                mlflow.log_param("student", "B")
                mlflow.log_param("specialization", "Model_Deployment")
                mlflow.log_param("model_type", "RandomForest")
                mlflow.log_metric("accuracy", 0.92)
                mlflow.log_metric("f1_score", 0.91)

            # Register the model
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, "Team_WineClassifier_B")
            self.log_action("Student B model registered", f"Version {mv.version}")

        except Exception as e:
            self.log_action("Error simulating Student B model", str(e))

    def start_team_servers(self):
        """Start model servers for all team members."""
        self.log_action("Starting team model servers")

        client = MlflowClient()

        # Get latest versions of team models
        team_models = {}
        for model_name in ["Team_WineClassifier_A", "Team_WineClassifier_B", "Team_WineClassifier_C"]:
            try:
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    latest_version = max(versions, key=lambda v: int(v.version))
                    team_models[model_name] = f"models:/{model_name}/{latest_version.version}"
                    self.log_action(f"Found {model_name}", f"Version {latest_version.version}")
            except Exception as e:
                self.log_action(f"Error getting {model_name}", str(e))

        # Start servers for each team member
        server_configs = [
            ("A", team_models.get("Team_WineClassifier_A"), 5001),
            ("B", team_models.get("Team_WineClassifier_B"), 5002),
            ("C", team_models.get("Team_WineClassifier_C"), 5003)
        ]

        for member_id, model_uri, port in server_configs:
            if model_uri:
                try:
                    tester = ModelServingTester(model_uri, port=port)
                    if tester.start_server():
                        self.active_servers[member_id] = tester
                        self.log_action(f"Started {self.team_members[member_id]['name']} server", f"Port {port}")
                    else:
                        self.log_action(f"Failed to start {self.team_members[member_id]['name']} server")
                except Exception as e:
                    self.log_action(f"Error starting {self.team_members[member_id]['name']} server", str(e))
            else:
                self.log_action(f"No model available for {self.team_members[member_id]['name']}")

    def run_team_ab_test(self, n_requests: int = 100, test_duration_minutes: int = 2):
        """Run A/B test comparing all team models."""
        self.log_action("Starting team A/B testing")

        if len(self.active_servers) < 2:
            self.log_action("Need at least 2 active servers for A/B testing")
            return None

        # Get model URIs for A/B testing
        model_uris = []
        for member_id, server in self.active_servers.items():
            model_uris.append(server.model_uri)

        if len(model_uris) >= 2:
            # Use first two models for A/B test
            ab_tester = ABTestingFramework(model_uris[0], model_uris[1])

            try:
                if ab_tester.start_servers():
                    self.log_action("A/B testing servers started")

                    # Run the test
                    results = ab_tester.run_ab_test(
                        n_requests=n_requests,
                        n_concurrent=5,
                        test_duration_minutes=test_duration_minutes
                    )

                    # Analyze results
                    analysis = ab_tester.analyze_results(results)
                    ab_tester.log_results_to_mlflow(analysis, "Team_AB_Testing_Results")

                    self.log_action("Team A/B testing completed")
                    return analysis
                else:
                    self.log_action("Failed to start A/B testing servers")
            except Exception as e:
                self.log_action("Error during A/B testing", str(e))
            finally:
                ab_tester.stop_servers()
        else:
            self.log_action("Insufficient models for A/B testing")

        return None

    def run_integrated_pipeline_demo(self):
        """Run integrated pipeline demo showing team collaboration."""
        self.log_action("Starting integrated pipeline demo")

        # Step 1: Data preprocessing (Student A style)
        self.log_action("Step 1: Data preprocessing phase")
        print("\n🔄 Student A Contribution: Advanced Data Preprocessing")
        print("   • Robust data validation and cleaning")
        print("   • Feature engineering and scaling")
        print("   • Data quality monitoring")

        # Step 2: Model training and tracking (Student C style)
        self.log_action("Step 2: Model training and MLflow tracking")
        print("\n🧠 Student C Contribution: End-to-End Pipeline")
        print("   • Comprehensive MLflow experiment tracking")
        print("   • Automated model validation and QA")
        print("   • Error handling and recovery")

        # Step 3: Model deployment and serving (Student B style)
        self.log_action("Step 3: Model deployment and serving")
        print("\n🚀 Student B Contribution: Production Deployment")
        print("   • Model serving with performance monitoring")
        print("   • A/B testing framework")
        print("   • Scalable deployment infrastructure")

        # Execute integrated pipeline
        self.log_action("Executing integrated team pipeline")

        config = {
            'data': {'source': 'sklearn'},
            'preprocessing': {'target_column': 'target', 'test_size': 0.2, 'random_state': 42},
            'model': {'type': 'random_forest', 'params': {'n_estimators': 100, 'random_state': 42}},
            'mlflow': {'experiment_name': 'Integrated_Team_Pipeline'},
            'deployment': {'register_model': True, 'model_name': 'Integrated_Team_Model'},
            'qa': {'run_qa_suite': True},
            'error_handling': {'enable_resource_monitoring': True}
        }

        try:
            orchestrator = PipelineOrchestrator(config)
            result = orchestrator.run_pipeline()

            if result['status'] == 'success':
                self.log_action("Integrated pipeline completed successfully")
                return result
            else:
                self.log_action("Integrated pipeline failed", result['error'])
                return result

        except Exception as e:
            self.log_action("Error in integrated pipeline", str(e))
            return {'status': 'failed', 'error': str(e)}

    def demonstrate_team_synergy(self):
        """Demonstrate how team components work together."""
        self.log_action("Demonstrating team synergy")

        synergy_points = [
            {
                'aspect': 'Data Pipeline',
                'student_a': 'Data ingestion and preprocessing',
                'student_b': 'Data validation and quality checks',
                'student_c': 'Pipeline orchestration and monitoring',
                'synergy': 'End-to-end data pipeline with quality assurance'
            },
            {
                'aspect': 'Model Development',
                'student_a': 'Feature engineering',
                'student_b': 'Model validation and testing',
                'student_c': 'Experiment tracking and optimization',
                'synergy': 'Comprehensive model development lifecycle'
            },
            {
                'aspect': 'Deployment & Serving',
                'student_a': 'Data preprocessing in production',
                'student_b': 'Model serving infrastructure',
                'student_c': 'Monitoring and automated retraining',
                'synergy': 'Production-ready ML serving pipeline'
            },
            {
                'aspect': 'Quality Assurance',
                'student_a': 'Data quality validation',
                'student_b': 'Integration testing',
                'student_c': 'Automated QA and error handling',
                'synergy': 'Robust, production-ready system'
            }
        ]

        print("\n🤝 Team Synergy Demonstration")
        print("=" * 60)

        for point in synergy_points:
            print(f"\n🎯 {point['aspect']}:")
            print(f"   Student A: {point['student_a']}")
            print(f"   Student B: {point['student_b']}")
            print(f"   Student C: {point['student_c']}")
            print(f"   → Synergy: {point['synergy']}")
            time.sleep(1)

    def generate_team_report(self):
        """Generate comprehensive team collaboration report."""
        self.log_action("Generating team collaboration report")

        report = {
            'timestamp': datetime.now().isoformat(),
            'team_members': self.team_members,
            'coordination_log': self.coordination_log,
            'active_servers': list(self.active_servers.keys()),
            'shared_registry': self.shared_registry_uri,
            'summary': {
                'total_actions': len(self.coordination_log),
                'servers_started': len(self.active_servers),
                'models_registered': 3,  # A, B, C
                'integration_status': 'successful' if len(self.active_servers) > 0 else 'pending'
            }
        }

        # Save report
        report_file = f"team_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.log_action("Team report generated", report_file)
        return report_file

    def cleanup_team_resources(self):
        """Clean up team resources and servers."""
        self.log_action("Cleaning up team resources")

        # Stop all active servers
        for member_id, server in self.active_servers.items():
            try:
                server.stop_server()
                self.log_action(f"Stopped {self.team_members[member_id]['name']} server")
            except Exception as e:
                self.log_action(f"Error stopping {self.team_members[member_id]['name']} server", str(e))

        self.active_servers.clear()
        self.log_action("Team resource cleanup completed")

class IntegratedDemoRunner:
    """Main runner for integrated team demonstrations."""

    def __init__(self):
        self.coordinator = TeamCoordinator()

    def run_full_integrated_demo(self):
        """Run complete integrated team demonstration."""
        print("🎭 Starting Integrated Team Demonstration")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        try:
            # Phase 1: Setup and Registration
            print("\n📋 Phase 1: Team Setup and Model Registration")
            self.coordinator.setup_shared_registry()
            self.coordinator.register_team_models()

            # Phase 2: Individual Contributions
            print("\n🔍 Phase 2: Individual Team Contributions")
            self.coordinator.demonstrate_team_synergy()

            # Phase 3: Integrated Pipeline
            print("\n🔗 Phase 3: Integrated Pipeline Execution")
            pipeline_result = self.coordinator.run_integrated_pipeline_demo()

            # Phase 4: Team Serving and Testing
            print("\n🚀 Phase 4: Team Model Serving and A/B Testing")
            self.coordinator.start_team_servers()
            ab_results = self.coordinator.run_team_ab_test()

            # Phase 5: Summary and Reporting
            print("\n📊 Phase 5: Team Summary and Reporting")
            report_file = self.coordinator.generate_team_report()

            print("\n🎉 Integrated team demonstration completed!")
            print(f"📄 Report saved: {report_file}")

            return {
                'status': 'success',
                'pipeline_result': pipeline_result,
                'ab_results': ab_results,
                'report_file': report_file
            }

        except Exception as e:
            print(f"\n❌ Integrated demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}

        finally:
            self.coordinator.cleanup_team_resources()

    def run_setup_only(self):
        """Run only the setup and registration phase."""
        print("🔧 Running Team Setup Only")
        self.coordinator.setup_shared_registry()
        self.coordinator.register_team_models()
        print("✅ Team setup completed")

    def run_serving_demo(self):
        """Run only the serving and A/B testing demo."""
        print("🚀 Running Team Serving Demo")
        self.coordinator.start_team_servers()
        ab_results = self.coordinator.run_team_ab_test()
        print("✅ Team serving demo completed")
        return ab_results

def main():
    """Main function with command line options."""
    import argparse

    parser = argparse.ArgumentParser(description='Student C - Integrated Team Demo')
    parser.add_argument('--full', action='store_true', help='Run complete integrated demo')
    parser.add_argument('--setup-only', action='store_true', help='Run only team setup')
    parser.add_argument('--serving-demo', action='store_true', help='Run only serving demo')
    parser.add_argument('--registry-uri', type=str, help='Shared registry URI')

    args = parser.parse_args()

    runner = IntegratedDemoRunner()

    if args.registry_uri:
        runner.coordinator.shared_registry_uri = args.registry_uri

    if args.setup_only:
        runner.run_setup_only()
    elif args.serving_demo:
        runner.run_serving_demo()
    else:
        # Default to full demo
        result = runner.run_full_integrated_demo()
        if result['status'] == 'failed':
            sys.exit(1)

if __name__ == "__main__":
    main()