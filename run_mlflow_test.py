#!/usr/bin/env python3
"""
MLflow Test Script - Runs key components from MLflow_main.ipynb
"""

import warnings
warnings.filterwarnings('ignore')

import os
import random
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Set tracking URI
TRACKING_URI = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

print("MLflow tracking URI:", mlflow.get_tracking_uri())

# Load data
iris = load_iris(as_frame=True)
X_iris = iris.data
y_iris = iris.target

wine = load_wine(as_frame=True)
X_wine = wine.data
y_wine = wine.target

# Titanic data
try:
    import seaborn as sns
    titanic = sns.load_dataset('titanic')
    titanic = titanic.dropna(subset=['survived'])
    titanic = titanic.select_dtypes(include=['number']).fillna(0)
    X_titanic = titanic.drop(columns=['survived'])
    y_titanic = titanic['survived']
except:
    print("Titanic dataset not available, skipping")
    X_titanic = None
    y_titanic = None

# Training function
def train_and_log_baselines(X, y, dataset_name='dataset', seed=42):
    experiment_name = f"{dataset_name} experiment"
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f'Warning: could not set experiment "{experiment_name}": {e}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    models = {
        'random_forest': RandomForestClassifier(random_state=seed),
        'logistic_regression': LogisticRegression(max_iter=500)
    }

    for name, model in models.items():
        with mlflow.start_run(nested=True):
            mlflow.log_param('model', name)
            mlflow.log_param('dataset', dataset_name)
            mlflow.log_param('seed', seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mlflow.log_metrics({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})

            try:
                input_example = X_train.head(5)
                sample_preds = model.predict(input_example)
                signature = infer_signature(input_example, sample_preds)
                mlflow.sklearn.log_model(model, artifact_path='./artifacts/', signature=signature, input_example=input_example)
            except Exception as e:
                print(f"Model logging failed: {e}")
                try:
                    mlflow.sklearn.log_model(model, artifact_path='./artifacts/')
                except Exception:
                    pass

# Generate seeds
seeds = [random.randint(0, 10000) for _ in range(3)]
print(f'Using seeds for testing: {seeds}')

# Train baselines
for seed in seeds:
    train_and_log_baselines(X_iris, y_iris, dataset_name='iris', seed=seed)
    if X_titanic is not None:
        train_and_log_baselines(X_titanic, y_titanic, dataset_name='titanic', seed=seed)
    train_and_log_baselines(X_wine, y_wine, dataset_name='wine', seed=seed)

print("Baseline training completed")

# Model Registry Demo
mlflow.set_experiment("Model Registry Demo")
model_name = "IrisClassifier"

model_configs = [
    {"n_estimators": 10, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10}
]

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

for i, config in enumerate(model_configs, 1):
    with mlflow.start_run(run_name=f"version_{i}"):
        model = RandomForestClassifier(n_estimators=config['n_estimators'], max_depth=config['max_depth'], random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_params(config)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        model_info = mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)
        print(f"Registered {model_name} version {i} - Accuracy: {accuracy:.4f}")

# Stage transitions
client.transition_model_version_stage(name=model_name, version=1, stage="Staging")
client.transition_model_version_stage(name=model_name, version=2, stage="Staging", archive_existing_versions=True)
client.transition_model_version_stage(name=model_name, version=3, stage="Production")

print("Model registry and stage transitions completed")

# Create directories
os.makedirs('model_artifacts', exist_ok=True)
os.makedirs('deployment', exist_ok=True)

# Load production model and log artifacts
model_uri = f"models:/{model_name}/Production"
prod_model = mlflow.sklearn.load_model(model_uri)

y_pred = prod_model.predict(X_test)
y_pred_proba = prod_model.predict_proba(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix - Production Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('model_artifacts/confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.close()
mlflow.log_artifact('model_artifacts/confusion_matrix.png')

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': prod_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Production Model')
plt.gca().invert_yaxis()
plt.savefig('model_artifacts/feature_importance.png', bbox_inches='tight', dpi=150)
plt.close()
mlflow.log_artifact('model_artifacts/feature_importance.png')

# Classification Report
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('model_artifacts/classification_report.csv')
mlflow.log_artifact('model_artifacts/classification_report.csv')

# Model Metadata
metadata = {
    "model_name": model_name,
    "version": "3 (Production)",
    "training_date": datetime.now().isoformat(),
    "framework": "scikit-learn",
    "algorithm": "RandomForestClassifier",
    "metrics": {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    },
    "feature_names": iris.feature_names,
    "target_names": iris.target_names.tolist(),
    "n_samples_train": len(X_train),
    "n_samples_test": len(X_test)
}
with open('model_artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
mlflow.log_artifact('model_artifacts/model_metadata.json')

print("Artifacts logged")

# Batch inference test
test_samples = pd.DataFrame({
    'sepal length (cm)': [5.1, 6.7, 4.9],
    'sepal width (cm)': [3.5, 3.0, 2.5],
    'petal length (cm)': [1.4, 5.2, 4.5],
    'petal width (cm)': [0.2, 2.3, 1.7]
})

predictions = prod_model.predict(test_samples)
probabilities = prod_model.predict_proba(test_samples)

print("\nBatch Inference Results:")
print("-" * 50)
for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
    pred_name = iris.target_names[pred]
    confidence = probs[pred] * 100
    print(f"Sample {i+1}: {pred_name} (confidence: {confidence:.1f}%)")

print("\nMLflow test completed successfully!")