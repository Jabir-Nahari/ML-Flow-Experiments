import time
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

# Load datasets
wine = load_wine()
iris = load_iris()

datasets = {'wine': wine, 'iris': iris}

# Benchmark results storage
benchmark_results = []

for dataset_name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLflow setup
    setup_start = time.time()
    mlflow.set_experiment(f"benchmark_{dataset_name}")
    setup_end = time.time()
    setup_time = setup_end - setup_start

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("seed", 42)

        # Training
        train_start = time.time()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        train_end = time.time()
        training_time = train_end - train_start

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Log metrics
        logging_start = time.time()
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, "model")
        logging_end = time.time()
        logging_overhead = logging_end - logging_start

        benchmark_results.append({
            'Platform': 'MLflow',
            'Dataset': dataset_name,
            'Setup Time (s)': setup_time,
            'Training Time (s)': training_time,
            'Logging Overhead (s)': logging_overhead,
            'Accuracy': accuracy
        })

# Print results
benchmark_df = pd.DataFrame(benchmark_results)
print(benchmark_df)