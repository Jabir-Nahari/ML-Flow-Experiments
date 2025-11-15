# ML-Flow-Experiments
Repo for the Big Data course.

# Running
To run the server run the file: `run_mlflow_server.sh`
To run the code: only run through the **`Project_entrypoint.ipynb`**

# Requirements

- **Docker**
- **mlflow**: experiment tracking, model registry and serving.
- **scikit-learn**: model training and utilities (`sklearn`).
- **pandas**: data handling and I/O.
- **numpy**: numerical computing.
- **seaborn**: plotting (notebooks / reports).
- **matplotlib**: plotting.
- **boto3**: AWS S3 integration for artifact storage.
- **psutil**: optional runtime/process monitoring used by tests and helpers.
- **joblib**: model persistence and utilities (used in `end_to_end_pipeline.py`).


# Deleting DB experiments:
**To remove the DBs Experiments**: run `sqlite3 mlflow.db < delete_mlflow_exp.sql` in the root directory.