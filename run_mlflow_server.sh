#!/usr/bin/env bash
set -euo pipefail

# run_mlflow_server.sh
# Simple helper to start an MLflow tracking server with a local SQLite backend
# Usage:
#   ./run_mlflow_server.sh [PORT] [HOST]
# Examples:
#   ./run_mlflow_server.sh           # starts on 0.0.0.0:5000
#   ./run_mlflow_server.sh 1234      # starts on 0.0.0.0:1234
#   ./run_mlflow_server.sh 1234 127.0.0.1

PORT=${1:-5000}
HOST=${2:-127.0.0.1}
BACKEND_URI="sqlite:///mlflow.db"
ARTIFACT_ROOT="./artifacts"

# Create artifact directory if missing
mkdir -p "$ARTIFACT_ROOT"

# Check mlflow is available
if ! command -v mlflow >/dev/null 2>&1; then
  echo "mlflow not found in PATH. Install with: pip install mlflow" >&2
  exit 1
fi

echo "Starting MLflow server"
echo "Backend store: $BACKEND_URI"
echo "Artifact root: $ARTIFACT_ROOT"
echo "Listening: $HOST:$PORT"
	mlflow server \
    --backend-store-uri "$BACKEND_URI" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --host "$HOST" \
    --port "$PORT"
