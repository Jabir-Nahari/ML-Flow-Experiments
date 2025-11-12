FROM python:3.9-slim

# Upgrade pip to avoid installation issues
RUN pip install --upgrade pip

# Install dependencies with verbose output and no cache
RUN pip install --no-cache-dir --verbose     mlflow     scikit-learn     pandas     numpy

# Set working directory
WORKDIR /app

# Copy MLflow runs directory
COPY ./mlruns ./mlruns

# Expose port
EXPOSE 5001

# Serve model
CMD ["mlflow", "models", "serve", "-m", "models:/IrisClassifier/Production", "-p", "5001", "--host", "0.0.0.0", "--no-conda"]
