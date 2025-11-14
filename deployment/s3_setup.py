"""
AWS S3 Integration for MLflow Artifact Storage
This script configures MLflow to use S3 for artifact storage with fallback to local.
"""

import os
import mlflow
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def setup_s3_artifact_store(bucket_name='mlflow-artifacts-bucket', region='us-east-1'):
    """
    Configure MLflow to use S3 for artifact storage with fallback to local.

    Args:
        bucket_name (str): S3 bucket name for artifacts
        region (str): AWS region for the bucket
    """
    try:
        # Check if AWS credentials are available
        boto3.client('s3').list_buckets()
        print("✓ AWS credentials found. Configuring S3 artifact storage...")

        # Set MLflow artifact URI to S3
        artifact_uri = f"s3://{bucket_name}/artifacts"
        mlflow.set_tracking_uri(f"sqlite:///mlruns/mlflow.db")
        os.environ['MLFLOW_ARTIFACT_ROOT'] = artifact_uri

        print(f"✓ MLflow artifact storage configured to: {artifact_uri}")

        # Test S3 connection
        s3_client = boto3.client('s3', region_name=region)
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✓ S3 bucket '{bucket_name}' exists and is accessible")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"⚠️  S3 bucket '{bucket_name}' does not exist. Creating it...")
                s3_client.create_bucket(Bucket=bucket_name)
                print(f"✓ Created S3 bucket '{bucket_name}'")
            else:
                raise e

    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"⚠️  AWS credentials not found or invalid: {e}")
        print("Falling back to local artifact storage...")
        setup_local_artifact_store()

    except Exception as e:
        print(f"⚠️  Error configuring S3: {e}")
        print("Falling back to local artifact storage...")
        setup_local_artifact_store()

def setup_local_artifact_store():
    """Configure MLflow to use local artifact storage."""
    local_path = "./mlruns"
    mlflow.set_tracking_uri(f"sqlite:///{local_path}/mlflow.db")
    os.environ['MLFLOW_ARTIFACT_ROOT'] = local_path
    print(f"✓ MLflow artifact storage configured to local: {local_path}")

def test_s3_connection(bucket_name='mlflow-artifacts-bucket', region='us-east-1'):
    """Test S3 connection and permissions."""
    try:
        s3_client = boto3.client('s3', region_name=region)

        # Test bucket access
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Successfully connected to S3 bucket: {bucket_name}")

        # Test write permissions by uploading a small test file
        test_key = "test_connection.txt"
        s3_client.put_object(Bucket=bucket_name, Key=test_key, Body=b"test")
        print("✓ Write permissions confirmed")

        # Clean up test file
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print("✓ Test file cleaned up")

        return True

    except Exception as e:
        print(f"✗ S3 connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    print("Setting up MLflow with S3 artifact storage...")

    # Configure with your bucket details
    BUCKET_NAME = "mlflow-artifacts-bucket"  # Change this to your bucket
    REGION = "us-east-1"  # Change this to your region

    setup_s3_artifact_store(bucket_name=BUCKET_NAME, region=REGION)

    # Test the connection
    if test_s3_connection(bucket_name=BUCKET_NAME, region=REGION):
        print("🎉 S3 integration successful!")
    else:
        print("⚠️  S3 integration failed, using local storage")