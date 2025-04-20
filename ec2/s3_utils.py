import boto3
import os
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

# Initialize constants
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
REGION = os.getenv("AWS_DEFAULT_REGION")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize boto3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION
)

def upload_to_s3(file_path: str, s3_key: str):
    """Uploads a local file to the specified S3 bucket."""
    try:
        s3.upload_file(file_path, BUCKET_NAME, s3_key)
        print(f"✅ Uploaded '{file_path}' to S3 as '{s3_key}'")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

def download_from_s3(s3_key: str, local_path: str):
    """Downloads a file from the specified S3 bucket to a local path."""
    try:
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"✅ Downloaded '{s3_key}' from S3 to '{local_path}'")
    except Exception as e:
        print(f"❌ Download failed: {e}")
