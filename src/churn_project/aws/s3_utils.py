import os
from typing import Tuple

import boto3
from botocore.exceptions import ClientError

from churn_project.logger import logger


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse S3 URI into bucket and prefix."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    parts = s3_uri[5:].split("/", 1)  # Remove 's3://' and split at first '/'
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""  # Empty prefix case
    return bucket, prefix.rstrip("/")  # strip trailing slash
    # Example: s3://my-bucket/my-prefix -> ("my-bucket", "my-prefix")


def upload_folder_to_s3(folder_path: str, s3_uri: str) -> None:
    """
    Upload entire local_folder contents into s3_uri (prefix).
    s3_uri example: s3://bucket/path/to/production/
    """
    s3 = boto3.client("s3")
    bucket, prefix = parse_s3_uri(s3_uri)

    for root, _, files in os.walk(folder_path):  # iterates over files in directory
        # root is current directory being traversed
        for file in files:
            local_path = os.path.join(root, file)
            # build key relative to the folder_path
            relative_path = os.path.relpath(local_path, folder_path)
            s3_key = f"{prefix}/{relative_path}".replace("\\", "/").lstrip("/")
            # replace backslashes for Windows compatibility

            try:
                s3.upload_file(local_path, bucket, s3_key)
                logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
            except ClientError as e:
                logger.error(
                    f"Failed to upload {local_path} to s3://{bucket}/{s3_key}: {e}"
                )
                raise


def download_s3_folder(s3_uri: str, local_dir: str) -> None:
    """
    Download all objects under s3_uri (prefix) into local_dir preserving folder structure.
    """
    s3 = boto3.client("s3")
    bucket, prefix = parse_s3_uri(s3_uri)

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]  # each key is a full path in the bucket

            rel_path = key[len(prefix) :].lstrip("/")
            if rel_path == "":
                continue

            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            try:
                s3.download_file(bucket, key, local_path)
                logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
            except ClientError as e:
                logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
                raise
