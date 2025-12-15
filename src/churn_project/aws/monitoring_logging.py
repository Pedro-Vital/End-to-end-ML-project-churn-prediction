import json
import traceback
import uuid
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

from churn_project.logger import logger


def upload_log_to_s3(log_data: dict, bucket: str, prefix: str):
    """
    Uploads a log entry as a JSON file to the specified S3 bucket and prefix.

    Args:
        log_data (dict): The log data to upload.
        bucket (str): The S3 bucket name.
        prefix (str): The S3 prefix (folder path) where the log will be stored.
    """
    s3_client = boto3.client("s3")

    # Partition by date
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    request_id = log_data.get("request_id", str(uuid.uuid4()))
    key = f"{prefix}/date={date_str}/id={request_id}.json"

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(log_data).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Uploaded log to s3://{bucket}/{key}")
    except ClientError as e:
        logger.error(f"Failed to upload log to s3://{bucket}/{key}: {e}")
        logger.error(traceback.format_exc())
        raise
