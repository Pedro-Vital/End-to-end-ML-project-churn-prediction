import json
import tempfile
from typing import Any, Dict

import boto3
import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from prefect import get_run_logger, task
from scipy.stats import ks_2samp
from sqlalchemy import create_engine

from churn_project.aws.s3_utils import upload_folder_to_s3


@task
def load_reference_data(
    *,
    user: str,
    password: str,
    host: str,
    database: str,
    query: str,
    target: str,
) -> pd.DataFrame:
    """Load reference data from a MySQL database using SQLAlchemy."""
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
    df = pd.read_sql(query, con=engine)
    df.drop(columns=[target], inplace=True)
    logger = get_run_logger()
    logger.info(f"Loaded reference data with {len(df)} records from database.")
    return df


@task
def fetch_data_from_s3(bucket: str, prefix: str, date: str) -> pd.DataFrame:
    """
    Load all JSON logs for a given date folder and convert them into a flat DataFrame.
    """
    s3 = boto3.client("s3")
    folder = f"{prefix}/date={date}/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=folder)

    if "Contents" not in response:
        return pd.DataFrame()  # No data for this date

    rows = []

    for obj in response["Contents"]:
        key = obj["Key"]
        file_obj = s3.get_object(Bucket=bucket, Key=key)
        content = json.loads(file_obj["Body"].read())

        # Unifies single and batch format
        if "input" in content:
            rows.append({**content["input"]})
            # content["input"] is Dict[str, float]
            # Result is a single dict to be appended
        elif "inputs" in content:
            for item in content["inputs"]:
                rows.append({**item})
            # content["inputs"] is List[Dict[str, float]]
            # Result is multiple dicts to be appended
    logger = get_run_logger()
    logger.info(f"Fetched {len(rows)} records from s3://{bucket}/{folder}")
    return pd.DataFrame(rows)


@task
def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.05,
) -> dict:
    """
    Perform KS test to detect drift on numerical columns.

    Returns:
        {
            "drift_detected": bool,
            "threshold": float,
            "features": {
                column_name: {
                    "p_value": float,
                    "statistic": float,
                    "drifted": bool
                }
            }
        }
    """
    logger = get_run_logger()

    if len(current_data) < 50:
        logger.warning(
            "Current data has less than 50 records. KS test may be unreliable."
        )

    drift_report: Dict[str, Any] = {
        "drift_detected": False,
        "threshold": threshold,
        "features": {},
    }

    for col in reference_data.columns:
        stat, p_value = ks_2samp(reference_data[col], current_data[col])
        logger.info(f"KS test for column {col}: statistic={stat}, p_value={p_value}")
        drifted = p_value < threshold
        drift_report["features"][col] = {
            "p_value": p_value,
            "statistic": stat,
            "drifted": drifted,
        }
        if drifted:
            drift_report["drift_detected"] = True
    logger.info(
        f"Drift detection completed. Drift detected: {drift_report['drift_detected']}"
    )
    return drift_report


@task
def generate_evidently_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drift_report: dict,
    output_s3_uri: str,
    date: str,
):
    """Generate Evidently report for data drift and summary."""
    eval_data_curr = Dataset.from_pandas(current_data)
    eval_data_ref = Dataset.from_pandas(reference_data)

    report = Report([DataDriftPreset(method="ks"), DataSummaryPreset()])

    eval = report.run(reference_data=eval_data_ref, current_data=eval_data_curr)
    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = f"{temp_dir}/evidently_report.html"
        metadata_path = f"{temp_dir}/drift_metadata.json"

        # Save evidently report
        eval.save_html(report_path)

        # Save metadata
        metadata = {
            "date": date,
            "drift_detected": drift_report["drift_detected"],
            "threshold": drift_report["threshold"],
            "num_features": len(drift_report["features"]),
            "num_drifted_features": sum(
                1 for v in drift_report["features"].values() if v["drifted"]
            ),
            "drifted_features": [
                k for k, v in drift_report["features"].items() if v["drifted"]
            ],
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Upload entire folder
        dated_s3_uri = f"{output_s3_uri}/date={date}"
        upload_folder_to_s3(temp_dir, dated_s3_uri)
        logger = get_run_logger()
        logger.info(f"Evidently report and metadata uploaded to {dated_s3_uri}")
