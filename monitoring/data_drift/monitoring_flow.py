import os

from dotenv import load_dotenv
from monitoring_tasks import (
    detect_data_drift,
    fetch_data_from_s3,
    generate_evidently_report,
    load_reference_data,
)
from prefect import flow, get_run_logger
from prefect.runtime import flow_run
from retraining_trigger import retraining_trigger

from churn_project.constants import SCHEMA_FILE_PATH
from churn_project.utils import read_yaml

# Local testing only.
load_dotenv()
# For deployed flows, secrets and credentials are handled by blocks.


@flow(name="DataMonitoringFlow")
def data_monitoring_flow(
    date: str | None = None,
    threshold: float = 0.05,
):
    """Main flow to perform data monitoring using Evidently."""
    logger = get_run_logger()
    if not date:
        date = flow_run.get_scheduled_start_time().strftime("%Y-%m-%d")
        logger.info("Using scheduled start time.")
    else:
        logger.info("Using provided date.")
    logger.info(f"Starting Data Monitoring Flow for date: {date}")

    schema = read_yaml(SCHEMA_FILE_PATH)
    columns = ", ".join(schema.columns.keys())
    target = schema.target_column
    reference_data = load_reference_data(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        query=f"SELECT {columns} FROM churners;",
        target=target,
    )

    log_bucket = os.getenv("LOG_S3_BUCKET")
    log_prefix = os.getenv("LOG_S3_PREFIX")
    current_data = fetch_data_from_s3(log_bucket, log_prefix, date)
    if current_data.empty:
        logger.error(f"No data found for date {date} in s3://{log_bucket}/{log_prefix}")
        return

    current_data = current_data[reference_data.columns]  # Ensure same column order
    drift_report = detect_data_drift(
        reference_data=reference_data,
        current_data=current_data,
        threshold=threshold,
    )

    generate_evidently_report(
        reference_data=reference_data,
        current_data=current_data,
        drift_report=drift_report,
        output_s3_uri=f"s3://{log_bucket}/monitoring_reports",
        date=date,
    )

    retraining_trigger(drift_report=drift_report, date=date)


data_monitoring_flow()
