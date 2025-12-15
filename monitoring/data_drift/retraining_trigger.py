from prefect import get_run_logger, task
from prefect.deployments import run_deployment


@task
def retraining_trigger(drift_report: dict, date: str):
    """
    Trigger retraining flow if data drift is detected.

    Args:
        drift_report (dict): The drift report containing drift detection results.
        date (str): The date string associated with the current data.
    """
    logger = get_run_logger()

    if not drift_report["drift_detected"]:
        logger.info("No drift detected. Retraining will not be triggered.")
        return

    logger.warning("Data drift detected. Triggering retraining flow.")

    # Trigger the training flow deployment
    run_deployment(
        name="TrainingPipelineFlow/churn_train",
        parameters={
            "trigger_reason": "Data Drift Detected",
            "drift_date": date,
            "threshold": drift_report["threshold"],
            "num_drifted_features": [
                k for k, v in drift_report["features"].items() if v["drifted"]
            ],
        },
    )

    logger.info("Retraining flow triggered successfully.")
