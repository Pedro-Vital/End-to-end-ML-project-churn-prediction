import mlflow
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

from churn_project.components.data_ingestion import DataIngestion
from churn_project.components.data_transformation import DataTransformation
from churn_project.components.data_validation import DataValidation
from churn_project.components.model_evaluation import ModelEvaluation
from churn_project.components.model_pusher import ModelPusher
from churn_project.components.model_trainer import ModelTrainer
from churn_project.config.configuration import ConfigurationManager

# Define Prefect Tasks


@task(retries=2, retry_delay_seconds=10)
def data_ingestion_task(config):
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting Data Ingestion task")
    ingestion = DataIngestion(config)
    return ingestion.initiate_data_ingestion()


@task
def data_validation_task(config, ingestion_artifact):
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting Data Validation task")
    validator = DataValidation(config)
    return validator.initiate_data_validation(ingestion_artifact)


@task
def data_transformation_task(config, validation_artifact, ingestion_artifact):
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting Data Transformation task")

    if not validation_artifact.validation_status:
        raise Exception(
            "Data validation failed. Cannot proceed to data transformation."
        )

    transformer = DataTransformation(config)
    return transformer.initiate_data_transformation(ingestion_artifact)


@task
def model_trainer_task(config, transformation_artifact):
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting Model Trainer task")
    trainer = ModelTrainer(config)
    return trainer.initiate_model_trainer(transformation_artifact)


@task
def model_evaluation_task(config, transformation_artifact, trainer_artifact):
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting Model Evaluation task")
    evaluator = ModelEvaluation(config)
    return evaluator.initiate_model_evaluation(
        data_transformation_artifact=transformation_artifact,
        model_trainer_artifact=trainer_artifact,
    )


@task
def model_pusher_task(config, evaluation_artifact, trainer_artifact):
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting Model Pusher task")
    pusher = ModelPusher(config)
    return pusher.initiate_model_pusher(
        model_evaluation_artifact=evaluation_artifact,
        model_trainer_artifact=trainer_artifact,
    )


# Define Prefect Flow


@flow(name="TrainingPipelineFlow")
def training_flow(
    trigger_reason: str | None = "Manual Trigger",
    drift_date: str | None = None,
    threshold: float | None = None,
    num_drifted_features: int | None = None,
):
    """
    This flow orchestrates the entire training pipeline using Prefect.
    """
    # Load environment variables
    load_dotenv()
    prefect_logger = get_run_logger()
    if trigger_reason:
        prefect_logger.info(
            f"Triggered by {trigger_reason} | "
            f"date={drift_date} | "
            f"threshold={threshold} | "
            f"drifted_features={num_drifted_features}"
        )
    prefect_logger.info("Starting Prefect Training Pipeline Flow")

    # Load configs
    config_manager = ConfigurationManager()

    mlflow_config = config_manager.get_mlflow_config()
    ingestion_config = config_manager.get_data_ingestion_config()
    validation_config = config_manager.get_data_validation_config()
    transformation_config = config_manager.get_data_transformation_config()
    trainer_config = config_manager.get_model_trainer_config()
    evaluation_config = config_manager.get_model_evaluation_config()
    pusher_config = config_manager.get_model_pusher_config()

    # MLflow Integration
    try:
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name)
    except mlflow.MlflowException as e:
        prefect_logger.error(f"MLflow setup failed: {e}")
        raise

    with mlflow.start_run(run_name="Pipeline_Run"):
        mlflow.set_tag("trigger_reason", trigger_reason)
        mlflow.set_tag("drift_date", drift_date)

        ingestion_artifact = data_ingestion_task(ingestion_config)
        validation_artifact = data_validation_task(
            validation_config, ingestion_artifact
        )
        transformation_artifact = data_transformation_task(
            transformation_config, validation_artifact, ingestion_artifact
        )
        trainer_artifact = model_trainer_task(trainer_config, transformation_artifact)
        evaluation_artifact = model_evaluation_task(
            evaluation_config, transformation_artifact, trainer_artifact
        )
        pusher_artifact = model_pusher_task(
            pusher_config, evaluation_artifact, trainer_artifact
        )

    prefect_logger.info("Prefect Training Pipeline Flow completed successfully.")
    # promoted if pusher_artifact["promoted"] else "not promoted"
    prefect_logger.info(
        f"Result: {'promoted' if pusher_artifact.promoted['promoted'] else 'not promoted'}"
    )
