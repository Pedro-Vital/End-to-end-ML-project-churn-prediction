import mlflow
from prefect import flow, task

from churn_project.components.data_ingestion import DataIngestion
from churn_project.components.data_transformation import DataTransformation
from churn_project.components.data_validation import DataValidation
from churn_project.components.model_evaluation import ModelEvaluation
from churn_project.components.model_pusher import ModelPusher
from churn_project.components.model_trainer import ModelTrainer
from churn_project.config.configuration import ConfigurationManager
from churn_project.logger import logger

# Define Prefect Tasks


@task(retries=2, retry_delay_seconds=10)
def data_ingestion_task(config):
    logger.info("Prefect Task: Data Ingestion")
    ingestion = DataIngestion(config)
    return ingestion.initiate_data_ingestion()


@task
def data_validation_task(config, ingestion_artifact):
    logger.info("Prefect Task: Data Validation")
    validator = DataValidation(config)
    return validator.initiate_data_validation(ingestion_artifact)


@task
def data_transformation_task(config, validation_artifact, ingestion_artifact):
    logger.info("Prefect Task: Data Transformation")

    if not validation_artifact.validation_status:
        raise Exception(
            "Data validation failed. Cannot proceed to data transformation."
        )

    transformer = DataTransformation(config)
    return transformer.initiate_data_transformation(ingestion_artifact)


@task
def model_trainer_task(config, transformation_artifact):
    logger.info("Prefect Task: Model Trainer")
    trainer = ModelTrainer(config)
    return trainer.initiate_model_trainer(transformation_artifact)


@task
def model_evaluation_task(config, transformation_artifact, trainer_artifact):
    logger.info("Prefect Task: Model Evaluation")
    evaluator = ModelEvaluation(config)
    return evaluator.initiate_model_evaluation(
        data_transformation_artifact=transformation_artifact,
        model_trainer_artifact=trainer_artifact,
    )


@task
def model_pusher_task(config, evaluation_artifact, trainer_artifact):
    logger.info("Prefect Task: Model Pusher")
    pusher = ModelPusher(config)
    return pusher.initiate_model_pusher(
        model_evaluation_artifact=evaluation_artifact,
        model_trainer_artifact=trainer_artifact,
    )


# Define Prefect Flow


@flow(name="TrainingPipelineFlow")
def training_flow():
    """
    This flow orchestrates the entire training pipeline using Prefect.
    """
    logger.info("Starting Prefect Training Pipeline Flow")

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
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(mlflow_config.experiment_name)

    with mlflow.start_run(run_name="Pipeline_Run"):
        mlflow.set_tag("orchestrator", "prefect")

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

        return {
            "ingestion": ingestion_artifact,
            "validation": validation_artifact,
            "transformation": transformation_artifact,
            "trainer": trainer_artifact,
            "evaluation": evaluation_artifact,
            "pusher": pusher_artifact,
        }
