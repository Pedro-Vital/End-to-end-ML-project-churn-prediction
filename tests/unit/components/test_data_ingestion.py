# These unit tests are intentionally lightweight.
# They exist to demonstrate pytest usage and CI integration.
# They are not designed to be production-level tests.

import pandas as pd

from churn_project.components.data_ingestion import DataIngestion
from churn_project.entity.config_entity import DataIngestionConfig


def test_split_data_smoke(tmp_path):
    raw = tmp_path / "raw.csv"
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"

    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 5, 6, 7]})
    df.to_csv(raw, index=False)

    config = DataIngestionConfig(
        db_host=None,
        db_user=None,
        db_password=None,
        db_name=None,
        base_query="",
        raw_data_path=raw,
        training_path=train,
        testing_path=test,
        train_test_split_ratio=0.25,
        random_state=42,
        columns={"A": "int64", "B": "int64"},
    )

    ingestion = DataIngestion(config)
    ingestion.split_data()

    assert train.exists()
    assert test.exists()
