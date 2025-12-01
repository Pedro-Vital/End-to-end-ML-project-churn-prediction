# These unit tests are intentionally lightweight.
# They exist to demonstrate pytest usage and CI integration.
# They are not designed to be production-level tests.

import pandas as pd

from churn_project.components.data_validation import DataValidation
from churn_project.entity.config_entity import DataValidationConfig


def test_validate_columns_success(tmp_path):
    df = pd.DataFrame({"A": [1], "B": [2]})

    # only tests logic — config only contains expected columns
    config = DataValidationConfig(
        validation_report_path=tmp_path / "report.json",
        columns={"A": "int64", "B": "int64"},
    )

    validator = DataValidation(config)
    assert validator.validate_columns(df) is True
