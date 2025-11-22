import pandas as pd
import pytest

from churn_project.exception import CustomException


def test_empty_dataframe_raises_error(predictor):
    """Empty input should raise CustomException."""
    df = pd.DataFrame()

    with pytest.raises(CustomException):
        predictor.predict(df)


def test_dataframe_with_all_null_values_raises_error(predictor):
    """A DataFrame with all nulls should raise an error."""
    df = pd.DataFrame(
        {
            "gender": [None],
            "tenure": [None],
            "MonthlyCharges": [None],
        }
    )

    with pytest.raises(CustomException):
        predictor.predict(df)


def test_invalid_columns_do_not_break_predictor(predictor):
    """
    If columns are wrong, the pipeline (inside MLflow) should throw,
    and our predictor must wrap it into CustomException.
    """
    df = pd.DataFrame({"wrong_column": [1]})

    with pytest.raises(CustomException):
        predictor.predict(df)
