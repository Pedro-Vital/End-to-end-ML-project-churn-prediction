# These unit tests are intentionally lightweight.
# They exist to demonstrate pytest usage and CI integration.
# They are not designed to be production-level tests.

import pandas as pd

from churn_project.components.data_transformation import FeatureEngineer


def test_feature_engineer_drops_configured_columns():
    df = pd.DataFrame(
        {
            "Total_Amt_Chng_Q4_Q1": [1.0],
            "Total_Trans_Amt": [10],
            "Total_Revolving_Bal": [5],
            "Avg_Utilization_Ratio": [0.5],
            "Credit_Limit": [100],
        }
    )

    fe = FeatureEngineer(drop_columns=["Credit_Limit"])
    transformed = fe.transform(df)

    assert "Credit_Limit" not in transformed.columns
    assert "Activity_Growth" in transformed.columns
    assert "Customer_Value" in transformed.columns
