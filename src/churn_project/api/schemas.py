from typing import Annotated, List

from pydantic import BaseModel, Field


class UserInput(BaseModel):
    # In the same order as the training features
    Total_Relationship_Count: Annotated[
        float, Field(..., description="Total number of relationships")
    ]
    Credit_Limit: Annotated[float, Field(..., description="Credit limit of the user")]
    Total_Revolving_Bal: Annotated[
        float, Field(..., description="Total revolving balance")
    ]
    Total_Amt_Chng_Q4_Q1: Annotated[
        float, Field(..., description="Total amount change from Q4 to Q1")
    ]
    Total_Trans_Amt: Annotated[
        float, Field(..., description="Total transaction amount")
    ]
    Total_Trans_Ct: Annotated[float, Field(..., description="Total transaction count")]
    Total_Ct_Chng_Q4_Q1: Annotated[
        float, Field(..., description="Total count change from Q4 to Q1")
    ]
    Avg_Utilization_Ratio: Annotated[
        float, Field(..., description="Average utilization ratio")
    ]


class BatchInput(BaseModel):
    records: List[UserInput]


class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    timestamp: str
    num_samples: int
