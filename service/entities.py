from tkinter.filedialog import Open
import typing as t
import typing_extensions as te

from pydantic import BaseModel, Field, ConstrainedInt, PositiveInt,NonNegativeFloat, NonNegativeInt
from typing import List

class ModelInput(BaseModel):
    Asset_ID: PositiveInt
    Count: NonNegativeFloat
    Open: NonNegativeFloat
    High: NonNegativeFloat
    Low: NonNegativeFloat
    Close: NonNegativeFloat
    Volume: NonNegativeFloat
    VWAP: NonNegativeFloat

#class ModelInput(BaseModel):
   # forecast_data: List[ForecastInput]

class NumPredictions(BaseModel):
    Steps: PositiveInt

