
import pickle
from pathlib import Path

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the saved model and vectorizer
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "logistic_regression_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
dv = artifact["vectorizer"]
threshold = artifact["threshold"]

# Load the latest dataset
DATA_PATH = BASE_DIR / "cleaned_data.csv"
data_df = pd.read_csv(DATA_PATH)
data_df["week"] = pd.to_datetime(data_df["week"])

FEATURES = [
    "fatalities_lag_w4_log",
    "fatalities_lag_w12_log",
    "population_exposure_lag_w4_log",
    "population_exposure_lag_w12_log",
]

# Check that the required columns exist
required_columns = ["admin1", "week"] + FEATURES
missing_columns = [
    col for col in required_columns if col not in data_df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns in CSV: {missing_columns}")

app = FastAPI(title="Humanitarian Conflict Early Warning API")


class ConflictInput(BaseModel):
    admin1: str = Field(min_length=1, description="Nigerian state or FCT")


@app.get("/")
def home():
    return {"status": "Conflict prediction API running"}


@app.post("/predict")
def predict(data: ConflictInput):

    # Find the requested state, ignoring capitalisation and extra spaces
    state = data.admin1.strip().casefold()

    state_data = data_df[
        data_df["admin1"].astype(str).str.strip().str.casefold() == state
    ]

    if state_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for state: {data.admin1}"
        )

    # Retrieve the latest observation for that state
    latest_row = state_data.sort_values("week").iloc[-1]

    # Extract the four features used by the trained model
    input_dict = {
        feature: float(latest_row[feature])
        for feature in FEATURES
    }

    if any(pd.isna(value) for value in input_dict.values()):
        raise HTTPException(
            status_code=422,
            detail="Missing lagged feature values for this state."
        )

    input_dict["admin1"] = latest_row["admin1"]

    # Generate prediction
    X = dv.transform([input_dict])
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "admin1": latest_row["admin1"],
        "observation_week": latest_row["week"].strftime("%Y-%m-%d"),
        "probability": round(probability, 4),
        "predicted_conflict": prediction,
        "threshold": threshold
    }
