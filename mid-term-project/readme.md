# Predicting conflict severity in Nigeria
---
This notebook goes through a pipline predicting severity of political violence at subnational level in Nigeria using historical features in ACLED. The selected model is deployed as a containarised API. An aribtary threshold of 5 fatalities was coded 'high severity conflict' for a given week. 


## Project Structure
 
```
nigeria-political-violence/
│
├── data/
│   ├── raw/ACLED Data_2026-11-02_event_date_from_1997-01-01_event_date_to_2026-10-30.csv
│   └── processed/cleaned_data.csv
│
├── notebooks/
│   ├── 01_data_cleaning_EDA.ipynb
│   └── 02_modelling_deployment.ipynb
│
├── models/
│   └── logistic_regression_model.pkl
│
├── conflict-prediction-api/
│   ├── main.py
│   ├── schemas.py
│   ├── Dockerfile
│   └── .dockerignore
|   └── requirements.txt
│   └── tests
│        └── tests_api.py
│
│
├── conflict-prediction-api.zip
| 
└── README.md
```

## Workflow
 
1. **Data cleaning and EDA** (`01_data_cleaning_EDA.ipynb`): cleans the raw ACLED export (events from 1997-01-01 to 2026-10-30) and explores patterns in political violence. Output is saved to `data/processed/cleaned_data.csv`.
2. **Modelling and deployment** (`02_modelling_deployment.ipynb`): trains and compares Logistic Regression, Random Forest and XGBoost, selects a decision threshold for each, and exports the chosen model to `models/`.
3. **API** (`conflict-prediction-api/`): serves the selected model for predictions, with request/response schemas in `schemas.py` and a Docker setup for deployment.
4. **Tests** (`tests/test_api.py`): tests for the API.


##  Feature selection:

| Features | Rationale for feature selection | Any issues |
|---|---|---|
| `fatalities_lag_w4_log` | Captures the short-term relationship between previous weeks' fatalities and the current outcome, accounting for recent conflict intensity. | Potential skewness in fatality counts; log transformation helps reduce the influence of extreme values. |
| `fatalities_lag_w12_log` | Captures medium-term conflict intensity by accounting for fatalities over the preceding 12 weeks. | Potential overlap and multicollinearity with the 4-week lag variable. With a r = 0.91|
| `population_exposure_lag_w4` | lagged so only past information is used | Highly correlated with fatalities|
| `admin1` (fixed effects) | Controls for time-invariant differences across first-level administrative regions, accounting for unobserved geographical and regional characteristics, we know from EDA that Borno accounts for majority of the conflict fatalities | Increases model complexity; does not account for time-varying regional characteristics. |
| `is_election_year` | Excluded due to lack of relationship in EDA, negative relationship with conflict/fatalities events|

- Note: Other features from other data sets such as demographic data from WorldPop and national census data to get population levels at admin 1 level should also be added, due to lack of time, these features already existing in ACLED data was used. 


## Model Comparison
 
| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| Validation ROC-AUC | 0.834 | 0.836 | 0.839 |
| Validation Average Precision | 0.638 | 0.637 | 0.635 |
| Validation Precision | 0.587 | 0.567 | 0.540 |
| Validation Recall | 0.593 | 0.599 | 0.617 |
| Test ROC-AUC | 0.883 | 0.891 | 0.893 |
| Test Average Precision | 0.779 | 0.795 | 0.797 |
| Test Precision | 0.749 | 0.712 | 0.689 |
| Test Recall | 0.660 | 0.712 | 0.723 |
| Test F1-score | 0.702 | 0.712 | 0.706 |
| Selected threshold | 0.275 | 0.28 | 0.25 |


 
## Final Model Selection
 
**Logistic Regression** was selected as the deployment model.
 
- Although XGBoost and Random Forest achieved slightly higher test ROC-AUC and Average Precision, all three models performed broadly comparably.
- Logistic Regression offers a strong balance of predictive performance, interpretability and operational simplicity. It also achieved the highest test precision (0.749), reducing the proportion of false-positive alerts relative to the two tree-based models at their selected thresholds.
- Given the importance of transparency and analyst oversight in humanitarian early warning, the marginal gains from the more complex models were not considered sufficient. 


# Requirements
Python 3.13 (matches the Docker image)
scikit-learn 1.9.1. The pickled model must be loaded with the same scikit-learn version it was trained with, otherwise predictions fail (for example with 'LogisticRegression' object has no attribute 'multi_class').

All dependencies are pinned in `conflict-prediction-api/requirements.txt`.
 
## Running the API Locally
 
```bash
cd conflict-prediction-api
python3.13 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload
```
 
Open <http://127.0.0.1:8000/docs> for the interactive API page.
 
### Example request
 
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"admin1": "Abia"}'
```
 
### Example response
 
```json
{
  "admin1": "Abia",
  "observation_week": "2026-10-26",
  "probability": 0.3124,
  "predicted_conflict": 1,
  "threshold": 0.275
}
```
 
The values above are illustrative. `predicted_conflict` is 1 when `probability` is at or above `threshold`.
### Endpoints
 
| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict conflict for a state using its latest observation. Body: `{"admin1": "<state>"}` |
 
Error responses: `404` if the state isn't in the data, `422` for invalid input or missing lagged feature values.



## Getting Started
 
```bash
pip install -r requirements.txt
```
 
Run the notebooks in order (`01_...` then `02_...`).
 
To run the API locally:
 
```bash
cd conflict-prediction-api
uvicorn main:app --reload
```
 
To build and run with Docker:
 
```bash
cd conflict-prediction-api
docker build -t conflict-prediction-api .
docker run -p 8000:8000 conflict-prediction-api
```
 
To run the tests:
 
```bash
pytest tests/
```

## On Azure Deployment
- The API is also deplohed as a container on Azure Container Apps, builds the docker image from the Dockerfile in the cloud. 
- Live API: https://conflict-api.ashypebble-e398be07.uksouth.azurecontainerapps.io/docs#/default/predict_predict_post

 
### Prerequisites
 
- An Azure account with an active subscription (a personal Microsoft account without a subscription can't use Azure Cloud Shell)
- The [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), or [Azure Cloud Shell](https://portal.azure.com) in the browser (upload the `conflict-prediction-api/` folder, including the `.pkl` and `.csv` files, and exclude `venv/`)

## Deploy
 
Run these from inside `conflict-prediction-api/`:
 
```bash
# One-time setup
az login
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az group create --name conflict-api-rg --location uksouth
 
# Build in Azure and deploy
az containerapp up \
  --name conflict-api \
  --resource-group conflict-api-rg \
  --location uksouth \
  --environment conflict-env \
  --source . \
  --ingress external \
  --target-port 8000
```
 
Get the public address:
 
```bash
az containerapp show --name conflict-api --resource-group conflict-api-rg \
  --query properties.configuration.ingress.fqdn -o tsv
```
 
Test it:
 
```bash
curl -X POST https://<app-address>/predict \
  -H "Content-Type: application/json" \
  -d '{"admin1": "Abia"}'
```
 
### Redeploying
 
After changing the code, model or data, re-run the `az containerapp up` command above.
 
### Viewing logs
 
```bash
az containerapp logs show --name conflict-api --resource-group conflict-api-rg --follow
```
 
### Cleaning up
 
Delete everything created for the deployment to stop charges:
 
```bash
az group delete --name conflict-api-rg
```
 
### Notes
 
- The model and `cleaned_data.csv` are baked into the image, so new data requires a rebuild and redeploy.
- The endpoint is public and has no authentication. Add an API key or restrict access before using it beyond a demo.
- Resources are billed by usage. Check current Azure pricing and consider setting a budget alert.
## Running the Tests
 
From inside `conflict-prediction-api/`, with the virtual environment active:
 
```bash
python -m pip install pytest httpx
python -m pytest tests/ -v
```
 
Use `python -m pytest` (not bare `pytest`) so the virtual environment's packages are used. The tests use the real model and dataset, so both files must be present next to `main.py`.

 
## Data
 
Data comes from ACLED. Please review ACLED's terms of use before redistributing the raw data.
 
