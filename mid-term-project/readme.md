# Predicting conflict severity in Nigeria
---
This notebook goes through a pipline predicting severity of political violence at subnational level in Nigeria using historical features in ACLED. The selected model is deployed as a containarised API. 


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
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
└── README.md
```

## Workflow
 
1. **Data cleaning and EDA** (`01_data_cleaning_EDA.ipynb`): cleans the raw ACLED export (events from 1997-01-01 to 2026-10-30) and explores patterns in political violence. Output is saved to `data/processed/cleaned_data.csv`.
2. **Modelling and deployment** (`02_modelling_deployment.ipynb`): trains and compares Logistic Regression, Random Forest and XGBoost, selects a decision threshold for each, and exports the chosen model to `models/`.
3. **API** (`conflict-prediction-api/`): serves the selected model for predictions, with request/response schemas in `schemas.py` and a Docker setup for deployment.
4. **Tests** (`tests/test_api.py`): tests for the API.


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
- Given the importance of transparency and analyst oversight in humanitarian early warning, the marginal gains from the more complex models were not considered sufficient to justify their added complexity.


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
 
## Data
 
Data comes from ACLED. Please review ACLED's terms of use before redistributing the raw data.
 
