"""Tests for the Humanitarian Conflict Early Warning API.


"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# The API folder has a hyphen in its name, so it can't be imported as a
# package. Add it to the path so `import main` works.
API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

import main  # noqa: E402

client = TestClient(main.app)


@pytest.fixture(scope="module")
def valid_state():
    """Return a state whose latest observation has all features present."""
    df = main.data_df
    for state in df["admin1"].dropna().unique():
        latest = df[df["admin1"] == state].sort_values("week").iloc[-1]
        if not latest[main.FEATURES].isna().any():
            return state
    pytest.skip("No state with a complete latest observation in the data")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_home_returns_status():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Conflict prediction API running"}


# ---------------------------------------------------------------------------
# Successful predictions
# ---------------------------------------------------------------------------

def test_predict_valid_state(valid_state):
    response = client.post("/predict", json={"admin1": valid_state})
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {
        "admin1",
        "observation_week",
        "probability",
        "predicted_conflict",
        "threshold",
    }
    assert body["admin1"] == valid_state
    assert 0.0 <= body["probability"] <= 1.0
    assert body["predicted_conflict"] in (0, 1)
    assert body["threshold"] == main.threshold


def test_prediction_is_consistent_with_threshold(valid_state):
    body = client.post("/predict", json={"admin1": valid_state}).json()
    # Probability is rounded to 4 d.p. in the response, so allow for that
    # only when it sits right on the threshold.
    if abs(body["probability"] - body["threshold"]) > 1e-4:
        expected = int(body["probability"] >= body["threshold"])
        assert body["predicted_conflict"] == expected


def test_observation_week_is_latest_for_state(valid_state):
    df = main.data_df
    expected_week = (
        df[df["admin1"] == valid_state]["week"].max().strftime("%Y-%m-%d")
    )
    body = client.post("/predict", json={"admin1": valid_state}).json()
    assert body["observation_week"] == expected_week


def test_state_lookup_ignores_case_and_whitespace(valid_state):
    baseline = client.post("/predict", json={"admin1": valid_state}).json()

    for variant in (valid_state.lower(), valid_state.upper(), f"  {valid_state}  "):
        response = client.post("/predict", json={"admin1": variant})
        assert response.status_code == 200
        assert response.json() == baseline


def test_predict_is_deterministic(valid_state):
    first = client.post("/predict", json={"admin1": valid_state}).json()
    second = client.post("/predict", json={"admin1": valid_state}).json()
    assert first == second


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_state_returns_404():
    response = client.post("/predict", json={"admin1": "Atlantis"})
    assert response.status_code == 404
    assert "Atlantis" in response.json()["detail"]


def test_empty_state_returns_422():
    response = client.post("/predict", json={"admin1": ""})
    assert response.status_code == 422


def test_missing_field_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_wrong_type_returns_422():
    response = client.post("/predict", json={"admin1": 123})
    assert response.status_code == 422


def test_get_on_predict_not_allowed():
    response = client.get("/predict")
    assert response.status_code == 405


def test_missing_lagged_features_returns_422(monkeypatch, valid_state):
    df = main.data_df.copy()
    mask = df["admin1"] == valid_state
    latest_idx = df[mask].sort_values("week").index[-1]
    df.loc[latest_idx, main.FEATURES[0]] = float("nan")
    monkeypatch.setattr(main, "data_df", df)

    response = client.post("/predict", json={"admin1": valid_state})
    assert response.status_code == 422
    assert "Missing lagged feature values" in response.json()["detail"]