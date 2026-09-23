# tests/test_monitoring.py
import contextlib
import io

import pytest

from src.data.features import SELECTED_FEATURES
from src.data.preprocess import prepare_dataset
from src.monitoring.monitor import CSV_PATH, run_drift_report, simulate_liquidity_shock


@pytest.fixture(scope="module")
def splits():
    with contextlib.redirect_stdout(io.StringIO()):
        X_train, X_test, _, _ = prepare_dataset(CSV_PATH, features=SELECTED_FEATURES)
    return X_train, X_test


def test_no_global_drift_on_normal_batch(splits):
    X_train, X_test = splits
    summary = run_drift_report(X_train, X_test, "test_normal")
    assert summary["dataset_drift"] is False


def test_global_drift_on_liquidity_shock(splits):
    X_train, X_test = splits
    summary = run_drift_report(X_train, simulate_liquidity_shock(X_test), "test_shock")
    assert summary["dataset_drift"] is True
    for col in ["Quick Ratio", "Quick Assets/Current Liability", "Borrowing dependency"]:
        assert summary["column_scores"][col] < 0.05


def test_shock_keeps_the_feature_contract(splits):
    _, X_test = splits
    shocked = simulate_liquidity_shock(X_test)
    assert list(shocked.columns) == SELECTED_FEATURES
    assert len(shocked) == len(X_test)


def test_prod_batch_files_reproduce_the_demo_results(splits, tmp_path, monkeypatch):
    import pandas as pd
    import src.monitoring.monitor as monitor

    X_train, X_test = splits
    monkeypatch.setattr(monitor, "BATCH_DIR", tmp_path)
    normal_path, shock_path = monitor.generate_prod_batches(X_test)

    normal = pd.read_csv(normal_path)
    assert list(normal.columns[:2]) == ["company_id", "scoring_month"]
    assert "Bankrupt?" not in normal.columns

    s_normal = run_drift_report(X_train, normal[SELECTED_FEATURES], "test_normal_file")
    s_shock = run_drift_report(X_train, pd.read_csv(shock_path)[SELECTED_FEATURES], "test_shock_file")
    assert (s_normal["drifted_columns"], s_normal["dataset_drift"]) == (1, False)
    assert (s_shock["drifted_columns"], s_shock["dataset_drift"]) == (4, True)