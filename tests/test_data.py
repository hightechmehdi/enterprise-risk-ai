# tests/test_data.py

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    load_data,
    enforce_numeric_types,
    clean_data,
    split_data,
)

CSV_PATH = "data/raw/data.csv"
TARGET = "Bankrupt?"


@pytest.fixture(scope="module")
def raw_df():
    return load_data(CSV_PATH)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    df = enforce_numeric_types(raw_df)
    df = clean_data(df)
    return df


@pytest.fixture(scope="module")
def split_dataset(clean_df):
    return split_data(clean_df)


def test_dataset_not_empty(raw_df):
    assert len(raw_df) > 0


def test_target_exists(raw_df):
    assert TARGET in raw_df.columns


def test_target_is_binary(clean_df):
    values = set(clean_df[TARGET].unique())
    assert values.issubset({0, 1})


def test_no_missing_values(clean_df):
    assert clean_df.isna().sum().sum() == 0


def test_no_infinite_values(clean_df):
    numeric_df = clean_df.select_dtypes(include=np.number)
    assert np.isfinite(numeric_df.to_numpy()).all()


def test_all_features_are_numeric(clean_df):
    features = clean_df.drop(columns=[TARGET])

    non_numeric = [
        col
        for col in features.columns
        if not pd.api.types.is_numeric_dtype(features[col])
    ]

    assert non_numeric == [], (
        f"Non-numeric columns found: {non_numeric}"
    )


def test_no_duplicates(clean_df):
    assert clean_df.duplicated().sum() == 0


def test_train_test_same_features(split_dataset):
    X_train, X_test, _, _ = split_dataset

    assert list(X_train.columns) == list(X_test.columns)


def test_train_test_no_nan(split_dataset):
    X_train, X_test, _, _ = split_dataset

    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_train_test_no_inf(split_dataset):
    X_train, X_test, _, _ = split_dataset

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_test.to_numpy()).all()


def test_split_is_stratified(split_dataset):
    _, _, y_train, y_test = split_dataset

    train_rate = y_train.mean()
    test_rate = y_test.mean()

    assert abs(train_rate - test_rate) < 0.01


def test_no_constant_features(split_dataset):
    X_train, _, _, _ = split_dataset

    constant_features = [
        col
        for col in X_train.columns
        if X_train[col].nunique() <= 1
    ]

    assert constant_features == [], (
        f"Constant features found: {constant_features}"
    )