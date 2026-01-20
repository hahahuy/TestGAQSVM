from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


@dataclass
class PCACache:
    # maps n_components -> fitted PCA
    pca_by_k: Dict[int, PCA] = field(default_factory=dict)


@dataclass
class DataStore:
    credit: DatasetSplit | None = None
    wine: DatasetSplit | None = None
    pca_credit: PCACache = field(default_factory=PCACache)
    pca_wine: PCACache = field(default_factory=PCACache)


def preprocess_credit(
    csv_path: str | Path,
    *,
    sample_frac: float = 0.5,
    random_state: int = 42,
    test_size: float = 0.3,
) -> DatasetSplit:
    df = pd.read_csv(Path(csv_path))
    df = df.sample(frac=sample_frac, random_state=random_state)

    X = df.drop(columns=["Default"])
    y = df["Default"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return DatasetSplit(
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        scaler=scaler,
    )


def preprocess_wine(*, random_state: int = 42, test_size: float = 0.2) -> DatasetSplit:
    wine = load_wine()
    X = wine.data[wine.target != 2]
    y = wine.target[wine.target != 2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return DatasetSplit(
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        scaler=scaler,
    )


def _apply_pca_cached(
    split: DatasetSplit,
    cache: PCACache,
    *,
    num_feature: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test = split.X_train, split.X_test

    if num_feature >= X_train.shape[1]:
        return X_train, split.y_train, X_test, split.y_test

    if num_feature not in cache.pca_by_k:
        pca = PCA(n_components=num_feature)
        pca.fit(X_train)
        cache.pca_by_k[num_feature] = pca

    pca = cache.pca_by_k[num_feature]
    return pca.transform(X_train), split.y_train, pca.transform(X_test), split.y_test


def generate_wine(store: DataStore, num_feature: int):
    if store.wine is None:
        raise RuntimeError("Wine dataset not preprocessed. Call preprocess_all().")
    return _apply_pca_cached(store.wine, store.pca_wine, num_feature=num_feature)


def generate_credit(store: DataStore, num_feature: int):
    if store.credit is None:
        raise RuntimeError("Credit dataset not preprocessed. Call preprocess_all().")
    return _apply_pca_cached(store.credit, store.pca_credit, num_feature=num_feature)


def preprocess_all(
    *,
    repo_root: str | Path,
    credit_csv: str = "germancredit_data_updated.csv",
) -> DataStore:
    root = Path(repo_root)
    store = DataStore()
    store.credit = preprocess_credit(root / credit_csv)
    store.wine = preprocess_wine()
    return store


