"""
drift_framework/data/loader.py

TableShift wrappers for loading Adult (Income) and Diabetes Readmission datasets.
Returns raw (unprocessed) pandas DataFrames split into reference, pre-drift, and
post-drift sets for use with the custom two-stage pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

# tableshift imports — install with:
#   pip install --no-deps git+https://github.com/mlfoundations/tableshift.git
#   pip install ray  # required at runtime
from tableshift import get_dataset
from tableshift.core.features import PreprocessorConfig

SEED = 42

# Maps thesis split roles to tableshift split names, per dataset.
# adult uses FixedSplitter  → splits: train / validation / test
# diabetes_readmission uses DomainSplitter → splits: train / validation / id_test / ood_test
_SPLIT_MAP: Dict[str, Dict[str, str]] = {
    "adult": {
        "reference": "train",
        "pre_drift": "validation",
        "post_drift": "test",
    },
    "diabetes_readmission": {
        "reference": "train",
        "pre_drift": "id_test",
        "post_drift": "ood_test",
    },
}


@dataclass
class DataBundle:
    """Container for one dataset's three splits plus feature-type metadata."""

    name: str
    X_ref: pd.DataFrame
    y_ref: pd.Series
    X_pre: pd.DataFrame
    y_pre: pd.Series
    X_post: pd.DataFrame
    y_post: pd.Series
    num_features: list[str]
    cat_features: list[str]


def _passthrough_preprocessor() -> PreprocessorConfig:
    """Return a PreprocessorConfig that skips all tableshift transformations.

    Allows the custom two-stage pipeline to apply its own preprocessing
    (QuantileTransformer / TargetEncoder / StandardScaler) on raw values.
    """
    return PreprocessorConfig(
        categorical_features="passthrough",
        numeric_features="passthrough",
        dropna="all",
    )


def _detect_feature_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) inferred from column dtypes."""
    cat_cols = [
        c for c in X.columns
        if isinstance(X[c].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(X[c])
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def load_dataset(name: str, cache_dir: str = "tableshift_cache") -> DataBundle:
    """Load a tableshift dataset and return reference, pre-drift, and post-drift splits.

    Args:
        name: tableshift dataset identifier. Supported: ``"adult"``,
              ``"diabetes_readmission"``.
        cache_dir: local directory where tableshift caches downloaded files.

    Returns:
        DataBundle with X/y for each of the three splits and feature-type lists.

    Raises:
        ValueError: if ``name`` is not a supported dataset.
    """
    if name not in _SPLIT_MAP:
        raise ValueError(
            f"Dataset '{name}' not supported. Choose from: {list(_SPLIT_MAP.keys())}"
        )

    print(f"\n{'='*60}")
    print(f"Loading dataset: {name}")
    print(f"{'='*60}")

    dset = get_dataset(
        name,
        cache_dir=cache_dir,
        preprocessor_config=_passthrough_preprocessor(),
    )

    split_map = _SPLIT_MAP[name]
    bundles: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for role, split_name in split_map.items():
        X, y, _groups, _domain = dset.get_pandas(split_name)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        bundles[role] = (X, y)

    X_ref, y_ref = bundles["reference"]
    X_pre, y_pre = bundles["pre_drift"]
    X_post, y_post = bundles["post_drift"]

    num_features, cat_features = _detect_feature_types(X_ref)

    print(f"Reference split  : X={X_ref.shape},  class balance={y_ref.mean():.3f}")
    print(f"Pre-drift split  : X={X_pre.shape},  class balance={y_pre.mean():.3f}")
    print(f"Post-drift split : X={X_post.shape}, class balance={y_post.mean():.3f}")
    print(f"Numeric features  ({len(num_features)}): {num_features[:5]}{'...' if len(num_features) > 5 else ''}")
    print(f"Categorical features ({len(cat_features)}): {cat_features[:5]}{'...' if len(cat_features) > 5 else ''}")

    return DataBundle(
        name=name,
        X_ref=X_ref,
        y_ref=y_ref,
        X_pre=X_pre,
        y_pre=y_pre,
        X_post=X_post,
        y_post=y_post,
        num_features=num_features,
        cat_features=cat_features,
    )


def load_all_datasets(cache_dir: str = "tableshift_cache") -> Dict[str, DataBundle]:
    """Load both supported datasets and return a dict keyed by dataset name."""
    return {name: load_dataset(name, cache_dir) for name in _SPLIT_MAP}


if __name__ == "__main__":
    bundles = load_all_datasets()
    for name, b in bundles.items():
        print(f"\n[{name}] ref X shape: {b.X_ref.shape}")
        print(f"  num_features sample : {b.num_features[:3]}")
        print(f"  cat_features sample : {b.cat_features[:3]}")
