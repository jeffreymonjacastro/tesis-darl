"""
darl.types.dataclass
-------------------
Reusable dataclasses for drift metadata.

"""

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class _NumericMeta:
    col: str
    col_min: float
    col_max: float
    alpha_0: float  # Beta original shape a
    beta_0: float  # Beta original shape b
    direction: str  # high/low/central/extreme
    alpha_q: float  # Beta target shape a
    beta_q: float  # Beta target shape b
    drift_severity: float  # α mixture weight
    extra: dict = field(default_factory=dict)  # KS, PSI after transform


@dataclass
class _CatMeta:
    col: str
    p0: pd.Series  # original distribution (proportions)
    q: pd.Series  # target distribution (proportions)
    p_alpha: pd.Series  # mixed distribution
    drift_severity: float
    strategy: str
    extra: dict = field(default_factory=dict)  # PSI, JS, Hellinger, chi2


@dataclass
class _LabelMeta:
    col: str
    drift_severity: float
    p_flip: float
    before_prevalence: float
    after_prevalence: float
    extra: dict = field(default_factory=dict)
