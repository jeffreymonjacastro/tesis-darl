"""Structured calibrated drift report."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DriftReport:
    """Observable drift evidence plus hidden labels reserved for evaluation."""

    p_covariate: float
    p_concept: float
    p_both: float
    severity_score: float
    confidence: float
    delta_auc: float
    delta_logloss: float
    psi_scores: dict[str, float]
    ks_scores: dict[str, float]
    c2st_auc: float
    true_drift_type: str | None = None
    window_idx: int = 0
    n_samples: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def max_psi(self) -> float:
        """Return largest feature PSI."""
        return max(self.psi_scores.values(), default=0.0)

    @property
    def has_drift(self) -> bool:
        """Return whether calibrated evidence favors any drift type."""
        return max(self.p_covariate, self.p_concept, self.p_both) >= 0.5

    @property
    def feature_psi(self) -> dict[str, float]:
        """Compatibility alias for numeric PSI values."""
        return self.psi_scores

    @property
    def feature_ks_pval(self) -> dict[str, float]:
        """Compatibility alias for KS p-values."""
        return self.ks_scores

    @property
    def cat_feature_psi(self) -> dict[str, float]:
        """Compatibility alias retained for earlier notebooks."""
        return self.metadata.get("categorical_psi", {})

    def get_state_vector(self, important_features: list[str]) -> list[float]:
        """Return compact compatibility state used by earlier demos."""
        return [self.c2st_auc, self.max_psi] + [
            self.psi_scores.get(feature, self.cat_feature_psi.get(feature, 0.0))
            for feature in important_features
        ]
