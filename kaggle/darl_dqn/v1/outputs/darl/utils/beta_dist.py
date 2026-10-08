import numpy as np
import pandas as pd
import scipy.stats as stats


def beta_distribution(data_clean: pd.Series, EPS: float):
    """
    Ajusta una distribución Beta a cada columna numérica del DataFrame
    y devuelve los parámetros alpha, beta, datos limpios, datos escalados,
    mínimo y máximo original.
    """
    col_min, col_max = float(data_clean.min()), float(data_clean.max())
    data_scaled = (data_clean - col_min) / (col_max - col_min + EPS)
    data_scaled = np.clip(data_scaled, EPS, 1.0 - EPS)
    α, β, _, _ = stats.beta.fit(data_scaled, floc=0, fscale=1)
    return α, β, data_clean, data_scaled, col_min, col_max
