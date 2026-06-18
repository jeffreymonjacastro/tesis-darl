import numpy as np
import pandas as pd
import scipy.stats as stats


def beta_distribution(data: pd.DataFrame, col: str):
    """
    Ajusta una distribución Beta a cada columna numérica del DataFrame
    y devuelve los parámetros alpha, beta, datos limpios, datos escalados,
    mínimo y máximo original.
    """
    data_clean = data[col].dropna()
    col_min, col_max = data_clean.min(), data_clean.max()
    data_scaled = (data_clean - col_min) / (col_max - col_min)
    data_scaled = np.clip(data_scaled, 1e-6, 1.0 - 1e-6)
    α, β, _, _ = stats.beta.fit(data_scaled, floc=0, fscale=1)
    return α, β, data_clean, data_scaled, col_min, col_max
