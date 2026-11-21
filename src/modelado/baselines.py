"""Baselines: predecir la demanda con un rezago que ya esta en la tabla.

- **estacional**: y_pred(t) = lag_168 = demanda de la misma hora de la semana
  anterior. Existe en r24 y en r120, y es la referencia de la mejora de MAPE.
- **diario**: y_pred(t) = lag_24 = demanda de la misma hora del dia anterior.
  Solo en r24: en r120 la demanda de t - 24 h aun no se conoce al predecir, y la
  tabla no tiene lag_24.

Que baseline aplica se decide por las columnas de X, no por el nombre de la
tabla: si una version futura gana o pierde un rezago, los baselines lo siguen.

Son estimadores de scikit-learn (`fit`/`predict`, `get_params`, `clone`), asi
que se miden, se serializan y se comparan igual que cualquier modelo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from modelado.datos import registro_operacion

# nombre -> columna de X que se usa como prediccion. El orden importa: la
# referencia va primero para poder calcular la mejora de los demas.
BASELINES = {"estacional": "lag_168", "diario": "lag_24"}
BASELINE_REFERENCIA = "estacional"
DESCRIPCIONES = {
    "estacional": "y_pred(t) = demanda(t - 168 h): la misma hora de la semana anterior",
    "diario": "y_pred(t) = demanda(t - 24 h): la misma hora del dia anterior",
}


class ErrorBaseline(Exception):
    """El baseline no se puede aplicar a estos datos."""


class BaselineRezago(RegressorMixin, BaseEstimator):
    """Predice con una columna de rezago de X: y_pred = X[columna].

    `fit` no aprende nada: comprueba que la columna existe y guarda las
    variables vistas, como cualquier estimador ajustado.
    """

    def __init__(self, columna: str = "lag_168"):
        self.columna = columna

    def _columna_de(self, X: pd.DataFrame) -> pd.Series:
        if not isinstance(X, pd.DataFrame):
            raise ErrorBaseline("X tiene que ser un DataFrame: el baseline busca su columna por nombre.")
        if self.columna not in X.columns:
            raise ErrorBaseline(f"X no tiene la columna {self.columna!r}.")
        return X[self.columna]

    def fit(self, X: pd.DataFrame, y=None) -> BaselineRezago:
        self._columna_de(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        return self._columna_de(X).to_numpy(dtype="float64")


def baselines_disponibles(variables: list[str]) -> tuple[dict[str, BaselineRezago], dict[str, Any]]:
    """Los baselines cuya columna esta entre `variables`, con la referencia primero.

    Devuelve ({nombre: estimador sin ajustar}, registro). El registro anota cada
    baseline omitido y por que.
    """
    disponibles: dict[str, BaselineRezago] = {}
    omitidos: dict[str, str] = {}
    for nombre, columna in BASELINES.items():
        if columna in variables:
            disponibles[nombre] = BaselineRezago(columna)
        else:
            omitidos[nombre] = (
                f"{columna} no esta entre las variables de la tabla: con un "
                f"rezago_minimo_h mayor que el de {columna} esa demanda aun no se "
                "conoce al predecir"
            )
    if not disponibles:
        raise ErrorBaseline(f"Ninguna de las columnas {list(BASELINES.values())} esta en X.")

    registro = registro_operacion(
        "baselines_disponibles",
        "un baseline por cada rezago de BASELINES presente en X",
        disponibles={nombre: {"columna": m.columna, "descripcion": DESCRIPCIONES[nombre]} for nombre, m in disponibles.items()},
        omitidos=omitidos,
        referencia=BASELINE_REFERENCIA if BASELINE_REFERENCIA in disponibles else None,
    )
    return disponibles, registro
