"""Ensembles de arboles para la demanda horaria, sobre el nivel y sobre la serie diferenciada.

    random_forest      RandomForestRegressor
    adaboost           AdaBoostRegressor sobre DecisionTreeRegressor(max_depth=PROFUNDIDAD_ADABOOST)
    gradient_boosting  GradientBoostingRegressor
    hist_gradient      HistGradientBoostingRegressor (la variante rapida)

**Por que dos formas.** Un arbol predice con medias (o medianas) de valores de
`y` vistos en entrenamiento, asi que no extrapola: no puede dar mas que el
maximo que vio. La demanda del SIN crece de un ano a otro (`crecimiento_anual`
en docs/resultados/eda.json), y la validacion es posterior al entrenamiento:
sobre el nivel, el ensemble queda por debajo justo en las horas mas altas.
Cada ensemble se entrena entonces de dos formas:

    nivel         y = demanda_mwh
    diferenciado  y = demanda_mwh - lag_168, y a la prediccion se le suma lag_168

En la forma diferenciada el arbol aprende el cambio frente a la misma hora de
la semana anterior, y el nivel lo pone `lag_168`, que si sigue el crecimiento.
`RegresorDiferenciado` envuelve esa transformacion para que el modelo guardado
prediga demanda directamente. `lag_168` sigue tambien entre las variables de X:
la diferencia no la retira.

**Que se reporta.** Ademas de las metricas habituales, `extrapolacion` cuenta
cuantas predicciones de validacion superan el maximo de `demanda_mwh` en
entrenamiento, cuantos valores reales lo superan y el sesgo medio en esas horas
y en el resto: es lo que muestra si la diferenciacion corrige la subestimacion.

**Sin preprocesado.** Los arboles son invariantes a transformaciones monotonas
de cada variable: no hay escalado ni imputacion que ajustar, y por eso no hay
Pipeline. Si alguno se anadiera, iria dentro de un Pipeline (CLAUDE.md).

**Hiperparametros.** Valores por defecto razonables, fijos en este paso
(`MODELOS`); la busqueda va despues. Dos decisiones no son de ajuste fino:

- `HistGradientBoostingRegressor(early_stopping=False)`: con `"auto"` y mas de
  10 000 filas, sklearn separa al azar un 10 % del entrenamiento para parar
  antes. Seria una particion aleatoria de una serie temporal.
- Random Forest con `oob_score=False`: el error out-of-bag no se usa como
  metrica, porque el bootstrap mezcla momentos distintos y cada arbol se
  evalua con horas vecinas de las que vio. Se mide en los pliegues de
  `TimeSeriesSplit` y en validacion, como el resto.

    from modelado.ensembles import MODELOS, FORMAS, evaluar_ensemble
    modelo, registro = evaluar_ensemble("random_forest", "diferenciado", X, y, fechas, indices, tscv, mape_base)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from modelado.baselines import BASELINE_REFERENCIA, BASELINES
from modelado.datos import registro_operacion
from modelado.metricas import mape, metricas_regresion
from modelado.tiempos import medir_ajuste, medir_prediccion, tamano_serializado

SEMILLA = 42
PROFUNDIDAD_ADABOOST = 6
# La forma diferenciada resta y vuelve a sumar el baseline estacional.
COLUMNA_DIFERENCIA = BASELINES[BASELINE_REFERENCIA]
FORMAS = ("nivel", "diferenciado")
OBJETIVOS = {
    "nivel": "y = demanda_mwh",
    "diferenciado": f"y = demanda_mwh - {COLUMNA_DIFERENCIA}; prediccion = modelo + {COLUMNA_DIFERENCIA}",
}

# Metricas que se resumen por pliegue (media y desviacion).
METRICAS_CV = (
    "mae_mwh",
    "rmse_mwh",
    "mape_pct",
    "mape_pico_pct",
    "sesgo_medio_mwh",
    "mejora_mape_pct",
    "predicciones_sobre_maximo",
    "reales_sobre_maximo",
)


class ErrorEnsemble(Exception):
    """El ensemble no se puede ajustar o evaluar con seguridad."""


@dataclass(frozen=True)
class Especificacion:
    descripcion: str
    estimador: Callable[[], Any]
    # Lo que se fija a mano, para el registro (los objetos no se serializan a JSON).
    parametros: dict[str, Any] = field(default_factory=dict)


def _random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_features=0.5,
        min_samples_leaf=10,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=SEMILLA,
    )


def _adaboost() -> AdaBoostRegressor:
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=PROFUNDIDAD_ADABOOST, random_state=SEMILLA),
        n_estimators=100,
        learning_rate=0.5,
        loss="linear",
        random_state=SEMILLA,
    )


def _gradient_boosting() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        subsample=1.0,
        loss="squared_error",
        random_state=SEMILLA,
    )


def _hist_gradient() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=31,
        early_stopping=False,
        random_state=SEMILLA,
    )


MODELOS: dict[str, Especificacion] = {
    "random_forest": Especificacion(
        "RandomForestRegressor",
        _random_forest,
        {"n_estimators": 200, "max_features": 0.5, "min_samples_leaf": 10, "bootstrap": True,
         "oob_score": False, "n_jobs": "-1 al ajustar, 1 al predecir (ver ajustar)"},
    ),
    "adaboost": Especificacion(
        f"AdaBoostRegressor sobre DecisionTreeRegressor(max_depth={PROFUNDIDAD_ADABOOST})",
        _adaboost,
        {"estimador_base": f"DecisionTreeRegressor(max_depth={PROFUNDIDAD_ADABOOST})", "n_estimators": 100,
         "learning_rate": 0.5, "loss": "linear"},
    ),
    "gradient_boosting": Especificacion(
        "GradientBoostingRegressor",
        _gradient_boosting,
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 4, "subsample": 1.0, "loss": "squared_error"},
    ),
    "hist_gradient": Especificacion(
        "HistGradientBoostingRegressor",
        _hist_gradient,
        {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31, "early_stopping": False},
    ),
}

NOTA_OOB = (
    "El error out-of-bag de Random Forest no se usa: el bootstrap mezcla momentos "
    "distintos y cada arbol se evaluaria con horas vecinas de las que vio. Se mide "
    "con TimeSeriesSplit y en validacion."
)


class RegresorDiferenciado(RegressorMixin, BaseEstimator):
    """Ajusta `estimador` sobre y - X[columna] y predice estimador.predict(X) + X[columna].

    X tiene que ser un DataFrame con `columna`; la columna se queda tambien
    como variable del estimador.
    """

    def __init__(self, estimador=None, columna: str = COLUMNA_DIFERENCIA):
        self.estimador = estimador
        self.columna = columna

    def _base(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise ErrorEnsemble("X tiene que ser un DataFrame: la diferencia busca su columna por nombre.")
        if self.columna not in X.columns:
            raise ErrorEnsemble(f"X no tiene la columna {self.columna!r} con la que se diferencia.")
        base = X[self.columna].to_numpy(dtype="float64")
        if np.isnan(base).any():
            raise ErrorEnsemble(f"{self.columna!r} tiene NaN: no se puede diferenciar.")
        return base

    def fit(self, X: pd.DataFrame, y) -> RegresorDiferenciado:
        if self.estimador is None:
            raise ErrorEnsemble("Falta el estimador que aprende la diferencia.")
        diferencia = np.asarray(y, dtype="float64") - self._base(X)
        self.estimador_ = clone(self.estimador).fit(X, diferencia)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        return np.asarray(self.estimador_.predict(X), dtype="float64") + self._base(X)


def _especificacion(nombre: str) -> Especificacion:
    if nombre not in MODELOS:
        raise ErrorEnsemble(f"Ensemble desconocido: {nombre!r}. Disponibles: {list(MODELOS)}")
    return MODELOS[nombre]


def crear_modelo(nombre: str, forma: str = "nivel") -> BaseEstimator:
    """Ensemble sin ajustar, sobre el nivel o envuelto en RegresorDiferenciado."""
    estimador = _especificacion(nombre).estimador()
    if forma == "nivel":
        return estimador
    if forma == "diferenciado":
        return RegresorDiferenciado(estimador, COLUMNA_DIFERENCIA)
    raise ErrorEnsemble(f"Forma desconocida: {forma!r}. Disponibles: {list(FORMAS)}")


def estimador_interno(modelo: BaseEstimator) -> BaseEstimator:
    """El ensemble de sklearn que hay dentro, en cualquiera de las dos formas."""
    return modelo.estimador_ if isinstance(modelo, RegresorDiferenciado) else modelo


def ajustar(modelo: BaseEstimator, X: pd.DataFrame, y) -> BaseEstimator:
    """`fit` y despues prediccion en un solo hilo, para que sea reproducible bit a bit.

    Random Forest ajusta en paralelo (`n_jobs=-1`, mismo resultado con la
    semilla fija), pero al predecir en paralelo suma los arboles en el orden en
    que acaban los hilos y la ultima cifra decimal cambia de una llamada a otra.
    """
    return ajustar_prediccion_en_un_hilo(modelo.fit(X, y))


def ajustar_prediccion_en_un_hilo(modelo: BaseEstimator) -> BaseEstimator:
    """Pone `n_jobs=1` en el ensemble ya ajustado (ver `ajustar`)."""
    interno = estimador_interno(modelo)
    if "n_jobs" in interno.get_params():
        interno.set_params(n_jobs=1)
    return modelo


# --------------------------------------------------------------------------
# Extrapolacion: lo que el arbol no puede ver
# --------------------------------------------------------------------------


def extrapolacion(y_true, y_pred, maximo_entrenamiento: float) -> dict[str, Any]:
    """Predicciones y valores reales por encima del maximo de entrenamiento, y el sesgo en cada lado.

    `sesgo_*` es pred - real (negativo: subestima), separado entre las horas
    cuyo valor real supera el maximo de entrenamiento y las demas.
    """
    real = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    if real.shape != pred.shape or real.size == 0:
        raise ErrorEnsemble("y_true e y_pred tienen que tener la misma longitud, no nula.")
    sobre = real > maximo_entrenamiento
    error = pred - real

    def _sesgo(mascara: np.ndarray) -> float | None:
        return float(error[mascara].mean()) if mascara.any() else None

    return {
        "maximo_entrenamiento_mwh": float(maximo_entrenamiento),
        "n": int(real.size),
        "predicciones_sobre_maximo": int((pred > maximo_entrenamiento).sum()),
        "proporcion_predicciones_sobre_maximo": float((pred > maximo_entrenamiento).mean()),
        "reales_sobre_maximo": int(sobre.sum()),
        "proporcion_reales_sobre_maximo": float(sobre.mean()),
        "prediccion_maxima_mwh": float(pred.max()),
        "real_maximo_mwh": float(real.max()),
        "sesgo_medio_reales_sobre_maximo_mwh": _sesgo(sobre),
        "sesgo_medio_resto_mwh": _sesgo(~sobre),
    }


def importancias(modelo: BaseEstimator, variables: list[str]) -> dict[str, Any] | None:
    """Importancia por reduccion de impureza, si el ensemble la tiene (HistGB no).

    Es la de sklearn, sesgada hacia variables continuas con muchos cortes:
    sirve para orientarse, no para concluir. La interpretacion va con SHAP.
    """
    interno = estimador_interno(modelo)
    try:
        valores = np.asarray(interno.feature_importances_, dtype="float64")
    except AttributeError:
        return None
    orden = np.argsort(valores)[::-1]
    return {
        "criterio": "feature_importances_ de sklearn (reduccion media de impureza)",
        "por_variable": {variables[i]: float(valores[i]) for i in orden},
    }


# --------------------------------------------------------------------------
# Validacion cruzada temporal
# --------------------------------------------------------------------------


def metricas_por_pliegue(
    nombre: str,
    forma: str,
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    tscv: TimeSeriesSplit,
) -> dict[str, Any]:
    """Metricas y extrapolacion en cada pliegue de `tscv`, con media y desviacion (ddof=1).

    Cada pliegue ajusta un modelo nuevo solo con su tramo de entrenamiento, y
    el maximo de referencia es el de ese tramo. La mejora de MAPE es frente al
    baseline estacional del mismo pliegue.
    """
    columna_base = BASELINES[BASELINE_REFERENCIA]
    pliegues = []
    inicio = time.perf_counter()
    for numero, (entrena, valida) in enumerate(tscv.split(X), start=1):
        modelo = ajustar(crear_modelo(nombre, forma), X.iloc[entrena], y.iloc[entrena])
        y_valida = y.iloc[valida]
        prediccion = modelo.predict(X.iloc[valida])
        metricas = metricas_regresion(
            y_valida, prediccion, fechas.iloc[valida],
            mape_baseline=mape(y_valida, X[columna_base].iloc[valida]),
        )
        tope = extrapolacion(y_valida, prediccion, float(y.iloc[entrena].max()))
        pliegues.append({
            "pliegue": numero,
            **metricas,
            "predicciones_sobre_maximo": tope["predicciones_sobre_maximo"],
            "reales_sobre_maximo": tope["reales_sobre_maximo"],
        })
    segundos = time.perf_counter() - inicio

    resumen = {}
    for clave in METRICAS_CV:
        valores = np.array([p[clave] for p in pliegues], dtype="float64")
        resumen[clave] = {
            "media": float(valores.mean()),
            "desviacion": float(valores.std(ddof=1)) if len(valores) > 1 else None,
            "por_pliegue": [float(v) for v in valores],
        }
    return registro_operacion(
        "metricas_por_pliegue",
        "un modelo nuevo por pliegue ajustado con su tramo de entrenamiento; media y "
        "desviacion (ddof=1) entre pliegues; maximo de referencia el del tramo de "
        f"entrenamiento del pliegue; mejora frente al baseline {BASELINE_REFERENCIA} del mismo pliegue",
        n_pliegues=len(pliegues),
        resumen=resumen,
        pliegues=pliegues,
        segundos=segundos,
    )


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------


def evaluar_ensemble(
    nombre: str,
    forma: str,
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    indices: dict[str, np.ndarray],
    tscv: TimeSeriesSplit,
    mape_baseline_validacion: float | None = None,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """CV en entrenamiento, ajuste con todo el entrenamiento y evaluacion en validacion.

    Solo usa `indices["entrenamiento"]` para ajustar y `indices["validacion"]`
    para evaluar; el bloque de prueba no entra. Devuelve (modelo ajustado, registro).
    """
    especificacion = _especificacion(nombre)
    if forma not in FORMAS:
        raise ErrorEnsemble(f"Forma desconocida: {forma!r}. Disponibles: {list(FORMAS)}")
    entrena, valida = indices["entrenamiento"], indices["validacion"]
    if len(entrena) == 0 or len(valida) == 0 or entrena.max() >= valida.min():
        raise ErrorEnsemble("El entrenamiento tiene que ser no vacio y anterior a la validacion.")
    X_entrena = X.iloc[entrena].reset_index(drop=True)
    y_entrena = y.iloc[entrena].reset_index(drop=True)
    fechas_entrena = fechas.iloc[entrena].reset_index(drop=True)

    validacion_cruzada = metricas_por_pliegue(nombre, forma, X_entrena, y_entrena, fechas_entrena, tscv)

    modelo, ajuste = medir_ajuste(crear_modelo(nombre, forma), X_entrena, y_entrena)
    ajustar_prediccion_en_un_hilo(modelo)
    prediccion, medida = medir_prediccion(modelo, X.iloc[valida])
    y_valida = y.iloc[valida]
    metricas = metricas_regresion(y_valida, prediccion, fechas.iloc[valida], mape_baseline=mape_baseline_validacion)

    registro = {
        "descripcion": especificacion.descripcion,
        "forma": forma,
        "objetivo": OBJETIVOS[forma],
        "hiperparametros": especificacion.parametros,
        "semilla": SEMILLA,
        "validacion_cruzada": validacion_cruzada,
        "metricas_validacion": metricas,
        "mejora_frente_a": BASELINE_REFERENCIA if mape_baseline_validacion is not None else None,
        "extrapolacion_validacion": extrapolacion(y_valida, prediccion, float(y_entrena.max())),
        "tiempos": {
            "validacion_cruzada_s": validacion_cruzada["detalle"]["segundos"],
            "ajuste_s": ajuste["detalle"]["segundos"],
            "prediccion_ms_por_1000_filas": medida["detalle"]["ms_por_1000_filas"],
            "repeticiones_prediccion": medida["detalle"]["repeticiones"],
            "tamano_serializado_bytes": tamano_serializado(modelo),
        },
        "importancias": importancias(modelo, list(X.columns)),
        "filas_ajuste": int(len(entrena)),
    }
    if nombre == "random_forest":
        registro["nota_oob"] = NOTA_OOB
    return modelo, registro


def comparar_formas(registros: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Nivel frente a diferenciado de un ensemble: sesgo, MAPE y predicciones sobre el maximo."""
    filas = {}
    for forma in FORMAS:
        m = registros[forma]["metricas_validacion"]
        e = registros[forma]["extrapolacion_validacion"]
        filas[forma] = {
            "mape_pct": m["mape_pct"],
            "mape_pico_pct": m["mape_pico_pct"],
            "rmse_mwh": m["rmse_mwh"],
            "sesgo_medio_mwh": m["sesgo_medio_mwh"],
            "sesgo_medio_pct": m["sesgo_medio_pct"],
            "predicciones_sobre_maximo": e["predicciones_sobre_maximo"],
            "sesgo_medio_reales_sobre_maximo_mwh": e["sesgo_medio_reales_sobre_maximo_mwh"],
        }
    nivel, dif = filas["nivel"], filas["diferenciado"]
    return {
        **filas,
        "reales_sobre_maximo": registros["nivel"]["extrapolacion_validacion"]["reales_sobre_maximo"],
        "diferenciado_reduce_sesgo": abs(dif["sesgo_medio_mwh"]) < abs(nivel["sesgo_medio_mwh"]),
        "diferenciado_reduce_mape": dif["mape_pct"] < nivel["mape_pct"],
    }


__all__ = [
    "FORMAS",
    "MODELOS",
    "RegresorDiferenciado",
    "comparar_formas",
    "ajustar",
    "crear_modelo",
    "estimador_interno",
    "evaluar_ensemble",
    "extrapolacion",
    "importancias",
    "metricas_por_pliegue",
]
