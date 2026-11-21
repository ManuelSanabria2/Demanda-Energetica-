"""Clasificadores de alerta_pico: cada uno es un Pipeline StandardScaler + modelo.

    logistica    LogisticRegression(class_weight="balanced")
    knn          KNeighborsClassifier
    svm_lineal   LinearSVC(class_weight="balanced"), sobre todo el entrenamiento
    svm_rbf      SVC(kernel="rbf", class_weight="balanced"), sobre una submuestra
                 temporal del entrenamiento (una de cada SUBMUESTRA_RBF horas)

y un baseline: alerta si lag_168 supera el mismo umbral de la fila.

**Seleccion.** Rejillas pequenas (C, n_neighbors, weights) con GridSearchCV
sobre el Pipeline completo y el TimeSeriesSplit del entrenamiento, eligiendo por
**PR-AUC** (average precision). No por accuracy: con ~12 % de positivos, decir
siempre "no hay alerta" ya acierta ~88 %. Tambien se registra el F1 de cada
candidato.

**SVC con RBF.** Su ajuste crece entre cuadratica y cubicamente con las filas:
con las ~32 000 del entrenamiento, una rejilla con 5 pliegues tardaria horas.
Se entrena con una de cada `SUBMUESTRA_RBF` horas, conservando el orden. El paso
es 5 porque es coprimo con 168: la submuestra recorre todas las horas de la
semana (con 6 o 7 se quedaria con una fraccion de las combinaciones hora x dia).
El TimeSeriesSplit de la submuestra lleva `gap = ceil(rezago / paso)` filas, que
cubren al menos el mismo rezago en horas. El registro anota el motivo, las filas
y los segundos.

**Puntajes.** `predict_proba` donde existe (umbral 0.5) y `decision_function`
en las SVM (umbral 0). La validacion nunca entra en el ajuste.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from modelado.baselines import BASELINES, BASELINE_REFERENCIA
from modelado.datos import registro_operacion, rezago_de
from modelado.metricas import metricas_clasificacion
from modelado.particion import validacion_cruzada
from modelado.tiempos import medir_ajuste, medir_prediccion, tamano_serializado

SEMILLA = 42
SUBMUESTRA_RBF = 5
CRITERIO = "pr_auc"
PUNTUACIONES = {"pr_auc": "average_precision", "f1": "f1"}


class ErrorClasificacion(Exception):
    """El clasificador no se puede ajustar o evaluar con seguridad."""


@dataclass(frozen=True)
class Especificacion:
    descripcion: str
    estimador: Callable[[], Any]
    rejilla: dict[str, tuple] = field(default_factory=dict)
    submuestra: int = 1


MODELOS: dict[str, Especificacion] = {
    "logistica": Especificacion(
        'StandardScaler + LogisticRegression(class_weight="balanced")',
        lambda: LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEMILLA),
        rejilla={"C": (0.01, 0.1, 1.0, 10.0, 100.0)},
    ),
    "knn": Especificacion(
        "StandardScaler + KNeighborsClassifier",
        KNeighborsClassifier,
        rejilla={"n_neighbors": (5, 15, 45, 135), "weights": ("uniform", "distance")},
    ),
    "svm_lineal": Especificacion(
        'StandardScaler + LinearSVC(class_weight="balanced"), todo el entrenamiento',
        lambda: LinearSVC(class_weight="balanced", dual="auto", max_iter=50_000, random_state=SEMILLA),
        rejilla={"C": (0.01, 0.1, 1.0, 10.0)},
    ),
    "svm_rbf": Especificacion(
        f'StandardScaler + SVC(kernel="rbf", class_weight="balanced"), una de cada {SUBMUESTRA_RBF} h',
        lambda: SVC(kernel="rbf", gamma="scale", class_weight="balanced", random_state=SEMILLA),
        rejilla={"C": (0.1, 1.0, 10.0)},
        submuestra=SUBMUESTRA_RBF,
    ),
}

MOTIVO_SUBMUESTRA = (
    "SVC con kernel RBF escala entre O(n^2) y O(n^3) con las filas: con todo el "
    "entrenamiento la rejilla con 5 pliegues tardaria horas. Una de cada {paso} h, "
    "en orden; {paso} es coprimo con 168, asi que se recorren todas las horas de la semana."
)


def _especificacion(nombre: str) -> Especificacion:
    if nombre not in MODELOS:
        raise ErrorClasificacion(f"Clasificador desconocido: {nombre!r}. Disponibles: {list(MODELOS)}")
    return MODELOS[nombre]


def crear_pipeline(nombre: str, **hiperparametros: Any) -> Pipeline:
    pipeline = Pipeline([("escalar", StandardScaler()), ("modelo", _especificacion(nombre).estimador())])
    if hiperparametros:
        pipeline.set_params(**{f"modelo__{k}": v for k, v in hiperparametros.items()})
    return pipeline


def puntaje(modelo, X: pd.DataFrame) -> tuple[np.ndarray, float, str]:
    """(puntaje de la clase 1, umbral natural, metodo): probabilidad o distancia al hiperplano."""
    if hasattr(modelo, "predict_proba"):
        try:
            return np.asarray(modelo.predict_proba(X))[:, 1], 0.5, "predict_proba"
        except AttributeError:  # SVC sin probability=True
            pass
    return np.asarray(modelo.decision_function(X), dtype="float64"), 0.0, "decision_function"


def _contando_convergencia(funcion: Callable[[], Any]) -> tuple[Any, int]:
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        resultado = funcion()
    contados = 0
    for aviso in avisos:
        if issubclass(aviso.category, ConvergenceWarning):
            contados += 1
        else:
            warnings.warn_explicit(aviso.message, aviso.category, aviso.filename, aviso.lineno)
    return resultado, contados


def posiciones_con_objetivo(objetivo: np.ndarray, posiciones: np.ndarray) -> np.ndarray:
    """Las posiciones de un bloque cuyo objetivo esta definido (fuera del arranque sin umbral)."""
    return posiciones[~np.isnan(objetivo[posiciones])]


# --------------------------------------------------------------------------
# Busqueda
# --------------------------------------------------------------------------


def buscar_hiperparametros(
    nombre: str, X: pd.DataFrame, y: np.ndarray, tscv: TimeSeriesSplit
) -> tuple[dict[str, Any], dict[str, Any]]:
    """GridSearchCV del Pipeline con `tscv`, eligiendo por PR-AUC. Devuelve (mejores, registro)."""
    especificacion = _especificacion(nombre)
    busqueda = GridSearchCV(
        crear_pipeline(nombre),
        {f"modelo__{k}": list(v) for k, v in especificacion.rejilla.items()},
        cv=tscv,
        scoring=PUNTUACIONES,
        refit=False,
        error_score="raise",
    )
    inicio = time.perf_counter()
    _, avisos = _contando_convergencia(lambda: busqueda.fit(X, y))
    segundos = time.perf_counter() - inicio

    resultados = busqueda.cv_results_
    mejor = int(np.argmin(resultados[f"rank_test_{CRITERIO}"]))
    mejores = {k.removeprefix("modelo__"): v for k, v in resultados["params"][mejor].items()}
    candidatos = [
        {
            "hiperparametros": {k.removeprefix("modelo__"): v for k, v in parametros.items()},
            **{
                f"{metrica}_media": float(resultados[f"mean_test_{metrica}"][i])
                for metrica in PUNTUACIONES
            },
            **{
                f"{metrica}_desviacion": float(resultados[f"std_test_{metrica}"][i])
                for metrica in PUNTUACIONES
            },
        }
        for i, parametros in enumerate(resultados["params"])
    ]
    por_pliegue = {
        metrica: [float(resultados[f"split{k}_test_{metrica}"][mejor]) for k in range(tscv.n_splits)]
        for metrica in PUNTUACIONES
    }
    registro = registro_operacion(
        "buscar_hiperparametros",
        f"GridSearchCV del Pipeline completo, cv=TimeSeriesSplit(n_splits={tscv.n_splits}, "
        f"gap={tscv.gap}), criterio PR-AUC (average precision) medio; tambien F1",
        rejilla={k: list(v) for k, v in especificacion.rejilla.items()},
        mejores=mejores,
        metrica_seleccion=CRITERIO,
        validacion_cruzada_mejor={
            metrica: {
                "media": float(np.mean(valores)),
                "desviacion": float(np.std(valores, ddof=1)),
                "por_pliegue": valores,
            }
            for metrica, valores in por_pliegue.items()
        },
        en_borde={
            k: v in (especificacion.rejilla[k][0], especificacion.rejilla[k][-1])
            for k, v in mejores.items()
            if isinstance(v, (int, float))
        },
        candidatos=candidatos,
        avisos_convergencia=avisos,
        segundos=segundos,
    )
    return mejores, registro


# --------------------------------------------------------------------------
# Evaluacion
# --------------------------------------------------------------------------


def evaluar_clasificador(
    nombre: str,
    X: pd.DataFrame,
    objetivo: np.ndarray,
    fechas: pd.Series,
    indices: dict[str, np.ndarray],
    rezago_minimo_h: int | str,
) -> tuple[Pipeline, dict[str, Any], np.ndarray]:
    """Busca, ajusta con entrenamiento (o su submuestra) y evalua en validacion.

    `objetivo` es alerta_pico alineado con X (NaN en el arranque sin umbral).
    Devuelve (Pipeline ajustado, registro, puntajes de validacion).
    """
    especificacion = _especificacion(nombre)
    rezago = rezago_de(rezago_minimo_h)
    objetivo = np.asarray(objetivo, dtype="float64")
    entrena = posiciones_con_objetivo(objetivo, indices["entrenamiento"])
    valida = indices["validacion"]
    if np.isnan(objetivo[valida]).any():
        raise ErrorClasificacion("La validacion tiene filas sin objetivo.")
    if len(entrena) == 0 or entrena.max() >= valida.min():
        raise ErrorClasificacion("El entrenamiento tiene que ser no vacio y anterior a la validacion.")

    paso = especificacion.submuestra
    if paso == 1:
        tscv, registro_cv = validacion_cruzada(rezago, fechas.iloc[entrena])
        submuestra = None
    else:
        entrena = entrena[::paso]
        gap = math.ceil(rezago / paso)
        tscv = TimeSeriesSplit(n_splits=5, gap=gap)
        registro_cv = registro_operacion(
            "validacion_cruzada",
            f"TimeSeriesSplit(n_splits=5, gap={gap}) sobre la submuestra: {gap} filas de "
            f"{paso} h = {gap * paso} h >= rezago de {rezago} h",
            n_splits=5,
            gap_filas=gap,
            gap_h=gap * paso,
            filas=int(len(entrena)),
        )
        submuestra = {
            "paso_h": paso,
            "filas": int(len(entrena)),
            "desde": str(fechas.iloc[entrena[0]]),
            "hasta": str(fechas.iloc[entrena[-1]]),
            "motivo": MOTIVO_SUBMUESTRA.format(paso=paso),
        }

    X_entrena = X.iloc[entrena].reset_index(drop=True)
    y_entrena = objetivo[entrena].astype(int)
    mejores, busqueda = buscar_hiperparametros(nombre, X_entrena, y_entrena, tscv)

    pipeline = crear_pipeline(nombre, **mejores)
    (pipeline, ajuste), avisos = _contando_convergencia(lambda: medir_ajuste(pipeline, X_entrena, y_entrena))
    _, medida = medir_prediccion(pipeline, X.iloc[valida])
    puntajes, umbral, metodo = puntaje(pipeline, X.iloc[valida])
    metricas = metricas_clasificacion(objetivo[valida].astype(int), puntajes, umbral=umbral)

    if submuestra is not None:
        submuestra["segundos_busqueda"] = busqueda["detalle"]["segundos"]
        submuestra["segundos_ajuste_final"] = ajuste["detalle"]["segundos"]
        submuestra["vectores_soporte"] = int(pipeline[-1].support_vectors_.shape[0])

    registro = {
        "descripcion": especificacion.descripcion,
        "hiperparametros": mejores,
        "semilla": SEMILLA,
        "puntaje": metodo,
        "umbral_decision": umbral,
        "filas_ajuste": int(len(entrena)),
        "positivos_ajuste": int(y_entrena.sum()),
        "submuestra": submuestra,
        "validacion_cruzada": registro_cv,
        "busqueda": busqueda,
        "metricas_validacion": metricas,
        "tiempos": {
            "busqueda_s": busqueda["detalle"]["segundos"],
            "ajuste_s": ajuste["detalle"]["segundos"],
            "prediccion_ms_por_1000_filas": medida["detalle"]["ms_por_1000_filas"],
            "repeticiones_prediccion": medida["detalle"]["repeticiones"],
            "tamano_serializado_bytes": tamano_serializado(pipeline),
        },
        "avisos_convergencia_ajuste_final": avisos,
    }
    return pipeline, registro, puntajes


def evaluar_baseline(
    X: pd.DataFrame, objetivo: np.ndarray, umbral: np.ndarray, indices: dict[str, np.ndarray]
) -> tuple[dict[str, Any], np.ndarray]:
    """Alerta si lag_168 supera el umbral de la fila. Puntaje = lag_168 - umbral (MWh), umbral 0.

    No se ajusta: el umbral es el mismo que define el objetivo.
    """
    columna = BASELINES[BASELINE_REFERENCIA]
    valida = indices["validacion"]
    puntajes = X[columna].to_numpy(dtype="float64")[valida] - np.asarray(umbral, dtype="float64")[valida]
    metricas = metricas_clasificacion(np.asarray(objetivo)[valida].astype(int), puntajes, umbral=0.0)
    return {
        "descripcion": f"alerta si {columna}(t) supera el umbral de la fila t",
        "puntaje": f"{columna} - umbral_alerta_mwh (MWh)",
        "umbral_decision": 0.0,
        "metricas_validacion": metricas,
    }, puntajes
