"""Modelos lineales de regresion: cada uno es un Pipeline StandardScaler + modelo.

    lineal             StandardScaler + LinearRegression
    polinomial         StandardScaler + PolynomialFeatures(2) + LinearRegression
    polinomial_ridge   StandardScaler + PolynomialFeatures(2) + Ridge
    ridge              StandardScaler + Ridge
    lasso              StandardScaler + Lasso
    elasticnet         StandardScaler + ElasticNet

La expansion polinomial sin regularizar queda mal condicionada: con variables
binarias estandarizadas, z^2 es una funcion afin de z, y sin^2 + cos^2 = 1 en
las ciclicas, asi que hay columnas exactamente dependientes. Por eso se reporta
tambien con Ridge, y el registro anota el numero de condicion y el rango de la
matriz de diseno de cada modelo.

**Hiperparametros.** Ridge, Lasso y ElasticNet buscan alpha (y l1_ratio) con
`GridSearchCV` sobre el Pipeline completo y el `TimeSeriesSplit` del bloque de
entrenamiento (`particion.validacion_cruzada`). No se usan RidgeCV/LassoCV
dentro del Pipeline a proposito: ahi el StandardScaler se ajustaria con todo el
entrenamiento antes de partirlo, y cada pliegue validaria con un escalado que
ya vio sus filas (CLAUDE.md: el escalado se ajusta en cada pliegue). El
criterio es el MAPE medio de los pliegues.

**Evaluacion.** Con los hiperparametros elegidos: metricas por pliegue (media
y desviacion, ddof=1), ajuste final con todo el entrenamiento, metricas en
validacion con la mejora frente al baseline estacional, tiempos y coeficientes
(los que Lasso anula, en `nulos`). La validacion nunca entra en el ajuste.

    from modelado.regresion import MODELOS, evaluar_modelo
    modelo, registro = evaluar_modelo("lasso", X, y, fechas, indices, tscv, mape_baseline)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from modelado.baselines import BASELINES, BASELINE_REFERENCIA
from modelado.datos import registro_operacion
from modelado.metricas import mape, metricas_regresion
from modelado.tiempos import medir_ajuste, medir_prediccion, tamano_serializado

SEMILLA = 42
GRADO_POLINOMIAL = 2
MAX_ITER = 50_000
# Criterio de seleccion: MAPE medio de los pliegues (sklearn lo da negativo y en fraccion).
CRITERIO = "neg_mean_absolute_percentage_error"

# Ridge penaliza ||w||^2 sin dividir por n: con ~30 000 filas, alphas grandes.
ALPHAS_RIDGE = tuple(float(a) for a in np.logspace(-2, 5, 15))
# Lasso y ElasticNet dividen el error por 2n: la escala util es mucho menor.
ALPHAS_LASSO = tuple(float(a) for a in np.logspace(-3, 2, 11))
L1_RATIOS = (0.1, 0.5, 0.9)

# Metricas de regresion que se resumen por pliegue.
METRICAS_CV = (
    "mae_mwh",
    "rmse_mwh",
    "mape_pct",
    "mape_pico_pct",
    "error_maximo_diario_pct",
    "sesgo_medio_mwh",
    "mejora_mape_pct",
)


class ErrorRegresion(Exception):
    """El modelo no se puede ajustar o evaluar con seguridad."""


@dataclass(frozen=True)
class Especificacion:
    descripcion: str
    estimador: Callable[[], Any]
    polinomial: bool = False
    rejilla: dict[str, tuple] = field(default_factory=dict)


MODELOS: dict[str, Especificacion] = {
    "lineal": Especificacion("StandardScaler + LinearRegression", LinearRegression),
    "polinomial": Especificacion(
        f"StandardScaler + PolynomialFeatures({GRADO_POLINOMIAL}) + LinearRegression",
        LinearRegression,
        polinomial=True,
    ),
    "polinomial_ridge": Especificacion(
        f"StandardScaler + PolynomialFeatures({GRADO_POLINOMIAL}) + Ridge",
        Ridge,
        polinomial=True,
        rejilla={"alpha": ALPHAS_RIDGE},
    ),
    "ridge": Especificacion("StandardScaler + Ridge", Ridge, rejilla={"alpha": ALPHAS_RIDGE}),
    "lasso": Especificacion(
        "StandardScaler + Lasso",
        lambda: Lasso(max_iter=MAX_ITER, random_state=SEMILLA),
        rejilla={"alpha": ALPHAS_LASSO},
    ),
    "elasticnet": Especificacion(
        "StandardScaler + ElasticNet",
        lambda: ElasticNet(max_iter=MAX_ITER, random_state=SEMILLA),
        rejilla={"alpha": ALPHAS_LASSO, "l1_ratio": L1_RATIOS},
    ),
}


def _especificacion(nombre: str) -> Especificacion:
    if nombre not in MODELOS:
        raise ErrorRegresion(f"Modelo desconocido: {nombre!r}. Disponibles: {list(MODELOS)}")
    return MODELOS[nombre]


def crear_pipeline(nombre: str, **hiperparametros: Any) -> Pipeline:
    """Pipeline sin ajustar: escalado, expansion polinomial si toca, y el modelo."""
    especificacion = _especificacion(nombre)
    pasos: list[tuple[str, Any]] = [("escalar", StandardScaler())]
    if especificacion.polinomial:
        pasos.append(("polinomios", PolynomialFeatures(degree=GRADO_POLINOMIAL, include_bias=False)))
    pasos.append(("modelo", especificacion.estimador()))
    pipeline = Pipeline(pasos)
    if hiperparametros:
        pipeline.set_params(**{f"modelo__{k}": v for k, v in hiperparametros.items()})
    return pipeline


def _con_avisos_de_convergencia(funcion: Callable[[], Any]) -> tuple[Any, int]:
    """Ejecuta `funcion` contando los ConvergenceWarning; los demas avisos se reemiten."""
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        resultado = funcion()
    convergencia = 0
    for aviso in avisos:
        if issubclass(aviso.category, ConvergenceWarning):
            convergencia += 1
        else:
            warnings.warn_explicit(aviso.message, aviso.category, aviso.filename, aviso.lineno)
    return resultado, convergencia


# --------------------------------------------------------------------------
# Busqueda de hiperparametros
# --------------------------------------------------------------------------


def buscar_hiperparametros(
    nombre: str, X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit
) -> tuple[dict[str, Any], dict[str, Any]]:
    """GridSearchCV del Pipeline completo con `tscv`. Devuelve (mejores, registro).

    Sin rejilla (lineal, polinomial) devuelve {} sin buscar. El registro anota
    cada candidato con su MAPE medio y desviacion, y si el elegido cae en el
    borde de la rejilla (habria que ampliarla).
    """
    especificacion = _especificacion(nombre)
    if not especificacion.rejilla:
        return {}, registro_operacion("buscar_hiperparametros", "sin hiperparametros que buscar", buscados=False)

    busqueda = GridSearchCV(
        crear_pipeline(nombre),
        {f"modelo__{k}": list(v) for k, v in especificacion.rejilla.items()},
        cv=tscv,
        scoring=CRITERIO,
        refit=False,
        error_score="raise",
    )
    inicio = time.perf_counter()
    _, avisos = _con_avisos_de_convergencia(lambda: busqueda.fit(X, y))
    segundos = time.perf_counter() - inicio

    mejores = {k.removeprefix("modelo__"): v for k, v in busqueda.best_params_.items()}
    resultados = busqueda.cv_results_
    candidatos = [
        {
            "hiperparametros": {k.removeprefix("modelo__"): v for k, v in parametros.items()},
            "mape_media_pct": float(-100 * media),
            "mape_desviacion_pct": float(100 * desviacion),
        }
        for parametros, media, desviacion in zip(
            resultados["params"], resultados["mean_test_score"], resultados["std_test_score"]
        )
    ]
    registro = registro_operacion(
        "buscar_hiperparametros",
        f"GridSearchCV del Pipeline completo, cv=TimeSeriesSplit(n_splits={tscv.n_splits}, "
        f"gap={tscv.gap}), criterio MAPE medio de los pliegues",
        buscados=True,
        rejilla={k: list(v) for k, v in especificacion.rejilla.items()},
        mejores=mejores,
        mape_cv_mejor_pct=float(-100 * busqueda.best_score_),
        en_borde={k: v in (min(especificacion.rejilla[k]), max(especificacion.rejilla[k])) for k, v in mejores.items()},
        candidatos=candidatos,
        avisos_convergencia=avisos,
        segundos=segundos,
    )
    return mejores, registro


# --------------------------------------------------------------------------
# Metricas por pliegue
# --------------------------------------------------------------------------


def metricas_por_pliegue(
    nombre: str,
    hiperparametros: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    tscv: TimeSeriesSplit,
) -> dict[str, Any]:
    """Todas las metricas en cada pliegue de `tscv`, con media y desviacion (ddof=1).

    Cada pliegue ajusta su propio Pipeline (escalado incluido) solo con su tramo
    de entrenamiento. La mejora de MAPE es frente al baseline estacional del
    mismo pliegue. Con hiperparametros elegidos en estos mismos pliegues, la
    cifra de los modelos con busqueda es algo optimista.
    """
    columna_base = BASELINES[BASELINE_REFERENCIA]
    pliegues, avisos = [], 0
    for numero, (entrena, valida) in enumerate(tscv.split(X), start=1):
        modelo = crear_pipeline(nombre, **hiperparametros)
        _, n_avisos = _con_avisos_de_convergencia(lambda: modelo.fit(X.iloc[entrena], y.iloc[entrena]))
        avisos += n_avisos
        y_valida = y.iloc[valida]
        metricas = metricas_regresion(
            y_valida,
            modelo.predict(X.iloc[valida]),
            fechas.iloc[valida],
            mape_baseline=mape(y_valida, X[columna_base].iloc[valida]),
        )
        pliegues.append({"pliegue": numero, **metricas})

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
        "un Pipeline nuevo por pliegue ajustado con su tramo de entrenamiento; "
        "media y desviacion (ddof=1) entre pliegues; mejora frente al baseline "
        f"{BASELINE_REFERENCIA} del mismo pliegue",
        n_pliegues=len(pliegues),
        resumen=resumen,
        pliegues=pliegues,
        avisos_convergencia=avisos,
    )


# --------------------------------------------------------------------------
# Lectura del modelo ajustado
# --------------------------------------------------------------------------


def coeficientes(pipeline: Pipeline) -> dict[str, Any]:
    """Coeficientes del modelo lineal final, en la escala estandarizada, por variable.

    `nulos` son los que valen exactamente 0 (los que anula Lasso o ElasticNet).
    """
    nombres = [str(n) for n in pipeline[:-1].get_feature_names_out()]
    coef = np.asarray(pipeline[-1].coef_, dtype="float64").ravel()
    nulos = [n for n, c in zip(nombres, coef) if c == 0.0]
    return {
        "escala": "variables estandarizadas (y polinomios de ellas): MWh por desviacion estandar",
        "intercepto": float(pipeline[-1].intercept_),
        "por_variable": {n: float(c) for n, c in zip(nombres, coef)},
        "n_coeficientes": len(coef),
        "nulos": nulos,
        "n_nulos": len(nulos),
        "norma_l2": float(np.linalg.norm(coef)),
    }


def condicionamiento(pipeline: Pipeline, X: pd.DataFrame) -> dict[str, Any]:
    """Numero de condicion y rango de la matriz de diseno (centrada) que ve el modelo final.

    Se centra porque el modelo ajusta el intercepto aparte. Un numero de
    condicion enorme o un rango menor que el numero de columnas indican
    columnas (casi) dependientes: coeficientes inestables sin regularizacion.
    """
    diseno = np.asarray(pipeline[:-1].transform(X), dtype="float64")
    diseno = diseno - diseno.mean(axis=0)
    condicion = float(np.linalg.cond(diseno))
    return {
        "columnas": int(diseno.shape[1]),
        "rango": int(np.linalg.matrix_rank(diseno)),
        "numero_condicion": condicion if np.isfinite(condicion) else None,
    }


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------


def evaluar_modelo(
    nombre: str,
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    indices: dict[str, np.ndarray],
    tscv: TimeSeriesSplit,
    mape_baseline_validacion: float | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Busca hiperparametros y mide en CV dentro de entrenamiento, ajusta y evalua en validacion.

    Solo usa `indices["entrenamiento"]` para ajustar y `indices["validacion"]`
    para evaluar; el bloque de prueba no entra. Devuelve (Pipeline ajustado con
    todo el entrenamiento, registro).
    """
    especificacion = _especificacion(nombre)
    entrena, valida = indices["entrenamiento"], indices["validacion"]
    if len(entrena) == 0 or len(valida) == 0 or entrena.max() >= valida.min():
        raise ErrorRegresion("El entrenamiento tiene que ser no vacio y anterior a la validacion.")
    X_entrena = X.iloc[entrena].reset_index(drop=True)
    y_entrena = y.iloc[entrena].reset_index(drop=True)
    fechas_entrena = fechas.iloc[entrena].reset_index(drop=True)

    mejores, busqueda = buscar_hiperparametros(nombre, X_entrena, y_entrena, tscv)
    validacion_cruzada = metricas_por_pliegue(nombre, mejores, X_entrena, y_entrena, fechas_entrena, tscv)

    pipeline = crear_pipeline(nombre, **mejores)
    (pipeline, ajuste), avisos = _con_avisos_de_convergencia(lambda: medir_ajuste(pipeline, X_entrena, y_entrena))
    prediccion, medida = medir_prediccion(pipeline, X.iloc[valida])
    metricas = metricas_regresion(
        y.iloc[valida], prediccion, fechas.iloc[valida], mape_baseline=mape_baseline_validacion
    )

    registro = {
        "descripcion": especificacion.descripcion,
        "pasos": [paso for paso, _ in pipeline.steps],
        "hiperparametros": mejores,
        "semilla": SEMILLA,
        "busqueda": busqueda,
        "validacion_cruzada": validacion_cruzada,
        "metricas_validacion": metricas,
        "mejora_frente_a": BASELINE_REFERENCIA if mape_baseline_validacion is not None else None,
        "tiempos": {
            "busqueda_s": busqueda["detalle"].get("segundos"),
            "ajuste_s": ajuste["detalle"]["segundos"],
            "prediccion_ms_por_1000_filas": medida["detalle"]["ms_por_1000_filas"],
            "repeticiones_prediccion": medida["detalle"]["repeticiones"],
            "tamano_serializado_bytes": tamano_serializado(pipeline),
        },
        "coeficientes": coeficientes(pipeline),
        "condicionamiento": condicionamiento(pipeline, X_entrena),
        "filas_ajuste": int(len(entrena)),
        "avisos_convergencia_ajuste_final": avisos,
    }
    return pipeline, registro


def ajustado_con(pipeline: Pipeline) -> int:
    """Filas con las que se ajusto el escalado del Pipeline (para comprobar que no vio validacion)."""
    return int(pipeline.named_steps["escalar"].n_samples_seen_)


__all__ = [
    "MODELOS",
    "ajustado_con",
    "buscar_hiperparametros",
    "coeficientes",
    "condicionamiento",
    "crear_pipeline",
    "evaluar_modelo",
    "metricas_por_pliegue",
]
