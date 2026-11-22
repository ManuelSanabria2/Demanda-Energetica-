"""Busqueda de hiperparametros: GridSearchCV frente a RandomizedSearchCV.

Candidatos, por tabla, leidos de los resultados ya escritos (nada a mano):

- los `N_REGRESION` mejores modelos de regresion en validacion por MAPE, entre
  la familia lineal (docs/resultados/modelado_regresion.json) y los ensembles
  en sus dos formas (modelado_ensembles.json). Los baselines y los modelos sin
  hiperparametros que buscar no compiten (quedan en `omitidos`);
- el mejor clasificador de alerta_pico en validacion, por PR-AUC, el criterio
  con el que se eligio en su paso (modelado_clasificacion.json).

Para cada candidato, sobre el mismo espacio:

- `GridSearchCV` con una rejilla acotada (`ESPACIOS[...].rejilla`);
- `RandomizedSearchCV` con distribuciones de scipy.stats que cubren el mismo
  rango (`ESPACIOS[...].distribuciones`) y tantas iteraciones como
  combinaciones tiene la rejilla.

Las dos con el TimeSeriesSplit del entrenamiento (`particion.validacion_cruzada`;
la SVC RBF, con su submuestra y su gap, `clasificacion.muestra_de_entrenamiento`),
scoring `neg_mean_absolute_error` (F1 en clasificacion), `n_jobs=-1`,
`random_state=42` y solo filas del bloque de entrenamiento. `refit=False`: el
mejor de cada busqueda se reentrena aparte con todo el entrenamiento, se mide
y se evalua en validacion.

**Referencia.** La configuracion con la que cada modelo se evaluo en su paso
(los valores fijos de `ensembles.MODELOS`, o los hiperparametros elegidos que
anota el JSON de su familia) se mide igual: CV con el mismo TimeSeriesSplit y
validacion. La mejora de cada busqueda es frente a ella.

**¿Llega Random a lo mismo con menos ajustes?** Cada busqueda deja su
trayectoria: el mejor puntaje de CV acumulado candidato a candidato y el tiempo
de computo acumulado (suma de ajuste y puntuacion de cada pliegue; con
`n_jobs=-1` el tiempo de reloj es menor y va aparte). `comparar_busquedas`
cuenta los ajustes que necesita cada metodo para quedar a menos de
`TOLERANCIA_RELATIVA` del mejor puntaje de Grid.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, cross_validate

from modelado import clasificacion, ensembles, regresion
from modelado.baselines import BASELINE_REFERENCIA, BASELINES
from modelado.datos import registro_operacion
from modelado.metricas import mape, metricas_clasificacion, metricas_regresion
from modelado.particion import validacion_cruzada
from modelado.tiempos import medir_ajuste, medir_prediccion, tamano_serializado

SEMILLA = 42
N_REGRESION = 3
N_JOBS = -1
PUNTUACION = {"regresion": "neg_mean_absolute_error", "clasificacion": "f1"}
# "Llega a lo mismo": mejor puntaje de CV a menos de este margen relativo del de Grid.
TOLERANCIA_RELATIVA = 0.005
METODOS = ("grid", "random")


class ErrorBusqueda(Exception):
    """La busqueda no se puede plantear o ejecutar con seguridad."""


# --------------------------------------------------------------------------
# Espacios de busqueda
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Distribucion:
    """Distribucion de scipy.stats con su descripcion legible para el registro."""

    texto: str
    crear: Callable[[], Any]


def entero(a: int, b: int) -> Distribucion:
    """Entero uniforme en [a, b] (ambos incluidos)."""
    return Distribucion(f"randint({a}, {b + 1})", lambda: stats.randint(a, b + 1))


def continua(a: float, b: float) -> Distribucion:
    return Distribucion(f"uniform({a}, {b})", lambda: stats.uniform(loc=a, scale=b - a))


def logaritmica(a: float, b: float) -> Distribucion:
    return Distribucion(f"loguniform({a}, {b})", lambda: stats.loguniform(a, b))


def categorica(*valores: Any) -> Distribucion:
    return Distribucion(f"uniforme sobre {list(valores)}", lambda: list(valores))


@dataclass(frozen=True)
class Espacio:
    """Rejilla acotada y distribuciones sobre el mismo rango, con nombres relativos al modelo."""

    rejilla: dict[str, tuple]
    distribuciones: dict[str, Distribucion]

    def __post_init__(self) -> None:
        if set(self.rejilla) != set(self.distribuciones):
            raise ErrorBusqueda("La rejilla y las distribuciones tienen que cubrir los mismos parametros.")

    @property
    def combinaciones(self) -> int:
        return math.prod(len(v) for v in self.rejilla.values())


ESPACIOS: dict[tuple[str, str], Espacio] = {
    ("ensembles", "random_forest"): Espacio(
        {"max_features": (0.3, 0.5, 0.8), "min_samples_leaf": (5, 10, 20), "max_samples": (0.5, 1.0)},
        {"max_features": continua(0.3, 0.8), "min_samples_leaf": entero(5, 20),
         "max_samples": continua(0.5, 1.0)},
    ),
    ("ensembles", "adaboost"): Espacio(
        {"estimator__max_depth": (4, 6, 8), "learning_rate": (0.1, 0.5, 1.0), "loss": ("linear", "square")},
        {"estimator__max_depth": entero(4, 8), "learning_rate": logaritmica(0.1, 1.0),
         "loss": categorica("linear", "square")},
    ),
    ("ensembles", "gradient_boosting"): Espacio(
        {"learning_rate": (0.05, 0.1, 0.2), "max_depth": (3, 4, 6), "n_estimators": (200, 400)},
        {"learning_rate": logaritmica(0.05, 0.2), "max_depth": entero(3, 6), "n_estimators": entero(200, 400)},
    ),
    ("ensembles", "hist_gradient"): Espacio(
        {"learning_rate": (0.05, 0.1, 0.2), "max_leaf_nodes": (15, 31, 63), "l2_regularization": (0.0, 1.0)},
        {"learning_rate": logaritmica(0.05, 0.2), "max_leaf_nodes": entero(15, 63),
         "l2_regularization": continua(0.0, 1.0)},
    ),
    ("lineal", "ridge"): Espacio(
        {"alpha": regresion.ALPHAS_RIDGE}, {"alpha": logaritmica(min(regresion.ALPHAS_RIDGE), max(regresion.ALPHAS_RIDGE))}
    ),
    ("lineal", "polinomial_ridge"): Espacio(
        {"alpha": regresion.ALPHAS_RIDGE}, {"alpha": logaritmica(min(regresion.ALPHAS_RIDGE), max(regresion.ALPHAS_RIDGE))}
    ),
    ("lineal", "lasso"): Espacio(
        {"alpha": regresion.ALPHAS_LASSO}, {"alpha": logaritmica(min(regresion.ALPHAS_LASSO), max(regresion.ALPHAS_LASSO))}
    ),
    ("lineal", "elasticnet"): Espacio(
        {"alpha": regresion.ALPHAS_LASSO, "l1_ratio": regresion.L1_RATIOS},
        {"alpha": logaritmica(min(regresion.ALPHAS_LASSO), max(regresion.ALPHAS_LASSO)),
         "l1_ratio": continua(min(regresion.L1_RATIOS), max(regresion.L1_RATIOS))},
    ),
    ("clasificacion", "svm_rbf"): Espacio(
        {"C": (0.3, 1.0, 3.0, 10.0, 30.0), "gamma": (0.01, 0.03, 0.1, 0.3)},
        {"C": logaritmica(0.3, 30.0), "gamma": logaritmica(0.01, 0.3)},
    ),
    ("clasificacion", "logistica"): Espacio(
        {"C": (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)}, {"C": logaritmica(0.001, 100.0)}
    ),
    ("clasificacion", "svm_lineal"): Espacio(
        {"C": (0.001, 0.01, 0.1, 1.0, 10.0)}, {"C": logaritmica(0.001, 10.0)}
    ),
    ("clasificacion", "knn"): Espacio(
        {"n_neighbors": (5, 15, 45, 135, 405), "weights": ("uniform", "distance")},
        {"n_neighbors": entero(5, 405), "weights": categorica("uniform", "distance")},
    ),
}


def espacio_de(familia: str, nombre: str) -> Espacio | None:
    return ESPACIOS.get((familia, nombre))


# --------------------------------------------------------------------------
# Candidatos: leidos de los resultados de los pasos anteriores
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidato:
    familia: str  # "lineal", "ensembles" o "clasificacion"
    nombre: str
    forma: str  # "nivel" / "diferenciado" en ensembles; "-" en el resto
    criterio_seleccion: str
    valor_seleccion: float
    hiperparametros_previos: dict[str, Any]

    @property
    def tipo(self) -> str:
        return "clasificacion" if self.familia == "clasificacion" else "regresion"

    @property
    def etiqueta(self) -> str:
        return self.nombre if self.forma == "-" else f"{self.nombre} ({self.forma})"

    @property
    def clave(self) -> str:
        return self.nombre if self.forma == "-" else f"{self.nombre}_{self.forma}"


def seleccionar_candidatos(
    resultados: dict[str, dict[str, Any]], clave: str, n_regresion: int = N_REGRESION
) -> tuple[list[Candidato], dict[str, Any]]:
    """Los n mejores de regresion por MAPE de validacion y el mejor clasificador por PR-AUC.

    `resultados` es {familia: contenido del JSON de esa familia}. Devuelve
    (candidatos, registro con la clasificacion completa y los omitidos).
    """
    for familia in ("lineal", "ensembles", "clasificacion"):
        if clave not in (resultados.get(familia) or {}).get("tablas", {}):
            raise ErrorBusqueda(f"Faltan los resultados de la familia {familia!r} para {clave}.")

    filas = []
    for nombre, registro in resultados["lineal"]["tablas"][clave]["modelos"].items():
        filas.append(("lineal", nombre, "-", registro["metricas_validacion"]["mape_pct"],
                      dict(registro.get("hiperparametros") or {})))
    for nombre, formas in resultados["ensembles"]["tablas"][clave]["modelos"].items():
        for forma, registro in formas.items():
            filas.append(("ensembles", nombre, forma, registro["metricas_validacion"]["mape_pct"], {}))
    filas.sort(key=lambda fila: fila[3])

    candidatos, omitidos, ranking = [], {}, []
    for familia, nombre, forma, valor, previos in filas:
        ranking.append({"familia": familia, "nombre": nombre, "forma": forma, "mape_validacion_pct": valor})
        if len(candidatos) == n_regresion:
            continue
        if espacio_de(familia, nombre) is None:
            omitidos[f"{nombre}" if forma == "-" else f"{nombre}_{forma}"] = (
                "sin hiperparametros que buscar (o sin espacio definido)"
            )
            continue
        candidatos.append(Candidato(familia, nombre, forma, "mape_validacion_pct", valor, previos))

    clasificadores = resultados["clasificacion"]["tablas"][clave]["modelos"]
    orden = sorted(clasificadores, key=lambda n: -clasificadores[n]["metricas_validacion"]["pr_auc"])
    mejor = next((n for n in orden if espacio_de("clasificacion", n) is not None), None)
    if mejor is None:
        raise ErrorBusqueda("Ningun clasificador tiene espacio de busqueda.")
    candidatos.append(Candidato(
        "clasificacion", mejor, "-", "pr_auc_validacion",
        clasificadores[mejor]["metricas_validacion"]["pr_auc"],
        dict(clasificadores[mejor].get("hiperparametros") or {}),
    ))

    registro = registro_operacion(
        "seleccionar_candidatos",
        f"los {n_regresion} mejores de regresion (lineal y ensembles) por MAPE de validacion "
        "con espacio de busqueda, y el mejor clasificador por PR-AUC de validacion",
        ranking_regresion=ranking,
        ranking_clasificacion=[
            {"nombre": n, "pr_auc_validacion": clasificadores[n]["metricas_validacion"]["pr_auc"]} for n in orden
        ],
        omitidos=omitidos,
        elegidos=[c.etiqueta for c in candidatos],
    )
    return candidatos, registro


# --------------------------------------------------------------------------
# Estimadores
# --------------------------------------------------------------------------


def prefijo(candidato: Candidato) -> str:
    """Prefijo de los parametros del modelo dentro del estimador que se busca."""
    if candidato.familia in ("lineal", "clasificacion"):
        return "modelo__"
    return "estimador__" if candidato.forma == "diferenciado" else ""


def crear_estimador(candidato: Candidato) -> BaseEstimator:
    """El estimador de referencia: la configuracion con la que se evaluo en su paso.

    Random Forest va con `n_jobs=1`: la busqueda ya reparte pliegues y
    candidatos entre procesos (`n_jobs=-1`).
    """
    if candidato.familia == "ensembles":
        estimador = ensembles.crear_modelo(candidato.nombre, candidato.forma)
        interno = estimador.estimador if candidato.forma == "diferenciado" else estimador
        if "n_jobs" in interno.get_params():
            interno.set_params(n_jobs=1)
        return estimador
    if candidato.familia == "lineal":
        return regresion.crear_pipeline(candidato.nombre, **candidato.hiperparametros_previos)
    if candidato.familia == "clasificacion":
        return clasificacion.crear_pipeline(candidato.nombre, **candidato.hiperparametros_previos)
    raise ErrorBusqueda(f"Familia desconocida: {candidato.familia!r}")


def _con_prefijo(candidato: Candidato, parametros: dict[str, Any]) -> dict[str, Any]:
    return {prefijo(candidato) + k: v for k, v in parametros.items()}


def _sin_prefijo(candidato: Candidato, parametros: dict[str, Any]) -> dict[str, Any]:
    return {k.removeprefix(prefijo(candidato)): _a_json(v) for k, v in parametros.items()}


def _a_json(valor: Any) -> Any:
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


# --------------------------------------------------------------------------
# Datos del candidato: solo el bloque de entrenamiento para buscar
# --------------------------------------------------------------------------


@dataclass
class Muestra:
    X_entrena: pd.DataFrame
    y_entrena: np.ndarray
    tscv: TimeSeriesSplit
    registro_cv: dict[str, Any]
    filas_entrenamiento: np.ndarray
    submuestra: dict[str, Any] | None


def muestra(
    candidato: Candidato,
    X: pd.DataFrame,
    y: pd.Series,
    objetivo_clasificacion: np.ndarray | None,
    fechas: pd.Series,
    indices: dict[str, np.ndarray],
    clave: str,
) -> Muestra:
    if candidato.tipo == "regresion":
        entrena = indices["entrenamiento"]
        tscv, registro_cv = validacion_cruzada(clave, fechas.iloc[entrena])
        objetivo, submuestra = y.to_numpy(dtype="float64"), None
    else:
        if objetivo_clasificacion is None:
            raise ErrorBusqueda("Falta alerta_pico para el clasificador.")
        objetivo = np.asarray(objetivo_clasificacion, dtype="float64")
        entrena, tscv, registro_cv, submuestra = clasificacion.muestra_de_entrenamiento(
            candidato.nombre, objetivo, fechas, indices, clave
        )
    if entrena.max() >= indices["validacion"].min():
        raise ErrorBusqueda("El entrenamiento de la busqueda llega a la validacion.")  # defensa
    y_entrena = objetivo[entrena]
    if candidato.tipo == "clasificacion":
        y_entrena = y_entrena.astype(int)
    return Muestra(X.iloc[entrena].reset_index(drop=True), y_entrena, tscv, registro_cv, entrena, submuestra)


# --------------------------------------------------------------------------
# Ajuste final y evaluacion en validacion
# --------------------------------------------------------------------------


def _ajustar_y_evaluar(
    candidato: Candidato,
    estimador: BaseEstimator,
    m: Muestra,
    X: pd.DataFrame,
    y: pd.Series,
    objetivo_clasificacion: np.ndarray | None,
    fechas: pd.Series,
    valida: np.ndarray,
    mape_baseline: float | None,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Ajusta con todo el entrenamiento (o su submuestra) y evalua en validacion."""
    modelo, ajuste = medir_ajuste(clone(estimador), m.X_entrena, m.y_entrena)
    if candidato.familia == "ensembles":
        ensembles.ajustar_prediccion_en_un_hilo(modelo)
    if candidato.tipo == "regresion":
        prediccion, medida = medir_prediccion(modelo, X.iloc[valida])
        metricas = metricas_regresion(y.iloc[valida], prediccion, fechas.iloc[valida], mape_baseline=mape_baseline)
    else:
        _, medida = medir_prediccion(modelo, X.iloc[valida])
        puntajes, umbral, _ = clasificacion.puntaje(modelo, X.iloc[valida])
        real = np.asarray(objetivo_clasificacion, dtype="float64")[valida].astype(int)
        metricas = metricas_clasificacion(real, puntajes, umbral=umbral)
    return modelo, {
        "metricas_validacion": metricas,
        "tiempos": {
            "ajuste_s": ajuste["detalle"]["segundos"],
            "prediccion_ms_por_1000_filas": medida["detalle"]["ms_por_1000_filas"],
            "tamano_serializado_bytes": tamano_serializado(modelo),
        },
    }


def _metrica_principal(candidato: Candidato, metricas: dict[str, Any]) -> dict[str, Any]:
    if candidato.tipo == "regresion":
        return {k: metricas[k] for k in ("mae_mwh", "mape_pct", "rmse_mwh", "sesgo_medio_mwh")}
    return {k: metricas[k] for k in ("f1", "pr_auc", "roc_auc", "precision", "recall")}


def mejora(candidato: Candidato, referencia: dict[str, Any], buscado: dict[str, Any]) -> dict[str, Any]:
    """Mejora en validacion frente a la referencia (positiva: mejor)."""
    r, b = referencia, buscado
    if candidato.tipo == "regresion":
        return {
            "mae_mwh": r["mae_mwh"] - b["mae_mwh"],
            "mae_pct": 100.0 * (r["mae_mwh"] - b["mae_mwh"]) / r["mae_mwh"],
            "mape_puntos": r["mape_pct"] - b["mape_pct"],
        }
    return {
        "f1": None if r["f1"] is None or b["f1"] is None else b["f1"] - r["f1"],
        "pr_auc": b["pr_auc"] - r["pr_auc"],
    }


# --------------------------------------------------------------------------
# Busquedas
# --------------------------------------------------------------------------


def _trayectoria(resultados: dict[str, Any], n_splits: int) -> list[dict[str, Any]]:
    """Por candidato en el orden en que se evaluo: puntaje, mejor acumulado y computo acumulado."""
    puntajes = np.asarray(resultados["mean_test_score"], dtype="float64")
    computo = (np.asarray(resultados["mean_fit_time"]) + np.asarray(resultados["mean_score_time"])) * n_splits
    mejor_acumulado = np.maximum.accumulate(puntajes)
    return [
        {
            "candidato": i + 1,
            "ajustes_cv": (i + 1) * n_splits,
            "puntaje_cv": float(puntajes[i]),
            "mejor_puntaje_cv_acumulado": float(mejor_acumulado[i]),
            "computo_acumulado_s": float(computo[: i + 1].sum()),
        }
        for i in range(len(puntajes))
    ]


def buscar(
    metodo: str,
    candidato: Candidato,
    espacio: Espacio,
    m: Muestra,
    n_jobs: int = N_JOBS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """GridSearchCV o RandomizedSearchCV con el TimeSeriesSplit de la muestra. Devuelve (mejores, registro)."""
    estimador = crear_estimador(candidato)
    puntuacion = PUNTUACION[candidato.tipo]
    if metodo == "grid":
        busqueda = GridSearchCV(
            estimador, _con_prefijo(candidato, {k: list(v) for k, v in espacio.rejilla.items()}),
            scoring=puntuacion, cv=m.tscv, n_jobs=n_jobs, refit=False, error_score="raise",
        )
        espacio_registro = {k: [_a_json(x) for x in v] for k, v in espacio.rejilla.items()}
    elif metodo == "random":
        busqueda = RandomizedSearchCV(
            estimador, _con_prefijo(candidato, {k: d.crear() for k, d in espacio.distribuciones.items()}),
            n_iter=espacio.combinaciones, scoring=puntuacion, cv=m.tscv, n_jobs=n_jobs, refit=False,
            random_state=SEMILLA, error_score="raise",
        )
        espacio_registro = {k: d.texto for k, d in espacio.distribuciones.items()}
    else:
        raise ErrorBusqueda(f"Metodo desconocido: {metodo!r}. Disponibles: {list(METODOS)}")

    inicio = time.perf_counter()
    busqueda.fit(m.X_entrena, m.y_entrena)
    segundos = time.perf_counter() - inicio

    resultados = busqueda.cv_results_
    n_candidatos = len(resultados["params"])
    mejores = _sin_prefijo(candidato, busqueda.best_params_)
    registro = {
        "metodo": "GridSearchCV" if metodo == "grid" else "RandomizedSearchCV",
        "espacio": espacio_registro,
        "n_iter": None if metodo == "grid" else espacio.combinaciones,
        "semilla": None if metodo == "grid" else SEMILLA,
        "scoring": puntuacion,
        "cv": f"TimeSeriesSplit(n_splits={m.tscv.n_splits}, gap={m.tscv.gap})",
        "n_jobs": n_jobs,
        "mejores": mejores,
        "mejor_puntaje_cv": float(busqueda.best_score_),
        "desviacion_mejor_puntaje_cv": float(resultados["std_test_score"][busqueda.best_index_]),
        "n_candidatos": n_candidatos,
        "n_ajustes_cv": n_candidatos * m.tscv.n_splits,
        "segundos_busqueda": segundos,
        "candidatos": [
            {"hiperparametros": _sin_prefijo(candidato, p), "puntaje_cv": float(s), "desviacion": float(d)}
            for p, s, d in zip(resultados["params"], resultados["mean_test_score"], resultados["std_test_score"])
        ],
        "trayectoria": _trayectoria(resultados, m.tscv.n_splits),
    }
    return mejores, registro


def comparar_busquedas(grid: dict[str, Any], random: dict[str, Any], tolerancia: float = TOLERANCIA_RELATIVA) -> dict[str, Any]:
    """Ajustes que necesita cada metodo para quedar a menos de `tolerancia` del mejor puntaje de Grid.

    Los puntajes son "mayor es mejor" (MAE negativo o F1). El umbral es el
    mejor de Grid menos `tolerancia` veces su valor absoluto.
    """
    objetivo = grid["mejor_puntaje_cv"] - tolerancia * abs(grid["mejor_puntaje_cv"])

    def _ajustes_hasta(registro: dict[str, Any]) -> int | None:
        for paso in registro["trayectoria"]:
            if paso["mejor_puntaje_cv_acumulado"] >= objetivo:
                return paso["ajustes_cv"]
        return None

    random_ajustes = _ajustes_hasta(random)
    return {
        "criterio": f"mejor puntaje de CV a menos del {100 * tolerancia:.1f} % del mejor de Grid",
        "umbral_puntaje_cv": objetivo,
        "diferencia_mejor_puntaje_cv": random["mejor_puntaje_cv"] - grid["mejor_puntaje_cv"],
        "random_alcanza_grid": random_ajustes is not None,
        "ajustes_cv_grid_hasta_su_mejor": _ajustes_hasta(grid),
        "ajustes_cv_random_hasta_alcanzar": random_ajustes,
        "ajustes_cv_totales": {"grid": grid["n_ajustes_cv"], "random": random["n_ajustes_cv"]},
        "segundos_busqueda": {"grid": grid["segundos_busqueda"], "random": random["segundos_busqueda"]},
    }


# --------------------------------------------------------------------------
# Orquestacion por candidato
# --------------------------------------------------------------------------


def evaluar_candidato(
    candidato: Candidato,
    X: pd.DataFrame,
    y: pd.Series,
    objetivo_clasificacion: np.ndarray | None,
    fechas: pd.Series,
    indices: dict[str, np.ndarray],
    clave: str,
    n_jobs: int = N_JOBS,
) -> tuple[dict[str, BaseEstimator], dict[str, Any]]:
    """Referencia, Grid y Random para un candidato. Devuelve ({metodo: modelo}, registro).

    Buscar y reentrenar solo ve el bloque de entrenamiento (o su submuestra);
    la validacion entra solo en la evaluacion final de cada configuracion.
    """
    espacio = espacio_de(candidato.familia, candidato.nombre)
    if espacio is None:
        raise ErrorBusqueda(f"{candidato.etiqueta} no tiene espacio de busqueda.")
    m = muestra(candidato, X, y, objetivo_clasificacion, fechas, indices, clave)
    valida = indices["validacion"]
    mape_base = None
    if candidato.tipo == "regresion":
        mape_base = mape(y.iloc[valida], X[BASELINES[BASELINE_REFERENCIA]].iloc[valida])

    # Referencia: la configuracion del paso anterior, medida igual.
    referencia_estimador = crear_estimador(candidato)
    inicio = time.perf_counter()
    cv = cross_validate(referencia_estimador, m.X_entrena, m.y_entrena, cv=m.tscv,
                        scoring=PUNTUACION[candidato.tipo], n_jobs=n_jobs, error_score="raise")
    segundos_cv = time.perf_counter() - inicio
    modelos: dict[str, BaseEstimator] = {}
    modelos["referencia"], evaluacion = _ajustar_y_evaluar(
        candidato, referencia_estimador, m, X, y, objetivo_clasificacion, fechas, valida, mape_base
    )
    referencia = {
        "hiperparametros": candidato.hiperparametros_previos or "los fijos de su modulo (ensembles.MODELOS)",
        "puntaje_cv": float(np.mean(cv["test_score"])),
        "desviacion_puntaje_cv": float(np.std(cv["test_score"])),
        "n_ajustes_cv": int(m.tscv.n_splits),
        "segundos_cv": segundos_cv,
        **evaluacion,
    }

    metodos: dict[str, Any] = {}
    for metodo in METODOS:
        mejores, registro = buscar(metodo, candidato, espacio, m, n_jobs=n_jobs)
        estimador = crear_estimador(candidato).set_params(**_con_prefijo(candidato, mejores))
        modelos[metodo], evaluacion = _ajustar_y_evaluar(
            candidato, estimador, m, X, y, objetivo_clasificacion, fechas, valida, mape_base
        )
        registro.update(evaluacion)
        registro["n_ajustes"] = registro["n_ajustes_cv"] + 1  # mas el reentreno con todo el entrenamiento
        registro["segundos_total"] = registro["segundos_busqueda"] + evaluacion["tiempos"]["ajuste_s"]
        registro["mejora_cv_frente_a_referencia"] = registro["mejor_puntaje_cv"] - referencia["puntaje_cv"]
        registro["mejora_validacion_frente_a_referencia"] = mejora(
            candidato,
            _metrica_principal(candidato, referencia["metricas_validacion"]),
            _metrica_principal(candidato, evaluacion["metricas_validacion"]),
        )
        metodos[metodo] = registro

    registro = {
        "familia": candidato.familia,
        "nombre": candidato.nombre,
        "forma": candidato.forma,
        "tipo": candidato.tipo,
        "seleccion": {"criterio": candidato.criterio_seleccion, "valor": candidato.valor_seleccion},
        "combinaciones_rejilla": espacio.combinaciones,
        "filas_busqueda": int(len(m.y_entrena)),
        "validacion_cruzada": m.registro_cv,
        "submuestra": m.submuestra,
        "mejora_frente_a": BASELINE_REFERENCIA if mape_base is not None else None,
        "referencia": referencia,
        **metodos,
        "comparacion": comparar_busquedas(metodos["grid"], metodos["random"]),
    }
    return modelos, registro


def combinaciones(rejilla: dict[str, tuple]) -> list[dict[str, Any]]:
    """Todas las combinaciones de una rejilla (para comprobar Grid en las pruebas)."""
    claves = list(rejilla)
    return [dict(zip(claves, valores)) for valores in itertools.product(*(rejilla[c] for c in claves))]


__all__ = [
    "ESPACIOS",
    "METODOS",
    "Candidato",
    "Espacio",
    "buscar",
    "combinaciones",
    "comparar_busquedas",
    "crear_estimador",
    "espacio_de",
    "evaluar_candidato",
    "mejora",
    "muestra",
    "seleccionar_candidatos",
]
