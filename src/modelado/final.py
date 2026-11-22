"""Evaluacion final en el bloque de prueba: el unico paso que lo usa.

**Candidatos** (por tabla, leidos de los resultados ya escritos; no se vuelve a buscar):

- `baseline`: el estacional, y = lag_168.
- `lineal`: el de menor MAPE de validacion de la familia lineal. Se excluyen los
  modelos sin regularizacion cuya matriz de diseno tiene rango incompleto (la
  polinomial con LinearRegression): su solucion no es unica y su MAPE de CV se
  disparo (modelado_regresion.json). Ridge sobre los mismos polinomios si entra.
- `ensemble`: el de menor MAPE de validacion entre los ensembles (dos formas,
  hiperparametros fijos) y sus versiones de la busqueda (Grid y Random).
- `clasificador`: el de mayor PR-AUC de validacion entre los clasificadores y
  sus versiones de la busqueda; PR-AUC es el criterio de su paso.

**Reentreno.** Cada candidato se reajusta con entrenamiento + validacion
(2021 .. 2025-09-02) y los hiperparametros elegidos. La SVC RBF conserva su
submuestra de una de cada 5 h.

**Probabilidad de alerta.** La SVC no da probabilidades (`probability=True`
calibraria con una CV aleatoria interna de libsvm). Se usa una sigmoide de
Platt ajustada con los puntajes de validacion del mismo clasificador entrenado
solo con entrenamiento, y se aplica al puntaje del modelo final. Es una
aproximacion (el modelo final ve mas datos) y se anota; la decision de alerta
usa el puntaje y su umbral natural, 0.

**Eleccion del modelo final** (`elegir_final`), con validacion y antes de
mirar la prueba:

1. Son elegibles los candidatos de regresion que en validacion cumplen el
   criterio de exito: mejora de MAPE >= `CRITERIO_MEJORA_PCT` frente al
   baseline estacional y MAPE en horas pico menor que el del baseline.
2. Gana el de menor MAPE de validacion. Si el siguiente esta a menos de
   `UMBRAL_EMPATE_PUNTOS` puntos, se prefiere el mas interpretable y despues
   el mas rapido al predecir.
3. Si ninguno es elegible, gana el de menor MAPE de validacion si mejora al
   baseline; si ninguno lo mejora, el baseline. Queda anotado como hallazgo.

La prueba se evalua despues, una vez, y dice si el criterio se sostiene.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from modelado import clasificacion, ensembles, regresion
from modelado.baselines import BASELINE_REFERENCIA, BASELINES, BaselineRezago
from modelado.datos import registro_operacion
from modelado.metricas import HORAS_PICO, mape, metricas_clasificacion, metricas_regresion
from modelado.particion import inicio_prueba
from modelado.tiempos import medir_ajuste, medir_prediccion, tamano_serializado

CRITERIO_MEJORA_PCT = 20.0
UMBRAL_EMPATE_PUNTOS = 0.05
SIN_REGULARIZACION = ("lineal", "polinomial")
ROLES_REGRESION = ("baseline", "lineal", "ensemble")
ORDEN_INTERPRETABILIDAD = {"alta": 0, "media": 1, "baja": 2}
DIAS = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")
TIPOS_DIA = ("laboral", "fin_de_semana", "festivo", "semana_santa")


class ErrorFinal(Exception):
    """La evaluacion final no se puede hacer con seguridad."""


# --------------------------------------------------------------------------
# Candidatos
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidato:
    rol: str
    familia: str
    nombre: str
    forma: str
    origen: str
    hiperparametros: dict[str, Any] = field(default_factory=dict)
    # Metricas de validacion del paso en que se eligio (para la regla y la tabla).
    validacion: dict[str, Any] = field(default_factory=dict)

    @property
    def etiqueta(self) -> str:
        if self.rol == "baseline":
            return f"baseline {self.nombre} ({BASELINES[self.nombre]})"
        return self.nombre if self.forma == "-" else f"{self.nombre} ({self.forma})"

    @property
    def clave_calificacion(self) -> str:
        return f"baseline_{self.nombre}" if self.rol == "baseline" else self.nombre


def _tabla(resultados: dict[str, Any], familia: str, clave: str) -> dict[str, Any]:
    try:
        return resultados[familia]["tablas"][clave]
    except KeyError as error:
        raise ErrorFinal(f"Faltan los resultados de la familia {familia!r} para {clave}.") from error


def seleccionar_candidatos(resultados: dict[str, dict[str, Any]], clave: str) -> tuple[dict[str, Candidato], dict[str, Any]]:
    """Baseline, mejor lineal, mejor ensemble y mejor clasificador, con los criterios de validacion.

    `resultados`: {"baseline", "lineal", "ensembles", "clasificacion"} y, opcional, "busqueda".
    """
    base = _tabla(resultados, "baseline", clave)["modelos"][BASELINE_REFERENCIA]
    candidatos = {"baseline": Candidato("baseline", "baseline", BASELINE_REFERENCIA, "-", "modelado_baselines.json",
                                        {}, base["metricas_validacion"])}

    lineales, excluidos = [], {}
    for nombre, r in _tabla(resultados, "lineal", clave)["modelos"].items():
        diseno = r.get("condicionamiento") or {}
        if nombre in SIN_REGULARIZACION and diseno.get("rango", 0) < diseno.get("columnas", 0):
            excluidos[nombre] = (f"sin regularizacion y matriz de diseno de rango {diseno['rango']} < "
                                 f"{diseno['columnas']} columnas: solucion no unica, CV inestable")
            continue
        lineales.append(Candidato("lineal", "lineal", nombre, "-", "modelado_regresion.json",
                                  dict(r.get("hiperparametros") or {}), r["metricas_validacion"]))
    candidatos["lineal"] = min(lineales, key=lambda c: c.validacion["mape_pct"])

    arboles = []
    for nombre, formas in _tabla(resultados, "ensembles", clave)["modelos"].items():
        for forma, r in formas.items():
            arboles.append(Candidato("ensemble", "ensembles", nombre, forma, "modelado_ensembles.json", {},
                                     r["metricas_validacion"]))
    clasificadores = []
    for nombre, r in _tabla(resultados, "clasificacion", clave)["modelos"].items():
        clasificadores.append(Candidato("clasificador", "clasificacion", nombre, "-", "modelado_clasificacion.json",
                                        dict(r.get("hiperparametros") or {}), r["metricas_validacion"]))
    busqueda = (resultados.get("busqueda") or {}).get("tablas", {}).get(clave)
    for r in (busqueda or {}).get("modelos", {}).values():
        for metodo in ("grid", "random"):
            m = r[metodo]
            origen = f"modelado_busqueda.json ({metodo})"
            if r["familia"] == "ensembles":
                arboles.append(Candidato("ensemble", "ensembles", r["nombre"], r["forma"], origen,
                                         dict(m["mejores"]), m["metricas_validacion"]))
            elif r["familia"] == "clasificacion":
                clasificadores.append(Candidato("clasificador", "clasificacion", r["nombre"], "-", origen,
                                                dict(m["mejores"]), m["metricas_validacion"]))
    candidatos["ensemble"] = min(arboles, key=lambda c: c.validacion["mape_pct"])
    candidatos["clasificador"] = max(clasificadores, key=lambda c: c.validacion["pr_auc"])

    registro = registro_operacion(
        "seleccionar_candidatos",
        "lineal y ensemble por menor MAPE de validacion; clasificador por mayor PR-AUC de validacion; "
        "sin volver a buscar hiperparametros",
        candidatos={rol: asdict(c) for rol, c in candidatos.items()},
        lineales_excluidos=excluidos,
        ensembles_considerados=len(arboles),
        clasificadores_considerados=len(clasificadores),
        busqueda_incluida=busqueda is not None,
    )
    return candidatos, registro


def crear_estimador(candidato: Candidato) -> BaseEstimator:
    """El estimador sin ajustar, con los hiperparametros elegidos."""
    if candidato.rol == "baseline":
        return BaselineRezago(BASELINES[candidato.nombre])
    if candidato.familia == "lineal":
        return regresion.crear_pipeline(candidato.nombre, **candidato.hiperparametros)
    if candidato.familia == "ensembles":
        modelo = ensembles.crear_modelo(candidato.nombre, candidato.forma)
        prefijo = "estimador__" if candidato.forma == "diferenciado" else ""
        return modelo.set_params(**{prefijo + k: v for k, v in candidato.hiperparametros.items()})
    if candidato.familia == "clasificacion":
        return clasificacion.crear_pipeline(candidato.nombre, **candidato.hiperparametros)
    raise ErrorFinal(f"Familia desconocida: {candidato.familia!r}")


# --------------------------------------------------------------------------
# Reentreno con entrenamiento + validacion
# --------------------------------------------------------------------------


def filas_de_ajuste(indices: dict[str, np.ndarray]) -> np.ndarray:
    """Entrenamiento + validacion, contiguos y anteriores a la prueba."""
    filas = np.concatenate([indices["entrenamiento"], indices["validacion"]])
    if np.any(np.diff(filas) != 1):
        raise ErrorFinal("Entrenamiento y validacion no forman un tramo continuo.")
    if len(indices["prueba"]) and filas.max() >= indices["prueba"].min():
        raise ErrorFinal("El ajuste final llega al bloque de prueba.")
    return filas


def _filas_clasificador(candidato: Candidato, objetivo: np.ndarray, filas: np.ndarray) -> np.ndarray:
    filas = filas[~np.isnan(objetivo[filas])]
    paso = clasificacion.MODELOS[candidato.nombre].submuestra
    return filas[::paso]


def reentrenar(
    candidato: Candidato,
    X: pd.DataFrame,
    y: pd.Series,
    objetivo: np.ndarray | None,
    fechas: pd.Series,
    indices: dict[str, np.ndarray],
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Ajusta el candidato con entrenamiento + validacion y mide el ajuste."""
    filas = filas_de_ajuste(indices)
    if candidato.rol == "clasificador":
        if objetivo is None:
            raise ErrorFinal("Falta alerta_pico para el clasificador.")
        filas = _filas_clasificador(candidato, np.asarray(objetivo, dtype="float64"), filas)
        objetivo_ajuste = np.asarray(objetivo, dtype="float64")[filas].astype(int)
    else:
        objetivo_ajuste = y.iloc[filas].to_numpy(dtype="float64")
    if fechas.iloc[filas].max() >= inicio_prueba():
        raise ErrorFinal("Filas del bloque de prueba en el ajuste final.")  # defensa
    modelo, ajuste = medir_ajuste(crear_estimador(candidato), X.iloc[filas].reset_index(drop=True), objetivo_ajuste)
    if candidato.familia == "ensembles":
        ensembles.ajustar_prediccion_en_un_hilo(modelo)
    return modelo, {
        "filas": int(len(filas)),
        "desde": str(fechas.iloc[filas[0]]),
        "hasta": str(fechas.iloc[filas[-1]]),
        "submuestra_paso_h": clasificacion.MODELOS[candidato.nombre].submuestra if candidato.rol == "clasificador" else 1,
        "ajuste_s": ajuste["detalle"]["segundos"],
    }


def calibrador_platt(
    candidato: Candidato, X: pd.DataFrame, objetivo: np.ndarray, indices: dict[str, np.ndarray]
) -> tuple[LogisticRegression, dict[str, Any]]:
    """Sigmoide de Platt: puntajes de validacion del clasificador entrenado solo con entrenamiento."""
    objetivo = np.asarray(objetivo, dtype="float64")
    entrena = _filas_clasificador(candidato, objetivo, indices["entrenamiento"])
    valida = indices["validacion"]
    previo = crear_estimador(candidato).fit(X.iloc[entrena], objetivo[entrena].astype(int))
    puntajes, _, metodo = clasificacion.puntaje(previo, X.iloc[valida])
    calibrador = LogisticRegression(C=1e6, max_iter=1000).fit(puntajes.reshape(-1, 1), objetivo[valida].astype(int))
    return calibrador, {
        "metodo": "sigmoide de Platt (LogisticRegression sobre el puntaje)",
        "puntaje": metodo,
        "ajustada_con": "puntajes de validacion del clasificador entrenado solo con entrenamiento",
        "pendiente": float(calibrador.coef_[0, 0]),
        "intercepto": float(calibrador.intercept_[0]),
        "nota": "aproximacion: se aplica al puntaje del modelo final, que ademas vio la validacion",
    }


# --------------------------------------------------------------------------
# Eleccion del modelo final (solo con validacion)
# --------------------------------------------------------------------------


def cumple_criterio(metricas: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Mejora de MAPE >= CRITERIO_MEJORA_PCT y MAPE en horas pico por debajo del baseline."""
    mejora = 100.0 * (baseline["mape_pct"] - metricas["mape_pct"]) / baseline["mape_pct"]
    pico = metricas["mape_pico_pct"] < baseline["mape_pico_pct"]
    return {
        "mejora_mape_pct": mejora,
        "mejora_suficiente": mejora >= CRITERIO_MEJORA_PCT,
        "mape_pico_pct": metricas["mape_pico_pct"],
        "mape_pico_baseline_pct": baseline["mape_pico_pct"],
        "ventaja_en_pico": pico,
        "cumple": bool(mejora >= CRITERIO_MEJORA_PCT and pico),
    }


def tiempos_prediccion(modelos: dict[str, BaseEstimator], X_validacion: pd.DataFrame) -> dict[str, float]:
    """ms por 1000 filas de cada modelo de regresion, medidos en validacion (para desempatar sin mirar la prueba)."""
    return {rol: medir_prediccion(modelos[rol], X_validacion)[1]["detalle"]["ms_por_1000_filas"]
            for rol in ROLES_REGRESION if rol in modelos}


def elegir_final(
    candidatos: dict[str, Candidato], calificaciones: dict[str, dict[str, str]], prediccion_ms: dict[str, float]
) -> tuple[str, dict[str, Any]]:
    """Rol del modelo final segun la regla del modulo, solo con metricas de validacion.

    `prediccion_ms` (solo para desempatar) tiene que medirse en validacion: ver `tiempos_prediccion`.
    """
    base = candidatos["baseline"].validacion
    evaluados = {}
    for rol in ROLES_REGRESION:
        if rol == "baseline":
            continue
        c = candidatos[rol]
        evaluados[rol] = {
            "mape_validacion_pct": c.validacion["mape_pct"],
            "criterio_validacion": cumple_criterio(c.validacion, base),
            "interpretabilidad": calificaciones.get(c.clave_calificacion, {}).get("calificacion"),
            "prediccion_ms_por_1000_filas": prediccion_ms.get(rol),
        }
    elegibles = [rol for rol, e in evaluados.items() if e["criterio_validacion"]["cumple"]]
    nota = None
    if elegibles:
        orden = sorted(elegibles, key=lambda r: evaluados[r]["mape_validacion_pct"])
        mejor = orden[0]
        empatados = [r for r in orden
                     if evaluados[r]["mape_validacion_pct"] - evaluados[mejor]["mape_validacion_pct"] < UMBRAL_EMPATE_PUNTOS]
        if len(empatados) > 1:
            mejor = min(empatados, key=lambda r: (ORDEN_INTERPRETABILIDAD.get(evaluados[r]["interpretabilidad"], 3),
                                                  evaluados[r]["prediccion_ms_por_1000_filas"] or np.inf))
            nota = f"empate a menos de {UMBRAL_EMPATE_PUNTOS} puntos entre {empatados}: decide la interpretabilidad"
    else:
        mejoran = [r for r, e in evaluados.items() if e["mape_validacion_pct"] < base["mape_pct"]]
        if mejoran:
            mejor = min(mejoran, key=lambda r: evaluados[r]["mape_validacion_pct"])
            nota = "ningun candidato cumple el criterio en validacion: gana el de menor MAPE que mejora al baseline"
        else:
            mejor = "baseline"
            nota = "ningun candidato mejora al baseline estacional en validacion: el modelo final es el baseline"
    return mejor, {
        "regla": (
            f"elegibles: mejora de MAPE >= {CRITERIO_MEJORA_PCT:.0f} % y MAPE en horas pico menor que el "
            f"del baseline, en validacion; gana el menor MAPE de validacion; empate a menos de "
            f"{UMBRAL_EMPATE_PUNTOS} puntos: mas interpretable y despues mas rapido al predecir (medido en validacion)"
        ),
        "bloque": "validacion (la prueba no interviene en la eleccion)",
        "desempate_tiempo": "ms por 1000 filas medidos en validacion",
        "evaluados": evaluados,
        "elegibles": elegibles,
        "elegido": mejor,
        "nota": nota,
    }


# --------------------------------------------------------------------------
# Residuales
# --------------------------------------------------------------------------


def tipo_de_dia(fechas: pd.Series, X: pd.DataFrame) -> np.ndarray:
    """semana_santa > festivo > fin_de_semana > laboral (la primera que aplique)."""
    momentos = pd.DatetimeIndex(fechas)
    tipo = np.where(momentos.dayofweek >= 5, "fin_de_semana", "laboral").astype(object)
    tipo[X["es_festivo"].to_numpy() == 1] = "festivo"
    tipo[X["es_semana_santa"].to_numpy() == 1] = "semana_santa"
    return tipo


def residuales(
    y: pd.Series, prediccion: np.ndarray, baseline: np.ndarray, fechas: pd.Series, X: pd.DataFrame
) -> dict[str, Any]:
    """MAE, MAPE y sesgo del modelo (y MAPE del baseline) por hora, dia de la semana y tipo de dia."""
    real = y.to_numpy(dtype="float64")
    pred = np.asarray(prediccion, dtype="float64")
    base = np.asarray(baseline, dtype="float64")
    momentos = pd.DatetimeIndex(fechas)
    grupos = {
        "hora": (momentos.hour.to_numpy(), list(range(24))),
        "dia_semana": (np.array([DIAS[d] for d in momentos.dayofweek]), list(DIAS)),
        "tipo_dia": (tipo_de_dia(fechas, X), list(TIPOS_DIA)),
    }
    resultado: dict[str, Any] = {"definicion_residual": "prediccion - real (negativo: subestima)"}
    for nombre, (etiquetas, orden) in grupos.items():
        filas = {}
        for valor in orden:
            m = etiquetas == valor
            if not m.any():
                continue
            filas[str(valor)] = {
                "n": int(m.sum()),
                "mae_mwh": float(np.abs(pred[m] - real[m]).mean()),
                "mape_pct": mape(real[m], pred[m]),
                "sesgo_mwh": float((pred[m] - real[m]).mean()),
                "mape_baseline_pct": mape(real[m], base[m]),
            }
        resultado[nombre] = filas
    return resultado


# --------------------------------------------------------------------------
# Evaluacion en prueba
# --------------------------------------------------------------------------


def evaluar_prueba(
    candidatos: dict[str, Candidato],
    modelos: dict[str, BaseEstimator],
    calibrador: LogisticRegression | None,
    X: pd.DataFrame,
    y: pd.Series,
    objetivo: np.ndarray | None,
    fechas: pd.Series,
    indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Metricas en prueba de cada candidato. Un MAPE < 1 % lanza AlertaFuga antes de devolver nada."""
    prueba = indices["prueba"]
    if len(prueba) == 0:
        raise ErrorFinal("No hay bloque de prueba.")
    X_prueba, y_prueba, fechas_prueba = X.iloc[prueba], y.iloc[prueba], fechas.iloc[prueba]
    if y_prueba.isna().any():
        raise ErrorFinal("La prueba tiene demanda sin dato.")

    predicciones, medidas, metricas = {}, {}, {}
    for rol in ROLES_REGRESION:
        predicciones[rol], medida = medir_prediccion(modelos[rol], X_prueba)
        medidas[rol] = medida["detalle"]["ms_por_1000_filas"]
    mape_base = mape(y_prueba, predicciones["baseline"])
    for rol in ROLES_REGRESION:
        metricas[rol] = metricas_regresion(
            y_prueba, predicciones[rol], fechas_prueba, mape_baseline=None if rol == "baseline" else mape_base
        )
    # Los arboles no pasan del maximo que vieron: cuantas horas de prueba lo superan y que sesgo hay en ellas.
    maximo = float(y.iloc[filas_de_ajuste(indices)].max())
    extrapolacion = {rol: ensembles.extrapolacion(y_prueba, predicciones[rol], maximo) for rol in ROLES_REGRESION}

    clasificador = None
    if "clasificador" in modelos:
        real = np.asarray(objetivo, dtype="float64")[prueba]
        if np.isnan(real).any():
            raise ErrorFinal("La prueba tiene filas sin alerta_pico.")
        _, medida = medir_prediccion(modelos["clasificador"], X_prueba)
        medidas["clasificador"] = medida["detalle"]["ms_por_1000_filas"]
        puntajes, umbral, metodo = clasificacion.puntaje(modelos["clasificador"], X_prueba)
        probabilidad = calibrador.predict_proba(puntajes.reshape(-1, 1))[:, 1] if calibrador is not None else None
        clasificador = {
            "puntajes": puntajes,
            "probabilidad": probabilidad,
            "umbral": umbral,
            "real": real.astype(int),
            "metricas": {
                **metricas_clasificacion(real.astype(int), puntajes, umbral=umbral),
                "puntaje": metodo,
                "umbral_decision": umbral,
                "prevalencia": float(real.mean()),
                "brier_probabilidad": None if probabilidad is None else float(np.mean((probabilidad - real) ** 2)),
            },
        }
    return {"predicciones": predicciones, "prediccion_ms": medidas, "metricas": metricas, "clasificador": clasificador,
            "mape_baseline_pct": mape_base, "extrapolacion": extrapolacion}


def costo_rezago(final_por_tabla: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Cuanto empeora el modelo final y cada rol de r24 a r120, en prueba."""
    if not {"r24", "r120"} <= set(final_por_tabla):
        return {}
    a, b = final_por_tabla["r24"], final_por_tabla["r120"]
    resultado = {}
    for rol in ROLES_REGRESION:
        ma, mb = a["metricas_prueba"][rol]["mape_pct"], b["metricas_prueba"][rol]["mape_pct"]
        resultado[rol] = {
            "modelo_r24": a["candidatos"][rol], "modelo_r120": b["candidatos"][rol],
            "mape_r24_pct": ma, "mape_r120_pct": mb,
            "aumento_puntos": mb - ma, "aumento_relativo_pct": 100.0 * (mb - ma) / ma,
        }
    ma, mb = a["final"]["metricas_prueba"]["mape_pct"], b["final"]["metricas_prueba"]["mape_pct"]
    resultado["final"] = {
        "modelo_r24": a["final"]["etiqueta"], "modelo_r120": b["final"]["etiqueta"],
        "mape_r24_pct": ma, "mape_r120_pct": mb,
        "aumento_puntos": mb - ma, "aumento_relativo_pct": 100.0 * (mb - ma) / ma,
    }
    return resultado


__all__ = [
    "CRITERIO_MEJORA_PCT",
    "Candidato",
    "calibrador_platt",
    "costo_rezago",
    "crear_estimador",
    "cumple_criterio",
    "elegir_final",
    "evaluar_prueba",
    "filas_de_ajuste",
    "reentrenar",
    "residuales",
    "seleccionar_candidatos",
    "tamano_serializado",
    "tiempos_prediccion",
    "tipo_de_dia",
]
