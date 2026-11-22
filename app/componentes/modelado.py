"""Tablas de las paginas de modelado, armadas desde docs/resultados/modelado_*.json.

Funciones puras (sin Streamlit) para poder probarlas: reciben los JSON ya
leidos y devuelven DataFrames. No recalculan metricas: las copian de los JSON,
salvo las que dependen de un control de la pagina (el umbral de la alerta o la
semana elegida), que se calculan sobre las predicciones exportadas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from componentes.carga import obtener

NOMBRE_FAMILIA = {
    "baseline": "Baseline",
    "lineal": "Lineal",
    "ensembles": "Ensemble",
    "busqueda": "Búsqueda",
}


def _fila(familia: str, modelo: str, forma: str, entrada: dict[str, Any], metricas: dict[str, Any] | None) -> dict[str, Any]:
    metricas = metricas or {}
    resumen = obtener(entrada, "validacion_cruzada", "detalle", "resumen", "mape_pct") or {}
    tiempos = entrada.get("tiempos") or {}
    return {
        "familia": NOMBRE_FAMILIA.get(familia, familia),
        "modelo": modelo,
        "forma": forma,
        "mape_pct": metricas.get("mape_pct"),
        "mape_pico_pct": metricas.get("mape_pico_pct"),
        "mejora_mape_pct": metricas.get("mejora_mape_pct"),
        "mae_mwh": metricas.get("mae_mwh"),
        "rmse_mwh": metricas.get("rmse_mwh"),
        "sesgo_medio_mwh": metricas.get("sesgo_medio_mwh"),
        "error_maximo_diario_pct": metricas.get("error_maximo_diario_pct"),
        "mape_cv_media_pct": resumen.get("media"),
        "mape_cv_desviacion_pct": resumen.get("desviacion"),
        "ajuste_s": tiempos.get("ajuste_s"),
        "prediccion_ms_por_1000_filas": tiempos.get("prediccion_ms_por_1000_filas"),
    }


def tabla_regresion(resultados: dict[str, dict | None], version: str) -> pd.DataFrame:
    """Una fila por modelo de regresion evaluado en validacion, de todas las familias."""
    filas: list[dict[str, Any]] = []
    base = obtener(resultados.get("baseline"), "tablas", version, "modelos") or {}
    for nombre, entrada in base.items():
        filas.append(_fila("baseline", nombre, "-", entrada, entrada.get("metricas_validacion")))
    lineal = obtener(resultados.get("lineal"), "tablas", version, "modelos") or {}
    for nombre, entrada in lineal.items():
        filas.append(_fila("lineal", nombre, "-", entrada, entrada.get("metricas_validacion")))
    ensembles = obtener(resultados.get("ensembles"), "tablas", version, "modelos") or {}
    for nombre, formas in ensembles.items():
        for forma, entrada in formas.items():
            filas.append(_fila("ensembles", nombre, forma, entrada, entrada.get("metricas_validacion")))
    busqueda = obtener(resultados.get("busqueda"), "tablas", version, "modelos") or {}
    for clave, entrada in busqueda.items():
        if entrada.get("tipo") != "regresion":
            continue
        for metodo in ("grid", "random"):
            if metodo in entrada:
                fila = _fila("busqueda", f"{entrada.get('nombre', clave)} · {metodo}", entrada.get("forma", "-"),
                             entrada[metodo], entrada[metodo].get("metricas_validacion"))
                filas.append(fila)
    return pd.DataFrame(filas)


def tabla_clasificacion(clasificacion: dict | None, version: str) -> pd.DataFrame:
    tabla = obtener(clasificacion, "tablas", version) or {}
    filas = []
    entradas = {"baseline (lag_168)": tabla.get("baseline")} | (tabla.get("modelos") or {})
    for nombre, entrada in entradas.items():
        if not entrada:
            continue
        m = entrada.get("metricas_validacion") or {}
        filas.append({
            "modelo": nombre,
            "pr_auc": m.get("pr_auc"),
            "roc_auc": m.get("roc_auc"),
            "f1": m.get("f1"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "positivos": m.get("positivos"),
            "n": m.get("n"),
        })
    return pd.DataFrame(filas)


def pliegues(resultados: dict[str, dict | None], version: str) -> pd.DataFrame:
    """MAPE de cada pliegue de TimeSeriesSplit, en formato largo (modelo, pliegue, mape_pct)."""
    filas = []
    lineal = obtener(resultados.get("lineal"), "tablas", version, "modelos") or {}
    fuentes = [(f"{n}", e) for n, e in lineal.items()]
    ensembles = obtener(resultados.get("ensembles"), "tablas", version, "modelos") or {}
    fuentes += [(f"{n} ({forma})", e) for n, formas in ensembles.items() for forma, e in formas.items()]
    for modelo, entrada in fuentes:
        valores = obtener(entrada, "validacion_cruzada", "detalle", "resumen", "mape_pct", "por_pliegue") or []
        filas += [{"modelo": modelo, "pliegue": i + 1, "mape_pct": v} for i, v in enumerate(valores)]
    return pd.DataFrame(filas, columns=["modelo", "pliegue", "mape_pct"])


def trayectorias_busqueda(busqueda: dict | None, version: str) -> pd.DataFrame:
    """Mejor puntaje de CV acumulado frente al computo, para Grid y Random de cada candidato."""
    filas = []
    for clave, entrada in (obtener(busqueda, "tablas", version, "modelos") or {}).items():
        regresion = entrada.get("tipo") == "regresion"
        for metodo in ("grid", "random"):
            for punto in obtener(entrada, metodo, "trayectoria") or []:
                mejor = punto.get("mejor_puntaje_cv_acumulado")
                filas.append({
                    "candidato": clave,
                    "metodo": "GridSearchCV" if metodo == "grid" else "RandomizedSearchCV",
                    "ajustes_cv": punto.get("ajustes_cv"),
                    "computo_s": punto.get("computo_acumulado_s"),
                    # neg_mean_absolute_error -> MAE en MWh; en clasificacion el puntaje se deja tal cual
                    "mejor_cv": -mejor if (regresion and mejor is not None) else mejor,
                })
    return pd.DataFrame(filas, columns=["candidato", "metodo", "ajustes_cv", "computo_s", "mejor_cv"])


def resumen_busqueda(busqueda: dict | None, version: str) -> pd.DataFrame:
    filas = []
    for clave, entrada in (obtener(busqueda, "tablas", version, "modelos") or {}).items():
        comp = entrada.get("comparacion") or {}
        segundos = comp.get("segundos_busqueda") or {}
        filas.append({
            "candidato": clave,
            "tipo": entrada.get("tipo"),
            "combinaciones de la rejilla": entrada.get("combinaciones_rejilla"),
            "mejores (Grid)": str(obtener(entrada, "grid", "mejores")),
            "mejores (Random)": str(obtener(entrada, "random", "mejores")),
            "Random alcanza a Grid": comp.get("random_alcanza_grid"),
            "s de reloj (Grid)": segundos.get("grid"),
            "s de reloj (Random)": segundos.get("random"),
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Prediccion y alertas
# --------------------------------------------------------------------------


def mape(real: pd.Series, prediccion: pd.Series) -> float:
    return float(100 * np.mean(np.abs(prediccion - real) / real))


def error_hora_dia(predicciones: pd.DataFrame) -> pd.DataFrame:
    """MAPE (%) por dia de la semana (filas, 0 = lunes) y hora (columnas)."""
    tiempo = pd.to_datetime(predicciones["fecha_hora"])
    ape = 100 * (predicciones["prediccion_mwh"] - predicciones["real_mwh"]).abs() / predicciones["real_mwh"]
    return ape.groupby([tiempo.dt.dayofweek.rename("dia"), tiempo.dt.hour.rename("hora")]).mean().unstack("hora")


def matriz_confusion(real: np.ndarray, probabilidad: np.ndarray, umbral: float) -> dict[str, float]:
    """Alerta si probabilidad >= umbral. Misma orientacion que los JSON: filas = real, columnas = predicha."""
    predicha = probabilidad >= umbral
    real = real.astype(bool)
    vp = int((predicha & real).sum())
    fp = int((predicha & ~real).sum())
    fn = int((~predicha & real).sum())
    vn = int((~predicha & ~real).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"vp": vp, "fp": fp, "fn": fn, "vn": vn, "precision": precision, "recall": recall, "f1": f1}


def curva_precision_recall(real: np.ndarray, puntaje: np.ndarray) -> pd.DataFrame:
    """Precision y recall para cada umbral posible (de mayor a menor puntaje)."""
    orden = np.argsort(-puntaje, kind="mergesort")
    puntaje, real = puntaje[orden], real[orden].astype(bool)
    vp = np.cumsum(real)
    fp = np.cumsum(~real)
    ultimos = np.r_[np.flatnonzero(np.diff(puntaje)), len(puntaje) - 1]  # un punto por umbral distinto
    vp, fp = vp[ultimos], fp[ultimos]
    total = real.sum()
    return pd.DataFrame({
        "umbral": puntaje[ultimos],
        "precision": vp / (vp + fp),
        "recall": vp / total if total else np.zeros_like(vp, dtype=float),
    })


def probabilidad_de_puntaje(puntaje: float, calibracion: dict | None) -> float | None:
    """Sigmoide de Platt guardada en modelado_seleccion.json: p = 1 / (1 + exp(-(a * puntaje + b)))."""
    if not calibracion or calibracion.get("pendiente") is None or calibracion.get("intercepto") is None:
        return None
    return float(1 / (1 + np.exp(-(calibracion["pendiente"] * puntaje + calibracion["intercepto"]))))
