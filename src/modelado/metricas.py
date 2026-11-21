"""Metricas de evaluacion: regresion de la demanda horaria y clasificacion.

Regresion, en MWh por hora:

- MAE, RMSE y MAPE (%);
- MAPE solo en las horas pico (18, 19 y 20 h locales), donde un error cuesta mas;
- error sobre el maximo diario: el pico de cada dia predicho frente al real;
- sesgo medio = media(prediccion - real): **negativo si el modelo subestima**;
- mejora porcentual de MAPE frente a un baseline: positiva si el modelo es mejor.

`metricas_regresion` hace sonar la alarma de fuga: con lo que se sabe de esta
serie, un MAPE por debajo del 1 % no es un buen modelo sino el objetivo colado
en X (CLAUDE.md). Lanza `AlertaFuga` en vez de devolver la cifra.

Clasificacion: precision, recall, F1, ROC-AUC, PR-AUC (average precision) y
matriz de confusion, con el umbral explicito.

Funciones puras: reciben arreglos y devuelven numeros o diccionarios
serializables a JSON. Ninguna alinea por indice de pandas, todo se convierte a
arreglos: y_true, y_pred y fechas tienen que venir en el mismo orden.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from modelado.datos import ZONA_HORARIA

HORAS_PICO = (18, 19, 20)
UMBRAL_MAPE_FUGA_PCT = 1.0
UMBRAL_CLASIFICACION = 0.5


class ErrorMetricas(ValueError):
    """Las entradas de una metrica no son validas."""


class AlertaFuga(Exception):
    """MAPE por debajo del umbral de fuga: hay que auditar antes de seguir."""


# --------------------------------------------------------------------------
# Entradas
# --------------------------------------------------------------------------


def _arreglos(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray(y_true, dtype="float64").ravel()
    pred = np.asarray(y_pred, dtype="float64").ravel()
    if real.shape != pred.shape:
        raise ErrorMetricas(f"y_true tiene {real.size} valores e y_pred {pred.size}.")
    if real.size == 0:
        raise ErrorMetricas("No hay valores que evaluar.")
    if not (np.isfinite(real).all() and np.isfinite(pred).all()):
        raise ErrorMetricas("Hay NaN o infinitos en y_true o y_pred.")
    return real, pred


def _momentos(fechas, n: int) -> pd.DatetimeIndex:
    """Marcas en hora local. Sin zona horaria se toman como ya locales."""
    momentos = pd.DatetimeIndex(fechas)
    if len(momentos) != n:
        raise ErrorMetricas(f"Hay {len(momentos)} fechas para {n} valores.")
    return momentos.tz_convert(ZONA_HORARIA) if momentos.tz is not None else momentos


# --------------------------------------------------------------------------
# Regresion
# --------------------------------------------------------------------------


def mae(y_true, y_pred) -> float:
    real, pred = _arreglos(y_true, y_pred)
    return float(np.mean(np.abs(pred - real)))


def rmse(y_true, y_pred) -> float:
    real, pred = _arreglos(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def mape(y_true, y_pred) -> float:
    """Error porcentual absoluto medio, en %. La demanda real tiene que ser positiva."""
    real, pred = _arreglos(y_true, y_pred)
    if (real <= 0).any():
        raise ErrorMetricas("MAPE no esta definido con valores reales <= 0.")
    return float(100.0 * np.mean(np.abs(pred - real) / real))


def sesgo_medio(y_true, y_pred) -> float:
    """media(prediccion - real), en MWh. Negativo: el modelo subestima."""
    real, pred = _arreglos(y_true, y_pred)
    return float(np.mean(pred - real))


def mape_horas_pico(y_true, y_pred, fechas, horas: tuple[int, ...] = HORAS_PICO) -> float:
    """MAPE solo en las horas locales `horas` (por defecto 18, 19 y 20 h)."""
    real, pred = _arreglos(y_true, y_pred)
    en_pico = np.isin(_momentos(fechas, real.size).hour, horas)
    if not en_pico.any():
        raise ErrorMetricas(f"Ninguna fila cae en las horas pico {list(horas)}.")
    return mape(real[en_pico], pred[en_pico])


def error_maximo_diario(y_true, y_pred, fechas) -> dict[str, float | int]:
    """Error al predecir el maximo de cada dia local: max(pred del dia) frente a max(real del dia).

    No exige que los dos maximos caigan a la misma hora: mide el nivel del pico,
    que es lo que dimensiona la generacion.
    """
    real, pred = _arreglos(y_true, y_pred)
    dias = _momentos(fechas, real.size).normalize()
    maximos = pd.DataFrame({"real": real, "pred": pred}, index=dias).groupby(level=0).max()
    return {
        "mae_mwh": mae(maximos["real"], maximos["pred"]),
        "mape_pct": mape(maximos["real"], maximos["pred"]),
        "n_dias": int(len(maximos)),
    }


def mejora_mape(mape_modelo: float, mape_baseline: float) -> float:
    """100 * (MAPE_baseline - MAPE_modelo) / MAPE_baseline. Positiva si el modelo es mejor."""
    if not mape_baseline > 0:
        raise ErrorMetricas(f"El MAPE del baseline tiene que ser positivo: {mape_baseline}")
    return float(100.0 * (mape_baseline - mape_modelo) / mape_baseline)


def verificar_fuga(mape_pct: float, umbral: float = UMBRAL_MAPE_FUGA_PCT) -> None:
    """Lanza AlertaFuga si el MAPE baja del umbral (1 %)."""
    if mape_pct < umbral:
        raise AlertaFuga(
            f"MAPE = {mape_pct:.3f} % < {umbral} %: con esta serie eso indica que el "
            "objetivo o algo derivado de el se colo en X. Audita la fuga antes de "
            "seguir (CLAUDE.md)."
        )


def metricas_regresion(
    y_true,
    y_pred,
    fechas,
    mape_baseline: float | None = None,
    alarma_fuga: bool = True,
) -> dict[str, Any]:
    """Todas las metricas de regresion de una prediccion, listas para el registro.

    Con `mape_baseline` anade la mejora porcentual frente a ese baseline. Con
    `alarma_fuga` (por defecto) un MAPE < 1 % lanza AlertaFuga.
    """
    real, pred = _arreglos(y_true, y_pred)
    momentos = _momentos(fechas, real.size)
    valor_mape = mape(real, pred)
    if alarma_fuga:
        verificar_fuga(valor_mape)

    maximo = error_maximo_diario(real, pred, momentos)
    sesgo = sesgo_medio(real, pred)
    return {
        "n": int(real.size),
        "mae_mwh": mae(real, pred),
        "rmse_mwh": rmse(real, pred),
        "mape_pct": valor_mape,
        "mape_pico_pct": mape_horas_pico(real, pred, momentos),
        "horas_pico": list(HORAS_PICO),
        "n_pico": int(np.isin(momentos.hour, HORAS_PICO).sum()),
        "error_maximo_diario_mwh": maximo["mae_mwh"],
        "error_maximo_diario_pct": maximo["mape_pct"],
        "n_dias": maximo["n_dias"],
        "sesgo_medio_mwh": sesgo,
        "sesgo_medio_pct": float(100.0 * sesgo / np.mean(real)),
        "mejora_mape_pct": None if mape_baseline is None else mejora_mape(valor_mape, mape_baseline),
    }


# --------------------------------------------------------------------------
# Clasificacion
# --------------------------------------------------------------------------


def metricas_clasificacion(
    y_true, y_score, umbral: float = UMBRAL_CLASIFICACION
) -> dict[str, Any]:
    """Metricas de una clasificacion binaria a partir de puntajes (probabilidad de la clase 1).

    La clase predicha es `y_score >= umbral`. Precision, recall o F1 sin
    denominador (ningun positivo predicho o real) quedan en None; ROC-AUC y
    PR-AUC tambien, si y_true tiene una sola clase.
    """
    real = np.asarray(y_true).ravel()
    puntaje = np.asarray(y_score, dtype="float64").ravel()
    if real.shape != puntaje.shape or real.size == 0:
        raise ErrorMetricas(f"y_true tiene {real.size} valores e y_score {puntaje.size}.")
    if not np.isfinite(puntaje).all():
        raise ErrorMetricas("Hay NaN o infinitos en y_score.")
    if not set(np.unique(real).tolist()) <= {0, 1}:
        raise ErrorMetricas(f"y_true tiene que ser 0/1: {sorted(set(np.unique(real).tolist()))[:5]}")
    real = real.astype(int)

    prediccion = (puntaje >= umbral).astype(int)
    vn, fp, fn, vp = (int(v) for v in confusion_matrix(real, prediccion, labels=[0, 1]).ravel())
    precision = vp / (vp + fp) if vp + fp else None
    recall = vp / (vp + fn) if vp + fn else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    dos_clases = len(np.unique(real)) == 2
    return {
        "n": int(real.size),
        "positivos": int(real.sum()),
        "umbral": float(umbral),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(real, puntaje)) if dos_clases else None,
        "pr_auc": float(average_precision_score(real, puntaje)) if dos_clases else None,
        "pr_auc_metodo": "average precision (sklearn.metrics.average_precision_score)",
        "matriz_confusion": {
            "vn": vn,
            "fp": fp,
            "fn": fn,
            "vp": vp,
            "matriz": [[vn, fp], [fn, vp]],
            "orden": "filas = clase real (0, 1); columnas = clase predicha (0, 1)",
        },
        "nota": None if dos_clases else "una sola clase en y_true: ROC-AUC y PR-AUC no estan definidos",
    }
