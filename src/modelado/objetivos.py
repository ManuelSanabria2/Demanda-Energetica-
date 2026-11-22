"""Objetivos derivados de la demanda para la clasificacion.

**alerta_pico(t) = 1 si demanda_mwh(t) > umbral(t), 0 si no**, donde umbral(t)
es el percentil 90 de la demanda en la ventana de 168 h que acaba en
t - rezago_minimo_h: la misma ventana que `ma_168` y `std_168`, es decir
`demanda(t - r - 167 h) .. demanda(t - r)`.

El umbral solo usa demanda ya publicada al predecir t, asi que se puede
conocer de antemano; lo que se predice es si la hora t lo superara. Igual que en
`variables.construir`, se calcula **por marca de tiempo**: se desplaza la serie
`r` horas y despues se aplica la ventana movil de 168 h, con la ventana completa
exigida (`min_periods=168`) sobre una rejilla horaria verificada.

Como la tabla de modelado empieza donde `ma_168` ya tiene ventana, pero aqui el
umbral se recalcula con la demanda de la propia tabla, sus primeras
`r + 167` filas no tienen ventana completa: quedan sin objetivo (NaN) y el
registro las cuenta. Son filas del arranque del entrenamiento.

La proporcion de positivos se reporta para entrenamiento y validacion. La del
bloque de prueba **no se calcula**: mirarla seria usar el objetivo de la prueba
antes de la evaluacion final (CLAUDE.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modelado.datos import ZONA_HORARIA, registro_operacion, rezago_de, verificar_rejilla_horaria
from variables.construir import VENTANA_MOVIL_H

PERCENTIL_ALERTA = 90
COLUMNA_ALERTA = "alerta_pico"
COLUMNA_UMBRAL = "umbral_alerta_mwh"
BLOQUES_VISIBLES = ("entrenamiento", "validacion")


class ErrorObjetivo(Exception):
    """El objetivo no se puede construir con seguridad."""


def umbral_movil(
    fechas: pd.Series | pd.DatetimeIndex,
    demanda: pd.Series | np.ndarray,
    rezago_minimo_h: int | str,
    percentil: float = PERCENTIL_ALERTA,
    ventana_h: int = VENTANA_MOVIL_H,
) -> pd.Series:
    """Percentil `percentil` de la demanda en las `ventana_h` horas que acaban en t - r.

    Por marca de tiempo: desplazar `r` horas y despues la ventana movil de
    tiempo, igual que `ma_168`. Interpolacion lineal entre valores ordenados,
    la misma de `numpy.percentile`. Devuelve una serie alineada con `fechas`
    (posicion a posicion), con NaN donde la ventana no esta completa.
    """
    r = rezago_de(rezago_minimo_h)
    if not 0 < percentil < 100:
        raise ErrorObjetivo(f"percentil debe estar entre 0 y 100: {percentil}")
    momentos = verificar_rejilla_horaria(fechas)
    serie = pd.Series(np.asarray(demanda, dtype="float64"), index=momentos)
    if len(serie) != len(momentos):  # pragma: no cover
        raise ErrorObjetivo("demanda y fechas tienen longitudes distintas.")

    desplazada = serie.shift(freq=pd.Timedelta(hours=r))
    ventana = desplazada.rolling(f"{ventana_h}h", min_periods=ventana_h)
    umbral = ventana.quantile(percentil / 100, interpolation="linear").reindex(momentos)
    return pd.Series(umbral.to_numpy(), name=COLUMNA_UMBRAL)


def construir_alerta_pico(
    fechas: pd.Series | pd.DatetimeIndex,
    demanda: pd.Series | np.ndarray,
    rezago_minimo_h: int | str,
    indices: dict[str, np.ndarray] | None = None,
    percentil: float = PERCENTIL_ALERTA,
    ventana_h: int = VENTANA_MOVIL_H,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Marco con fecha_hora, umbral_alerta_mwh y alerta_pico (0/1, NaN sin ventana).

    Con `indices` (de `particion.particionar`) el registro anade filas,
    positivos y proporcion por bloque, salvo en prueba.
    """
    r = rezago_de(rezago_minimo_h)
    momentos = verificar_rejilla_horaria(fechas)
    valores = np.asarray(demanda, dtype="float64")
    if len(valores) != len(momentos):
        raise ErrorObjetivo(f"{len(valores)} valores de demanda para {len(momentos)} marcas.")

    umbral = umbral_movil(momentos, valores, r, percentil, ventana_h).to_numpy()
    con_umbral = ~np.isnan(umbral)
    if not con_umbral.any():
        raise ErrorObjetivo(
            f"La serie ({len(valores)} h) es mas corta que la ventana desplazada ({r + ventana_h} h)."
        )
    # 0/1 en float: NaN donde no hay ventana completa (el arranque).
    alerta = np.where(con_umbral, (valores > umbral).astype("float64"), np.nan)
    marco = pd.DataFrame({"fecha_hora": pd.Series(momentos), COLUMNA_UMBRAL: umbral, COLUMNA_ALERTA: alerta})

    sin_umbral = int((~con_umbral).sum())
    if sin_umbral and np.flatnonzero(~con_umbral).max() >= np.flatnonzero(con_umbral).min():  # pragma: no cover
        raise ErrorObjetivo("Hay filas sin umbral despues del arranque: la rejilla deberia evitarlo.")

    por_bloque: dict[str, Any] = {}
    if indices is not None:
        for nombre, posiciones in indices.items():
            if nombre not in BLOQUES_VISIBLES:
                por_bloque[nombre] = {
                    "filas": int(len(posiciones)),
                    "proporcion_positivos": None,
                    "nota": "no se calcula: bloque reservado para la evaluacion final",
                }
                continue
            objetivo = alerta[posiciones]
            definidos = objetivo[~np.isnan(objetivo)]
            por_bloque[nombre] = {
                "filas": int(len(posiciones)),
                "filas_con_objetivo": int(len(definidos)),
                "positivos": int(definidos.sum()),
                "proporcion_positivos": float(definidos.mean()) if len(definidos) else None,
            }

    registro = registro_operacion(
        "construir_alerta_pico",
        f"{COLUMNA_ALERTA}(t) = 1 si demanda_mwh(t) > percentil {percentil:g} de demanda_mwh "
        f"en las {ventana_h} h que acaban en t - {r} h; 0 si no",
        rezago_minimo_h=r,
        percentil=percentil,
        ventana_h=ventana_h,
        ventana=f"demanda_mwh(t - {r + ventana_h - 1} h) .. demanda_mwh(t - {r} h), la misma de ma_{ventana_h}",
        comparacion="estricta (>)",
        interpolacion="lineal entre valores ordenados (numpy.percentile)",
        calculo="por marca de tiempo: shift con freq + ventana movil de tiempo, min_periods = ventana",
        zona_horaria=ZONA_HORARIA,
        filas=len(marco),
        filas_sin_umbral=sin_umbral,
        nota_sin_umbral=(
            f"las primeras {r + ventana_h - 1} filas de la tabla no tienen ventana completa "
            "con la demanda de la propia tabla; quedan sin objetivo"
        ),
        por_bloque=por_bloque,
    )
    return marco, registro
