"""Particion cronologica de la tabla de modelado y validacion cruzada temporal.

Tres bloques consecutivos, con las fechas de corte como constantes (hora local):

    entrenamiento   inicio de la tabla   .. 2024-09-02 23:00
    validacion      2024-09-03 00:00     .. 2025-09-02 23:00
    prueba          2025-09-03 00:00     .. 2026-09-02 23:00

Nada se baraja: cada bloque contiene solo horas posteriores a las del anterior.
El bloque de prueba se devuelve para que conste cuantas filas tiene y que rango
cubre, pero **no se usa hasta la evaluacion final**: la seleccion de modelos y
de hiperparametros se hace en validacion (CLAUDE.md).

Dentro del entrenamiento, la validacion cruzada es `TimeSeriesSplit` con
`gap = rezago_minimo_h`: 24 h en r24 y 120 h en r120. Entre la ultima hora con
la que se entrena y la primera que se valida quedan las mismas horas que entre
el ultimo dato publicado y la hora que se predice. `gap` cuenta filas, asi que
antes se comprueba que la rejilla horaria esta completa.

    from modelado.particion import particionar, validacion_cruzada
    indices, registro = particionar(fechas)
    entrenamiento = indices["entrenamiento"]
    tscv, registro_cv = validacion_cruzada("r24", fechas.iloc[entrenamiento])
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from modelado.datos import ZONA_HORARIA, registro_operacion, rezago_de, verificar_rejilla_horaria

# Ultimo dia (incluido) de cada bloque, en hora local.
FIN_ENTRENAMIENTO = dt.date(2024, 9, 2)
FIN_VALIDACION = dt.date(2025, 9, 2)
FIN_PRUEBA = dt.date(2026, 9, 2)
CORTES = {
    "entrenamiento": FIN_ENTRENAMIENTO,
    "validacion": FIN_VALIDACION,
    "prueba": FIN_PRUEBA,
}
BLOQUES = tuple(CORTES)

N_PLIEGUES = 5


class ErrorParticion(Exception):
    """La particion temporal no se puede hacer con seguridad."""


def limite_superior(fecha: dt.date) -> pd.Timestamp:
    """Primera hora del dia siguiente a `fecha`, en hora local: el bloque acaba justo antes."""
    return pd.Timestamp((fecha + dt.timedelta(days=1)).isoformat()).tz_localize(ZONA_HORARIA)


def inicio_prueba() -> pd.Timestamp:
    """Primera hora del bloque de prueba. Ninguna hora igual o posterior se usa antes de la evaluacion final."""
    return limite_superior(FIN_VALIDACION)


def _momentos(fechas: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    momentos = pd.DatetimeIndex(fechas)
    if momentos.tz is None:
        raise ErrorParticion(f"Las marcas de tiempo no tienen zona horaria; se esperan en {ZONA_HORARIA}.")
    if momentos.hasnans:
        raise ErrorParticion(f"{int(momentos.isna().sum())} marcas de tiempo nulas.")
    if momentos.has_duplicates or not momentos.is_monotonic_increasing:
        raise ErrorParticion(
            "Las marcas de tiempo tienen que estar ordenadas y sin repetidas: ordena "
            "la tabla por fecha_hora antes de particionar."
        )
    return momentos.tz_convert(ZONA_HORARIA)


def _rango(momentos: pd.DatetimeIndex, posiciones: np.ndarray) -> dict[str, Any]:
    return {
        "filas": int(len(posiciones)),
        "desde": str(momentos[posiciones[0]]),
        "hasta": str(momentos[posiciones[-1]]),
    }


def particionar(
    fechas: pd.Series | pd.DatetimeIndex,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Posiciones de las filas de entrenamiento, validacion y prueba.

    Devuelve ({bloque: posiciones}, registro). Las posiciones son enteras y
    crecientes, para usar con `.iloc`. Falla si hay filas posteriores al fin de
    la prueba o si algun bloque queda vacio.
    """
    momentos = _momentos(fechas)

    posteriores = int((momentos >= limite_superior(FIN_PRUEBA)).sum())
    if posteriores:
        raise ErrorParticion(
            f"{posteriores} filas posteriores a {FIN_PRUEBA}: no pertenecen a ningun "
            "bloque. Revisa las fechas de corte o la tabla."
        )

    indices: dict[str, np.ndarray] = {}
    desde: pd.Timestamp | None = None
    for nombre, fin in CORTES.items():
        hasta = limite_superior(fin)
        mascara = momentos < hasta if desde is None else (momentos >= desde) & (momentos < hasta)
        indices[nombre] = np.flatnonzero(mascara)
        desde = hasta

    vacios = [nombre for nombre, posiciones in indices.items() if len(posiciones) == 0]
    if vacios:
        raise ErrorParticion(
            f"Bloques vacios: {vacios}. La tabla cubre {momentos[0]} .. {momentos[-1]}."
        )
    # Defensa: consecutivos, en orden y cubriendo todas las filas una sola vez.
    unidas = np.concatenate([indices[nombre] for nombre in BLOQUES])
    if not np.array_equal(unidas, np.arange(len(momentos))):  # pragma: no cover
        raise ErrorParticion("Los bloques no cubren la tabla una sola vez y en orden.")

    bloques = {
        nombre: {**_rango(momentos, indices[nombre]), "fecha_corte": CORTES[nombre].isoformat()}
        for nombre in BLOQUES
    }
    bloques["prueba"]["uso"] = (
        "reservado para la evaluacion final: no se usa para seleccionar modelos, "
        "ajustar hiperparametros ni escalar"
    )

    registro = registro_operacion(
        "particionar",
        "tres bloques cronologicos consecutivos por fecha de corte (ultimo dia "
        "incluido, hora local), sin barajar",
        filas=len(momentos),
        zona_horaria=ZONA_HORARIA,
        cortes={nombre: fecha.isoformat() for nombre, fecha in CORTES.items()},
        bloques=bloques,
    )
    return indices, registro


def validacion_cruzada(
    rezago_minimo_h: int | str,
    fechas_entrenamiento: pd.Series | pd.DatetimeIndex,
    n_splits: int = N_PLIEGUES,
) -> tuple[TimeSeriesSplit, dict[str, Any]]:
    """`TimeSeriesSplit` para usar dentro del bloque de entrenamiento.

    `gap = rezago_minimo_h` filas, que son horas porque antes se comprueba la
    rejilla horaria: 24 en r24 y 120 en r120. El registro describe cada pliegue
    con sus filas y su rango, y comprueba que ninguno valida antes de entrenar.
    """
    gap = rezago_de(rezago_minimo_h)
    momentos = verificar_rejilla_horaria(fechas_entrenamiento)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    try:
        pliegues = list(tscv.split(np.arange(len(momentos))))
    except ValueError as error:
        raise ErrorParticion(
            f"No caben {n_splits} pliegues con gap={gap} en {len(momentos)} filas: {error}"
        ) from error

    descripcion = []
    for numero, (entrena, valida) in enumerate(pliegues, start=1):
        separacion = int(valida[0] - entrena[-1] - 1)
        if separacion != gap:  # pragma: no cover - defensa
            raise ErrorParticion(f"Pliegue {numero}: separacion de {separacion} filas, se esperaban {gap}.")
        descripcion.append(
            {
                "pliegue": numero,
                "entrenamiento": _rango(momentos, entrena),
                "validacion": _rango(momentos, valida),
                "gap_h": separacion,
            }
        )

    registro = registro_operacion(
        "validacion_cruzada",
        f"TimeSeriesSplit(n_splits={n_splits}, gap={gap}) sobre el bloque de "
        "entrenamiento; gap = rezago_minimo_h, en filas de una rejilla horaria completa",
        n_splits=n_splits,
        gap_h=gap,
        rezago_minimo_h=gap,
        filas=len(momentos),
        pliegues=descripcion,
    )
    return tscv, registro
