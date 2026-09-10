"""Normalizacion de ambas fuentes a un esquema horario comun.

La descarga y el paso de formato ancho a largo los hace `clientes`; aqui solo
se unifica el esquema, se colapsan las versiones de liquidacion de SIMEM y se
mide la cobertura.

Esquema de salida:

    fecha_hora   datetime64[ns]  hora local de Colombia (naive; no hay DST)
    fuente       str             "xm" o "simem"
    metrica      str             p. ej. "DemaReal" o "DdaReal"
    entidad      str             p. ej. "Sistema" o el codigo SIC del agente
    valor_kwh    float64
    version      str | None      version de liquidacion (solo SIMEM)
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd

from . import config

log = logging.getLogger(__name__)

COLUMNAS_SALIDA = ["fecha_hora", "fuente", "metrica", "entidad", "valor_kwh", "version"]


def xm_cliente_a_esquema_comun(tabla: pd.DataFrame) -> pd.DataFrame:
    """Pasa la salida de ClienteXM al esquema comun de las tablas procesadas.

    El cliente devuelve (timestamp, fuente, identificador, entidad, valor);
    aqui solo se renombra. La conversion de formato ancho a largo ya la hizo
    el cliente, que es quien conoce la convencion HourNN.
    """
    if tabla.empty:
        return pd.DataFrame(columns=COLUMNAS_SALIDA)

    tabla = tabla.copy()
    tabla["timestamp"] = config.a_zona_colombia(tabla["timestamp"])
    comun = tabla.rename(
        columns={
            "timestamp": "fecha_hora",
            "identificador": "metrica",
            "valor": "valor_kwh",
        }
    ).copy()
    comun["version"] = None
    return comun[COLUMNAS_SALIDA].sort_values("fecha_hora").reset_index(drop=True)


def simem_colapsar_versiones(
    registros: pd.DataFrame,
    columnas_dimension: list[str],
    precedencia: tuple[str, ...] = config.PRECEDENCIA_VERSIONES,
) -> pd.DataFrame:
    """Deja una sola version de liquidacion por marca de tiempo.

    Sobre una misma FechaHora pueden coexistir varias versiones (verificado:
    TX2 y TXR juntas en 14fabb), asi que agregar sin filtrar duplica la
    demanda. Para cada FechaHora se elige la version mas definitiva presente
    segun `precedencia`, y despues se verifica que no queden duplicados sobre
    (FechaHora + dimensiones). Si el error se colara sesgaria el objetivo del
    modelo por un factor entero, asi que aqui falla en vez de continuar.
    """
    if registros.empty:
        return registros

    if "Version" not in registros.columns:
        log.info("El dataset no tiene columna Version; no hay nada que colapsar")
        return registros

    presentes = set(registros["Version"].dropna().unique())
    desconocidas = presentes - set(precedencia)
    if desconocidas:
        raise ValueError(
            f"Versiones sin precedencia definida: {sorted(desconocidas)}. "
            f"Anadelas a config.PRECEDENCIA_VERSIONES en el orden correcto."
        )

    rango = {version: orden for orden, version in enumerate(precedencia)}
    tabla = registros.copy()
    tabla["_rango"] = tabla["Version"].map(rango)

    # La version elegida se decide por FechaHora, no globalmente: una fecha
    # reciente puede tener solo TX2 mientras otra antigua ya tiene TXR.
    mejor = tabla.groupby("FechaHora")["_rango"].transform("min")
    tabla = tabla[tabla["_rango"] == mejor].drop(columns="_rango")

    claves = ["FechaHora", *columnas_dimension]
    duplicados = int(tabla.duplicated(subset=claves).sum())
    if duplicados:
        raise ValueError(
            f"Quedan {duplicados} filas duplicadas sobre {claves} tras colapsar "
            f"versiones. Agregar en este estado duplicaria la demanda."
        )

    por_hora = tabla.groupby("FechaHora")["Version"].nunique()
    if (por_hora > 1).any():
        conflictivas = por_hora[por_hora > 1].index.tolist()[:5]
        raise ValueError(
            f"Hay marcas de tiempo con mas de una version tras el colapso: {conflictivas}"
        )

    log.info(
        "SIMEM: %d filas -> %d tras colapsar versiones %s",
        len(registros),
        len(tabla),
        sorted(presentes),
    )
    return tabla.reset_index(drop=True)


def simem_agregar_nacional(
    registros: pd.DataFrame,
    metrica: str = "DdaReal",
    columnas_dimension: tuple[str, ...] = ("CodigoSICAgente", "TipoMercado"),
) -> pd.DataFrame:
    """Colapsa versiones y suma a una serie horaria nacional en el esquema comun.

    El colapso de versiones es obligatorio antes de sumar; esta funcion lo
    aplica ella misma para que no se pueda olvidar.
    """
    if registros.empty:
        return pd.DataFrame(columns=COLUMNAS_SALIDA)

    dimensiones = [c for c in columnas_dimension if c in registros.columns]
    limpio = simem_colapsar_versiones(registros, dimensiones)

    limpio = limpio.copy()
    limpio["fecha_hora"] = config.a_zona_colombia(limpio["FechaHora"])

    agregado = (
        limpio.groupby("fecha_hora")
        .agg(valor_kwh=("Valor", "sum"), version=("Version", "first"))
        .reset_index()
    )
    agregado["fuente"] = "simem"
    agregado["metrica"] = metrica
    agregado["entidad"] = "Nacional"

    return agregado[COLUMNAS_SALIDA].sort_values("fecha_hora").reset_index(drop=True)


def reporte_cobertura(
    tabla: pd.DataFrame,
    inicio: dt.date | None = None,
    fin: dt.date | None = None,
) -> dict[str, Any]:
    """Compara la serie contra el calendario horario completo esperado.

    Detecta los huecos que ninguna de las dos APIs senala como error: XM
    devuelve 200 con Items vacio y SIMEM simplemente omite las filas.
    """
    if tabla.empty:
        return {
            "n_filas": 0,
            "horas_esperadas": 0,
            "horas_presentes": 0,
            "horas_faltantes": 0,
            "horas_con_nan": 0,
            "duplicados": 0,
            "primeros_huecos": [],
        }

    momentos = config.a_zona_colombia(tabla["fecha_hora"])
    zona = momentos.dt.tz

    # La rejilla esperada tiene que construirse en la misma zona que los datos.
    # Compararla naive contra datos con zona no da error: da que faltan TODAS
    # las horas, que es peor.
    desde = (
        pd.Timestamp(inicio, tz=zona) if inicio else momentos.min().normalize()
    )
    hasta = (
        pd.Timestamp(fin, tz=zona) + pd.Timedelta(hours=23)
        if fin
        else momentos.max().normalize() + pd.Timedelta(hours=23)
    )

    esperadas = pd.date_range(desde, hasta, freq="h")
    presentes = pd.DatetimeIndex(momentos.unique())
    faltantes = esperadas.difference(presentes)

    return {
        "n_filas": int(len(tabla)),
        "horas_esperadas": int(len(esperadas)),
        "horas_presentes": int(len(presentes)),
        "horas_faltantes": int(len(faltantes)),
        "horas_con_nan": int(tabla["valor_kwh"].isna().sum()),
        "duplicados": int(momentos.duplicated().sum()),
        "primeros_huecos": [str(m) for m in faltantes[:20]],
    }
