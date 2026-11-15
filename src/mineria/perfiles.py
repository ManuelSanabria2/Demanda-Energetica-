"""Perfiles diarios de carga: del panel horario a una matriz dia x hora.

    from mineria.perfiles import perfiles_diarios, forma_y_nivel, atributos_dia

    perfiles, registro = perfiles_diarios(panel)       # MWh, 24 columnas
    forma, nivel = forma_y_nivel(perfiles)            # forma: media del dia = 1
    atributos = atributos_dia(panel, perfiles.index)  # calendario y clima por dia

Por que separar forma y nivel: la demanda crece de un ano a otro (~2-3 %), asi
que un K-means sobre MWh agruparia los dias por ano antes que por tipo de dia.
Dividir cada dia por su media deja solo la *forma* de la curva, que es lo que
distingue un lunes de un domingo o de un festivo. El nivel se conserva aparte
para describir los grupos.

Solo entran dias completos: las 24 horas presentes, ningun NaN y ningun valor
imputado. Los ultimos `dias_parciales` dias se descartan por la publicacion
parcial documentada en notas/hallazgos_calidad.md (seccion 1).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

COLUMNA_TIEMPO = "fecha_hora"
COLUMNA_VALOR = "valor_kwh"
DIAS_PARCIALES = 5
HORAS = [f"h{h:02d}" for h in range(24)]

BANDERAS = (
    "es_festivo",
    "es_vispera_festivo",
    "es_puente",
    "es_semana_santa",
    "es_ultima_semana_diciembre",
)

NOMBRES_DIA = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")


class ErrorPerfiles(ValueError):
    """La entrada no permite construir perfiles diarios fiables."""


def _hora_local(marco: pd.DataFrame) -> pd.Series:
    tiempo = pd.to_datetime(marco[COLUMNA_TIEMPO])
    if tiempo.dt.tz is not None:
        tiempo = tiempo.dt.tz_convert("America/Bogota").dt.tz_localize(None)
    return tiempo


def perfiles_diarios(
    panel: pd.DataFrame,
    dias_parciales: int = DIAS_PARCIALES,
    excluir_imputados: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Matriz dia x hora en MWh, solo con dias completos y fiables.

    Devuelve el marco (indice `fecha`, columnas h00..h23) y un registro con
    cuantos dias entraron y por que salio cada uno de los demas.
    """
    if COLUMNA_TIEMPO not in panel or COLUMNA_VALOR not in panel:
        raise ErrorPerfiles(f"faltan columnas: se esperan {COLUMNA_TIEMPO} y {COLUMNA_VALOR}")
    tiempo = _hora_local(panel)
    if tiempo.duplicated().any():
        raise ErrorPerfiles("hay marcas de tiempo repetidas: el pivote mezclaria valores")

    datos = pd.DataFrame(
        {
            "fecha": tiempo.dt.date,
            "hora": tiempo.dt.hour,
            "mwh": panel[COLUMNA_VALOR].to_numpy(dtype=float) / 1000.0,
        }
    )
    if excluir_imputados and "imputado" in panel:
        datos.loc[panel["imputado"].fillna(False).to_numpy(dtype=bool), "mwh"] = np.nan

    matriz = datos.pivot(index="fecha", columns="hora", values="mwh").reindex(columns=range(24))
    matriz.columns = HORAS
    dias_totales = len(matriz)

    incompletos = matriz.isna().any(axis=1)
    matriz = matriz[~incompletos]

    ultimos: list[dt.date] = []
    if dias_parciales > 0 and len(matriz):
        ultimo_dia = max(datos["fecha"])
        corte = ultimo_dia - dt.timedelta(days=dias_parciales - 1)
        ultimos = [d for d in matriz.index if d >= corte]
        matriz = matriz.drop(index=ultimos)

    if matriz.empty:
        raise ErrorPerfiles("ningun dia completo tras los filtros")

    registro = {
        "operacion": "perfiles_diarios",
        "criterio": (
            "un dia entra si tiene las 24 horas con valor observado (sin NaN"
            + (" ni imputados" if excluir_imputados else "")
            + f"); se descartan los ultimos {dias_parciales} dias por publicacion parcial"
        ),
        "dias_totales": int(dias_totales),
        "dias_incompletos": int(incompletos.sum()),
        "dias_publicacion_parcial": len(ultimos),
        "dias_resultantes": int(len(matriz)),
        "rango": [str(matriz.index.min()), str(matriz.index.max())],
    }
    matriz.index = pd.Index(matriz.index, name="fecha")
    return matriz, registro


def forma_y_nivel(perfiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa cada dia en forma (media 1) y nivel (media diaria en MWh)."""
    nivel = perfiles.mean(axis=1).rename("nivel_mwh")
    if (nivel <= 0).any():
        raise ErrorPerfiles("hay dias con media no positiva: la forma no esta definida")
    forma = perfiles.div(nivel, axis=0)
    return forma, nivel


def atributos_dia(panel: pd.DataFrame, fechas: pd.Index | None = None) -> pd.DataFrame:
    """Calendario y clima de cada dia: lo que se usa para *leer* los grupos.

    Ninguno de estos atributos entra al K-means; sirven para interpretar a
    posteriori que tienen en comun los dias de cada grupo.
    """
    tiempo = _hora_local(panel)
    datos = pd.DataFrame({"fecha": tiempo.dt.date})
    for bandera in BANDERAS:
        if bandera in panel:
            datos[bandera] = panel[bandera].fillna(False).astype(bool).to_numpy()
    if "temperatura_c" in panel:
        datos["temperatura_c"] = panel["temperatura_c"].to_numpy(dtype=float)
    if "nombre_festivo" in panel:
        datos["nombre_festivo"] = panel["nombre_festivo"].to_numpy()

    agregaciones: dict[str, Any] = {b: "max" for b in BANDERAS if b in datos}
    if "temperatura_c" in datos:
        agregaciones["temperatura_c"] = "mean"
    if "nombre_festivo" in datos:
        agregaciones["nombre_festivo"] = lambda s: next((v for v in s if isinstance(v, str) and v), None)
    dias = datos.groupby("fecha").agg(agregaciones) if agregaciones else datos.groupby("fecha").size().to_frame("n")

    fechas_dt = pd.to_datetime(pd.Series(dias.index, index=dias.index))
    dias["dia_semana"] = fechas_dt.dt.dayofweek.to_numpy()
    dias["nombre_dia"] = [NOMBRES_DIA[d] for d in dias["dia_semana"]]
    dias["mes"] = fechas_dt.dt.month.to_numpy()
    dias["anio"] = fechas_dt.dt.year.to_numpy()
    festivo = dias["es_festivo"] if "es_festivo" in dias else pd.Series(False, index=dias.index)
    dias["tipo_dia"] = np.select(
        [festivo.to_numpy(), dias["dia_semana"].to_numpy() == 6, dias["dia_semana"].to_numpy() == 5],
        ["festivo", "domingo", "sabado"],
        default="laboral",
    )
    if fechas is not None:
        dias = dias.reindex(fechas)
    dias.index = pd.Index(dias.index, name="fecha")
    return dias
