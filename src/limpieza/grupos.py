"""Limpieza de tablas desagregadas, grupo a grupo.

`limpiar()` esta pensada para una serie: una fila por hora. Una tabla como la
demanda comercial por CIIU tiene cientos de filas por hora -- una por
subactividad -- y por eso `deduplicar()` se niega a tratarla: deduplicar por
hora borraria el 99 % de los datos.

La forma correcta es partir la tabla en series, una por grupo, y limpiar cada
una por separado. Asi los estadisticos de un sector no contaminan a otro: un
valor de 10 MWh es normal en la industria manufacturera y disparatado en una
biblioteca, y un criterio de atipicos calculado sobre la mezcla no veria ni lo
uno ni lo otro.

    from limpieza.grupos import limpiar_por_grupos, fila_de_resumen

    for etiqueta, limpio, registro in limpiar_por_grupos(ciiu, ["Activity", "Subactivity"]):
        ...  # escribir `limpio`, acumular fila_de_resumen(etiqueta, limpio, registro)

Es un generador a proposito: con 17,5 millones de filas no conviene juntar el
resultado en memoria, sino escribir cada grupo en cuanto esta limpio.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import pandas as pd

from limpieza.limpiar import ErrorLimpieza, a_snake_case, limpiar

log = logging.getLogger(__name__)


def limpiar_por_grupos(
    marco: pd.DataFrame,
    columnas_grupo: list[str],
    **opciones: Any,
) -> Iterator[tuple[dict[str, str], pd.DataFrame, dict[str, Any]]]:
    """Limpia cada grupo como una serie propia y los va entregando uno a uno.

    `opciones` se pasan tal cual a `limpiar()` (metodo de imputacion, limite de
    horas, etc.). Cada grupo se entrega como `(etiqueta, limpio, registro)`.

    Tras limpiar se comprueba que todas las filas del grupo -- tambien las que
    se crearon para completar la rejilla horaria -- conservan su etiqueta. Una
    fila sin etiqueta no se podria volver a atribuir a su sector, asi que en
    ese caso falla en vez de seguir.
    """
    if marco.empty:
        raise ErrorLimpieza("La tabla esta vacia: no hay grupos que limpiar.")
    if not columnas_grupo:
        raise ErrorLimpieza("Hace falta al menos una columna de grupo.")
    faltan = [c for c in columnas_grupo if c not in marco.columns]
    if faltan:
        raise ErrorLimpieza(
            f"Columnas de grupo ausentes: {faltan}. Columnas: {list(marco.columns)}"
        )

    for clave, grupo in marco.groupby(list(columnas_grupo), observed=True, sort=True):
        clave = clave if isinstance(clave, tuple) else (clave,)
        etiqueta = {columna: str(valor) for columna, valor in zip(columnas_grupo, clave)}

        limpio, registro = limpiar(grupo, **opciones)

        for columna, valor in etiqueta.items():
            nombre = a_snake_case(columna)
            etiquetas = limpio[nombre]
            sin_etiqueta = int(etiquetas.isna().sum())
            ajenas = int((etiquetas.dropna().astype(str) != valor).sum())
            if sin_etiqueta or ajenas:
                raise ErrorLimpieza(
                    f"Grupo {etiqueta}: {sin_etiqueta} filas sin etiqueta y {ajenas} "
                    f"con otra etiqueta en '{nombre}' tras limpiar."
                )

        registro["grupo"] = etiqueta
        yield etiqueta, limpio, registro


def fila_de_resumen(
    etiqueta: dict[str, str], limpio: pd.DataFrame, registro: dict[str, Any]
) -> dict[str, Any]:
    """Una fila compacta por grupo, para el registro y para inspeccionar."""
    procedencia = registro["procedencia"]["por_origen"]
    rejilla = next(
        op["detalle"] for op in registro["operaciones"] if op["operacion"] == "completar_rejilla"
    )
    tiempo = registro["columna_tiempo"]
    return {
        **etiqueta,
        "desde": str(limpio[tiempo].min()),
        "hasta": str(limpio[tiempo].max()),
        "filas_iniciales": int(registro["filas_iniciales"]),
        "filas_finales": int(registro["filas_finales"]),
        "horas_creadas": int(rejilla["filas_creadas_para_completar_rejilla"]),
        "observados": int(procedencia.get("observado", 0)),
        "interpolados": int(procedencia.get("interpolado", 0)),
        "faltantes": int(procedencia.get("faltante", 0)),
        "tramos_no_imputados": int(rejilla["n_tramos_no_imputados"]),
        "atipicos": int(registro["procedencia"]["n_atipicos"]),
        "no_evaluables": int(
            next(
                op["detalle"]["n_no_evaluables"]
                for op in registro["operaciones"]
                if op["operacion"] == "marcar_atipicos"
            )
        ),
    }


def resumir(filas: list[dict[str, Any]]) -> dict[str, Any]:
    """Totales de todos los grupos a partir de sus filas de resumen."""
    if not filas:
        return {"n_grupos": 0}
    tabla = pd.DataFrame(filas)
    total = {
        clave: int(tabla[clave].sum())
        for clave in (
            "filas_iniciales", "filas_finales", "horas_creadas", "observados",
            "interpolados", "faltantes", "atipicos", "no_evaluables",
        )
    }
    return {
        "n_grupos": int(len(tabla)),
        **total,
        "grupos_con_interpolados": int((tabla["interpolados"] > 0).sum()),
        "grupos_con_faltantes": int((tabla["faltantes"] > 0).sum()),
        "pct_observado": round(100 * total["observados"] / max(1, total["filas_finales"]), 4),
    }
