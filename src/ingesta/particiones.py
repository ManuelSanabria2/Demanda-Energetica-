"""Escritura de Parquet particionado por anio y mes, sin destruir lo anterior.

Escribir no es reemplazar. `pyarrow.parquet.write_to_dataset` con
`existing_data_behavior="delete_matching"` borra la particion entera antes de
escribir, asi que una ingesta acotada a unos pocos dias se lleva por delante el
resto del mes. Ya paso una vez en este proyecto: una prueba de cinco dias borro
los 26 restantes de marzo de 2025, en silencio.

Aqui, para cada particion afectada, se lee lo que ya habia, se combina con lo
nuevo y se deduplica por una clave explicita antes de reescribirla completa.
Lo recien escrito gana, de modo que una revision de la fuente sustituye al dato
previo en vez de convivir con el.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config

log = logging.getLogger(__name__)

COLUMNAS_PARTICION = ("anio", "mes")


def leer_particiones(
    destino: Path, pares: set[tuple[int, int]], columnas: list[str]
) -> pd.DataFrame:
    """Lee lo ya almacenado en las particiones que se van a tocar.

    Devuelve las filas sin las columnas de particion: se recalculan despues a
    partir de la marca de tiempo, que es la fuente de verdad.
    """
    marcos = []
    for anio, mes in sorted(pares):
        carpeta = destino / f"anio={anio}" / f"mes={mes}"
        if not carpeta.exists():
            continue
        for archivo in sorted(carpeta.glob("*.parquet")):
            marcos.append(pd.read_parquet(archivo))

    if not marcos:
        return pd.DataFrame(columns=columnas)

    leido = pd.concat(marcos, ignore_index=True)
    # Las particiones antiguas se escribieron sin zona. Normalizarlas al leer
    # evita que un concat mezcle naive con tz-aware y reviente, y hace la
    # migracion transparente.
    for columna in leido.columns:
        if pd.api.types.is_datetime64_any_dtype(leido[columna]):
            leido[columna] = config.a_zona_colombia(leido[columna])
    return leido


def escribir_particionado(
    tabla: pd.DataFrame,
    destino: Path,
    esquema: pa.Schema,
    claves: list[str],
    columna_tiempo: str = "fecha_hora",
    etiqueta: str = "",
) -> int:
    """Escribe `tabla` fusionandola con lo que ya hubiera en cada particion.

    `esquema` debe incluir las columnas de datos mas `anio` y `mes`. `claves`
    identifica una observacion: dos filas que coincidan en esas columnas son la
    misma, y se conserva la mas reciente.
    """
    if tabla.empty:
        log.warning("%s: nada que escribir", etiqueta or destino.name)
        return 0

    nombre = etiqueta or destino.name
    nuevas = tabla.copy()
    nuevas[columna_tiempo] = config.a_zona_colombia(nuevas[columna_tiempo])
    destino.mkdir(parents=True, exist_ok=True)

    columnas = [c for c in esquema.names if c not in COLUMNAS_PARTICION]
    for columna in columnas:
        if columna not in nuevas.columns:
            nuevas[columna] = None

    pares = set(zip(nuevas[columna_tiempo].dt.year, nuevas[columna_tiempo].dt.month))
    existentes = leer_particiones(destino, pares, columnas)

    # Concatenar con un marco vacio confunde la inferencia de tipos de pandas
    # (y esta en vias de deprecacion), asi que se evita cuando no hay historico.
    combinado = (
        nuevas[columnas].copy()
        if existentes.empty
        else pd.concat([existentes, nuevas[columnas]], ignore_index=True)
    )

    antes = len(combinado)
    combinado = combinado.drop_duplicates(subset=claves, keep="last")
    sustituidas = antes - len(combinado)
    preservadas = max(0, len(combinado) - len(nuevas))

    if sustituidas:
        log.info("%s: %d filas existentes sustituidas por lo recien escrito", nombre, sustituidas)
    if preservadas:
        log.info("%s: %d filas del historico conservadas en las particiones tocadas", nombre, preservadas)

    salida = combinado.sort_values(columna_tiempo).reset_index(drop=True)
    salida["anio"] = salida[columna_tiempo].dt.year.astype("int32")
    salida["mes"] = salida[columna_tiempo].dt.month.astype("int32")

    pq.write_to_dataset(
        pa.Table.from_pandas(salida[esquema.names], schema=esquema, preserve_index=False),
        root_path=str(destino),
        partition_cols=list(COLUMNAS_PARTICION),
        existing_data_behavior="delete_matching",
    )
    log.info(
        "%s: escritas %d filas (%d nuevas, %d preservadas)",
        nombre, len(salida), len(nuevas), preservadas,
    )
    return len(salida)
