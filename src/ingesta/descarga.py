"""Capa de persistencia de la capa cruda: Parquet particionado e incremental.

Estructura en disco:

    data/raw/{fuente}/{identificador}/anio={AAAA}/mes={MM}/datos.parquet

Las columnas `anio` y `mes` viven solo en la ruta, no dentro del archivo:
al leer el conjunto con `pandas.read_parquet(ruta)` pyarrow las reconstruye
como columnas a partir del particionado Hive.

Tres propiedades que el modulo garantiza:

**Incremental.** Cada ejecucion mira la fecha maxima ya almacenada y pide solo
lo que falta. No arranca justo despues de esa fecha, sino `VENTANA_REFRESCO_DIAS`
antes: la demanda de XM llega con rezago y las versiones de liquidacion de SIMEM
se revisan durante semanas, asi que los ultimos dias almacenados pueden estar
incompletos o desactualizados y hay que volver a pedirlos.

**Idempotente.** Escribir no es anadir. Para cada particion afectada se lee lo
que ya habia, se combina con lo nuevo, se deduplica por una clave explicita
(ganando siempre lo recien descargado) y se reescribe la particion entera.
Ejecutar dos veces seguidas deja el mismo contenido y el mismo hash.

**Trazable.** Cada descarga deja una entrada en `data/raw/manifiesto.json`, que
es una bitacora append-only: rango pedido, rango realmente obtenido, numero de
registros, hash del resultado, hash por particion y version de las librerias.
Sirve para reconstruir despues que datos habia en cada momento.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config
from .clientes import ClienteAPI, ClienteSIMEM, ClienteXM

log = logging.getLogger(__name__)

CLIENTES: dict[str, type[ClienteAPI]] = {
    "xm": ClienteXM,
    "simem": ClienteSIMEM,
}

NOMBRE_ARCHIVO = "datos.parquet"

# Columnas que contienen el valor observado y por tanto NO forman parte de la
# identidad de una fila: si dos filas coinciden en todo lo demas son la misma
# observacion, y la mas reciente sustituye a la anterior.
COLUMNAS_VALOR = frozenset({"valor", "Valor"})


class ErrorDescarga(Exception):
    """Fallo al persistir o leer la capa cruda."""


# --------------------------------------------------------------------------
# Rutas y lectura del estado actual
# --------------------------------------------------------------------------


def ruta_dataset(fuente: str, identificador: str) -> Path:
    """Directorio raiz del conjunto en la capa cruda."""
    return config.DIR_CRUDO / fuente / identificador


def ruta_particion(fuente: str, identificador: str, anio: int, mes: int) -> Path:
    """Directorio de una particion concreta, con el mes a dos digitos."""
    return ruta_dataset(fuente, identificador) / f"anio={anio}" / f"mes={mes:02d}"


def ultima_fecha_almacenada(fuente: str, identificador: str) -> dt.date | None:
    """Fecha maxima presente en el conjunto, o None si aun no hay nada.

    Lee unicamente la columna `timestamp`, sin traer el resto del conjunto.
    """
    raiz = ruta_dataset(fuente, identificador)
    if not raiz.exists() or not any(raiz.rglob(f"*{NOMBRE_ARCHIVO}")):
        return None

    tabla = pq.read_table(raiz, columns=["timestamp"])
    if tabla.num_rows == 0:
        return None

    maximo = pc.max(tabla.column("timestamp")).as_py()
    return maximo.date() if isinstance(maximo, dt.datetime) else None


def leer(fuente: str, identificador: str, **kwargs: Any) -> pd.DataFrame:
    """Lee el conjunto completo de la capa cruda."""
    raiz = ruta_dataset(fuente, identificador)
    if not raiz.exists():
        return pd.DataFrame()
    return pd.read_parquet(raiz, **kwargs)


def n_registros_almacenados(fuente: str, identificador: str) -> int:
    """Cuenta las filas del conjunto leyendo solo los metadatos de Parquet."""
    raiz = ruta_dataset(fuente, identificador)
    if not raiz.exists():
        return 0
    return sum(
        pq.ParquetFile(archivo).metadata.num_rows
        for archivo in raiz.rglob(f"*{NOMBRE_ARCHIVO}")
    )


# --------------------------------------------------------------------------
# Identidad de una fila y hash
# --------------------------------------------------------------------------


def clave_dedup(marco: pd.DataFrame) -> list[str]:
    """Columnas que identifican una observacion, sin contar el valor.

    Para XM son (timestamp, fuente, identificador, entidad). Para SIMEM son
    esas mas las dimensiones propias del dataset, `Version` incluida: sobre una
    misma marca de tiempo conviven varias versiones de liquidacion y las dos
    son observaciones legitimas, asi que la version forma parte de la identidad
    y no debe colapsarse aqui. Colapsarla es trabajo de `normalizar`.
    """
    claves = [c for c in marco.columns if c not in COLUMNAS_VALOR]
    if "timestamp" not in claves:
        raise ErrorDescarga(
            f"El marco no tiene columna 'timestamp'. Columnas: {list(marco.columns)}"
        )
    return claves


def hash_marco(marco: pd.DataFrame) -> str:
    """Hash sha256 del contenido de un DataFrame, independiente del orden.

    Se ordena por la clave y por las columnas antes de hashear, de forma que
    el mismo contenido produzca siempre el mismo hash. El valor depende de la
    version de pandas, que por eso queda anotada en el manifiesto.
    """
    if marco.empty:
        return "sha256:" + hashlib.sha256(b"").hexdigest()

    ordenado = marco.sort_index(axis=1)
    ordenado = ordenado.sort_values(list(ordenado.columns)).reset_index(drop=True)
    digest = pd.util.hash_pandas_object(ordenado, index=False).values.tobytes()
    return "sha256:" + hashlib.sha256(digest).hexdigest()


def _entorno() -> dict[str, str]:
    """Versiones que hacen falta para reproducir un hash."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "plataforma": sys.platform,
    }


# --------------------------------------------------------------------------
# Escritura idempotente
# --------------------------------------------------------------------------


def _escribir_particion(
    grupo: pd.DataFrame, destino: Path, claves: list[str]
) -> tuple[int, str]:
    """Combina el grupo con lo que ya hubiera en la particion y la reescribe.

    Devuelve (filas resultantes, hash de la particion). Lo recien descargado
    gana ante lo antiguo, de modo que una revision posterior de la fuente
    sustituye al dato previo en vez de convivir con el.
    """
    archivo = destino / NOMBRE_ARCHIVO

    if archivo.exists():
        existente = pd.read_parquet(archivo)
        combinado = pd.concat([existente, grupo], ignore_index=True)
        antes = len(combinado)
        combinado = combinado.drop_duplicates(subset=claves, keep="last")
        repetidas = antes - len(combinado)
        log.info(
            "%s: %d existentes + %d nuevas -> %d tras deduplicar (%d sustituidas)",
            destino.name, len(existente), len(grupo), len(combinado), repetidas,
        )
    else:
        antes = len(grupo)
        combinado = grupo.drop_duplicates(subset=claves, keep="last")
        if antes != len(combinado):
            # Repetidas dentro de la propia descarga: no es una sustitucion de
            # historico, es que la fuente devolvio la misma fila dos veces.
            log.warning(
                "%s: la descarga traia %d filas repetidas dentro de si misma",
                destino.name, antes - len(combinado),
            )

    combinado = combinado.sort_values("timestamp").reset_index(drop=True)

    # Se reescribe la particion entera: escribir a un temporal y sustituir
    # evita dejarla a medias si algo falla en mitad de la escritura.
    destino.mkdir(parents=True, exist_ok=True)
    temporal = destino / f".{NOMBRE_ARCHIVO}.tmp"
    combinado.to_parquet(temporal, index=False)
    temporal.replace(archivo)

    return len(combinado), hash_marco(combinado)


def escribir(
    marco: pd.DataFrame, fuente: str, identificador: str
) -> dict[str, dict[str, Any]]:
    """Escribe un marco en la capa cruda, particionado por anio y mes.

    Es idempotente: reescribe cada particion afectada con la union deduplicada
    de lo que ya habia y lo nuevo.
    """
    if marco.empty:
        log.warning("%s/%s: nada que escribir", fuente, identificador)
        return {}

    # La validacion va antes de tocar nada, para que un marco con las columnas
    # equivocadas de un mensaje util en vez de un KeyError.
    claves = clave_dedup(marco)

    datos = marco.copy()
    datos["timestamp"] = pd.to_datetime(datos["timestamp"])

    anios = datos["timestamp"].dt.year
    meses = datos["timestamp"].dt.month

    resumen: dict[str, dict[str, Any]] = {}
    for (anio, mes), grupo in datos.groupby([anios, meses], sort=True):
        destino = ruta_particion(fuente, identificador, int(anio), int(mes))
        filas, hash_particion = _escribir_particion(grupo, destino, claves)
        etiqueta = f"anio={int(anio)}/mes={int(mes):02d}"
        resumen[etiqueta] = {"n_registros": filas, "hash": hash_particion}

    log.info(
        "%s/%s: %d filas escritas en %d particiones",
        fuente,
        identificador,
        len(datos),
        len(resumen),
    )
    return resumen


# --------------------------------------------------------------------------
# Manifiesto
# --------------------------------------------------------------------------


def cargar_manifiesto() -> dict[str, Any]:
    """Lee la bitacora de descargas, o devuelve una vacia si aun no existe."""
    ruta = config.RUTA_MANIFIESTO_CRUDO
    if not ruta.exists():
        return {"version_formato": 1, "descargas": []}
    with ruta.open("r", encoding="utf-8") as origen:
        return json.load(origen)


def anotar_manifiesto(entrada: dict[str, Any]) -> None:
    """Anade una entrada al final de la bitacora. Nunca reescribe las anteriores."""
    manifiesto = cargar_manifiesto()
    manifiesto["descargas"].append(entrada)

    ruta = config.RUTA_MANIFIESTO_CRUDO
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as destino:
        json.dump(manifiesto, destino, ensure_ascii=False, indent=2)
    temporal.replace(ruta)

    log.info("Manifiesto: entrada %d anotada", len(manifiesto["descargas"]))


def historial(fuente: str | None = None, identificador: str | None = None) -> pd.DataFrame:
    """Bitacora como tabla, opcionalmente filtrada. Util para el informe."""
    descargas = cargar_manifiesto()["descargas"]
    if not descargas:
        return pd.DataFrame()

    tabla = pd.DataFrame(descargas)
    if fuente:
        tabla = tabla[tabla["fuente"] == fuente]
    if identificador:
        tabla = tabla[tabla["identificador"] == identificador]
    return tabla.reset_index(drop=True)


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------


def rango_pendiente(
    fuente: str,
    identificador: str,
    desde: dt.date | None,
    hasta: dt.date,
    forzar: bool,
) -> tuple[dt.date, str]:
    """Decide desde que fecha hay que pedir, y con que modo.

    Sin `--desde` ni `--forzar`, arranca `VENTANA_REFRESCO_DIAS` antes de la
    ultima fecha almacenada: los ultimos dias guardados pueden estar
    incompletos por el rezago de publicacion o desactualizados por una
    reliquidacion posterior.
    """
    if forzar:
        inicio = desde or config.FECHA_INICIO_DEFECTO
        return inicio, "forzado"

    if desde is not None:
        return desde, "rango explicito"

    ultima = ultima_fecha_almacenada(fuente, identificador)
    if ultima is None:
        return config.FECHA_INICIO_DEFECTO, "primera descarga"

    solape = dt.timedelta(days=config.VENTANA_REFRESCO_DIAS)
    inicio = max(ultima - solape, config.FECHA_INICIO_DEFECTO)
    log.info(
        "%s/%s: ultima fecha almacenada %s; se re-pide desde %s (solape de %d dias)",
        fuente,
        identificador,
        ultima,
        inicio,
        config.VENTANA_REFRESCO_DIAS,
    )
    return inicio, "incremental"


def descargar(
    fuente: str,
    identificador: str,
    desde: dt.date | None = None,
    hasta: dt.date | None = None,
    forzar: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Descarga, persiste y anota en el manifiesto. Devuelve la entrada anotada."""
    if fuente not in CLIENTES:
        raise ErrorDescarga(
            f"Fuente desconocida: {fuente!r}. Conocidas: {sorted(CLIENTES)}"
        )

    hasta = hasta or dt.date.today()
    inicio, modo = rango_pendiente(fuente, identificador, desde, hasta, forzar)
    comienzo = dt.datetime.now()

    if inicio > hasta:
        log.info(
            "%s/%s: no hay nada pendiente (%s > %s)", fuente, identificador, inicio, hasta
        )
        entrada = _entrada_manifiesto(
            fuente, identificador, kwargs, (inicio, hasta), None, 0, {}, modo, comienzo
        )
        anotar_manifiesto(entrada)
        return entrada

    # Con --forzar se ignora tambien la cache del cliente: la intencion es
    # volver a preguntarle a la API, no releer lo mismo desde disco.
    cliente = CLIENTES[fuente](usar_cache=not forzar)
    datos = cliente.consultar(identificador, inicio, hasta, **kwargs)

    if datos.empty:
        log.warning(
            "%s/%s: %s..%s no devolvio registros", fuente, identificador, inicio, hasta
        )
        obtenido = None
        particiones: dict[str, dict[str, Any]] = {}
    else:
        marcas = pd.to_datetime(datos["timestamp"])
        obtenido = (marcas.min().date(), marcas.max().date())
        particiones = escribir(datos, fuente, identificador)

    entrada = _entrada_manifiesto(
        fuente,
        identificador,
        kwargs,
        (inicio, hasta),
        obtenido,
        len(datos),
        particiones,
        modo,
        comienzo,
        hash_resultado=hash_marco(datos),
    )
    anotar_manifiesto(entrada)
    return entrada


def _entrada_manifiesto(
    fuente: str,
    identificador: str,
    parametros: dict[str, Any],
    solicitado: tuple[dt.date, dt.date],
    obtenido: tuple[dt.date, dt.date] | None,
    n_registros: int,
    particiones: dict[str, dict[str, Any]],
    modo: str,
    comienzo: dt.datetime,
    hash_resultado: str | None = None,
) -> dict[str, Any]:
    """Construye la entrada de bitacora de una descarga."""
    return {
        "timestamp_ejecucion": comienzo.isoformat(timespec="seconds"),
        "duracion_segundos": round((dt.datetime.now() - comienzo).total_seconds(), 2),
        "fuente": fuente,
        "identificador": identificador,
        "parametros": {k: str(v) for k, v in parametros.items()},
        "modo": modo,
        "rango_solicitado": [solicitado[0].isoformat(), solicitado[1].isoformat()],
        "rango_obtenido": (
            [obtenido[0].isoformat(), obtenido[1].isoformat()] if obtenido else None
        ),
        "n_registros": n_registros,
        "hash_resultado": hash_resultado,
        "particiones_escritas": particiones,
        "n_registros_total": n_registros_almacenados(fuente, identificador),
        "entorno": _entorno(),
    }


def borrar_dataset(fuente: str, identificador: str) -> None:
    """Borra el conjunto entero de la capa cruda. Solo para pruebas."""
    raiz = ruta_dataset(fuente, identificador)
    if raiz.exists():
        shutil.rmtree(raiz)
        log.warning("%s/%s: conjunto borrado de %s", fuente, identificador, raiz)
