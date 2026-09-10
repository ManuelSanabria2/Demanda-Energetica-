"""Cliente de la API de SIMEM.

Forma de la respuesta, verificada contra el servidor el 2026-09-09:

    {"parameters": {...},
     "success": true,
     "result": {"idDataset": "14fabb",
                "name": "Demanda real nacional",
                "metadata": {...},
                "filterDate": ...,
                "records": [{"CodigoVariable": "DdaReal",
                             "FechaHora": "2026-08-05 01:00:00",
                             "CodigoSICAgente": "CHVC",
                             "TipoMercado": "No Regulado",
                             "Version": "TX2",
                             "Valor": 62463.18,
                             "UnidadMedida": "kWh",
                             "CodigoDuracion": "PT1H"}, ...],
                "variables": [...], "columns": ..., "tags": [...]}}

`records` es formato largo y las columnas de dimension cambian segun el
dataset. No hay paginacion: la respuesta llega completa, y para 31 dias de
datos horarios son decenas de MB, asi que cada llamado se vuelca a disco
mientras se descarga en vez de mantenerlo entero en memoria como texto.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd
import requests

from . import config, ventanas
from .http import ErrorAPISIMEM, describir_error_simem

log = logging.getLogger(__name__)


def obtener_catalogo(sesion: requests.Session) -> pd.DataFrame:
    """Descarga el catalogo de conjuntos de datos publicados en SIMEM.

    El catalogo es a su vez un dataset (config.DATASET_CATALOGO_SIMEM). Trae
    idDataset, nombreConjuntoDatos, inicioDato, finDato y fechaActualizacion,
    que es la forma de descubrir el id de 6 caracteres de cualquier conjunto.
    """
    ruta = _descargar_a_archivo(
        sesion,
        config.DATASET_CATALOGO_SIMEM,
        dt.date(1990, 1, 1),
        dt.date.today(),
    )
    registros = _leer_registros(ruta)
    return pd.DataFrame(registros)


def buscar_datasets(catalogo: pd.DataFrame, texto: str) -> pd.DataFrame:
    """Filtra el catalogo por coincidencia parcial en el nombre del conjunto."""
    coincide = catalogo["nombreConjuntoDatos"].str.contains(texto, case=False, na=False)
    return catalogo.loc[
        coincide, ["idDataset", "nombreConjuntoDatos", "inicioDato", "finDato"]
    ]


def descargar_dataset(
    sesion: requests.Session,
    dataset_id: str,
    inicio: dt.date,
    fin: dt.date,
    max_dias: int = config.MAX_DIAS_SIMEM,
    conservar_crudo: bool = True,
) -> tuple[pd.DataFrame, list[tuple[dt.date, dt.date]]]:
    """Descarga un dataset de SIMEM partiendo el rango en ventanas.

    Devuelve los registros de todas las ventanas concatenados en un DataFrame
    y la lista de ventanas que volvieron sin registros.

    Con `conservar_crudo=False` el JSON de cada llamado se borra tras leerlo:
    un mes de 14fabb ocupa unos 27 MB y el historico completo pasa de 1.5 GB.
    """
    trozos = ventanas.partir_rango(inicio, fin, max_dias)
    log.info(
        "SIMEM %s: %s..%s en %d llamados de hasta %d dias",
        dataset_id,
        inicio,
        fin,
        len(trozos),
        max_dias,
    )

    marcos: list[pd.DataFrame] = []
    vacias: list[tuple[dt.date, dt.date]] = []

    for desde, hasta in trozos:
        ruta = _descargar_a_archivo(sesion, dataset_id, desde, hasta)
        registros = _leer_registros(ruta)

        if not registros:
            log.warning("SIMEM %s: sin registros en %s..%s", dataset_id, desde, hasta)
            vacias.append((desde, hasta))
            if not conservar_crudo:
                ruta.unlink(missing_ok=True)
            continue

        # Se convierte cada trozo a DataFrame de inmediato: la lista de dicts
        # ocupa varias veces mas memoria que la tabla equivalente.
        marcos.append(pd.DataFrame(registros))

        if not conservar_crudo:
            ruta.unlink(missing_ok=True)

    if not marcos:
        return pd.DataFrame(), vacias

    return pd.concat(marcos, ignore_index=True), vacias


def _descargar_a_archivo(
    sesion: requests.Session,
    dataset_id: str,
    desde: dt.date,
    hasta: dt.date,
) -> Path:
    """Descarga un llamado a SIMEM volcandolo a disco por trozos.

    Devuelve la ruta del JSON crudo, que queda guardado para auditoria.
    """
    config.DIR_CRUDO_SIMEM.mkdir(parents=True, exist_ok=True)
    nombre = f"{dataset_id}_{desde:%Y%m%d}_{hasta:%Y%m%d}.json"
    ruta = config.DIR_CRUDO_SIMEM / nombre

    parametros = {
        "datasetId": dataset_id,
        "startDate": ventanas.a_iso(desde),
        "endDate": ventanas.a_iso(hasta),
    }

    with sesion.get(
        config.URL_SIMEM,
        params=parametros,
        timeout=config.TIMEOUT_SEGUNDOS,
        stream=True,
    ) as respuesta:
        if respuesta.status_code != 200:
            raise ErrorAPISIMEM(
                f"SIMEM dataset {dataset_id} [{desde}..{hasta}] devolvio "
                f"{respuesta.status_code}: {describir_error_simem(respuesta)}"
            )

        with ruta.open("wb") as destino:
            for bloque in respuesta.iter_content(chunk_size=1 << 20):
                destino.write(bloque)

    return ruta


def _leer_registros(ruta: Path) -> list[dict]:
    """Lee `result.records` de una respuesta cruda de SIMEM guardada en disco."""
    with ruta.open("r", encoding="utf-8") as origen:
        datos = json.load(origen)

    if not datos.get("success", False):
        raise ErrorAPISIMEM(
            f"{ruta.name} contiene una respuesta con success=False: "
            f"{datos.get('message', '(sin mensaje)')}"
        )

    return datos.get("result", {}).get("records") or []
