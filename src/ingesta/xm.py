"""Cliente de la API de XM (SINERGOX).

Forma de la respuesta horaria, verificada contra el servidor el 2026-09-09:

    {"Metric": {"Id": "DemaReal", "Name": "...", "StartDate": "...", "EndDate": "..."},
     "Items": [{"Date": "2025-01-01",
                "HourlyEntities": [{"Id": "Sistema",
                                    "Values": {"code": "Sistema",
                                               "Hour01": "7306335.34000",
                                               ...,
                                               "Hour24": "7925012.28000"}}]}]}

Dos comportamientos que condicionan el diseno:

- Los valores de `Values` son cadenas, no numeros.
- Un rango sin datos publicados devuelve 200 con `"Items": []`, no un error.
  Los huecos son silenciosos y hay que detectarlos explicitamente.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

import pandas as pd
import requests

from . import config, ventanas
from .http import pedir_xm

log = logging.getLogger(__name__)

# Limite por defecto si el catalogo no informa MaxDays para la metrica pedida.
MAX_DIAS_POR_DEFECTO = 30


def obtener_catalogo_metricas(sesion: requests.Session) -> pd.DataFrame:
    """Descarga el listado de metricas de la API de XM.

    Devuelve un DataFrame con, entre otras, las columnas MetricId, Entity,
    MaxDays, Type, MetricUnits. MaxDays es el limite real de dias por llamado
    y varia por metrica, asi que se lee de aqui en vez de asumirlo.
    """
    cuerpo = {
        "MetricId": "ListadoMetricas",
        "StartDate": "2020-01-01",
        "EndDate": "2020-01-02",
        "Entity": "Sistema",
    }
    datos = pedir_xm(sesion, "lists", cuerpo)

    filas = [
        entidad["Values"]
        for item in datos.get("Items", [])
        for entidad in item.get("ListEntities", [])
    ]
    if not filas:
        raise RuntimeError("El listado de metricas de XM vino vacio")

    return pd.DataFrame(filas)


def max_dias_de_metrica(
    catalogo: pd.DataFrame,
    metric_id: str,
    entity: str,
) -> int:
    """Busca el MaxDays de una metrica en el catalogo.

    Si la metrica no aparece, cae al limite conservador por defecto en vez de
    fallar: es preferible una ingesta lenta a una ingesta que no arranca.
    """
    coincide = catalogo[
        (catalogo["MetricId"] == metric_id) & (catalogo["Entity"] == entity)
    ]
    if coincide.empty:
        log.warning(
            "La metrica %s/%s no esta en el catalogo de XM; se usan %d dias por llamado",
            metric_id,
            entity,
            MAX_DIAS_POR_DEFECTO,
        )
        return MAX_DIAS_POR_DEFECTO

    return int(coincide["MaxDays"].iloc[0])


def descargar_horaria(
    sesion: requests.Session,
    metric_id: str,
    entity: str,
    inicio: dt.date,
    fin: dt.date,
    catalogo: pd.DataFrame | None = None,
    guardar_crudo: bool = True,
) -> tuple[list[dict[str, Any]], list[tuple[dt.date, dt.date]]]:
    """Descarga una metrica horaria partiendo el rango segun su MaxDays.

    Devuelve los `Items` concatenados de todos los llamados y la lista de
    ventanas que volvieron vacias (huecos de publicacion o datos inexistentes).
    """
    if catalogo is None:
        catalogo = obtener_catalogo_metricas(sesion)

    max_dias = max_dias_de_metrica(catalogo, metric_id, entity)
    trozos = ventanas.partir_rango(inicio, fin, max_dias)
    log.info(
        "XM %s/%s: %s..%s en %d llamados de hasta %d dias",
        metric_id,
        entity,
        inicio,
        fin,
        len(trozos),
        max_dias,
    )

    items: list[dict[str, Any]] = []
    vacias: list[tuple[dt.date, dt.date]] = []

    for desde, hasta in trozos:
        cuerpo = {
            "MetricId": metric_id,
            "StartDate": ventanas.a_iso(desde),
            "EndDate": ventanas.a_iso(hasta),
            "Entity": entity,
        }
        datos = pedir_xm(sesion, "hourly", cuerpo)

        if guardar_crudo:
            _guardar_crudo(datos, metric_id, entity, desde, hasta)

        trozo_items = datos.get("Items") or []
        if not trozo_items:
            log.warning("XM %s/%s: sin datos en %s..%s", metric_id, entity, desde, hasta)
            vacias.append((desde, hasta))
        items.extend(trozo_items)

    return items, vacias


def ultima_fecha_con_datos(items: list[dict[str, Any]]) -> dt.date | None:
    """Devuelve la fecha maxima presente en los Items, o None si vinieron vacios.

    Sirve para medir el rezago de publicacion: al 2026-09-09 la demanda real
    solo estaba publicada hasta el 2026-09-06.
    """
    fechas = [item["Date"] for item in items if item.get("Date")]
    if not fechas:
        return None
    return dt.date.fromisoformat(max(fechas)[:10])


def _guardar_crudo(
    datos: dict[str, Any],
    metric_id: str,
    entity: str,
    desde: dt.date,
    hasta: dt.date,
) -> None:
    """Guarda la respuesta cruda de un llamado, para poder auditarla despues."""
    config.DIR_CRUDO_XM.mkdir(parents=True, exist_ok=True)
    nombre = f"{metric_id}_{entity}_{desde:%Y%m%d}_{hasta:%Y%m%d}.json"
    ruta = config.DIR_CRUDO_XM / nombre
    ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
