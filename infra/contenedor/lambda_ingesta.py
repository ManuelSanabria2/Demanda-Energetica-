"""Handler de AWS Lambda para la ingesta incremental diaria.

    raw/ + staging/ (S3) -> /tmp/data -> ingesta incremental -> raw/ + staging/ (S3)

Dos pasos, los mismos que se ejecutan en local:

1. `ingesta.descarga.descargar`: capa cruda particionada con manifiesto
   (XM DemaReal y SIMEM 14fabb), incremental con 45 dias de solape.
2. `ingesta.cli --fuente todas --desde hoy-45`: series normalizadas y clima en
   data/processed/, que es la entrada de scripts/construir_panel.py.

La logica es la del repositorio, sin cambios. Lo
unico que hace este modulo es apuntar las rutas de `ingesta.config` a /tmp (el
unico directorio escribible en Lambda) y sincronizar con S3 antes y despues.

Evento opcional: {"series": [["xm", "DemaReal"], ["simem", "14fabb"]], "forzar": false}
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import s3sync

SERIES_DEFECTO = [("xm", "DemaReal"), ("simem", "14fabb")]
BASE = Path(os.environ.get("DIR_TRABAJO", "/tmp/data"))

logging.getLogger().setLevel(logging.INFO)


def reconfigurar_rutas(base: Path) -> None:
    """Redirige todas las rutas de datos del paquete a `base` (p. ej. /tmp/data)."""
    from ingesta import config

    config.DIR_DATOS = base
    config.DIR_MUESTRAS = base / "raw_samples"
    config.DIR_CRUDO = base / "raw"
    config.DIR_PROCESADO = base / "processed"
    config.DIR_CACHE = base / "cache"
    config.RUTA_MANIFIESTO_CRUDO = base / "raw" / "manifiesto.json"
    config.RUTA_MANIFIESTO = base / "manifiesto.json"
    for directorio in (config.DIR_CRUDO, config.DIR_CACHE, config.DIR_PROCESADO):
        directorio.mkdir(parents=True, exist_ok=True)


def handler(evento, contexto):  # noqa: ARG001 - firma de Lambda
    import datetime as dt

    bucket = os.environ["BUCKET_DATOS"]
    prefijo = os.environ.get("PREFIJO_CRUDO", "raw/")
    prefijo_staging = os.environ.get("PREFIJO_STAGING", "staging/")
    dias = int(os.environ.get("DIAS_REFRESCO", "45"))
    reconfigurar_rutas(BASE)

    from ingesta import cli, descarga

    s3sync.bajar(bucket, prefijo, BASE / "raw")
    s3sync.bajar(bucket, prefijo_staging, BASE / "processed")
    resultados = []
    for fuente, identificador in (evento or {}).get("series", SERIES_DEFECTO):
        entrada = descarga.descargar(fuente, identificador, forzar=bool((evento or {}).get("forzar")))
        resultados.append(
            {
                "fuente": fuente,
                "identificador": identificador,
                "registros": entrada.get("n_registros"),
                "rango_obtenido": entrada.get("rango_obtenido"),
            }
        )
    desde = (dt.date.today() - dt.timedelta(days=dias)).isoformat()
    codigo = cli.main(["--fuente", "todas", "--desde", desde])
    if codigo:
        raise RuntimeError(f"ingesta.cli termino con codigo {codigo}; no se sube staging/")
    subidos = s3sync.subir(BASE / "raw", bucket, prefijo) + s3sync.subir(BASE / "processed", bucket, prefijo_staging)
    return {"estado": "ok", "series": resultados, "desde_staging": desde, "archivos_subidos": subidos}
