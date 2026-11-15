"""Sincronizacion minima entre un prefijo de S3 y una carpeta local.

La usan la Lambda de ingesta y las tareas de Fargate: el codigo del proyecto
lee y escribe `data/` en disco (asi lo prueban las 270+ pruebas), y este modulo
trae esa carpeta desde S3 antes de ejecutar y la devuelve despues. Solo sube
los archivos cuyo contenido cambio (comparando MD5 con el ETag), para no
generar versiones nuevas de objetos identicos en un bucket versionado.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _cliente():
    import boto3  # disponible en la imagen base de Lambda y en requirements-nube.txt

    return boto3.client("s3")


def _md5(ruta: Path) -> str:
    resumen = hashlib.md5()  # noqa: S324 - comparacion con el ETag, no seguridad
    with ruta.open("rb") as origen:
        for bloque in iter(lambda: origen.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def bajar(bucket: str, prefijo: str, destino: Path, cliente=None) -> int:
    """Copia s3://bucket/prefijo* a destino/, conservando la estructura. Devuelve archivos."""
    cliente = cliente or _cliente()
    n = 0
    for pagina in cliente.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefijo):
        for objeto in pagina.get("Contents", []):
            clave = objeto["Key"]
            if clave.endswith("/"):
                continue
            local = destino / clave[len(prefijo):]
            local.parent.mkdir(parents=True, exist_ok=True)
            cliente.download_file(bucket, clave, str(local))
            n += 1
    log.info("s3://%s/%s -> %s: %d archivos", bucket, prefijo, destino, n)
    return n


def subir(origen: Path, bucket: str, prefijo: str, cliente=None) -> int:
    """Sube origen/** a s3://bucket/prefijo, solo los archivos nuevos o cambiados."""
    cliente = cliente or _cliente()
    remotos: dict[str, str] = {}
    for pagina in cliente.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefijo):
        for objeto in pagina.get("Contents", []):
            remotos[objeto["Key"]] = objeto["ETag"].strip('"')
    n = 0
    for ruta in sorted(p for p in origen.rglob("*") if p.is_file()):
        clave = prefijo + ruta.relative_to(origen).as_posix()
        if remotos.get(clave) == _md5(ruta):
            continue
        cliente.upload_file(str(ruta), bucket, clave)
        n += 1
    log.info("%s -> s3://%s/%s: %d archivos nuevos o cambiados", origen, bucket, prefijo, n)
    return n
