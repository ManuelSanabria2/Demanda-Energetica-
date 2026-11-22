"""Lectura de los artefactos del pipeline, con cache.

La interfaz no calcula nada que no este ya en un artefacto: lee JSON, CSV,
Parquet e imagenes que generan los scripts del repositorio. Cada lector
devuelve `None` si el archivo no existe, para que la pagina muestre el aviso de
pendiente (componentes.pendiente) en vez de fallar.

Las rutas son relativas a la raiz del repositorio. La variable de entorno
`SIEM_INTERFAZ_RAIZ` la cambia: las pruebas la apuntan a un directorio vacio
para comprobar que ninguna pagina falla sin artefactos.

La cache se indexa por ruta, fecha de modificacion y tamano: si un script
regenera un artefacto mientras la app esta abierta, se vuelve a leer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

VARIABLE_RAIZ = "SIEM_INTERFAZ_RAIZ"


def raiz() -> Path:
    """Raiz del repositorio (o la de las pruebas). Se lee en cada llamada, no al importar."""
    valor = os.environ.get(VARIABLE_RAIZ)
    return Path(valor) if valor else Path(__file__).resolve().parents[2]


def ruta(relativa: str) -> Path:
    return raiz() / relativa


def existe(relativa: str) -> bool:
    return ruta(relativa).is_file()


def _firma(relativa: str) -> tuple[str, int, int] | None:
    destino = ruta(relativa)
    if not destino.is_file():
        return None
    estado = destino.stat()
    return str(destino), estado.st_mtime_ns, estado.st_size


@st.cache_data(show_spinner=False)
def _leer_json(destino: str, marca: int, tamano: int) -> Any:
    return json.loads(Path(destino).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _leer_parquet(destino: str, marca: int, tamano: int) -> pd.DataFrame:
    return pd.read_parquet(destino)


@st.cache_data(show_spinner=False)
def _leer_csv(destino: str, marca: int, tamano: int) -> pd.DataFrame:
    return pd.read_csv(destino, encoding="utf-8")


def json_(relativa: str) -> Any | None:
    firma = _firma(relativa)
    return _leer_json(*firma) if firma else None


def parquet(relativa: str) -> pd.DataFrame | None:
    firma = _firma(relativa)
    return _leer_parquet(*firma) if firma else None


def csv(relativa: str) -> pd.DataFrame | None:
    firma = _firma(relativa)
    return _leer_csv(*firma) if firma else None


def imagen(relativa: str) -> str | None:
    """Ruta de la imagen como texto (st.image la acepta), o None si no existe."""
    return str(ruta(relativa)) if existe(relativa) else None


def imagenes(carpeta: str, patron: str) -> list[str]:
    base = ruta(carpeta)
    return [str(p) for p in sorted(base.glob(patron))] if base.is_dir() else []


def obtener(datos: Any, *claves: Any, defecto: Any = None) -> Any:
    """datos[c1][c2]... sin KeyError: un JSON de una version anterior no tumba la pagina."""
    actual = datos
    for clave in claves:
        if isinstance(actual, dict) and clave in actual:
            actual = actual[clave]
        elif isinstance(actual, list) and isinstance(clave, int) and -len(actual) <= clave < len(actual):
            actual = actual[clave]
        else:
            return defecto
    return actual
