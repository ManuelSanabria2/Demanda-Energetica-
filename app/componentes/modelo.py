"""Modelo final de cada version, para la pagina Probar el modelo.

La app no entrena: carga el modelo que dejo scripts/exportar_interfaz.py en
app/datos/ y comprueba, antes de deserializarlo, que su SHA-256 es el anotado en
el manifiesto (joblib ejecuta codigo al cargar; un archivo cambiado no se abre).
Modelo y explicador de SHAP se guardan con st.cache_resource: se cargan una vez
por proceso y se comparten entre sesiones.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from componentes import carga

MANIFIESTO = "app/datos/manifiesto_interfaz.json"


class ModeloAlterado(RuntimeError):
    """El archivo del modelo no es el que describe el manifiesto."""


def relativa(version: str) -> str:
    return f"app/datos/modelo_final_{version}.joblib"


def _sha256(ruta: Path) -> str:
    huella = hashlib.sha256()
    with ruta.open("rb") as origen:
        for bloque in iter(lambda: origen.read(1 << 20), b""):
            huella.update(bloque)
    return huella.hexdigest()


@st.cache_resource(show_spinner="Cargando el modelo…")
def _cargar(ruta: str, sha256: str) -> Any:
    huella = _sha256(Path(ruta))
    if huella != sha256:
        raise ModeloAlterado(f"{Path(ruta).name}: SHA-256 {huella[:12]}… distinto del manifiesto ({sha256[:12]}…). "
                             "Vuelve a correr python scripts/exportar_interfaz.py.")
    return joblib.load(ruta)


@st.cache_resource(show_spinner="Preparando SHAP…")
def _explicador(ruta: str, sha256: str) -> shap.TreeExplainer:
    return shap.TreeExplainer(_cargar(ruta, sha256))


def descripcion(version: str) -> dict[str, Any] | None:
    """Entrada del manifiesto del modelo exportado (con su registro), o None si no existe."""
    entrada = carga.obtener(carga.json_(MANIFIESTO), "archivos", f"modelo_final_{version}.joblib")
    return entrada if entrada and carga.existe(relativa(version)) else None


def cargar(version: str) -> Any | None:
    entrada = descripcion(version)
    return None if entrada is None else _cargar(str(carga.ruta(relativa(version))), entrada["sha256"])


def explicar(version: str, fila: pd.DataFrame) -> tuple[float, pd.Series]:
    """Valor base y contribucion SHAP de cada variable para una fila (exacto en arboles)."""
    entrada = descripcion(version)
    explicacion = _explicador(str(carga.ruta(relativa(version))), entrada["sha256"])(fila)
    base = float(np.ravel(explicacion.base_values)[0])
    return base, pd.Series(explicacion.values[0], index=fila.columns)


def variables(modelo: Any, fila: pd.DataFrame) -> pd.DataFrame:
    """Las columnas que el modelo espera, en su orden y como float64 (igual que en el entrenamiento)."""
    return fila[list(modelo.feature_names_in_)].astype("float64")
