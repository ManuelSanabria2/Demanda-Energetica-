"""Coste de un modelo: tiempo de ajuste, tiempo de prediccion y tamano serializado.

- `medir_ajuste`: segundos de `fit`.
- `medir_prediccion`: milisegundos por 1000 filas, mediana de varias
  repeticiones para que un pico del sistema no decida la cifra.
- `tamano_serializado`: bytes del modelo serializado con joblib, los mismos que
  escribe `guardar_modelo` en models/. Ese archivo va con su SHA-256, que se
  anota en el registro del experimento y `cargar_modelo` comprueba al leerlo.

Los tiempos dependen de la maquina: cada medicion va acompanada de `entorno()`,
y solo se comparan entre si los tiempos medidos en la misma ejecucion.
"""

from __future__ import annotations

import hashlib
import io
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from modelado.datos import registro_operacion, sha256_archivo

FILAS_REFERENCIA = 1000
REPETICIONES = 5


class ErrorModelo(Exception):
    """El modelo guardado no es el que se espera."""


def entorno() -> dict[str, str]:
    """Versiones y plataforma con las que se midieron los tiempos y se serializo el modelo."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "sistema": f"{platform.system()} {platform.machine()}",
    }


def medir_ajuste(modelo, X, y) -> tuple[Any, dict[str, Any]]:
    """Ajusta `modelo` con (X, y) y mide cuanto tarda. Devuelve (modelo ajustado, registro)."""
    inicio = time.perf_counter()
    modelo.fit(X, y)
    segundos = time.perf_counter() - inicio
    registro = registro_operacion(
        "medir_ajuste",
        "time.perf_counter alrededor de una sola llamada a fit",
        segundos=segundos,
        filas=int(len(X)),
        columnas=int(X.shape[1]),
    )
    return modelo, registro


def medir_prediccion(
    modelo, X, repeticiones: int = REPETICIONES
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predice X `repeticiones` veces y da la mediana en ms por 1000 filas.

    Devuelve (prediccion, registro). Falla si dos repeticiones no dan la misma
    prediccion: un modelo ajustado tiene que ser determinista al predecir.
    """
    if repeticiones < 1:
        raise ValueError(f"repeticiones debe ser >= 1: {repeticiones}")
    if len(X) == 0:
        raise ValueError("No hay filas que predecir.")

    duraciones, prediccion = [], None
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        actual = np.asarray(modelo.predict(X))
        duraciones.append(time.perf_counter() - inicio)
        if prediccion is not None and not np.array_equal(prediccion, actual):
            raise ErrorModelo("Dos predicciones del mismo modelo sobre las mismas filas difieren.")
        prediccion = actual

    mediana = statistics.median(duraciones)
    registro = registro_operacion(
        "medir_prediccion",
        f"mediana de {repeticiones} llamadas a predict, escalada a {FILAS_REFERENCIA} filas",
        ms_por_1000_filas=1000.0 * mediana * FILAS_REFERENCIA / len(X),
        segundos_mediana=mediana,
        filas=int(len(X)),
        repeticiones=repeticiones,
    )
    return prediccion, registro


def serializar(modelo) -> bytes:
    """El modelo serializado con joblib, sin comprimir."""
    destino = io.BytesIO()
    joblib.dump(modelo, destino)
    return destino.getvalue()


def tamano_serializado(modelo) -> int:
    """Bytes del modelo serializado: el tamano del archivo que escribe `guardar_modelo`."""
    return len(serializar(modelo))


def guardar_modelo(modelo, ruta: Path) -> dict[str, Any]:
    """Escribe el modelo en `ruta` (de forma atomica) y devuelve su tamano y SHA-256."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contenido = serializar(modelo)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_bytes(contenido)
    temporal.replace(ruta)
    return {
        "archivo": ruta.name,
        "formato": "joblib",
        "bytes": len(contenido),
        "sha256": hashlib.sha256(contenido).hexdigest(),
    }


def cargar_modelo(ruta: Path, sha256: str | None = None):
    """Lee un modelo de models/. Con `sha256`, falla si el archivo no es el anotado."""
    ruta = Path(ruta)
    if sha256 is not None:
        huella = sha256_archivo(ruta)
        if huella != sha256:
            raise ErrorModelo(
                f"El SHA-256 de {ruta.name} ({huella[:12]}...) no es el anotado "
                f"({sha256[:12]}...): no es el modelo del registro."
            )
    return joblib.load(ruta)
