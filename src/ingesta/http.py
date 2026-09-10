"""Sesion HTTP compartida y traduccion de los errores de ambas APIs.

Las dos APIs senalan los errores de forma distinta y ninguna de las dos es
uniforme, asi que el parseo de la respuesta nunca asume JSON:

- XM devuelve 400 con cuerpo de texto plano que empieza por "error:".
- SIMEM devuelve 400 en dos formatos JSON distintos: el de validacion de
  ASP.NET (clave "errors") y uno propio (claves "status"/"message").
"""

from __future__ import annotations

import json
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config


class ErrorAPI(Exception):
    """Error devuelto por una de las APIs de XM S.A. E.S.P."""


class ErrorAPIXM(ErrorAPI):
    """Error devuelto por la API de XM (SINERGOX)."""


class ErrorAPISIMEM(ErrorAPI):
    """Error devuelto por la API de SIMEM."""


def crear_sesion() -> requests.Session:
    """Crea una sesion con reintentos y backoff exponencial.

    Solo se reintentan fallos transitorios (5xx y 429). Un 400 significa
    "consulta invalida" en ambas APIs y reintentarlo nunca ayuda.
    """
    sesion = requests.Session()
    reintentos = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adaptador = HTTPAdapter(max_retries=reintentos)
    sesion.mount("https://", adaptador)
    sesion.mount("http://", adaptador)
    sesion.headers.update({"Accept": "application/json"})
    return sesion


def _texto_recortado(respuesta: requests.Response, limite: int = 400) -> str:
    """Devuelve el cuerpo de la respuesta recortado, para mensajes de error."""
    texto = respuesta.text.strip().replace("\n", " ")
    return texto[:limite]


def pedir_xm(
    sesion: requests.Session,
    granularidad: str,
    cuerpo: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta un POST contra la API de XM y devuelve el JSON de respuesta.

    `granularidad` es uno de: hourly, daily, monthly, lists.
    """
    url = f"{config.URL_XM}/{granularidad}"
    respuesta = sesion.post(url, json=cuerpo, timeout=config.TIMEOUT_SEGUNDOS)

    if respuesta.status_code != 200:
        raise ErrorAPIXM(
            f"XM {granularidad} devolvio {respuesta.status_code} "
            f"para {cuerpo}: {_texto_recortado(respuesta)}"
        )

    try:
        return respuesta.json()
    except json.JSONDecodeError as exc:
        # XM responde 200 con texto plano en algunos casos de borde.
        raise ErrorAPIXM(
            f"XM {granularidad} devolvio 200 con cuerpo no-JSON "
            f"para {cuerpo}: {_texto_recortado(respuesta)}"
        ) from exc


def pedir_simem(
    sesion: requests.Session,
    dataset_id: str,
    fecha_inicio: str,
    fecha_fin: str,
) -> dict[str, Any]:
    """Ejecuta un GET contra la API de SIMEM y devuelve el JSON de respuesta.

    Las fechas van en formato ISO (YYYY-MM-DD).
    """
    parametros = {
        "datasetId": dataset_id,
        "startDate": fecha_inicio,
        "endDate": fecha_fin,
    }
    respuesta = sesion.get(
        config.URL_SIMEM, params=parametros, timeout=config.TIMEOUT_SEGUNDOS
    )

    if respuesta.status_code != 200:
        raise ErrorAPISIMEM(
            f"SIMEM dataset {dataset_id} [{fecha_inicio}..{fecha_fin}] devolvio "
            f"{respuesta.status_code}: {describir_error_simem(respuesta)}"
        )

    try:
        datos = respuesta.json()
    except json.JSONDecodeError as exc:
        raise ErrorAPISIMEM(
            f"SIMEM dataset {dataset_id} devolvio 200 con cuerpo no-JSON: "
            f"{_texto_recortado(respuesta)}"
        ) from exc

    # SIMEM puede devolver 200 con success=False.
    if not datos.get("success", False):
        raise ErrorAPISIMEM(
            f"SIMEM dataset {dataset_id} [{fecha_inicio}..{fecha_fin}] respondio "
            f"success=False: {_texto_recortado(respuesta)}"
        )

    return datos


def describir_error_simem(respuesta: requests.Response) -> str:
    """Extrae el mensaje de error de SIMEM, sea cual sea de sus dos formatos."""
    try:
        datos = respuesta.json()
    except json.JSONDecodeError:
        return _texto_recortado(respuesta)

    if isinstance(datos, dict):
        # Formato propio de SIMEM: {"status": false, "message": "..."}
        if "message" in datos:
            return str(datos["message"])
        # Formato de validacion de ASP.NET: {"errors": {"campo": ["..."]}}
        if "errors" in datos:
            return json.dumps(datos["errors"], ensure_ascii=False)

    return _texto_recortado(respuesta)
