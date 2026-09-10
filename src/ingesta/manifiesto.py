"""Manifiesto de ingesta: que se descargo, hasta donde llega y que falta.

El rezago de publicacion no se oculta, se registra. La demanda real de XM se
publica con unos 3 dias de retraso, asi que `ultima_fecha_con_datos` casi
nunca coincide con la fecha de ingesta, y esa diferencia es una limitacion
real del uso operativo del modelo a 24 horas.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from . import config


def cargar() -> dict[str, Any]:
    """Lee el manifiesto existente, o devuelve uno vacio si aun no hay."""
    if not config.RUTA_MANIFIESTO.exists():
        return {"entradas": {}}
    with config.RUTA_MANIFIESTO.open("r", encoding="utf-8") as origen:
        return json.load(origen)


def guardar(manifiesto: dict[str, Any]) -> None:
    """Escribe el manifiesto completo en disco."""
    config.RUTA_MANIFIESTO.parent.mkdir(parents=True, exist_ok=True)
    with config.RUTA_MANIFIESTO.open("w", encoding="utf-8") as destino:
        json.dump(manifiesto, destino, ensure_ascii=False, indent=2)


def registrar(
    clave: str,
    rango_solicitado: tuple[dt.date, dt.date],
    ultima_fecha_con_datos: dt.date | None,
    n_registros: int,
    cobertura: dict[str, Any],
) -> dict[str, Any]:
    """Anade o reemplaza la entrada de una fuente/metrica y guarda el manifiesto.

    `clave` identifica la serie, p. ej. "xm:DemaReal:Sistema".
    """
    hoy = dt.date.today()
    ultima = ultima_fecha_con_datos

    manifiesto = cargar()
    manifiesto["entradas"][clave] = {
        "fecha_ingesta": hoy.isoformat(),
        "rango_solicitado": [rango_solicitado[0].isoformat(), rango_solicitado[1].isoformat()],
        "ultima_fecha_con_datos": ultima.isoformat() if ultima else None,
        "rezago_dias": (hoy - ultima).days if ultima else None,
        "n_registros": n_registros,
        "dias_sin_datos": dias_incompletos(cobertura),
        "cobertura": cobertura,
    }
    guardar(manifiesto)
    return manifiesto


def dias_incompletos(cobertura: dict[str, Any]) -> list[str]:
    """Dias que aparecen entre las horas faltantes del reporte de cobertura."""
    huecos = cobertura.get("primeros_huecos") or []
    return sorted({str(hueco)[:10] for hueco in huecos})
