"""Particion de un rango de fechas en ventanas admitidas por las APIs."""

from __future__ import annotations

import datetime as dt


def partir_rango(
    inicio: dt.date,
    fin: dt.date,
    max_dias: int,
) -> list[tuple[dt.date, dt.date]]:
    """Parte [inicio, fin] en ventanas de a lo sumo `max_dias` dias.

    Ambos extremos son inclusivos y las ventanas no se solapan: la cobertura
    del rango original es exacta. Una ventana de N dias va de d a d+N-1.
    """
    if max_dias < 1:
        raise ValueError(f"max_dias debe ser >= 1, se recibio {max_dias}")
    if fin < inicio:
        raise ValueError(f"El rango [{inicio}..{fin}] esta invertido")

    ventanas: list[tuple[dt.date, dt.date]] = []
    actual = inicio
    while actual <= fin:
        cierre = min(actual + dt.timedelta(days=max_dias - 1), fin)
        ventanas.append((actual, cierre))
        actual = cierre + dt.timedelta(days=1)

    return ventanas


def a_iso(fecha: dt.date) -> str:
    """Formatea una fecha como YYYY-MM-DD, el formato que aceptan ambas APIs."""
    return fecha.isoformat()
