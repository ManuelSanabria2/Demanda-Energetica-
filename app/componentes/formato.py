"""Numeros en formato espanol: coma decimal y espacio fino de miles (49 824; 2,21 %)."""

from __future__ import annotations

MILES = " "  # espacio fino no separable


def num(valor: float | int | None, decimales: int = 0) -> str:
    if valor is None:
        return "—"
    texto = f"{float(valor):,.{decimales}f}"
    return texto.replace(",", MILES).replace(".", ",")


def pct(valor: float | None, decimales: int = 1) -> str:
    return "—" if valor is None else f"{num(valor, decimales)} %"


def fecha(texto: str | None) -> str:
    """'2025-09-03 00:00:00-05:00' -> '2025-09-03'."""
    return "—" if not texto else str(texto)[:10]
