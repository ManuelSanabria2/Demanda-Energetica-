"""Sondeo reproducible de ambas APIs, a traves de la capa de clientes.

Regenera la evidencia sobre la que se diseno la ingesta (notas/hallazgos_apis.md).
Sirve como defensa: si XM o SIMEM cambian la forma de sus respuestas o sus
limites, este script lo hace evidente antes de que el cambio contamine los datos.

A diferencia de scripts/explorar_apis.py, que mira el JSON crudo sin
intermediarios, este comprueba los invariantes de los que depende el codigo.

    python scripts/verificar_apis.py
"""

from __future__ import annotations

import collections
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingesta import config  # noqa: E402
from ingesta.clientes import ClienteSIMEM, ClienteXM, ErrorCliente, ErrorHTTP  # noqa: E402


def titulo(texto: str) -> None:
    """Imprime un encabezado de seccion."""
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def verificar_xm() -> None:
    """Comprueba catalogo, MaxDays, error de rango y rezago de publicacion."""
    titulo("API XM (SINERGOX)")

    cliente = ClienteXM(usar_cache=False)
    catalogo = cliente.catalogo()
    print(f"Metricas en el catalogo  : {len(catalogo)}")

    fila = catalogo[
        (catalogo["MetricId"] == config.METRICA_OBJETIVO_XM)
        & (catalogo["Entity"] == config.ENTIDAD_OBJETIVO_XM)
    ].iloc[0]
    print(f"DemaReal/Sistema MaxDays : {fila['MaxDays']}  (unidades: {fila['MetricUnits']})")
    print(f"Limites presentes        : {sorted(catalogo['MaxDays'].unique())}")

    tabla = cliente.consultar(
        config.METRICA_OBJETIVO_XM,
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 1),
        entidad=config.ENTIDAD_OBJETIVO_XM,
    )
    print(f"\nUn dia de DemaReal       : {len(tabla)} filas (esperadas 24)")
    print(f"Primera hora del dia     : {tabla['timestamp'].iloc[0]} = {tabla['valor'].iloc[0]}")
    print(f"Tipo de valor            : {tabla['valor'].dtype}")

    print("\nRango que excede el limite de la metrica:")
    try:
        # 6 meses de golpe: la API rechaza cualquier cosa por encima de MaxDays.
        cliente._pedir(
            config.METRICA_OBJETIVO_XM,
            dt.date(2025, 1, 1),
            dt.date(2025, 6, 30),
            entidad=config.ENTIDAD_OBJETIVO_XM,
        )
        print("  no fallo (inesperado)")
    except ErrorHTTP as exc:
        print(f"  HTTP {exc.status}: {exc.cuerpo[:150]}")

    hoy = dt.date.today()
    reciente = cliente.consultar(
        config.METRICA_OBJETIVO_XM,
        hoy - dt.timedelta(days=20),
        hoy,
        entidad=config.ENTIDAD_OBJETIVO_XM,
    )
    ultima = reciente["timestamp"].max().date() if not reciente.empty else None
    print(f"\nUltima fecha publicada   : {ultima}")
    print(f"Rezago de publicacion    : {(hoy - ultima).days if ultima else 'n/d'} dias")


def verificar_simem() -> None:
    """Comprueba catalogo, errores y coexistencia de versiones de liquidacion."""
    titulo("API SIMEM")

    cliente = ClienteSIMEM(usar_cache=False)

    print("datasetId inexistente:")
    try:
        cliente.consultar("zzzzzz", dt.date(2025, 1, 1), dt.date(2025, 1, 1))
        print("  no fallo (inesperado)")
    except (ErrorHTTP, ErrorCliente) as exc:
        print(f"  {str(exc)[:160]}")

    catalogo = cliente.catalogo()
    print(f"\nConjuntos en el catalogo : {len(catalogo)}")
    print(cliente.buscar("Demanda real nacional").to_string(index=False))

    tabla = cliente.consultar(
        config.DATASET_DEMANDA_SIMEM, dt.date(2026, 8, 1), dt.date(2026, 8, 5)
    )
    print(f"\nRegistros 2026-08-01..05 : {len(tabla)}")
    print(f"Columnas                 : {list(tabla.columns)}")

    versiones = collections.Counter(tabla["Version"])
    print(f"Versiones coexistentes   : {dict(versiones)}")
    if len(versiones) > 1:
        print(
            "  ATENCION: agregar sin colapsar versiones multiplicaria la demanda "
            f"por {len(versiones)}."
        )

    por_hora = tabla.groupby("timestamp").size()
    print(f"Filas por marca de tiempo: min={por_hora.min()} max={por_hora.max()}")

    faltan = set(tabla["Version"].dropna().unique()) - set(config.PRECEDENCIA_VERSIONES)
    if faltan:
        print(f"  ATENCION: versiones sin precedencia definida: {sorted(faltan)}")
    else:
        print("Precedencia de versiones : cubre todas las presentes")


def main() -> int:
    """Ejecuta los dos sondeos."""
    verificar_xm()
    verificar_simem()
    print("\nSondeo completo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
