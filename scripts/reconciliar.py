"""Reconcilia XM contra SIMEM y resuelve empiricamente la convencion HourNN.

La correspondencia entre la columna HourNN de XM y la hora de reloj no esta
documentada en la respuesta de la API. En vez de asumirla, se mide: si el
alineamiento es correcto, la serie horaria de XM y la de SIMEM (misma fuente
primaria, XM S.A. E.S.P.) deben correlacionar casi perfectamente. Un desfase
de una hora lo delata de inmediato.

    python scripts/reconciliar.py --mes 2026-08
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingesta import config, normalizar  # noqa: E402
from ingesta.clientes import ClienteSIMEM, ClienteXM  # noqa: E402

DESFASES_A_PROBAR = (-1, 0, 1)


def serie_xm(inicio: dt.date, fin: dt.date) -> pd.Series:
    """Serie horaria nacional de demanda real segun la API de XM."""
    crudo = ClienteXM().consultar(
        config.METRICA_OBJETIVO_XM, inicio, fin, entidad=config.ENTIDAD_OBJETIVO_XM
    )
    tabla = normalizar.xm_cliente_a_esquema_comun(crudo)
    return tabla.set_index("fecha_hora")["valor_kwh"].sort_index()


def serie_simem(inicio: dt.date, fin: dt.date) -> pd.Series:
    """Serie horaria nacional de demanda real segun SIMEM, ya sin duplicar versiones."""
    crudo = ClienteSIMEM().consultar(config.DATASET_DEMANDA_SIMEM, inicio, fin)
    tabla = normalizar.simem_agregar_nacional(crudo)
    return tabla.set_index("fecha_hora")["valor_kwh"].sort_index()


def comparar(a: pd.Series, b: pd.Series, desfase: int) -> dict[str, float]:
    """Compara ambas series desplazando XM `desfase` horas."""
    desplazada = a.copy()
    desplazada.index = desplazada.index + pd.Timedelta(hours=desfase)

    juntas = pd.concat([desplazada.rename("xm"), b.rename("simem")], axis=1).dropna()
    if juntas.empty:
        return {"n": 0, "correlacion": float("nan"), "error_medio_pct": float("nan")}

    error_pct = ((juntas["xm"] - juntas["simem"]).abs() / juntas["simem"]).mean() * 100
    return {
        "n": int(len(juntas)),
        "correlacion": float(juntas["xm"].corr(juntas["simem"])),
        "error_medio_pct": float(error_pct),
    }


def main(argv: list[str] | None = None) -> int:
    """Descarga un mes de ambas fuentes y reporta el desfase optimo."""
    analizador = argparse.ArgumentParser(description="Reconciliacion XM vs SIMEM")
    analizador.add_argument("--mes", default="2026-08", help="mes a comparar, YYYY-MM")
    args = analizador.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    anio, mes = (int(p) for p in args.mes.split("-"))
    inicio = dt.date(anio, mes, 1)
    fin = dt.date(anio, mes, calendar.monthrange(anio, mes)[1])

    print(f"Comparando {inicio} .. {fin}\n")
    a = serie_xm(inicio, fin)
    b = serie_simem(inicio, fin)
    print(f"XM   : {len(a)} horas, {a.index.min()} .. {a.index.max()}")
    print(f"SIMEM: {len(b)} horas, {b.index.min()} .. {b.index.max()}\n")

    resultados = {d: comparar(a, b, d) for d in DESFASES_A_PROBAR}

    print(f"{'desfase':>8} {'n':>6} {'correlacion':>13} {'error medio %':>15}")
    for desfase, r in resultados.items():
        print(
            f"{desfase:>8} {r['n']:>6} {r['correlacion']:>13.6f} "
            f"{r['error_medio_pct']:>15.3f}"
        )

    validos = {d: r for d, r in resultados.items() if r["n"] > 0}
    if not validos:
        print("\nNo hubo horas comparables. Revisa que ambas fuentes cubran el mes.")
        return 1

    mejor = max(validos, key=lambda d: validos[d]["correlacion"])
    print(f"\nDesfase optimo: {mejor} horas (config.DESFASE_HORA_XM = {config.DESFASE_HORA_XM})")
    if mejor == 0:
        print("La convencion asumida para HourNN queda confirmada.")
    else:
        print(
            f"AJUSTAR: cambia config.DESFASE_HORA_XM a "
            f"{config.DESFASE_HORA_XM - mejor} y vuelve a ejecutar la ingesta."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
