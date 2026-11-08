"""Construye las tablas de modelado a partir del panel.

    panel_modelado.parquet -> tabla_modelado_r24.parquet  (backtesting academico)
                           -> tabla_modelado_r120.parquet (disponibilidad real)
                           -> registro_variables.json

    python scripts/construir_variables.py
    python scripts/construir_variables.py --rezagos-minimos 24 120 --dias-parciales 5

El panel se construye antes con:

    python scripts/construir_panel.py --imputacion causal
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from variables.construir import (  # noqa: E402
    DIAS_PUBLICACION_PARCIAL,
    DIR_SALIDA,
    REZAGOS_MINIMOS_DEFECTO,
    construir_tabla,
    guardar,
)

log = logging.getLogger("variables")

NOMBRE_PANEL = "panel_modelado.parquet"
NOMBRE_INFORME_PANEL = "informe_panel.json"


def _relativa(ruta: Path) -> str:
    """Ruta relativa al repositorio: el registro no debe llevar rutas locales."""
    try:
        return ruta.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return ruta.name


def contexto_panel(ruta_panel: Path, panel: pd.DataFrame) -> dict:
    """De que panel salen las tablas: ruta, filas y como se construyo."""
    contexto = {
        "ruta": _relativa(ruta_panel),
        "filas": len(panel),
        "columnas": len(panel.columns),
    }
    ruta_informe = ruta_panel.with_name(NOMBRE_INFORME_PANEL)
    if ruta_informe.exists():
        with ruta_informe.open(encoding="utf-8") as origen:
            informe = json.load(origen)
        contexto.update(
            {
                "momento_panel": informe.get("momento"),
                "imputacion": (informe.get("limpieza") or {}).get("imputacion"),
                "rango": informe.get("rango"),
            }
        )
    return contexto


def main(argv: list[str] | None = None) -> int:
    """Construye una tabla por rezago minimo y las guarda con su registro."""
    analizador = argparse.ArgumentParser(description="Tablas de modelado")
    analizador.add_argument("--panel", type=Path, default=DIR_SALIDA / NOMBRE_PANEL)
    analizador.add_argument("--salida", type=Path, default=DIR_SALIDA)
    analizador.add_argument(
        "--rezagos-minimos",
        type=int,
        nargs="+",
        default=list(REZAGOS_MINIMOS_DEFECTO),
        help="ultima hora de demanda disponible al predecir, en horas (>= 24)",
    )
    analizador.add_argument(
        "--dias-parciales",
        type=int,
        default=DIAS_PUBLICACION_PARCIAL,
        help="dias finales que se descartan por publicacion parcial",
    )
    args = analizador.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    if not args.panel.exists():
        raise FileNotFoundError(
            f"No existe {args.panel}. Ejecuta antes: "
            "python scripts/construir_panel.py --imputacion causal"
        )
    panel = pd.read_parquet(args.panel)

    versiones = {
        rezago: construir_tabla(panel, rezago, args.dias_parciales)
        for rezago in args.rezagos_minimos
    }
    rutas = guardar(versiones, args.salida, contexto_panel(args.panel, panel))

    print("\n=== TABLAS DE MODELADO ===")
    print(f"  panel          : {args.panel}  ({len(panel):,} filas)")
    for rezago, (tabla, registro) in sorted(versiones.items()):
        balance = registro["balance_filas"]
        omitidos = [o["columna"] for o in registro["rezagos_omitidos"]]
        print(f"\n  r{rezago}: {rutas[f'r{rezago}']}")
        print(f"    filas        : {len(tabla):,}   columnas: {len(tabla.columns)}")
        print(f"    rango        : {registro['rango'][0]} .. {registro['rango'][1]}")
        print(
            f"    eliminadas   : {balance['arranque_sin_historico']} de arranque, "
            f"{balance['publicacion_parcial']} por publicacion parcial"
        )
        print(f"    omitidos     : {', '.join(omitidos) or 'ninguno'}")
    print(f"\n  registro       : {rutas['registro']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
