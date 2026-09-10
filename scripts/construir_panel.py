"""Construye el panel de modelado a partir de la capa limpia.

Antes habia dos tablas "finales" que no se hablaban: la capa limpia, con la
procedencia de cada valor pero sin variables exogenas, y un panel unido con
clima y calendario pero sin procedencia. Ninguna de las dos servia para
modelar, y no estaba dicho en ninguna parte cual era la buena.

Esta es la buena. Se construye siempre en el mismo orden:

    demanda cruda -> limpieza (procedencia + marcas) -> union con clima y
    calendario -> data/procesado/panel_modelado.parquet

    python scripts/construir_panel.py
    python scripts/construir_panel.py --imputacion causal
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingesta import config  # noqa: E402
from ingesta.complementarias import calendario_horario, unir  # noqa: E402
from limpieza.limpiar import DIR_SALIDA, guardar, limpiar  # noqa: E402

log = logging.getLogger("panel")

NOMBRE_LIMPIA = "demanda_horaria_sin"
NOMBRE_PANEL = "panel_modelado.parquet"
NOMBRE_INFORME = "informe_panel.json"

# Columnas que NO deben usarse como variables: se calculan mirando la serie
# entera y meterlas en un modelo seria fuga temporal.
COLUMNAS_SOLO_DIAGNOSTICO = ("atipico_global", "atipico_iqr_global")


def construir(
    fuente_demanda: str = "xm_demanda_real_sistema",
    metodo_imputacion: str = "temporal",
    incluir_clima: bool = True,
    incluir_calendario: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Limpia la demanda y la une con las fuentes exogenas."""
    ruta = config.DIR_PROCESADO / fuente_demanda
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes: python -m ingesta.cli --fuente xm"
        )

    crudo = pd.read_parquet(ruta)
    crudo = crudo.drop(columns=[c for c in ("anio", "mes") if c in crudo.columns])
    log.info("Demanda cruda: %d filas", len(crudo))

    limpio, registro_limpieza = limpiar(crudo, metodo_imputacion=metodo_imputacion)
    guardar(limpio, registro_limpieza, NOMBRE_LIMPIA)

    inicio = limpio["fecha_hora"].min().date()
    fin = limpio["fecha_hora"].max().date()

    clima = None
    if incluir_clima:
        ruta_clima = config.DIR_PROCESADO / "clima_nacional"
        if ruta_clima.exists():
            clima = pd.read_parquet(ruta_clima)
            clima = clima.drop(columns=[c for c in ("anio", "mes") if c in clima.columns])
        else:
            log.warning("No hay clima en %s; el panel ira sin temperatura", ruta_clima)

    calendario = None
    registro_calendario = {}
    if incluir_calendario:
        calendario, registro_calendario = calendario_horario(inicio, fin)

    panel, informe_union = unir(limpio, clima, calendario)

    informe = {
        "momento": registro_limpieza["momento"],
        "fuente_demanda": fuente_demanda,
        "filas": len(panel),
        "rango": [str(panel["fecha_hora"].min()), str(panel["fecha_hora"].max())],
        "zona_horaria": str(panel["fecha_hora"].dt.tz),
        "limpieza": registro_limpieza["politica"],
        "procedencia": registro_limpieza["procedencia"],
        "calendario": {
            k: registro_calendario.get(k) for k in ("fuente", "n_festivos", "definiciones")
        },
        "union": informe_union,
        "columnas_solo_diagnostico": list(COLUMNAS_SOLO_DIAGNOSTICO),
        "aviso": (
            "Las columnas atipico_global y atipico_iqr_global se calculan sobre "
            "la serie entera: son fuga temporal si entran en un modelo. Usa "
            "atipico y atipico_iqr, que son causales."
        ),
    }
    return panel, informe


def main(argv: list[str] | None = None) -> int:
    """Construye el panel y lo guarda junto a su informe."""
    analizador = argparse.ArgumentParser(description="Panel de modelado")
    analizador.add_argument("--fuente-demanda", default="xm_demanda_real_sistema")
    analizador.add_argument(
        "--imputacion",
        choices=("temporal", "causal"),
        default="temporal",
        help="'temporal' interpola con el valor posterior; 'causal' no mira al futuro",
    )
    analizador.add_argument("--sin-clima", action="store_true")
    analizador.add_argument("--sin-calendario", action="store_true")
    args = analizador.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    panel, informe = construir(
        args.fuente_demanda, args.imputacion, not args.sin_clima, not args.sin_calendario
    )

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta_panel = DIR_SALIDA / NOMBRE_PANEL
    panel.to_parquet(ruta_panel, index=False)
    with (DIR_SALIDA / NOMBRE_INFORME).open("w", encoding="utf-8") as salida:
        json.dump(informe, salida, ensure_ascii=False, indent=2, default=str)

    proc = informe["procedencia"]
    print(f"\n=== PANEL DE MODELADO ===")
    print(f"  archivo        : {ruta_panel}")
    print(f"  filas          : {informe['filas']:,}   columnas: {len(panel.columns)}")
    print(f"  rango          : {informe['rango'][0]} .. {informe['rango'][1]}")
    print(f"  zona horaria   : {informe['zona_horaria']}")
    print(f"  observados     : {proc['pct_observado']}%")
    print(f"  interpolados   : {proc['pct_interpolado']}%   faltantes: {proc['pct_faltante']}%")
    print(f"  atipicos       : {proc['n_atipicos']:,} (causales, aptos como variable)")
    for union in informe["union"]["uniones"]:
        if union["aplicada"]:
            print(f"  {union['fuente']:<14} : {union['pct_sin_cobertura']}% sin cobertura")
    print(f"\n  NO usar como variables: {', '.join(COLUMNAS_SOLO_DIAGNOSTICO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
