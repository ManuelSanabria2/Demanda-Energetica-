"""Predice la demanda horaria con el modelo final a partir de una tabla de modelado.

    python scripts/predecir.py --tabla r120
    python scripts/predecir.py --tabla r24 --entrada otra_tabla.parquet --salida pred.csv
    python scripts/predecir.py --tabla r120 --desde 2026-08-01 --hasta 2026-08-31

El modelo final de cada tabla y su SHA-256 se leen de
docs/resultados/modelado_seleccion.json (los escribe scripts/evaluar_final.py);
el modelo no se carga si su huella no coincide. La entrada es una tabla de
modelado (columnas de `variables.construir`): se valida con `modelado.datos.separar`,
que rechaza columnas desconocidas o con fuga, y sus variables tienen que ser
exactamente las del modelo. `demanda_mwh` puede faltar o venir vacia: es lo
que se predice. Las filas con alguna variable sin dato no se predicen.

Salida: fecha_hora, prediccion_mwh y, si la entrada trae demanda, demanda_mwh
(parquet o csv segun la extension; sin --salida, se imprime un resumen).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado.datos import COLUMNA_OBJETIVO, COLUMNA_TIEMPO, DIR_SALIDA, ZONA_HORARIA, clave_version, separar  # noqa: E402
from modelado.tiempos import cargar_modelo  # noqa: E402

SELECCION = RAIZ / "docs" / "resultados" / "modelado_seleccion.json"
MODELOS = RAIZ / "models"


class ErrorPrediccion(Exception):
    """La tabla o el modelo no permiten predecir con seguridad."""


def modelo_final(clave: str, seleccion: Path = SELECCION, dir_modelos: Path = MODELOS) -> tuple[Any, dict[str, Any]]:
    """El modelo final de la tabla `clave`, comprobando su SHA-256. Devuelve (modelo, descripcion)."""
    if not Path(seleccion).exists():
        raise ErrorPrediccion(f"No existe {seleccion}: corre antes scripts/evaluar_final.py.")
    contenido = json.loads(Path(seleccion).read_text(encoding="utf-8"))
    if clave not in contenido.get("tablas", {}):
        raise ErrorPrediccion(f"{seleccion} no tiene modelo final para {clave}.")
    final = contenido["tablas"][clave]["final"]
    ruta = Path(dir_modelos) / Path(final["modelo"]["ruta"]).name
    if not ruta.exists():
        raise ErrorPrediccion(f"No esta {ruta}: vuelve a correr scripts/evaluar_final.py.")
    return cargar_modelo(ruta, final["modelo"]["sha256"]), {
        "tabla": clave, "modelo": final["etiqueta"], "archivo": ruta.name, "sha256": final["modelo"]["sha256"],
    }


def predecir(modelo, tabla: pd.DataFrame, desde: str | None = None, hasta: str | None = None) -> pd.DataFrame:
    """Predicciones del modelo para las filas de `tabla` (y el rango pedido) con todas sus variables."""
    tabla = tabla.copy()
    if COLUMNA_OBJETIVO not in tabla.columns:
        tabla[COLUMNA_OBJETIVO] = np.nan
    (X, y, fechas), _ = separar(tabla)
    esperadas = list(getattr(modelo, "feature_names_in_", []))
    if esperadas and list(X.columns) != esperadas:
        raise ErrorPrediccion(
            f"Las variables de la tabla no son las del modelo: tabla {list(X.columns)}, modelo {esperadas}. "
            "¿Es la tabla de la otra version (r24 / r120)?"
        )
    momentos = pd.DatetimeIndex(fechas)
    mascara = np.ones(len(X), dtype=bool)
    if desde:
        mascara &= momentos >= pd.Timestamp(desde, tz=ZONA_HORARIA)
    if hasta:
        mascara &= momentos < pd.Timestamp(hasta, tz=ZONA_HORARIA) + pd.Timedelta(days=1)
    completas = mascara & X.notna().all(axis=1).to_numpy()
    if not completas.any():
        raise ErrorPrediccion("No hay filas con todas las variables en el rango pedido.")
    salida = pd.DataFrame({
        COLUMNA_TIEMPO: momentos[completas],
        "prediccion_mwh": np.asarray(modelo.predict(X[completas].astype("float64")), dtype="float64"),
    })
    if y[completas].notna().any():
        salida[COLUMNA_OBJETIVO] = y[completas].to_numpy()
    return salida


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description="Predice con el modelo final")
    analizador.add_argument("--tabla", choices=("r24", "r120"), required=True, help="version del modelo y de la tabla")
    analizador.add_argument("--entrada", type=Path, help="tabla de modelado (por defecto la de data/procesado)")
    analizador.add_argument("--salida", type=Path, help="archivo .parquet o .csv")
    analizador.add_argument("--desde", help="fecha inicial (YYYY-MM-DD, hora local)")
    analizador.add_argument("--hasta", help="fecha final incluida (YYYY-MM-DD)")
    analizador.add_argument("--seleccion", type=Path, default=SELECCION)
    analizador.add_argument("--modelos", type=Path, default=MODELOS)
    args = analizador.parse_args(argv)

    clave = clave_version(args.tabla)
    modelo, descripcion = modelo_final(clave, args.seleccion, args.modelos)
    entrada = args.entrada or DIR_SALIDA / f"tabla_modelado_{clave}.parquet"
    tabla = pd.read_parquet(entrada) if Path(entrada).suffix == ".parquet" else pd.read_csv(entrada)
    if not pd.api.types.is_datetime64_any_dtype(tabla[COLUMNA_TIEMPO]):
        tabla[COLUMNA_TIEMPO] = pd.to_datetime(tabla[COLUMNA_TIEMPO], utc=True).dt.tz_convert(ZONA_HORARIA)
    salida = predecir(modelo, tabla, args.desde, args.hasta)

    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        if args.salida.suffix == ".csv":
            salida.to_csv(args.salida, index=False)
        else:
            salida.to_parquet(args.salida, index=False)
    print(f"{descripcion['modelo']} ({descripcion['archivo']}, SHA-256 {descripcion['sha256'][:12]}...)")
    print(f"{len(salida)} horas: {salida[COLUMNA_TIEMPO].iloc[0]} .. {salida[COLUMNA_TIEMPO].iloc[-1]}")
    if COLUMNA_OBJETIVO in salida:
        real = salida[COLUMNA_OBJETIVO].to_numpy()
        valido = ~np.isnan(real)
        if valido.any():
            error = 100 * np.mean(np.abs(salida["prediccion_mwh"].to_numpy()[valido] - real[valido]) / real[valido])
            print(f"MAPE frente a la demanda de la tabla: {error:.2f} % ({valido.sum()} horas)")
    if args.salida:
        print(f"-> {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
