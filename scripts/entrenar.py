"""Entrena y evalua una familia de modelos sobre una tabla de modelado.

    python scripts/entrenar.py --tabla r24  --familia baseline
    python scripts/entrenar.py --tabla r120 --familia baseline
    python scripts/entrenar.py --tabla r24  --familia lineal

Ajusta con el bloque de entrenamiento, evalua en validacion y escribe una
entrada por tabla en docs/resultados (FAMILIAS: baseline ->
modelado_baselines.json, lineal -> modelado_regresion.json): correr r24 y
despues r120 deja las dos en el mismo archivo. La familia lineal busca sus
hiperparametros con el TimeSeriesSplit del bloque de entrenamiento. Los modelos ajustados van a
models/ (fuera de git) y su SHA-256 queda en el registro.

El bloque de prueba (2025-09-03 .. 2026-09-02) no se toca: el registro de la
particion solo dice cuantas filas tiene y que rango cubre. Se evalua en la
evaluacion final, no aqui.

Las tablas se construyen antes con:

    python scripts/construir_panel.py --imputacion causal
    python scripts/construir_variables.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado.baselines import (  # noqa: E402
    BASELINE_REFERENCIA,
    BASELINES,
    DESCRIPCIONES,
    baselines_disponibles,
)
from modelado.datos import DIR_SALIDA, ahora, cargar_tabla, clave_version, rezago_de  # noqa: E402
from modelado.metricas import mape, metricas_regresion  # noqa: E402
from modelado.particion import inicio_prueba, particionar, validacion_cruzada  # noqa: E402
from modelado.regresion import MODELOS as MODELOS_LINEALES, evaluar_modelo  # noqa: E402
from modelado.tiempos import entorno, guardar_modelo, medir_ajuste, medir_prediccion  # noqa: E402

log = logging.getLogger("modelado")

VERSION_FORMATO = 1
TABLAS = ("r24", "r120")
# familia -> archivo de resultados en docs/resultados.
FAMILIAS = {"baseline": "modelado_baselines.json", "lineal": "modelado_regresion.json"}
RESULTADOS = RAIZ / "docs" / "resultados"
MODELOS = RAIZ / "models"


def _relativa(ruta: Path) -> str:
    """Ruta relativa al repositorio: el registro no debe llevar rutas locales."""
    try:
        return Path(ruta).resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return Path(ruta).name


def evaluar_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    indices: dict,
    clave: str,
    dir_modelos: Path,
) -> dict[str, Any]:
    """Ajusta cada baseline con entrenamiento, lo evalua en validacion y lo guarda.

    Solo usa las posiciones de entrenamiento y validacion; la prueba no entra.
    """
    entrena, valida = indices["entrenamiento"], indices["validacion"]
    if fechas.iloc[valida].max() >= inicio_prueba() or fechas.iloc[entrena].max() >= inicio_prueba():
        raise RuntimeError("Filas del bloque de prueba en el ajuste o la evaluacion.")  # defensa

    modelos, registro = baselines_disponibles(list(X.columns))
    resultados: dict[str, Any] = {}
    mape_referencia = None
    for nombre, modelo in modelos.items():  # la referencia va primero
        modelo, ajuste = medir_ajuste(modelo, X.iloc[entrena], y.iloc[entrena])
        prediccion, medida = medir_prediccion(modelo, X.iloc[valida])
        es_referencia = nombre == BASELINE_REFERENCIA
        metricas = metricas_regresion(
            y.iloc[valida],
            prediccion,
            fechas.iloc[valida],
            mape_baseline=None if es_referencia else mape_referencia,
        )
        if es_referencia:
            mape_referencia = metricas["mape_pct"]

        ruta = dir_modelos / f"baseline_{nombre}_{clave}.joblib"
        archivo = guardar_modelo(modelo, ruta)
        resultados[nombre] = {
            "descripcion": DESCRIPCIONES[nombre],
            "columna": modelo.columna,
            "metricas_validacion": metricas,
            "mejora_frente_a": None if es_referencia else BASELINE_REFERENCIA,
            "tiempos": {
                "ajuste_s": ajuste["detalle"]["segundos"],
                "prediccion_ms_por_1000_filas": medida["detalle"]["ms_por_1000_filas"],
                "repeticiones_prediccion": medida["detalle"]["repeticiones"],
            },
            "modelo": {**archivo, "ruta": _relativa(ruta)},
        }
    return {"baselines": registro, "modelos": resultados}


def evaluar_lineales(
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    indices: dict,
    clave: str,
    dir_modelos: Path,
) -> dict[str, Any]:
    """Cada modelo de modelado.regresion: busqueda y CV en entrenamiento, evaluacion en validacion.

    La mejora de MAPE es frente al baseline estacional en las mismas horas de
    validacion. La prueba no entra.
    """
    entrena, valida = indices["entrenamiento"], indices["validacion"]
    if fechas.iloc[valida].max() >= inicio_prueba() or fechas.iloc[entrena].max() >= inicio_prueba():
        raise RuntimeError("Filas del bloque de prueba en el ajuste o la evaluacion.")  # defensa

    tscv, registro_cv = validacion_cruzada(clave, fechas.iloc[entrena])
    columna_base = BASELINES[BASELINE_REFERENCIA]
    mape_base = mape(y.iloc[valida], X[columna_base].iloc[valida])

    resultados: dict[str, Any] = {}
    for nombre in MODELOS_LINEALES:
        log.info("%s %s: busqueda, CV y validacion", clave, nombre)
        modelo, registro = evaluar_modelo(nombre, X, y, fechas, indices, tscv, mape_base)
        ruta = dir_modelos / f"lineal_{nombre}_{clave}.joblib"
        resultados[nombre] = {**registro, "modelo": {**guardar_modelo(modelo, ruta), "ruta": _relativa(ruta)}}
    return {
        "validacion_cruzada": registro_cv,
        "baseline_referencia": {
            "nombre": BASELINE_REFERENCIA,
            "descripcion": DESCRIPCIONES[BASELINE_REFERENCIA],
            "mape_validacion_pct": mape_base,
        },
        "modelos": resultados,
    }


EVALUADORES = {"baseline": evaluar_baselines, "lineal": evaluar_lineales}


def escribir_resultados(ruta: Path, familia: str, clave: str, entrada: dict[str, Any]) -> dict[str, Any]:
    """Actualiza la entrada de una tabla sin tocar las demas y reescribe el JSON de forma atomica."""
    resultados: dict[str, Any] = {}
    if ruta.exists():
        resultados = json.loads(ruta.read_text(encoding="utf-8"))
        if resultados.get("familia") != familia:
            raise ValueError(f"{ruta} es de la familia {resultados.get('familia')!r}, no {familia!r}.")
    tablas = dict(resultados.get("tablas") or {})
    tablas[clave] = entrada
    resultados = {
        "version_formato": VERSION_FORMATO,
        "familia": familia,
        "momento": ahora(),
        "bloque_ajuste": "entrenamiento",
        "bloque_evaluacion": "validacion",
        "nota": (
            "El bloque de prueba no se usa: se evalua solo en la evaluacion final. "
            "Los tiempos dependen de la maquina (ver entorno de cada tabla)."
        ),
        "tablas": dict(sorted(tablas.items(), key=lambda par: rezago_de(par[0]))),
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as destino:
        json.dump(resultados, destino, ensure_ascii=False, indent=2, default=str)
    temporal.replace(ruta)
    return resultados


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description="Entrena y evalua una familia de modelos")
    analizador.add_argument("--tabla", choices=TABLAS, required=True, help="version de la tabla de modelado")
    analizador.add_argument("--familia", choices=sorted(FAMILIAS), required=True)
    analizador.add_argument("--datos", type=Path, default=DIR_SALIDA, help="carpeta de las tablas y su registro")
    analizador.add_argument("--resultados", type=Path, default=RESULTADOS)
    analizador.add_argument("--modelos", type=Path, default=MODELOS)
    args = analizador.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    clave = clave_version(args.tabla)

    (X, y, fechas), registro_datos = cargar_tabla(clave, args.datos)
    indices, registro_particion = particionar(fechas)
    evaluacion = EVALUADORES[args.familia](X, y, fechas, indices, clave, args.modelos)

    valida = indices["validacion"]
    entrada = {
        "momento": ahora(),
        "entorno": entorno(),
        "datos": registro_datos,
        "particion": registro_particion,
        "evaluacion": {
            "filas_ajuste": int(len(indices["entrenamiento"])),
            "filas_evaluacion": int(len(valida)),
            "rango_evaluacion": [str(fechas.iloc[valida].min()), str(fechas.iloc[valida].max())],
        },
        **evaluacion,
    }
    ruta = args.resultados / FAMILIAS[args.familia]
    escribir_resultados(ruta, args.familia, clave, entrada)

    rango = entrada["evaluacion"]["rango_evaluacion"]
    print(f"\n=== {args.familia.upper()} · {clave} === validacion {rango[0][:16]} .. {rango[1][:16]}")
    for nombre, resultado in evaluacion["modelos"].items():
        m = resultado["metricas_validacion"]
        mejora = "" if m["mejora_mape_pct"] is None else f" · mejora {m['mejora_mape_pct']:+.1f} %"
        print(
            f"  {nombre:<17} MAPE {m['mape_pct']:.2f} % · pico {m['mape_pico_pct']:.2f} % · "
            f"MAE {m['mae_mwh']:.0f} · RMSE {m['rmse_mwh']:.0f} MWh · max diario "
            f"{m['error_maximo_diario_pct']:.2f} % · sesgo {m['sesgo_medio_mwh']:+.0f} MWh{mejora}"
        )
        if resultado.get("hiperparametros"):
            print(f"  {'':<17} hiperparametros {resultado['hiperparametros']}")
    if "baselines" in evaluacion:
        for nombre, motivo in evaluacion["baselines"]["detalle"]["omitidos"].items():
            print(f"  {nombre:<17} omitido: {motivo}")
    print(f"  -> {_relativa(ruta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
