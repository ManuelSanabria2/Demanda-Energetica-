"""Entrena y evalua una familia de modelos sobre una tabla de modelado.

    python scripts/entrenar.py --tabla r24  --familia baseline
    python scripts/entrenar.py --tabla r120 --familia baseline
    python scripts/entrenar.py --tabla r24  --familia lineal
    python scripts/entrenar.py --tabla r24  --familia clasificacion

Ajusta con el bloque de entrenamiento, evalua en validacion y escribe una
entrada por tabla en docs/resultados (FAMILIAS: baseline ->
modelado_baselines.json, lineal -> modelado_regresion.json, clasificacion ->
modelado_clasificacion.json, con su figura precision-recall en
docs/figuras/modelado): correr r24 y
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

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado.clasificacion import MODELOS as MODELOS_CLASIFICACION  # noqa: E402
from modelado.clasificacion import evaluar_baseline, evaluar_clasificador  # noqa: E402
from modelado.baselines import (  # noqa: E402
    BASELINE_REFERENCIA,
    BASELINES,
    DESCRIPCIONES,
    baselines_disponibles,
)
from modelado.datos import DIR_SALIDA, ahora, cargar_tabla, clave_version, rezago_de  # noqa: E402
from modelado.metricas import mape, metricas_regresion  # noqa: E402
from modelado.objetivos import construir_alerta_pico  # noqa: E402
from modelado.particion import inicio_prueba, particionar, validacion_cruzada  # noqa: E402
from modelado.regresion import MODELOS as MODELOS_LINEALES, evaluar_modelo  # noqa: E402
from modelado.tiempos import entorno, guardar_modelo, medir_ajuste, medir_prediccion  # noqa: E402

log = logging.getLogger("modelado")

VERSION_FORMATO = 1
TABLAS = ("r24", "r120")
# familia -> archivo de resultados en docs/resultados.
FAMILIAS = {
    "baseline": "modelado_baselines.json",
    "lineal": "modelado_regresion.json",
    "clasificacion": "modelado_clasificacion.json",
}
RESULTADOS = RAIZ / "docs" / "resultados"
MODELOS = RAIZ / "models"
FIGURAS = RAIZ / "docs" / "figuras" / "modelado"


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
    dir_figuras: Path | None = None,
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
    dir_figuras: Path | None = None,
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


def figura_precision_recall(curvas: dict[str, tuple[np.ndarray, np.ndarray]], prevalencia: float, clave: str, ruta: Path) -> Path:
    """Curvas precision-recall en validacion, una por clasificador y el baseline."""
    from sklearn.metrics import precision_recall_curve

    from mineria import estilo

    estilo.aplicar()
    figura, ejes = estilo.plt.subplots(figsize=(5.2, 4.0))
    for posicion, (nombre, (real, puntajes)) in enumerate(curvas.items()):
        precision, recall, _ = precision_recall_curve(real, puntajes)
        es_base = nombre == "baseline"
        ejes.plot(
            recall,
            precision,
            label=nombre,
            color=estilo.TINTA_SECUNDARIA if es_base else estilo.CATEGORICA[posicion % len(estilo.CATEGORICA)],
            linestyle="--" if es_base else "-",
            linewidth=1.3 if es_base else 1.6,
        )
    ejes.axhline(prevalencia, color=estilo.TINTA_SECUNDARIA, linewidth=0.8, linestyle=":")
    ejes.annotate(f"prevalencia {100 * prevalencia:.1f} %", (1.0, prevalencia), xytext=(-4, 4),
                  textcoords="offset points", ha="right", fontsize=7.5, color=estilo.TINTA_SECUNDARIA)
    ejes.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="recall", ylabel="precision",
             title=f"alerta_pico en validacion · {clave}")
    ejes.legend(loc="upper right", fontsize=8)
    return estilo.guardar(figura, ruta)


def evaluar_clasificadores(
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    indices: dict,
    clave: str,
    dir_modelos: Path,
    dir_figuras: Path | None = None,
) -> dict[str, Any]:
    """alerta_pico: cada clasificador y el baseline, evaluados en validacion, con la figura PR."""
    entrena, valida = indices["entrenamiento"], indices["validacion"]
    if fechas.iloc[valida].max() >= inicio_prueba() or fechas.iloc[entrena].max() >= inicio_prueba():
        raise RuntimeError("Filas del bloque de prueba en el ajuste o la evaluacion.")  # defensa

    marco, registro_objetivo = construir_alerta_pico(fechas, y, clave, indices)
    objetivo = marco["alerta_pico"].to_numpy()
    real = objetivo[valida].astype(int)

    resultados: dict[str, Any] = {}
    curvas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for nombre in MODELOS_CLASIFICACION:
        log.info("%s %s: busqueda y validacion", clave, nombre)
        modelo, registro, puntajes = evaluar_clasificador(nombre, X, objetivo, fechas, indices, clave)
        ruta = dir_modelos / f"clasificacion_{nombre}_{clave}.joblib"
        resultados[nombre] = {**registro, "modelo": {**guardar_modelo(modelo, ruta), "ruta": _relativa(ruta)}}
        curvas[nombre] = (real, puntajes)

    baseline, puntajes_base = evaluar_baseline(X, objetivo, marco["umbral_alerta_mwh"].to_numpy(), indices)
    curvas["baseline"] = (real, puntajes_base)
    ruta_figura = figura_precision_recall(
        curvas, float(real.mean()), clave, (dir_figuras or FIGURAS) / f"precision_recall_{clave}.png"
    )
    return {
        "objetivo": registro_objetivo,
        "baseline": baseline,
        "modelos": resultados,
        "figura": _relativa(ruta_figura),
    }


EVALUADORES = {"baseline": evaluar_baselines, "lineal": evaluar_lineales, "clasificacion": evaluar_clasificadores}


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
    analizador.add_argument("--figuras", type=Path, default=FIGURAS)
    args = analizador.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    clave = clave_version(args.tabla)

    (X, y, fechas), registro_datos = cargar_tabla(clave, args.datos)
    indices, registro_particion = particionar(fechas)
    evaluacion = EVALUADORES[args.familia](X, y, fechas, indices, clave, args.modelos, args.figuras)

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
    if args.familia == "clasificacion":
        filas = {**evaluacion["modelos"], "baseline": evaluacion["baseline"]}
        for nombre, resultado in filas.items():
            m = resultado["metricas_validacion"]
            print(
                f"  {nombre:<11} PR-AUC {m['pr_auc']:.3f} · ROC-AUC {m['roc_auc']:.3f} · F1 {m['f1'] or 0:.3f} · "
                f"precision {m['precision'] or 0:.3f} · recall {m['recall'] or 0:.3f} · "
                f"{resultado.get('hiperparametros', '')}"
            )
        print(f"  figura -> {evaluacion['figura']}")
    for nombre, resultado in ({} if args.familia == "clasificacion" else evaluacion["modelos"]).items():
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
