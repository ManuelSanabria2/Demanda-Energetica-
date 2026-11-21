"""Entrena y evalua una familia de modelos sobre una tabla de modelado.

    python scripts/entrenar.py --tabla r24  --familia baseline
    python scripts/entrenar.py --tabla r120 --familia baseline
    python scripts/entrenar.py --tabla r24  --familia lineal
    python scripts/entrenar.py --tabla r24  --familia clasificacion
    python scripts/entrenar.py --tabla r24  --familia ensembles
    python scripts/entrenar.py --tabla r24  --familia busqueda

Ajusta con el bloque de entrenamiento, evalua en validacion y escribe una
entrada por tabla en docs/resultados (FAMILIAS: baseline ->
modelado_baselines.json, lineal -> modelado_regresion.json, clasificacion ->
modelado_clasificacion.json, con su figura precision-recall en
docs/figuras/modelado; ensembles -> modelado_ensembles.json; busqueda ->
modelado_busqueda.json, que lee los resultados de lineal, ensembles y
clasificacion para elegir sus candidatos): correr r24 y
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
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado import busqueda  # noqa: E402
from modelado.clasificacion import MODELOS as MODELOS_CLASIFICACION  # noqa: E402
from modelado.clasificacion import evaluar_baseline, evaluar_clasificador  # noqa: E402
from modelado.baselines import (  # noqa: E402
    BASELINE_REFERENCIA,
    BASELINES,
    DESCRIPCIONES,
    baselines_disponibles,
)
from modelado.ensembles import FORMAS, MODELOS as MODELOS_ENSEMBLES  # noqa: E402
from modelado.ensembles import comparar_formas, evaluar_ensemble  # noqa: E402
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
    "ensembles": "modelado_ensembles.json",
    "busqueda": "modelado_busqueda.json",
}
# Familias cuyos resultados lee la busqueda para elegir candidatos.
ENTRADAS_BUSQUEDA = ("lineal", "ensembles", "clasificacion")
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


def evaluar_ensembles(
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    indices: dict,
    clave: str,
    dir_modelos: Path,
    dir_figuras: Path | None = None,
) -> dict[str, Any]:
    """Cada ensemble sobre el nivel y sobre la serie diferenciada, con su comparacion.

    CV en entrenamiento y evaluacion en validacion; la mejora de MAPE es frente
    al baseline estacional en las mismas horas. La prueba no entra.
    """
    entrena, valida = indices["entrenamiento"], indices["validacion"]
    if fechas.iloc[valida].max() >= inicio_prueba() or fechas.iloc[entrena].max() >= inicio_prueba():
        raise RuntimeError("Filas del bloque de prueba en el ajuste o la evaluacion.")  # defensa

    tscv, registro_cv = validacion_cruzada(clave, fechas.iloc[entrena])
    columna_base = BASELINES[BASELINE_REFERENCIA]
    mape_base = mape(y.iloc[valida], X[columna_base].iloc[valida])

    resultados: dict[str, Any] = {}
    comparacion: dict[str, Any] = {}
    for nombre in MODELOS_ENSEMBLES:
        resultados[nombre] = {}
        for forma in FORMAS:
            log.info("%s %s (%s): CV y validacion", clave, nombre, forma)
            modelo, registro = evaluar_ensemble(nombre, forma, X, y, fechas, indices, tscv, mape_base)
            ruta = dir_modelos / f"ensemble_{nombre}_{forma}_{clave}.joblib"
            resultados[nombre][forma] = {**registro, "modelo": {**guardar_modelo(modelo, ruta), "ruta": _relativa(ruta)}}
        comparacion[nombre] = comparar_formas(resultados[nombre])
    return {
        "validacion_cruzada": registro_cv,
        "baseline_referencia": {
            "nombre": BASELINE_REFERENCIA,
            "descripcion": DESCRIPCIONES[BASELINE_REFERENCIA],
            "mape_validacion_pct": mape_base,
        },
        "maximo_entrenamiento_mwh": float(y.iloc[entrena].max()),
        "comparacion_formas": comparacion,
        "modelos": resultados,
    }


def figura_busqueda(registros: dict[str, Any], clave: str, ruta: Path) -> Path:
    """Mejor puntaje de CV acumulado frente al computo acumulado, Grid y Random, un panel por candidato."""
    from mineria import estilo

    estilo.aplicar()
    n = len(registros)
    columnas = 2
    filas = math.ceil(n / columnas)
    figura, ejes = estilo.plt.subplots(filas, columnas, figsize=(9.0, 3.3 * filas), squeeze=False)
    colores = {"grid": estilo.CATEGORICA[0], "random": estilo.CATEGORICA[1]}
    for eje, (etiqueta, registro) in zip(ejes.ravel(), registros.items()):
        regresion = registro["tipo"] == "regresion"
        signo = -1.0 if regresion else 1.0  # MAE positivo en el eje
        for metodo in busqueda.METODOS:
            r = registro[metodo]
            computo = [p["computo_acumulado_s"] for p in r["trayectoria"]]
            eje.scatter(computo, [signo * p["puntaje_cv"] for p in r["trayectoria"]], s=9, alpha=0.35,
                        color=colores[metodo], linewidths=0)
            eje.step(computo, [signo * p["mejor_puntaje_cv_acumulado"] for p in r["trayectoria"]], where="post",
                     color=colores[metodo], linewidth=1.6,
                     label=f"{'Grid' if metodo == 'grid' else 'Random'} · {r['n_ajustes_cv']} ajustes · "
                           f"{r['segundos_busqueda']:.0f} s de reloj")
        eje.axhline(signo * registro["referencia"]["puntaje_cv"], color=estilo.TINTA_SECUNDARIA,
                    linewidth=0.8, linestyle=":", label="referencia (paso anterior)")
        eje.set(xlabel="computo acumulado (s, suma de ajustes)",
                ylabel="MAE de CV (MWh)" if regresion else "F1 de CV", title=etiqueta)
        if regresion:
            # Los candidatos malos estiran el eje: se recorta al rango de lo que importa.
            mejores = [signo * registro[m]["mejor_puntaje_cv"] for m in busqueda.METODOS]
            techo = max(signo * registro["referencia"]["puntaje_cv"], *mejores)
            eje.set_ylim(min(mejores) * 0.97, techo * 1.12)
        eje.legend(fontsize=7, loc="upper right" if regresion else "lower right")
    for eje in ejes.ravel()[n:]:
        eje.set_visible(False)
    figura.suptitle(f"Grid frente a Random Search · {clave}", fontsize=10)
    figura.tight_layout()
    return estilo.guardar(figura, ruta)


def evaluar_busquedas(
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    indices: dict,
    clave: str,
    dir_modelos: Path,
    dir_figuras: Path | None = None,
    dir_resultados: Path | None = None,
) -> dict[str, Any]:
    """Grid y Random para los mejores de regresion y el mejor clasificador de los pasos anteriores."""
    entrena, valida = indices["entrenamiento"], indices["validacion"]
    if fechas.iloc[valida].max() >= inicio_prueba() or fechas.iloc[entrena].max() >= inicio_prueba():
        raise RuntimeError("Filas del bloque de prueba en el ajuste o la evaluacion.")  # defensa

    carpeta = dir_resultados or RESULTADOS
    previos = {}
    for familia in ENTRADAS_BUSQUEDA:
        ruta = carpeta / FAMILIAS[familia]
        if not ruta.exists():
            raise FileNotFoundError(f"La busqueda necesita {_relativa(ruta)}: corre antes --familia {familia}.")
        previos[familia] = json.loads(ruta.read_text(encoding="utf-8"))
    candidatos, registro_seleccion = busqueda.seleccionar_candidatos(previos, clave)

    objetivo = None
    if any(c.tipo == "clasificacion" for c in candidatos):
        marco, _ = construir_alerta_pico(fechas, y, clave, indices)
        objetivo = marco["alerta_pico"].to_numpy()

    resultados: dict[str, Any] = {}
    for candidato in candidatos:
        log.info("%s %s: referencia, Grid y Random", clave, candidato.etiqueta)
        modelos, registro = busqueda.evaluar_candidato(candidato, X, y, objetivo, fechas, indices, clave)
        for metodo in busqueda.METODOS:
            ruta = dir_modelos / f"busqueda_{candidato.clave}_{metodo}_{clave}.joblib"
            registro[metodo]["modelo"] = {**guardar_modelo(modelos[metodo], ruta), "ruta": _relativa(ruta)}
        resultados[candidato.etiqueta] = registro

    ruta_figura = figura_busqueda(resultados, clave, (dir_figuras or FIGURAS) / f"busqueda_puntaje_tiempo_{clave}.png")
    return {
        "seleccion": registro_seleccion,
        "baseline_referencia": {
            "nombre": BASELINE_REFERENCIA,
            "mape_validacion_pct": mape(y.iloc[valida], X[BASELINES[BASELINE_REFERENCIA]].iloc[valida]),
        },
        "tolerancia_relativa": busqueda.TOLERANCIA_RELATIVA,
        "modelos": resultados,
        "figura": _relativa(ruta_figura),
    }


EVALUADORES = {
    "baseline": evaluar_baselines,
    "lineal": evaluar_lineales,
    "clasificacion": evaluar_clasificadores,
    "ensembles": evaluar_ensembles,
    "busqueda": evaluar_busquedas,
}


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
    extra = {"dir_resultados": args.resultados} if args.familia == "busqueda" else {}
    evaluacion = EVALUADORES[args.familia](X, y, fechas, indices, clave, args.modelos, args.figuras, **extra)

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
    if args.familia == "ensembles":
        print(f"  maximo de entrenamiento {evaluacion['maximo_entrenamiento_mwh']:.0f} MWh")
        for nombre, formas in evaluacion["modelos"].items():
            for forma, resultado in formas.items():
                m, e = resultado["metricas_validacion"], resultado["extrapolacion_validacion"]
                print(
                    f"  {nombre:<17} {forma:<12} MAPE {m['mape_pct']:.2f} % · pico {m['mape_pico_pct']:.2f} % · "
                    f"RMSE {m['rmse_mwh']:.0f} · sesgo {m['sesgo_medio_mwh']:+.0f} MWh · mejora "
                    f"{m['mejora_mape_pct']:+.1f} % · sobre el maximo {e['predicciones_sobre_maximo']} pred / "
                    f"{e['reales_sobre_maximo']} reales · ajuste {resultado['tiempos']['ajuste_s']:.1f} s"
                )
    if args.familia == "busqueda":
        for etiqueta, registro in evaluacion["modelos"].items():
            regresion = registro["tipo"] == "regresion"
            metrica = "mae_mwh" if regresion else "f1"
            ref = registro["referencia"]
            print(f"  {etiqueta:<30} referencia CV {ref['puntaje_cv']:+.4g} · validacion "
                  f"{metrica} {ref['metricas_validacion'][metrica]:.4g}")
            for metodo in busqueda.METODOS:
                r = registro[metodo]
                print(f"  {'':<30} {metodo:<6} CV {r['mejor_puntaje_cv']:+.4g} · validacion {metrica} "
                      f"{r['metricas_validacion'][metrica]:.4g} · {r['n_ajustes']} ajustes · "
                      f"{r['segundos_total']:.0f} s · {r['mejores']}")
            c = registro["comparacion"]
            print(f"  {'':<30} random alcanza a grid: {c['random_alcanza_grid']} "
                  f"({c['ajustes_cv_random_hasta_alcanzar']} ajustes; grid llega a su mejor en "
                  f"{c['ajustes_cv_grid_hasta_su_mejor']})")
        print(f"  figura -> {evaluacion['figura']}")
    for nombre, resultado in ({} if args.familia in ("clasificacion", "ensembles", "busqueda") else evaluacion["modelos"]).items():
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
