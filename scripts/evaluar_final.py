"""Evaluacion final en el bloque de prueba (2025-09-03 .. 2026-09-02): el unico paso que lo usa.

    python scripts/evaluar_final.py            # r24 y r120
    python scripts/evaluar_final.py --forzar   # rehace una evaluacion ya escrita

Lee los resultados de los pasos anteriores (baselines, lineal, ensembles,
clasificacion, busqueda e interpretabilidad), reentrena los candidatos con
entrenamiento + validacion y los hiperparametros elegidos, y evalua una vez
en prueba. La eleccion del modelo final usa solo validacion
(`modelado.final.elegir_final`); la prueba dice si el criterio de exito se
sostiene. Todo se calcula antes de escribir: un MAPE < 1 % (AlertaFuga) para
el script sin dejar ningun archivo.

Escribe:
    docs/resultados/modelado_seleccion.json      tabla de seleccion y registro
    docs/resultados/predicciones_prueba.parquet  real y prediccion de cada candidato
    docs/resultados/probabilidades_alerta.parquet
    docs/modelado/seleccion.md                   justificacion (generada: ningun numero a mano)
    docs/figuras/modelado/{residuales_r24,residuales_r120,semana_prueba,mape_prueba}.png
    models/final_{tabla}.joblib y models/prueba_*_{tabla}.joblib (SHA-256 en el JSON)

Si ya existe modelado_seleccion.json no se repite sin --forzar: la prueba se
evalua una vez.
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

from modelado import final  # noqa: E402
from modelado.datos import DIR_SALIDA, ahora, cargar_tabla, clave_version  # noqa: E402
from modelado.objetivos import construir_alerta_pico  # noqa: E402
from modelado.particion import particionar  # noqa: E402
from modelado.tiempos import entorno, guardar_modelo, tamano_serializado  # noqa: E402

log = logging.getLogger("modelado")

VERSION_FORMATO = 1
TABLAS = ("r24", "r120")
RESULTADOS = RAIZ / "docs" / "resultados"
MODELOS = RAIZ / "models"
FIGURAS = RAIZ / "docs" / "figuras" / "modelado"
DOCUMENTO = RAIZ / "docs" / "modelado" / "seleccion.md"
ENTRADAS = {
    "baseline": "modelado_baselines.json",
    "lineal": "modelado_regresion.json",
    "ensembles": "modelado_ensembles.json",
    "clasificacion": "modelado_clasificacion.json",
    "busqueda": "modelado_busqueda.json",
    "interpretabilidad": "modelado_interpretabilidad.json",
}
OPCIONALES = ("busqueda",)
SALIDA = "modelado_seleccion.json"
PREDICCIONES = "predicciones_prueba.parquet"
PROBABILIDADES = "probabilidades_alerta.parquet"
NOMBRES_ROL = {"baseline": "baseline estacional", "lineal": "mejor lineal", "ensemble": "mejor ensemble",
               "clasificador": "mejor clasificador"}


def _relativa(ruta: Path) -> str:
    try:
        return Path(ruta).resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return Path(ruta).name


def _escribir_json(ruta: Path, contenido: dict[str, Any]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as destino:
        json.dump(contenido, destino, ensure_ascii=False, indent=2, default=str)
    temporal.replace(ruta)


def _escribir_parquet(ruta: Path, marco: pd.DataFrame) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".parquet.tmp")
    marco.to_parquet(temporal, index=False)
    temporal.replace(ruta)


def _escribir_texto(ruta: Path, texto: str) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(texto, encoding="utf-8")
    temporal.replace(ruta)


# --------------------------------------------------------------------------
# Calculo (nada se escribe aqui)
# --------------------------------------------------------------------------


def _validacion_sobre_maximo(previos: dict[str, Any], clave: str) -> int | None:
    """Horas de validacion por encima del maximo de entrenamiento, del JSON de ensembles (igual en todos)."""
    try:
        formas = next(iter(previos["ensembles"]["tablas"][clave]["modelos"].values()))
        return next(iter(formas.values()))["extrapolacion_validacion"]["reales_sobre_maximo"]
    except (KeyError, StopIteration):
        return None


def evaluar_tabla(clave: str, previos: dict[str, Any], dir_datos: Path) -> dict[str, Any]:
    (X, y, fechas), registro_datos = cargar_tabla(clave, dir_datos)
    X = X.astype("float64")
    indices, registro_particion = particionar(fechas)
    marco, registro_objetivo = construir_alerta_pico(fechas, y, clave, indices)
    objetivo = marco["alerta_pico"].to_numpy()

    candidatos, registro_seleccion = final.seleccionar_candidatos(previos, clave)
    modelos, ajustes = {}, {}
    for rol, candidato in candidatos.items():
        log.info("%s %s: reentreno con entrenamiento + validacion", clave, candidato.etiqueta)
        modelos[rol], ajustes[rol] = final.reentrenar(candidato, X, y, objetivo, fechas, indices)
    calibrador, registro_calibrador = final.calibrador_platt(candidatos["clasificador"], X, objetivo, indices)

    # La eleccion se hace antes de tocar la prueba: metricas de validacion de cada paso y, para
    # desempatar, el tiempo de prediccion medido sobre las filas de validacion.
    calificaciones = previos["interpretabilidad"]["tablas"][clave]["calificaciones"]
    ms_validacion = final.tiempos_prediccion(modelos, X.iloc[indices["validacion"]])
    rol_final, eleccion = final.elegir_final(candidatos, calificaciones, ms_validacion)

    log.info("%s: evaluacion en prueba", clave)
    evaluacion = final.evaluar_prueba(candidatos, modelos, calibrador, X, y, objetivo, fechas, indices)

    prueba = indices["prueba"]
    return {
        "clave": clave, "X": X, "y": y, "fechas": fechas, "indices": indices, "objetivo": objetivo,
        "umbral": marco["umbral_alerta_mwh"].to_numpy(), "candidatos": candidatos, "modelos": modelos,
        "ajustes": ajustes, "calibrador": calibrador, "registro_calibrador": registro_calibrador,
        "evaluacion": evaluacion, "calificaciones": calificaciones, "rol_final": rol_final, "eleccion": eleccion,
        "residuales": final.residuales(y.iloc[prueba], evaluacion["predicciones"][rol_final],
                                       evaluacion["predicciones"]["baseline"], fechas.iloc[prueba], X.iloc[prueba]),
        "registros": {"datos": registro_datos, "particion": registro_particion, "objetivo": registro_objetivo,
                      "seleccion": registro_seleccion},
    }


def entrada_json(r: dict[str, Any], archivos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """La entrada de una tabla en modelado_seleccion.json: tabla de seleccion de tres ejes."""
    candidatos, evaluacion = r["candidatos"], r["evaluacion"]
    tabla = []
    for rol, c in candidatos.items():
        calificacion = r["calificaciones"].get(c.clave_calificacion, {})
        if rol == "clasificador":
            validacion = {k: c.validacion.get(k) for k in ("pr_auc", "roc_auc", "f1", "precision", "recall")}
            prueba = evaluacion["clasificador"]["metricas"]
        else:
            validacion = {k: c.validacion.get(k) for k in ("mape_pct", "mape_pico_pct", "mejora_mape_pct", "mae_mwh",
                                                           "rmse_mwh", "sesgo_medio_mwh")}
            prueba = evaluacion["metricas"][rol]
        tabla.append({
            "rol": rol,
            "modelo": c.etiqueta,
            "nombre": c.nombre,
            "forma": c.forma,
            "origen": c.origen,
            "hiperparametros": c.hiperparametros,
            "es_final": rol == r["rol_final"],
            "rendimiento": {"validacion": validacion, "prueba": prueba},
            "interpretabilidad": {"calificacion": calificacion.get("calificacion"), "motivo": calificacion.get("motivo")},
            "eficiencia": {
                "ajuste_s": r["ajustes"][rol]["ajuste_s"],
                "prediccion_ms_por_1000_filas": evaluacion["prediccion_ms"][rol],
                "tamano_bytes": tamano_serializado(r["modelos"][rol]),
            },
            "ajuste": r["ajustes"][rol],
            "modelo_guardado": archivos[rol],
        })
    metricas_final = evaluacion["metricas"][r["rol_final"]]
    criterio_prueba = {
        rol: final.cumple_criterio(evaluacion["metricas"][rol], evaluacion["metricas"]["baseline"])
        for rol in final.ROLES_REGRESION if rol != "baseline"
    }
    return {
        "datos": r["registros"]["datos"],
        "particion": r["registros"]["particion"],
        "objetivo_alerta": r["registros"]["objetivo"],
        "seleccion_candidatos": r["registros"]["seleccion"],
        "tabla_seleccion": tabla,
        "eleccion": r["eleccion"],
        "final": {
            "rol": r["rol_final"],
            "etiqueta": candidatos[r["rol_final"]].etiqueta,
            "metricas_prueba": metricas_final,
            "criterio_exito_prueba": criterio_prueba.get(r["rol_final"]),
            "modelo": archivos["final"],
        },
        "criterio_exito_prueba": criterio_prueba,
        "metricas_prueba": evaluacion["metricas"],
        "extrapolacion_prueba": evaluacion["extrapolacion"],
        "clasificador_prueba": evaluacion["clasificador"]["metricas"],
        "calibracion_alerta": r["registro_calibrador"],
        "residuales_final": r["residuales"],
        "candidatos": {rol: c.etiqueta for rol, c in candidatos.items()},
    }


def predicciones_largas(resultados: dict[str, dict[str, Any]]) -> pd.DataFrame:
    partes = []
    for clave, r in resultados.items():
        prueba = r["indices"]["prueba"]
        for rol in final.ROLES_REGRESION:
            partes.append(pd.DataFrame({
                "fecha_hora": r["fechas"].iloc[prueba].to_numpy(),
                "tabla": clave,
                "rol": rol,
                "modelo": r["candidatos"][rol].etiqueta,
                "es_final": rol == r["rol_final"],
                "real_mwh": r["y"].iloc[prueba].to_numpy(dtype="float64"),
                "prediccion_mwh": np.asarray(r["evaluacion"]["predicciones"][rol], dtype="float64"),
            }))
    return pd.concat(partes, ignore_index=True)


def probabilidades(resultados: dict[str, dict[str, Any]]) -> pd.DataFrame:
    partes = []
    for clave, r in resultados.items():
        prueba = r["indices"]["prueba"]
        c = r["evaluacion"]["clasificador"]
        partes.append(pd.DataFrame({
            "fecha_hora": r["fechas"].iloc[prueba].to_numpy(),
            "tabla": clave,
            "modelo": r["candidatos"]["clasificador"].etiqueta,
            "umbral_alerta_mwh": r["umbral"][prueba],
            "real_mwh": r["y"].iloc[prueba].to_numpy(dtype="float64"),
            "alerta_real": c["real"],
            "puntaje": c["puntajes"],
            "probabilidad": c["probabilidad"],
            "alerta_predicha": (c["puntajes"] >= c["umbral"]).astype(int),
        }))
    return pd.concat(partes, ignore_index=True)


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------


def semana_representativa(r: dict[str, Any]) -> pd.Timestamp:
    """Lunes de la semana completa de prueba cuyo MAPE semanal del modelo final es la mediana."""
    prueba = r["indices"]["prueba"]
    momentos = pd.DatetimeIndex(r["fechas"].iloc[prueba])
    real = r["y"].iloc[prueba].to_numpy(dtype="float64")
    pred = np.asarray(r["evaluacion"]["predicciones"][r["rol_final"]], dtype="float64")
    lunes = (momentos - pd.to_timedelta(momentos.dayofweek, unit="D")).normalize()
    marco = pd.DataFrame({"lunes": lunes, "ape": np.abs(pred - real) / real})
    semanas = marco.groupby("lunes")["ape"].agg(["mean", "size"])
    semanas = semanas[semanas["size"] == 168].sort_values("mean")
    return semanas.index[len(semanas) // 2]


def figura_semana(resultados: dict[str, dict[str, Any]], ruta: Path) -> tuple[Path, str]:
    from mineria import estilo

    estilo.aplicar()
    lunes = semana_representativa(resultados["r24"] if "r24" in resultados else next(iter(resultados.values())))
    figura, ejes = estilo.plt.subplots(len(resultados), 1, figsize=(10.5, 3.3 * len(resultados)), squeeze=False,
                                       sharex=True)
    for eje, (clave, r) in zip(ejes.ravel(), resultados.items()):
        prueba = r["indices"]["prueba"]
        momentos = pd.DatetimeIndex(r["fechas"].iloc[prueba])
        m = (momentos >= lunes) & (momentos < lunes + pd.Timedelta(days=7))
        real = r["y"].iloc[prueba].to_numpy()[m]
        base = np.asarray(r["evaluacion"]["predicciones"]["baseline"])[m]
        pred = np.asarray(r["evaluacion"]["predicciones"][r["rol_final"]])[m]
        eje.plot(momentos[m], real, color=estilo.TINTA, linewidth=1.6, label="real")
        eje.plot(momentos[m], base, color=estilo.TINTA_SECUNDARIA, linewidth=1.1, linestyle="--",
                 label=f"baseline (lag_168) · MAPE {100 * np.mean(np.abs(base - real) / real):.2f} %")
        eje.plot(momentos[m], pred, color=estilo.CATEGORICA[1], linewidth=1.4,
                 label=f"{r['candidatos'][r['rol_final']].etiqueta} · MAPE {100 * np.mean(np.abs(pred - real) / real):.2f} %")
        eje.set(ylabel="demanda (MWh)", title=f"{clave}")
        eje.legend(fontsize=7.5, loc="upper left", ncols=3)
    figura.suptitle(f"Semana de prueba del {lunes:%Y-%m-%d} (MAPE semanal mediano del modelo final)", fontsize=10)
    figura.autofmt_xdate()
    figura.tight_layout()
    return estilo.guardar(figura, ruta), str(lunes.date())


def figura_mape(resultados: dict[str, dict[str, Any]], ruta: Path) -> Path:
    from mineria import estilo

    estilo.aplicar()
    figura, ejes = estilo.plt.subplots(1, len(resultados), figsize=(5.4 * len(resultados), 3.8), squeeze=False)
    for eje, (clave, r) in zip(ejes.ravel(), resultados.items()):
        metricas = r["evaluacion"]["metricas"]
        roles = list(final.ROLES_REGRESION)
        valores = [metricas[rol]["mape_pct"] for rol in roles]
        etiquetas = [f"{rol}\n{r['candidatos'][rol].nombre}" for rol in roles]
        colores = [estilo.TINTA_SECUNDARIA if rol == "baseline"
                   else (estilo.CATEGORICA[1] if rol == r["rol_final"] else estilo.CATEGORICA[0]) for rol in roles]
        barras = eje.bar(etiquetas, valores, color=colores)
        base = metricas["baseline"]["mape_pct"]
        eje.axhline(base, color=estilo.TINTA_SECUNDARIA, linewidth=0.8, linestyle=":")
        eje.axhline(base * (1 - final.CRITERIO_MEJORA_PCT / 100), color=estilo.CATEGORICA[2], linewidth=0.9,
                    linestyle="--", label=f"criterio: -{final.CRITERIO_MEJORA_PCT:.0f} % del baseline")
        for barra, rol in zip(barras, roles):
            mejora = metricas[rol]["mejora_mape_pct"]
            texto = f"{barra.get_height():.2f} %" + ("" if mejora is None else f"\n{mejora:+.1f} %")
            eje.annotate(texto, (barra.get_x() + barra.get_width() / 2, barra.get_height()), ha="center",
                         va="bottom", fontsize=7.5, xytext=(0, 2), textcoords="offset points")
        eje.set(ylabel="MAPE en prueba (%)", title=f"{clave} · naranja: modelo final", ylim=(0, max(valores) * 1.3))
        eje.tick_params(axis="x", labelsize=7.5)
        eje.legend(fontsize=7.5, loc="upper right")
    figura.suptitle("MAPE en el año de prueba frente al baseline estacional", fontsize=10)
    figura.tight_layout()
    return estilo.guardar(figura, ruta)


def figura_residuales(r: dict[str, Any], ruta: Path) -> Path:
    from mineria import estilo

    estilo.aplicar()
    residuales = r["residuales"]
    figura, ejes = estilo.plt.subplots(2, 3, figsize=(12.5, 6.2), gridspec_kw={"width_ratios": [2.2, 1.4, 1.1]})
    for columna, grupo in enumerate(("hora", "dia_semana", "tipo_dia")):
        filas = residuales[grupo]
        nombres = list(filas)
        x = np.arange(len(nombres))
        mape_modelo = [filas[n]["mape_pct"] for n in nombres]
        mape_base = [filas[n]["mape_baseline_pct"] for n in nombres]
        sesgo = [filas[n]["sesgo_mwh"] for n in nombres]
        arriba, abajo = ejes[0, columna], ejes[1, columna]
        ancho = 0.4
        arriba.bar(x - ancho / 2, mape_base, ancho, color=estilo.TINTA_SECUNDARIA, alpha=0.55, label="baseline")
        arriba.bar(x + ancho / 2, mape_modelo, ancho, color=estilo.CATEGORICA[1], label="modelo final")
        abajo.bar(x, sesgo, color=[estilo.CATEGORICA[0] if s >= 0 else estilo.CATEGORICA[1] for s in sesgo])
        abajo.axhline(0, color=estilo.TINTA_SECUNDARIA, linewidth=0.7)
        etiquetas = [n if grupo != "tipo_dia" else f"{n}\n(n={filas[n]['n']})" for n in nombres]
        for eje in (arriba, abajo):
            eje.set_xticks(x)
            eje.set_xticklabels(etiquetas, fontsize=7, rotation=0 if grupo == "hora" else 30)
        arriba.set(ylabel="MAPE (%)" if columna == 0 else None, title=grupo.replace("_", " "))
        abajo.set(ylabel="sesgo medio (MWh)" if columna == 0 else None)
        if columna == 0:
            arriba.legend(fontsize=7.5)
    figura.suptitle(f"Residuales en prueba del modelo final · {r['clave']} · "
                    f"{r['candidatos'][r['rol_final']].etiqueta} (sesgo = prediccion - real)", fontsize=10)
    figura.tight_layout()
    return estilo.guardar(figura, ruta)


# --------------------------------------------------------------------------
# Documento
# --------------------------------------------------------------------------


def _pct(valor: float | None, decimales: int = 2) -> str:
    return "—" if valor is None else f"{valor:.{decimales}f} %".replace(".", ",")


def _num(valor: float | None, decimales: int = 0) -> str:
    if valor is None:
        return "—"
    texto = f"{valor:,.{decimales}f}".replace(",", " ").replace(".", ",")
    return texto


def _tamano(bytes_: int) -> str:
    return f"{_num(bytes_ / 1e6, 1)} MB" if bytes_ >= 1e6 else f"{_num(bytes_ / 1e3, 1)} KB"


def _signo(valor: float, decimales: int = 1) -> str:
    return f"{valor:+.{decimales}f}".replace(".", ",")


def documento(contenido: dict[str, Any]) -> str:
    """docs/modelado/seleccion.md a partir del JSON: todas las cifras salen de el."""
    lineas = [
        "# Selección del modelo final",
        "",
        "> Documento generado por `scripts/evaluar_final.py` a partir de "
        "`docs/resultados/modelado_seleccion.json`. Ninguna cifra está escrita a mano.",
        "",
        "Este es el único paso que usa el bloque de prueba "
        f"({contenido['bloque_prueba'][0][:10]} → {contenido['bloque_prueba'][1][:10]}). Los candidatos se "
        "reentrenaron con entrenamiento + validación y los hiperparámetros ya elegidos, sin volver a buscar, y se "
        "evaluaron una sola vez en prueba.",
        "",
        "## Regla de elección",
        "",
        "La elección se hizo **con validación, antes de mirar la prueba**: "
        f"{contenido['regla_eleccion']}. La prueba solo dice si el criterio de éxito se sostiene fuera de la "
        "muestra con la que se eligió.",
        "",
        f"Criterio de éxito del proyecto: reducir al menos un {final.CRITERIO_MEJORA_PCT:.0f} % el MAPE del "
        "baseline estacional (`lag_168`) y mantener la ventaja en las horas pico "
        f"({', '.join(f'{h}:00' for h in contenido['horas_pico'])}).",
        "",
    ]
    for clave, t in contenido["tablas"].items():
        f = t["final"]
        lineas += [f"## {clave}", "", "### Tabla de selección", "",
                   "| rol | modelo | MAPE val. | MAPE prueba | MAPE pico prueba | mejora prueba | sesgo prueba | "
                   "interpretabilidad | ajuste | predicción / 1000 filas | tamaño |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
        for fila in t["tabla_seleccion"]:
            if fila["rol"] == "clasificador":
                continue
            p, v, e = fila["rendimiento"]["prueba"], fila["rendimiento"]["validacion"], fila["eficiencia"]
            marca = " **(final)**" if fila["es_final"] else ""
            lineas.append(
                f"| {fila['rol']} | {fila['modelo']}{marca} | {_pct(v['mape_pct'])} | {_pct(p['mape_pct'])} | "
                f"{_pct(p['mape_pico_pct'])} | {'—' if p['mejora_mape_pct'] is None else _signo(p['mejora_mape_pct']) + ' %'} | "
                f"{_num(p['sesgo_medio_mwh'])} MWh | {fila['interpretabilidad']['calificacion']} | "
                f"{_num(e['ajuste_s'], 1)} s | {_num(e['prediccion_ms_por_1000_filas'], 2)} ms | "
                f"{_tamano(e['tamano_bytes'])} |"
            )
        lineas += ["", "MAE, RMSE y error sobre el máximo diario de cada candidato están en el JSON "
                   "(`metricas_prueba`).", ""]

        eleccion = t["eleccion"]
        lineas += ["### Por qué este modelo", ""]
        elegido = eleccion["elegido"]
        lineas.append(f"**Modelo final en {clave}: {f['etiqueta']}** (rol `{elegido}`).")
        lineas.append("")
        for rol, ev in eleccion["evaluados"].items():
            cv = ev["criterio_validacion"]
            estado = "cumple" if cv["cumple"] else "no cumple"
            motivo = []
            if not cv["mejora_suficiente"]:
                motivo.append(f"mejora de {_signo(cv['mejora_mape_pct'])} %, por debajo del "
                              f"{final.CRITERIO_MEJORA_PCT:.0f} %")
            if not cv["ventaja_en_pico"]:
                motivo.append(f"MAPE pico de {_pct(cv['mape_pico_pct'])} frente a {_pct(cv['mape_pico_baseline_pct'])} "
                              "del baseline")
            lineas.append(
                f"- `{rol}` ({t['candidatos'][rol]}): MAPE de validación {_pct(ev['mape_validacion_pct'])}, "
                f"{estado} el criterio en validación" + (f" ({'; '.join(motivo)})" if motivo else "")
                + f"; interpretabilidad {ev['interpretabilidad']}."
            )
        if eleccion["nota"]:
            lineas += ["", f"Nota: {eleccion['nota']}."]
        otros = [fila for fila in t["tabla_seleccion"] if fila["rol"] in ("lineal", "ensemble") and not fila["es_final"]]
        fila_final = next(fila for fila in t["tabla_seleccion"] if fila["es_final"])
        for otro in otros:
            ef, eo = fila_final["eficiencia"], otro["eficiencia"]
            lineas += ["", f"Frente a {otro['modelo']}: el final tiene "
                       f"{_pct(fila_final['rendimiento']['validacion']['mape_pct'])} de MAPE de validación contra "
                       f"{_pct(otro['rendimiento']['validacion']['mape_pct'])}; a cambio, su interpretabilidad es "
                       f"{fila_final['interpretabilidad']['calificacion']} (la del otro, "
                       f"{otro['interpretabilidad']['calificacion']}) y predice en "
                       f"{_num(ef['prediccion_ms_por_1000_filas'], 2)} ms por 1000 filas frente a "
                       f"{_num(eo['prediccion_ms_por_1000_filas'], 2)} ms."]
        if fila_final["rol"] != "baseline":
            lineas += ["", "El baseline estacional se queda como referencia: no tiene parámetros y es la cota que "
                       "cualquier modelo tiene que superar."]

        criterio = f["criterio_exito_prueba"]
        lineas += ["", "### Criterio de éxito en prueba", ""]
        if criterio is None:
            lineas.append("El modelo final es el baseline: el criterio no aplica. **Hallazgo:** ningún candidato "
                          "superó al baseline estacional en validación.")
        else:
            lineas.append(
                f"- Mejora de MAPE frente al baseline: **{_signo(criterio['mejora_mape_pct'])} %** "
                f"({'cumple' if criterio['mejora_suficiente'] else 'no cumple'} el mínimo del "
                f"{final.CRITERIO_MEJORA_PCT:.0f} %)."
            )
            lineas.append(
                f"- MAPE en horas pico: {_pct(criterio['mape_pico_pct'])} frente a "
                f"{_pct(criterio['mape_pico_baseline_pct'])} del baseline "
                f"({'mantiene' if criterio['ventaja_en_pico'] else 'pierde'} la ventaja)."
            )
            if criterio["cumple"]:
                lineas.append(f"- **Cumple el criterio de éxito en prueba.**")
            else:
                lineas.append(f"- **Hallazgo: no cumple el criterio de éxito en prueba**, aunque lo cumplía en "
                              "validación. Se reporta tal cual: la elección no se rehízo mirando la prueba.")
        otros_prueba = {rol: c for rol, c in t["criterio_exito_prueba"].items() if rol != f["rol"]}
        for rol, c in otros_prueba.items():
            lineas.append(f"- Como contraste, `{rol}` ({t['candidatos'][rol]}) mejora un "
                          f"{_signo(c['mejora_mape_pct'])} % en prueba, con {_pct(c['mape_pico_pct'])} en horas pico "
                          f"({'mantiene' if c['ventaja_en_pico'] else 'no mantiene'} la ventaja).")
        if not any(c["cumple"] for c in t["criterio_exito_prueba"].values()):
            lineas.append("- **Ningún candidato cumple el criterio de éxito en prueba en esta tabla.**")

        extra = t["extrapolacion_prueba"]
        base_extra = extra["baseline"]
        lineas += ["", "### Por encima de lo visto", "",
                   f"{base_extra['reales_sobre_maximo']} de las {base_extra['n']} horas de prueba "
                   f"({_pct(100 * base_extra['proporcion_reales_sobre_maximo'], 1)}) superan el máximo de "
                   f"entrenamiento + validación ({_num(base_extra['maximo_entrenamiento_mwh'])} MWh). En esas horas:", ""]
        for rol, e in extra.items():
            lineas.append(f"- `{rol}` ({t['candidatos'][rol]}): {e['predicciones_sobre_maximo']} predicciones por "
                          f"encima del máximo; sesgo medio {_num(e['sesgo_medio_reales_sobre_maximo_mwh'])} MWh "
                          f"(en el resto de horas, {_num(e['sesgo_medio_resto_mwh'])} MWh).")
        forma_ensemble = next(fila["forma"] for fila in t["tabla_seleccion"] if fila["rol"] == "ensemble")
        validacion_extra = contenido.get("validacion_sobre_maximo", {}).get(clave)
        lineas += ["", "Un árbol predice con valores que vio y no pasa de ese máximo; la demanda de prueba sí lo pasa."]
        if forma_ensemble == "nivel":
            lineas[-1] += (" El ensemble elegido trabaja sobre el nivel: en validación"
                           + ("" if validacion_extra is None else f" solo {validacion_extra} horas superaban el máximo de "
                              "entrenamiento y")
                           + " la forma diferenciada (`y - lag_168`), la que permitiría pasar ese techo, no ganó "
                           "(`docs/resultados/modelado_ensembles.json`).")

        res = t["residuales_final"]
        peor_hora = max(res["hora"].items(), key=lambda par: par[1]["mape_pct"])
        mejor_hora = min(res["hora"].items(), key=lambda par: par[1]["mape_pct"])
        lineas += ["", "### Residuales del modelo final en prueba", "",
                   f"Figura: `{t['figuras']['residuales']}`. Residual = predicción − real.", "",
                   f"- Por hora: el MAPE va de {_pct(mejor_hora[1]['mape_pct'])} ({mejor_hora[0]}:00) a "
                   f"{_pct(peor_hora[1]['mape_pct'])} ({peor_hora[0]}:00).",
                   "- Por tipo de día:"]
        for tipo, valores in res["tipo_dia"].items():
            lineas.append(f"  - {tipo.replace('_', ' ')} (n = {valores['n']}): MAPE {_pct(valores['mape_pct'])} "
                          f"frente a {_pct(valores['mape_baseline_pct'])} del baseline; sesgo "
                          f"{_num(valores['sesgo_mwh'])} MWh.")
        peor_dia = max(res["dia_semana"].items(), key=lambda par: par[1]["mape_pct"])
        lineas.append(f"- Por día de la semana, el peor es el {peor_dia[0]} (MAPE {_pct(peor_dia[1]['mape_pct'])}, "
                      f"sesgo {_num(peor_dia[1]['sesgo_mwh'])} MWh).")

        c = t["clasificador_prueba"]
        lineas += ["", "### Alerta de pico (`alerta_pico`) en prueba", "",
                   f"{t['candidatos']['clasificador']}: PR-AUC {_num(c['pr_auc'], 3)} (prevalencia "
                   f"{_pct(100 * c['prevalencia'], 1)}), ROC-AUC {_num(c['roc_auc'], 3)}, F1 {_num(c['f1'], 3)}, "
                   f"precisión {_num(c['precision'], 3)}, recall {_num(c['recall'], 3)}. Brier de la probabilidad "
                   f"calibrada: {_num(c['brier_probabilidad'], 4)}.", ""]

    costo = contenido.get("costo_rezago") or {}
    if costo:
        cf = costo["final"]
        lineas += ["## Lo que cuesta el rezago de publicación (r24 → r120)", "",
                   "r120 es el escenario operativo: la demanda real se publica con días de rezago (README, "
                   "«Limitación importante para el modelado»), y r24 supone que se conoce la de ayer.", "",
                   f"- Modelo final: MAPE de {_pct(cf['mape_r24_pct'])} en r24 ({cf['modelo_r24']}) a "
                   f"{_pct(cf['mape_r120_pct'])} en r120 ({cf['modelo_r120']}): "
                   f"{_signo(cf['aumento_puntos'], 2)} puntos, un {_signo(cf['aumento_relativo_pct'])} %."]
        for rol in final.ROLES_REGRESION:
            c = costo[rol]
            lineas.append(f"- `{rol}`: {_pct(c['mape_r24_pct'])} → {_pct(c['mape_r120_pct'])} "
                          f"({_signo(c['aumento_puntos'], 2)} puntos).")
        lineas.append("")

    lineas += ["## Limitaciones", "",
               "- La probabilidad de alerta es una sigmoide de Platt ajustada con el clasificador entrenado solo "
               "con entrenamiento y aplicada al modelo final: es aproximada. La decisión de alerta usa el puntaje "
               "del modelo y su umbral natural, 0.",
               "- Los tiempos dependen de la máquina (ver `entorno` en el JSON) y solo se comparan entre sí.",
               "- La prueba se evaluó una vez; un nuevo ajuste a partir de estos resultados ya no tendría una "
               "estimación limpia del error.", ""]
    return "\n".join(lineas)


# --------------------------------------------------------------------------
# Principal
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description="Evaluacion final en el bloque de prueba")
    analizador.add_argument("--tablas", nargs="+", choices=TABLAS, default=list(TABLAS))
    analizador.add_argument("--datos", type=Path, default=DIR_SALIDA)
    analizador.add_argument("--resultados", type=Path, default=RESULTADOS)
    analizador.add_argument("--modelos", type=Path, default=MODELOS)
    analizador.add_argument("--figuras", type=Path, default=FIGURAS)
    analizador.add_argument("--documento", type=Path, default=DOCUMENTO)
    analizador.add_argument("--forzar", action="store_true", help="rehace una evaluacion ya escrita")
    args = analizador.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    salida = args.resultados / SALIDA
    if salida.exists() and not args.forzar:
        raise SystemExit(f"{_relativa(salida)} ya existe: la prueba se evalua una vez. Usa --forzar para rehacerla.")

    previos = {}
    for familia, archivo in ENTRADAS.items():
        ruta = args.resultados / archivo
        if ruta.exists():
            previos[familia] = json.loads(ruta.read_text(encoding="utf-8"))
        elif familia not in OPCIONALES:
            raise SystemExit(f"Falta {_relativa(ruta)}: corre antes su paso.")

    # 1. Calculo completo en memoria (AlertaFuga detiene todo antes de escribir).
    claves = [clave_version(t) for t in args.tablas]
    resultados = {clave: evaluar_tabla(clave, previos, args.datos) for clave in claves}

    # 2. Escritura.
    archivos_por_tabla = {}
    for clave, r in resultados.items():
        archivos = {}
        for rol, modelo in r["modelos"].items():
            nombre = f"final_{clave}.joblib" if rol == r["rol_final"] else f"prueba_{rol}_{clave}.joblib"
            ruta = args.modelos / nombre
            archivos[rol] = {**guardar_modelo(modelo, ruta), "ruta": _relativa(ruta)}
        archivos["final"] = archivos[r["rol_final"]]
        ruta = args.modelos / f"prueba_calibrador_{clave}.joblib"
        archivos["calibrador"] = {**guardar_modelo(r["calibrador"], ruta), "ruta": _relativa(ruta)}
        archivos_por_tabla[clave] = archivos

    figuras_tabla = {clave: {"residuales": _relativa(figura_residuales(r, args.figuras / f"residuales_{clave}.png"))}
                     for clave, r in resultados.items()}
    ruta_semana, lunes = figura_semana(resultados, args.figuras / "semana_prueba.png")
    ruta_mape = figura_mape(resultados, args.figuras / "mape_prueba.png")

    ruta_predicciones = args.resultados / PREDICCIONES
    ruta_probabilidades = args.resultados / PROBABILIDADES
    _escribir_parquet(ruta_predicciones, predicciones_largas(resultados))
    _escribir_parquet(ruta_probabilidades, probabilidades(resultados))

    tablas = {}
    for clave, r in resultados.items():
        tablas[clave] = {**entrada_json(r, archivos_por_tabla[clave]), "figuras": figuras_tabla[clave]}
        tablas[clave]["calibracion_alerta"]["modelo"] = archivos_por_tabla[clave]["calibrador"]
    prueba = next(iter(resultados.values()))
    fechas_prueba = prueba["fechas"].iloc[prueba["indices"]["prueba"]]
    contenido = {
        "version_formato": VERSION_FORMATO,
        "momento": ahora(),
        "entorno": entorno(),
        "bloque_ajuste": "entrenamiento + validacion",
        "bloque_evaluacion": "prueba",
        "bloque_prueba": [str(fechas_prueba.iloc[0]), str(fechas_prueba.iloc[-1])],
        "regla_eleccion": next(iter(tablas.values()))["eleccion"]["regla"],
        "criterio_exito": {"mejora_mape_minima_pct": final.CRITERIO_MEJORA_PCT,
                           "ventaja_en_pico": "MAPE en horas pico menor que el del baseline estacional"},
        "horas_pico": list(final.HORAS_PICO),
        "validacion_sobre_maximo": {clave: _validacion_sobre_maximo(previos, clave) for clave in tablas},
        "tablas": tablas,
        "costo_rezago": final.costo_rezago(tablas),
        "archivos": {
            "predicciones": _relativa(ruta_predicciones),
            "probabilidades": _relativa(ruta_probabilidades),
            "figura_semana": _relativa(ruta_semana),
            "semana_figura_desde": lunes,
            "figura_mape": _relativa(ruta_mape),
            "documento": _relativa(args.documento),
        },
    }
    _escribir_json(salida, contenido)
    _escribir_texto(args.documento, documento(contenido))

    for clave, t in tablas.items():
        print(f"\n=== PRUEBA · {clave} === {contenido['bloque_prueba'][0][:16]} .. {contenido['bloque_prueba'][1][:16]}")
        for fila in t["tabla_seleccion"]:
            p = fila["rendimiento"]["prueba"]
            marca = " <- final" if fila["es_final"] else ""
            if fila["rol"] == "clasificador":
                print(f"  {fila['rol']:<12} {fila['modelo']:<32} PR-AUC {p['pr_auc']:.3f} · F1 {p['f1'] or 0:.3f} · "
                      f"Brier {p['brier_probabilidad']:.4f}")
                continue
            mejora = "" if p["mejora_mape_pct"] is None else f" · mejora {p['mejora_mape_pct']:+.1f} %"
            print(f"  {fila['rol']:<12} {fila['modelo']:<32} MAPE {p['mape_pct']:.2f} % · pico {p['mape_pico_pct']:.2f} % "
                  f"· sesgo {p['sesgo_medio_mwh']:+.0f} MWh{mejora}{marca}")
    print(f"\n  -> {_relativa(salida)}, {_relativa(args.documento)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
