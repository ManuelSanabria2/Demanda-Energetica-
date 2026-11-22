"""Exporta a app/datos/ lo que la interfaz necesita y no esta en git.

La interfaz (app/) no entrena ni descarga: solo lee artefactos. La mayoria ya
estan versionados (docs/resultados/*.json, docs/figuras/, infra/costos/), pero
la serie diaria y los perfiles salen del panel de modelado, que vive fuera de
git. Este script los resume en tablas pequenas y las deja en app/datos/ con un
manifiesto (fecha y SHA-256 de cada archivo):

    python scripts/exportar_interfaz.py

| archivo                       | contenido                                                    |
|-------------------------------|--------------------------------------------------------------|
| serie_diaria.parquet          | energia diaria (MWh), temperatura media y tipo de dia        |
| perfil_horario.parquet        | demanda media por dia de la semana y hora (168 filas)        |
| autocorrelacion.parquet       | ACF hasta 336 h y PACF hasta 48 h de la serie horaria        |
| centroides_grupos.parquet     | forma media de cada grupo de dias del K-means, hora a hora   |
| predicciones_prueba.parquet   | por hora de prueba y version: real, baseline, cada candidato |
|                               | y el final, mas las banderas del calendario                  |
| probabilidades_alerta.parquet | probabilidad y etiqueta de alerta_pico del clasificador      |
| variables_prueba.parquet      | variables (X) y real de cada hora de prueba, para el modelo  |
| modelo_final_r24.joblib, ...  | modelo final de cada version, para la pagina Probar el modelo |
| manifiesto_interfaz.json      | momento, origen y SHA-256 de cada archivo; lo que falto      |

Las predicciones salen de docs/resultados/predicciones_prueba.parquet, que
escribe scripts/evaluar_final.py. Si no esta, se calculan cargando los modelos
guardados en models/ que anota modelado_seleccion.json (comprobando su SHA-256 y
el de la tabla de modelado): no se reentrena nada.

Los modelos finales se leen de models/ comprobando el SHA-256 anotado en
modelado_seleccion.json. Los que pesan mas de LIMITE_MB se guardan comprimidos
(joblib, compress=3); si aun asi lo superan, se exporta en su lugar la variante
HistGradientBoosting (nivel) de esa version, y el manifiesto lo dice para que la
pagina lo muestre. Asi ningun archivo de app/datos/ pasa de LIMITE_MB en git.

Si falta una entrada opcional (las etiquetas de la mineria o las predicciones
del modelado), se anota en `faltantes` del manifiesto y se sigue: la pagina que
la usa muestra el aviso de pendiente. Sin panel no hay nada que exportar y el
script falla con el comando que lo genera.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from mineria import eda  # noqa: E402
from mineria.perfiles import NOMBRES_DIA, forma_y_nivel, perfiles_diarios  # noqa: E402

VERSION_FORMATO = 1
DIAS_PARCIALES = 5
TIPOS_DIA = ("laboral", "sabado", "domingo", "festivo", "semana_santa")

ROLES_REGRESION = ("baseline", "lineal", "ensemble")
BANDERAS = ("es_festivo", "es_vispera_festivo", "es_puente", "es_semana_santa", "es_ultima_semana_diciembre")
GENERA_MODELADO = "python scripts/evaluar_final.py"
LIMITE_MB = 25


# --------------------------------------------------------------------------
# Escritura atomica y huellas
# --------------------------------------------------------------------------


def sha256(ruta: Path) -> str:
    huella = hashlib.sha256()
    with ruta.open("rb") as origen:
        for bloque in iter(lambda: origen.read(1 << 20), b""):
            huella.update(bloque)
    return huella.hexdigest()


def _escribir_parquet(marco: pd.DataFrame, ruta: Path) -> None:
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    marco.to_parquet(temporal, index=False)
    temporal.replace(ruta)


def _escribir_json(datos: dict, ruta: Path) -> None:
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporal.replace(ruta)


def _relativa(ruta: Path) -> str:
    try:
        return ruta.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return ruta.as_posix()


# --------------------------------------------------------------------------
# Tablas: cada una devuelve (marco, registro)
# --------------------------------------------------------------------------


def _serie_y_panel_local(panel: pd.DataFrame, dias_parciales: int) -> tuple[pd.Series, pd.DataFrame]:
    """Serie horaria en MWh sin los dias parciales, y el panel con el mismo indice local."""
    completa = eda.serie_mwh(panel)
    serie = eda.recortar_parcial(completa, dias_parciales)
    panel_local = panel.set_index(completa.index).reindex(serie.index)
    return serie, panel_local


def tipo_dia(indice: pd.DatetimeIndex, festivo: pd.Series, semana_santa: pd.Series) -> pd.Series:
    """laboral / sabado / domingo / festivo / semana_santa.

    Igual que `mineria.eda.tipo_dia` (el festivo manda sobre el dia de la
    semana), y la Semana Santa manda sobre todo: es una semana entera con un
    perfil propio, festivos del jueves y el viernes incluidos.
    """
    base = eda.tipo_dia(indice, festivo)
    return base.mask(semana_santa.to_numpy(bool), "semana_santa")


def serie_diaria(panel: pd.DataFrame, dias_parciales: int = DIAS_PARCIALES) -> tuple[pd.DataFrame, dict[str, Any]]:
    serie, local = _serie_y_panel_local(panel, dias_parciales)
    energia = serie.resample("D").sum(min_count=24)
    completos = energia.notna()
    dias = energia.index[completos]

    def por_dia(columna: str, como: str) -> pd.Series:
        if columna not in local:
            return pd.Series(np.nan, index=dias)
        datos = local[columna]
        if como == "max":
            datos = datos.fillna(False).astype(bool)
        return getattr(datos.resample("D"), como)().reindex(dias)

    festivo = por_dia("es_festivo", "max")
    santa = por_dia("es_semana_santa", "max")
    nombres = (
        local["nombre_festivo"].resample("D").first().reindex(dias)
        if "nombre_festivo" in local
        else pd.Series(None, index=dias, dtype=object)
    )
    marco = pd.DataFrame(
        {
            "fecha": dias,
            "demanda_mwh": energia[completos].to_numpy(),
            "demanda_media_mwh": serie.resample("D").mean().reindex(dias).to_numpy(),
            "temperatura_media_c": por_dia("temperatura_c", "mean").to_numpy(),
            "tipo_dia": tipo_dia(dias, festivo, santa).to_numpy(),
            "es_festivo": festivo.to_numpy(bool),
            "nombre_festivo": nombres.to_numpy(),
            "horas_atipicas": por_dia("atipico", "sum").fillna(0).astype(int).to_numpy(),
        }
    )
    registro = {
        "operacion": "serie_diaria",
        "criterio": (
            "energia diaria = suma de las 24 h en MWh; solo dias con las 24 h; se descartan los "
            f"ultimos {dias_parciales} dias por publicacion parcial; tipo de dia: semana_santa > "
            "festivo > domingo > sabado > laboral; horas_atipicas = marca causal `atipico`"
        ),
        "dias": int(len(marco)),
        "dias_incompletos": int((~completos).sum()),
        "rango": [str(dias.min().date()), str(dias.max().date())],
        "tipos": {t: int((marco["tipo_dia"] == t).sum()) for t in TIPOS_DIA},
    }
    return marco, registro


def perfil_horario(panel: pd.DataFrame, dias_parciales: int = DIAS_PARCIALES) -> tuple[pd.DataFrame, dict[str, Any]]:
    serie, _ = _serie_y_panel_local(panel, dias_parciales)
    datos = pd.DataFrame({"dia_semana": serie.index.dayofweek, "hora": serie.index.hour, "mwh": serie.to_numpy()})
    marco = (
        datos.groupby(["dia_semana", "hora"])["mwh"]
        .agg(demanda_media_mwh="mean", demanda_mediana_mwh="median", n="count")
        .reset_index()
    )
    marco.insert(1, "nombre_dia", marco["dia_semana"].map(dict(enumerate(NOMBRES_DIA))))
    registro = {
        "operacion": "perfil_horario",
        "criterio": "media y mediana de la demanda horaria (MWh) por dia de la semana (0 = lunes) y hora local",
        "filas": int(len(marco)),
        "horas": int(datos["mwh"].notna().sum()),
    }
    return marco, registro


def autocorrelacion(panel: pd.DataFrame, dias_parciales: int = DIAS_PARCIALES) -> tuple[pd.DataFrame, dict[str, Any]]:
    """La misma ACF/PACF que scripts/eda.py: eda.json solo guarda cinco rezagos."""
    serie, _ = _serie_y_panel_local(panel, dias_parciales)
    acf, pacf = eda.autocorrelacion(serie)
    marco = pd.DataFrame({"rezago_h": np.arange(len(acf)), "acf": acf.to_numpy()})
    marco["pacf"] = pacf.reindex(marco.index).to_numpy()
    registro = {
        "operacion": "autocorrelacion",
        "criterio": "mineria.eda.autocorrelacion sobre la serie horaria sin dias parciales: ACF hasta 336 h, PACF hasta 48 h",
        "rezagos": int(len(marco) - 1),
        "banda_95": float(1.96 / np.sqrt(serie.notna().sum())),
    }
    return marco, registro


def centroides_grupos(
    panel: pd.DataFrame, etiquetas: pd.DataFrame, dias_parciales: int = DIAS_PARCIALES
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Forma media (media del dia = 1) de cada grupo, con su banda p10-p90.

    Los grupos son los de docs/resultados/mineria_etiquetas_dia.csv: aqui no se
    vuelve a agrupar, solo se promedia la forma de los dias de cada etiqueta.
    """
    perfiles, _ = perfiles_diarios(panel, dias_parciales=dias_parciales)
    forma, _ = forma_y_nivel(perfiles)
    forma.index = pd.to_datetime(forma.index)
    grupo = etiquetas.assign(fecha=pd.to_datetime(etiquetas["fecha"])).set_index("fecha")["grupo"]
    comunes = forma.index.intersection(grupo.index)
    largo = (
        forma.loc[comunes]
        .assign(grupo=grupo.loc[comunes])
        .melt(id_vars="grupo", var_name="hora", value_name="forma")
    )
    largo["hora"] = largo["hora"].str[1:].astype(int)
    marco = (
        largo.groupby(["grupo", "hora"])["forma"]
        .agg(forma_media="mean", p10=lambda x: x.quantile(0.10), p90=lambda x: x.quantile(0.90), dias="count")
        .reset_index()
    )
    registro = {
        "operacion": "centroides_grupos",
        "criterio": "media y percentiles 10/90 de la forma diaria (dia / su media) de los dias de cada grupo",
        "dias_etiquetados": int(len(grupo)),
        "dias_con_perfil": int(len(comunes)),
        "dias_sin_perfil": int(len(grupo.index.difference(forma.index))),
        "grupos": sorted(marco["grupo"].unique().tolist()),
    }
    return marco, registro


def predicciones_desde_modelos(
    seleccion: dict[str, Any], procesado: Path, modelos: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predicciones de prueba con los modelos ya guardados, en el formato largo de evaluar_final.py.

    Solo carga y predice: cada modelo se lee con su SHA-256 (el de
    modelado_seleccion.json) y la tabla de modelado con el suyo (el de
    registro_variables.json). El bloque de prueba es el registrado.
    """
    from modelado.datos import cargar_tabla
    from modelado.tiempos import cargar_modelo

    desde, hasta = (pd.Timestamp(m) for m in seleccion["bloque_prueba"])
    partes, cargados = [], {}
    for clave, tabla in seleccion["tablas"].items():
        datos, _ = cargar_tabla(clave, procesado)
        fechas = pd.to_datetime(datos.fechas)
        filas = ((fechas >= desde) & (fechas <= hasta)).to_numpy()
        final = (tabla.get("final") or {}).get("rol")
        for fila in tabla.get("tabla_seleccion", []):
            if fila.get("rol") not in ROLES_REGRESION:
                continue
            guardado = fila["modelo_guardado"]
            modelo = cargar_modelo(modelos / Path(guardado["ruta"]).name, guardado["sha256"])
            partes.append(pd.DataFrame({
                "fecha_hora": fechas[filas].to_numpy(),
                "tabla": clave,
                "rol": fila["rol"],
                "modelo": fila.get("modelo"),
                "es_final": fila["rol"] == final,
                "real_mwh": datos.y[filas].to_numpy(dtype="float64"),
                "prediccion_mwh": np.asarray(modelo.predict(datos.X[filas].astype("float64")), dtype="float64"),
            }))
            cargados[f"{clave}/{fila['rol']}"] = guardado["sha256"]
    registro = {"operacion": "predecir_con_modelos_guardados",
                "criterio": "modelos de models/ con el SHA-256 de modelado_seleccion.json sobre el bloque de prueba; sin reentrenar",
                "modelos": cargados}
    return pd.concat(partes, ignore_index=True), registro


def predicciones_ancho(largo: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Una fila por hora de prueba y version: real, una columna por rol, la del final y el calendario."""
    largo = largo.assign(fecha_hora=pd.to_datetime(largo["fecha_hora"]))
    ancho = largo.pivot_table(index=["fecha_hora", "tabla"], columns="rol", values="prediccion_mwh", aggfunc="first")
    ancho.columns = [f"{rol}_mwh" for rol in ancho.columns]
    real = largo.groupby(["fecha_hora", "tabla"])["real_mwh"].first()
    finales = largo[largo["es_final"]].set_index(["fecha_hora", "tabla"])["prediccion_mwh"].rename("final_mwh")
    marco = pd.concat([real, ancho, finales], axis=1).reset_index().sort_values(["tabla", "fecha_hora"])

    calendario = panel.assign(fecha_hora=pd.to_datetime(panel["fecha_hora"]))
    columnas = [c for c in (*BANDERAS, "nombre_festivo") if c in calendario]
    calendario = calendario[["fecha_hora", *columnas]]
    if calendario["fecha_hora"].dt.tz is None and marco["fecha_hora"].dt.tz is not None:
        calendario["fecha_hora"] = calendario["fecha_hora"].dt.tz_localize(marco["fecha_hora"].dt.tz)
    elif calendario["fecha_hora"].dt.tz is not None and marco["fecha_hora"].dt.tz is not None:
        calendario["fecha_hora"] = calendario["fecha_hora"].dt.tz_convert(marco["fecha_hora"].dt.tz)
    marco = marco.merge(calendario, on="fecha_hora", how="left")
    sin_calendario = int(marco[BANDERAS[0]].isna().sum()) if BANDERAS[0] in marco else len(marco)
    for bandera in BANDERAS:
        if bandera in marco:
            marco[bandera] = marco[bandera].fillna(False).astype(bool)
    marco = marco.reset_index(drop=True)

    modelos = largo.groupby(["tabla", "rol"])["modelo"].first()
    registro = {
        "operacion": "predicciones_ancho",
        "criterio": "pivote de las predicciones de prueba por rol; final_mwh = el rol con es_final; calendario del panel por marca de tiempo",
        "filas_por_tabla": {t: int(n) for t, n in marco.groupby("tabla").size().items()},
        "modelos": {t: {r: m for (tt, r), m in modelos.items() if tt == t} for t in sorted(largo["tabla"].unique())},
        "final": {t: str(g.loc[g["es_final"], "rol"].iloc[0]) for t, g in largo.groupby("tabla") if g["es_final"].any()},
        "horas_sin_calendario": sin_calendario,
    }
    return marco, registro


def variables_prueba(seleccion: dict[str, Any], procesado: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """X y demanda real de cada hora del bloque de prueba, para predecir y explicar una hora en la app.

    Una fila por hora y version; las variables que una version no tiene (lag_24 en r120) quedan vacias.
    """
    from modelado.datos import cargar_tabla

    desde, hasta = (pd.Timestamp(m) for m in seleccion["bloque_prueba"])
    partes, sha = [], {}
    for clave in seleccion["tablas"]:
        datos, paso = cargar_tabla(clave, procesado)
        fechas = pd.to_datetime(datos.fechas)
        filas = ((fechas >= desde) & (fechas <= hasta)).to_numpy()
        partes.append(datos.X[filas].assign(fecha_hora=fechas[filas].to_numpy(), tabla=clave,
                                            real_mwh=datos.y[filas].to_numpy(dtype="float64")))
        sha[clave] = paso["sha256"]
    marco = pd.concat(partes, ignore_index=True)
    marco = marco[["fecha_hora", "tabla", "real_mwh", *[c for c in marco.columns if c not in ("fecha_hora", "tabla", "real_mwh")]]]
    registro = {"operacion": "variables_prueba",
                "criterio": "X de la tabla de modelado (SHA-256 verificado contra registro_variables.json) en el bloque de prueba",
                "sha256_tablas": sha, "filas_por_tabla": {t: int(n) for t, n in marco.groupby("tabla").size().items()}}
    return marco, registro


def exportar_modelos(seleccion: dict[str, Any], ensembles: dict[str, Any] | None, modelos: Path, destino: Path,
                     limite_mb: float = LIMITE_MB, anterior: dict[str, Any] | None = None,
                     ) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Modelo final de cada version en destino/modelo_final_<version>.joblib, sin pasar de `limite_mb`.

    Nunca se reentrena: se copia el archivo, se vuelve a guardar comprimido o se toma la variante
    HistGradientBoosting ya entrenada (modelado_ensembles.json), siempre comprobando su SHA-256.

    Volver a comprimir un bosque no da los mismos bytes, asi que si `anterior` (el manifiesto previo)
    dice que el archivo ya exportado salio del mismo modelo de origen y sigue intacto, se conserva:
    sin esto, cada exportacion reescribiria decenas de MB en git sin que nada haya cambiado.
    """
    import joblib
    from modelado.tiempos import cargar_modelo

    archivos, faltantes = {}, []
    limite = limite_mb * 1e6
    for clave, tabla in seleccion["tablas"].items():
        nombre = f"modelo_final_{clave}.joblib"
        final = tabla.get("final") or {}
        guardado = final.get("modelo") or {}
        try:
            origen = modelos / Path(guardado["ruta"]).name
            modelo = cargar_modelo(origen, guardado["sha256"])
        except Exception as error:  # noqa: BLE001
            faltantes.append({"archivo": nombre, "falta": f"{guardado.get('ruta')} ({error})", "script": GENERA_MODELADO})
            continue
        ruta = destino / nombre
        previo = ((anterior or {}).get("archivos") or {}).get(nombre)
        if (previo and ruta.exists() and sha256(ruta) == previo.get("sha256")
                and previo.get("registro", {}).get("sha256_final") == guardado["sha256"]):
            archivos[nombre] = previo
            continue
        temporal = ruta.with_suffix(ruta.suffix + ".tmp")
        info = {"sha256_final": guardado["sha256"], "modelo": final.get("etiqueta"), "rol": final.get("rol"), "variante": False,
                "origen": f"models/{origen.name}", "sha256_origen": guardado["sha256"], "bytes_origen": origen.stat().st_size}
        if origen.stat().st_size <= limite:
            temporal.write_bytes(origen.read_bytes())
            info["guardado"] = "copia byte a byte"
        else:
            joblib.dump(modelo, temporal, compress=3)
            info["guardado"] = "joblib compress=3"
            if temporal.stat().st_size > limite:
                variante = obtener_variante_hgb(ensembles, clave)
                if variante is None:
                    temporal.unlink()
                    faltantes.append({"archivo": nombre, "falta": f"{origen.name} comprimido pasa de {limite_mb} MB y no hay "
                                      "variante HistGradientBoosting", "script": GENERA_MODELADO})
                    continue
                origen = modelos / Path(variante["ruta"]).name
                joblib.dump(cargar_modelo(origen, variante["sha256"]), temporal, compress=3)
                info.update({"modelo": f"hist_gradient (nivel), variante de {final.get('etiqueta')}", "variante": True,
                             "origen": f"models/{origen.name}", "sha256_origen": variante["sha256"],
                             "bytes_origen": origen.stat().st_size, "mape_validacion_pct": variante.get("mape_validacion_pct"),
                             "motivo_variante": f"el modelo final comprimido pasa de {limite_mb} MB"})
        temporal.replace(ruta)
        archivos[nombre] = {"sha256": sha256(ruta), "bytes": ruta.stat().st_size, "origen": info["origen"],
                            "registro": {"operacion": "exportar_modelo", "criterio":
                                         f"modelo final verificado por SHA-256; si pasa de {limite_mb} MB, comprimido y, "
                                         "si aun asi, la variante HistGradientBoosting", **info}}
    return archivos, faltantes


def obtener_variante_hgb(ensembles: dict[str, Any] | None, clave: str) -> dict[str, Any] | None:
    entrada = (((ensembles or {}).get("tablas") or {}).get(clave) or {}).get("modelos", {}).get("hist_gradient", {}).get("nivel")
    if not entrada or not entrada.get("modelo"):
        return None
    return {"ruta": entrada["modelo"]["ruta"], "sha256": entrada["modelo"]["sha256"],
            "mape_validacion_pct": (entrada.get("metricas_validacion") or {}).get("mape_pct")}


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------


def exportar(
    panel_ruta: Path, resultados: Path, destino: Path, dias_parciales: int = DIAS_PARCIALES,
    procesado: Path | None = None, modelos: Path | None = None,
) -> dict[str, Any]:
    if not panel_ruta.exists():
        raise FileNotFoundError(
            f"No existe {panel_ruta}. Ejecuta antes: python scripts/construir_panel.py --imputacion causal"
        )
    destino.mkdir(parents=True, exist_ok=True)
    procesado = procesado or panel_ruta.parent
    modelos = modelos or RAIZ / "models"
    panel = pd.read_parquet(panel_ruta)

    tablas: dict[str, tuple[pd.DataFrame, dict[str, Any], str]] = {
        "serie_diaria.parquet": (*serie_diaria(panel, dias_parciales), _relativa(panel_ruta)),
        "perfil_horario.parquet": (*perfil_horario(panel, dias_parciales), _relativa(panel_ruta)),
        "autocorrelacion.parquet": (*autocorrelacion(panel, dias_parciales), _relativa(panel_ruta)),
    }
    faltantes: list[dict[str, str]] = []

    ruta_etiquetas = resultados / "mineria_etiquetas_dia.csv"
    if ruta_etiquetas.exists():
        tablas["centroides_grupos.parquet"] = (
            *centroides_grupos(panel, pd.read_csv(ruta_etiquetas), dias_parciales), _relativa(ruta_etiquetas))
    else:
        faltantes.append({"archivo": "centroides_grupos.parquet", "falta": _relativa(ruta_etiquetas),
                          "script": "python scripts/mineria_no_supervisada.py"})

    # Predicciones de prueba: las de evaluar_final.py o, si no estan, las de los modelos guardados.
    ruta_largas = resultados / "predicciones_prueba.parquet"
    ruta_seleccion = resultados / "modelado_seleccion.json"
    largas = origen = None
    if ruta_largas.exists():
        largas, origen = pd.read_parquet(ruta_largas), _relativa(ruta_largas)
    elif ruta_seleccion.exists():
        try:
            largas, paso = predicciones_desde_modelos(json.loads(ruta_seleccion.read_text(encoding="utf-8")), procesado, modelos)
            origen = f"{_relativa(modelos)} ({paso['operacion']})"
        except Exception as error:  # noqa: BLE001 -- se anota y se sigue: la pagina muestra el aviso
            faltantes.append({"archivo": "predicciones_prueba.parquet", "falta": f"{_relativa(ruta_largas)} y los modelos guardados ({error})",
                              "script": GENERA_MODELADO})
    else:
        faltantes.append({"archivo": "predicciones_prueba.parquet", "falta": _relativa(ruta_largas), "script": GENERA_MODELADO})
    if largas is not None:
        tablas["predicciones_prueba.parquet"] = (*predicciones_ancho(largas, panel), origen)

    ruta_probabilidades = resultados / "probabilidades_alerta.parquet"
    if ruta_probabilidades.exists():
        probabilidades = pd.read_parquet(ruta_probabilidades)
        tablas["probabilidades_alerta.parquet"] = (probabilidades, {
            "operacion": "copiar", "criterio": "probabilidad calibrada y etiqueta real de alerta_pico del clasificador final, por hora de prueba",
            "filas_por_tabla": {t: int(n) for t, n in probabilidades.groupby("tabla").size().items()},
        }, _relativa(ruta_probabilidades))
    else:
        faltantes.append({"archivo": "probabilidades_alerta.parquet", "falta": _relativa(ruta_probabilidades),
                          "script": GENERA_MODELADO})

    seleccion = json.loads(ruta_seleccion.read_text(encoding="utf-8")) if ruta_seleccion.exists() else None
    if seleccion is not None:
        try:
            tablas["variables_prueba.parquet"] = (*variables_prueba(seleccion, procesado), _relativa(procesado))
        except Exception as error:  # noqa: BLE001
            faltantes.append({"archivo": "variables_prueba.parquet", "falta": f"tablas de modelado ({error})",
                              "script": "python scripts/construir_variables.py"})
    else:
        faltantes += [{"archivo": n, "falta": _relativa(ruta_seleccion), "script": GENERA_MODELADO}
                      for n in ("variables_prueba.parquet", "modelo_final_r24.joblib", "modelo_final_r120.joblib")]

    archivos: dict[str, dict[str, Any]] = {}
    if seleccion is not None:
        ruta_ensembles = resultados / "modelado_ensembles.json"
        ensembles = json.loads(ruta_ensembles.read_text(encoding="utf-8")) if ruta_ensembles.exists() else None
        ruta_manifiesto = destino / "manifiesto_interfaz.json"
        anterior = json.loads(ruta_manifiesto.read_text(encoding="utf-8")) if ruta_manifiesto.exists() else None
        de_modelos, sin_modelo = exportar_modelos(seleccion, ensembles, modelos, destino, anterior=anterior)
        archivos.update(de_modelos)
        faltantes += sin_modelo
    for nombre, (marco, registro, origen_archivo) in tablas.items():
        ruta = destino / nombre
        _escribir_parquet(marco, ruta)
        archivos[nombre] = {"sha256": sha256(ruta), "bytes": ruta.stat().st_size, "filas": int(len(marco)),
                            "columnas": list(marco.columns), "origen": origen_archivo, "registro": registro}

    manifiesto = {
        "version_formato": VERSION_FORMATO,
        "momento": dt.datetime.now().isoformat(timespec="seconds"),
        "panel": {"ruta": _relativa(panel_ruta), "sha256": sha256(panel_ruta), "filas": int(len(panel))},
        "dias_parciales_excluidos": dias_parciales,
        "archivos": archivos,
        "faltantes": faltantes,
    }
    _escribir_json(manifiesto, destino / "manifiesto_interfaz.json")
    return manifiesto


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description="Exporta a app/datos/ los artefactos de la interfaz")
    analizador.add_argument("--panel", type=Path, default=RAIZ / "data" / "procesado" / "panel_modelado.parquet")
    analizador.add_argument("--resultados", type=Path, default=RAIZ / "docs" / "resultados")
    analizador.add_argument("--destino", type=Path, default=RAIZ / "app" / "datos")
    analizador.add_argument("--dias-parciales", type=int, default=DIAS_PARCIALES)
    args = analizador.parse_args(argv)

    manifiesto = exportar(args.panel, args.resultados, args.destino, args.dias_parciales)
    print(f"\n=== Interfaz === {_relativa(args.destino)}")
    for nombre, info in manifiesto["archivos"].items():
        filas = f"{info['filas']:>7,} filas" if 'filas' in info else "     modelo"
        print(f"  {nombre:<32} {filas}  {info['bytes'] / 1024:>7.0f} KB  {info['sha256'][:12]}")
    for falta in manifiesto["faltantes"]:
        print(f"  PENDIENTE {falta['archivo']}: falta {falta['falta']} ({falta['script']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
