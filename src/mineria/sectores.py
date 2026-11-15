"""Agrupamiento de los sectores CIIU por su perfil semanal de consumo.

    from mineria.sectores import perfiles_sectoriales, describir_sectores

    perfiles, energia, registro = perfiles_sectoriales(ciiu)   # 168 columnas
    ...K-means con mineria.clustering...
    tabla = describir_sectores(perfiles, energia, etiquetas)

Responde a la pregunta de alcance del proyecto: dentro de la demanda no
regulada, que subactividades se comportan como industria de proceso continuo
(curva plana, sin fin de semana) y cuales como jornada laboral o comercio.
Una subactividad de "INDUSTRIAS MANUFACTURERAS" con perfil de oficina, o una de
servicios con perfil de proceso continuo, es justo lo que la etiqueta CIIU no
dice y el agrupamiento si.

El perfil es la media de cada hora de la semana (lunes 00 h ... domingo 23 h)
dividida por la media del sector: forma, no tamano. Solo se usan valores
observados (`origen_valor == "observado"`); los interpolados y faltantes de la
limpieza por grupos no entran.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

HORAS_SEMANA = 168
OBSERVACIONES_MINIMAS = 24 * 7 * 8     # al menos ~8 semanas de datos
COLUMNAS_SEMANA = [f"d{d}h{h:02d}" for d in range(7) for h in range(24)]


def perfiles_sectoriales(
    ciiu: pd.DataFrame,
    columnas_grupo: tuple[str, str] = ("activity", "subactivity"),
    columna_tiempo: str = "timestamp",
    columna_valor: str = "valor",
    observaciones_minimas: int = OBSERVACIONES_MINIMAS,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Perfil medio por hora de la semana, normalizado, uno por sector."""
    datos = ciiu
    if "origen_valor" in datos:
        datos = datos[datos["origen_valor"].astype(str) == "observado"]
    datos = datos[datos[columna_valor].notna()]

    tiempo = pd.to_datetime(datos[columna_tiempo])
    if tiempo.dt.tz is not None:
        tiempo = tiempo.dt.tz_convert("America/Bogota")
    hora_semana = (tiempo.dt.dayofweek * 24 + tiempo.dt.hour).to_numpy()

    claves = [datos[c].astype(str).to_numpy() for c in columnas_grupo]
    marco = pd.DataFrame(
        {"g0": claves[0], "g1": claves[1], "hs": hora_semana, "v": datos[columna_valor].to_numpy(float)}
    )
    conteo = marco.groupby(["g0", "g1"]).size()
    validos = conteo[conteo >= observaciones_minimas].index
    marco = marco.set_index(["g0", "g1"]).loc[validos].reset_index()

    medias = marco.groupby(["g0", "g1", "hs"])["v"].mean().unstack("hs").reindex(columns=range(HORAS_SEMANA))
    completos = medias.notna().all(axis=1)
    medias = medias[completos]
    media_sector = medias.mean(axis=1)
    positivos = media_sector > 0
    medias, media_sector = medias[positivos], media_sector[positivos]

    perfiles = medias.div(media_sector, axis=0)
    perfiles.columns = COLUMNAS_SEMANA
    perfiles.index = perfiles.index.set_names(list(columnas_grupo))
    energia = marco.groupby(["g0", "g1"])["v"].sum().reindex(perfiles.index)
    energia.index = perfiles.index
    energia.name = "energia_kwh"

    registro = {
        "operacion": "perfiles_sectoriales",
        "criterio": (
            "media de cada hora de la semana sobre valores observados, dividida por la "
            f"media del sector; se exigen >= {observaciones_minimas} observaciones y las "
            "168 horas de la semana con dato"
        ),
        "sectores_entrada": int(len(conteo)),
        "sectores_pocos_datos": int((conteo < observaciones_minimas).sum()),
        "sectores_semana_incompleta": int((~completos).sum()),
        "sectores_media_no_positiva": int((~positivos).sum()),
        "sectores_resultantes": int(len(perfiles)),
    }
    return perfiles, energia, registro


def indicadores_forma(perfiles: pd.DataFrame) -> pd.DataFrame:
    """Rasgos interpretables de cada perfil semanal (ya normalizado, media 1)."""
    matriz = perfiles.to_numpy().reshape(len(perfiles), 7, 24)
    laborables = matriz[:, :5, :].mean(axis=(1, 2))
    domingo = matriz[:, 6, :].mean(axis=1)
    noche = matriz[:, :5, 0:5].mean(axis=(1, 2))
    dia = matriz[:, :5, 9:17].mean(axis=(1, 2))
    return pd.DataFrame(
        {
            "domingo_sobre_laboral": domingo / laborables,
            "noche_sobre_dia": noche / dia,
            "factor_carga": 1.0 / perfiles.max(axis=1).to_numpy(),
        },
        index=perfiles.index,
    )


def describir_sectores(
    perfiles: pd.DataFrame, energia: pd.Series, etiquetas: pd.Series, actividad_industrial: str = "INDUSTRIAS MANUFACTURERAS"
) -> pd.DataFrame:
    """Una fila por grupo: tamano, energia, rasgos de forma y peso de la manufactura."""
    rasgos = indicadores_forma(perfiles)
    datos = rasgos.join(energia).assign(grupo=etiquetas.to_numpy())
    actividad = perfiles.index.get_level_values(0)
    datos["manufactura"] = actividad.str.upper().str.contains(actividad_industrial.upper(), regex=False)
    total = datos["energia_kwh"].sum()
    filas = []
    for grupo, sub in datos.groupby("grupo"):
        filas.append(
            {
                "grupo": grupo,
                "sectores": int(len(sub)),
                "pct_energia": round(100 * float(sub["energia_kwh"].sum() / total), 2),
                "domingo_sobre_laboral": round(float(sub["domingo_sobre_laboral"].median()), 3),
                "noche_sobre_dia": round(float(sub["noche_sobre_dia"].median()), 3),
                "factor_carga": round(float(sub["factor_carga"].median()), 3),
                "pct_sectores_manufactura": round(100 * float(sub["manufactura"].mean()), 1),
                "pct_energia_manufactura": round(
                    100 * float(sub.loc[sub["manufactura"], "energia_kwh"].sum() / sub["energia_kwh"].sum()), 1
                ),
            }
        )
    return pd.DataFrame(filas).set_index("grupo")


def mayores_por_grupo(energia: pd.Series, etiquetas: pd.Series, n: int = 5) -> pd.DataFrame:
    """Los `n` sectores de mas energia en cada grupo, para nombrarlos en el informe."""
    datos = energia.to_frame().assign(grupo=etiquetas.to_numpy()).reset_index()
    datos = datos.sort_values(["grupo", "energia_kwh"], ascending=[True, False])
    return datos.groupby("grupo").head(n).reset_index(drop=True)


def ordenar_por_tamano(etiquetas_crudas: np.ndarray) -> np.ndarray:
    """Renombra etiquetas enteras a 'S1'..'Sk' con S1 el grupo mas numeroso."""
    tamanos = np.bincount(etiquetas_crudas)
    orden = np.argsort(-tamanos, kind="stable")
    mapa = {int(v): f"S{i + 1}" for i, v in enumerate(orden)}
    return np.array([mapa[int(e)] for e in etiquetas_crudas])
