"""Limpieza del catalogo de metricas de XM (`ListadoMetricas`).

El catalogo no es una serie temporal sino una tabla de referencia de 193 filas,
asi que no le aplican huecos ni atipicos. Sus problemas son de texto y de
coherencia, y todos se encontraron mirando el archivo real, no suponiendolos:

- el mismo centinela escrito de dos formas: `"No aplica"` y `"No Aplica"`;
- la misma entidad con dos grafias: `"SubArea"` y `"Subarea"`;
- 7 unidades vacias, 13 descripciones con espacios sobrantes y 5 nombres con
  doble espacio (`"Aportes  Energia por Rio"`, entre otros);
- una URL equivocada: el catalogo anuncia `/list` para las metricas de listado,
  pero ese endpoint responde 404; el que funciona es `/lists`.

Como en el resto del proyecto, cada operacion devuelve `(marco, registro)` y el
registro dice que se cambio, cuantas filas y con que criterio. Nada se inventa:
lo que no se puede corregir con evidencia se marca, no se reescribe.

    from limpieza.catalogo import limpiar_catalogo
    limpio, registro = limpiar_catalogo(pd.read_csv("datasets/xm/xm_catalogo_metricas.csv"))
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

import pandas as pd

from limpieza.limpiar import ErrorLimpieza, _registro, a_snake_case

log = logging.getLogger(__name__)

# Granularidad que corresponde a cada `Type` del catalogo, y el endpoint que la
# sirve. Un Type nuevo hace fallar la limpieza en vez de quedar sin clasificar.
GRANULARIDAD_POR_TIPO = {
    "HourlyEntities": "horaria",
    "DailyEntities": "diaria",
    "MonthlyEntities": "mensual",
    "ListsEntities": "listado",
}
ENDPOINT_POR_GRANULARIDAD = {
    "horaria": "/hourly",
    "diaria": "/daily",
    "mensual": "/monthly",
    "listado": "/lists",
}

# Correcciones de URL respaldadas por una comprobacion contra el servidor. Solo
# se corrige lo que esta verificado; cualquier otra URL se deja como viene.
#   2026-09-10: POST /list  -> 404 sin cuerpo
#               POST /lists -> 200, 193 metricas
CORRECCIONES_URL = {
    "https://servapibi.xm.com.co/list": "https://servapibi.xm.com.co/lists",
}

# Valores que significan "sin filtro". Se comparan sin distinguir mayusculas,
# que es justo como aparecen en el catalogo: "No aplica" y "No Aplica".
CENTINELAS_SIN_FILTRO = {"no aplica"}

COLUMNAS_TEXTO = (
    "metric_id", "metric_name", "entity", "type", "url",
    "filter", "metric_units", "metric_description",
)


def _ahora() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Operaciones
# --------------------------------------------------------------------------


def normalizar_nombres(marco: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Columnas a snake_case: `MetricId` -> `metric_id`."""
    renombres = {c: a_snake_case(c) for c in marco.columns}
    limpio = marco.rename(columns=renombres)
    return limpio, _registro(
        "normalizar_nombres", "nombres de columna a snake_case",
        len(marco), len(limpio), filas_afectadas=0,
        renombradas={k: v for k, v in renombres.items() if k != v},
    )


def limpiar_texto(marco: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Quita espacios en los extremos y colapsa los espacios repetidos."""
    limpio = marco.copy()
    cambios: dict[str, int] = {}
    ejemplos: dict[str, list[str]] = {}

    for columna in (c for c in COLUMNAS_TEXTO if c in limpio.columns):
        original = limpio[columna]
        texto = original.astype("string")
        nuevo = texto.str.strip().str.replace(r"\s{2,}", " ", regex=True)
        distinto = (texto != nuevo).fillna(False)
        if distinto.any():
            cambios[columna] = int(distinto.sum())
            ejemplos[columna] = [repr(v) for v in original[distinto].head(3)]
        limpio[columna] = nuevo.astype(object).where(nuevo.notna(), None)

    return limpio, _registro(
        "limpiar_texto",
        "espacios en los extremos eliminados y espacios repetidos colapsados a uno",
        len(marco), len(limpio),
        filas_afectadas=int(sum(cambios.values())),
        cambios_por_columna=cambios,
        ejemplos_originales=ejemplos,
    )


def vacios_a_nulo(marco: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Las cadenas vacias pasan a nulo: un vacio no es una unidad ni una descripcion."""
    limpio = marco.copy()
    nulos: dict[str, int] = {}
    for columna in (c for c in COLUMNAS_TEXTO if c in limpio.columns):
        vacio = limpio[columna].astype("string").fillna("").eq("") & limpio[columna].notna()
        if vacio.any():
            nulos[columna] = int(vacio.sum())
            limpio.loc[vacio, columna] = None

    return limpio, _registro(
        "vacios_a_nulo", "cadenas vacias convertidas a nulo",
        len(marco), len(limpio),
        filas_afectadas=int(sum(nulos.values())),
        nulos_por_columna=nulos,
    )


def normalizar_filtro(marco: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """El centinela "No aplica" (en cualquier grafia) pasa a nulo y se anade `tiene_filtro`."""
    limpio = marco.copy()
    texto = limpio["filter"].astype("string")
    es_centinela = texto.str.casefold().isin(CENTINELAS_SIN_FILTRO).fillna(False)
    variantes = texto[es_centinela].value_counts().to_dict()

    limpio.loc[es_centinela, "filter"] = None
    limpio["tiene_filtro"] = limpio["filter"].notna()

    return limpio, _registro(
        "normalizar_filtro",
        "valores 'No aplica' (sin distinguir mayusculas) a nulo; columna tiene_filtro",
        len(marco), len(limpio),
        filas_afectadas=int(es_centinela.sum()),
        grafias_encontradas={str(k): int(v) for k, v in variantes.items()},
    )


def normalizar_entidad(marco: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Anade `entidad_normalizada` unificando grafias que solo difieren en mayusculas.

    `entity` NO se toca: es el literal que se envia a la API en el campo
    `Entity`, y no esta verificado que la API acepte las dos grafias de un mismo
    nombre. La columna nueva sirve para agrupar y contar sin duplicar entidades.
    """
    limpio = marco.copy()
    conteo = limpio["entity"].value_counts()
    canonica: dict[str, str] = {}
    unificadas: dict[str, list[str]] = {}

    for clave, grafias in conteo.groupby(conteo.index.str.casefold()):
        # La grafia mas frecuente gana; en empate, la primera en orden alfabetico.
        orden = sorted(grafias.items(), key=lambda par: (-par[1], par[0]))
        elegida = orden[0][0]
        for grafia, _ in orden:
            canonica[grafia] = elegida
        if len(orden) > 1:
            unificadas[elegida] = [g for g, _ in orden]

    limpio["entidad_normalizada"] = limpio["entity"].map(canonica)
    afectadas = int((limpio["entidad_normalizada"] != limpio["entity"]).sum())

    return limpio, _registro(
        "normalizar_entidad",
        "entidad_normalizada unifica grafias que solo difieren en mayusculas; "
        "entity conserva el literal que espera la API",
        len(marco), len(limpio),
        filas_afectadas=afectadas,
        grafias_unificadas=unificadas,
        n_entidades_originales=int(conteo.size),
        n_entidades_normalizadas=int(limpio["entidad_normalizada"].nunique()),
    )


def corregir_url(marco: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Corrige las URL con una correccion verificada y marca las filas tocadas."""
    limpio = marco.copy()
    corregir = limpio["url"].isin(CORRECCIONES_URL)
    antes = limpio.loc[corregir, "url"].value_counts().to_dict()

    limpio["url_corregida"] = corregir
    limpio.loc[corregir, "url"] = limpio.loc[corregir, "url"].map(CORRECCIONES_URL)

    return limpio, _registro(
        "corregir_url",
        "URL sustituidas solo cuando hay una correccion verificada contra el "
        "servidor (CORRECCIONES_URL)",
        len(marco), len(limpio),
        filas_afectadas=int(corregir.sum()),
        correcciones={str(k): CORRECCIONES_URL[k] for k in antes},
        filas_por_url_original={str(k): int(v) for k, v in antes.items()},
        evidencia="2026-09-10: POST /list -> 404; POST /lists -> 200 con 193 metricas",
    )


def derivar_granularidad(marco: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Anade `granularidad` desde `type` y comprueba que la URL le corresponde."""
    limpio = marco.copy()
    desconocidos = sorted(set(limpio["type"].dropna()) - set(GRANULARIDAD_POR_TIPO))
    if desconocidos:
        raise ErrorLimpieza(
            f"Tipos de metrica sin granularidad definida: {desconocidos}. "
            "Anadelos a GRANULARIDAD_POR_TIPO antes de limpiar."
        )

    limpio["granularidad"] = limpio["type"].map(GRANULARIDAD_POR_TIPO)
    esperado = limpio["granularidad"].map(ENDPOINT_POR_GRANULARIDAD)
    limpio["url_coherente"] = [
        isinstance(u, str) and u.endswith(e) for u, e in zip(limpio["url"], esperado)
    ]
    incoherentes = limpio.loc[~limpio["url_coherente"], ["metric_id", "entity", "url"]]

    return limpio, _registro(
        "derivar_granularidad",
        "granularidad = traduccion de type; url_coherente = la URL apunta al "
        "endpoint de esa granularidad",
        len(marco), len(limpio),
        filas_afectadas=0,
        por_granularidad=limpio["granularidad"].value_counts().to_dict(),
        n_url_incoherentes=int(len(incoherentes)),
        url_incoherentes=incoherentes.head(10).to_dict("records"),
    )


def validar(marco: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Comprueba la identidad de cada fila y el tipo de max_days. Falla si no cuadra."""
    duplicados = marco[marco.duplicated(["metric_id", "entity"], keep=False)]
    if not duplicados.empty:
        raise ErrorLimpieza(
            f"El catalogo repite (metric_id, entity) en {len(duplicados)} filas: "
            f"{duplicados[['metric_id', 'entity']].head(5).to_dict('records')}"
        )

    max_dias = pd.to_numeric(marco["max_days"], errors="coerce")
    if max_dias.isna().any() or (max_dias <= 0).any():
        raise ErrorLimpieza("max_days debe ser un entero positivo en todas las filas.")

    limpio = marco.copy()
    limpio["max_days"] = max_dias.astype("int64")
    return limpio, _registro(
        "validar",
        "(metric_id, entity) unico; max_days entero positivo",
        len(marco), len(limpio), filas_afectadas=0,
        valores_max_days=sorted(int(v) for v in limpio["max_days"].unique()),
    )


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------


def limpiar_catalogo(marco: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica todas las operaciones en orden y devuelve (catalogo, registro)."""
    if marco.empty:
        raise ErrorLimpieza("El catalogo esta vacio.")

    operaciones: list[dict[str, Any]] = []
    limpio = marco
    for paso in (
        normalizar_nombres, limpiar_texto, vacios_a_nulo, normalizar_filtro,
        normalizar_entidad, corregir_url, derivar_granularidad, validar,
    ):
        limpio, registro = paso(limpio)
        operaciones.append(registro)

    registro = {
        "version_formato": 1,
        "conjunto": "catalogo_metricas_xm",
        "momento": _ahora(),
        "filas_iniciales": int(len(marco)),
        "filas_finales": int(len(limpio)),
        "columnas_finales": list(limpio.columns),
        "operaciones": operaciones,
        "politica": (
            "entity se conserva literal para la API; solo se corrigen URL con "
            "evidencia; lo demas se normaliza o se marca, nunca se inventa."
        ),
    }
    return limpio.reset_index(drop=True), registro


def resumen(registro: dict[str, Any]) -> str:
    """Version legible del registro de limpieza del catalogo."""
    lineas = ["=" * 74, "LIMPIEZA DEL CATALOGO DE METRICAS", "=" * 74]
    lineas.append(f"Filas: {registro['filas_iniciales']} -> {registro['filas_finales']}")
    for paso in registro["operaciones"]:
        lineas.append("")
        lineas.append(f"[{paso['operacion']}]  {paso['filas_afectadas']} filas afectadas")
        lineas.append(f"  {paso['criterio']}")
        for clave, valor in paso["detalle"].items():
            if valor in ({}, [], None, 0, ""):
                continue
            lineas.append(f"  {clave}: {valor}")
    return "\n".join(lineas)
