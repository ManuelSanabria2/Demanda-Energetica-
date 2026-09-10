"""Diagnostico de calidad de una serie horaria. No corrige nada.

Este modulo solo mira y reporta. Ninguna funcion rellena huecos, elimina
duplicados ni recorta valores extremos: el objetivo es saber que hay antes de
decidir que hacer con ello.

    from calidad.diagnostico import diagnosticar, resumen

    informe = diagnosticar(marco)      # dict serializable a JSON
    print(resumen(informe))            # version legible

El informe distingue dos formas de tabla, porque no se diagnostican igual:

- **Agregada**: una fila por marca de tiempo (p. ej. XM DemaReal/Sistema). Un
  timestamp repetido es un defecto.
- **Desagregada**: varias filas por marca de tiempo (p. ej. SIMEM 14fabb, con
  93 combinaciones de agente y mercado por cada version de liquidacion). Ahi
  los timestamps repetidos son lo normal, y lo que hay que vigilar es que no se
  repita la clave completa.

Se detecta sola por el numero medio de filas por marca de tiempo, y el informe
dice cual asumio.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Nombres de columna que usa el proyecto, en orden de preferencia.
CANDIDATAS_TIEMPO = ("fecha_hora", "timestamp", "FechaHora")
CANDIDATAS_VALOR = ("valor_kwh", "valor", "Valor")

# Columnas que nunca son dimensiones categoricas de interes.
COLUMNAS_TECNICAS = frozenset({"anio", "mes"})

# Umbrales por defecto.
FACTOR_IQR = 1.5
# 3.5 es el umbral habitual para la puntuacion z modificada (Iglewicz y Hoaglin).
UMBRAL_Z = 3.5
# Un grupo (dia de semana, hora) con pocas observaciones no da una mediana
# fiable: la MAD subestima la dispersion en muestras pequenas y eso infla la
# puntuacion z, produciendo atipicos que no lo son. Medido sobre una serie
# sintetica limpia con ruido gaussiano del 2% (la tasa teorica para |z|>3.5
# seria ~0.05%):
#
#     dias   obs/grupo   falsos positivos
#       60         8.6              2.50%
#      120        17.1              0.83%
#      180        25.7              0.58%
#      365        52.1              0.24%
#      730       104.3              0.07%
#
# 20 observaciones equivalen a unos cinco meses de historico. Por debajo de
# eso el grupo se declara no evaluable en vez de emitir un juicio poco fiable.
MINIMO_POR_GRUPO = 20

# Por debajo de esta mediana de observaciones por grupo, el informe avisa de
# que la tasa de atipicos estacionales esta inflada. Vive en config para que
# diagnostico y limpieza no puedan reportar fiabilidades distintas.
from ingesta.config import GRUPO_COMODO, UMBRAL_FILAS_POR_MARCA  # noqa: E402

HORAS_DEL_DIA = 24


# --------------------------------------------------------------------------
# Deteccion de columnas y forma
# --------------------------------------------------------------------------


def _elegir_columna(
    marco: pd.DataFrame, candidatas: tuple[str, ...], explicita: str | None, papel: str
) -> str:
    """Devuelve la columna a usar, explicita o detectada entre las candidatas."""
    if explicita is not None:
        if explicita not in marco.columns:
            raise KeyError(
                f"La columna de {papel} {explicita!r} no esta en el marco. "
                f"Columnas: {list(marco.columns)}"
            )
        return explicita

    for candidata in candidatas:
        if candidata in marco.columns:
            return candidata

    raise KeyError(
        f"No se encontro columna de {papel}. Se buscaron {candidatas} y el marco "
        f"tiene {list(marco.columns)}. Pasala explicitamente."
    )


def _columnas_categoricas(marco: pd.DataFrame, excluir: set[str]) -> list[str]:
    """Columnas no numericas ni temporales que actuan como dimension."""
    return [
        columna
        for columna in marco.columns
        if columna not in excluir
        and columna not in COLUMNAS_TECNICAS
        and not pd.api.types.is_numeric_dtype(marco[columna])
        and not pd.api.types.is_datetime64_any_dtype(marco[columna])
    ]


# --------------------------------------------------------------------------
# Completitud temporal
# --------------------------------------------------------------------------


def _clasificar_hueco(n_horas: int) -> str:
    """Clasifica un tramo faltante por duracion.

    La distincion no es cosmetica: tres horas sueltas se pueden interpolar sin
    mayor riesgo, mientras que dos semanas seguidas no se rellenan, se excluyen
    del entrenamiento o se tratan como un periodo aparte.
    """
    if n_horas == 1:
        return "aislado"
    if n_horas <= 5:
        return "corto"
    if n_horas <= HORAS_DEL_DIA:
        return "medio"
    return "largo"


def _tramos_faltantes(faltantes: pd.DatetimeIndex) -> list[dict[str, Any]]:
    """Agrupa horas faltantes consecutivas en tramos."""
    if len(faltantes) == 0:
        return []

    serie = faltantes.to_series().reset_index(drop=True)
    corte = serie.diff() != pd.Timedelta(hours=1)
    grupos = corte.cumsum()

    tramos = []
    for _, bloque in serie.groupby(grupos):
        n_horas = int(len(bloque))
        tramos.append(
            {
                "inicio": str(bloque.iloc[0]),
                "fin": str(bloque.iloc[-1]),
                "n_horas": n_horas,
                "clase": _clasificar_hueco(n_horas),
            }
        )
    return tramos


def _completitud(
    marco: pd.DataFrame,
    columna_tiempo: str,
    columnas_clave: list[str] | None,
    top: int,
) -> dict[str, Any]:
    """Huecos, duplicados y dias que no tienen 24 horas."""
    momentos = pd.to_datetime(marco[columna_tiempo])
    unicos = pd.DatetimeIndex(momentos.unique()).sort_values()

    esperadas = pd.date_range(unicos.min(), unicos.max(), freq="h")
    faltantes = esperadas.difference(unicos)
    tramos = _tramos_faltantes(faltantes)

    por_clase: dict[str, int] = {}
    for tramo in tramos:
        por_clase[tramo["clase"]] = por_clase.get(tramo["clase"], 0) + 1

    # Muchos tramos de una hora significan huecos dispersos; pocos tramos con
    # muchas horas, huecos agrupados.
    horas_por_tramo = (len(faltantes) / len(tramos)) if tramos else 0.0
    if not tramos:
        forma = "sin huecos"
    elif horas_por_tramo < 1.5:
        forma = "dispersos"
    elif horas_por_tramo >= 12:
        forma = "agrupados"
    else:
        forma = "mixtos"

    # Dias que no tienen 24 horas distintas. Colombia no aplica horario de
    # verano, asi que un dia de 23 o 25 horas no es un cambio de hora legitimo:
    # es un error de parseo o de deduplicacion.
    horas_por_dia = momentos.dt.normalize().groupby(momentos.dt.normalize()).size()
    horas_distintas = momentos.groupby(momentos.dt.normalize()).nunique()
    anomalos = horas_distintas[horas_distintas != HORAS_DEL_DIA]

    dias_incompletos = [
        {"dia": str(dia.date()), "horas_distintas": int(n)}
        for dia, n in anomalos.items()
    ]
    # El primer y el ultimo dia suelen estar truncados por el propio rango
    # pedido; se marcan aparte para no confundirlos con un defecto de los datos.
    bordes = {str(unicos.min().date()), str(unicos.max().date())}
    for entrada in dias_incompletos:
        entrada["es_borde_del_rango"] = entrada["dia"] in bordes

    informe: dict[str, Any] = {
        "rango": {"inicio": str(unicos.min()), "fin": str(unicos.max())},
        "horas_esperadas": int(len(esperadas)),
        "horas_presentes": int(len(unicos)),
        "horas_faltantes": int(len(faltantes)),
        "completitud_pct": round(100 * len(unicos) / len(esperadas), 4),
        "huecos": {
            "n_tramos": len(tramos),
            "forma": forma,
            "horas_por_tramo_media": round(horas_por_tramo, 2),
            "tramos_por_clase": por_clase,
            "tramo_mas_largo": max(tramos, key=lambda t: t["n_horas"]) if tramos else None,
            "tramos": sorted(tramos, key=lambda t: -t["n_horas"])[:top],
            "n_tramos_no_listados": max(0, len(tramos) - top),
        },
        "dias_sin_24_horas": {
            "n_dias": len(dias_incompletos),
            "n_dias_interiores": sum(
                1 for d in dias_incompletos if not d["es_borde_del_rango"]
            ),
            "dias": dias_incompletos[:top],
            "nota": (
                "Colombia no aplica horario de verano: un dia con 23 o 25 horas "
                "indica un error de parseo o de deduplicacion, no un cambio de hora."
            ),
        },
    }

    # Duplicados: se miden de dos formas porque significan cosas distintas
    # segun la tabla sea agregada o desagregada.
    n_dup_tiempo = int(momentos.duplicated().sum())
    informe["duplicados"] = {
        "timestamps_repetidos": n_dup_tiempo,
        "filas_por_timestamp_media": round(len(marco) / len(unicos), 2),
        "ejemplos": [
            str(m) for m in momentos[momentos.duplicated(keep=False)].unique()[:top]
        ],
    }

    if columnas_clave:
        faltan = [c for c in columnas_clave if c not in marco.columns]
        if faltan:
            raise KeyError(f"Columnas de clave ausentes en el marco: {faltan}")
        n_dup_clave = int(marco.duplicated(subset=columnas_clave).sum())
        informe["duplicados"]["clave"] = columnas_clave
        informe["duplicados"]["clave_repetida"] = n_dup_clave

    return informe


# --------------------------------------------------------------------------
# Valores
# --------------------------------------------------------------------------


def _nulos_por_columna(marco: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Recuento y porcentaje de nulos, columna a columna."""
    total = len(marco)
    nulos = {}
    for columna in marco.columns:
        n = int(marco[columna].isna().sum())
        nulos[str(columna)] = {
            "n_nulos": n,
            "pct": round(100 * n / total, 4) if total else 0.0,
        }
    return nulos


def _sospechosos_fisicos(
    valores: pd.Series, momentos: pd.Series, top: int
) -> dict[str, Any]:
    """Ceros y negativos: la demanda electrica agregada no puede ser ninguno."""
    ceros = valores == 0
    negativos = valores < 0

    return {
        "n_ceros": int(ceros.sum()),
        "n_negativos": int(negativos.sum()),
        "ejemplos_ceros": [str(m) for m in momentos[ceros].head(top)],
        "ejemplos_negativos": [
            {"momento": str(m), "valor": float(v)}
            for m, v in zip(momentos[negativos].head(top), valores[negativos].head(top))
        ],
        "nota": (
            "La demanda agregada de un sistema no puede ser cero ni negativa. "
            "Un cero suele ser un hueco codificado como numero; un negativo, un "
            "signo invertido o una compensacion de liquidacion."
        ),
    }


def _outliers_iqr(
    valores: pd.Series, momentos: pd.Series, factor: float, top: int
) -> dict[str, Any]:
    """Atipicos por rango intercuartilico sobre toda la serie.

    Criterio global: ignora que la demanda tiene forma horaria y semanal, asi
    que tiende a senalar las horas punta y los valles como atipicos aunque sean
    perfectamente normales. Se incluye como referencia, no como criterio
    principal.
    """
    limpios = valores.dropna()
    if limpios.empty:
        return {"aplicable": False, "motivo": "no hay valores no nulos"}

    q1, q3 = limpios.quantile([0.25, 0.75])
    iqr = q3 - q1
    bajo, alto = q1 - factor * iqr, q3 + factor * iqr
    fuera = (valores < bajo) | (valores > alto)

    extremos = (
        pd.DataFrame({"momento": momentos, "valor": valores})[fuera.fillna(False)]
        .assign(desviacion=lambda d: (d["valor"] - limpios.median()).abs())
        .nlargest(top, "desviacion")
    )

    return {
        "aplicable": True,
        "factor": factor,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "limite_inferior": float(bajo),
        "limite_superior": float(alto),
        "n_outliers": int(fuera.sum()),
        "pct": round(100 * fuera.sum() / len(valores), 4),
        "ejemplos": [
            {"momento": str(f["momento"]), "valor": float(f["valor"])}
            for _, f in extremos.iterrows()
        ],
        "nota": (
            "Criterio global: no distingue la hora del dia, asi que marca picos "
            "y valles normales. Ver el criterio estacional."
        ),
    }


def z_estacional(
    marco: pd.DataFrame,
    columna_tiempo: str,
    columna_valor: str,
    columnas_grupo: list[str] | None = None,
    minimo_por_grupo: int = MINIMO_POR_GRUPO,
    causal: bool = False,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Puntuacion z modificada de cada fila frente a su (dia de semana, hora).

    Es el nucleo del criterio estacional, y vive aqui como funcion publica para
    que el diagnostico y la limpieza compartan una unica implementacion: si el
    criterio cambia, cambia para los dos a la vez.

    Devuelve (z, no_evaluable, contexto). Las filas no evaluables llevan z nula
    y no deben contarse como normales ni como atipicas: simplemente no se sabe.

    **`causal` decide si hay fuga temporal.**

    Con `causal=False` (por defecto) la mediana y la MAD de cada grupo se
    calculan sobre la serie entera, futuro incluido. Para *diagnosticar* eso es
    lo correcto: se quiere el mejor juicio posible con todo lo que se sabe.

    Con `causal=True` cada fila se juzga solo contra las observaciones
    **anteriores** de su mismo grupo, con estadisticos expansivos. Es lo que
    hay que usar si la marca va a entrar en un modelo -- como variable o como
    filtro de entrenamiento -- porque si no, el pasado se estaria etiquetando
    con conocimiento del futuro y la validacion saldria optimista.

    Medido sobre la serie real: cortando en 2023-12-31, el modo global marca 90
    atipicos y el causal 255; 165 banderas cambian segun se incluya o no el
    futuro.
    """
    datos = marco[[columna_tiempo, columna_valor]].copy()
    datos[columna_valor] = pd.to_numeric(datos[columna_valor], errors="coerce")
    momentos = pd.to_datetime(marco[columna_tiempo])
    datos["_dia_semana"] = momentos.dt.dayofweek
    datos["_hora"] = momentos.dt.hour

    claves = ["_dia_semana", "_hora"] + list(columnas_grupo or [])
    for columna in columnas_grupo or []:
        datos[columna] = marco[columna]

    agrupado = datos.groupby(claves, dropna=False)[columna_valor]

    if causal:
        # Estadisticos expansivos sobre las observaciones ANTERIORES de cada
        # grupo. El shift(1) es lo que garantiza que la propia fila no entre en
        # el estadistico que la juzga.
        orden = momentos.sort_values().index
        ordenados = datos.loc[orden]
        agrupado_ord = ordenados.groupby(claves, dropna=False)[columna_valor]

        mediana = agrupado_ord.transform(
            lambda s: s.shift(1).expanding().median()
        ).reindex(datos.index)
        tamano = agrupado_ord.transform(
            lambda s: s.shift(1).expanding().count()
        ).reindex(datos.index)

        desviacion = (datos[columna_valor] - mediana).abs()
        desv_ord = desviacion.loc[orden]
        claves_ord = [ordenados[c] for c in claves]
        mad = (
            desv_ord.groupby(claves_ord, dropna=False)
            .transform(lambda s: s.shift(1).expanding().median())
            .reindex(datos.index)
        )
        desviacion_media = (
            desv_ord.groupby(claves_ord, dropna=False)
            .transform(lambda s: s.shift(1).expanding().mean())
            .reindex(datos.index)
        )
    else:
        mediana = agrupado.transform("median")
        tamano = agrupado.transform("size")

        desviacion = (datos[columna_valor] - mediana).abs()
        por_grupo = [datos[c] for c in claves]
        mad = desviacion.groupby(por_grupo, dropna=False).transform("median")

    # La MAD tiene un punto ciego: si un grupo es constante salvo por un unico
    # valor extremo, su mediana de desviaciones es cero y el atipico queda
    # invisible. Iglewicz y Hoaglin proponen para ese caso recurrir a la
    # desviacion absoluta media, que si reacciona a un solo valor.
        desviacion_media = desviacion.groupby(por_grupo, dropna=False).transform("mean")

    with np.errstate(divide="ignore", invalid="ignore"):
        z_mad = 0.6745 * (datos[columna_valor] - mediana) / mad
        z_media = (datos[columna_valor] - mediana) / (1.253314 * desviacion_media)

    # 0.6745 y 1.253314 escalan cada estimador para que su z sea comparable a
    # la de una normal, y por tanto ambas al mismo umbral.
    z = z_mad.where(mad > 0, z_media)

    # Solo es inevaluable un grupo verdaderamente constante (ninguna dispersion
    # por ninguna de las dos medidas) o con muy pocas observaciones.
    sin_dispersion = (mad == 0) & (desviacion_media == 0)
    poco_poblados = tamano.fillna(0) < minimo_por_grupo
    no_evaluable = (
        sin_dispersion.fillna(True) | poco_poblados | datos[columna_valor].isna()
    )
    z = z.where(~no_evaluable)

    contexto = {
        "causal": causal,
        "mediana_grupo": mediana,
        "tamano_grupo": tamano,
        "mad": mad,
        "desviacion_media": desviacion_media,
        "n_grupos": int(agrupado.ngroups),
        "minimo_por_grupo": minimo_por_grupo,
    }
    return z, no_evaluable, contexto


def limites_iqr_causal(
    valores: pd.Series,
    momentos: pd.Series,
    factor: float = FACTOR_IQR,
    minimo: int = MINIMO_POR_GRUPO,
) -> tuple[pd.Series, pd.Series]:
    """Limites IQR calculados solo con las observaciones anteriores a cada fila.

    Devuelve dos series alineadas con `valores`. Las primeras filas, sin
    historico suficiente, llevan NaN: no se puede juzgar y no se inventa.

    La version global de esta funcion mira la serie entera; sobre los datos
    reales el limite superior se desplaza 333 092 kWh segun se incluya o no el
    futuro, asi que para marcar filas destinadas a un modelo hay que usar esta.
    """
    numeros = pd.to_numeric(valores, errors="coerce")
    orden = pd.to_datetime(momentos).sort_values().index
    ordenados = numeros.loc[orden]

    previos = ordenados.shift(1)
    q1 = previos.expanding(min_periods=minimo).quantile(0.25).reindex(numeros.index)
    q3 = previos.expanding(min_periods=minimo).quantile(0.75).reindex(numeros.index)
    iqr = q3 - q1
    return (q1 - factor * iqr, q3 + factor * iqr)


def limites_iqr(
    valores: pd.Series, factor: float = FACTOR_IQR
) -> tuple[float, float]:
    """Limites inferior y superior del criterio global por rango intercuartilico.

    Usa la serie entera, futuro incluido. Correcto para diagnosticar; para
    marcar filas que van a un modelo, usar `limites_iqr_causal`.
    """
    limpios = pd.to_numeric(valores, errors="coerce").dropna()
    if limpios.empty:
        return (float("nan"), float("nan"))
    q1, q3 = limpios.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (float(q1 - factor * iqr), float(q3 + factor * iqr))


def _outliers_estacionales(
    marco: pd.DataFrame,
    columna_tiempo: str,
    columna_valor: str,
    columnas_grupo: list[str] | None,
    umbral: float,
    minimo_por_grupo: int,
    top: int,
) -> dict[str, Any]:
    """Atipicos frente a la misma hora del mismo dia de la semana.

    Este es el criterio que importa. Un valor alto a las 2 p.m. de un martes es
    normal; el mismo valor a las 3 a.m. de un domingo no lo es. Se compara cada
    observacion contra la mediana de su grupo (dia de la semana, hora) usando
    la puntuacion z modificada, que se apoya en la mediana y la desviacion
    absoluta mediana y por tanto no se deja arrastrar por los propios atipicos.
    """
    momentos = pd.to_datetime(marco[columna_tiempo])
    valores = pd.to_numeric(marco[columna_valor], errors="coerce")

    z, no_evaluable, contexto = z_estacional(
        marco, columna_tiempo, columna_valor, columnas_grupo, minimo_por_grupo
    )
    mediana = contexto["mediana_grupo"]
    tamano = contexto["tamano_grupo"]
    mad = contexto["mad"]
    desviacion_media = contexto["desviacion_media"]
    agrupado_n = contexto["n_grupos"]

    fuera = z.abs() > umbral

    detalle = pd.DataFrame(
        {
            "momento": momentos,
            "valor": valores,
            "mediana_grupo": mediana,
            "z": z,
            "dia_semana": momentos.dt.dayofweek,
            "hora": momentos.dt.hour,
        }
    )
    peores = detalle[fuera.fillna(False)].reindex(
        detalle[fuera.fillna(False)]["z"].abs().sort_values(ascending=False).index
    )

    dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    por_hora = detalle[fuera.fillna(False)].groupby("hora").size()

    observaciones = tamano[~valores.isna()]
    mediana_por_grupo = float(observaciones.median()) if len(observaciones) else 0.0

    fiabilidad = (
        "alta"
        if mediana_por_grupo >= GRUPO_COMODO
        else ("limitada" if mediana_por_grupo >= minimo_por_grupo else "insuficiente")
    )

    return {
        "aplicable": True,
        "criterio": "z modificada frente a la mediana de (dia de semana, hora)",
        "observaciones_por_grupo_mediana": round(mediana_por_grupo, 1),
        "fiabilidad": fiabilidad,
        "umbral_z": umbral,
        "columnas_grupo": list(columnas_grupo or []),
        "n_evaluables": int((~no_evaluable).sum()),
        "n_no_evaluables": int(no_evaluable.sum()),
        "n_por_desviacion_media": int(((mad == 0) & (desviacion_media > 0)).sum()),
        "n_grupos": int(agrupado_n),
        "n_outliers": int(fuera.sum()),
        "pct": round(100 * fuera.sum() / max(1, (~no_evaluable).sum()), 4),
        "por_hora_del_dia": {int(h): int(n) for h, n in por_hora.items()},
        "ejemplos": [
            {
                "momento": str(f["momento"]),
                "valor": float(f["valor"]),
                "mediana_grupo": float(f["mediana_grupo"]),
                "z": round(float(f["z"]), 2),
                "dia_semana": dias[int(f["dia_semana"])],
                "hora": int(f["hora"]),
            }
            for _, f in peores.head(top).iterrows()
        ],
    }


def _valores(
    marco: pd.DataFrame,
    columna_tiempo: str,
    columna_valor: str,
    columnas_grupo: list[str] | None,
    factor_iqr: float,
    umbral_z: float,
    minimo_por_grupo: int,
    top: int,
) -> dict[str, Any]:
    """Nulos, sospechosos fisicos y atipicos por los dos criterios."""
    valores = pd.to_numeric(marco[columna_valor], errors="coerce")
    momentos = pd.to_datetime(marco[columna_tiempo])

    limpios = valores.dropna()
    descriptivos = (
        {
            "n": int(len(limpios)),
            "media": float(limpios.mean()),
            "mediana": float(limpios.median()),
            "desviacion": float(limpios.std()),
            "minimo": float(limpios.min()),
            "maximo": float(limpios.max()),
        }
        if not limpios.empty
        else {"n": 0}
    )

    return {
        "columna": columna_valor,
        "descriptivos": descriptivos,
        "nulos_por_columna": _nulos_por_columna(marco),
        "sospechosos_fisicos": _sospechosos_fisicos(valores, momentos, top),
        "outliers_iqr": _outliers_iqr(valores, momentos, factor_iqr, top),
        "outliers_estacionales": _outliers_estacionales(
            marco, columna_tiempo, columna_valor, columnas_grupo,
            umbral_z, minimo_por_grupo, top,
        ),
    }


# --------------------------------------------------------------------------
# Estructura
# --------------------------------------------------------------------------


def _estructura(
    marco: pd.DataFrame, columna_tiempo: str, columna_valor: str, top: int
) -> dict[str, Any]:
    """Rango temporal, continuidad y cardinalidad de las categoricas."""
    momentos = pd.to_datetime(marco[columna_tiempo])
    unicos = pd.DatetimeIndex(momentos.unique()).sort_values()
    esperadas = pd.date_range(unicos.min(), unicos.max(), freq="h")

    dias_cubiertos = int(momentos.dt.normalize().nunique())
    categoricas = _columnas_categoricas(marco, {columna_tiempo, columna_valor})

    cardinalidad = {}
    for columna in categoricas:
        conteo = marco[columna].value_counts(dropna=False)
        cardinalidad[str(columna)] = {
            "n_distintos": int(marco[columna].nunique(dropna=False)),
            "valores_mas_frecuentes": {
                str(k): int(v) for k, v in conteo.head(top).items()
            },
            "todos_los_valores": (
                [str(v) for v in conteo.index] if len(conteo) <= top else None
            ),
        }

    return {
        "n_filas": int(len(marco)),
        "n_columnas": int(len(marco.columns)),
        "columnas": {str(c): str(marco[c].dtype) for c in marco.columns},
        "rango_temporal": {
            "inicio": str(unicos.min()),
            "fin": str(unicos.max()),
            "dias_cubiertos": dias_cubiertos,
            "dias_del_rango": int((unicos.max().date() - unicos.min().date()).days) + 1,
            "horas_del_rango": int(len(esperadas)),
        },
        "continuidad_pct": round(100 * len(unicos) / len(esperadas), 4),
        "cardinalidad_categoricas": cardinalidad,
    }


# --------------------------------------------------------------------------
# Serializacion
# --------------------------------------------------------------------------


def _json_seguro(valor: Any) -> Any:
    """Convierte tipos de numpy y pandas a tipos nativos de Python.

    Sin esto el informe no pasa por json.dumps: numpy.int64 y numpy.float64 no
    son serializables, y los NaN producen JSON invalido para muchos lectores.
    """
    if isinstance(valor, dict):
        return {str(k): _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json_seguro(v) for v in valor]
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating, float)):
        numero = float(valor)
        return None if math.isnan(numero) or math.isinf(numero) else numero
    if isinstance(valor, (np.bool_, bool)):
        return bool(valor)
    if isinstance(valor, (pd.Timestamp, dt.datetime, dt.date)):
        return str(valor)
    if valor is pd.NaT or valor is None:
        return None
    return valor


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------


def diagnosticar(
    marco: pd.DataFrame,
    columna_tiempo: str | None = None,
    columna_valor: str | None = None,
    columnas_clave: list[str] | None = None,
    columnas_grupo: list[str] | None = None,
    factor_iqr: float = FACTOR_IQR,
    umbral_z: float = UMBRAL_Z,
    minimo_por_grupo: int = MINIMO_POR_GRUPO,
    top: int = 10,
) -> dict[str, Any]:
    """Diagnostica una serie horaria y devuelve un informe serializable a JSON.

    No modifica el marco ni corrige nada: solo describe lo que hay.

    `columnas_clave` identifica una observacion en una tabla desagregada (por
    ejemplo timestamp + agente + version); sirve para distinguir un duplicado
    real de una desagregacion legitima. `columnas_grupo` separa el calculo de
    atipicos estacionales por entidad, para no comparar agentes entre si.
    """
    if marco.empty:
        return {"meta": {"vacio": True}, "avisos": ["El marco esta vacio."]}

    columna_tiempo = _elegir_columna(marco, CANDIDATAS_TIEMPO, columna_tiempo, "tiempo")
    columna_valor = _elegir_columna(marco, CANDIDATAS_VALOR, columna_valor, "valor")

    momentos = pd.to_datetime(marco[columna_tiempo])
    filas_por_momento = len(marco) / momentos.nunique()
    desagregada = filas_por_momento > UMBRAL_FILAS_POR_MARCA

    avisos: list[str] = []
    if desagregada:
        avisos.append(
            f"La tabla parece desagregada: {filas_por_momento:.1f} filas por marca "
            "de tiempo. Los timestamps repetidos son esperables aqui; lo que "
            "indica un defecto es que se repita la clave completa."
        )
        if not columnas_clave:
            avisos.append(
                "No se paso `columnas_clave`, asi que no se puede distinguir un "
                "duplicado real de la desagregacion."
            )
        if not columnas_grupo:
            avisos.append(
                "No se paso `columnas_grupo`: los atipicos estacionales mezclan "
                "todas las entidades en un mismo grupo (dia de semana, hora), "
                "asi que compararan entre si observaciones no comparables."
            )

    informe = {
        "meta": {
            "generado": dt.datetime.now().isoformat(timespec="seconds"),
            "columna_tiempo": columna_tiempo,
            "columna_valor": columna_valor,
            "forma": "desagregada" if desagregada else "agregada",
            "filas_por_timestamp": round(filas_por_momento, 2),
            "vacio": False,
        },
        "completitud": _completitud(marco, columna_tiempo, columnas_clave, top),
        "valores": _valores(
            marco, columna_tiempo, columna_valor, columnas_grupo,
            factor_iqr, umbral_z, minimo_por_grupo, top,
        ),
        "estructura": _estructura(marco, columna_tiempo, columna_valor, top),
        "avisos": avisos,
    }

    estacional = informe["valores"]["outliers_estacionales"]
    if estacional.get("fiabilidad") == "limitada":
        avisos.append(
            f"Solo hay {estacional['observaciones_por_grupo_mediana']} observaciones "
            f"por grupo (dia de semana, hora). La MAD subestima la dispersion en "
            f"muestras pequenas, asi que los {estacional['n_outliers']} atipicos "
            f"estacionales estan inflados: tratalos como candidatos a revisar, no "
            f"como un recuento firme."
        )
    elif estacional.get("fiabilidad") == "insuficiente":
        avisos.append(
            "No hay suficientes observaciones por grupo (dia de semana, hora) para "
            "juzgar atipicos estacionales. Hacen falta unos cinco meses de historico."
        )

    return _json_seguro(informe)


# --------------------------------------------------------------------------
# Resumen legible
# --------------------------------------------------------------------------

_ANCHO = 74


def _seccion(titulo: str) -> str:
    return f"\n{'-' * _ANCHO}\n{titulo}\n{'-' * _ANCHO}"


def resumen(informe: dict[str, Any]) -> str:
    """Version legible del informe, para leer en terminal o pegar en el anexo."""
    if informe.get("meta", {}).get("vacio"):
        return "El marco esta vacio: no hay nada que diagnosticar."

    meta = informe["meta"]
    comp = informe["completitud"]
    val = informe["valores"]
    est = informe["estructura"]
    lineas: list[str] = []

    lineas.append("=" * _ANCHO)
    lineas.append("DIAGNOSTICO DE CALIDAD")
    lineas.append("=" * _ANCHO)
    lineas.append(f"Generado    : {meta['generado']}")
    lineas.append(f"Forma       : {meta['forma']} ({meta['filas_por_timestamp']} filas/hora)")
    lineas.append(f"Columnas    : tiempo={meta['columna_tiempo']}  valor={meta['columna_valor']}")
    lineas.append(f"Filas       : {est['n_filas']:,}")

    # --- completitud ---
    lineas.append(_seccion("COMPLETITUD TEMPORAL"))
    lineas.append(f"Rango           : {comp['rango']['inicio']} .. {comp['rango']['fin']}")
    lineas.append(
        f"Horas           : {comp['horas_presentes']:,} de {comp['horas_esperadas']:,} "
        f"esperadas ({comp['completitud_pct']}%)"
    )

    huecos = comp["huecos"]
    if comp["horas_faltantes"] == 0:
        lineas.append("Huecos          : ninguno")
    else:
        lineas.append(
            f"Huecos          : {comp['horas_faltantes']:,} horas en "
            f"{huecos['n_tramos']} tramos -> {huecos['forma'].upper()} "
            f"({huecos['horas_por_tramo_media']} horas por tramo)"
        )
        clases = huecos["tramos_por_clase"]
        etiquetas = {
            "aislado": "1 hora suelta",
            "corto": "2-5 horas",
            "medio": "6-24 horas",
            "largo": "mas de un dia",
        }
        for clase, etiqueta in etiquetas.items():
            if clases.get(clase):
                lineas.append(f"  {clases[clase]:>5} tramos de {etiqueta}")
        lineas.append("  tramos mas largos:")
        for tramo in huecos["tramos"][:5]:
            lineas.append(
                f"    {tramo['inicio']} .. {tramo['fin']}  "
                f"({tramo['n_horas']} h, {tramo['clase']})"
            )
        if huecos["n_tramos_no_listados"]:
            lineas.append(f"    ... y {huecos['n_tramos_no_listados']} tramos mas")

    dup = comp["duplicados"]
    lineas.append(f"Timestamps rep. : {dup['timestamps_repetidos']:,}")
    if "clave_repetida" in dup:
        lineas.append(
            f"Clave repetida  : {dup['clave_repetida']:,}  (clave: {', '.join(dup['clave'])})"
        )

    dias = comp["dias_sin_24_horas"]
    if dias["n_dias"] == 0:
        lineas.append("Dias != 24 h    : ninguno")
    else:
        lineas.append(
            f"Dias != 24 h    : {dias['n_dias']} "
            f"({dias['n_dias_interiores']} sin contar los bordes del rango)"
        )
        for entrada in dias["dias"][:5]:
            borde = "  <- borde del rango" if entrada["es_borde_del_rango"] else ""
            lineas.append(
                f"    {entrada['dia']}: {entrada['horas_distintas']} horas{borde}"
            )
        if dias["n_dias_interiores"]:
            lineas.append("    OJO: sin horario de verano, esto es error de parseo.")

    # --- valores ---
    lineas.append(_seccion("VALORES"))
    desc = val["descriptivos"]
    if desc.get("n"):
        lineas.append(
            f"Descriptivos    : n={desc['n']:,}  media={desc['media']:,.1f}  "
            f"mediana={desc['mediana']:,.1f}"
        )
        lineas.append(
            f"                  min={desc['minimo']:,.1f}  max={desc['maximo']:,.1f}"
        )

    con_nulos = {
        c: d for c, d in val["nulos_por_columna"].items() if d["n_nulos"] > 0
    }
    if not con_nulos:
        lineas.append("Nulos           : ninguno en ninguna columna")
    else:
        lineas.append(f"Nulos           : en {len(con_nulos)} columnas")
        for columna, detalle in sorted(
            con_nulos.items(), key=lambda x: -x[1]["n_nulos"]
        )[:8]:
            lineas.append(f"    {columna:<24} {detalle['n_nulos']:>8,}  ({detalle['pct']}%)")

    fis = val["sospechosos_fisicos"]
    lineas.append(
        f"Ceros/negativos : {fis['n_ceros']:,} ceros, {fis['n_negativos']:,} negativos"
    )
    if fis["n_ceros"] or fis["n_negativos"]:
        lineas.append("    fisicamente imposibles en demanda agregada; revisar origen")
        for ejemplo in fis["ejemplos_negativos"][:3]:
            lineas.append(f"    {ejemplo['momento']}: {ejemplo['valor']:,.1f}")
        for ejemplo in fis["ejemplos_ceros"][:3]:
            lineas.append(f"    {ejemplo}: 0")

    iqr = val["outliers_iqr"]
    if iqr.get("aplicable"):
        lineas.append(
            f"Atipicos IQR    : {iqr['n_outliers']:,} ({iqr['pct']}%)  "
            f"fuera de [{iqr['limite_inferior']:,.0f}, {iqr['limite_superior']:,.0f}]"
        )
        lineas.append("                  criterio global: no distingue la hora del dia")

    est_out = val["outliers_estacionales"]
    lineas.append(
        f"Atipicos hora/dia: {est_out['n_outliers']:,} ({est_out['pct']}% de los "
        f"{est_out['n_evaluables']:,} evaluables, |z|>{est_out['umbral_z']})"
    )
    if est_out.get("fiabilidad") != "alta":
        lineas.append(
            f"                  fiabilidad {est_out.get('fiabilidad')}: "
            f"{est_out.get('observaciones_por_grupo_mediana')} observaciones por grupo"
        )
    if est_out["n_no_evaluables"]:
        lineas.append(
            f"                  {est_out['n_no_evaluables']:,} no evaluables "
            "(grupo sin dispersion o con pocas observaciones)"
        )
    if est_out["ejemplos"]:
        lineas.append("    peores desviaciones frente a su (dia de semana, hora):")
        for ejemplo in est_out["ejemplos"][:5]:
            lineas.append(
                f"    {ejemplo['momento']} ({ejemplo['dia_semana']} {ejemplo['hora']:02d}h)"
                f"  valor={ejemplo['valor']:,.0f}  mediana={ejemplo['mediana_grupo']:,.0f}"
                f"  z={ejemplo['z']}"
            )
    if est_out["por_hora_del_dia"]:
        peores_horas = sorted(
            est_out["por_hora_del_dia"].items(), key=lambda x: -x[1]
        )[:5]
        lineas.append(
            "    horas con mas atipicos: "
            + ", ".join(f"{int(h):02d}h ({n})" for h, n in peores_horas)
        )

    # --- estructura ---
    lineas.append(_seccion("ESTRUCTURA"))
    rango = est["rango_temporal"]
    lineas.append(
        f"Cobertura       : {rango['dias_cubiertos']:,} dias de "
        f"{rango['dias_del_rango']:,} en el rango"
    )
    lineas.append(f"Continuidad     : {est['continuidad_pct']}%")
    lineas.append(f"Columnas        : {est['n_columnas']}")

    cardinalidad = est["cardinalidad_categoricas"]
    if not cardinalidad:
        lineas.append("Categoricas     : ninguna")
    else:
        lineas.append("Categoricas     : niveles de desagregacion")
        for columna, detalle in sorted(
            cardinalidad.items(), key=lambda x: -x[1]["n_distintos"]
        ):
            valores = detalle["todos_los_valores"]
            muestra = f"  {valores}" if valores else ""
            lineas.append(f"    {columna:<24} {detalle['n_distintos']:>6} distintos{muestra}")

    if informe["avisos"]:
        lineas.append(_seccion("AVISOS"))
        for aviso in informe["avisos"]:
            lineas.append(f"  - {aviso}")

    lineas.append("")
    lineas.append("Este informe no corrige nada. Solo describe.")
    return "\n".join(lineas)
