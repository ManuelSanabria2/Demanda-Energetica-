"""Ingenieria de variables (Fase A): del panel a la tabla de modelado.

    data/procesado/panel_modelado.parquet
        -> tabla_modelado_r24.parquet    backtesting academico
        -> tabla_modelado_r120.parquet   disponibilidad real

Principio rector: **ninguna variable de la fila t usa informacion que no
estuviera disponible al predecir t**. Lo fija un solo parametro,
`rezago_minimo_h`: cuantas horas separan la hora que se predice de la ultima
hora de demanda publicada en ese momento.

- `rezago_minimo_h=24`: al predecir t se conoce la demanda hasta t-24. Es el
  supuesto academico del README, util para backtesting y no para operar.
- `rezago_minimo_h=120`: la disponibilidad real. La demanda se publica con 3
  dias de rezago y los dos ultimos dias publicados llegan parciales, asi que el
  ultimo dato fiable tiene ~5 dias (notas/hallazgos_calidad.md, seccion 1).

Un rezago menor que `rezago_minimo_h` no se genera, y el registro lo anota. No
hay `lag_1` ni nada por debajo del horizonte de 24 h: con un modelo a 24 h, la
demanda mas reciente que se puede conocer al predecir t es la de t-24.

Los rezagos y las ventanas moviles se calculan **por marca de tiempo**, no por
posicion, y solo despues de comprobar que la rejilla horaria esta completa: con
un hueco, "168 filas atras" dejaria de ser "168 horas atras" sin avisar.

Igual que en la limpieza, cada operacion devuelve `(marco, registro)` y el
registro dice que criterio se aplico y a cuantas filas.

Lo que esta capa **no** hace, a proposito: normalizar, escalar ni imputar con
estadisticos del conjunto completo. Eso va en el modelado, despues de la
particion temporal; hacerlo aqui meteria en el entrenamiento la media y la
varianza del periodo de prueba.

    from variables.construir import construir_tabla
    tabla, registro = construir_tabla(panel, rezago_minimo_h=24)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

# Horizonte del modelo. Ningun rezago puede ser menor: al predecir t con 24 h
# de antelacion, lo mas reciente que se puede conocer es t-24.
HORIZONTE_H = 24

REZAGOS_H = (24, 168)
VENTANA_MOVIL_H = 168

# Las dos versiones de la tabla. Ver el docstring del modulo.
REZAGOS_MINIMOS_DEFECTO = (24, 120)

# Dias finales que se descartan por publicacion parcial.
DIAS_PUBLICACION_PARCIAL = 5

KWH_POR_MWH = 1000.0

COLUMNA_TIEMPO = "fecha_hora"
COLUMNA_KWH = "valor_kwh"
COLUMNA_OBJETIVO = "demanda_mwh"

# Banderas del calendario: pasan de bool a 0/1 sin cambiar su definicion, que
# vive en ingesta.complementarias.DEFINICIONES_CALENDARIO.
BANDERAS_CALENDARIO = (
    "es_festivo",
    "es_vispera_festivo",
    "es_puente",
    "es_semana_santa",
    "es_ultima_semana_diciembre",
)

# Exogenas del panel que entran tal cual, sin transformar.
EXOGENAS_DIRECTAS = ("temperatura_c",)

# Componente del calendario -> periodo. Los valores van de 0 a periodo-1.
PERIODOS_CICLICOS = {"hora": 24, "dia_semana": 7, "mes": 12}

# Columnas del panel que no entran en la tabla, con el motivo. Cualquier columna
# que no este aqui ni sea variable hace fallar la construccion: decidir si una
# columna nueva del panel es variable es una decision, no un descuido.
COLUMNAS_EXCLUIDAS: dict[str, str] = {
    # Fuga temporal: se calculan con la serie entera (lo advierte el README).
    "atipico_global": (
        "fuga temporal: criterio estacional calculado con la serie entera, "
        "futuro incluido"
    ),
    "atipico_iqr_global": (
        "fuga temporal: limites IQR calculados con la serie entera, futuro "
        "incluido"
    ),
    # Fuga del objetivo: son causales respecto al pasado, pero usan la demanda
    # de la propia hora t, que es justo lo que se quiere predecir.
    "z_estacional": (
        "fuga del objetivo: (demanda de t - mediana pasada) / MAD; contiene el "
        "valor que se quiere predecir"
    ),
    "atipico": (
        "fuga del objetivo: |z_estacional| > umbral, calculada con la demanda "
        "de la propia hora t"
    ),
    "atipico_iqr": (
        "fuga del objetivo: compara la demanda de la propia hora t con limites "
        "IQR causales"
    ),
    "atipico_evaluable": (
        "procedencia de la marca de atipicos: dice si el grupo tenia historico "
        "suficiente, no describe la demanda"
    ),
    # Procedencia del objetivo.
    "origen_valor": "procedencia del objetivo (texto: observado / interpolado / faltante)",
    "imputado": "procedencia del objetivo: si el valor de t fue imputado",
    "hueco_horas": "procedencia del objetivo: longitud del hueco al que pertenecia t",
    "periodo_atipico": (
        "marca de limpieza del periodo de pandemia (2020-03..2020-12), no una "
        "bandera del calendario"
    ),
    "etiqueta_periodo": "texto: etiqueta del periodo atipico",
    # Procedencia de las fuentes.
    "fuente": "texto de procedencia de la demanda",
    "metrica": "texto de procedencia de la demanda",
    "entidad": "texto de procedencia de la demanda",
    "version": "version de liquidacion: solo existe en SIMEM, nula en XM",
    "ciudades_disponibles": (
        "procedencia del clima: cuantas ciudades entraron en la media ponderada "
        "de temperatura_c"
    ),
    "nombre_festivo": "texto: nombre del festivo; la variable es es_festivo",
    # Sustituida por el objetivo en MWh.
    "valor_kwh": (
        "sustituida por demanda_mwh = valor_kwh / 1000; ninguna variable la "
        "necesita en kWh"
    ),
}

# Las que nunca pueden aparecer en la tabla final, pase lo que pase.
COLUMNAS_FUGA = ("atipico_global", "atipico_iqr_global", "z_estacional", "atipico", "atipico_iqr")

DIR_SALIDA = Path(__file__).resolve().parents[2] / "data" / "procesado"
NOMBRE_REGISTRO = "registro_variables.json"


class ErrorVariables(Exception):
    """La tabla de modelado no se puede construir con seguridad."""


def _ahora() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _registro(
    operacion: str, criterio: str, antes: int, despues: int, **detalle: Any
) -> dict[str, Any]:
    """Entrada de registro de una operacion, con la misma forma que la limpieza."""
    return {
        "operacion": operacion,
        "criterio": criterio,
        "momento": _ahora(),
        "filas_antes": int(antes),
        "filas_despues": int(despues),
        "filas_afectadas": int(detalle.pop("filas_afectadas", abs(despues - antes))),
        "detalle": detalle,
    }


def nombre_tabla(rezago_minimo_h: int) -> str:
    """Nombre del archivo de cada version: tabla_modelado_r24.parquet, etc."""
    return f"tabla_modelado_r{rezago_minimo_h}.parquet"


# --------------------------------------------------------------------------
# 1. Rejilla horaria
# --------------------------------------------------------------------------


def verificar_rejilla(
    marco: pd.DataFrame, columna_tiempo: str = COLUMNA_TIEMPO
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Comprueba que la serie es una rejilla horaria completa, sin repetidos.

    No corrige nada: si falta una hora o hay una repetida, falla. Rellenar la
    rejilla es trabajo de la limpieza, y hacerlo aqui en silencio esconderia un
    problema de la capa anterior.
    """
    if marco.empty:
        raise ErrorVariables("El panel esta vacio.")
    if columna_tiempo not in marco.columns:
        raise ErrorVariables(f"El panel no tiene la columna {columna_tiempo!r}.")

    antes = len(marco)
    datos = marco.copy()
    momentos = pd.to_datetime(datos[columna_tiempo])
    if momentos.isna().any():
        raise ErrorVariables(f"{int(momentos.isna().sum())} marcas de tiempo nulas.")

    repetidas = int(momentos.duplicated().sum())
    if repetidas:
        raise ErrorVariables(
            f"{repetidas} marcas de tiempo repetidas: un rezago por marca de "
            "tiempo no estaria definido. Deduplica en la limpieza."
        )

    estaba_ordenada = bool(momentos.is_monotonic_increasing)
    datos[columna_tiempo] = momentos
    datos = datos.sort_values(columna_tiempo).reset_index(drop=True)

    presentes = pd.DatetimeIndex(datos[columna_tiempo])
    esperada = pd.date_range(presentes.min(), presentes.max(), freq="h")
    faltan = esperada.difference(presentes)
    fuera = presentes.difference(esperada)
    if len(faltan) or len(fuera):
        raise ErrorVariables(
            f"La rejilla horaria no esta completa: faltan {len(faltan)} horas "
            f"(primeras: {[str(m) for m in faltan[:5]]}) y {len(fuera)} marcas "
            f"no caen en hora en punto (primeras: {[str(m) for m in fuera[:5]]}). "
            "Los rezagos de 24 y 168 h necesitan la rejilla completa."
        )

    registro = _registro(
        "verificar_rejilla",
        "rejilla horaria completa, sin marcas repetidas ni fuera de la hora en punto",
        antes,
        len(datos),
        filas_afectadas=0,
        rango=[str(presentes.min()), str(presentes.max())],
        n_horas=len(presentes),
        zona_horaria=str(presentes.tz),
        estaba_ordenada=estaba_ordenada,
    )
    return datos, registro


# --------------------------------------------------------------------------
# 2. Variable objetivo
# --------------------------------------------------------------------------


def crear_objetivo(
    marco: pd.DataFrame, columna_kwh: str = COLUMNA_KWH
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """`demanda_mwh = valor_kwh / 1000`.

    `valor_kwh` es la energia demandada en la hora, asi que `demanda_mwh` es la
    energia de la hora en MWh, numericamente igual a la potencia media en MW.
    `valor_kwh` se queda en el marco hasta la seleccion de columnas, que la
    excluye con su motivo.
    """
    if columna_kwh not in marco.columns:
        raise ErrorVariables(f"El panel no tiene la columna {columna_kwh!r}.")

    datos = marco.copy()
    datos[COLUMNA_OBJETIVO] = pd.to_numeric(datos[columna_kwh]).astype("float64") / KWH_POR_MWH

    registro = _registro(
        "crear_objetivo",
        f"{COLUMNA_OBJETIVO} = {columna_kwh} / {KWH_POR_MWH:g}",
        len(marco),
        len(datos),
        filas_afectadas=0,
        columna=COLUMNA_OBJETIVO,
        unidad="MWh por hora (equivale a la potencia media en MW durante la hora)",
        columna_origen=columna_kwh,
        unidad_origen="kWh por hora",
        n_nan=int(datos[COLUMNA_OBJETIVO].isna().sum()),
    )
    return datos, registro


# --------------------------------------------------------------------------
# 3. Rezagos y estadisticos moviles
# --------------------------------------------------------------------------


def _validar_rezago_minimo(rezago_minimo_h: int) -> None:
    if isinstance(rezago_minimo_h, bool) or not isinstance(rezago_minimo_h, (int, np.integer)):
        raise ErrorVariables(f"rezago_minimo_h debe ser un entero de horas: {rezago_minimo_h!r}")
    if rezago_minimo_h < HORIZONTE_H:
        raise ErrorVariables(
            f"rezago_minimo_h={rezago_minimo_h} es menor que el horizonte de "
            f"{HORIZONTE_H} h. Con un modelo a {HORIZONTE_H} h la demanda mas "
            f"reciente que se conoce al predecir t es la de t-{HORIZONTE_H}: un "
            "rezago menor (lag_1 incluido) seria fuga temporal."
        )


def _desplazar(serie: pd.Series, horas: int) -> pd.Series:
    """Valor de la serie `horas` horas antes, buscado por marca de tiempo."""
    return serie.shift(freq=pd.Timedelta(hours=horas)).reindex(serie.index)


def crear_rezagos(
    marco: pd.DataFrame,
    rezago_minimo_h: int,
    rezagos: tuple[int, ...] = REZAGOS_H,
    ventana_h: int = VENTANA_MOVIL_H,
    columna_tiempo: str = COLUMNA_TIEMPO,
    columna_objetivo: str = COLUMNA_OBJETIVO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rezagos de la demanda y media y desviacion moviles, todo por marca de tiempo.

    - `lag_k(t) = demanda(t - k h)`, solo si `k >= rezago_minimo_h`.
    - `ma_168(t)` y `std_168(t)`: media y desviacion estandar (ddof=1) de las
      168 horas que acaban en `t - rezago_minimo_h`, es decir de
      `demanda(t - rezago_minimo_h - 167 h) .. demanda(t - rezago_minimo_h)`.
      El desplazamiento va **antes** de la ventana: una ventana que acabase en
      t usaria demanda aun no publicada.

    Se exige la ventana completa (`min_periods = ventana_h`): una media de 3
    horas no es una media de 168, y mezclarlas en la misma columna engana al
    modelo. Las filas iniciales sin historico quedan en NaN y las elimina
    `eliminar_arranque`.
    """
    _validar_rezago_minimo(rezago_minimo_h)
    if any(k <= 0 for k in rezagos) or ventana_h < 2:
        raise ErrorVariables(f"Rezagos {rezagos} o ventana {ventana_h} no validos.")

    datos = marco.copy()
    indice = pd.DatetimeIndex(datos[columna_tiempo])
    serie = pd.Series(datos[columna_objetivo].to_numpy(), index=indice)

    generadas: list[str] = []
    omitidos: list[dict[str, Any]] = []
    definiciones: dict[str, str] = {}

    for k in sorted(rezagos):
        nombre = f"lag_{k}"
        if k < rezago_minimo_h:
            omitidos.append(
                {
                    "columna": nombre,
                    "rezago_h": k,
                    "motivo": (
                        f"{k} h < rezago_minimo_h={rezago_minimo_h} h: al predecir t "
                        f"la demanda de t-{k} h aun no esta disponible"
                    ),
                }
            )
            continue
        datos[nombre] = _desplazar(serie, k).to_numpy()
        generadas.append(nombre)
        definiciones[nombre] = f"{columna_objetivo} en la marca de tiempo t - {k} h"

    # Desplazar primero, luego la ventana: la ventana de t acaba en t - rezago.
    desplazada = serie.shift(freq=pd.Timedelta(hours=rezago_minimo_h))
    ventana = desplazada.rolling(f"{ventana_h}h", min_periods=ventana_h)
    media, desviacion = f"ma_{ventana_h}", f"std_{ventana_h}"
    datos[media] = ventana.mean().reindex(indice).to_numpy()
    datos[desviacion] = ventana.std(ddof=1).reindex(indice).to_numpy()
    generadas += [media, desviacion]

    tramo = (
        f"{columna_objetivo}(t - {rezago_minimo_h + ventana_h - 1} h) .. "
        f"{columna_objetivo}(t - {rezago_minimo_h} h)"
    )
    definiciones[media] = f"media de las {ventana_h} horas {tramo}"
    definiciones[desviacion] = f"desviacion estandar (ddof=1) de las {ventana_h} horas {tramo}"

    registro = _registro(
        "crear_rezagos",
        (
            f"rezagos {list(rezagos)} h solo si >= rezago_minimo_h={rezago_minimo_h}; "
            f"media y desviacion moviles de {ventana_h} h sobre la serie desplazada "
            f"{rezago_minimo_h} h; todo por marca de tiempo"
        ),
        len(marco),
        len(datos),
        filas_afectadas=0,
        rezago_minimo_h=int(rezago_minimo_h),
        horizonte_h=HORIZONTE_H,
        columnas_generadas=generadas,
        rezagos_omitidos=omitidos,
        definiciones=definiciones,
        ventana_h=ventana_h,
        min_periods=ventana_h,
        ddof=1,
        calculo="por marca de tiempo (shift con freq + reindex; ventana movil de tiempo)",
        nan_por_columna={c: int(datos[c].isna().sum()) for c in generadas},
    )
    log.info(
        "Rezagos (r=%d): %s generadas, %d omitidos", rezago_minimo_h, generadas, len(omitidos)
    )
    return datos, registro


# --------------------------------------------------------------------------
# 4. Codificacion ciclica
# --------------------------------------------------------------------------


def codificar_ciclicas(
    marco: pd.DataFrame, columna_tiempo: str = COLUMNA_TIEMPO
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Seno y coseno de la hora, el dia de la semana y el mes.

    Con un entero, la hora 23 y la 0 quedan a 23 unidades; en el circulo quedan
    tan cerca como la 0 y la 1. Los valores se cuentan desde 0 (hora 0..23,
    lunes=0..domingo=6, enero=0..diciembre=11) en hora local de Colombia.
    Son deterministas: se conocen de antemano para cualquier t.
    """
    datos = marco.copy()
    momentos = pd.to_datetime(datos[columna_tiempo])
    componentes = {
        "hora": momentos.dt.hour,
        "dia_semana": momentos.dt.dayofweek,
        "mes": momentos.dt.month - 1,
    }

    generadas: list[str] = []
    definiciones: dict[str, str] = {}
    for nombre, valores in componentes.items():
        periodo = PERIODOS_CICLICOS[nombre]
        angulo = 2.0 * np.pi * valores.to_numpy(dtype="float64") / periodo
        datos[f"{nombre}_sin"] = np.sin(angulo)
        datos[f"{nombre}_cos"] = np.cos(angulo)
        generadas += [f"{nombre}_sin", f"{nombre}_cos"]
        definiciones[nombre] = f"sin y cos de 2*pi*{nombre}/{periodo}"

    registro = _registro(
        "codificar_ciclicas",
        "seno y coseno de hora (24), dia de la semana (7) y mes (12), en hora local",
        len(marco),
        len(datos),
        filas_afectadas=0,
        columnas_generadas=generadas,
        definiciones=definiciones,
        origen={"hora": "0..23", "dia_semana": "lunes=0 .. domingo=6", "mes": "enero=0 .. diciembre=11"},
        zona_horaria=str(momentos.dt.tz),
    )
    return datos, registro


# --------------------------------------------------------------------------
# 5. Banderas del calendario
# --------------------------------------------------------------------------


def banderas_a_enteros(
    marco: pd.DataFrame, banderas: tuple[str, ...] = BANDERAS_CALENDARIO
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Banderas booleanas del calendario a enteros 0/1. No cambia su definicion.

    Falla ante un nulo o un valor que no sea booleano: el calendario cubre el
    100 % de las horas por construccion, y convertir un nulo a 0 diria "no es
    festivo" cuando en realidad no se sabe.
    """
    datos = marco.copy()
    convertidas = [b for b in banderas if b in datos.columns]
    ausentes = [b for b in banderas if b not in datos.columns]

    for bandera in convertidas:
        columna = datos[bandera]
        if columna.isna().any():
            raise ErrorVariables(
                f"{bandera} tiene {int(columna.isna().sum())} nulos: el calendario "
                "no cubre todas las horas."
            )
        valores = set(pd.unique(columna))
        if not valores <= {True, False}:
            raise ErrorVariables(f"{bandera} no es booleana: {sorted(map(str, valores))[:5]}")
        datos[bandera] = columna.astype(bool).astype("int8")

    if ausentes:
        log.warning("Banderas del calendario ausentes del panel: %s", ausentes)

    registro = _registro(
        "banderas_a_enteros",
        "banderas booleanas del calendario a int8 0/1, sin cambiar su definicion",
        len(marco),
        len(datos),
        filas_afectadas=0,
        convertidas=convertidas,
        ausentes=ausentes,
        unos_por_bandera={b: int(datos[b].sum()) for b in convertidas},
        definiciones="ingesta.complementarias.DEFINICIONES_CALENDARIO (sin cambios)",
    )
    return datos, registro


# --------------------------------------------------------------------------
# 6. Seleccion de columnas
# --------------------------------------------------------------------------


def seleccionar_columnas(
    marco: pd.DataFrame,
    variables: list[str],
    columna_tiempo: str = COLUMNA_TIEMPO,
    columna_objetivo: str = COLUMNA_OBJETIVO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deja la marca de tiempo, el objetivo y las variables; excluye el resto.

    Cada columna excluida queda en el registro con su motivo. Una columna que no
    sea variable ni figure en `COLUMNAS_EXCLUIDAS` hace fallar la seleccion: si
    el panel gana una columna nueva, decidir si es variable es una decision que
    no debe tomarse en silencio.
    """
    conservadas = [columna_tiempo, columna_objetivo] + [
        c for c in variables if c in marco.columns
    ]
    excluidas = [c for c in marco.columns if c in COLUMNAS_EXCLUIDAS]
    sin_clasificar = [c for c in marco.columns if c not in conservadas and c not in excluidas]
    if sin_clasificar:
        raise ErrorVariables(
            f"Columnas del panel sin clasificar: {sin_clasificar}. Anadelas a las "
            "variables o a COLUMNAS_EXCLUIDAS con su motivo."
        )
    faltan = [c for c in variables if c not in marco.columns]
    if faltan:
        raise ErrorVariables(f"Variables pedidas que no estan en el marco: {faltan}")

    resultado = marco[conservadas].copy()
    fuga = [c for c in COLUMNAS_FUGA if c in resultado.columns]
    if fuga:  # pragma: no cover - defensa final
        raise ErrorVariables(f"Columnas con fuga en la tabla final: {fuga}")

    # Lo que dicen las columnas de procedencia antes de quitarlas.
    procedencia: dict[str, Any] = {}
    if "imputado" in marco.columns:
        procedencia["objetivo_imputado"] = int(marco["imputado"].sum())
    if "origen_valor" in marco.columns:
        procedencia["origen_valor"] = {
            str(k): int(v) for k, v in marco["origen_valor"].value_counts().items()
        }

    registro = _registro(
        "seleccionar_columnas",
        "marca de tiempo + objetivo + variables; el resto se excluye con su motivo",
        len(marco),
        len(resultado),
        filas_afectadas=0,
        columnas_conservadas=conservadas,
        columnas_excluidas={
            c: {
                "motivo": COLUMNAS_EXCLUIDAS[c],
                "valores_distintos": int(marco[c].nunique(dropna=False)),
            }
            for c in excluidas
        },
        procedencia_del_objetivo=procedencia,
    )
    return resultado, registro


# --------------------------------------------------------------------------
# 7. Filas eliminadas
# --------------------------------------------------------------------------


def eliminar_arranque(
    marco: pd.DataFrame,
    columnas_historicas: list[str],
    columna_tiempo: str = COLUMNA_TIEMPO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Elimina el bloque inicial de filas sin historico para los rezagos.

    Solo el bloque inicial. Un NaN en mitad de la serie (un hueco largo que la
    limpieza dejo como NaN) no se elimina aqui: se reporta, y quitarlo o no es
    decision del modelado.
    """
    completas = marco[columnas_historicas].notna().all(axis=1).to_numpy()
    if not completas.any():
        raise ErrorVariables(
            f"Ninguna fila tiene historico completo para {columnas_historicas}: "
            "la serie es mas corta que el rezago mas largo."
        )
    primera = int(np.argmax(completas))
    resultado = marco.iloc[primera:].reset_index(drop=True)

    nan_interiores = {
        c: int(resultado[c].isna().sum()) for c in resultado.columns if resultado[c].isna().any()
    }
    if nan_interiores:
        log.warning("NaN fuera del arranque, conservados: %s", nan_interiores)

    momentos = marco[columna_tiempo]
    registro = _registro(
        "eliminar_arranque",
        f"se eliminan las filas iniciales hasta la primera con {columnas_historicas} completas",
        len(marco),
        len(resultado),
        filas_afectadas=primera,
        rango_eliminado=(
            [str(momentos.iloc[0]), str(momentos.iloc[primera - 1])] if primera else None
        ),
        primera_fila_conservada=str(momentos.iloc[primera]),
        nan_fuera_del_arranque=nan_interiores,
        nota=(
            "Solo se elimina el bloque inicial. Los NaN posteriores, si los hay, "
            "se conservan y se reportan aqui."
        ),
    )
    return resultado, registro


def descartar_publicacion_parcial(
    marco: pd.DataFrame,
    dias: int = DIAS_PUBLICACION_PARCIAL,
    columna_tiempo: str = COLUMNA_TIEMPO,
    columna_objetivo: str = COLUMNA_OBJETIVO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Elimina los ultimos `dias` dias de calendario de la serie.

    Los dos ultimos dias publicados llegan al ~24 % de su valor aunque tengan
    las 24 horas y ningun nulo, y el rezago util es de ~5 dias
    (notas/hallazgos_calidad.md, seccion 1). Se cuentan dias locales: un ultimo
    dia incompleto cuenta igual como uno de los cinco.

    El registro guarda el total de cada dia descartado frente a la mediana de
    los 28 dias anteriores, para que se vea que se descarto y por que.
    """
    if dias < 0:
        raise ErrorVariables(f"dias debe ser >= 0: {dias}")

    momentos = pd.to_datetime(marco[columna_tiempo])
    fechas = momentos.dt.date
    if dias == 0:
        descartar = pd.Series(False, index=marco.index)
    else:
        corte = fechas.max() - dt.timedelta(days=dias - 1)
        descartar = fechas >= corte

    resultado = marco[~descartar].reset_index(drop=True)
    if resultado.empty:
        raise ErrorVariables(f"Descartar {dias} dias deja la tabla vacia.")

    totales = marco.groupby(fechas)[columna_objetivo].sum()
    descartadas = sorted(set(fechas[descartar]))
    referencia = totales[[f for f in totales.index if f not in descartadas]].tail(28).median()
    diario = {
        str(f): {
            "total_mwh": round(float(totales[f]), 1),
            "pct_frente_mediana_28d": round(100 * float(totales[f]) / referencia, 1)
            if referencia
            else None,
        }
        for f in descartadas
    }

    registro = _registro(
        "descartar_publicacion_parcial",
        f"se eliminan los ultimos {dias} dias de calendario por publicacion parcial",
        len(marco),
        len(resultado),
        filas_afectadas=int(descartar.sum()),
        dias=dias,
        rango_eliminado=(
            [str(momentos[descartar].min()), str(momentos[descartar].max())]
            if descartar.any()
            else None
        ),
        dias_eliminados=diario,
        motivo=(
            "Los ultimos dias publicados llegan parciales (~24 % de lo normal) "
            "aunque esten completos en horas; el rezago util es de ~5 dias. Ver "
            "notas/hallazgos_calidad.md, seccion 1."
        ),
    )
    return resultado, registro


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------


def construir_tabla(
    panel: pd.DataFrame,
    rezago_minimo_h: int,
    dias_parciales: int = DIAS_PUBLICACION_PARCIAL,
    columna_tiempo: str = COLUMNA_TIEMPO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Del panel de modelado a la tabla de modelado de una version de rezago.

    Devuelve (tabla, registro). El numero de filas solo cambia por las dos
    eliminaciones documentadas (arranque sin historico y publicacion parcial);
    cualquier otra diferencia hace fallar la construccion.
    """
    _validar_rezago_minimo(rezago_minimo_h)
    operaciones: list[dict[str, Any]] = []
    filas_panel = len(panel)

    marco, paso = verificar_rejilla(panel, columna_tiempo)
    operaciones.append(paso)

    marco, paso = crear_objetivo(marco)
    operaciones.append(paso)

    marco, paso = crear_rezagos(marco, rezago_minimo_h, columna_tiempo=columna_tiempo)
    operaciones.append(paso)
    historicas = paso["detalle"]["columnas_generadas"]
    omitidos = paso["detalle"]["rezagos_omitidos"]

    marco, paso = codificar_ciclicas(marco, columna_tiempo)
    operaciones.append(paso)
    ciclicas = paso["detalle"]["columnas_generadas"]

    marco, paso = banderas_a_enteros(marco)
    operaciones.append(paso)
    banderas = paso["detalle"]["convertidas"]

    exogenas = [c for c in EXOGENAS_DIRECTAS if c in marco.columns]
    variables = historicas + exogenas + ciclicas + banderas
    marco, paso = seleccionar_columnas(marco, variables, columna_tiempo)
    operaciones.append(paso)
    excluidas = paso["detalle"]["columnas_excluidas"]

    marco, paso = eliminar_arranque(marco, historicas, columna_tiempo)
    operaciones.append(paso)
    n_arranque = paso["filas_afectadas"]

    marco, paso = descartar_publicacion_parcial(marco, dias_parciales, columna_tiempo)
    operaciones.append(paso)
    n_parcial = paso["filas_afectadas"]

    if len(marco) != filas_panel - n_arranque - n_parcial:  # pragma: no cover
        raise ErrorVariables(
            f"Balance de filas roto: {filas_panel} - {n_arranque} - {n_parcial} != {len(marco)}"
        )

    registro = {
        "version_formato": 1,
        "momento": _ahora(),
        "rezago_minimo_h": int(rezago_minimo_h),
        "horizonte_h": HORIZONTE_H,
        "archivo": nombre_tabla(rezago_minimo_h),
        "filas_panel": int(filas_panel),
        "filas_finales": int(len(marco)),
        "n_columnas": len(marco.columns),
        "balance_filas": {
            "panel": int(filas_panel),
            "arranque_sin_historico": int(n_arranque),
            "publicacion_parcial": int(n_parcial),
            "final": int(len(marco)),
        },
        "rango": [str(marco[columna_tiempo].min()), str(marco[columna_tiempo].max())],
        "objetivo": {"columna": COLUMNA_OBJETIVO, "unidad": "MWh por hora"},
        "variables": variables,
        "rezagos_omitidos": omitidos,
        "columnas_excluidas": {c: d["motivo"] for c, d in excluidas.items()},
        "operaciones": operaciones,
        "politica": {
            "fuga_temporal": (
                f"toda variable historica de la fila t usa demanda de t - "
                f"{rezago_minimo_h} h o antes; las marcas de atipicos se excluyen "
                "porque usan la demanda de la propia hora t (o la serie entera)"
            ),
            "escalado": (
                "ninguno: normalizar, escalar e imputar se hace en el modelado, "
                "despues de la particion temporal"
            ),
            "nan": "solo se elimina el arranque; un NaN posterior se conserva y se reporta",
            "temperatura": (
                "temperatura_c es la observada en la propia hora t (Open-Meteo "
                "archive). En backtesting equivale a suponer un pronostico "
                "meteorologico perfecto; para operar habria que sustituirla por un "
                "pronostico emitido antes del momento de prediccion."
            ),
        },
    }
    return marco, registro


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------


def _sha256(ruta: Path) -> str:
    digesto = hashlib.sha256()
    with ruta.open("rb") as origen:
        for bloque in iter(lambda: origen.read(1 << 20), b""):
            digesto.update(bloque)
    return digesto.hexdigest()


def guardar(
    versiones: dict[int, tuple[pd.DataFrame, dict[str, Any]]],
    directorio: Path | None = None,
    contexto: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Escribe una tabla por version y un registro comun para todas.

    `versiones` es {rezago_minimo_h: (tabla, registro)}. El registro se
    reescribe en cada ejecucion: describe las tablas que hay en disco, con el
    SHA-256 de cada una para poder comprobar despues que son esas.
    """
    destino = directorio or DIR_SALIDA
    destino.mkdir(parents=True, exist_ok=True)

    rutas: dict[str, str] = {}
    entradas: dict[str, Any] = {}
    for rezago, (tabla, registro) in sorted(versiones.items()):
        ruta = destino / nombre_tabla(rezago)
        tabla.to_parquet(ruta, index=False)
        rutas[f"r{rezago}"] = str(ruta)
        entradas[f"r{rezago}"] = {**registro, "sha256": _sha256(ruta)}

    bitacora = {
        "version_formato": 1,
        "momento": _ahora(),
        "panel": contexto or {},
        "versiones": entradas,
    }
    ruta_registro = destino / NOMBRE_REGISTRO
    temporal = ruta_registro.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as salida:
        json.dump(bitacora, salida, ensure_ascii=False, indent=2, default=str)
    temporal.replace(ruta_registro)
    rutas["registro"] = str(ruta_registro)

    log.info("Guardadas %d versiones y registro en %s", len(versiones), ruta_registro)
    return rutas
