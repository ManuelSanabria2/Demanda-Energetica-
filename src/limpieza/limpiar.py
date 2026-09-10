"""Limpieza trazable de la serie horaria.

Principio rector: **toda transformacion queda registrada**. Cada operacion
devuelve `(marco, registro)`, donde el registro dice que criterio se aplico,
cuantas filas afecto y con que resultado. El registro completo se guarda en
`data/procesado/registro_limpieza.json`.

Y para cualquier celda concreta se puede responder de donde salio, porque el
marco resultante lleva la procedencia fila a fila:

    origen_valor   "observado" | "interpolado" | "faltante"
    imputado       bool
    hueco_horas    duracion del tramo faltante al que pertenecia la fila
    atipico        marcado, nunca eliminado
    periodo_atipico  marzo-diciembre de 2020

    from limpieza.limpiar import limpiar, procedencia
    marco, registro = limpiar(crudo)
    procedencia(marco, "2025-03-06 04:00")   # explica una celda

Politica de imputacion: **conservadora a proposito**. Solo se interpolan huecos
de hasta 3 horas. Un hueco mayor se queda como NaN y se marca, porque un NaN
honesto se distingue del dato real y un valor inventado ya no.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from calidad.diagnostico import (
    FACTOR_IQR,
    MINIMO_POR_GRUPO,
    UMBRAL_Z,
    limites_iqr,
    z_estacional,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

# Colombia no aplica horario de verano, asi que un desplazamiento fijo de -5 es
# exacto y no depende de la base de datos de zonas horarias del sistema.
ZONA_COLOMBIA = dt.timezone(dt.timedelta(hours=-5), name="UTC-05:00")

# Maximo de horas consecutivas que se interpolan. Por encima, NaN honesto.
MAX_HORAS_INTERPOLACION = 3

# Evento estructural no reproducible: los confinamientos de 2020 cambiaron el
# perfil de demanda de forma que no volvera a repetirse igual. Se marca para
# que el modelado pueda aislarlo, no se elimina.
PANDEMIA_INICIO = dt.date(2020, 3, 1)
PANDEMIA_FIN = dt.date(2020, 12, 31)

COLUMNAS_PROCEDENCIA = ["origen_valor", "imputado", "hueco_horas", "atipico"]

DIR_SALIDA = Path(__file__).resolve().parents[2] / "data" / "procesado"
NOMBRE_REGISTRO = "registro_limpieza.json"


class ErrorLimpieza(Exception):
    """La limpieza no se puede aplicar con seguridad sobre este marco."""


def _ahora() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _registro(
    operacion: str, criterio: str, antes: int, despues: int, **detalle: Any
) -> dict[str, Any]:
    """Construye la entrada de registro de una operacion."""
    return {
        "operacion": operacion,
        "criterio": criterio,
        "momento": _ahora(),
        "filas_antes": int(antes),
        "filas_despues": int(despues),
        "filas_afectadas": int(detalle.pop("filas_afectadas", abs(despues - antes))),
        "detalle": detalle,
    }


# --------------------------------------------------------------------------
# 1. Normalizacion de esquema
# --------------------------------------------------------------------------


def a_snake_case(nombre: str) -> str:
    """Convierte un nombre de columna a snake_case.

    Maneja los tres estilos que llegan de las fuentes: `fecha_hora` (ya bien),
    `CodigoSICAgente` (CamelCase con siglas) y `Valor` (capitalizado).
    """
    texto = str(nombre).strip()
    # Separa la ultima mayuscula de una sigla de la palabra que la sigue:
    # CodigoSICAgente -> CodigoSIC_Agente
    texto = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", texto)
    # Separa minuscula o digito seguido de mayuscula: FechaHora -> Fecha_Hora
    texto = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", texto)
    texto = re.sub(r"[\s\-.]+", "_", texto)
    texto = re.sub(r"_+", "_", texto)
    return texto.strip("_").lower()


def normalizar_esquema(
    marco: pd.DataFrame,
    columna_tiempo: str | None = None,
    columna_valor: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nombres en snake_case, marca de tiempo en UTC-5 explicita y tipos fijos.

    La marca de tiempo se localiza en UTC-5, no se convierte: los datos de XM y
    SIMEM ya vienen en hora local de Colombia, asi que lo que falta es declarar
    esa zona, no desplazar los valores.
    """
    antes = len(marco)
    columna_tiempo = columna_tiempo or _detectar(marco, ("fecha_hora", "timestamp", "FechaHora"))
    columna_valor = columna_valor or _detectar(marco, ("valor_kwh", "valor", "Valor"))

    renombres = {c: a_snake_case(c) for c in marco.columns}
    colisiones = [n for n in set(renombres.values()) if list(renombres.values()).count(n) > 1]
    if colisiones:
        raise ErrorLimpieza(
            f"El paso a snake_case colisiona en {sorted(set(colisiones))}: "
            f"dos columnas distintas quedarian con el mismo nombre."
        )

    limpio = marco.rename(columns=renombres).copy()
    tiempo = a_snake_case(columna_tiempo)
    valor = a_snake_case(columna_valor)

    momentos = pd.to_datetime(limpio[tiempo])
    ya_tenia_zona = momentos.dt.tz is not None
    if ya_tenia_zona:
        momentos = momentos.dt.tz_convert(ZONA_COLOMBIA)
    else:
        momentos = momentos.dt.tz_localize(ZONA_COLOMBIA)
    limpio[tiempo] = momentos

    limpio[valor] = pd.to_numeric(limpio[valor], errors="coerce")
    no_numericos = int(limpio[valor].isna().sum() - marco[columna_valor].isna().sum())

    # Las categoricas a texto, para que el Parquet no arrastre dtypes distintos
    # segun de que particion venga cada trozo.
    categoricas = [
        c
        for c in limpio.columns
        if c not in (tiempo, valor)
        and not pd.api.types.is_numeric_dtype(limpio[c])
        and not pd.api.types.is_datetime64_any_dtype(limpio[c])
    ]
    for columna in categoricas:
        limpio[columna] = limpio[columna].astype("object").where(limpio[columna].notna(), None)

    registro = _registro(
        "normalizar_esquema",
        "nombres a snake_case; marca de tiempo localizada en UTC-5; valor a float64",
        antes,
        len(limpio),
        filas_afectadas=0,
        columna_tiempo=tiempo,
        columna_valor=valor,
        renombradas={k: v for k, v in renombres.items() if k != v},
        zona_horaria=str(ZONA_COLOMBIA),
        zona_ya_presente=bool(ya_tenia_zona),
        valores_no_numericos_a_nan=max(0, no_numericos),
        columnas_normalizadas_a_texto=categoricas,
    )
    return limpio, registro


def _detectar(marco: pd.DataFrame, candidatas: tuple[str, ...]) -> str:
    """Primera columna presente de entre las candidatas."""
    for candidata in candidatas:
        if candidata in marco.columns:
            return candidata
    raise ErrorLimpieza(
        f"No se encontro ninguna de {candidatas} en {list(marco.columns)}."
    )


# --------------------------------------------------------------------------
# 2. Deduplicacion
# --------------------------------------------------------------------------


def deduplicar(
    marco: pd.DataFrame,
    columna_tiempo: str = "fecha_hora",
    columna_recencia: str | None = None,
    permitir_desagregada: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deja una fila por marca de tiempo, conservando la mas reciente.

    "Mas reciente" significa, por defecto, la ultima en el orden actual del
    marco. Si hay una columna que ordena por recencia de verdad (una version de
    liquidacion, una fecha de descarga), pasala en `columna_recencia`.

    Sobre una tabla desagregada esto seria destructivo: en SIMEM hay cientos de
    filas legitimas por hora, y deduplicar por marca de tiempo se cargaria el
    99% de los datos. Por eso se detecta y se rechaza salvo confirmacion
    explicita.
    """
    antes = len(marco)
    momentos = pd.to_datetime(marco[columna_tiempo])
    filas_por_momento = antes / momentos.nunique() if momentos.nunique() else 0

    if filas_por_momento > 1.5 and not permitir_desagregada:
        raise ErrorLimpieza(
            f"El marco tiene {filas_por_momento:.1f} filas por marca de tiempo: "
            "parece desagregado (varias entidades o versiones por hora). "
            "Deduplicar por marca de tiempo eliminaria datos legitimos. Agrega "
            "primero a una fila por hora, o pasa permitir_desagregada=True si "
            "de verdad es lo que quieres."
        )

    ordenado = marco
    if columna_recencia is not None:
        if columna_recencia not in marco.columns:
            raise ErrorLimpieza(
                f"La columna de recencia {columna_recencia!r} no esta en el marco."
            )
        ordenado = marco.sort_values(columna_recencia, kind="stable")

    duplicadas = ordenado.duplicated(subset=[columna_tiempo], keep="last")
    ejemplos = [
        str(m) for m in pd.to_datetime(ordenado.loc[duplicadas, columna_tiempo]).unique()[:10]
    ]
    resultado = ordenado[~duplicadas].sort_values(columna_tiempo).reset_index(drop=True)

    registro = _registro(
        "deduplicar",
        (
            f"una fila por {columna_tiempo}, conservando la ultima segun "
            + (f"{columna_recencia}" if columna_recencia else "el orden de llegada")
        ),
        antes,
        len(resultado),
        filas_afectadas=int(duplicadas.sum()),
        eliminadas=int(duplicadas.sum()),
        columna_recencia=columna_recencia,
        marcas_de_tiempo_afectadas=ejemplos,
        filas_por_marca_antes=round(filas_por_momento, 2),
    )
    log.info("Deduplicacion: %d filas eliminadas", duplicadas.sum())
    return resultado, registro


# --------------------------------------------------------------------------
# 3. Huecos: rejilla completa e interpolacion acotada
# --------------------------------------------------------------------------


def completar_rejilla(
    marco: pd.DataFrame,
    columna_tiempo: str = "fecha_hora",
    columna_valor: str = "valor_kwh",
    max_horas: int = MAX_HORAS_INTERPOLACION,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Completa la rejilla horaria e interpola solo los huecos cortos.

    Un hueco de hasta `max_horas` se interpola linealmente en el tiempo. Uno
    mayor se queda como NaN: rellenar dias enteros seria inventar el perfil de
    demanda de esos dias, y despues no habria forma de distinguirlo del dato
    real.

    Tampoco se extrapola en los extremos: si la serie empieza o acaba con
    huecos, se quedan como estan.
    """
    antes = len(marco)
    datos = marco.copy()
    datos[columna_tiempo] = pd.to_datetime(datos[columna_tiempo])
    datos = datos.sort_values(columna_tiempo)

    rejilla = pd.date_range(
        datos[columna_tiempo].min(), datos[columna_tiempo].max(), freq="h"
    )
    completo = (
        datos.set_index(columna_tiempo)
        .reindex(rejilla)
        .rename_axis(columna_tiempo)
        .reset_index()
    )
    filas_creadas = len(completo) - antes

    faltante = completo[columna_valor].isna()

    # Longitud del tramo de NaN al que pertenece cada fila.
    tramo = (faltante != faltante.shift()).cumsum()
    longitud = faltante.groupby(tramo).transform("sum").where(faltante, 0).astype(int)

    interpolable = faltante & (longitud <= max_horas)

    # limit_area="inside" impide extrapolar antes del primer dato o despues del
    # ultimo, que seria inventar fuera del rango observado.
    serie = completo.set_index(columna_tiempo)[columna_valor]
    interpolada = serie.interpolate(method="time", limit_area="inside")
    valores_interpolados = interpolada.to_numpy()

    completo[columna_valor] = completo[columna_valor].where(
        ~interpolable, valores_interpolados
    )
    # Un hueco corto pegado al borde no se puede interpolar: sigue siendo NaN.
    interpolado_real = interpolable & completo[columna_valor].notna()

    completo["origen_valor"] = "observado"
    completo.loc[interpolado_real, "origen_valor"] = "interpolado"
    completo.loc[completo[columna_valor].isna(), "origen_valor"] = "faltante"
    completo["imputado"] = interpolado_real
    completo["hueco_horas"] = longitud

    por_clase = (
        pd.Series(longitud[faltante]).value_counts().sort_index().to_dict()
        if faltante.any()
        else {}
    )
    tramos_largos = _tramos_no_imputados(completo, columna_tiempo, max_horas)

    registro = _registro(
        "completar_rejilla",
        (
            f"rejilla horaria completa; interpolacion temporal solo para huecos "
            f"de hasta {max_horas} h; sin extrapolar en los extremos"
        ),
        antes,
        len(completo),
        filas_afectadas=int(interpolado_real.sum()),
        max_horas_interpolacion=max_horas,
        filas_creadas_para_completar_rejilla=int(filas_creadas),
        horas_faltantes_totales=int(faltante.sum()),
        horas_interpoladas=int(interpolado_real.sum()),
        horas_dejadas_como_nan=int(completo[columna_valor].isna().sum()),
        huecos_por_longitud={str(k): int(v // k) for k, v in por_clase.items() if k},
        tramos_no_imputados=tramos_largos[:20],
        n_tramos_no_imputados=len(tramos_largos),
        nota=(
            "Los huecos mayores al limite se dejan como NaN a proposito. Un NaN "
            "se distingue del dato real; un valor inventado, no."
        ),
    )
    log.info(
        "Huecos: %d horas faltantes, %d interpoladas, %d dejadas como NaN",
        faltante.sum(),
        interpolado_real.sum(),
        completo[columna_valor].isna().sum(),
    )
    return completo, registro


def _tramos_no_imputados(
    completo: pd.DataFrame, columna_tiempo: str, max_horas: int
) -> list[dict[str, Any]]:
    """Lista los tramos que se dejaron sin imputar, para poder auditarlos."""
    largos = completo[completo["hueco_horas"] > max_horas]
    if largos.empty:
        return []

    momentos = pd.to_datetime(largos[columna_tiempo])
    corte = (momentos.diff() != pd.Timedelta(hours=1)).cumsum()
    tramos = []
    for _, bloque in momentos.groupby(corte):
        tramos.append(
            {
                "inicio": str(bloque.iloc[0]),
                "fin": str(bloque.iloc[-1]),
                "n_horas": int(len(bloque)),
            }
        )
    return sorted(tramos, key=lambda t: -t["n_horas"])


# --------------------------------------------------------------------------
# 4. Atipicos: marcar, nunca eliminar
# --------------------------------------------------------------------------


def marcar_atipicos(
    marco: pd.DataFrame,
    columna_tiempo: str = "fecha_hora",
    columna_valor: str = "valor_kwh",
    columnas_grupo: list[str] | None = None,
    umbral_z: float = UMBRAL_Z,
    factor_iqr: float = FACTOR_IQR,
    minimo_por_grupo: int = MINIMO_POR_GRUPO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Anade columnas de marcado de atipicos. No elimina ni corrige ningun valor.

    Usa los mismos criterios que el diagnostico, importando su implementacion en
    vez de repetirla: si el criterio cambia, cambia en los dos sitios a la vez.

    `atipico` se decide por el criterio estacional, que es el que distingue un
    valor alto a las 2 p.m. (normal) de uno igual a las 3 a.m. (no lo es). El
    criterio global por IQR se marca aparte, como referencia.

    Excluir o no las filas marcadas es decision del modelado, no de la limpieza.
    """
    antes = len(marco)
    datos = marco.copy()
    valores = pd.to_numeric(datos[columna_valor], errors="coerce")

    z, no_evaluable, contexto = z_estacional(
        datos, columna_tiempo, columna_valor, columnas_grupo, minimo_por_grupo
    )
    bajo, alto = limites_iqr(valores, factor_iqr)

    datos["z_estacional"] = z
    datos["atipico"] = (z.abs() > umbral_z).fillna(False)
    datos["atipico_iqr"] = ((valores < bajo) | (valores > alto)).fillna(False)
    datos["atipico_evaluable"] = ~no_evaluable

    observaciones = contexto["tamano_grupo"][~valores.isna()]
    mediana_grupo = float(observaciones.median()) if len(observaciones) else 0.0
    fiabilidad = (
        "alta"
        if mediana_grupo >= 30
        else ("limitada" if mediana_grupo >= minimo_por_grupo else "insuficiente")
    )

    registro = _registro(
        "marcar_atipicos",
        (
            f"columna atipico = |z modificada frente a (dia de semana, hora)| > "
            f"{umbral_z}; atipico_iqr = fuera de {factor_iqr} rangos "
            "intercuartilicos. Marcado, sin eliminar ni corregir."
        ),
        antes,
        len(datos),
        filas_afectadas=int(datos["atipico"].sum()),
        n_atipicos_estacionales=int(datos["atipico"].sum()),
        n_atipicos_iqr=int(datos["atipico_iqr"].sum()),
        n_no_evaluables=int(no_evaluable.sum()),
        umbral_z=umbral_z,
        factor_iqr=factor_iqr,
        limite_iqr_inferior=bajo,
        limite_iqr_superior=alto,
        observaciones_por_grupo_mediana=round(mediana_grupo, 1),
        fiabilidad=fiabilidad,
        columnas_grupo=list(columnas_grupo or []),
        nota="La decision de excluirlos corresponde al modelado, no a la limpieza.",
    )
    log.info(
        "Atipicos: %d por criterio estacional, %d por IQR (marcados, no eliminados)",
        datos["atipico"].sum(),
        datos["atipico_iqr"].sum(),
    )
    return datos, registro


# --------------------------------------------------------------------------
# 5. Periodo atipico: pandemia
# --------------------------------------------------------------------------


def marcar_periodo_atipico(
    marco: pd.DataFrame,
    columna_tiempo: str = "fecha_hora",
    inicio: dt.date = PANDEMIA_INICIO,
    fin: dt.date = PANDEMIA_FIN,
    etiqueta: str = "pandemia_2020",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Marca un periodo estructuralmente distinto para que el modelado lo aisle.

    Los confinamientos de 2020 cambiaron el perfil horario de demanda de una
    forma que no se va a repetir. No se eliminan las filas: se marcan, y el
    modelado decide si las excluye, las pondera o las modela aparte.
    """
    antes = len(marco)
    datos = marco.copy()
    momentos = pd.to_datetime(datos[columna_tiempo])
    fechas = momentos.dt.date

    dentro = (fechas >= inicio) & (fechas <= fin)
    datos["periodo_atipico"] = dentro
    datos["etiqueta_periodo"] = pd.Series([None] * len(datos), dtype="object")
    datos.loc[dentro, "etiqueta_periodo"] = etiqueta

    registro = _registro(
        "marcar_periodo_atipico",
        f"periodo_atipico = fecha entre {inicio} y {fin} ({etiqueta})",
        antes,
        len(datos),
        filas_afectadas=int(dentro.sum()),
        etiqueta=etiqueta,
        rango=[inicio.isoformat(), fin.isoformat()],
        n_filas_marcadas=int(dentro.sum()),
        cubre_el_periodo=bool(dentro.any()),
        nota=(
            "Evento estructural no reproducible. Se marca, no se elimina: aislarlo "
            "o no es decision del modelado."
        ),
    )
    if not dentro.any():
        log.info(
            "El periodo %s..%s no se solapa con los datos (%s..%s): 0 filas marcadas",
            inicio, fin, fechas.min(), fechas.max(),
        )
    return datos, registro


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------


def limpiar(
    marco: pd.DataFrame,
    columna_tiempo: str | None = None,
    columna_valor: str | None = None,
    columna_recencia: str | None = None,
    max_horas_interpolacion: int = MAX_HORAS_INTERPOLACION,
    columnas_grupo: list[str] | None = None,
    permitir_desagregada: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica la limpieza completa y devuelve (marco, registro).

    El registro lleva una entrada por operacion, en orden, con el criterio
    aplicado y las filas afectadas.
    """
    if marco.empty:
        raise ErrorLimpieza("El marco esta vacio: no hay nada que limpiar.")

    operaciones: list[dict[str, Any]] = []
    filas_iniciales = len(marco)

    limpio, paso = normalizar_esquema(marco, columna_tiempo, columna_valor)
    operaciones.append(paso)
    tiempo = paso["detalle"]["columna_tiempo"]
    valor = paso["detalle"]["columna_valor"]
    recencia = a_snake_case(columna_recencia) if columna_recencia else None
    grupos = [a_snake_case(c) for c in columnas_grupo] if columnas_grupo else None

    limpio, paso = deduplicar(limpio, tiempo, recencia, permitir_desagregada)
    operaciones.append(paso)

    limpio, paso = completar_rejilla(limpio, tiempo, valor, max_horas_interpolacion)
    operaciones.append(paso)

    limpio, paso = marcar_atipicos(limpio, tiempo, valor, grupos)
    operaciones.append(paso)

    limpio, paso = marcar_periodo_atipico(limpio, tiempo)
    operaciones.append(paso)

    registro = {
        "version_formato": 1,
        "momento": _ahora(),
        "filas_iniciales": int(filas_iniciales),
        "filas_finales": int(len(limpio)),
        "columna_tiempo": tiempo,
        "columna_valor": valor,
        "operaciones": operaciones,
        "procedencia": _resumen_procedencia(limpio, valor),
        "politica": {
            "imputacion": (
                f"solo huecos de hasta {max_horas_interpolacion} h, por "
                "interpolacion temporal, sin extrapolar en los extremos"
            ),
            "atipicos": "marcados, nunca eliminados ni corregidos",
            "periodos_estructurales": "marcados, nunca eliminados",
            "nan": "se prefiere un NaN honesto a un valor inventado",
        },
    }
    return limpio, registro


def _resumen_procedencia(marco: pd.DataFrame, columna_valor: str) -> dict[str, Any]:
    """Recuento de donde sale cada valor del marco final."""
    conteo = marco["origen_valor"].value_counts().to_dict()
    total = len(marco)
    return {
        "por_origen": {str(k): int(v) for k, v in conteo.items()},
        "pct_observado": round(100 * conteo.get("observado", 0) / total, 4),
        "pct_interpolado": round(100 * conteo.get("interpolado", 0) / total, 4),
        "pct_faltante": round(100 * conteo.get("faltante", 0) / total, 4),
        "n_atipicos": int(marco["atipico"].sum()),
        "n_periodo_atipico": int(marco["periodo_atipico"].sum()),
        "n_utilizables_sin_marcas": int(
            (
                marco[columna_valor].notna()
                & ~marco["atipico"]
                & ~marco["periodo_atipico"]
                & ~marco["imputado"]
            ).sum()
        ),
    }


# --------------------------------------------------------------------------
# Trazabilidad de una celda
# --------------------------------------------------------------------------


def procedencia(
    marco: pd.DataFrame,
    momento: str | dt.datetime | pd.Timestamp,
    columna_tiempo: str = "fecha_hora",
    columna_valor: str = "valor_kwh",
) -> dict[str, Any]:
    """Responde de donde salio el valor de una hora concreta.

        procedencia(marco, "2025-03-06 04:00")
    """
    objetivo = pd.Timestamp(momento)
    momentos = pd.to_datetime(marco[columna_tiempo])
    if momentos.dt.tz is not None and objetivo.tz is None:
        objetivo = objetivo.tz_localize(momentos.dt.tz)

    fila = marco[momentos == objetivo]
    if fila.empty:
        return {
            "momento": str(objetivo),
            "encontrado": False,
            "explicacion": "Esa marca de tiempo no esta en el marco.",
        }

    fila = fila.iloc[0]
    origen = fila.get("origen_valor")
    valor = fila.get(columna_valor)

    explicaciones = {
        "observado": "Valor tal como lo publico la fuente; no se modifico.",
        "interpolado": (
            f"Valor imputado por interpolacion temporal: pertenecia a un hueco de "
            f"{fila.get('hueco_horas')} h, dentro del limite de "
            f"{MAX_HORAS_INTERPOLACION} h."
        ),
        "faltante": (
            f"Sin valor. Pertenece a un hueco de {fila.get('hueco_horas')} h, por "
            f"encima del limite de {MAX_HORAS_INTERPOLACION} h, asi que se dejo "
            "como NaN en vez de inventarlo."
        ),
    }

    return {
        "momento": str(objetivo),
        "encontrado": True,
        "valor": None if pd.isna(valor) else float(valor),
        "origen_valor": origen,
        "imputado": bool(fila.get("imputado", False)),
        "hueco_horas": int(fila.get("hueco_horas", 0)),
        "atipico": bool(fila.get("atipico", False)),
        "atipico_iqr": bool(fila.get("atipico_iqr", False)),
        "z_estacional": (
            None if pd.isna(fila.get("z_estacional")) else round(float(fila["z_estacional"]), 3)
        ),
        "periodo_atipico": bool(fila.get("periodo_atipico", False)),
        "etiqueta_periodo": fila.get("etiqueta_periodo"),
        "explicacion": explicaciones.get(origen, "Origen desconocido."),
    }


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------


def guardar(
    marco: pd.DataFrame,
    registro: dict[str, Any],
    nombre: str,
    directorio: Path | None = None,
) -> dict[str, str]:
    """Guarda el marco limpio en Parquet y anade el registro a la bitacora."""
    destino = directorio or DIR_SALIDA
    destino.mkdir(parents=True, exist_ok=True)

    ruta_datos = destino / f"{nombre}.parquet"
    marco.to_parquet(ruta_datos, index=False)

    ruta_registro = destino / NOMBRE_REGISTRO
    bitacora: dict[str, Any] = {"version_formato": 1, "limpiezas": []}
    if ruta_registro.exists():
        with ruta_registro.open(encoding="utf-8") as origen:
            bitacora = json.load(origen)

    entrada = dict(registro)
    entrada["conjunto"] = nombre
    entrada["archivo"] = ruta_datos.name
    bitacora["limpiezas"].append(entrada)

    temporal = ruta_registro.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as salida:
        json.dump(bitacora, salida, ensure_ascii=False, indent=2, default=str)
    temporal.replace(ruta_registro)

    log.info("Guardado %s (%d filas) y registro en %s", ruta_datos, len(marco), ruta_registro)
    return {"datos": str(ruta_datos), "registro": str(ruta_registro)}


# --------------------------------------------------------------------------
# Resumen legible
# --------------------------------------------------------------------------


def resumen(registro: dict[str, Any]) -> str:
    """Version legible del registro de limpieza."""
    lineas = ["=" * 74, "REGISTRO DE LIMPIEZA", "=" * 74]
    lineas.append(f"Momento : {registro['momento']}")
    lineas.append(
        f"Filas   : {registro['filas_iniciales']:,} -> {registro['filas_finales']:,}"
    )

    for paso in registro["operaciones"]:
        lineas.append("")
        lineas.append(f"[{paso['operacion']}]  {paso['filas_afectadas']:,} filas afectadas")
        lineas.append(f"  criterio: {paso['criterio']}")
        for clave, valor in paso["detalle"].items():
            if clave == "nota" or valor in ({}, [], None, 0, False, ""):
                continue
            if isinstance(valor, (dict, list)) and len(str(valor)) > 160:
                valor = f"({len(valor)} entradas)"
            lineas.append(f"  {clave}: {valor}")

    proc = registro["procedencia"]
    lineas.append("")
    lineas.append("-" * 74)
    lineas.append("PROCEDENCIA DE LOS VALORES")
    lineas.append("-" * 74)
    lineas.append(f"  observados  : {proc['por_origen'].get('observado', 0):>8,}  ({proc['pct_observado']}%)")
    lineas.append(f"  interpolados: {proc['por_origen'].get('interpolado', 0):>8,}  ({proc['pct_interpolado']}%)")
    lineas.append(f"  faltantes   : {proc['por_origen'].get('faltante', 0):>8,}  ({proc['pct_faltante']}%)")
    lineas.append(f"  atipicos marcados      : {proc['n_atipicos']:,}")
    lineas.append(f"  en periodo atipico     : {proc['n_periodo_atipico']:,}")
    lineas.append(f"  sin ninguna marca      : {proc['n_utilizables_sin_marcas']:,}")
    lineas.append("")
    lineas.append("Ningun valor fue eliminado ni corregido: solo marcado.")
    return "\n".join(lineas)
