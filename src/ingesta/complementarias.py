"""Fuentes complementarias: clima y calendario colombiano.

**Clima.** Temperatura horaria historica de las cuatro ciudades principales,
desde la API de archivo de Open-Meteo, agregada a una serie nacional por media
ponderada. Los pesos son un parametro, no una constante escondida: se pasan a
cada llamada y quedan anotados en el registro de la descarga, de modo que
cualquier analisis posterior puede decir con que ponderacion se construyo.

**Calendario.** Se apoya en la libreria `holidays` con `country='CO'`, que ya
implementa la Ley Emiliani (el traslado de varios festivos al lunes siguiente).
Construir ese calendario a mano seria reimplementar una regla no trivial que ya
esta resuelta y mantenida.

**Integracion.** `unir()` junta demanda, clima y calendario por marca de tiempo
y **falla ruidosamente** si el numero de filas no cuadra. Un merge que duplica
o pierde filas en silencio envenena todo lo que venga despues.

Forma de la respuesta de Open-Meteo, verificada contra el servidor el 2026-09-10:

    {"latitude": 4.7451673, "longitude": -74.10025, "utc_offset_seconds": -18000,
     "timezone": "America/Bogota", "timezone_abbreviation": "GMT-5",
     "elevation": 2557.0,
     "hourly_units": {"time": "iso8601", "temperature_2m": "°C"},
     "hourly": {"time": ["2025-01-01T00:00", ...],
                "temperature_2m": [10.8, 10.6, 9.8, ...]}}

Dos detalles de esa respuesta que condicionan el codigo: la latitud y longitud
que vuelven **no son las pedidas** sino las del centro de la celda de la malla
(4.745 frente a 4.711 para Bogota), y por eso se registran; y el archivo no
tiene rezago apreciable, a diferencia de la demanda.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as pa
import requests

from . import config, particiones, ventanas

log = logging.getLogger(__name__)

URL_OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
ZONA_CONSULTA = "America/Bogota"
VARIABLE_TEMPERATURA = "temperature_2m"

# Open-Meteo no documenta un limite de dias por llamado tan estricto como XM,
# pero trocear evita respuestas enormes y permite reintentar por partes.
MAX_DIAS_OPEN_METEO = 365


@dataclass(frozen=True)
class Ciudad:
    """Punto de consulta de temperatura."""

    clave: str
    nombre: str
    latitud: float
    longitud: float


CIUDADES: dict[str, Ciudad] = {
    "bogota": Ciudad("bogota", "Bogota", 4.7110, -74.0721),
    "medellin": Ciudad("medellin", "Medellin", 6.2442, -75.5812),
    "cali": Ciudad("cali", "Cali", 3.4516, -76.5320),
    "barranquilla": Ciudad("barranquilla", "Barranquilla", 10.9685, -74.7813),
}

# --------------------------------------------------------------------------
# PESOS: valor por defecto PROVISIONAL, pensado para ser sustituido
# --------------------------------------------------------------------------
# Estos pesos son una aproximacion de orden de magnitud por tamano de area
# metropolitana. NO estan medidos sobre la demanda real y no deben presentarse
# como si lo estuvieran.
#
# La base correcta seria la participacion de cada zona en la demanda del SIN.
# SIMEM publica conjuntos que permiten calcularla -- "Pronostico oficial de
# demanda por Area operativa" (d91840) y "Demandas agrupadas por Sistema de
# Transmision Regional" (38FF5B), ambos verificados en el catalogo -- pero el
# mapeo de area operativa a ciudad no es inmediato y no se ha hecho aqui.
#
# Mientras tanto: pasa tus propios pesos a cada funcion. Se anotan en el
# registro de cada descarga, asi que siempre queda constancia de cual se uso.
PESOS_DEFECTO: dict[str, float] = {
    "bogota": 0.45,
    "medellin": 0.22,
    "cali": 0.18,
    "barranquilla": 0.15,
}

ESQUEMA_CLIMA = pa.schema(
    [
        ("fecha_hora", pa.timestamp("ns")),
        ("temperatura_c", pa.float64()),
        ("ciudades_disponibles", pa.int32()),
        ("anio", pa.int32()),
        ("mes", pa.int32()),
    ]
)
CLAVE_CLIMA = ["fecha_hora"]

DIR_CLIMA = "clima_nacional"
RUTA_REGISTRO_CLIMA = "registro_clima.json"


class ErrorComplementaria(Exception):
    """Fallo al obtener o integrar una fuente complementaria."""


class ErrorUnion(ErrorComplementaria):
    """La union de tablas no conserva el numero de filas esperado."""


# --------------------------------------------------------------------------
# Pesos
# --------------------------------------------------------------------------


def normalizar_pesos(pesos: dict[str, float]) -> dict[str, float]:
    """Escala unos pesos para que sumen 1, validandolos antes.

    Acepta magnitudes crudas (por ejemplo GWh por zona) y las convierte en
    participaciones, que es la via para sustituir el proxy por datos reales.
    """
    if not pesos:
        raise ErrorComplementaria("No se dieron pesos.")

    desconocidas = set(pesos) - set(CIUDADES)
    if desconocidas:
        raise ErrorComplementaria(
            f"Ciudades sin coordenadas definidas: {sorted(desconocidas)}. "
            f"Conocidas: {sorted(CIUDADES)}."
        )
    negativos = {c: p for c, p in pesos.items() if p < 0}
    if negativos:
        raise ErrorComplementaria(f"Los pesos no pueden ser negativos: {negativos}")

    total = sum(pesos.values())
    if total <= 0:
        raise ErrorComplementaria("Los pesos suman cero: no definen una ponderacion.")

    return {ciudad: peso / total for ciudad, peso in pesos.items()}


# --------------------------------------------------------------------------
# Clima
# --------------------------------------------------------------------------


def descargar_temperatura(
    ciudad: Ciudad,
    desde: dt.date,
    hasta: dt.date,
    sesion: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Temperatura horaria de una ciudad. Devuelve (tabla, metadatos)."""
    sesion = sesion or requests.Session()
    trozos = ventanas.partir_rango(desde, hasta, MAX_DIAS_OPEN_METEO)

    marcos: list[pd.DataFrame] = []
    metadatos: dict[str, Any] = {}

    for inicio, fin in trozos:
        respuesta = sesion.get(
            URL_OPEN_METEO,
            params={
                "latitude": ciudad.latitud,
                "longitude": ciudad.longitud,
                "start_date": ventanas.a_iso(inicio),
                "end_date": ventanas.a_iso(fin),
                "hourly": VARIABLE_TEMPERATURA,
                "timezone": ZONA_CONSULTA,
            },
            timeout=config.TIMEOUT_SEGUNDOS,
        )
        if respuesta.status_code != 200:
            raise ErrorComplementaria(
                f"Open-Meteo {ciudad.nombre} [{inicio}..{fin}] devolvio "
                f"{respuesta.status_code}: {respuesta.text[:300]}"
            )

        datos = respuesta.json()
        horario = datos.get("hourly") or {}
        if not horario.get("time"):
            log.warning("Open-Meteo %s: sin datos en %s..%s", ciudad.nombre, inicio, fin)
            continue

        marcos.append(
            pd.DataFrame(
                {
                    "fecha_hora": pd.to_datetime(horario["time"]),
                    f"temp_{ciudad.clave}": horario[VARIABLE_TEMPERATURA],
                }
            )
        )
        # La malla no coincide con el punto pedido: se registra lo devuelto.
        metadatos = {
            "ciudad": ciudad.nombre,
            "lat_pedida": ciudad.latitud,
            "lon_pedida": ciudad.longitud,
            "lat_malla": datos.get("latitude"),
            "lon_malla": datos.get("longitude"),
            "elevacion_m": datos.get("elevation"),
            "zona": datos.get("timezone"),
            "desfase_utc_s": datos.get("utc_offset_seconds"),
            "unidad": (datos.get("hourly_units") or {}).get(VARIABLE_TEMPERATURA),
        }

    if not marcos:
        return pd.DataFrame(columns=["fecha_hora", f"temp_{ciudad.clave}"]), metadatos

    tabla = pd.concat(marcos, ignore_index=True).drop_duplicates(subset=["fecha_hora"])
    return tabla.sort_values("fecha_hora").reset_index(drop=True), metadatos


def descargar_clima(
    desde: dt.date,
    hasta: dt.date,
    pesos: dict[str, float] | None = None,
    sesion: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Temperatura de las ciudades y su media ponderada nacional.

    Devuelve la tabla con una columna por ciudad mas `temperatura_c`, y el
    registro de la descarga, que incluye los pesos usados.
    """
    pesos = normalizar_pesos(pesos or PESOS_DEFECTO)
    sesion = sesion or requests.Session()
    log.info("Clima %s..%s con pesos %s", desde, hasta, pesos)

    tabla: pd.DataFrame | None = None
    detalles: list[dict[str, Any]] = []

    for clave in pesos:
        ciudad = CIUDADES[clave]
        parcial, metadatos = descargar_temperatura(ciudad, desde, hasta, sesion)
        detalles.append({**metadatos, "n_horas": len(parcial)})
        if parcial.empty:
            continue
        tabla = parcial if tabla is None else tabla.merge(parcial, on="fecha_hora", how="outer")

    if tabla is None or tabla.empty:
        raise ErrorComplementaria(
            f"Ninguna ciudad devolvio temperatura para {desde}..{hasta}."
        )

    tabla = tabla.sort_values("fecha_hora").reset_index(drop=True)
    tabla, resumen_agregado = agregar_nacional(tabla, pesos)

    registro = {
        "momento": dt.datetime.now().isoformat(timespec="seconds"),
        "fuente": "open-meteo-archive",
        "variable": VARIABLE_TEMPERATURA,
        "rango_solicitado": [desde.isoformat(), hasta.isoformat()],
        "rango_obtenido": [
            str(tabla["fecha_hora"].min()),
            str(tabla["fecha_hora"].max()),
        ],
        "n_registros": len(tabla),
        "pesos_usados": pesos,
        "pesos_son_proxy": pesos == normalizar_pesos(PESOS_DEFECTO),
        "nota_pesos": (
            "Los pesos por defecto son una aproximacion por tamano de area "
            "metropolitana, no una medicion de la demanda por zona."
        ),
        "ciudades": detalles,
        **resumen_agregado,
    }
    return tabla, registro


def agregar_nacional(
    tabla: pd.DataFrame, pesos: dict[str, float]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Anade `temperatura_c` como media ponderada de las ciudades presentes.

    Si a una hora le falta la temperatura de alguna ciudad, se renormalizan los
    pesos de las que si estan en vez de propagar un NaN a toda la serie. La
    columna `ciudades_disponibles` dice cuantas entraron en cada hora, para que
    esa renormalizacion sea visible y no un apano silencioso.
    """
    columnas = [f"temp_{c}" for c in pesos if f"temp_{c}" in tabla.columns]
    if not columnas:
        raise ErrorComplementaria(
            f"Ninguna columna de temperatura en la tabla: {list(tabla.columns)}"
        )

    datos = tabla.copy()
    valores = datos[columnas]
    peso_serie = pd.Series({f"temp_{c}": p for c, p in pesos.items()})[columnas]

    # A float explicito: si una columna llega entera de nulos, pandas la trata
    # como object y el fillna posterior emite un aviso de downcasting.
    valores = valores.astype("float64")
    presentes = valores.notna()
    peso_efectivo = presentes.mul(peso_serie, axis=1)
    suma_pesos = peso_efectivo.sum(axis=1)

    datos["temperatura_c"] = (valores.fillna(0.0) * peso_efectivo).sum(axis=1) / suma_pesos
    datos.loc[suma_pesos == 0, "temperatura_c"] = float("nan")
    datos["ciudades_disponibles"] = presentes.sum(axis=1).astype("int32")

    horas_parciales = int((datos["ciudades_disponibles"].between(1, len(columnas) - 1)).sum())
    resumen = {
        "ciudades_incluidas": [c.replace("temp_", "") for c in columnas],
        "horas_con_todas_las_ciudades": int(
            (datos["ciudades_disponibles"] == len(columnas)).sum()
        ),
        "horas_con_ponderacion_parcial": horas_parciales,
        "horas_sin_ninguna_ciudad": int((datos["ciudades_disponibles"] == 0).sum()),
    }
    return datos, resumen


def guardar_clima(
    tabla: pd.DataFrame, registro: dict[str, Any], directorio=None
) -> dict[str, str]:
    """Persiste el clima nacional en Parquet particionado y anota el registro."""
    base = directorio or config.DIR_PROCESADO
    destino = base / DIR_CLIMA
    particiones.escribir_particionado(
        tabla[["fecha_hora", "temperatura_c", "ciudades_disponibles"]],
        destino,
        ESQUEMA_CLIMA,
        CLAVE_CLIMA,
        etiqueta=DIR_CLIMA,
    )

    ruta_registro = base / RUTA_REGISTRO_CLIMA
    bitacora: dict[str, Any] = {"version_formato": 1, "descargas": []}
    if ruta_registro.exists():
        with ruta_registro.open(encoding="utf-8") as origen:
            bitacora = json.load(origen)
    bitacora["descargas"].append(registro)

    temporal = ruta_registro.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as salida:
        json.dump(bitacora, salida, ensure_ascii=False, indent=2, default=str)
    temporal.replace(ruta_registro)

    return {"datos": str(destino), "registro": str(ruta_registro)}


# --------------------------------------------------------------------------
# Calendario colombiano
# --------------------------------------------------------------------------

# Definiciones de las banderas, explicitas porque varias son decisiones y no
# hechos: "puente" o "semana santa" no tienen una unica definicion obvia.
DEFINICIONES_CALENDARIO = {
    "es_festivo": "La fecha figura en holidays.country_holidays('CO').",
    "es_vispera_festivo": "El dia siguiente es festivo.",
    "es_puente": (
        "La fecha cae en un fin de semana largo generado por la Ley Emiliani: "
        "sabado, domingo o lunes cuando ese lunes es festivo."
    ),
    "es_semana_santa": (
        "La fecha cae en la semana de lunes a domingo que contiene el Viernes "
        "Santo."
    ),
    "es_ultima_semana_diciembre": "Del 25 al 31 de diciembre, ambos incluidos.",
}


def calendario(desde: dt.date, hasta: dt.date) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calendario diario colombiano con las banderas del proyecto.

    Usa `holidays` con `country='CO'`, que ya aplica la Ley Emiliani: San Jose,
    Ascension, Corpus Christi, Sagrado Corazon, San Pedro y San Pablo, la
    Asuncion, el Dia de la Raza, Todos los Santos y la Independencia de
    Cartagena se trasladan al lunes siguiente. Reimplementar eso a mano seria
    reescribir una regla no trivial que ya esta resuelta y mantenida.
    """
    try:
        import holidays
    except ImportError as exc:  # pragma: no cover
        raise ErrorComplementaria(
            "Falta la libreria 'holidays' (pip install holidays)."
        ) from exc

    anios = list(range(desde.year, hasta.year + 1))
    festivos = holidays.country_holidays("CO", years=anios)

    fechas = pd.date_range(desde, hasta, freq="D")
    tabla = pd.DataFrame({"fecha": fechas})
    dias = tabla["fecha"].dt.date

    tabla["es_festivo"] = dias.map(lambda d: d in festivos)
    tabla["nombre_festivo"] = dias.map(lambda d: festivos.get(d))
    tabla["es_vispera_festivo"] = dias.map(
        lambda d: (d + dt.timedelta(days=1)) in festivos
    )

    # Puente: fin de semana largo por lunes festivo. Se mira, para cada fecha,
    # el lunes de su propia semana.
    dia_semana = tabla["fecha"].dt.dayofweek
    lunes_de_la_semana = tabla["fecha"] - pd.to_timedelta(dia_semana, unit="D")
    lunes_festivo = lunes_de_la_semana.dt.date.map(lambda d: d in festivos)
    tabla["es_puente"] = lunes_festivo & dia_semana.isin([0, 5, 6])
    # El sabado y el domingo del puente pertenecen a la semana anterior.
    lunes_siguiente = tabla["fecha"] + pd.to_timedelta(7 - dia_semana, unit="D")
    puente_fin_de_semana = dia_semana.isin([5, 6]) & lunes_siguiente.dt.date.map(
        lambda d: d in festivos
    )
    tabla["es_puente"] = tabla["es_puente"] | puente_fin_de_semana

    tabla["es_semana_santa"] = _marcar_semana_santa(tabla["fecha"], festivos)

    tabla["es_ultima_semana_diciembre"] = (
        (tabla["fecha"].dt.month == 12) & (tabla["fecha"].dt.day >= 25)
    )

    registro = {
        "momento": dt.datetime.now().isoformat(timespec="seconds"),
        "fuente": f"holidays {holidays.__version__} country_holidays('CO')",
        "rango": [desde.isoformat(), hasta.isoformat()],
        "n_dias": len(tabla),
        "n_festivos": int(tabla["es_festivo"].sum()),
        "festivos_por_anio": {
            str(anio): int(sum(1 for f in festivos if f.year == anio)) for anio in anios
        },
        "definiciones": DEFINICIONES_CALENDARIO,
        "banderas": {
            columna: int(tabla[columna].sum())
            for columna in DEFINICIONES_CALENDARIO
        },
        "ley_emiliani": (
            "Ley Emiliani aplicada por la libreria holidays: los festivos "
            "trasladables aparecen en el lunes siguiente, marcados como "
            "'(observado)'. No se implementa a mano en este proyecto."
        ),
    }
    return tabla, registro


def _marcar_semana_santa(fechas: pd.Series, festivos) -> pd.Series:
    """Semana de lunes a domingo que contiene cada Viernes Santo."""
    viernes_santo = [
        fecha
        for fecha, nombre in festivos.items()
        if "Viernes Santo" in str(nombre)
    ]
    if not viernes_santo:
        return pd.Series(False, index=fechas.index)

    semanas = set()
    for viernes in viernes_santo:
        lunes = viernes - dt.timedelta(days=viernes.weekday())
        semanas.update(lunes + dt.timedelta(days=n) for n in range(7))

    return fechas.dt.date.map(lambda d: d in semanas)


def calendario_horario(desde: dt.date, hasta: dt.date) -> tuple[pd.DataFrame, dict]:
    """El calendario diario expandido a marcas de tiempo horarias."""
    diario, registro = calendario(desde, hasta)
    horas = pd.date_range(
        dt.datetime.combine(desde, dt.time(0)),
        dt.datetime.combine(hasta, dt.time(23)),
        freq="h",
    )
    marco = pd.DataFrame({"fecha_hora": horas})
    marco["fecha"] = marco["fecha_hora"].dt.normalize()
    horario = marco.merge(diario, on="fecha", how="left").drop(columns="fecha")
    registro["n_horas"] = len(horario)
    return horario, registro


# --------------------------------------------------------------------------
# Integracion
# --------------------------------------------------------------------------


def unir(
    demanda: pd.DataFrame,
    clima: pd.DataFrame | None = None,
    calendario_h: pd.DataFrame | None = None,
    columna_tiempo: str = "fecha_hora",
    exigir_cobertura_total: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Une demanda, clima y calendario por marca de tiempo, validando el merge.

    La demanda manda: el resultado tiene exactamente sus filas, ni una mas ni
    una menos. Si alguna union las altera, esto **falla**. Un merge que duplica
    filas en silencio -- por ejemplo porque la tabla de la derecha trae marcas
    de tiempo repetidas -- multiplica observaciones y envenena todo lo que venga
    despues sin dejar rastro visible.

    Con `exigir_cobertura_total=True` tambien falla si alguna fuente deja huecos
    en la demanda, en vez de limitarse a reportarlos.
    """
    if demanda.empty:
        raise ErrorUnion("La tabla de demanda esta vacia.")

    filas_esperadas = len(demanda)
    resultado = demanda.copy()
    resultado[columna_tiempo] = pd.to_datetime(resultado[columna_tiempo])

    duplicados_demanda = int(resultado[columna_tiempo].duplicated().sum())
    if duplicados_demanda:
        raise ErrorUnion(
            f"La demanda trae {duplicados_demanda} marcas de tiempo repetidas. "
            "Unir sobre una clave no unica multiplicaria filas: deduplica antes."
        )

    informe: dict[str, Any] = {
        "momento": dt.datetime.now().isoformat(timespec="seconds"),
        "filas_demanda": filas_esperadas,
        "uniones": [],
    }

    for nombre, tabla in (("clima", clima), ("calendario", calendario_h)):
        if tabla is None or tabla.empty:
            informe["uniones"].append({"fuente": nombre, "aplicada": False})
            continue

        derecha = tabla.copy()
        derecha[columna_tiempo] = pd.to_datetime(derecha[columna_tiempo])

        repetidas = int(derecha[columna_tiempo].duplicated().sum())
        if repetidas:
            raise ErrorUnion(
                f"La tabla de {nombre} trae {repetidas} marcas de tiempo repetidas. "
                f"Unirla multiplicaria filas de demanda. Deduplicala antes."
            )

        solapadas = [
            c
            for c in derecha.columns
            if c != columna_tiempo and c in resultado.columns
        ]
        if solapadas:
            raise ErrorUnion(
                f"La tabla de {nombre} comparte columnas con el resultado: "
                f"{solapadas}. Renombralas para no pisar datos en silencio."
            )

        antes = len(resultado)
        resultado = resultado.merge(derecha, on=columna_tiempo, how="left")
        despues = len(resultado)

        if despues != filas_esperadas:
            raise ErrorUnion(
                f"La union con {nombre} cambio el numero de filas: "
                f"{antes} -> {despues}, se esperaban {filas_esperadas}. "
                "El merge no es uno a uno."
            )

        nuevas = [c for c in derecha.columns if c != columna_tiempo]
        sin_cobertura = int(resultado[nuevas[0]].isna().sum()) if nuevas else 0
        informe["uniones"].append(
            {
                "fuente": nombre,
                "aplicada": True,
                "columnas_anadidas": nuevas,
                "filas_derecha": len(derecha),
                "filas_sin_cobertura": sin_cobertura,
                "pct_sin_cobertura": round(100 * sin_cobertura / filas_esperadas, 4),
            }
        )
        if sin_cobertura:
            mensaje = (
                f"{nombre}: {sin_cobertura} de {filas_esperadas} horas de demanda "
                f"sin dato ({100 * sin_cobertura / filas_esperadas:.2f}%)"
            )
            if exigir_cobertura_total:
                raise ErrorUnion(mensaje)
            log.warning(mensaje)

    if len(resultado) != filas_esperadas:  # pragma: no cover - defensa final
        raise ErrorUnion(
            f"El resultado tiene {len(resultado)} filas y la demanda {filas_esperadas}."
        )

    informe["filas_resultado"] = len(resultado)
    informe["columnas_resultado"] = list(resultado.columns)
    informe["cuadra"] = True
    log.info(
        "Union correcta: %d filas, %d columnas", len(resultado), len(resultado.columns)
    )
    return resultado, informe
