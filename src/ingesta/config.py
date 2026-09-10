"""Rutas, constantes y parametros por defecto de la ingesta."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

# --- Rutas ---------------------------------------------------------------

RAIZ = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ / "data"
DIR_MUESTRAS = DIR_DATOS / "raw_samples"  # scripts/explorar_apis.py
DIR_CRUDO = DIR_DATOS / "raw"  # capa cruda particionada (descarga.py)
DIR_PROCESADO = DIR_DATOS / "processed"
DIR_CACHE = DIR_DATOS / "cache"
DIR_NOTAS = RAIZ / "notas"

# Dos manifiestos, con dos propositos distintos:
#   RUTA_MANIFIESTO_CRUDO  bitacora append-only de descargas, con hashes; es el
#                          registro de reproducibilidad (que datos habia cuando)
#   RUTA_MANIFIESTO        cobertura de la capa procesada (huecos, rezago)
RUTA_MANIFIESTO_CRUDO = DIR_CRUDO / "manifiesto.json"
RUTA_MANIFIESTO = DIR_DATOS / "manifiesto.json"

# Los tramos que terminan dentro de esta ventana no se sirven desde la cache.
# Dos razones: la demanda real llega con ~3 dias de rezago (un tramo reciente
# cacheado quedaria incompleto para siempre) y las versiones de liquidacion de
# SIMEM se revisan durante semanas (TX2 -> TXR -> TXF).
VENTANA_REFRESCO_DIAS = 45

# --- APIs ----------------------------------------------------------------

URL_XM = "https://servapibi.xm.com.co"
URL_SIMEM = "https://www.simem.co/backend-files/api/PublicData"

TIMEOUT_SEGUNDOS = 180

# Dataset del catalogo de conjuntos de datos de SIMEM (verificado 2026-09-09).
DATASET_CATALOGO_SIMEM = "e007fb"
# Dataset del inventario de variables de SIMEM.
DATASET_VARIABLES_SIMEM = "a5a6c4"
# "Demanda real nacional", granularidad horaria, datos desde 2021-01-01.
DATASET_DEMANDA_SIMEM = "14fabb"

# La API de SIMEM no publica MaxDays por dataset; el limite documentado para
# granularidad horaria y diaria es de 31 dias por llamado.
MAX_DIAS_SIMEM = 31

# Metrica objetivo: demanda real del Sistema Interconectado Nacional, en kWh.
METRICA_OBJETIVO_XM = "DemaReal"
ENTIDAD_OBJETIVO_XM = "Sistema"

# --- Rango historico -----------------------------------------------------

# 2021-01-01 es el inicio de los datos del dataset 14fabb de SIMEM; ambas
# fuentes cubren desde esa fecha.
FECHA_INICIO_DEFECTO = dt.date(2021, 1, 1)

# --- Semantica de los datos ----------------------------------------------

# --------------------------------------------------------------------------
# CONVENCION HORARIA DE XM  --  unica fuente de verdad del proyecto
# --------------------------------------------------------------------------
# Desfase entre el indice N de la columna HourNN de XM y la hora de reloj:
#
#     hora_de_reloj = N - DESFASE_HORA_XM
#
# Con DESFASE_HORA_XM = 1, Hour01 es el intervalo 00:00-01:00 y Hour24 el de
# 23:00-00:00.
#
# VERIFICADO EMPIRICAMENTE (2026-09-09, scripts/reconciliar.py --mes 2026-08):
# se comparo la serie de XM contra la de SIMEM probando tres desfases, y el
# alineamiento correcto es el que produce la correlacion maxima:
#
#     desfase -1 -> correlacion 0.948753
#     desfase  0 -> correlacion 0.991236   <-- optimo
#     desfase +1 -> correlacion 0.952222
#
# Confirmado ademas sobre el historico completo (49 752 horas, 2021-2026):
# correlacion 0.992516 con un sesgo de signo estable, propio de una diferencia
# de alcance entre ambas series, no de un desalineamiento temporal.
#
# Si alguna vez hubiera que corregirlo, se corrige AQUI y en ningun otro sitio:
# clientes.ClienteXM es el unico que traduce HourNN, y lee esta constante.
DESFASE_HORA_XM = 1

# Precedencia de versiones de liquidacion de SIMEM, de mas a menos definitiva.
# Sobre una misma FechaHora pueden coexistir varias versiones: se han observado
# hasta cuatro a la vez (TX2, TX3, TXR y TXF en mayo de 2026), asi que agregar
# sin colapsar multiplicaria la demanda por cuatro, no por dos.
#
# El orden entre TXn es el de las liquidaciones sucesivas; TXR (reliquidacion) y
# TXF se toman como posteriores a todas ellas. No esta confirmado contra
# documentacion oficial de XM: es una decision del proyecto. Medido sobre mayo
# de 2026, la eleccion entre versiones mueve el total mensual menos de un 0.05%,
# asi que el riesgo real esta en no colapsar, no en el orden exacto.
PRECEDENCIA_VERSIONES = (
    "TXF",
    "TXR",
    "TX8",
    "TX7",
    "TX6",
    "TX5",
    "TX4",
    "TX3",
    "TX2",
    "TX1",
)

# --------------------------------------------------------------------------
# ZONA HORARIA  --  unica en todo el proyecto
# --------------------------------------------------------------------------
# Colombia no aplica horario de verano, asi que un desplazamiento fijo de -5 es
# exacto y no depende de la base de datos de zonas del sistema.
# Se usa el nombre IANA y no un desplazamiento fijo porque pandas, pyarrow y
# Parquet lo entienden por igual, mientras que una cadena tipo "UTC-05:00" no es
# una zona valida para pyarrow. Colombia no aplica horario de verano, asi que
# America/Bogota equivale siempre y exactamente a -05:00.
ZONA_COLOMBIA = "America/Bogota"
ZONA_HORARIA = ZONA_COLOMBIA

# Una tabla con mas de estas filas por marca de tiempo esta desagregada: hay
# varias entidades o versiones por hora. El umbral es unico para todo el
# proyecto; tenerlo por duplicado hacia que un modulo considerase agregada una
# tabla que otro veia desagregada.
UMBRAL_FILAS_POR_MARCA = 1.01

# Observaciones por grupo (dia de semana, hora) a partir de las cuales el
# criterio estacional se considera fiable. Compartido por diagnostico y
# limpieza, para que ambos reporten la misma fiabilidad.
GRUPO_COMODO = 30


def a_zona_colombia(serie):
    """Devuelve la serie de marcas de tiempo en UTC-5, sea cual sea su origen.

    Localiza si viene sin zona y convierte si ya la trae. Es idempotente, asi
    que se puede aplicar en cualquier frontera sin comprobar antes que llego.
    """
    import pandas as pd

    momentos = pd.to_datetime(serie)
    if getattr(momentos.dt, "tz", None) is None:
        return momentos.dt.tz_localize(ZONA_COLOMBIA)
    return momentos.dt.tz_convert(ZONA_COLOMBIA)


def asegurar_directorios() -> None:
    """Crea los directorios de datos si aun no existen."""
    for directorio in (DIR_CACHE, DIR_PROCESADO, DIR_NOTAS):
        directorio.mkdir(parents=True, exist_ok=True)
