"""Capa de acceso a datos: un cliente por API, con interfaz comun.

    ClienteXM().consultar("DemaReal", inicio, fin, entidad="Sistema")
    ClienteSIMEM().consultar("14fabb", inicio, fin)

Ambos devuelven un DataFrame ya en formato largo, con estas columnas siempre
presentes:

    timestamp        datetime64[ns]  hora local de Colombia (naive, sin DST)
    fuente           str             "xm" o "simem"
    identificador    str             MetricId o datasetId consultado

XM anade `entidad` y `valor` (float). SIMEM anade las columnas propias del
dataset tal como llegan (`Valor`, `Version`, `CodigoSICAgente`, ...), porque
sus dimensiones cambian de un conjunto a otro y aplanarlas aqui perderia
informacion. Las dimensiones de SIMEM no se tocan: colapsar versiones de
liquidacion es trabajo de `normalizar`, no de la capa de acceso.

Los clientes se encargan de:

- fragmentar el rango segun el limite de cada API y concatenar el resultado;
- reintentar hasta 3 veces con backoff exponencial ante 5xx y timeouts, y
  nunca ante 4xx;
- cachear cada tramo en disco (Parquet) y reutilizarlo en llamados posteriores.

Dependencias: requests, pandas, pyarrow.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from . import config, ventanas

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Politica de reintentos
# --------------------------------------------------------------------------

MAX_INTENTOS = 3
# Espera antes del reintento n: BACKOFF_BASE * 2**(n-1) -> 2 s, 4 s.
BACKOFF_BASE_SEGUNDOS = 2.0

# Limite por llamado cuando el catalogo de XM no informa MaxDays.
MAX_DIAS_POR_DEFECTO = 30


class ErrorCliente(Exception):
    """Fallo al consultar una de las APIs."""


class ErrorHTTP(ErrorCliente):
    """La API respondio con un codigo de error.

    Conserva el status y el cuerpo porque ambas APIs ponen ahi el diagnostico
    util, y ninguna de las dos lo hace de forma uniforme.
    """

    def __init__(self, status: int, cuerpo: str, contexto: str) -> None:
        self.status = status
        self.cuerpo = cuerpo
        super().__init__(f"{contexto}: HTTP {status} -- {cuerpo[:300]}")


# --------------------------------------------------------------------------
# Cliente base
# --------------------------------------------------------------------------


class ClienteAPI(ABC):
    """Interfaz comun a las dos APIs de XM S.A. E.S.P.

    Las subclases implementan como se pide un tramo (`_pedir`) y como se
    convierte la respuesta cruda a formato largo (`_a_dataframe`). Todo lo
    demas -- fragmentacion, reintentos, cache, concatenacion -- vive aqui.
    """

    fuente: str

    def __init__(
        self,
        sesion: requests.Session | None = None,
        usar_cache: bool = True,
        dir_cache: Path | None = None,
        timeout: int = config.TIMEOUT_SEGUNDOS,
    ) -> None:
        self.sesion = sesion or self._crear_sesion()
        self.usar_cache = usar_cache
        self.dir_cache = dir_cache or config.DIR_CACHE
        self.timeout = timeout

    @staticmethod
    def _crear_sesion() -> requests.Session:
        """Sesion sin reintentos propios: la politica la aplica el cliente."""
        sesion = requests.Session()
        sesion.headers.update({"Accept": "application/json"})
        return sesion

    # --- interfaz publica -------------------------------------------------

    def consultar(
        self,
        identificador: str,
        fecha_inicio: dt.date,
        fecha_fin: dt.date,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Consulta un rango completo, fragmentandolo y concatenando el resultado.

        Devuelve un DataFrame vacio con las columnas esperadas si ningun tramo
        trajo datos, en vez de None o de una excepcion: un rango sin publicar
        es un resultado legitimo, no un error.
        """
        fecha_inicio = _a_fecha(fecha_inicio)
        fecha_fin = _a_fecha(fecha_fin)

        max_dias = self._max_dias(identificador, **kwargs)
        tramos = ventanas.partir_rango(fecha_inicio, fecha_fin, max_dias)

        log.info(
            "%s: %s %s..%s -> %d tramos de hasta %d dias",
            self.fuente,
            identificador,
            fecha_inicio,
            fecha_fin,
            len(tramos),
            max_dias,
        )

        marcos: list[pd.DataFrame] = []
        for numero, (desde, hasta) in enumerate(tramos, start=1):
            marco = self._obtener_tramo(
                identificador, desde, hasta, numero, len(tramos), **kwargs
            )
            if not marco.empty:
                marcos.append(marco)

        if not marcos:
            log.warning(
                "%s: %s %s..%s no devolvio ningun registro",
                self.fuente,
                identificador,
                fecha_inicio,
                fecha_fin,
            )
            return self._marco_vacio()

        resultado = pd.concat(marcos, ignore_index=True)
        resultado = resultado.sort_values("timestamp").reset_index(drop=True)
        log.info(
            "%s: %s -> %d filas, %s..%s",
            self.fuente,
            identificador,
            len(resultado),
            resultado["timestamp"].min(),
            resultado["timestamp"].max(),
        )
        return resultado

    # --- fragmento a fragmento -------------------------------------------

    def _obtener_tramo(
        self,
        identificador: str,
        desde: dt.date,
        hasta: dt.date,
        numero: int,
        total: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Devuelve un tramo, desde la cache si es posible y si no de la API."""
        ruta = self._ruta_cache(identificador, desde, hasta, **kwargs)

        if self._cache_utilizable(ruta, hasta):
            marco = pd.read_parquet(ruta)
            # Los tramos cacheados antes de unificar la zona se guardaron sin
            # ella. Normalizarlos al leer evita mezclar naive con tz-aware al
            # concatenar, y hace la migracion transparente.
            if "timestamp" in marco.columns:
                marco["timestamp"] = config.a_zona_colombia(marco["timestamp"])
            log.info(
                "%s: tramo %d/%d %s..%s desde cache (%d filas)",
                self.fuente,
                numero,
                total,
                desde,
                hasta,
                len(marco),
            )
            return marco

        log.info(
            "%s: tramo %d/%d %s..%s -> API", self.fuente, numero, total, desde, hasta
        )
        crudo = self._pedir_con_reintentos(identificador, desde, hasta, **kwargs)
        marco = self._a_dataframe(crudo, identificador, **kwargs)

        if marco.empty:
            log.warning(
                "%s: tramo %s..%s de %s vino vacio "
                "(la API responde 200 sin datos, no es un error)",
                self.fuente,
                desde,
                hasta,
                identificador,
            )

        self._guardar_cache(marco, ruta, hasta)
        return marco

    def _pedir_con_reintentos(
        self, identificador: str, desde: dt.date, hasta: dt.date, **kwargs: Any
    ) -> Any:
        """Ejecuta `_pedir` reintentando ante 5xx y timeouts, nunca ante 4xx.

        Un 4xx significa que la consulta esta mal formada (rango excedido,
        dataset inexistente): reintentarla solo gasta tiempo y da el mismo
        error.
        """
        contexto = f"{self.fuente} {identificador} {desde}..{hasta}"

        for intento in range(1, MAX_INTENTOS + 1):
            try:
                return self._pedir(identificador, desde, hasta, **kwargs)

            except ErrorHTTP as exc:
                if exc.status < 500:
                    log.error(
                        "%s: HTTP %d, no se reintenta (consulta invalida). Cuerpo: %s",
                        contexto,
                        exc.status,
                        exc.cuerpo[:300],
                    )
                    raise
                motivo = f"HTTP {exc.status}"
                ultimo: Exception = exc

            except (requests.Timeout, requests.ConnectionError) as exc:
                motivo = type(exc).__name__
                ultimo = exc

            except requests.RequestException as exc:
                log.error("%s: fallo no reintentable: %r", contexto, exc)
                raise ErrorCliente(f"{contexto}: {exc}") from exc

            if intento == MAX_INTENTOS:
                log.error(
                    "%s: agotados los %d intentos (%s)", contexto, MAX_INTENTOS, motivo
                )
                raise ErrorCliente(
                    f"{contexto}: fallo tras {MAX_INTENTOS} intentos ({motivo})"
                ) from ultimo

            espera = BACKOFF_BASE_SEGUNDOS * (2 ** (intento - 1))
            log.warning(
                "%s: %s en el intento %d/%d, reintentando en %.0f s",
                contexto,
                motivo,
                intento,
                MAX_INTENTOS,
                espera,
            )
            time.sleep(espera)

        raise AssertionError("inalcanzable")  # pragma: no cover

    # --- cache ------------------------------------------------------------

    def _ruta_cache(
        self, identificador: str, desde: dt.date, hasta: dt.date, **kwargs: Any
    ) -> Path:
        """Ruta del tramo en cache.

        La clave es fuente + identificador + rango + los kwargs que cambian la
        consulta. Incluir los kwargs no es opcional: sin ellos, DemaReal para
        Entity=Sistema y para Entity=Agente compartirian archivo y el segundo
        leeria los datos del primero.
        """
        sufijo = self._clave_extra(**kwargs)
        nombre = f"{identificador}{sufijo}_{desde:%Y%m%d}_{hasta:%Y%m%d}.parquet"
        return self.dir_cache / self.fuente / nombre

    def _cache_utilizable(self, ruta: Path, hasta: dt.date) -> bool:
        """Decide si un tramo cacheado se puede reutilizar.

        Los tramos recientes se vuelven a pedir siempre: la demanda llega con
        rezago y las versiones de liquidacion se revisan durante semanas, asi
        que un tramo reciente cacheado se quedaria congelado en datos parciales.
        """
        if not self.usar_cache or not ruta.exists():
            return False

        antiguedad = (dt.date.today() - hasta).days
        if antiguedad < config.VENTANA_REFRESCO_DIAS:
            log.info(
                "%s: %s cierra hace %d dias (< %d), se vuelve a pedir en vez de usar cache",
                self.fuente,
                ruta.name,
                antiguedad,
                config.VENTANA_REFRESCO_DIAS,
            )
            return False

        return True

    def _guardar_cache(self, marco: pd.DataFrame, ruta: Path, hasta: dt.date) -> None:
        """Guarda un tramo en cache, salvo que sea demasiado reciente o vacio."""
        if not self.usar_cache or marco.empty:
            return
        if (dt.date.today() - hasta).days < config.VENTANA_REFRESCO_DIAS:
            return

        ruta.parent.mkdir(parents=True, exist_ok=True)
        marco.to_parquet(ruta, index=False)
        log.debug("%s: tramo cacheado en %s", self.fuente, ruta)

    # --- a implementar por cada API --------------------------------------

    @abstractmethod
    def _pedir(
        self, identificador: str, desde: dt.date, hasta: dt.date, **kwargs: Any
    ) -> Any:
        """Pide un tramo y devuelve el JSON crudo. Lanza ErrorHTTP si falla."""

    @abstractmethod
    def _a_dataframe(
        self, crudo: Any, identificador: str, **kwargs: Any
    ) -> pd.DataFrame:
        """Convierte la respuesta cruda de un tramo a formato largo."""

    @abstractmethod
    def _max_dias(self, identificador: str, **kwargs: Any) -> int:
        """Maximo de dias por llamado para este identificador."""

    @abstractmethod
    def _marco_vacio(self) -> pd.DataFrame:
        """DataFrame vacio con las columnas que devuelve este cliente."""

    def _clave_extra(self, **kwargs: Any) -> str:
        """Parte de la clave de cache que depende de los kwargs. Vacia por defecto."""
        if not kwargs:
            return ""
        firma = json.dumps(kwargs, sort_keys=True, default=str)
        return "_" + hashlib.sha1(firma.encode("utf-8")).hexdigest()[:8]

    def _respuesta_json(self, respuesta: requests.Response, contexto: str) -> Any:
        """Valida el status y devuelve el JSON, o lanza ErrorHTTP con el cuerpo."""
        if respuesta.status_code >= 400:
            raise ErrorHTTP(respuesta.status_code, respuesta.text, contexto)
        try:
            return respuesta.json()
        except json.JSONDecodeError as exc:
            raise ErrorCliente(
                f"{contexto}: respuesta {respuesta.status_code} no es JSON: "
                f"{respuesta.text[:300]}"
            ) from exc


# --------------------------------------------------------------------------
# XM (SINERGOX)
# --------------------------------------------------------------------------


class ClienteXM(ClienteAPI):
    """Cliente de la API de XM.

    Estructura de la respuesta horaria, verificada contra el servidor:

        {"Metric": {"Id": "DemaReal", "Name": "Demanda Real por Sistema",
                    "StartDate": "...", "EndDate": "..."},
         "Items": [{"Date": "2025-01-01",
                    "HourlyEntities": [{"Id": "Sistema",
                                        "Values": {"code": "Sistema",
                                                   "Hour01": "7306335.34000",
                                                   ...,
                                                   "Hour24": "7925012.28000"}}]}]}

    Dos detalles que condicionan el parseo: los valores son cadenas, y un
    rango sin datos publicados devuelve 200 con `"Items": []`.
    """

    fuente = "xm"

    COLUMNAS = ["timestamp", "fuente", "identificador", "entidad", "valor"]

    # Claves de `Values` que no son horas.
    _NO_HORARIAS = frozenset({"code", "Code", "Id"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._catalogo: pd.DataFrame | None = None

    # --- catalogo ---------------------------------------------------------

    def catalogo(self) -> pd.DataFrame:
        """Inventario de metricas (POST /lists con ListadoMetricas).

        Se pide una sola vez por instancia. De aqui sale `MaxDays`, que varia
        por metrica: para DemaReal son 31 dias, no 30.
        """
        if self._catalogo is not None:
            return self._catalogo

        contexto = "xm catalogo de metricas"
        respuesta = self.sesion.post(
            f"{config.URL_XM}/lists",
            json={
                "MetricId": "ListadoMetricas",
                "StartDate": "2020-01-01",
                "EndDate": "2020-01-02",
                "Entity": "Sistema",
            },
            timeout=self.timeout,
        )
        datos = self._respuesta_json(respuesta, contexto)

        filas = [
            entidad["Values"]
            for item in datos.get("Items", [])
            for entidad in item.get("ListEntities", [])
        ]
        if not filas:
            raise ErrorCliente("El catalogo de metricas de XM vino vacio")

        self._catalogo = pd.DataFrame(filas)
        log.info("xm: catalogo con %d metricas", len(self._catalogo))
        return self._catalogo

    def _max_dias(self, identificador: str, **kwargs: Any) -> int:
        """Lee MaxDays del catalogo en vez de asumir un limite fijo."""
        entidad = kwargs.get("entidad", config.ENTIDAD_OBJETIVO_XM)
        try:
            catalogo = self.catalogo()
        except ErrorCliente as exc:
            log.warning(
                "xm: no se pudo leer el catalogo (%s); se usan %d dias por llamado",
                exc,
                MAX_DIAS_POR_DEFECTO,
            )
            return MAX_DIAS_POR_DEFECTO

        coincide = catalogo[
            (catalogo["MetricId"] == identificador) & (catalogo["Entity"] == entidad)
        ]
        if coincide.empty:
            log.warning(
                "xm: %s/%s no esta en el catalogo; se usan %d dias por llamado",
                identificador,
                entidad,
                MAX_DIAS_POR_DEFECTO,
            )
            return MAX_DIAS_POR_DEFECTO

        return int(coincide["MaxDays"].iloc[0])

    # --- consulta ---------------------------------------------------------

    def _pedir(
        self, identificador: str, desde: dt.date, hasta: dt.date, **kwargs: Any
    ) -> Any:
        """POST a /{granularidad} con el cuerpo que espera la API."""
        entidad = kwargs.get("entidad", config.ENTIDAD_OBJETIVO_XM)
        granularidad = kwargs.get("granularidad", "hourly")

        respuesta = self.sesion.post(
            f"{config.URL_XM}/{granularidad}",
            json={
                "MetricId": identificador,
                "StartDate": ventanas.a_iso(desde),
                "EndDate": ventanas.a_iso(hasta),
                "Entity": entidad,
            },
            timeout=self.timeout,
        )
        return self._respuesta_json(
            respuesta, f"xm {identificador}/{entidad} {desde}..{hasta}"
        )

    def _a_dataframe(
        self, crudo: Any, identificador: str, **kwargs: Any
    ) -> pd.DataFrame:
        """Pasa de HourNN (formato ancho) a una fila por hora (formato largo).

        La correspondencia entre HourNN y la hora de reloj es
        `config.DESFASE_HORA_XM`, la unica constante del proyecto que codifica
        esa convencion. Ver su documentacion en config.py: esta verificada
        empiricamente, pero si hubiera que corregirla se corrige alli y este
        codigo la sigue.

        **Dimensiones extra.** No todas las metricas tienen la misma forma. La
        mayoria trae en `Values` solo `code` y las 24 horas, y su dimension es
        el `Id` de la entidad. Pero `DemaComeNoReg`/`Entity=CIIU` mete ademas
        `Activity` y `Subactivity` dentro de `Values`, y deja `Id` fijo en
        "CIIU": para dos dias devuelve 349 combinaciones actividad/subactividad.
        Descartar esas claves, como se hacia antes, dejaba 16 704 filas con la
        misma marca de tiempo y sin forma de saber a que actividad pertenecia
        cada una. Ahora cualquier clave no horaria distinta de `code`/`Id` se
        conserva como columna propia.
        """
        filas: list[dict[str, Any]] = []
        dimensiones_extra: set[str] = set()

        for item in crudo.get("Items") or []:
            fecha = dt.date.fromisoformat(str(item["Date"])[:10])

            for entidad in item.get("HourlyEntities", []):
                valores = entidad.get("Values", {})
                codigo = entidad.get("Id") or valores.get("code") or "desconocida"

                extra = {
                    clave: bruto
                    for clave, bruto in valores.items()
                    if clave not in self._NO_HORARIAS and not clave.startswith("Hour")
                }
                dimensiones_extra.update(extra)

                for clave, bruto in valores.items():
                    if clave in self._NO_HORARIAS or not clave.startswith("Hour"):
                        continue

                    hora = int(clave[4:]) - config.DESFASE_HORA_XM
                    if not 0 <= hora <= 23:
                        log.warning("xm: hora fuera de rango en %s: %s", fecha, clave)
                        continue

                    filas.append(
                        {
                            "timestamp": dt.datetime.combine(fecha, dt.time(hora)),
                            "fuente": self.fuente,
                            "identificador": identificador,
                            "entidad": str(codigo),
                            "valor": _a_float(bruto),
                            **extra,
                        }
                    )

        if dimensiones_extra:
            log.info(
                "xm %s: la metrica trae dimensiones extra en Values, se conservan "
                "como columnas: %s",
                identificador,
                sorted(dimensiones_extra),
            )

        if not filas:
            return self._marco_vacio()

        # Las columnas fijas primero y las dimensiones extra despues: fijar
        # `columns=self.COLUMNAS` las habria descartado en silencio.
        columnas = self.COLUMNAS + sorted(dimensiones_extra)
        marco = pd.DataFrame(filas, columns=columnas)
        marco["timestamp"] = config.a_zona_colombia(marco["timestamp"])
        return marco

    def _marco_vacio(self) -> pd.DataFrame:
        vacio = pd.DataFrame(columns=self.COLUMNAS)
        vacio = vacio.astype({"valor": "float64"})
        vacio["timestamp"] = pd.Series(dtype=f"datetime64[ns, {config.ZONA_COLOMBIA}]")
        return vacio

    def _clave_extra(self, **kwargs: Any) -> str:
        """La entidad y la granularidad forman parte de la identidad del tramo."""
        entidad = kwargs.get("entidad", config.ENTIDAD_OBJETIVO_XM)
        granularidad = kwargs.get("granularidad", "hourly")
        return f"_{entidad}_{granularidad}"


# --------------------------------------------------------------------------
# SIMEM
# --------------------------------------------------------------------------


class ClienteSIMEM(ClienteAPI):
    """Cliente de la API de SIMEM.

    Estructura de la respuesta, verificada contra el servidor:

        {"parameters": {...},
         "success": true,
         "result": {"idDataset": "14fabb", "name": "Demanda real nacional",
                    "metadata": {...}, "records": [...],
                    "variables": [...], "columns": null, "tags": [...]}}

    `result.records` ya viene en formato largo, una fila por combinacion de
    dimensiones y hora:

        {"CodigoVariable": "DdaReal", "FechaHora": "2026-08-05 01:00:00",
         "CodigoSICAgente": "CHVC", "TipoMercado": "No Regulado",
         "Version": "TX2", "Valor": 62463.18, "UnidadMedida": "kWh",
         "CodigoDuracion": "PT1H"}

    Las columnas de dimension cambian segun el dataset, asi que se devuelven
    tal cual llegan. En particular `Version` se conserva sin tocar: sobre una
    misma FechaHora pueden coexistir varias versiones de liquidacion, y
    colapsarlas es responsabilidad de `normalizar`, no de esta capa.
    """

    fuente = "simem"

    COLUMNAS_BASE = ["timestamp", "fuente", "identificador"]

    # --- catalogo ---------------------------------------------------------

    def catalogo(self) -> pd.DataFrame:
        """Catalogo de conjuntos de datos publicados en SIMEM.

        El catalogo es a su vez un dataset (config.DATASET_CATALOGO_SIMEM), y
        es la unica forma de descubrir el id de 6 caracteres de un conjunto.
        No pasa por `consultar` porque sus registros no tienen `FechaHora`.
        """
        contexto = "simem catalogo de datasets"
        crudo = self._pedir_con_reintentos(
            config.DATASET_CATALOGO_SIMEM, dt.date(1990, 1, 1), dt.date.today()
        )
        registros = (crudo.get("result") or {}).get("records") or []
        if not registros:
            raise ErrorCliente(f"{contexto}: vino vacio")

        catalogo = pd.DataFrame(registros)
        log.info("simem: catalogo con %d conjuntos de datos", len(catalogo))
        return catalogo

    def buscar(self, texto: str) -> pd.DataFrame:
        """Busca conjuntos cuyo nombre contenga `texto`."""
        catalogo = self.catalogo()
        coincide = catalogo["nombreConjuntoDatos"].str.contains(
            texto, case=False, na=False
        )
        return catalogo.loc[
            coincide, ["idDataset", "nombreConjuntoDatos", "inicioDato", "finDato"]
        ].reset_index(drop=True)

    # --- consulta ---------------------------------------------------------

    def _max_dias(self, identificador: str, **kwargs: Any) -> int:
        """SIMEM no publica el limite por dataset; el documentado es 31 dias."""
        return int(kwargs.get("max_dias", config.MAX_DIAS_SIMEM))

    def _pedir(
        self, identificador: str, desde: dt.date, hasta: dt.date, **kwargs: Any
    ) -> Any:
        """GET a PublicData con datasetId y el rango en ISO."""
        respuesta = self.sesion.get(
            config.URL_SIMEM,
            params={
                "datasetId": identificador,
                "startDate": ventanas.a_iso(desde),
                "endDate": ventanas.a_iso(hasta),
            },
            timeout=self.timeout,
        )
        contexto = f"simem {identificador} {desde}..{hasta}"
        datos = self._respuesta_json(respuesta, contexto)

        # SIMEM puede responder 200 con success=False.
        if not datos.get("success", False):
            raise ErrorCliente(
                f"{contexto}: la API respondio success=False -- "
                f"{datos.get('message', '(sin mensaje)')}"
            )
        return datos

    def _a_dataframe(
        self, crudo: Any, identificador: str, **kwargs: Any
    ) -> pd.DataFrame:
        """Convierte `result.records` a DataFrame anadiendo las columnas comunes."""
        registros = (crudo.get("result") or {}).get("records") or []
        if not registros:
            return self._marco_vacio()

        marco = pd.DataFrame(registros)

        if "FechaHora" not in marco.columns:
            raise ErrorCliente(
                f"simem {identificador}: los registros no traen FechaHora. "
                f"Columnas recibidas: {list(marco.columns)}"
            )

        marco.insert(0, "timestamp", config.a_zona_colombia(marco["FechaHora"]))
        marco.insert(1, "fuente", self.fuente)
        marco.insert(2, "identificador", identificador)
        return marco

    def _marco_vacio(self) -> pd.DataFrame:
        vacio = pd.DataFrame(columns=self.COLUMNAS_BASE)
        vacio["timestamp"] = pd.Series(dtype=f"datetime64[ns, {config.ZONA_COLOMBIA}]")
        return vacio

    def _clave_extra(self, **kwargs: Any) -> str:
        """Ningun kwarg de SIMEM cambia los datos devueltos, solo el troceado."""
        return ""


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------


def _a_fecha(valor: dt.date | dt.datetime | str) -> dt.date:
    """Normaliza fechas dadas como date, datetime o cadena ISO."""
    if isinstance(valor, dt.datetime):
        return valor.date()
    if isinstance(valor, dt.date):
        return valor
    return dt.date.fromisoformat(str(valor)[:10])


def _a_float(bruto: Any) -> float:
    """Convierte un valor de XM a float; ausente, nulo o vacio dan NaN.

    Nunca devuelve 0 por defecto: un cero es un dato real y un hueco no lo es.
    """
    if bruto is None:
        return float("nan")
    texto = str(bruto).strip()
    if not texto:
        return float("nan")
    try:
        return float(texto)
    except ValueError:
        log.warning("xm: valor no numerico %r", bruto)
        return float("nan")
