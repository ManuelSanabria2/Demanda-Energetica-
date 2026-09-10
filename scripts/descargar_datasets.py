"""Genera la carpeta `datasets/` con los CSV de trabajo.

Reutiliza los clientes de API ya verificados (troceado del rango, reintentos con
backoff y cache en disco), asi que reejecutarlo es barato: lo que ya esta
cacheado no se vuelve a pedir.

    python scripts/descargar_datasets.py                     # todo
    python scripts/descargar_datasets.py --fuente xm --sin-ciiu
    python scripts/descargar_datasets.py --desde 2024-01-01

Que produce:

    datasets/simem/simem_14fabb_{anio}.csv    demanda desagregada, un CSV por anio
    datasets/simem/simem_14fabb_nacional.csv  serie horaria nacional agregada
    datasets/xm/xm_catalogo_metricas.csv      las 193 metricas del catalogo
    datasets/xm/xm_demareal_sistema.csv       demanda real del SIN
    datasets/xm/xm_demacome_sistema.csv       demanda comercial del SIN
    datasets/xm/ciiu/xm_demacomenoreg_ciiu_{anio}.csv.gz

**Memoria acotada.** Los conjuntos desagregados no se cargan enteros: un anio
de SIMEM llega a 2,1 millones de filas de texto, y la primera version de este
script, que lo cargaba de golpe, se quedo sin memoria. Ahora cada anio se
escribe tramo a tramo -- ventanas de 31 dias, las mismas que usa la cache del
cliente -- y en memoria nunca hay mas que un tramo.

CIIU va comprimido porque son ~349 subactividades por hora: unos 17 millones de
filas en el historico, que en CSV plano pasarian de 2,5 GB por lo largas y
repetidas que son las descripciones de actividad. `pandas.read_csv` abre los
.csv.gz igual que un CSV.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import gzip
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from ingesta import config, normalizar, ventanas  # noqa: E402
from ingesta.clientes import ClienteAPI, ClienteSIMEM, ClienteXM  # noqa: E402

log = logging.getLogger("datasets")

DIR_DATASETS = RAIZ / "datasets"
RUTA_MANIFIESTO = DIR_DATASETS / "manifiesto_datasets.json"

# Metricas de XM que se bajan como datos, no solo como catalogo.
METRICAS_XM = [
    ("DemaReal", "Sistema", "xm_demareal_sistema"),
    ("DemaCome", "Sistema", "xm_demacome_sistema"),
]
METRICA_CIIU = ("DemaComeNoReg", "CIIU", "xm_demacomenoreg_ciiu")

# Orden de escritura de cada conjunto. Fijarlo hace que dos ejecuciones sobre
# los mismos datos produzcan el mismo archivo byte a byte, y por tanto el mismo
# hash en el manifiesto, sin depender del orden en que la API devuelve filas.
ORDEN_SIMEM = ["timestamp", "CodigoSICAgente", "TipoMercado", "Version"]
ORDEN_CIIU = ["timestamp", "Activity", "Subactivity"]

# CIIU se pide en tramos de 7 dias aunque el catalogo permita 31. Medido el
# 2026-09-10: con el servidor libre una semana tarda 2-3 s, pero con el servidor
# cargado una semana llego a 34 s (1,6 MB) y una peticion de 3 dias agoto 300 s.
# A ese ritmo un tramo de 31 dias rebasaria el timeout de 180 s del cliente.
# Tramos cortos dejan margen y, si algo falla, se reanuda desde la cache
# perdiendo solo una semana.
DIAS_POR_TRAMO_CIIU = 7


# --------------------------------------------------------------------------
# Escritura incremental
# --------------------------------------------------------------------------


class EscritorCSV:
    """Escribe un CSV por partes y resume lo escrito al cerrar.

    Mantiene el recuento de filas y el rango temporal sin retener los datos, y
    calcula el hash del archivo final leyendolo en bloques.
    """

    def __init__(self, ruta: Path, comprimir: bool = False, orden: list[str] | None = None):
        if comprimir and ruta.suffix != ".gz":
            ruta = ruta.with_suffix(ruta.suffix + ".gz")
        self.ruta = ruta
        # Se escribe a un temporal y se renombra al cerrar: un archivo a medio
        # escribir (proceso interrumpido, fallo de red) nunca tiene el nombre
        # final, asi que nadie puede tomarlo por un dataset completo.
        self._temporal = ruta.with_name(ruta.name + ".tmp")
        self.orden = orden or []
        self.columnas: list[str] | None = None
        self.n_filas = 0
        self.minimo: pd.Timestamp | None = None
        self.maximo: pd.Timestamp | None = None
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self._archivo = (
            gzip.open(self._temporal, "wt", encoding="utf-8", newline="")
            if ruta.suffix == ".gz"
            else open(self._temporal, "w", encoding="utf-8", newline="")
        )

    def escribir(self, tabla: pd.DataFrame) -> None:
        if tabla.empty:
            return
        if self.columnas is None:
            self.columnas = list(tabla.columns)
        faltan = set(self.columnas) ^ set(tabla.columns)
        if faltan:
            # Un tramo con columnas distintas desalinearia el CSV sin avisar.
            raise RuntimeError(
                f"{self.ruta.name}: un tramo trae columnas distintas del resto: {sorted(faltan)}"
            )

        claves = [c for c in self.orden if c in tabla.columns]
        if claves:
            tabla = tabla.sort_values(claves, kind="stable")

        tabla[self.columnas].to_csv(
            self._archivo, index=False, header=self.n_filas == 0
        )
        self.n_filas += len(tabla)

        if "timestamp" in tabla.columns:
            momentos = pd.to_datetime(tabla["timestamp"])
            self.minimo = min(filter(None, [self.minimo, momentos.min()]))
            self.maximo = max(filter(None, [self.maximo, momentos.max()]))

    def cerrar(self) -> dict[str, Any] | None:
        """Cierra el archivo y devuelve su entrada de manifiesto (o None si vacio)."""
        self._archivo.close()
        if self.n_filas == 0:
            self._temporal.unlink(missing_ok=True)
            log.warning("%s: sin datos, no se crea", self.ruta.name)
            return None
        self._temporal.replace(self.ruta)

        resumen = hashlib.sha256()
        with open(self.ruta, "rb") as origen:
            for bloque in iter(lambda: origen.read(1 << 20), b""):
                resumen.update(bloque)

        tamano = self.ruta.stat().st_size
        log.info("%s: %d filas, %.1f MB", self.ruta.name, self.n_filas, tamano / 1e6)
        return {
            "archivo": self.ruta.relative_to(RAIZ).as_posix(),
            "n_filas": self.n_filas,
            "columnas": self.columnas,
            "bytes": tamano,
            "rango": [str(self.minimo), str(self.maximo)] if self.minimo is not None else None,
            "sha256_archivo": resumen.hexdigest(),
            "comprimido": self.ruta.suffix == ".gz",
        }

    def descartar(self) -> None:
        """Cierra y borra el temporal sin publicar nada."""
        self._archivo.close()
        self._temporal.unlink(missing_ok=True)


def escribir_entero(
    tabla: pd.DataFrame, ruta: Path, orden: list[str] | None = None
) -> dict[str, Any] | None:
    """Atajo para tablas pequenas que caben en memoria sin problema."""
    escritor = EscritorCSV(ruta, orden=orden)
    escritor.escribir(tabla)
    return escritor.cerrar()


def _anios(desde: dt.date, hasta: dt.date) -> list[tuple[int, dt.date, dt.date]]:
    """Anios del rango, cada uno recortado a los limites pedidos."""
    return [
        (anio, max(dt.date(anio, 1, 1), desde), min(dt.date(anio, 12, 31), hasta))
        for anio in range(desde.year, hasta.year + 1)
    ]


def _por_tramos(
    cliente: ClienteAPI,
    identificador: str,
    inicio: dt.date,
    fin: dt.date,
    max_dias: int,
    **kwargs: Any,
):
    """Consulta un rango tramo a tramo, cediendo cada tramo en cuanto llega.

    Los cortes son los mismos que haria el propio cliente, asi que lo que ya
    esta en su cache se sirve desde disco.
    """
    for desde, hasta in ventanas.partir_rango(inicio, fin, max_dias):
        tabla = cliente.consultar(identificador, desde, hasta, **kwargs)
        if not tabla.empty:
            yield tabla


# --------------------------------------------------------------------------
# SIMEM
# --------------------------------------------------------------------------


def descargar_simem(desde: dt.date, hasta: dt.date, entradas: list[dict[str, Any]]) -> None:
    """Un CSV por anio con el detalle, mas la serie nacional agregada."""
    cliente = ClienteSIMEM()
    destino = DIR_DATASETS / "simem"
    nacionales: list[pd.DataFrame] = []

    for anio, inicio, fin in _anios(desde, hasta):
        log.info("SIMEM %s: %s..%s", anio, inicio, fin)
        escritor = EscritorCSV(destino / f"simem_14fabb_{anio}.csv", orden=ORDEN_SIMEM)

        for tramo in _por_tramos(
            cliente, config.DATASET_DEMANDA_SIMEM, inicio, fin, config.MAX_DIAS_SIMEM
        ):
            escritor.escribir(tramo)
            # El agregado nacional colapsa versiones: sumar sin eso multiplica la
            # demanda. El colapso se decide por FechaHora, asi que hacerlo tramo
            # a tramo da exactamente lo mismo que sobre el anio entero.
            nacionales.append(normalizar.simem_agregar_nacional(tramo))
            del tramo
            gc.collect()

        entrada = escritor.cerrar()
        if entrada:
            entradas.append(entrada)

    if nacionales:
        nacional = pd.concat(nacionales, ignore_index=True)
        nacional = nacional.drop_duplicates(subset=["fecha_hora"]).sort_values("fecha_hora")
        entrada = escribir_entero(
            nacional.reset_index(drop=True), destino / "simem_14fabb_nacional.csv"
        )
        if entrada:
            entradas.append(entrada)


# --------------------------------------------------------------------------
# XM
# --------------------------------------------------------------------------


def descargar_xm(
    desde: dt.date,
    hasta: dt.date,
    entradas: list[dict[str, Any]],
    incluir_ciiu: bool = True,
    comprimir_ciiu: bool = True,
) -> None:
    """Catalogo de metricas mas los datos de las metricas de demanda."""
    cliente = ClienteXM()
    destino = DIR_DATASETS / "xm"

    for entrada in (
        escribir_entero(cliente.catalogo(), destino / "xm_catalogo_metricas.csv"),
    ):
        if entrada:
            entradas.append(entrada)

    for metrica, entidad, nombre in METRICAS_XM:
        log.info("XM %s/%s: %s..%s", metrica, entidad, desde, hasta)
        tabla = cliente.consultar(metrica, desde, hasta, entidad=entidad)
        entrada = escribir_entero(tabla, destino / f"{nombre}.csv", orden=["timestamp"])
        if entrada:
            entradas.append(entrada)

    if incluir_ciiu:
        descargar_ciiu(cliente, desde, hasta, entradas, destino, comprimir_ciiu)


def descargar_ciiu(
    cliente: ClienteXM,
    desde: dt.date,
    hasta: dt.date,
    entradas: list[dict[str, Any]],
    destino: Path,
    comprimir: bool,
) -> None:
    """Demanda comercial no regulada por CIIU, un archivo por anio, por tramos."""
    metrica, entidad, nombre = METRICA_CIIU
    max_dias = min(cliente._max_dias(metrica, entidad=entidad), DIAS_POR_TRAMO_CIIU)

    for anio, inicio, fin in _anios(desde, hasta):
        log.info("XM %s/%s %s: %s..%s", metrica, entidad, anio, inicio, fin)
        escritor = EscritorCSV(
            destino / "ciiu" / f"{nombre}_{anio}.csv", comprimir=comprimir, orden=ORDEN_CIIU
        )

        for tramo in _por_tramos(cliente, metrica, inicio, fin, max_dias, entidad=entidad):
            faltan = [c for c in ("Activity", "Subactivity") if c not in tramo.columns]
            if faltan:
                escritor.descartar()
                raise RuntimeError(
                    f"La respuesta de CIIU no trae {faltan}. Sin esas columnas las "
                    "filas no se distinguen entre si; revisa ClienteXM._a_dataframe."
                )
            escritor.escribir(tramo)
            del tramo
            gc.collect()

        entrada = escritor.cerrar()
        if entrada:
            entradas.append(entrada)


# --------------------------------------------------------------------------


def guardar_manifiesto(entradas: list[dict[str, Any]], argumentos: dict) -> None:
    """Bitacora append-only de cada generacion, con el hash de cada archivo."""
    DIR_DATASETS.mkdir(parents=True, exist_ok=True)
    manifiesto: dict[str, Any] = {"version_formato": 2, "generaciones": []}
    if RUTA_MANIFIESTO.exists():
        with RUTA_MANIFIESTO.open(encoding="utf-8") as origen:
            manifiesto = json.load(origen)

    manifiesto["generaciones"].append(
        {
            "momento": dt.datetime.now().isoformat(timespec="seconds"),
            "argumentos": argumentos,
            "entorno": {"python": sys.version.split()[0], "pandas": pd.__version__},
            "archivos": entradas,
            "bytes_totales": sum(e["bytes"] for e in entradas),
        }
    )

    temporal = RUTA_MANIFIESTO.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as salida:
        json.dump(manifiesto, salida, ensure_ascii=False, indent=2, default=str)
    temporal.replace(RUTA_MANIFIESTO)


def _fecha(texto: str) -> dt.date:
    if texto == "hoy":
        return dt.date.today()
    return dt.date.fromisoformat(texto)


def main(argv: list[str] | None = None) -> int:
    """Genera los CSV pedidos y anota la generacion en el manifiesto."""
    analizador = argparse.ArgumentParser(description="Genera datasets/ en CSV")
    analizador.add_argument("--fuente", choices=("simem", "xm", "todas"), default="todas")
    analizador.add_argument("--desde", type=_fecha, default=config.FECHA_INICIO_DEFECTO)
    analizador.add_argument("--hasta", type=_fecha, default=dt.date.today())
    analizador.add_argument(
        "--sin-ciiu", action="store_true", help="omite CIIU, que es la descarga larga"
    )
    analizador.add_argument(
        "--solo-ciiu", action="store_true", help="solo CIIU, sin catalogo ni metricas de Sistema"
    )
    analizador.add_argument(
        "--sin-comprimir",
        action="store_true",
        help="CIIU en CSV plano en vez de .csv.gz (ocupa unas diez veces mas)",
    )
    args = analizador.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    entradas: list[dict[str, Any]] = []
    if args.fuente in ("simem", "todas") and not args.solo_ciiu:
        descargar_simem(args.desde, args.hasta, entradas)
    if args.fuente in ("xm", "todas"):
        if args.solo_ciiu:
            descargar_ciiu(
                ClienteXM(), args.desde, args.hasta, entradas,
                DIR_DATASETS / "xm", not args.sin_comprimir,
            )
        else:
            descargar_xm(
                args.desde, args.hasta, entradas,
                incluir_ciiu=not args.sin_ciiu,
                comprimir_ciiu=not args.sin_comprimir,
            )

    guardar_manifiesto(
        entradas,
        {
            "fuente": args.fuente,
            "desde": args.desde.isoformat(),
            "hasta": args.hasta.isoformat(),
            "sin_ciiu": args.sin_ciiu,
            "solo_ciiu": args.solo_ciiu,
        },
    )

    print(f"\n=== {len(entradas)} archivos en {DIR_DATASETS} ===")
    for entrada in entradas:
        print(f"  {entrada['n_filas']:>11,} filas  {entrada['bytes']/1e6:>8.1f} MB  {entrada['archivo']}")
    print(f"\n  total: {sum(e['bytes'] for e in entradas)/1e6:.1f} MB")
    print(f"  manifiesto: {RUTA_MANIFIESTO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
