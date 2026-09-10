"""CLI de descarga incremental a la capa cruda.

    python descargar.py --fuente xm --id DemaReal
    python descargar.py --fuente xm --id DemaReal --desde 2021-01-01
    python descargar.py --fuente simem --id 14fabb --desde 2025-01-01 --hasta 2025-03-31
    python descargar.py --fuente xm --id DemaReal --forzar

Sin `--desde` la descarga es incremental: mira la fecha maxima ya almacenada y
pide desde ahi hacia atras una ventana de solape, para recoger revisiones. Es
idempotente, asi que ejecutarlo dos veces seguidas no duplica nada.

Ademas:

    python descargar.py --historial                 # bitacora de descargas
    python descargar.py --estado --fuente xm --id DemaReal
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ingesta import config, descarga  # noqa: E402

log = logging.getLogger("descargar")


def _fecha(texto: str) -> dt.date:
    """Convierte un argumento de linea de comandos a fecha."""
    if texto == "hoy":
        return dt.date.today()
    return dt.date.fromisoformat(texto)


def _construir_analizador() -> argparse.ArgumentParser:
    """Define los argumentos del CLI."""
    analizador = argparse.ArgumentParser(
        description="Descarga incremental a la capa cruda particionada",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Identificadores habituales:\n"
            "  --fuente xm     --id DemaReal   (metrica; ver scripts/verificar_apis.py)\n"
            "  --fuente simem  --id 14fabb     (datasetId de 6 caracteres)\n"
        ),
    )
    analizador.add_argument(
        "--fuente", choices=sorted(descarga.CLIENTES), help="API de origen"
    )
    analizador.add_argument("--id", dest="identificador", help="MetricId o datasetId")
    analizador.add_argument(
        "--desde",
        type=_fecha,
        default=None,
        help="fecha inicial YYYY-MM-DD; si se omite, se calcula de forma incremental",
    )
    analizador.add_argument(
        "--hasta", type=_fecha, default=None, help="fecha final YYYY-MM-DD (por defecto hoy)"
    )
    analizador.add_argument(
        "--forzar",
        action="store_true",
        help="ignora el estado almacenado y la cache, y vuelve a pedirlo todo a la API",
    )
    analizador.add_argument(
        "--entidad",
        default=config.ENTIDAD_OBJETIVO_XM,
        help="Entity de XM (Sistema, Agente...); se ignora para SIMEM",
    )
    analizador.add_argument(
        "--historial", action="store_true", help="muestra la bitacora y termina"
    )
    analizador.add_argument(
        "--estado", action="store_true", help="muestra que hay almacenado y termina"
    )
    analizador.add_argument("--verboso", action="store_true")
    return analizador


def mostrar_historial(fuente: str | None, identificador: str | None) -> int:
    """Imprime la bitacora de descargas registrada en el manifiesto."""
    tabla = descarga.historial(fuente, identificador)
    if tabla.empty:
        print("La bitacora esta vacia.")
        return 0

    columnas = [
        "timestamp_ejecucion",
        "fuente",
        "identificador",
        "modo",
        "rango_obtenido",
        "n_registros",
        "n_registros_total",
    ]
    print(tabla[columnas].to_string(index=False))
    print(f"\n{len(tabla)} descargas registradas en {config.RUTA_MANIFIESTO_CRUDO}")
    return 0


def mostrar_estado(fuente: str, identificador: str) -> int:
    """Imprime que hay almacenado ahora mismo para un conjunto."""
    ruta = descarga.ruta_dataset(fuente, identificador)
    ultima = descarga.ultima_fecha_almacenada(fuente, identificador)
    total = descarga.n_registros_almacenados(fuente, identificador)

    print(f"Conjunto   : {fuente}/{identificador}")
    print(f"Ruta       : {ruta}")
    print(f"Existe     : {'si' if ruta.exists() else 'no'}")
    print(f"Registros  : {total:,}")
    print(f"Hasta      : {ultima}")
    if ultima:
        print(f"Rezago     : {(dt.date.today() - ultima).days} dias")

    particiones = sorted(p.parent.name for p in ruta.rglob("datos.parquet"))
    if particiones:
        print(f"Particiones: {len(particiones)} ({particiones[0]} .. {particiones[-1]})")
    return 0


def resumir(entrada: dict) -> None:
    """Imprime el resultado de una descarga."""
    print(f"\n=== {entrada['fuente']}/{entrada['identificador']} ===")
    print(f"  modo              : {entrada['modo']}")
    print(f"  rango solicitado  : {' .. '.join(entrada['rango_solicitado'])}")
    obtenido = entrada["rango_obtenido"]
    print(f"  rango obtenido    : {' .. '.join(obtenido) if obtenido else '(sin datos)'}")
    print(f"  registros nuevos  : {entrada['n_registros']:,}")
    print(f"  registros totales : {entrada['n_registros_total']:,}")
    print(f"  hash resultado    : {entrada['hash_resultado']}")
    print(f"  particiones       : {len(entrada['particiones_escritas'])}")
    print(f"  duracion          : {entrada['duracion_segundos']} s")


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del CLI."""
    analizador = _construir_analizador()
    args = analizador.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verboso else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.historial:
        return mostrar_historial(args.fuente, args.identificador)

    if not args.fuente or not args.identificador:
        analizador.error("--fuente y --id son obligatorios (salvo con --historial)")

    if args.estado:
        return mostrar_estado(args.fuente, args.identificador)

    # `entidad` solo tiene sentido para XM; pasarselo a SIMEM cambiaria la
    # clave de cache sin cambiar los datos.
    extra = {"entidad": args.entidad} if args.fuente == "xm" else {}

    entrada = descarga.descargar(
        fuente=args.fuente,
        identificador=args.identificador,
        desde=args.desde,
        hasta=args.hasta,
        forzar=args.forzar,
        **extra,
    )
    resumir(entrada)
    print(f"\nManifiesto: {config.RUTA_MANIFIESTO_CRUDO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
