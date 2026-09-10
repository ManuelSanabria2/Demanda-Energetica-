"""Punto de entrada de la ingesta.

    python -m ingesta.cli --fuente todas --desde 2021-01-01
    python -m ingesta.cli --fuente xm --desde 2025-01-01 --hasta 2025-01-02

La descarga la hace la capa de clientes (`clientes.py`), que se encarga de
fragmentar el rango, reintentar y cachear. Aqui solo queda la orquestacion:
normalizar al esquema comun, medir cobertura, escribir Parquet y registrar el
manifiesto.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd
import pyarrow as pa

from . import config, manifiesto, normalizar, particiones
from .clientes import ClienteSIMEM, ClienteXM

log = logging.getLogger("ingesta")

ESQUEMA = pa.schema(
    [
        ("fecha_hora", pa.timestamp("ns")),
        ("fuente", pa.string()),
        ("metrica", pa.string()),
        ("entidad", pa.string()),
        ("valor_kwh", pa.float64()),
        ("version", pa.string()),
        ("anio", pa.int32()),
        ("mes", pa.int32()),
    ]
)


# Identifica una observacion en el esquema procesado. Si dos filas coinciden en
# estas columnas son la misma hora de la misma serie, y la mas reciente sustituye
# a la anterior.
CLAVE_PROCESADA = ["fecha_hora", "fuente", "metrica", "entidad"]


def escribir_parquet(tabla: pd.DataFrame, subcarpeta: str) -> int:
    """Escribe la serie en Parquet particionado por anio y mes.

    La fusion con lo ya almacenado la hace `particiones.escribir_particionado`:
    leer, combinar, deduplicar y reescribir la particion completa, en vez de
    borrarla. Ver alli por que.
    """
    return particiones.escribir_particionado(
        tabla,
        config.DIR_PROCESADO / subcarpeta,
        ESQUEMA,
        CLAVE_PROCESADA,
        etiqueta=subcarpeta,
    )


def ingestar_xm(
    desde: dt.date, hasta: dt.date, usar_cache: bool = True
) -> pd.DataFrame:
    """Descarga, normaliza y persiste la demanda real del sistema desde XM."""
    cliente = ClienteXM(usar_cache=usar_cache)
    crudo = cliente.consultar(
        config.METRICA_OBJETIVO_XM,
        desde,
        hasta,
        entidad=config.ENTIDAD_OBJETIVO_XM,
    )
    tabla = normalizar.xm_cliente_a_esquema_comun(crudo)
    ultima = _ultima_fecha(tabla)

    # La cobertura se mide hasta la ultima fecha publicada, no hasta la pedida:
    # los dias aun no publicados son rezago conocido, no huecos.
    cobertura = normalizar.reporte_cobertura(tabla, inicio=desde, fin=ultima or hasta)
    escribir_parquet(tabla, "xm_demanda_real_sistema")

    manifiesto.registrar(
        clave=f"xm:{config.METRICA_OBJETIVO_XM}:{config.ENTIDAD_OBJETIVO_XM}",
        rango_solicitado=(desde, hasta),
        ultima_fecha_con_datos=ultima,
        n_registros=len(tabla),
        cobertura=cobertura,
    )
    _resumir("XM DemaReal/Sistema", ultima, cobertura)
    return tabla


def ingestar_simem(
    desde: dt.date, hasta: dt.date, usar_cache: bool = True
) -> pd.DataFrame:
    """Descarga la demanda real nacional de SIMEM y la agrega a serie horaria.

    El colapso de versiones de liquidacion lo hace `normalizar`, no el cliente:
    es la decision mas delicada del proyecto y no debe quedar escondida dentro
    de una descarga.
    """
    cliente = ClienteSIMEM(usar_cache=usar_cache)
    crudo = cliente.consultar(config.DATASET_DEMANDA_SIMEM, desde, hasta)
    tabla = normalizar.simem_agregar_nacional(crudo)
    ultima = _ultima_fecha(tabla)

    cobertura = normalizar.reporte_cobertura(tabla, inicio=desde, fin=ultima or hasta)
    escribir_parquet(tabla, "simem_demanda_real_nacional")

    manifiesto.registrar(
        clave=f"simem:{config.DATASET_DEMANDA_SIMEM}:Nacional",
        rango_solicitado=(desde, hasta),
        ultima_fecha_con_datos=ultima,
        n_registros=len(tabla),
        cobertura=cobertura,
    )
    _resumir("SIMEM 14fabb/Nacional", ultima, cobertura)
    return tabla


def _ultima_fecha(tabla: pd.DataFrame) -> dt.date | None:
    """Ultima fecha con dato de una tabla en el esquema comun."""
    if tabla.empty:
        return None
    return pd.to_datetime(tabla["fecha_hora"]).max().date()


def _resumir(etiqueta: str, ultima: dt.date | None, cobertura: dict) -> None:
    """Imprime el resumen de una ingesta, incluido el rezago de publicacion."""
    rezago = (dt.date.today() - ultima).days if ultima else None
    print(f"\n=== {etiqueta} ===")
    print(f"  filas                : {cobertura['n_filas']}")
    print(f"  ultima fecha con dato: {ultima}  (rezago: {rezago} dias)")
    print(f"  horas esperadas      : {cobertura['horas_esperadas']}")
    print(f"  horas faltantes      : {cobertura['horas_faltantes']}")
    print(f"  horas con NaN        : {cobertura['horas_con_nan']}")
    print(f"  duplicados           : {cobertura['duplicados']}")
    if cobertura["primeros_huecos"]:
        print(f"  primeros huecos      : {cobertura['primeros_huecos'][:5]}")


def _fecha(texto: str) -> dt.date:
    """Convierte un argumento de linea de comandos a fecha."""
    if texto == "hoy":
        return dt.date.today()
    return dt.date.fromisoformat(texto)


def main(argv: list[str] | None = None) -> int:
    """Orquesta la ingesta segun los argumentos de linea de comandos."""
    analizador = argparse.ArgumentParser(description="Ingesta de demanda horaria del SIN")
    analizador.add_argument(
        "--fuente", choices=("xm", "simem", "todas"), default="todas"
    )
    analizador.add_argument("--desde", type=_fecha, default=config.FECHA_INICIO_DEFECTO)
    analizador.add_argument("--hasta", type=_fecha, default=dt.date.today())
    analizador.add_argument("--verboso", action="store_true")
    analizador.add_argument(
        "--sin-cache",
        action="store_true",
        help="ignora la cache en disco y vuelve a pedir todos los tramos",
    )
    args = analizador.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verboso else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    config.asegurar_directorios()
    usar_cache = not args.sin_cache

    if args.fuente in ("xm", "todas"):
        ingestar_xm(args.desde, args.hasta, usar_cache=usar_cache)
    if args.fuente in ("simem", "todas"):
        ingestar_simem(args.desde, args.hasta, usar_cache=usar_cache)

    print(f"\nManifiesto: {config.RUTA_MANIFIESTO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
