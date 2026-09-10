"""Exploracion cruda de las APIs de XM (SINERGOX) y SIMEM.

Script autonomo: solo depende de `requests` y la libreria estandar. No importa
nada del paquete `ingesta`, no construye DataFrames y no transforma los datos.
Su unico proposito es ver que devuelven realmente las dos APIs.

Que hace:

1. Consulta el inventario completo de metricas de XM (POST /lists con
   MetricId "ListadoMetricas") y lo guarda entero.
2. Consulta un solo dia de una metrica horaria de XM.
3. Consulta un solo dia de un dataset de SIMEM.
4. Guarda cada respuesta cruda, byte a byte, en data/raw_samples/.
5. Imprime la estructura de cada JSON: claves de primer nivel, tipo de cada
   valor y, para las listas, la forma del primer elemento.

Cuando un llamado falla imprime el status code, el content-type y el cuerpo de
la respuesta, y sigue con el resto de sondeos en vez de abortar: si una API
esta caida, la otra igual se explora.

Uso:

    python scripts/explorar_apis.py
    python scripts/explorar_apis.py --fecha 2025-06-15 --dataset 14fabb
    python scripts/explorar_apis.py --metrica DemaCome --entidad Sistema
    python scripts/explorar_apis.py --profundidad 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import requests

# La salida de SIMEM trae acentos; sin esto la consola de Windows (cp1252)
# rompe la impresion o muestra mojibake.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

URL_XM = "https://servapibi.xm.com.co"
URL_SIMEM = "https://www.simem.co/backend-files/api/PublicData"

DIR_MUESTRAS = Path(__file__).resolve().parents[1] / "data" / "raw_samples"

TIMEOUT = 180
ANCHO = 78

# Cuantas claves listar antes de resumir el resto (Values trae 25).
MAX_CLAVES = 12
# Cuantos caracteres de un valor escalar mostrar como muestra.
MAX_ESCALAR = 60


# --------------------------------------------------------------------------
# Descripcion de la estructura
# --------------------------------------------------------------------------


def tipo_de(valor: Any) -> str:
    """Nombre legible del tipo de un valor JSON."""
    if valor is None:
        return "null"
    if isinstance(valor, bool):
        return "bool"
    if isinstance(valor, int):
        return "int"
    if isinstance(valor, float):
        return "float"
    if isinstance(valor, str):
        return "str"
    if isinstance(valor, list):
        return f"list[{len(valor)}]"
    if isinstance(valor, dict):
        return f"dict[{len(valor)} claves]"
    return type(valor).__name__


def muestra_de(valor: Any) -> str:
    """Representacion corta de un escalar, para ver su forma real."""
    if isinstance(valor, (dict, list)):
        return ""
    texto = repr(valor)
    if len(texto) > MAX_ESCALAR:
        texto = texto[:MAX_ESCALAR] + "..."
    return f" = {texto}"


def describir(
    valor: Any,
    nombre: str = "(raiz)",
    profundidad: int = 0,
    profundidad_max: int = 4,
    sangria: str = "",
) -> None:
    """Imprime recursivamente la estructura de un valor JSON.

    De las listas solo se describe el primer elemento: la intencion es ver la
    forma de los registros, no volcar los datos.
    """
    print(f"{sangria}{nombre}: {tipo_de(valor)}{muestra_de(valor)}")

    if profundidad >= profundidad_max:
        if isinstance(valor, (dict, list)) and valor:
            print(f"{sangria}  ... (profundidad maxima alcanzada)")
        return

    hija = sangria + "  "

    if isinstance(valor, dict):
        claves = list(valor.keys())
        for clave in claves[:MAX_CLAVES]:
            describir(valor[clave], clave, profundidad + 1, profundidad_max, hija)
        if len(claves) > MAX_CLAVES:
            restantes = claves[MAX_CLAVES:]
            print(
                f"{hija}... {len(restantes)} claves mas: "
                f"{restantes[0]!r} ... {restantes[-1]!r}"
            )

    elif isinstance(valor, list):
        if not valor:
            print(f"{hija}(lista vacia)")
            return
        describir(valor[0], "[0]", profundidad + 1, profundidad_max, hija)
        if len(valor) > 1:
            tipos = {tipo_de(elemento) for elemento in valor[:50]}
            if len(tipos) == 1:
                print(f"{hija}... {len(valor) - 1} elementos mas, del mismo tipo")
            else:
                print(f"{hija}... {len(valor) - 1} elementos mas, tipos: {sorted(tipos)}")


# --------------------------------------------------------------------------
# Llamados con diagnostico
# --------------------------------------------------------------------------


def titulo(texto: str) -> None:
    """Encabezado de seccion."""
    print(f"\n{'=' * ANCHO}\n{texto}\n{'=' * ANCHO}")


def informar_fallo(respuesta: requests.Response, contexto: str) -> None:
    """Imprime todo lo necesario para diagnosticar una respuesta que fallo."""
    print(f"\n  FALLO: {contexto}")
    print(f"  status code  : {respuesta.status_code} {respuesta.reason}")
    print(f"  content-type : {respuesta.headers.get('Content-Type', '(sin cabecera)')}")
    print(f"  longitud     : {len(respuesta.content)} bytes")
    print(f"  url          : {respuesta.url}")
    print("  cuerpo       :")
    cuerpo = respuesta.text.strip()
    if not cuerpo:
        print("    (vacio)")
        return
    for linea in cuerpo[:2000].splitlines():
        print(f"    {linea}")
    if len(cuerpo) > 2000:
        print(f"    ... ({len(cuerpo) - 2000} caracteres mas)")


def guardar_crudo(respuesta: requests.Response, nombre: str) -> Path:
    """Guarda la respuesta tal cual llego, sin reserializar ni reformatear."""
    DIR_MUESTRAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_MUESTRAS / nombre
    ruta.write_bytes(respuesta.content)
    print(f"  guardado     : {ruta.relative_to(DIR_MUESTRAS.parents[1])} "
          f"({len(respuesta.content):,} bytes)")
    return ruta


def sondear(
    sesion: requests.Session,
    contexto: str,
    nombre_archivo: str,
    metodo: str,
    url: str,
    **kwargs: Any,
) -> Any | None:
    """Ejecuta un llamado, guarda el crudo y devuelve el JSON, o None si fallo.

    Nunca lanza excepcion: el objetivo es diagnosticar, asi que un fallo se
    imprime y el script continua con los demas sondeos.
    """
    print(f"\n{contexto}")
    if kwargs.get("json"):
        print(f"  cuerpo enviado: {json.dumps(kwargs['json'], ensure_ascii=False)}")
    if kwargs.get("params"):
        print(f"  parametros    : {kwargs['params']}")

    try:
        respuesta = sesion.request(metodo, url, timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        print(f"\n  FALLO DE RED: {type(exc).__name__}: {exc}")
        return None

    print(f"  status code  : {respuesta.status_code}")
    print(f"  content-type : {respuesta.headers.get('Content-Type', '(sin cabecera)')}")

    # El crudo se guarda tambien cuando la respuesta es un error: el cuerpo del
    # error es justamente lo que hay que poder revisar despues.
    guardar_crudo(respuesta, nombre_archivo)

    if respuesta.status_code != 200:
        informar_fallo(respuesta, contexto)
        return None

    try:
        return respuesta.json()
    except json.JSONDecodeError as exc:
        print(f"\n  El cuerpo devolvio 200 pero no es JSON valido: {exc}")
        informar_fallo(respuesta, contexto)
        return None


# --------------------------------------------------------------------------
# Sondeos
# --------------------------------------------------------------------------


def explorar_lista_metricas_xm(
    sesion: requests.Session, profundidad: int
) -> Any | None:
    """Inventario completo de metricas de XM: POST /lists con ListadoMetricas."""
    titulo("XM  ·  1. inventario de metricas (POST /lists)")

    datos = sondear(
        sesion,
        contexto="POST /lists  MetricId=ListadoMetricas",
        nombre_archivo="xm_lists_ListadoMetricas.json",
        metodo="POST",
        url=f"{URL_XM}/lists",
        json={
            "MetricId": "ListadoMetricas",
            "StartDate": "2020-01-01",
            "EndDate": "2020-01-02",
            "Entity": "Sistema",
        },
    )
    if datos is None:
        return None

    print("\n  --- estructura ---")
    describir(datos, "(raiz)", profundidad_max=profundidad)

    # Recuento plano del inventario, sin construir tablas: solo contar.
    entradas = [
        entidad.get("Values", {})
        for item in datos.get("Items", [])
        for entidad in item.get("ListEntities", [])
    ]
    print(f"\n  --- inventario: {len(entradas)} entradas ---")
    if entradas:
        print(f"  claves de cada entrada: {list(entradas[0].keys())}")

        tipos: dict[str, int] = {}
        for entrada in entradas:
            clave = str(entrada.get("Type"))
            tipos[clave] = tipos.get(clave, 0) + 1
        print(f"  por Type              : {tipos}")

        limites = sorted({entrada.get("MaxDays") for entrada in entradas})
        print(f"  valores de MaxDays    : {limites}")
        print("\n  primeras 5 entradas:")
        for entrada in entradas[:5]:
            print(
                f"    {entrada.get('MetricId'):<18} {str(entrada.get('Entity')):<24} "
                f"MaxDays={entrada.get('MaxDays')} {entrada.get('MetricUnits')}"
            )

    return datos


def explorar_horaria_xm(
    sesion: requests.Session,
    metrica: str,
    entidad: str,
    fecha: dt.date,
    profundidad: int,
) -> Any | None:
    """Un solo dia de una metrica horaria de XM."""
    titulo(f"XM  ·  2. un dia de {metrica}/{entidad} (POST /hourly)")

    datos = sondear(
        sesion,
        contexto=f"POST /hourly  {metrica}/{entidad}  {fecha}",
        nombre_archivo=f"xm_hourly_{metrica}_{entidad}_{fecha:%Y%m%d}.json",
        metodo="POST",
        url=f"{URL_XM}/hourly",
        json={
            "MetricId": metrica,
            "StartDate": fecha.isoformat(),
            "EndDate": fecha.isoformat(),
            "Entity": entidad,
        },
    )
    if datos is None:
        return None

    print("\n  --- estructura ---")
    describir(datos, "(raiz)", profundidad_max=profundidad)

    # Un 200 con Items vacio no es un error de la API, pero si un dia sin datos.
    if not datos.get("Items"):
        print(
            "\n  AVISO: la respuesta es 200 pero Items viene vacio. "
            "La API no senala como error un dia sin datos publicados."
        )

    return datos


def explorar_simem(
    sesion: requests.Session,
    dataset: str,
    fecha: dt.date,
    profundidad: int,
) -> Any | None:
    """Un solo dia de un dataset de SIMEM."""
    titulo(f"SIMEM  ·  3. un dia del dataset {dataset} (GET PublicData)")

    datos = sondear(
        sesion,
        contexto=f"GET PublicData  datasetId={dataset}  {fecha}",
        nombre_archivo=f"simem_{dataset}_{fecha:%Y%m%d}.json",
        metodo="GET",
        url=URL_SIMEM,
        params={
            "datasetId": dataset,
            "startDate": fecha.isoformat(),
            "endDate": fecha.isoformat(),
        },
    )
    if datos is None:
        return None

    print("\n  --- estructura ---")
    describir(datos, "(raiz)", profundidad_max=profundidad)

    registros = datos.get("result", {}).get("records") or []
    print(f"\n  --- records: {len(registros)} ---")
    if registros:
        print(f"  claves de cada record: {list(registros[0].keys())}")
        print("  primeros 3 records (crudos):")
        for registro in registros[:3]:
            print(f"    {json.dumps(registro, ensure_ascii=False)}")

        # Cardinalidad por columna: revela que dimensiones desagregan el dataset
        # y si hay mas de un valor por marca de tiempo.
        print("\n  valores distintos por clave (sobre esta muestra):")
        for clave in registros[0]:
            distintos = {
                json.dumps(registro.get(clave), ensure_ascii=False)
                for registro in registros
            }
            muestra = sorted(distintos)[:4]
            extra = " ..." if len(distintos) > 4 else ""
            print(f"    {clave:<22} {len(distintos):>6} distintos  {muestra}{extra}")

    return datos


def explorar_errores(sesion: requests.Session) -> None:
    """Provoca un error en cada API para ver la forma exacta de sus fallos.

    Las dos APIs senalan los errores de forma distinta, y ninguna de las dos
    devuelve algo que se pueda parsear a ciegas con `r.json()`.
    """
    titulo("4. forma de los errores (llamados invalidos a proposito)")

    sondear(
        sesion,
        contexto="SIMEM sin datasetId",
        nombre_archivo="error_simem_sin_datasetid.json",
        metodo="GET",
        url=URL_SIMEM,
        params={"startDate": "2025-01-01", "endDate": "2025-01-01"},
    )

    sondear(
        sesion,
        contexto="SIMEM con datasetId inexistente",
        nombre_archivo="error_simem_dataset_inexistente.json",
        metodo="GET",
        url=URL_SIMEM,
        params={
            "datasetId": "zzzzzz",
            "startDate": "2025-01-01",
            "endDate": "2025-01-01",
        },
    )

    sondear(
        sesion,
        contexto="XM con rango que excede el limite de la metrica",
        nombre_archivo="error_xm_rango_excedido.json",
        metodo="POST",
        url=f"{URL_XM}/hourly",
        json={
            "MetricId": "DemaReal",
            "StartDate": "2025-01-01",
            "EndDate": "2025-06-30",
            "Entity": "Sistema",
        },
    )


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Ejecuta los sondeos y reporta cuales salieron bien."""
    analizador = argparse.ArgumentParser(
        description="Exploracion cruda de las APIs de XM y SIMEM"
    )
    analizador.add_argument(
        "--fecha",
        default="2025-01-01",
        help="dia a consultar, YYYY-MM-DD (por defecto uno con datos ya liquidados)",
    )
    analizador.add_argument("--metrica", default="DemaReal", help="MetricId de XM")
    analizador.add_argument("--entidad", default="Sistema", help="Entity de XM")
    analizador.add_argument("--dataset", default="14fabb", help="datasetId de SIMEM")
    analizador.add_argument(
        "--profundidad",
        type=int,
        default=6,
        help="niveles de anidamiento a describir (por defecto 6)",
    )
    analizador.add_argument(
        "--sin-errores",
        action="store_true",
        help="omite los llamados invalidos de la seccion 4",
    )
    args = analizador.parse_args(argv)

    fecha = dt.date.fromisoformat(args.fecha)
    sesion = requests.Session()
    sesion.headers.update({"Accept": "application/json"})

    print(f"Exploracion del {dt.datetime.now():%Y-%m-%d %H:%M}")
    print(f"Dia consultado : {fecha}")
    print(f"Muestras crudas: {DIR_MUESTRAS}")

    resultados = {
        "XM /lists": explorar_lista_metricas_xm(sesion, args.profundidad) is not None,
        "XM /hourly": explorar_horaria_xm(
            sesion, args.metrica, args.entidad, fecha, args.profundidad
        )
        is not None,
        "SIMEM PublicData": explorar_simem(
            sesion, args.dataset, fecha, args.profundidad
        )
        is not None,
    }

    if not args.sin_errores:
        explorar_errores(sesion)

    titulo("resumen")
    for etiqueta, ok in resultados.items():
        print(f"  {etiqueta:<20} {'ok' if ok else 'FALLO (ver detalle arriba)'}")

    guardadas = sorted(DIR_MUESTRAS.glob("*.json")) if DIR_MUESTRAS.exists() else []
    print(f"\n  {len(guardadas)} muestras crudas en {DIR_MUESTRAS}:")
    for ruta in guardadas:
        print(f"    {ruta.name:<52} {ruta.stat().st_size:>12,} bytes")

    return 0 if all(resultados.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
