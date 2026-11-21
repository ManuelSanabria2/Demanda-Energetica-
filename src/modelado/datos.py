"""Carga de las tablas de modelado: del Parquet a X, y y fechas.

    data/procesado/tabla_modelado_r24.parquet   -> X, y, fechas
    data/procesado/tabla_modelado_r120.parquet  -> X, y, fechas

La lista de variables **sale de la tabla**: son sus columnas menos `fecha_hora`
y `demanda_mwh`. No se escribe a mano en ningun script: `lag_24` existe en r24
y no en r120, y una lista fija se romperia en una de las dos.

Que la lista salga de la tabla no significa aceptar cualquier columna. Cada una
se contrasta con lo que `variables.construir` sabe generar:

- una columna que esa capa excluye (`atipico`, `z_estacional`, `valor_kwh`...)
  hace fallar la carga con el motivo de la exclusion;
- una que no reconoce, tambien: decidir si es variable no debe ocurrir en
  silencio;
- un rezago menor que el `rezago_minimo_h` de la tabla (lag_24 en r120) es
  fuga temporal y tambien falla.

Antes de leer el Parquet se comprueba su SHA-256 contra
`registro_variables.json`. Si no coincide, la tabla en disco no es la que
describe el registro y hay que regenerarla (README, "Tabla de modelado").

    from modelado.datos import cargar_tabla
    (X, y, fechas), registro = cargar_tabla("r24")
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from variables.construir import (
    BANDERAS_CALENDARIO,
    COLUMNA_OBJETIVO,
    COLUMNA_TIEMPO,
    COLUMNAS_EXCLUIDAS,
    DIR_SALIDA,
    EXOGENAS_DIRECTAS,
    HORIZONTE_H,
    NOMBRE_REGISTRO,
    PERIODOS_CICLICOS,
    REZAGOS_H,
    VENTANA_MOVIL_H,
    nombre_tabla,
)

ZONA_HORARIA = "America/Bogota"
VERSION_FORMATO = 1

REGENERAR = (
    "python scripts/construir_panel.py --imputacion causal && "
    "python scripts/construir_variables.py"
)


class ErrorDatos(Exception):
    """La tabla de modelado no se puede usar con seguridad."""


class Datos(NamedTuple):
    """Una tabla de modelado separada. Las tres piezas comparten el orden de filas."""

    X: pd.DataFrame
    y: pd.Series
    fechas: pd.Series


# --------------------------------------------------------------------------
# Utilidades comunes del paquete
# --------------------------------------------------------------------------


def ahora() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def registro_operacion(operacion: str, criterio: str, **detalle: Any) -> dict[str, Any]:
    """Entrada de registro de una operacion del modelado: que se hizo y con que criterio."""
    return {"operacion": operacion, "criterio": criterio, "momento": ahora(), "detalle": detalle}


def sha256_archivo(ruta: Path) -> str:
    digesto = hashlib.sha256()
    with Path(ruta).open("rb") as origen:
        for bloque in iter(lambda: origen.read(1 << 20), b""):
            digesto.update(bloque)
    return digesto.hexdigest()


def rezago_de(version: str | int) -> int:
    """'r24' -> 24 y 'r120' -> 120; un entero se devuelve tal cual.

    Falla por debajo del horizonte de 24 h: una tabla asi tendria fuga temporal.
    """
    if isinstance(version, bool):
        raise ErrorDatos(f"Version de tabla no valida: {version!r}")
    if isinstance(version, (int, np.integer)):
        rezago = int(version)
    else:
        coincidencia = re.fullmatch(r"r(\d+)", str(version).strip().lower())
        if not coincidencia:
            raise ErrorDatos(f"Version de tabla no valida: {version!r}. Usa r24 o r120.")
        rezago = int(coincidencia.group(1))
    if rezago < HORIZONTE_H:
        raise ErrorDatos(
            f"rezago_minimo_h={rezago} es menor que el horizonte de {HORIZONTE_H} h: "
            "una tabla asi tendria fuga temporal."
        )
    return rezago


def clave_version(version: str | int) -> str:
    """24, 'r24' -> 'r24': la clave de la version en los registros."""
    return f"r{rezago_de(version)}"


def variables_conocidas() -> set[str]:
    """Todas las variables que `variables.construir` sabe generar, en cualquier version."""
    rezagos = {f"lag_{k}" for k in REZAGOS_H}
    moviles = {f"ma_{VENTANA_MOVIL_H}", f"std_{VENTANA_MOVIL_H}"}
    ciclicas = {f"{c}_{f}" for c in PERIODOS_CICLICOS for f in ("sin", "cos")}
    return rezagos | moviles | ciclicas | set(EXOGENAS_DIRECTAS) | set(BANDERAS_CALENDARIO)


def verificar_rejilla_horaria(fechas: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Marcas en hora local, comprobando que son horas consecutivas sin huecos.

    `TimeSeriesSplit` cuenta el `gap` en filas, y los bloques se recorren por
    posicion: solo sobre una rejilla completa 24 filas son 24 horas.
    """
    momentos = pd.DatetimeIndex(fechas)
    if len(momentos) == 0:
        raise ErrorDatos("No hay marcas de tiempo.")
    if momentos.tz is None:
        raise ErrorDatos(
            "Las marcas de tiempo no tienen zona horaria: las tablas de modelado "
            f"van en {ZONA_HORARIA}."
        )
    if momentos.hasnans:
        raise ErrorDatos(f"{int(momentos.isna().sum())} marcas de tiempo nulas.")
    pasos = np.diff(momentos.asi8)
    saltos = np.flatnonzero(pasos != pd.Timedelta(hours=1).value)
    if len(saltos):
        ejemplos = [f"{momentos[i]} -> {momentos[i + 1]}" for i in saltos[:3]]
        raise ErrorDatos(
            f"La rejilla horaria no es continua: {len(saltos)} saltos distintos de "
            f"1 h (primeros: {ejemplos}). El gap de TimeSeriesSplit cuenta filas y "
            "dejaria de contar horas."
        )
    return momentos.tz_convert(ZONA_HORARIA)


# --------------------------------------------------------------------------
# Separacion en X, y, fechas
# --------------------------------------------------------------------------


def separar(
    tabla: pd.DataFrame, rezago_minimo_h: int | str | None = None
) -> tuple[Datos, dict[str, Any]]:
    """De una tabla de modelado a (X, y, fechas), con la lista de variables de la tabla.

    X son todas las columnas menos `fecha_hora` y `demanda_mwh`, en el orden de
    la tabla. Falla ante una columna excluida por `variables.construir`, una
    desconocida, una no numerica o, si se da `rezago_minimo_h`, un rezago menor
    que ese minimo. Los NaN no se tocan: imputar va dentro del Pipeline.
    """
    for columna in (COLUMNA_TIEMPO, COLUMNA_OBJETIVO):
        if columna not in tabla.columns:
            raise ErrorDatos(f"La tabla no tiene la columna {columna!r}.")

    variables = [c for c in tabla.columns if c not in (COLUMNA_TIEMPO, COLUMNA_OBJETIVO)]
    if not variables:
        raise ErrorDatos("La tabla no tiene ninguna variable.")

    excluidas = {c: COLUMNAS_EXCLUIDAS[c] for c in variables if c in COLUMNAS_EXCLUIDAS}
    if excluidas:
        motivos = "; ".join(f"{c}: {m}" for c, m in excluidas.items())
        raise ErrorDatos(f"La tabla trae columnas que no pueden ser variables. {motivos}")

    conocidas = variables_conocidas()
    desconocidas = [c for c in variables if c not in conocidas]
    if desconocidas:
        raise ErrorDatos(
            f"Columnas desconocidas en la tabla: {desconocidas}. Clasificalas en "
            "variables.construir y en docs/diccionario_variables.md antes de usarlas."
        )

    rezago = None if rezago_minimo_h is None else rezago_de(rezago_minimo_h)
    if rezago is not None:
        cortos = [
            c
            for c in variables
            if (m := re.fullmatch(r"lag_(\d+)", c)) and int(m.group(1)) < rezago
        ]
        if cortos:
            raise ErrorDatos(
                f"{cortos} usan demanda posterior a t - {rezago} h: en r{rezago} "
                "serian fuga temporal."
            )

    no_numericas = [c for c in variables if not pd.api.types.is_numeric_dtype(tabla[c])]
    if no_numericas:
        raise ErrorDatos(f"Variables no numericas: {no_numericas}")

    momentos = verificar_rejilla_horaria(tabla[COLUMNA_TIEMPO])

    X = tabla[variables].reset_index(drop=True).copy()
    y = pd.to_numeric(tabla[COLUMNA_OBJETIVO]).astype("float64").reset_index(drop=True)
    y.name = COLUMNA_OBJETIVO
    fechas = pd.Series(momentos, name=COLUMNA_TIEMPO)

    registro = registro_operacion(
        "separar",
        f"X = columnas de la tabla menos {COLUMNA_TIEMPO} y {COLUMNA_OBJETIVO}, "
        "cada una reconocida por variables.construir",
        filas=len(X),
        variables=variables,
        n_variables=len(variables),
        objetivo=COLUMNA_OBJETIVO,
        rezago_minimo_h=rezago,
        rango=[str(momentos[0]), str(momentos[-1])],
        zona_horaria=ZONA_HORARIA,
        nan_por_variable={c: int(n) for c, n in X.isna().sum().items() if n},
        nan_objetivo=int(y.isna().sum()),
    )
    return Datos(X, y, fechas), registro


# --------------------------------------------------------------------------
# Carga desde disco
# --------------------------------------------------------------------------


def cargar_tabla(
    version: str | int, directorio: Path | None = None
) -> tuple[Datos, dict[str, Any]]:
    """Lee una tabla de modelado, comprueba su SHA-256 y la separa en X, y, fechas.

    `version` es 'r24', 'r120' o el rezago en horas. `directorio` es donde estan
    las tablas y `registro_variables.json` (por defecto, data/procesado).
    """
    rezago = rezago_de(version)
    clave = f"r{rezago}"
    carpeta = Path(directorio) if directorio is not None else DIR_SALIDA

    ruta_registro = carpeta / NOMBRE_REGISTRO
    if not ruta_registro.exists():
        raise ErrorDatos(f"No existe {ruta_registro}. Regenera las tablas: {REGENERAR}")
    bitacora = json.loads(ruta_registro.read_text(encoding="utf-8"))
    entrada = (bitacora.get("versiones") or {}).get(clave)
    if entrada is None:
        raise ErrorDatos(f"{NOMBRE_REGISTRO} no describe la version {clave}. Regenera: {REGENERAR}")
    if entrada.get("rezago_minimo_h") not in (None, rezago):
        raise ErrorDatos(
            f"La entrada {clave} del registro dice rezago_minimo_h="
            f"{entrada['rezago_minimo_h']}."
        )

    ruta = carpeta / entrada.get("archivo", nombre_tabla(rezago))
    if not ruta.exists():
        raise ErrorDatos(f"No existe {ruta}. Regenera las tablas: {REGENERAR}")
    huella = sha256_archivo(ruta)
    if huella != entrada.get("sha256"):
        raise ErrorDatos(
            f"El SHA-256 de {ruta.name} ({huella[:12]}...) no coincide con el de "
            f"{NOMBRE_REGISTRO} ({str(entrada.get('sha256'))[:12]}...): la tabla en "
            f"disco no es la que describe el registro. Regenerala: {REGENERAR}"
        )

    datos, paso = separar(pd.read_parquet(ruta), rezago)

    esperadas = entrada.get("variables")
    if esperadas is not None and list(datos.X.columns) != list(esperadas):
        raise ErrorDatos(
            f"Las variables de {ruta.name} {list(datos.X.columns)} no son las del "
            f"registro {list(esperadas)}."
        )

    registro = {
        "version_formato": VERSION_FORMATO,
        "momento": ahora(),
        "tabla": clave,
        "archivo": ruta.name,
        "sha256": huella,
        "rezago_minimo_h": rezago,
        "filas": len(datos.X),
        "variables": list(datos.X.columns),
        "rango": paso["detalle"]["rango"],
        "operaciones": [
            registro_operacion(
                "verificar_sha256",
                f"SHA-256 del Parquet igual al anotado en {NOMBRE_REGISTRO}",
                archivo=ruta.name,
                sha256=huella,
                coincide=True,
            ),
            paso,
        ],
    }
    return datos, registro
