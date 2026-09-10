"""Tests de las piezas puras de la ingesta (sin red)."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingesta import normalizar  # noqa: E402
from ingesta.ventanas import partir_rango  # noqa: E402


# --- ventanas -------------------------------------------------------------


def test_partir_rango_cubre_todo_sin_solapes():
    inicio, fin = dt.date(2025, 1, 1), dt.date(2025, 3, 15)
    trozos = partir_rango(inicio, fin, 31)

    assert trozos[0][0] == inicio
    assert trozos[-1][1] == fin
    for (_, cierre), (siguiente, _) in zip(trozos, trozos[1:]):
        assert siguiente == cierre + dt.timedelta(days=1)

    dias = sum((b - a).days + 1 for a, b in trozos)
    assert dias == (fin - inicio).days + 1


def test_partir_rango_respeta_max_dias():
    trozos = partir_rango(dt.date(2025, 1, 1), dt.date(2025, 12, 31), 31)
    assert all((b - a).days + 1 <= 31 for a, b in trozos)


def test_partir_rango_un_solo_dia():
    assert partir_rango(dt.date(2025, 1, 1), dt.date(2025, 1, 1), 31) == [
        (dt.date(2025, 1, 1), dt.date(2025, 1, 1))
    ]


def test_partir_rango_invertido_falla():
    with pytest.raises(ValueError):
        partir_rango(dt.date(2025, 2, 1), dt.date(2025, 1, 1), 31)


# --- XM ancho a largo -----------------------------------------------------


def _item_xm(fecha: str, **horas: str) -> dict:
    valores = {"code": "Sistema"}
    valores.update({f"Hour{n:02d}": "1000.0" for n in range(1, 25)})
    valores.update(horas)
    return {"Date": fecha, "HourlyEntities": [{"Id": "Sistema", "Values": valores}]}


def test_xm_ancho_a_largo_produce_24_horas():
    tabla = normalizar.xm_ancho_a_largo([_item_xm("2025-01-01")], "DemaReal")

    assert len(tabla) == 24
    assert list(tabla.columns) == normalizar.COLUMNAS_SALIDA
    assert tabla["fecha_hora"].min() == pd.Timestamp("2025-01-01 00:00:00")
    assert tabla["fecha_hora"].max() == pd.Timestamp("2025-01-01 23:00:00")
    assert tabla["valor_kwh"].dtype == float


def test_xm_valores_ausentes_son_nan_no_cero():
    item = _item_xm("2025-01-01", Hour05="", Hour06=None)
    tabla = normalizar.xm_ancho_a_largo([item], "DemaReal").set_index("fecha_hora")

    assert pd.isna(tabla.loc["2025-01-01 04:00:00", "valor_kwh"])
    assert pd.isna(tabla.loc["2025-01-01 05:00:00", "valor_kwh"])
    assert (tabla["valor_kwh"] == 0).sum() == 0


def test_xm_sin_items_devuelve_tabla_vacia_con_esquema():
    tabla = normalizar.xm_ancho_a_largo([], "DemaReal")
    assert tabla.empty
    assert list(tabla.columns) == normalizar.COLUMNAS_SALIDA


# --- SIMEM: colapso de versiones -----------------------------------------


def _registros_simem_dos_versiones() -> pd.DataFrame:
    filas = []
    for version in ("TX2", "TXR"):
        for hora in range(3):
            for agente in ("AAAA", "BBBB"):
                filas.append(
                    {
                        "FechaHora": f"2026-08-01 {hora:02d}:00:00",
                        "CodigoSICAgente": agente,
                        "TipoMercado": "Regulado",
                        "Version": version,
                        "Valor": 100.0,
                    }
                )
    return pd.DataFrame(filas)


def test_colapsar_versiones_no_duplica_la_demanda():
    registros = _registros_simem_dos_versiones()
    assert registros["Valor"].sum() == 1200.0  # el total ingenuo, duplicado

    limpio = normalizar.simem_colapsar_versiones(
        registros, ["CodigoSICAgente", "TipoMercado"]
    )

    assert len(limpio) == len(registros) / 2
    assert limpio["Valor"].sum() == 600.0
    assert set(limpio["Version"]) == {"TXR"}  # TXR tiene precedencia sobre TX2


def test_colapsar_elige_la_mejor_version_por_fecha():
    registros = _registros_simem_dos_versiones()
    # La hora 02 solo llego en version preliminar.
    registros = registros[
        ~((registros["FechaHora"].str.endswith("02:00:00")) & (registros["Version"] == "TXR"))
    ]

    limpio = normalizar.simem_colapsar_versiones(
        registros, ["CodigoSICAgente", "TipoMercado"]
    )
    por_hora = limpio.groupby("FechaHora")["Version"].unique()

    assert por_hora["2026-08-01 00:00:00"].tolist() == ["TXR"]
    assert por_hora["2026-08-01 02:00:00"].tolist() == ["TX2"]


def test_version_desconocida_falla_en_vez_de_pasar():
    registros = _registros_simem_dos_versiones()
    registros.loc[0, "Version"] = "TXZ"

    with pytest.raises(ValueError, match="precedencia"):
        normalizar.simem_colapsar_versiones(registros, ["CodigoSICAgente", "TipoMercado"])


def test_agregar_nacional_suma_una_sola_version():
    agregado = normalizar.simem_agregar_nacional(_registros_simem_dos_versiones())

    assert len(agregado) == 3
    assert agregado["valor_kwh"].tolist() == [200.0, 200.0, 200.0]
    assert list(agregado.columns) == normalizar.COLUMNAS_SALIDA


# --- cobertura ------------------------------------------------------------


def test_reporte_cobertura_detecta_huecos():
    tabla = normalizar.xm_ancho_a_largo([_item_xm("2025-01-01")], "DemaReal")
    tabla = tabla[tabla["fecha_hora"] != pd.Timestamp("2025-01-01 10:00:00")]

    reporte = normalizar.reporte_cobertura(
        tabla, inicio=dt.date(2025, 1, 1), fin=dt.date(2025, 1, 1)
    )

    assert reporte["horas_esperadas"] == 24
    assert reporte["horas_faltantes"] == 1
    assert reporte["primeros_huecos"] == ["2025-01-01 10:00:00"]


def test_reporte_cobertura_cuenta_nan_aparte():
    item = _item_xm("2025-01-01", Hour03="")
    tabla = normalizar.xm_ancho_a_largo([item], "DemaReal")

    reporte = normalizar.reporte_cobertura(
        tabla, inicio=dt.date(2025, 1, 1), fin=dt.date(2025, 1, 1)
    )

    assert reporte["horas_faltantes"] == 0
    assert reporte["horas_con_nan"] == 1


def test_precedencia_cubre_las_versiones_observadas():
    """Las 9 versiones vistas en el historico 2021-2026 deben tener precedencia.

    Si SIMEM introduce una nueva, el colapso falla en vez de agregar mal; este
    test deja constancia de las que ya se conocen.
    """
    from ingesta import config

    observadas = {"TX2", "TX3", "TX4", "TX5", "TX6", "TX7", "TX8", "TXF", "TXR"}
    assert observadas <= set(config.PRECEDENCIA_VERSIONES)


def test_colapso_con_cuatro_versiones_simultaneas():
    """Mayo de 2026 trae TX2, TX3, TXR y TXF sobre la misma hora: la suma seria 4x."""
    filas = [
        {
            "FechaHora": "2026-05-01 00:00:00",
            "CodigoSICAgente": "AAAA",
            "TipoMercado": "Regulado",
            "Version": version,
            "Valor": 100.0,
        }
        for version in ("TX2", "TX3", "TXR", "TXF")
    ]
    agregado = normalizar.simem_agregar_nacional(pd.DataFrame(filas))

    assert len(agregado) == 1
    assert agregado["valor_kwh"].iloc[0] == 100.0
    assert agregado["version"].iloc[0] == "TXF"
