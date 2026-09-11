"""Tests de la limpieza del catalogo de metricas de XM.

La tabla de prueba reproduce, fila a fila, los defectos que se encontraron en
el catalogo real: grafias distintas de un mismo valor, vacios, espacios
sobrantes y la URL /list que responde 404.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from limpieza.catalogo import limpiar_catalogo, resumen  # noqa: E402
from limpieza.limpiar import ErrorLimpieza  # noqa: E402

BASE = "https://servapibi.xm.com.co"


def catalogo() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["DemaReal", "Demanda Real por Sistema", "Sistema", 31, "HourlyEntities",
             f"{BASE}/hourly", "No aplica", "kWh", "Demanda del SIN"],
            ["AporEner", "Aportes  Energía por Rio", "Rio", 31, "DailyEntities",
             f"{BASE}/daily", "Nombre Río", "kWh", " Aportes en energia  "],
            ["ListadoAgentes", "Listado de agentes", "Sistema", 731, "ListsEntities",
             f"{BASE}/list", "No Aplica", "", "Listado"],
            ["DemaSub", "Demanda por subarea", "SubArea", 31, "HourlyEntities",
             f"{BASE}/hourly", "No aplica", "kWh", "x"],
            ["GeneSub", "Generacion por subarea", "Subarea", 31, "HourlyEntities",
             f"{BASE}/hourly", "Codigo Agente", "kWh", ""],
        ],
        columns=["MetricId", "MetricName", "Entity", "MaxDays", "Type", "Url",
                 "Filter", "MetricUnits", "MetricDescription"],
    )


def fila(limpio: pd.DataFrame, metrica: str) -> pd.Series:
    return limpio.set_index("metric_id").loc[metrica]


def test_las_columnas_pasan_a_snake_case():
    limpio, _ = limpiar_catalogo(catalogo())
    for columna in ("metric_id", "metric_name", "max_days", "metric_units"):
        assert columna in limpio.columns


def test_las_dos_grafias_de_no_aplica_se_tratan_igual():
    limpio, registro = limpiar_catalogo(catalogo())

    assert pd.isna(fila(limpio, "DemaReal")["filter"])
    assert pd.isna(fila(limpio, "ListadoAgentes")["filter"])
    assert fila(limpio, "AporEner")["filter"] == "Nombre Río"
    assert fila(limpio, "AporEner")["tiene_filtro"]
    assert not fila(limpio, "DemaReal")["tiene_filtro"]

    paso = next(o for o in registro["operaciones"] if o["operacion"] == "normalizar_filtro")
    assert paso["detalle"]["grafias_encontradas"] == {"No aplica": 2, "No Aplica": 1}


def test_los_espacios_sobrantes_se_limpian_y_quedan_registrados():
    limpio, registro = limpiar_catalogo(catalogo())

    assert fila(limpio, "AporEner")["metric_name"] == "Aportes Energía por Rio"
    assert fila(limpio, "AporEner")["metric_description"] == "Aportes en energia"
    paso = next(o for o in registro["operaciones"] if o["operacion"] == "limpiar_texto")
    assert paso["detalle"]["cambios_por_columna"] == {"metric_name": 1, "metric_description": 1}


def test_los_vacios_pasan_a_nulo():
    limpio, _ = limpiar_catalogo(catalogo())

    assert pd.isna(fila(limpio, "ListadoAgentes")["metric_units"])
    assert pd.isna(fila(limpio, "GeneSub")["metric_description"])


def test_la_entidad_literal_se_conserva_para_la_api():
    """entity es lo que se envia a la API: no se reescribe."""
    limpio, registro = limpiar_catalogo(catalogo())

    assert fila(limpio, "DemaSub")["entity"] == "SubArea"
    assert fila(limpio, "GeneSub")["entity"] == "Subarea"
    assert fila(limpio, "DemaSub")["entidad_normalizada"] == fila(limpio, "GeneSub")["entidad_normalizada"]

    paso = next(o for o in registro["operaciones"] if o["operacion"] == "normalizar_entidad")
    assert paso["detalle"]["n_entidades_normalizadas"] == paso["detalle"]["n_entidades_originales"] - 1


def test_la_url_de_listados_se_corrige_y_se_marca():
    limpio, registro = limpiar_catalogo(catalogo())

    assert fila(limpio, "ListadoAgentes")["url"] == f"{BASE}/lists"
    assert fila(limpio, "ListadoAgentes")["url_corregida"]
    assert not fila(limpio, "DemaReal")["url_corregida"]
    assert limpio["url_coherente"].all()

    paso = next(o for o in registro["operaciones"] if o["operacion"] == "corregir_url")
    assert "404" in paso["detalle"]["evidencia"]


def test_una_url_no_verificada_no_se_toca_pero_se_marca_incoherente():
    marco = catalogo()
    marco.loc[0, "Url"] = f"{BASE}/daily"  # horaria apuntando a /daily

    limpio, registro = limpiar_catalogo(marco)

    assert fila(limpio, "DemaReal")["url"] == f"{BASE}/daily"
    assert not fila(limpio, "DemaReal")["url_coherente"]
    paso = next(o for o in registro["operaciones"] if o["operacion"] == "derivar_granularidad")
    assert paso["detalle"]["n_url_incoherentes"] == 1


def test_la_granularidad_se_deriva_del_tipo():
    limpio, _ = limpiar_catalogo(catalogo())

    assert fila(limpio, "DemaReal")["granularidad"] == "horaria"
    assert fila(limpio, "AporEner")["granularidad"] == "diaria"
    assert fila(limpio, "ListadoAgentes")["granularidad"] == "listado"


def test_un_tipo_desconocido_falla_en_vez_de_quedar_sin_clasificar():
    marco = catalogo()
    marco.loc[0, "Type"] = "WeeklyEntities"

    with pytest.raises(ErrorLimpieza, match="WeeklyEntities"):
        limpiar_catalogo(marco)


def test_una_metrica_repetida_falla():
    marco = pd.concat([catalogo(), catalogo().iloc[[0]]], ignore_index=True)

    with pytest.raises(ErrorLimpieza, match="repite"):
        limpiar_catalogo(marco)


def test_no_se_pierden_filas_ni_se_modifica_la_entrada():
    marco = catalogo()
    copia = marco.copy(deep=True)

    limpio, registro = limpiar_catalogo(marco)

    assert len(limpio) == len(marco) == registro["filas_finales"]
    pd.testing.assert_frame_equal(marco, copia)


def test_el_resumen_lista_cada_operacion():
    _, registro = limpiar_catalogo(catalogo())
    texto = resumen(registro)

    for operacion in ("limpiar_texto", "normalizar_filtro", "corregir_url", "validar"):
        assert operacion in texto
