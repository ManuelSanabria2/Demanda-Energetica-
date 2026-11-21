"""Pruebas de los modelos lineales (src/modelado/regresion.py). Sin red.

Dos riesgos que no lanzan ninguna excepcion: que la validacion se cuele en el
ajuste (en el escalado, en la busqueda de hiperparametros o en el modelo final)
y que dos ejecuciones iguales den resultados distintos. Los dos se comprueban
para cada modelo y para r24 y r120, sobre las mismas tablas sinteticas que
tests/test_modelado_base.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado.datos import cargar_tabla  # noqa: E402
from modelado.particion import particionar, validacion_cruzada  # noqa: E402
from modelado.regresion import (  # noqa: E402
    MODELOS,
    ajustado_con,
    coeficientes,
    condicionamiento,
    crear_pipeline,
    evaluar_modelo,
)
from test_modelado_base import VERSIONES, guardar_tablas, panel_minimo  # noqa: E402

# Lo que cambia entre ejecuciones sin que cambie el modelo.
CLAVES_VARIABLES = {"momento", "segundos", "tiempos"}


def _sin_variables(objeto):
    """El registro sin marcas de tiempo ni duraciones, para comparar dos ejecuciones."""
    if isinstance(objeto, dict):
        return {k: _sin_variables(v) for k, v in objeto.items() if k not in CLAVES_VARIABLES}
    if isinstance(objeto, list):
        return [_sin_variables(v) for v in objeto]
    return objeto


@pytest.fixture(scope="module")
def carpeta_tablas(tmp_path_factory) -> Path:
    return guardar_tablas(panel_minimo("2024-06-01", "2026-09-07 23:00"), tmp_path_factory.mktemp("procesado"))


@pytest.fixture(scope="module")
def entradas(carpeta_tablas):
    """Por version: X (en float), y, fechas, indices y el TimeSeriesSplit del entrenamiento."""
    resultado = {}
    for version in VERSIONES:
        (X, y, fechas), _ = cargar_tabla(version, carpeta_tablas)
        indices, _ = particionar(fechas)
        tscv, _ = validacion_cruzada(version, fechas.iloc[indices["entrenamiento"]])
        resultado[version] = (X.astype("float64"), y, fechas, indices, tscv)
    return resultado


@pytest.fixture(scope="module")
def evaluados(entradas):
    """Cada modelo evaluado una vez por version, para no repetir las busquedas."""
    return {
        (version, nombre): evaluar_modelo(nombre, *entradas[version], mape_baseline_validacion=5.0)
        for version in VERSIONES
        for nombre in MODELOS
    }


@pytest.mark.parametrize("nombre", list(MODELOS))
@pytest.mark.parametrize("version", VERSIONES)
def test_el_pipeline_no_se_ajusta_con_datos_de_validacion(entradas, evaluados, version, nombre):
    X, y, fechas, indices, tscv = entradas[version]
    valida = indices["validacion"]
    modelo, registro = evaluados[(version, nombre)]

    # El escalado del modelo final vio exactamente las filas de entrenamiento.
    assert ajustado_con(modelo) == len(indices["entrenamiento"]) == registro["filas_ajuste"]

    # Si la validacion cambia por completo, nada de lo ajustado cambia.
    X_alterada, y_alterada = X.copy(), y.copy()
    X_alterada.iloc[valida] = X_alterada.iloc[valida] * 10 + 1000
    y_alterada.iloc[valida] = y_alterada.iloc[valida] * 3
    alterado, registro_alterado = evaluar_modelo(
        nombre, X_alterada, y_alterada, fechas, indices, tscv, mape_baseline_validacion=5.0
    )

    assert registro_alterado["hiperparametros"] == registro["hiperparametros"]
    assert _sin_variables(registro_alterado["busqueda"]) == _sin_variables(registro["busqueda"])
    assert _sin_variables(registro_alterado["validacion_cruzada"]) == _sin_variables(registro["validacion_cruzada"])
    assert registro_alterado["coeficientes"] == registro["coeficientes"]
    np.testing.assert_array_equal(modelo.named_steps["escalar"].mean_, alterado.named_steps["escalar"].mean_)
    # Control: la validacion si cambio, y las metricas de validacion lo notan.
    assert registro_alterado["metricas_validacion"]["mape_pct"] != registro["metricas_validacion"]["mape_pct"]


@pytest.mark.parametrize("version", VERSIONES)
def test_el_resultado_es_reproducible_con_la_misma_semilla(entradas, evaluados, version):
    for nombre in MODELOS:
        modelo, registro = evaluados[(version, nombre)]
        otra_vez, registro_otra_vez = evaluar_modelo(nombre, *entradas[version], mape_baseline_validacion=5.0)
        assert _sin_variables(registro_otra_vez) == _sin_variables(registro), nombre
        X = entradas[version][0]
        np.testing.assert_array_equal(modelo.predict(X), otra_vez.predict(X))


def test_los_modelos_con_busqueda_eligen_dentro_de_su_rejilla(evaluados):
    for (version, nombre), (_, registro) in evaluados.items():
        busqueda = registro["busqueda"]["detalle"]
        if not MODELOS[nombre].rejilla:
            assert not busqueda["buscados"] and registro["hiperparametros"] == {}
            continue
        for parametro, valor in registro["hiperparametros"].items():
            assert valor in busqueda["rejilla"][parametro]
        assert len(busqueda["candidatos"]) == np.prod([len(v) for v in busqueda["rejilla"].values()])
        cv = registro["validacion_cruzada"]["detalle"]
        assert cv["n_pliegues"] == 5
        assert len(cv["resumen"]["mape_pct"]["por_pliegue"]) == 5


def test_lasso_anula_coeficientes_y_el_registro_dice_cuales(entradas):
    X, y, _, indices, _ = entradas["r24"]
    entrena = indices["entrenamiento"]
    fuerte = crear_pipeline("lasso", alpha=1e6).fit(X.iloc[entrena], y.iloc[entrena])
    assert coeficientes(fuerte)["n_nulos"] == X.shape[1]

    moderado = crear_pipeline("lasso", alpha=5.0).fit(X.iloc[entrena], y.iloc[entrena])
    leido = coeficientes(moderado)
    esperados = [n for n, c in zip(X.columns, moderado[-1].coef_) if c == 0.0]
    assert leido["nulos"] == esperados and leido["n_nulos"] == len(esperados)
    assert list(leido["por_variable"]) == list(X.columns)


def test_la_expansion_polinomial_tiene_columnas_dependientes(entradas):
    X, y, _, indices, _ = entradas["r24"]
    entrena = indices["entrenamiento"]
    polinomial = crear_pipeline("polinomial").fit(X.iloc[entrena], y.iloc[entrena])
    diseno = condicionamiento(polinomial, X.iloc[entrena])
    # sin^2 + cos^2 = 1 y z^2 afin en z para cada binaria: rango por debajo de las columnas.
    assert diseno["columnas"] == X.shape[1] * (X.shape[1] + 3) // 2
    assert diseno["rango"] < diseno["columnas"]

