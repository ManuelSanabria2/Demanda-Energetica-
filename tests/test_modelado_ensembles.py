"""Pruebas de los ensembles (src/modelado/ensembles.py). Sin red.

El panel sintetico de tests/test_modelado_base.py se usa con una tendencia
creciente: asi la validacion queda por encima de lo visto en entrenamiento,
como en la serie real, y se puede comprobar lo que motiva la forma
diferenciada (un arbol sobre el nivel no pasa del maximo de entrenamiento).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado.datos import cargar_tabla  # noqa: E402
from modelado.ensembles import (  # noqa: E402
    FORMAS,
    MODELOS,
    ErrorEnsemble,
    RegresorDiferenciado,
    ajustar,
    comparar_formas,
    crear_modelo,
    estimador_interno,
    evaluar_ensemble,
    extrapolacion,
)
from modelado.particion import particionar, validacion_cruzada  # noqa: E402
from modelado.tiempos import cargar_modelo  # noqa: E402
from test_modelado_base import VERSIONES, _entrenar, guardar_tablas, panel_minimo  # noqa: E402

# Crecimiento total del 30 % entre el primer y el ultimo dia del panel.
CRECIMIENTO = 0.30


@pytest.fixture(scope="module")
def carpeta_tablas(tmp_path_factory) -> Path:
    panel = panel_minimo("2024-06-01", "2026-09-07 23:00")
    panel["valor_kwh"] *= 1 + CRECIMIENTO * np.linspace(0, 1, len(panel))
    return guardar_tablas(panel, tmp_path_factory.mktemp("procesado_creciente"))


@pytest.fixture(scope="module")
def entradas(carpeta_tablas):
    """Por version: X, y, fechas, indices y el TimeSeriesSplit del entrenamiento."""
    resultado = {}
    for version in VERSIONES:
        (X, y, fechas), _ = cargar_tabla(version, carpeta_tablas)
        indices, _ = particionar(fechas)
        tscv, _ = validacion_cruzada(version, fechas.iloc[indices["entrenamiento"]])
        resultado[version] = (X.astype("float64"), y, fechas, indices, tscv)
    return resultado


@pytest.fixture(scope="module")
def evaluados(entradas):
    """Cada ensemble en cada forma evaluado una vez sobre r24."""
    return {
        (nombre, forma): evaluar_ensemble(nombre, forma, *entradas["r24"], mape_baseline_validacion=5.0)
        for nombre in MODELOS
        for forma in FORMAS
    }


# ==========================================================================
# RegresorDiferenciado y extrapolacion
# ==========================================================================


def test_el_regresor_diferenciado_aprende_la_diferencia_y_le_suma_el_rezago():
    X = pd.DataFrame({"lag_168": [100.0, 200.0, 300.0, 400.0], "hora": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([110.0, 230.0, 310.0, 450.0])  # diferencias 10, 30, 10, 50: media 25
    modelo = RegresorDiferenciado(DummyRegressor(strategy="mean")).fit(X, y)

    assert modelo.estimador_.constant_.ravel()[0] == pytest.approx(25.0)
    nuevo = pd.DataFrame({"lag_168": [1000.0, 5.0], "hora": [0.0, 0.0]})
    np.testing.assert_allclose(modelo.predict(nuevo), [1025.0, 30.0])
    assert list(modelo.feature_names_in_) == ["lag_168", "hora"]


def test_el_regresor_diferenciado_rechaza_entradas_sin_su_columna():
    X = pd.DataFrame({"lag_168": [1.0, 2.0, 3.0], "hora": [0.0, 1.0, 2.0]})
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ErrorEnsemble, match="no tiene la columna"):
        RegresorDiferenciado(DummyRegressor()).fit(X.drop(columns="lag_168"), y)
    with pytest.raises(ErrorEnsemble, match="DataFrame"):
        RegresorDiferenciado(DummyRegressor()).fit(X.to_numpy(), y)
    with pytest.raises(ErrorEnsemble, match="NaN"):
        RegresorDiferenciado(DummyRegressor()).fit(X.assign(lag_168=[1.0, np.nan, 3.0]), y)
    with pytest.raises(ErrorEnsemble, match="Falta el estimador"):
        RegresorDiferenciado().fit(X, y)


def test_extrapolacion_coincide_con_el_conteo_a_mano():
    real = np.array([90.0, 105.0, 110.0, 95.0])
    pred = np.array([92.0, 100.0, 101.0, 99.0])
    tope = extrapolacion(real, pred, maximo_entrenamiento=100.0)

    assert tope["predicciones_sobre_maximo"] == 1  # 101
    assert tope["reales_sobre_maximo"] == 2  # 105 y 110
    assert tope["proporcion_reales_sobre_maximo"] == pytest.approx(0.5)
    assert tope["sesgo_medio_reales_sobre_maximo_mwh"] == pytest.approx(((100 - 105) + (101 - 110)) / 2)
    assert tope["sesgo_medio_resto_mwh"] == pytest.approx(((92 - 90) + (99 - 95)) / 2)
    assert tope["prediccion_maxima_mwh"] == 101.0 and tope["real_maximo_mwh"] == 110.0
    assert extrapolacion([1.0], [1.0], 5.0)["sesgo_medio_reales_sobre_maximo_mwh"] is None
    with pytest.raises(ErrorEnsemble):
        extrapolacion([1.0, 2.0], [1.0], 5.0)


# ==========================================================================
# Configuracion: nada de particiones aleatorias ni de OOB
# ==========================================================================


def test_los_ensembles_fijan_la_semilla_y_no_parten_al_azar():
    for nombre in MODELOS:
        for forma in FORMAS:
            interno = crear_modelo(nombre, forma)
            interno = interno.estimador if isinstance(interno, RegresorDiferenciado) else interno
            assert interno.get_params()["random_state"] == 42, nombre
    bosque = crear_modelo("random_forest")
    assert isinstance(bosque, RandomForestRegressor) and bosque.oob_score is False
    rapido = crear_modelo("hist_gradient")
    # early_stopping="auto" separaria al azar un 10 % del entrenamiento con mas de 10 000 filas.
    assert isinstance(rapido, HistGradientBoostingRegressor) and rapido.early_stopping is False
    assert crear_modelo("adaboost").estimator.max_depth is not None
    with pytest.raises(ErrorEnsemble):
        crear_modelo("random_forest", "logaritmo")
    with pytest.raises(ErrorEnsemble):
        crear_modelo("xgboost")


# ==========================================================================
# Evaluacion: extrapolacion, fuga y reproducibilidad
# ==========================================================================


def test_sobre_el_nivel_el_bosque_no_pasa_del_maximo_y_diferenciado_si(evaluados, entradas):
    X, y, _, indices, _ = entradas["r24"]
    maximo = y.iloc[indices["entrenamiento"]].max()
    nivel = evaluados[("random_forest", "nivel")][1]
    diferenciado = evaluados[("random_forest", "diferenciado")][1]

    # Con la tendencia, buena parte de la validacion supera el maximo de entrenamiento.
    assert nivel["extrapolacion_validacion"]["reales_sobre_maximo"] > 0
    assert nivel["extrapolacion_validacion"]["maximo_entrenamiento_mwh"] == pytest.approx(maximo)
    # Las hojas de un bosque son medias de valores vistos: nunca pasan del maximo.
    assert nivel["extrapolacion_validacion"]["predicciones_sobre_maximo"] == 0
    assert nivel["extrapolacion_validacion"]["prediccion_maxima_mwh"] <= maximo
    assert nivel["metricas_validacion"]["sesgo_medio_mwh"] < 0  # subestima
    # Con lag_168 poniendo el nivel, si pasa del maximo y el sesgo se reduce.
    assert diferenciado["extrapolacion_validacion"]["predicciones_sobre_maximo"] > 0
    comparacion = comparar_formas({"nivel": nivel, "diferenciado": diferenciado})
    assert comparacion["diferenciado_reduce_sesgo"]
    assert comparacion["reales_sobre_maximo"] == nivel["extrapolacion_validacion"]["reales_sobre_maximo"]


def test_el_registro_trae_cv_metricas_extrapolacion_y_tiempos(evaluados):
    for (nombre, forma), (modelo, registro) in evaluados.items():
        assert registro["forma"] == forma and registro["semilla"] == 42
        assert registro["validacion_cruzada"]["detalle"]["n_pliegues"] == 5
        resumen = registro["validacion_cruzada"]["detalle"]["resumen"]
        assert len(resumen["sesgo_medio_mwh"]["por_pliegue"]) == 5
        assert resumen["mape_pct"]["desviacion"] is not None
        assert registro["metricas_validacion"]["n"] == 8760
        assert registro["metricas_validacion"]["mejora_mape_pct"] is not None
        assert registro["extrapolacion_validacion"]["n"] == 8760
        tiempos = registro["tiempos"]
        assert tiempos["ajuste_s"] > 0 and tiempos["tamano_serializado_bytes"] > 0
        assert ("nota_oob" in registro) == (nombre == "random_forest")
        assert (registro["importancias"] is None) == (nombre == "hist_gradient")
        json.dumps(registro)  # todo serializable para el JSON de resultados


@pytest.mark.parametrize("nombre", list(MODELOS))
def test_alterar_la_validacion_no_cambia_lo_ajustado(entradas, evaluados, nombre):
    """Lo ajustado depende solo del entrenamiento; ademas, repetir da lo mismo (semilla fija)."""
    X, y, fechas, indices, tscv = entradas["r24"]
    entrena, valida = indices["entrenamiento"], indices["validacion"]
    X_alterada, y_alterada = X.copy(), y.copy()
    X_alterada.iloc[valida] = X_alterada.iloc[valida] * 10 + 1000
    y_alterada.iloc[valida] = y_alterada.iloc[valida] * 3

    for forma in FORMAS:
        modelo, registro = evaluados[(nombre, forma)]
        otro, registro_otro = evaluar_ensemble(
            nombre, forma, X_alterada, y_alterada, fechas, indices, tscv, mape_baseline_validacion=5.0
        )
        np.testing.assert_array_equal(modelo.predict(X.iloc[entrena]), otro.predict(X.iloc[entrena]))
        cv, cv_otro = registro["validacion_cruzada"]["detalle"], registro_otro["validacion_cruzada"]["detalle"]
        assert cv["resumen"] == cv_otro["resumen"] and cv["pliegues"] == cv_otro["pliegues"]
        assert registro["importancias"] == registro_otro["importancias"]
        assert (registro["extrapolacion_validacion"]["maximo_entrenamiento_mwh"]
                == registro_otro["extrapolacion_validacion"]["maximo_entrenamiento_mwh"])
        # Control: la validacion si cambio, y sus metricas lo notan.
        assert registro_otro["metricas_validacion"]["mape_pct"] != registro["metricas_validacion"]["mape_pct"]


def test_el_mismo_ensemble_con_la_misma_semilla_da_lo_mismo(entradas):
    X, y, _, indices, _ = entradas["r120"]
    entrena = indices["entrenamiento"]
    for nombre in MODELOS:
        uno = ajustar(crear_modelo(nombre, "diferenciado"), X.iloc[entrena], y.iloc[entrena])
        dos = ajustar(crear_modelo(nombre, "diferenciado"), X.iloc[entrena], y.iloc[entrena])
        np.testing.assert_array_equal(uno.predict(X), dos.predict(X), err_msg=nombre)
        assert estimador_interno(uno) is uno.estimador_


# ==========================================================================
# scripts/entrenar.py --familia ensembles
# ==========================================================================


def test_entrenar_familia_ensembles_de_punta_a_punta(carpeta_tablas, tmp_path):
    for version in VERSIONES:
        assert _entrenar(carpeta_tablas, tmp_path, version, familia="ensembles") == 0
    resultados = json.loads((tmp_path / "resultados" / "modelado_ensembles.json").read_text(encoding="utf-8"))

    assert resultados["familia"] == "ensembles" and list(resultados["tablas"]) == ["r24", "r120"]
    for version, entrada in resultados["tablas"].items():
        assert entrada["validacion_cruzada"]["detalle"]["gap_h"] == (24 if version == "r24" else 120)
        assert list(entrada["modelos"]) == list(MODELOS) == list(entrada["comparacion_formas"])
        referencia = entrada["baseline_referencia"]["mape_validacion_pct"]
        for nombre, formas in entrada["modelos"].items():
            assert list(formas) == list(FORMAS)
            for forma, registro in formas.items():
                m = registro["metricas_validacion"]
                assert m["mejora_mape_pct"] == pytest.approx(100 * (referencia - m["mape_pct"]) / referencia)
                archivo = tmp_path / "models" / Path(registro["modelo"]["ruta"]).name
                assert archivo.name == f"ensemble_{nombre}_{forma}_{version}.joblib"
                leido = cargar_modelo(archivo, registro["modelo"]["sha256"])
                assert isinstance(leido, RegresorDiferenciado) == (forma == "diferenciado")
        bosque = entrada["comparacion_formas"]["random_forest"]
        assert bosque["nivel"]["predicciones_sobre_maximo"] == 0 and bosque["diferenciado_reduce_sesgo"]
