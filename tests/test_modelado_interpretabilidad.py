"""Pruebas de la interpretabilidad (src/modelado/interpretabilidad.py). Sin red.

Comprueban lo que se puede comprobar a mano: que los valores SHAP suman la
prediccion (tambien en la forma diferenciada), que LinearExplainer da
coef * (x - media del fondo), que la importancia por ganancia de
HistGradientBoosting encuentra la unica variable util, que los casos de LIME
cumplen su regla y que la seleccion de candidatos sale de los resultados.
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

from modelado import clasificacion, ensembles, interpretabilidad, regresion  # noqa: E402
from modelado.baselines import BASELINES  # noqa: E402
from modelado.datos import cargar_tabla  # noqa: E402
from modelado.ensembles import RegresorDiferenciado  # noqa: E402
from modelado.interpretabilidad import (  # noqa: E402
    CALIFICACIONES,
    ErrorInterpretabilidad,
    casos_lime,
    contraste_eda,
    explicador_lime,
    explicar_lime,
    explicar_shap,
    importancia_impureza,
    importancia_permutacion,
    seleccionar_candidatos,
)
from modelado.particion import particionar  # noqa: E402
from test_modelado_base import _entrenar, guardar_tablas, panel_minimo  # noqa: E402


@pytest.fixture(scope="module")
def carpeta_tablas(tmp_path_factory) -> Path:
    return guardar_tablas(panel_minimo("2024-06-01", "2026-09-07 23:00"), tmp_path_factory.mktemp("procesado"))


@pytest.fixture(scope="module")
def datos(carpeta_tablas):
    (X, y, fechas), _ = cargar_tabla("r24", carpeta_tablas)
    indices, _ = particionar(fechas)
    return X.astype("float64"), y, fechas, indices


@pytest.fixture(scope="module")
def ajustados(datos):
    X, y, _, indices = datos
    entrena = indices["entrenamiento"]
    hist = HistGradientBoostingRegressor(max_iter=50, early_stopping=False, random_state=42)
    dif = RegresorDiferenciado(HistGradientBoostingRegressor(max_iter=50, early_stopping=False, random_state=42))
    lineal = regresion.crear_pipeline("lineal")
    return {nombre: m.fit(X.iloc[entrena], y.iloc[entrena])
            for nombre, m in {"hist": hist, "dif": dif, "lineal": lineal}.items()}


# ==========================================================================
# Importancias
# ==========================================================================


def test_la_ganancia_de_histgb_encuentra_la_unica_variable_util():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(2000, 4)), columns=["a", "b", "c", "d"])
    y = 3 * X["c"] + rng.normal(0, 0.01, 2000)
    hist = HistGradientBoostingRegressor(max_iter=30, early_stopping=False, random_state=42).fit(X, y)
    impureza = importancia_impureza(hist, list(X.columns))
    assert list(impureza["por_variable"])[0] == "c"
    assert sum(impureza["por_variable"].values()) == pytest.approx(1.0)
    assert "ganancia" in impureza["metodo"]

    bosque = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
    assert importancia_impureza(bosque, list(X.columns))["metodo"] == "feature_importances_ de sklearn"
    with pytest.raises(ErrorInterpretabilidad):
        importancia_impureza(DummyRegressor().fit(X, y), list(X.columns))


def test_la_permutacion_mide_en_validacion_y_ordena_de_mayor_a_menor(datos, ajustados):
    X, y, _, indices = datos
    valida = indices["validacion"][:2000]
    resultado = importancia_permutacion(ajustados["hist"], X.iloc[valida], y.iloc[valida], n_jobs=1)
    medias = [v["media"] for v in resultado["por_variable"].values()]
    assert medias == sorted(medias, reverse=True) and resultado["filas"] == 2000
    otra = importancia_permutacion(ajustados["hist"], X.iloc[valida], y.iloc[valida], n_jobs=1)
    assert otra["por_variable"] == resultado["por_variable"]  # random_state fijo


# ==========================================================================
# SHAP
# ==========================================================================


@pytest.mark.parametrize("nombre, rol", [("hist", "ensemble"), ("dif", "ensemble"), ("lineal", "lineal")])
def test_los_valores_shap_suman_la_prediccion(datos, ajustados, nombre, rol):
    X, _, _, indices = datos
    X_valida = X.iloc[indices["validacion"][:500]]
    posiciones, valores, base, registro = explicar_shap(rol, ajustados[nombre], X_valida, X.iloc[indices["entrenamiento"]])
    prediccion = ajustados[nombre].predict(X_valida.iloc[posiciones])
    np.testing.assert_allclose(valores.sum(axis=1) + base, prediccion, rtol=0, atol=1e-6 * np.abs(prediccion).max())
    assert registro["filas_explicadas"] == 500 and not registro["muestra"]
    assert list(registro["media_abs_por_variable_mwh"])[0] in X.columns


def test_linear_explainer_es_coef_por_x_menos_la_media_del_fondo(datos, ajustados):
    X, _, _, indices = datos
    modelo = ajustados["lineal"]
    X_valida = X.iloc[indices["validacion"][:200]]
    _, valores, _, _ = explicar_shap("lineal", modelo, X_valida, X.iloc[indices["entrenamiento"]])
    fondo = modelo[:-1].transform(interpretabilidad.muestra_fondo(X.iloc[indices["entrenamiento"]]))
    esperado = modelo[-1].coef_ * (modelo[:-1].transform(X_valida) - fondo.mean(axis=0))
    np.testing.assert_allclose(valores, esperado, atol=1e-6)


def test_si_shap_es_lento_usa_una_muestra_y_lo_anota(monkeypatch, datos, ajustados):
    X, _, _, indices = datos
    monkeypatch.setattr(interpretabilidad, "LIMITE_SHAP_S", 0.0)
    monkeypatch.setattr(interpretabilidad, "MUESTRA_SHAP", 150)
    X_valida = X.iloc[indices["validacion"][:1000]]
    posiciones, valores, _, registro = explicar_shap("ensemble", ajustados["hist"], X_valida, X.iloc[indices["entrenamiento"]])
    assert registro["muestra"] and registro["filas_explicadas"] == 150 == len(posiciones) == len(valores)
    assert np.all(np.diff(posiciones) > 0) and "muestra aleatoria de 150" in registro["motivo_muestra"]


# ==========================================================================
# LIME
# ==========================================================================


def test_los_casos_de_lime_cumplen_su_regla(datos, ajustados):
    X, y, fechas, indices = datos
    valida = indices["validacion"]
    X_valida, y_valida, fechas_valida = X.iloc[valida], y.iloc[valida], fechas.iloc[valida]
    predicciones = {rol: ajustados[m].predict(X_valida) for rol, m in (("ensemble", "hist"), ("lineal", "lineal"))}
    casos = casos_lime(fechas_valida, X_valida, y_valida, predicciones)

    assert set(casos) == {"pico_laboral", "festivo", "semana_santa", "mayor_error_ensemble", "mayor_error_lineal"}
    momento = {k: pd.Timestamp(c["fecha_hora"]) for k, c in casos.items()}
    fila = {k: X_valida.iloc[c["posicion"]] for k, c in casos.items()}
    assert momento["pico_laboral"].hour == 19 and momento["pico_laboral"].dayofweek < 5
    assert fila["pico_laboral"][["es_festivo", "es_semana_santa", "es_puente", "es_vispera_festivo"]].sum() == 0
    assert fila["festivo"]["es_festivo"] == 1 and fila["festivo"]["es_semana_santa"] == 0
    assert fila["semana_santa"]["es_semana_santa"] == 1 and fila["semana_santa"]["es_festivo"] == 0
    for rol, prediccion in predicciones.items():
        error = np.abs(prediccion - y_valida.to_numpy())
        assert casos[f"mayor_error_{rol}"]["posicion"] == int(np.argmax(error))


def test_lime_es_reproducible_y_explica_la_prediccion_del_modelo(datos, ajustados):
    X, y, fechas, indices = datos
    valida = indices["validacion"]
    explicador, binarias = explicador_lime(X.iloc[indices["entrenamiento"]])
    assert [X.columns[i] for i in binarias] == [c for c in X.columns if c.startswith("es_")]
    caso = {"posicion": 100}
    una = explicar_lime(explicador, ajustados["lineal"], X.iloc[valida], caso)
    explicador_otra_vez, _ = explicador_lime(X.iloc[indices["entrenamiento"]])
    otra = explicar_lime(explicador_otra_vez, ajustados["lineal"], X.iloc[valida], caso)
    assert una == otra
    assert una["prediccion_mwh"] == pytest.approx(ajustados["lineal"].predict(X.iloc[valida[[100]]])[0])
    assert len(una["pesos"]) == interpretabilidad.VARIABLES_LIME


# ==========================================================================
# Contraste, calificaciones y seleccion
# ==========================================================================


def test_el_contraste_cuenta_donde_domina_lag_168():
    medidas = {
        "ensemble": {"impureza": ["lag_168", "lag_24", "x"], "permutacion": ["lag_24", "lag_168", "x"]},
        "lineal": {"shap": ["lag_168", "x"]},
    }
    contraste = contraste_eda(medidas, {"24": 0.857, "168": 0.924})
    assert contraste["lag_168_primera_en"] == 2 and contraste["de"] == 3 and not contraste["lag_168_domina"]
    assert contraste["acf_168"] > contraste["acf_24"]
    fila = next(f for f in contraste["medidas"] if f["medida"] == "permutacion")
    assert fila["puesto_lag_168"] == 2 and fila["lag_168_antes_que_lag_24"] is False
    assert next(f for f in contraste["medidas"] if f["modelo"] == "lineal")["puesto_lag_24"] is None
    assert "lag_24" in contraste["conclusion"]


def test_hay_una_calificacion_para_cada_modelo_del_proyecto():
    nombres = (set(regresion.MODELOS) | set(ensembles.MODELOS) | set(clasificacion.MODELOS)
               | {f"baseline_{b}" for b in BASELINES})
    assert nombres <= set(CALIFICACIONES)
    for nombre, calificacion in CALIFICACIONES.items():
        assert calificacion["calificacion"] in {"alta", "media", "baja"} and calificacion["motivo"], nombre


def _resultados(busqueda: bool = True) -> dict:
    def modelo(mape_pct, archivo):
        return {"metricas_validacion": {"mape_pct": mape_pct}, "modelo": {"ruta": f"models/{archivo}", "sha256": "x"}}

    resultados = {
        "lineal": {"tablas": {"r24": {"modelos": {
            "polinomial": modelo(2.1, "p.joblib"), "ridge": modelo(2.7, "r.joblib"), "lineal": modelo(2.8, "l.joblib"),
        }}}},
        "ensembles": {"tablas": {"r24": {"modelos": {
            "hist_gradient": {"nivel": modelo(1.9, "hn.joblib"), "diferenciado": modelo(2.0, "hd.joblib")},
        }}}},
    }
    if busqueda:
        resultados["busqueda"] = {"tablas": {"r24": {"modelos": {
            "hist_gradient (nivel)": {"familia": "ensembles", "nombre": "hist_gradient", "forma": "nivel",
                                      "grid": modelo(1.8, "bg.joblib"), "random": modelo(1.85, "br.joblib")},
            "svm_rbf": {"familia": "clasificacion", "nombre": "svm_rbf", "forma": "-"},
        }}}}
    return resultados


def test_la_seleccion_excluye_la_polinomial_e_incluye_la_busqueda():
    candidatos, registro = seleccionar_candidatos(_resultados(), "r24")
    assert candidatos["lineal"].nombre == "ridge"  # la polinomial tiene menos MAPE pero no es lineal en X
    assert "polinomial" in registro["detalle"]["lineales_excluidos"]
    assert candidatos["ensemble"].ruta == "models/bg.joblib" and "grid" in candidatos["ensemble"].origen

    sin_busqueda, registro = seleccionar_candidatos(_resultados(busqueda=False), "r24")
    assert sin_busqueda["ensemble"].ruta == "models/hn.joblib" and not registro["detalle"]["busqueda_incluida"]
    with pytest.raises(ErrorInterpretabilidad, match="Faltan"):
        seleccionar_candidatos(_resultados(), "r120")


# ==========================================================================
# scripts/entrenar.py --familia interpretabilidad
# ==========================================================================


def test_entrenar_familia_interpretabilidad_de_punta_a_punta(monkeypatch, carpeta_tablas, tmp_path):
    # Solo HistGB entre los ensembles, para que la prueba tarde poco.
    monkeypatch.setattr(ensembles, "MODELOS", {"hist_gradient": ensembles.MODELOS["hist_gradient"]})
    for familia in ("lineal", "ensembles"):
        assert _entrenar(carpeta_tablas, tmp_path, "r24", familia=familia) == 0
    eda = {"acf": {"24": 0.857, "168": 0.924}}
    (tmp_path / "resultados" / "eda.json").write_text(json.dumps(eda), encoding="utf-8")

    assert _entrenar(carpeta_tablas, tmp_path, "r24", familia="interpretabilidad") == 0
    resultados = json.loads((tmp_path / "resultados" / "modelado_interpretabilidad.json").read_text(encoding="utf-8"))
    entrada = resultados["tablas"]["r24"]

    assert entrada["modelos"]["ensemble"]["candidato"]["nombre"] == "hist_gradient"
    assert entrada["modelos"]["lineal"]["candidato"]["nombre"] in interpretabilidad.LINEALES_EN_VARIABLES
    assert entrada["contraste_eda"]["acf_168"] == 0.924
    assert entrada["modelos"]["ensemble"]["calificacion"]["calificacion"] == "baja"
    assert entrada["modelos"]["lineal"]["calificacion"]["calificacion"] == "alta"
    for clave in ("pico_laboral", "festivo", "semana_santa", "mayor_error_ensemble", "mayor_error_lineal"):
        assert clave in entrada["lime"]["casos"]
    for ruta in [*entrada["figuras"].values(), entrada["modelos"]["ensemble"]["shap"]["figura_resumen"]]:
        assert (tmp_path / "figuras" / Path(ruta).name).exists()


def test_interpretabilidad_sin_eda_o_sin_familias_falla(carpeta_tablas, tmp_path):
    with pytest.raises(FileNotFoundError, match="corre antes"):
        _entrenar(carpeta_tablas, tmp_path, "r24", familia="interpretabilidad")
