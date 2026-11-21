"""Pruebas del objetivo alerta_pico y de los clasificadores (src/modelado). Sin red.

La prueba central es de fuga temporal, en el estilo de tests/test_variables.py:
el umbral de la fila t solo puede depender de demanda de t - rezago_minimo_h o
antes. Si alterar la demanda posterior lo cambiara, el objetivo llevaria
informacion que no se tiene al predecir. Todo corre para r24 y para r120.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado.clasificacion import (  # noqa: E402
    MODELOS,
    SUBMUESTRA_RBF,
    evaluar_baseline,
    evaluar_clasificador,
)
from modelado.datos import cargar_tabla  # noqa: E402
from modelado.objetivos import ErrorObjetivo, construir_alerta_pico, umbral_movil  # noqa: E402
from modelado.particion import particionar  # noqa: E402
from test_modelado_base import VERSIONES, _entrenar, guardar_tablas, panel_minimo  # noqa: E402

REZAGOS = [24, 120]
VENTANA = 168
ZONA = "America/Bogota"


def serie_horaria(dias: int = 40, semilla: int = 11) -> tuple[pd.Series, np.ndarray]:
    """Demanda con ciclo diario, tendencia y ruido: cada hora con un valor distinto."""
    fechas = pd.Series(pd.date_range("2025-02-03", periods=dias * 24, freq="h", tz=ZONA))
    rng = np.random.default_rng(semilla)
    hora = fechas.dt.hour.to_numpy()
    demanda = 9000 + 1200 * np.sin(2 * np.pi * (hora - 9) / 24) + 2.0 * np.arange(len(fechas)) + rng.normal(0, 150, len(fechas))
    return fechas, demanda


@pytest.fixture(scope="module")
def carpeta_tablas(tmp_path_factory) -> Path:
    return guardar_tablas(panel_minimo("2024-06-01", "2026-09-07 23:00"), tmp_path_factory.mktemp("procesado"))


# ==========================================================================
# objetivos: alerta_pico y su umbral, sin fuga temporal
# ==========================================================================


@pytest.mark.parametrize("r", REZAGOS, ids=lambda r: f"r{r}")
def test_alterar_la_demanda_posterior_a_t_menos_r_no_cambia_el_umbral(r):
    fechas, demanda = serie_horaria()
    base = umbral_movil(fechas, demanda, r)
    corte = len(fechas) // 2

    alterada = demanda.copy()
    alterada[corte] *= 10
    alterada[corte + 1 :] += 1.0e6
    perturbado = umbral_movil(fechas, alterada, r)

    # La fila t usa demanda hasta t - r: hasta corte + r - 1 nada cambia, con
    # exactitud, aunque la demanda de esas mismas filas si cambio.
    np.testing.assert_array_equal(base.iloc[: corte + r].to_numpy(), perturbado.iloc[: corte + r].to_numpy())
    # Y la perturbacion se ve donde debe: en t = corte + r la ventana ya incluye el corte.
    assert perturbado.iloc[corte + r] != base.iloc[corte + r]

    # El objetivo de esas filas si puede cambiar (usa la demanda de t), el umbral no.
    marco_base, _ = construir_alerta_pico(fechas, demanda, r)
    marco_pert, _ = construir_alerta_pico(fechas, alterada, r)
    np.testing.assert_array_equal(
        marco_base["umbral_alerta_mwh"].iloc[: corte + r], marco_pert["umbral_alerta_mwh"].iloc[: corte + r]
    )
    assert marco_pert["alerta_pico"].iloc[corte + 1 : corte + r].eq(1).all()


@pytest.mark.parametrize("r", REZAGOS, ids=lambda r: f"r{r}")
def test_el_umbral_es_el_percentil_90_de_la_ventana_de_ma_168(r):
    fechas, demanda = serie_horaria()
    marco, registro = construir_alerta_pico(fechas, demanda, r)

    arranque = r + VENTANA - 1
    assert marco["umbral_alerta_mwh"].iloc[:arranque].isna().all()
    assert marco["alerta_pico"].iloc[:arranque].isna().all()
    assert registro["detalle"]["filas_sin_umbral"] == arranque
    for t in (arranque, arranque + 37, len(fechas) - 1):
        ventana = demanda[t - r - VENTANA + 1 : t - r + 1]
        assert len(ventana) == VENTANA
        assert marco["umbral_alerta_mwh"].iloc[t] == pytest.approx(np.percentile(ventana, 90), rel=1e-12)
        assert marco["alerta_pico"].iloc[t] == float(demanda[t] > np.percentile(ventana, 90))
    assert set(marco["alerta_pico"].dropna().unique()) <= {0.0, 1.0}


def test_el_umbral_exige_rejilla_y_ventana_completas():
    fechas, demanda = serie_horaria(dias=5)
    with pytest.raises(ErrorObjetivo, match="mas corta"):
        construir_alerta_pico(fechas, demanda, 24)
    fechas, demanda = serie_horaria()
    with pytest.raises(Exception, match="no es continua"):
        construir_alerta_pico(fechas.drop(index=300), np.delete(demanda, 300), 24)


@pytest.mark.parametrize("version", VERSIONES)
def test_la_proporcion_de_positivos_no_mira_el_bloque_de_prueba(carpeta_tablas, version):
    (X, y, fechas), _ = cargar_tabla(version, carpeta_tablas)
    indices, _ = particionar(fechas)
    marco, registro = construir_alerta_pico(fechas, y, version, indices)

    por_bloque = registro["detalle"]["por_bloque"]
    assert por_bloque["prueba"]["proporcion_positivos"] is None
    assert "reservado" in por_bloque["prueba"]["nota"]
    valida = marco["alerta_pico"].iloc[indices["validacion"]]
    assert por_bloque["validacion"]["proporcion_positivos"] == pytest.approx(valida.mean())
    assert 0 < por_bloque["validacion"]["proporcion_positivos"] < 1
    assert por_bloque["entrenamiento"]["filas_con_objetivo"] == len(indices["entrenamiento"]) - (
        registro["detalle"]["filas_sin_umbral"]
    )


# ==========================================================================
# clasificacion: sin validacion en el ajuste, reproducible, submuestra temporal
# ==========================================================================

CLAVES_VARIABLES = {"momento", "segundos", "tiempos", "segundos_busqueda", "segundos_ajuste_final"}


def _sin_variables(objeto):
    if isinstance(objeto, dict):
        return {k: _sin_variables(v) for k, v in objeto.items() if k not in CLAVES_VARIABLES}
    if isinstance(objeto, list):
        return [_sin_variables(v) for v in objeto]
    return objeto


@pytest.fixture(scope="module")
def entradas(carpeta_tablas):
    resultado = {}
    for version in VERSIONES:
        (X, y, fechas), _ = cargar_tabla(version, carpeta_tablas)
        indices, _ = particionar(fechas)
        marco, _ = construir_alerta_pico(fechas, y, version, indices)
        resultado[version] = (X.astype("float64"), marco, fechas, indices)
    return resultado


@pytest.fixture(scope="module")
def evaluados(entradas):
    return {
        (version, nombre): evaluar_clasificador(
            nombre, X, marco["alerta_pico"].to_numpy(), fechas, indices, version
        )
        for version, (X, marco, fechas, indices) in entradas.items()
        for nombre in MODELOS
    }


@pytest.mark.parametrize("nombre", list(MODELOS))
@pytest.mark.parametrize("version", VERSIONES)
def test_el_clasificador_no_se_ajusta_con_datos_de_validacion(entradas, evaluados, version, nombre):
    X, marco, fechas, indices = entradas[version]
    valida = indices["validacion"]
    modelo, registro, _ = evaluados[(version, nombre)]
    assert modelo.named_steps["escalar"].n_samples_seen_ == registro["filas_ajuste"]

    X_alterada = X.copy()
    X_alterada.iloc[valida] = X_alterada.iloc[valida] * 10 + 1000
    objetivo = marco["alerta_pico"].to_numpy().copy()
    objetivo[valida] = 1 - objetivo[valida]
    alterado, registro_alterado, _ = evaluar_clasificador(nombre, X_alterada, objetivo, fechas, indices, version)

    assert registro_alterado["hiperparametros"] == registro["hiperparametros"]
    assert _sin_variables(registro_alterado["busqueda"]) == _sin_variables(registro["busqueda"])
    np.testing.assert_array_equal(modelo.named_steps["escalar"].mean_, alterado.named_steps["escalar"].mean_)
    entrena = indices["entrenamiento"]
    np.testing.assert_array_equal(modelo.predict(X.iloc[entrena]), alterado.predict(X.iloc[entrena]))
    # Control: la validacion si cambio.
    assert registro_alterado["metricas_validacion"] != registro["metricas_validacion"]


@pytest.mark.parametrize("version", VERSIONES)
def test_la_clasificacion_es_reproducible(entradas, evaluados, version):
    X, marco, fechas, indices = entradas[version]
    for nombre in MODELOS:
        _, registro, puntajes = evaluados[(version, nombre)]
        _, otra_vez, puntajes_otra_vez = evaluar_clasificador(
            nombre, X, marco["alerta_pico"].to_numpy(), fechas, indices, version
        )
        assert _sin_variables(otra_vez) == _sin_variables(registro), nombre
        np.testing.assert_array_equal(puntajes, puntajes_otra_vez)


def test_se_elige_por_pr_auc_y_no_por_accuracy(evaluados):
    for (_, nombre), (_, registro, _) in evaluados.items():
        busqueda = registro["busqueda"]["detalle"]
        assert busqueda["metrica_seleccion"] == "pr_auc"
        mejor = max(busqueda["candidatos"], key=lambda c: c["pr_auc_media"])
        assert mejor["hiperparametros"] == registro["hiperparametros"], nombre


@pytest.mark.parametrize("version", VERSIONES)
def test_la_svm_rbf_usa_una_submuestra_temporal_en_orden(entradas, evaluados, version):
    _, marco, fechas, indices = entradas[version]
    _, registro, _ = evaluados[(version, "svm_rbf")]
    submuestra = registro["submuestra"]
    definidas = int(marco["alerta_pico"].iloc[indices["entrenamiento"]].notna().sum())

    assert submuestra["paso_h"] == SUBMUESTRA_RBF and np.gcd(SUBMUESTRA_RBF, 168) == 1
    assert submuestra["filas"] == registro["filas_ajuste"] == -(-definidas // SUBMUESTRA_RBF)
    desde, hasta = pd.Timestamp(submuestra["desde"]), pd.Timestamp(submuestra["hasta"])
    assert (hasta - desde) == pd.Timedelta(hours=SUBMUESTRA_RBF * (submuestra["filas"] - 1))
    assert submuestra["segundos_busqueda"] > 0 and submuestra["segundos_ajuste_final"] > 0
    assert "O(n^2)" in submuestra["motivo"]
    assert registro["validacion_cruzada"]["detalle"]["gap_h"] >= (24 if version == "r24" else 120)
    # Las otras se ajustan con todo el entrenamiento con objetivo.
    assert evaluados[(version, "svm_lineal")][1]["filas_ajuste"] == definidas


@pytest.mark.parametrize("version", VERSIONES)
def test_el_baseline_alerta_si_lag_168_supera_el_umbral(entradas, version):
    X, marco, _, indices = entradas[version]
    valida = indices["validacion"]
    registro, puntajes = evaluar_baseline(X, marco["alerta_pico"].to_numpy(), marco["umbral_alerta_mwh"].to_numpy(), indices)

    esperado = X["lag_168"].to_numpy()[valida] > marco["umbral_alerta_mwh"].to_numpy()[valida]
    np.testing.assert_array_equal(puntajes > 0, esperado)
    real = marco["alerta_pico"].to_numpy()[valida].astype(int)
    matriz = registro["metricas_validacion"]["matriz_confusion"]
    assert matriz["vp"] == int((esperado & (real == 1)).sum())
    assert matriz["fp"] == int((esperado & (real == 0)).sum())


def test_entrenar_familia_clasificacion_de_punta_a_punta(carpeta_tablas, tmp_path):
    for version in VERSIONES:
        assert _entrenar(carpeta_tablas, tmp_path, version, familia="clasificacion") == 0
    resultados = json.loads((tmp_path / "resultados" / "modelado_clasificacion.json").read_text(encoding="utf-8"))

    assert resultados["familia"] == "clasificacion" and list(resultados["tablas"]) == ["r24", "r120"]
    for version, entrada in resultados["tablas"].items():
        assert list(entrada["modelos"]) == list(MODELOS)
        assert entrada["objetivo"]["detalle"]["por_bloque"]["prueba"]["proporcion_positivos"] is None
        assert (tmp_path / "figuras" / f"precision_recall_{version}.png").stat().st_size > 0
        for registro in [*entrada["modelos"].values(), entrada["baseline"]]:
            m = registro["metricas_validacion"]
            assert m["n"] == 8760 and 0 <= m["pr_auc"] <= 1
