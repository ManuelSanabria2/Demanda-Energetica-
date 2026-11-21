"""Pruebas del objetivo alerta_pico y de los clasificadores (src/modelado). Sin red.

La prueba central es de fuga temporal, en el estilo de tests/test_variables.py:
el umbral de la fila t solo puede depender de demanda de t - rezago_minimo_h o
antes. Si alterar la demanda posterior lo cambiara, el objetivo llevaria
informacion que no se tiene al predecir. Todo corre para r24 y para r120.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado.datos import cargar_tabla  # noqa: E402
from modelado.objetivos import ErrorObjetivo, construir_alerta_pico, umbral_movil  # noqa: E402
from modelado.particion import particionar  # noqa: E402
from test_modelado_base import VERSIONES, guardar_tablas, panel_minimo  # noqa: E402

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
