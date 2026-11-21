"""Pruebas de la base del modelado supervisado (src/modelado). Sin red.

Lo que cubren falla en silencio: una variable que no deberia estar en X, un
bloque de validacion que se solapa con el entrenamiento, un pliegue de
TimeSeriesSplit que valida antes de lo que entrena, un escalado que ve el
periodo de prueba. Todo eso da metricas plausibles y demasiado buenas.

Las tablas sinteticas se construyen con `variables.construir`, igual que las
reales, a partir de un panel minimo (demanda, temperatura y calendario) que
cubre los tres bloques de la particion. Todo corre para r24 y para r120.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from ingesta.complementarias import calendario_horario  # noqa: E402
from modelado.datos import (  # noqa: E402
    ErrorDatos,
    cargar_tabla,
    clave_version,
    rezago_de,
    separar,
    variables_conocidas,
)
from modelado.particion import (  # noqa: E402
    BLOQUES,
    ErrorParticion,
    inicio_prueba,
    particionar,
    validacion_cruzada,
)
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from variables.construir import BANDERAS_CALENDARIO, construir_tabla, guardar  # noqa: E402

VERSIONES = ["r24", "r120"]
ZONA = "America/Bogota"


def panel_minimo(desde: str, hasta: str, semilla: int = 42) -> pd.DataFrame:
    """Lo justo para `construir_tabla`: demanda, temperatura y banderas del calendario.

    La demanda tiene forma diaria y semanal y un 3 % de ruido: lag_168 no la
    reproduce exactamente, asi que el MAPE de los baselines queda lejos del
    umbral de fuga del 1 %.
    """
    horas = pd.date_range(desde, hasta, freq="h", tz=ZONA)
    rng = np.random.default_rng(semilla)
    hora, dia = horas.hour.to_numpy(), horas.dayofweek.to_numpy()
    forma = 1 + 0.15 * np.sin(2 * np.pi * (hora - 9) / 24) - 0.08 * (dia >= 5)
    valor = 9.0e6 * forma * (1 + rng.normal(0, 0.03, len(horas)))
    calendario, _ = calendario_horario(horas[0].date(), horas[-1].date())
    banderas = calendario.set_index("fecha_hora")[list(BANDERAS_CALENDARIO)].reindex(horas)
    panel = pd.DataFrame(
        {
            "fecha_hora": horas,
            "valor_kwh": valor,
            "temperatura_c": 20 + 4 * np.sin(2 * np.pi * (hora - 9) / 24) + rng.normal(0, 0.5, len(horas)),
        }
    )
    for bandera in BANDERAS_CALENDARIO:
        panel[bandera] = banderas[bandera].fillna(False).to_numpy(bool)
    return panel


def guardar_tablas(panel: pd.DataFrame, carpeta: Path) -> Path:
    """Construye r24 y r120 y las guarda con su registro, como scripts/construir_variables.py."""
    guardar({r: construir_tabla(panel, r) for r in (24, 120)}, carpeta)
    return carpeta


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    # Del 2024-06-01 al 2026-09-07: tras quitar el arranque y los 5 ultimos dias,
    # la tabla cubre entrenamiento, validacion y prueba hasta 2026-09-02 23:00.
    return panel_minimo("2024-06-01", "2026-09-07 23:00")


@pytest.fixture(scope="module")
def carpeta_tablas(panel, tmp_path_factory) -> Path:
    return guardar_tablas(panel, tmp_path_factory.mktemp("procesado"))


@pytest.fixture(scope="module")
def tablas(carpeta_tablas) -> dict[str, pd.DataFrame]:
    return {v: pd.read_parquet(carpeta_tablas / f"tabla_modelado_{v}.parquet") for v in VERSIONES}


# ==========================================================================
# datos: X, y y fechas, con la lista de variables tomada de la tabla
# ==========================================================================


@pytest.mark.parametrize("version", VERSIONES)
def test_x_sale_de_la_tabla_sin_tiempo_ni_objetivo(carpeta_tablas, tablas, version):
    (X, y, fechas), registro = cargar_tabla(version, carpeta_tablas)
    tabla = tablas[version]

    assert "fecha_hora" not in X.columns and "demanda_mwh" not in X.columns
    assert list(X.columns) == [c for c in tabla.columns if c not in ("fecha_hora", "demanda_mwh")]
    assert list(X.columns) == registro["variables"]
    assert ("lag_24" in X.columns) == (version == "r24")
    assert len(X) == len(y) == len(fechas) == len(tabla)
    np.testing.assert_array_equal(y.to_numpy(), tabla["demanda_mwh"].to_numpy())
    assert str(fechas.dt.tz) == ZONA
    assert registro["operaciones"][0]["detalle"]["coincide"] is True


@pytest.mark.parametrize(
    ("columna", "mensaje"),
    [
        ("precio_bolsa", "desconocidas"),
        ("atipico", "fuga del objetivo"),
        ("atipico_global", "fuga temporal"),
        ("valor_kwh", "sustituida por demanda_mwh"),
    ],
)
def test_una_columna_que_no_es_variable_conocida_hace_fallar(tablas, columna, mensaje):
    tabla = tablas["r24"].copy()
    tabla[columna] = 1.0
    with pytest.raises(ErrorDatos, match=mensaje):
        separar(tabla, 24)


def test_un_rezago_menor_que_el_minimo_de_la_tabla_es_fuga(tablas):
    with pytest.raises(ErrorDatos, match="lag_24"):
        separar(tablas["r24"], 120)


def test_una_rejilla_con_hueco_hace_fallar(tablas):
    tabla = tablas["r24"].drop(index=500)
    with pytest.raises(ErrorDatos, match="no es continua"):
        separar(tabla, 24)


def test_una_tabla_que_no_coincide_con_su_sha256_no_se_carga(carpeta_tablas, tmp_path):
    copia = Path(shutil.copytree(carpeta_tablas, tmp_path / "copia"))
    ruta = copia / "tabla_modelado_r24.parquet"
    tabla = pd.read_parquet(ruta)
    tabla.loc[100, "demanda_mwh"] += 1.0
    tabla.to_parquet(ruta, index=False)
    with pytest.raises(ErrorDatos, match="SHA-256"):
        cargar_tabla("r24", copia)


def test_versiones_y_variables_conocidas():
    assert rezago_de("r24") == 24 and rezago_de("R120") == 120 and rezago_de(120) == 120
    assert clave_version(24) == "r24"
    with pytest.raises(ErrorDatos):
        rezago_de("r1")
    with pytest.raises(ErrorDatos):
        rezago_de("24h")
    assert {"lag_24", "lag_168", "ma_168", "temperatura_c", "hora_sin", "es_festivo"} <= variables_conocidas()
    assert not {"demanda_mwh", "fecha_hora", "atipico", "valor_kwh"} & variables_conocidas()


# ==========================================================================
# particion: bloques cronologicos y TimeSeriesSplit
# ==========================================================================


def _hora(texto: str) -> pd.Timestamp:
    return pd.Timestamp(texto, tz=ZONA)


@pytest.mark.parametrize("version", VERSIONES)
def test_los_bloques_no_se_solapan_y_van_en_orden(carpeta_tablas, version):
    (_, _, fechas), _ = cargar_tabla(version, carpeta_tablas)
    indices, registro = particionar(fechas)

    entrena, valida, prueba = (indices[b] for b in BLOQUES)
    assert not (set(entrena) & set(valida)) and not (set(valida) & set(prueba)) and not (set(entrena) & set(prueba))
    assert entrena.max() < valida.min() and valida.max() < prueba.min()
    assert len(entrena) + len(valida) + len(prueba) == len(fechas)

    assert fechas.iloc[entrena].max() == _hora("2024-09-02 23:00")
    assert fechas.iloc[valida].min() == _hora("2024-09-03 00:00")
    assert fechas.iloc[valida].max() == _hora("2025-09-02 23:00")
    assert fechas.iloc[prueba].min() == _hora("2025-09-03 00:00") == inicio_prueba()
    assert fechas.iloc[prueba].max() == _hora("2026-09-02 23:00")

    bloques = registro["detalle"]["bloques"]
    assert [bloques[b]["filas"] for b in BLOQUES] == [len(entrena), len(valida), len(prueba)]
    assert bloques["validacion"]["desde"] == str(_hora("2024-09-03 00:00"))
    assert "evaluacion final" in bloques["prueba"]["uso"]


def test_particionar_falla_con_filas_fuera_de_los_bloques_o_desordenadas():
    horas = pd.date_range("2024-01-01", "2026-09-03 05:00", freq="h", tz=ZONA)
    with pytest.raises(ErrorParticion, match="posteriores"):
        particionar(horas)
    with pytest.raises(ErrorParticion, match="ordenadas"):
        particionar(horas[:-6][::-1])
    with pytest.raises(ErrorParticion, match="vacios"):
        particionar(pd.date_range("2025-01-01", periods=100, freq="h", tz=ZONA))
    with pytest.raises(ErrorParticion, match="zona horaria"):
        particionar(pd.date_range("2024-01-01", "2026-09-02 23:00", freq="h"))


@pytest.mark.parametrize(("version", "gap"), [("r24", 24), ("r120", 120)])
def test_ningun_pliegue_valida_antes_de_lo_que_entrena(carpeta_tablas, version, gap):
    (_, _, fechas), _ = cargar_tabla(version, carpeta_tablas)
    indices, _ = particionar(fechas)
    fechas_entrena = fechas.iloc[indices["entrenamiento"]].reset_index(drop=True)
    tscv, registro = validacion_cruzada(version, fechas_entrena)

    assert tscv.n_splits == 5 and tscv.gap == gap
    pliegues = list(tscv.split(np.arange(len(fechas_entrena))))
    assert len(pliegues) == 5
    for entrena, valida in pliegues:
        assert valida.min() > entrena.max()
        assert valida.min() - entrena.max() - 1 == gap
        separacion = fechas_entrena.iloc[valida].min() - fechas_entrena.iloc[entrena].max()
        assert separacion == pd.Timedelta(hours=gap + 1)
    assert [p["gap_h"] for p in registro["detalle"]["pliegues"]] == [gap] * 5


def test_la_validacion_cruzada_exige_rejilla_horaria():
    horas = pd.date_range("2023-01-01", periods=2000, freq="h", tz=ZONA).delete(700)
    with pytest.raises(ErrorDatos, match="no es continua"):
        validacion_cruzada(24, horas)
    with pytest.raises(ErrorDatos, match="horizonte"):
        validacion_cruzada(12, horas)


@pytest.mark.parametrize("version", VERSIONES)
def test_un_pipeline_ajustado_en_entrenamiento_no_cambia_si_se_altera_la_prueba(carpeta_tablas, version):
    (X, y, fechas), _ = cargar_tabla(version, carpeta_tablas)
    X = X.astype("float64")
    indices, _ = particionar(fechas)
    entrena, valida, prueba = (indices[b] for b in BLOQUES)

    def ajustar(X_, y_):
        modelo = Pipeline([("escalar", StandardScaler()), ("regresion", Ridge(alpha=1.0))])
        return modelo.fit(X_.iloc[entrena], y_.iloc[entrena])

    original = ajustar(X, y)
    X_alterada, y_alterada = X.copy(), y.copy()
    X_alterada.iloc[prueba] = X_alterada.iloc[prueba] * 10 + 1000
    y_alterada.iloc[prueba] = y_alterada.iloc[prueba] * 3
    alterado = ajustar(X_alterada, y_alterada)

    for paso in ("mean_", "scale_"):
        np.testing.assert_array_equal(
            getattr(original.named_steps["escalar"], paso), getattr(alterado.named_steps["escalar"], paso)
        )
    np.testing.assert_array_equal(original.named_steps["regresion"].coef_, alterado.named_steps["regresion"].coef_)
    np.testing.assert_array_equal(original.predict(X.iloc[valida]), alterado.predict(X_alterada.iloc[valida]))

    # Control: la alteracion si se notaria si el escalado viera la tabla entera.
    completo = StandardScaler().fit(X), StandardScaler().fit(X_alterada)
    assert not np.allclose(completo[0].mean_, completo[1].mean_)
