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
from modelado.baselines import (  # noqa: E402
    BaselineRezago,
    ErrorBaseline,
    baselines_disponibles,
)
from modelado.datos import (  # noqa: E402
    ErrorDatos,
    cargar_tabla,
    clave_version,
    rezago_de,
    separar,
    variables_conocidas,
)
from modelado.metricas import (  # noqa: E402
    AlertaFuga,
    ErrorMetricas,
    error_maximo_diario,
    mae,
    mape,
    mape_horas_pico,
    mejora_mape,
    metricas_clasificacion,
    metricas_regresion,
    rmse,
    sesgo_medio,
)
from modelado.particion import (  # noqa: E402
    BLOQUES,
    ErrorParticion,
    inicio_prueba,
    particionar,
    validacion_cruzada,
)
from modelado.tiempos import (  # noqa: E402
    ErrorModelo,
    cargar_modelo,
    entorno,
    guardar_modelo,
    medir_ajuste,
    medir_prediccion,
    tamano_serializado,
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


# ==========================================================================
# metricas: contra calculos a mano en un ejemplo pequeno
# ==========================================================================


@pytest.fixture
def ejemplo_regresion():
    """Dos dias, real = 1000 MWh en todas las horas; tres errores puestos a mano.

    dia 1, 19 h: +100   (hora pico; sube el maximo del dia 1 a 1100)
    dia 2,  3 h: -100   (fuera de pico)
    dia 2, 18 h:  -50   (hora pico; el maximo del dia 2 sigue en 1000)
    """
    fechas = pd.Series(pd.date_range("2025-01-06", periods=48, freq="h", tz=ZONA))
    real = np.full(48, 1000.0)
    pred = real.copy()
    pred[19], pred[24 + 3], pred[24 + 18] = 1100.0, 900.0, 950.0
    return real, pred, fechas


def test_metricas_de_regresion_coinciden_con_el_calculo_a_mano(ejemplo_regresion):
    real, pred, fechas = ejemplo_regresion

    assert mae(real, pred) == pytest.approx(250 / 48)
    assert rmse(real, pred) == pytest.approx(np.sqrt(22500 / 48))
    assert mape(real, pred) == pytest.approx(100 * 0.25 / 48)
    assert sesgo_medio(real, pred) == pytest.approx(-50 / 48)  # negativo: subestima en neto
    # 6 horas pico (18-20 h de cada dia), errores de 10 % y 5 %.
    assert mape_horas_pico(real, pred, fechas) == pytest.approx(100 * 0.15 / 6)
    # Maximos: dia 1 real 1000 / pred 1100, dia 2 real 1000 / pred 1000.
    assert error_maximo_diario(real, pred, fechas) == pytest.approx({"mae_mwh": 50.0, "mape_pct": 5.0, "n_dias": 2})
    assert mejora_mape(4.0, 5.0) == pytest.approx(20.0)
    assert mejora_mape(6.0, 5.0) == pytest.approx(-20.0)

    todas = metricas_regresion(real, pred, fechas, mape_baseline=1.0, alarma_fuga=False)
    assert todas["n"] == 48 and todas["n_pico"] == 6 and todas["n_dias"] == 2
    assert todas["mape_pct"] == pytest.approx(100 * 0.25 / 48)
    # Mejora frente a un baseline de MAPE 1 %: 100 * (1 - 0.5208) / 1.
    assert todas["mejora_mape_pct"] == pytest.approx(100 * (1.0 - 100 * 0.25 / 48) / 1.0)
    assert todas["sesgo_medio_pct"] == pytest.approx(100 * (-50 / 48) / 1000)


def test_un_mape_menor_que_1_pct_dispara_la_alarma_de_fuga(ejemplo_regresion):
    real, pred, fechas = ejemplo_regresion  # MAPE = 0.52 %
    with pytest.raises(AlertaFuga, match="Audita la fuga"):
        metricas_regresion(real, pred, fechas)
    peor = pred.copy()
    peor[:10] = 900.0  # MAPE > 1 %: pasa
    assert metricas_regresion(real, peor, fechas)["mape_pct"] > 1


def test_las_metricas_rechazan_entradas_invalidas(ejemplo_regresion):
    real, pred, fechas = ejemplo_regresion
    with pytest.raises(ErrorMetricas):
        mae(real, pred[:-1])
    with pytest.raises(ErrorMetricas):
        mape(np.r_[real[:-1], 0.0], pred)
    with pytest.raises(ErrorMetricas):
        rmse(np.r_[real[:-1], np.nan], pred)
    with pytest.raises(ErrorMetricas):
        mape_horas_pico(real[:5], pred[:5], fechas[:5])  # 0-4 h: ninguna hora pico


def test_metricas_de_clasificacion_coinciden_con_el_calculo_a_mano():
    real = [0, 0, 1, 1, 1, 0]
    puntaje = [0.1, 0.6, 0.8, 0.4, 0.9, 0.2]
    # Umbral 0.5 -> predichos [0, 1, 1, 0, 1, 0]: vp=2, fp=1, fn=1, vn=2.
    resultado = metricas_clasificacion(real, puntaje)

    assert resultado["matriz_confusion"]["matriz"] == [[2, 1], [1, 2]]
    assert resultado["precision"] == pytest.approx(2 / 3)
    assert resultado["recall"] == pytest.approx(2 / 3)
    assert resultado["f1"] == pytest.approx(2 / 3)
    # ROC-AUC: de los 9 pares (positivo, negativo), 8 tienen el positivo por encima.
    assert resultado["roc_auc"] == pytest.approx(8 / 9)
    # Average precision: positivos en las posiciones 1, 2 y 4 del orden descendente.
    assert resultado["pr_auc"] == pytest.approx((1 + 1 + 3 / 4) / 3)

    estricto = metricas_clasificacion(real, puntaje, umbral=0.85)  # solo 0.9 es positivo
    assert estricto["precision"] == pytest.approx(1.0) and estricto["recall"] == pytest.approx(1 / 3)


def test_clasificacion_con_una_sola_clase_no_inventa_auc():
    resultado = metricas_clasificacion([0, 0, 0], [0.2, 0.7, 0.1])
    assert resultado["roc_auc"] is None and resultado["pr_auc"] is None
    assert resultado["recall"] is None and resultado["precision"] == 0.0 and resultado["f1"] is None
    with pytest.raises(ErrorMetricas):
        metricas_clasificacion([0, 2, 1], [0.1, 0.2, 0.3])


# ==========================================================================
# baselines: lag_168 en las dos tablas, lag_24 solo donde existe
# ==========================================================================


@pytest.mark.parametrize("version", VERSIONES)
def test_los_baselines_predicen_exactamente_su_rezago(carpeta_tablas, version):
    (X, y, fechas), _ = cargar_tabla(version, carpeta_tablas)
    modelos, registro = baselines_disponibles(list(X.columns))

    assert list(modelos)[0] == "estacional" and registro["detalle"]["referencia"] == "estacional"
    assert ("diario" in modelos) == (version == "r24")
    if version == "r120":
        assert "lag_24" in registro["detalle"]["omitidos"]["diario"]

    for modelo in modelos.values():
        modelo.fit(X.iloc[:100], y.iloc[:100])
        np.testing.assert_array_equal(modelo.predict(X.iloc[100:]), X[modelo.columna].iloc[100:].to_numpy(float))


def test_el_baseline_se_comporta_como_un_estimador_de_sklearn(tablas):
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError

    X = tablas["r120"].drop(columns=["fecha_hora", "demanda_mwh"])
    modelo = BaselineRezago("lag_168")
    with pytest.raises(NotFittedError):
        modelo.predict(X)
    assert clone(modelo).get_params() == {"columna": "lag_168"}
    with pytest.raises(ErrorBaseline, match="lag_24"):
        BaselineRezago("lag_24").fit(X)
    with pytest.raises(ErrorBaseline, match="DataFrame"):
        modelo.fit(X.to_numpy())
    with pytest.raises(ErrorBaseline):
        baselines_disponibles(["temperatura_c"])


# ==========================================================================
# tiempos: ajuste, prediccion por 1000 filas y tamano serializado
# ==========================================================================


def test_tiempos_y_tamano_del_modelo_guardado(tablas, tmp_path):
    tabla = tablas["r24"]
    X, y = tabla.drop(columns=["fecha_hora", "demanda_mwh"]).astype(float), tabla["demanda_mwh"]
    modelo = Pipeline([("escalar", StandardScaler()), ("regresion", Ridge(alpha=1.0))])

    modelo, ajuste = medir_ajuste(modelo, X.iloc[:2000], y.iloc[:2000])
    assert ajuste["detalle"]["segundos"] > 0 and ajuste["detalle"]["filas"] == 2000

    prediccion, medida = medir_prediccion(modelo, X.iloc[2000:4500], repeticiones=3)
    np.testing.assert_array_equal(prediccion, modelo.predict(X.iloc[2000:4500]))
    detalle = medida["detalle"]
    assert detalle["ms_por_1000_filas"] == pytest.approx(1000 * detalle["segundos_mediana"] * 1000 / 2500)

    ruta = tmp_path / "modelos" / "ridge_r24.joblib"
    archivo = guardar_modelo(modelo, ruta)
    assert archivo["bytes"] == ruta.stat().st_size == tamano_serializado(modelo)
    leido = cargar_modelo(ruta, archivo["sha256"])
    np.testing.assert_array_equal(leido.predict(X.iloc[:50]), modelo.predict(X.iloc[:50]))
    with pytest.raises(ErrorModelo, match="SHA-256"):
        cargar_modelo(ruta, "0" * 64)
    assert {"python", "numpy", "scikit-learn", "joblib"} <= set(entorno())


# ==========================================================================
# scripts/entrenar.py: de punta a punta, sin tocar el bloque de prueba
# ==========================================================================


def _entrenar(carpeta_datos: Path, salida: Path, version: str) -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location("entrenar", RAIZ / "scripts" / "entrenar.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.main(
        ["--tabla", version, "--familia", "baseline", "--datos", str(carpeta_datos),
         "--resultados", str(salida / "resultados"), "--modelos", str(salida / "models")]
    )


def _baselines_de(carpeta_datos: Path, salida: Path) -> dict:
    for version in VERSIONES:
        assert _entrenar(carpeta_datos, salida, version) == 0
    import json

    return json.loads((salida / "resultados" / "modelado_baselines.json").read_text(encoding="utf-8"))


def test_entrenar_baselines_evalua_en_validacion_y_guarda_las_dos_tablas(carpeta_tablas, tmp_path):
    resultados = _baselines_de(carpeta_tablas, tmp_path)

    assert resultados["familia"] == "baseline" and list(resultados["tablas"]) == ["r24", "r120"]
    for version, entrada in resultados["tablas"].items():
        assert entrada["evaluacion"]["rango_evaluacion"] == [
            str(pd.Timestamp("2024-09-03 00:00", tz=ZONA)), str(pd.Timestamp("2025-09-02 23:00", tz=ZONA))
        ]
        assert entrada["datos"]["sha256"] and entrada["datos"]["tabla"] == version
        assert ("diario" in entrada["modelos"]) == (version == "r24")
        for nombre, modelo in entrada["modelos"].items():
            m = modelo["metricas_validacion"]
            assert m["n"] == entrada["evaluacion"]["filas_evaluacion"] == 8760
            assert m["mape_pct"] > 1  # el ruido del 3 % deja a los baselines lejos del umbral de fuga
            assert (m["mejora_mape_pct"] is None) == (nombre == "estacional")
            archivo = tmp_path / "models" / Path(modelo["modelo"]["ruta"]).name
            assert cargar_modelo(archivo, modelo["modelo"]["sha256"]).columna == modelo["columna"]
    assert "diario" in resultados["tablas"]["r120"]["baselines"]["detalle"]["omitidos"]


def test_alterar_la_demanda_del_bloque_de_prueba_no_cambia_ninguna_metrica(panel, carpeta_tablas, tmp_path):
    alterado = panel.copy()
    en_prueba = alterado["fecha_hora"] >= inicio_prueba()
    alterado.loc[en_prueba, "valor_kwh"] *= 2.0
    carpeta_alterada = guardar_tablas(alterado, tmp_path / "procesado_alterado")

    original = _baselines_de(carpeta_tablas, tmp_path / "original")
    cambiado = _baselines_de(carpeta_alterada, tmp_path / "alterado")
    for version in VERSIONES:
        # Las tablas si difieren (la prueba cambio)...
        assert original["tablas"][version]["datos"]["sha256"] != cambiado["tablas"][version]["datos"]["sha256"]
        # ...pero ninguna metrica de validacion se entera.
        for nombre, modelo in original["tablas"][version]["modelos"].items():
            assert modelo["metricas_validacion"] == cambiado["tablas"][version]["modelos"][nombre]["metricas_validacion"]
