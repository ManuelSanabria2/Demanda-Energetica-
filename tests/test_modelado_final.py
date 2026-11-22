"""Pruebas de la evaluacion final (src/modelado/final.py, scripts/evaluar_final.py, scripts/predecir.py).

Sin red, con las tablas sinteticas de tests/test_modelado_base.py y resultados
de los pasos anteriores fabricados a mano. Lo que importa: el ajuste final usa
entrenamiento + validacion y nunca la prueba; la eleccion sale solo de la
validacion; un MAPE < 1 % detiene todo antes de escribir; y predecir.py da lo
mismo que el modelo evaluado.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado import final  # noqa: E402
from modelado.datos import cargar_tabla  # noqa: E402
from modelado.ensembles import RegresorDiferenciado  # noqa: E402
from modelado.interpretabilidad import CALIFICACIONES  # noqa: E402
from modelado.metricas import AlertaFuga  # noqa: E402
from modelado.objetivos import construir_alerta_pico  # noqa: E402
from modelado.particion import particionar  # noqa: E402
from test_modelado_base import VERSIONES, guardar_tablas, panel_minimo  # noqa: E402


def _script(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _m(mape_pct, pico=2.5, **extra):
    return {"mape_pct": mape_pct, "mape_pico_pct": pico, "mejora_mape_pct": None, "mae_mwh": 1.0,
            "rmse_mwh": 1.0, "sesgo_medio_mwh": 0.0, **extra}


def previos(claves=tuple(VERSIONES), busqueda=True) -> dict:
    """Los JSON de los pasos anteriores, con solo lo que lee la evaluacion final."""
    def tabla(modelos):
        return {"tablas": {c: {"modelos": modelos} for c in claves}}

    lineal = {
        "lineal": {"metricas_validacion": _m(3.0), "hiperparametros": {}, "condicionamiento": {"rango": 16, "columnas": 16}},
        "polinomial": {"metricas_validacion": _m(2.0), "hiperparametros": {},
                       "condicionamiento": {"rango": 138, "columnas": 152}},
        "polinomial_ridge": {"metricas_validacion": _m(2.4), "hiperparametros": {"alpha": 1000.0},
                             "condicionamiento": {"rango": 138, "columnas": 152}},
    }
    resultado = {
        "baseline": tabla({"estacional": {"metricas_validacion": _m(3.35, 2.9)}}),
        "lineal": tabla(lineal),
        "ensembles": tabla({"hist_gradient": {"nivel": {"metricas_validacion": _m(2.0)},
                                              "diferenciado": {"metricas_validacion": _m(2.2)}}}),
        "clasificacion": tabla({
            "svm_rbf": {"metricas_validacion": {"pr_auc": 0.70, "f1": 0.6}, "hiperparametros": {"C": 1.0}},
            "logistica": {"metricas_validacion": {"pr_auc": 0.65, "f1": 0.6}, "hiperparametros": {"C": 0.1}},
        }),
        "interpretabilidad": {"tablas": {c: {"calificaciones": CALIFICACIONES} for c in claves}},
    }
    if busqueda:
        resultado["busqueda"] = {"tablas": {c: {"modelos": {
            "hist_gradient (diferenciado)": {
                "familia": "ensembles", "nombre": "hist_gradient", "forma": "diferenciado",
                "grid": {"mejores": {"learning_rate": 0.1, "max_leaf_nodes": 15}, "metricas_validacion": _m(1.8, 2.0)},
                "random": {"mejores": {"learning_rate": 0.07, "max_leaf_nodes": 20}, "metricas_validacion": _m(1.9)},
            },
            "svm_rbf": {
                "familia": "clasificacion", "nombre": "svm_rbf", "forma": "-",
                "grid": {"mejores": {"C": 3.0, "gamma": 0.03}, "metricas_validacion": {"pr_auc": 0.75, "f1": 0.6}},
                "random": {"mejores": {"C": 2.0, "gamma": 0.05}, "metricas_validacion": {"pr_auc": 0.72, "f1": 0.6}},
            },
        }} for c in claves}}
    return resultado


def escribir_previos(carpeta: Path, contenido: dict) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    for familia, archivo in _script("evaluar_final").ENTRADAS.items():
        if familia in contenido:
            (carpeta / archivo).write_text(json.dumps(contenido[familia]), encoding="utf-8")
    return carpeta


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return panel_minimo("2024-06-01", "2026-09-07 23:00")


@pytest.fixture(scope="module")
def carpeta_tablas(panel, tmp_path_factory) -> Path:
    return guardar_tablas(panel, tmp_path_factory.mktemp("procesado"))


@pytest.fixture(scope="module")
def datos(carpeta_tablas):
    (X, y, fechas), _ = cargar_tabla("r24", carpeta_tablas)
    indices, _ = particionar(fechas)
    marco, _ = construir_alerta_pico(fechas, y, "r24", indices)
    return X.astype("float64"), y, fechas, indices, marco["alerta_pico"].to_numpy()


# ==========================================================================
# Candidatos y estimadores
# ==========================================================================


def test_los_candidatos_salen_de_la_validacion_y_excluyen_la_polinomial_sin_regularizar():
    candidatos, registro = final.seleccionar_candidatos(previos(), "r24")
    assert candidatos["baseline"].nombre == "estacional"
    assert candidatos["lineal"].nombre == "polinomial_ridge" and candidatos["lineal"].hiperparametros == {"alpha": 1000.0}
    assert "polinomial" in registro["detalle"]["lineales_excluidos"]
    assert candidatos["ensemble"].forma == "diferenciado" and "grid" in candidatos["ensemble"].origen
    assert candidatos["clasificador"].hiperparametros == {"C": 3.0, "gamma": 0.03}

    sin_busqueda, _ = final.seleccionar_candidatos(previos(busqueda=False), "r24")
    assert sin_busqueda["ensemble"].forma == "nivel" and sin_busqueda["clasificador"].hiperparametros == {"C": 1.0}
    with pytest.raises(final.ErrorFinal, match="Faltan"):
        final.seleccionar_candidatos(previos(claves=("r24",)), "r120")


def test_el_estimador_lleva_los_hiperparametros_elegidos():
    candidatos, _ = final.seleccionar_candidatos(previos(), "r24")
    ensemble = final.crear_estimador(candidatos["ensemble"])
    assert isinstance(ensemble, RegresorDiferenciado)
    assert ensemble.estimador.max_leaf_nodes == 15 and ensemble.estimador.learning_rate == 0.1
    assert final.crear_estimador(candidatos["lineal"]).named_steps["modelo"].alpha == 1000.0
    assert final.crear_estimador(candidatos["clasificador"]).named_steps["modelo"].gamma == 0.03
    assert final.crear_estimador(candidatos["baseline"]).columna == "lag_168"


def test_el_ajuste_final_usa_entrenamiento_y_validacion_y_nunca_la_prueba(datos):
    X, y, fechas, indices, objetivo = datos
    filas = final.filas_de_ajuste(indices)
    assert len(filas) == len(indices["entrenamiento"]) + len(indices["validacion"])
    assert filas.max() < indices["prueba"].min()

    candidatos, _ = final.seleccionar_candidatos(previos(), "r24")
    X_alt, y_alt, objetivo_alt = X.copy(), y.copy(), objetivo.copy()
    prueba = indices["prueba"]
    X_alt.iloc[prueba] = X_alt.iloc[prueba] * 7 + 100
    y_alt.iloc[prueba] = y_alt.iloc[prueba] * 5
    objetivo_alt[prueba] = 1 - objetivo_alt[prueba]
    for rol in ("lineal", "ensemble", "clasificador"):
        modelo, ajuste = final.reentrenar(candidatos[rol], X, y, objetivo, fechas, indices)
        otro, _ = final.reentrenar(candidatos[rol], X_alt, y_alt, objetivo_alt, fechas, indices)
        assert pd.Timestamp(ajuste["hasta"]) < fechas.iloc[prueba[0]]
        muestra = X.iloc[filas[-300:]]
        if rol == "clasificador":
            np.testing.assert_array_equal(modelo.decision_function(muestra), otro.decision_function(muestra))
            assert ajuste["submuestra_paso_h"] == 5
        else:
            np.testing.assert_array_equal(modelo.predict(muestra), otro.predict(muestra))
            assert ajuste["filas"] == len(filas)


# ==========================================================================
# Regla de eleccion
# ==========================================================================


def _candidatos(lineal, ensemble, base=(3.35, 2.9)):
    def c(rol, nombre, metricas):
        return final.Candidato(rol, rol, nombre, "-", "x", {}, {"mape_pct": metricas[0], "mape_pico_pct": metricas[1]})
    return {"baseline": c("baseline", "estacional", base), "lineal": c("lineal", "ridge", lineal),
            "ensemble": c("ensemble", "hist_gradient", ensemble)}


@pytest.mark.parametrize("lineal, ensemble, esperado, nota", [
    ((2.5, 2.5), (2.0, 2.4), "ensemble", None),                  # los dos cumplen: gana el menor MAPE
    ((2.03, 2.5), (2.0, 2.4), "lineal", "empate"),               # empate < 0,05 puntos: el mas interpretable
    ((2.5, 3.5), (2.0, 3.0), "ensemble", "ningun candidato cumple"),  # mejoran pero pierden en pico
    ((3.5, 2.0), (3.4, 2.0), "baseline", "el baseline"),         # ninguno mejora al baseline
    ((2.9, 2.5), (2.0, 2.4), "ensemble", None),                  # el lineal no llega al 20 %
])
def test_la_regla_de_eleccion(lineal, ensemble, esperado, nota):
    rol, registro = final.elegir_final(_candidatos(lineal, ensemble), CALIFICACIONES,
                                       {"lineal": 1.0, "ensemble": 5.0})
    assert rol == esperado
    assert (registro["nota"] is None) if nota is None else (nota in registro["nota"])
    assert "validacion" in registro["bloque"]


def test_la_eleccion_no_depende_del_bloque_de_prueba(carpeta_tablas, panel, tmp_path):
    """Con la prueba alterada por completo, el modelo final elegido y sus razones son los mismos."""
    evaluar_final = _script("evaluar_final")
    alterado = panel.copy()
    en_prueba = alterado["fecha_hora"] >= pd.Timestamp("2025-09-03", tz="America/Bogota")
    alterado.loc[en_prueba, "valor_kwh"] *= 3.0
    alterado.loc[en_prueba, "temperatura_c"] += 15.0
    carpeta_alterada = guardar_tablas(alterado, tmp_path / "alterado")
    contenido = previos(claves=("r24",))
    original = evaluar_final.evaluar_tabla("r24", contenido, carpeta_tablas)
    otro = evaluar_final.evaluar_tabla("r24", contenido, carpeta_alterada)

    assert original["rol_final"] == otro["rol_final"]
    for rol, evaluado in original["eleccion"]["evaluados"].items():
        assert evaluado["criterio_validacion"] == otro["eleccion"]["evaluados"][rol]["criterio_validacion"]
    assert "validacion" in original["eleccion"]["desempate_tiempo"]
    # Control: la prueba si cambio, y sus metricas lo notan.
    assert original["evaluacion"]["metricas"]["baseline"] != otro["evaluacion"]["metricas"]["baseline"]


def test_cumple_criterio_a_mano():
    c = final.cumple_criterio({"mape_pct": 2.68, "mape_pico_pct": 2.0}, {"mape_pct": 3.35, "mape_pico_pct": 2.9})
    assert c["mejora_mape_pct"] == pytest.approx(20.0) and c["mejora_suficiente"] and c["cumple"]
    assert not final.cumple_criterio({"mape_pct": 2.0, "mape_pico_pct": 3.0},
                                     {"mape_pct": 3.35, "mape_pico_pct": 2.9})["cumple"]


# ==========================================================================
# Residuales
# ==========================================================================


def test_residuales_por_grupo_a_mano():
    fechas = pd.Series(pd.to_datetime(["2026-03-30 10:00", "2026-03-30 11:00", "2026-04-04 10:00",
                                       "2026-05-01 10:00"]).tz_localize("America/Bogota"))
    X = pd.DataFrame({"es_festivo": [0, 0, 0, 1], "es_semana_santa": [1, 1, 0, 0]})
    y = pd.Series([100.0, 200.0, 100.0, 50.0])
    pred = np.array([110.0, 180.0, 100.0, 60.0])
    base = np.array([100.0, 100.0, 50.0, 50.0])
    assert list(final.tipo_de_dia(fechas, X)) == ["semana_santa", "semana_santa", "fin_de_semana", "festivo"]
    r = final.residuales(y, pred, base, fechas, X)
    santa = r["tipo_dia"]["semana_santa"]
    assert santa["n"] == 2 and santa["sesgo_mwh"] == pytest.approx((10 - 20) / 2)
    assert santa["mape_pct"] == pytest.approx(100 * (0.1 + 0.1) / 2)
    assert santa["mape_baseline_pct"] == pytest.approx(100 * (0 + 0.5) / 2)
    assert r["hora"]["10"]["n"] == 3 and r["dia_semana"]["lunes"]["n"] == 2
    assert "laboral" not in r["tipo_dia"]


# ==========================================================================
# De punta a punta
# ==========================================================================


def _rutas(tmp: Path, previos_dir: Path) -> list[str]:
    return ["--datos", str(tmp / "procesado"), "--resultados", str(previos_dir), "--modelos", str(tmp / "models"),
            "--figuras", str(tmp / "figuras"), "--documento", str(tmp / "doc" / "seleccion.md")]


def test_un_mape_menor_que_1_pct_detiene_todo_antes_de_escribir(panel, tmp_path):
    perfecto = panel.copy()
    horas = pd.DatetimeIndex(perfecto["fecha_hora"])
    # Demanda que se repite cada semana: lag_168 la reproduce exactamente.
    perfecto["valor_kwh"] = 9.0e6 * (1 + 0.15 * np.sin(2 * np.pi * horas.hour / 24) - 0.08 * (horas.dayofweek >= 5))
    guardar_tablas(perfecto, tmp_path / "procesado")
    carpeta = escribir_previos(tmp_path / "resultados", previos(busqueda=False))
    antes = set(carpeta.iterdir())

    with pytest.raises(AlertaFuga):
        _script("evaluar_final").main(_rutas(tmp_path, carpeta) + ["--tablas", "r24"])
    assert set(carpeta.iterdir()) == antes
    assert not (tmp_path / "models").exists() and not (tmp_path / "figuras").exists()
    assert not (tmp_path / "doc").exists()


@pytest.fixture(scope="module")
def evaluacion(carpeta_tablas, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("final")
    (tmp / "procesado").mkdir()
    for archivo in carpeta_tablas.iterdir():
        (tmp / "procesado" / archivo.name).write_bytes(archivo.read_bytes())
    carpeta = escribir_previos(tmp / "resultados", previos())
    assert _script("evaluar_final").main(_rutas(tmp, carpeta)) == 0
    return tmp, carpeta


def test_evaluar_final_escribe_la_tabla_de_seleccion_y_sus_archivos(evaluacion):
    tmp, carpeta = evaluacion
    seleccion = json.loads((carpeta / "modelado_seleccion.json").read_text(encoding="utf-8"))
    assert list(seleccion["tablas"]) == ["r24", "r120"] and seleccion["bloque_evaluacion"] == "prueba"
    for clave, t in seleccion["tablas"].items():
        roles = [fila["rol"] for fila in t["tabla_seleccion"]]
        assert roles == ["baseline", "lineal", "ensemble", "clasificador"]
        assert sum(fila["es_final"] for fila in t["tabla_seleccion"]) == 1
        for fila in t["tabla_seleccion"]:
            assert set(fila) >= {"rendimiento", "interpretabilidad", "eficiencia"}
            assert fila["eficiencia"]["tamano_bytes"] > 0 and fila["interpretabilidad"]["calificacion"]
        m = t["metricas_prueba"]
        assert m["baseline"]["n"] == 8760 and m["baseline"]["mejora_mape_pct"] is None
        referencia = m["baseline"]["mape_pct"]
        for rol in ("lineal", "ensemble"):
            assert m[rol]["mejora_mape_pct"] == pytest.approx(100 * (referencia - m[rol]["mape_pct"]) / referencia)
        assert t["final"]["modelo"]["ruta"].endswith(f"final_{clave}.joblib")
        assert (tmp / "figuras" / f"residuales_{clave}.png").exists()
    assert set(seleccion["costo_rezago"]) == {"baseline", "lineal", "ensemble", "final"}
    for figura in ("semana_prueba.png", "mape_prueba.png"):
        assert (tmp / "figuras" / figura).exists()
    texto = (tmp / "doc" / "seleccion.md").read_text(encoding="utf-8")
    assert "Criterio de éxito en prueba" in texto and "r24 → r120" in texto


def test_las_predicciones_y_probabilidades_cubren_la_prueba(evaluacion):
    _, carpeta = evaluacion
    pred = pd.read_parquet(carpeta / "predicciones_prueba.parquet")
    assert len(pred) == 2 * 3 * 8760
    assert set(pred["rol"]) == {"baseline", "lineal", "ensemble"} and set(pred["tabla"]) == {"r24", "r120"}
    assert pred["fecha_hora"].min() == pd.Timestamp("2025-09-03 00:00", tz="America/Bogota")
    assert pred["fecha_hora"].max() == pd.Timestamp("2026-09-02 23:00", tz="America/Bogota")
    assert pred.groupby("tabla")["es_final"].sum().eq(8760).all()
    prob = pd.read_parquet(carpeta / "probabilidades_alerta.parquet")
    assert len(prob) == 2 * 8760 and prob["probabilidad"].between(0, 1).all()
    assert ((prob["puntaje"] >= 0).astype(int) == prob["alerta_predicha"]).all()


def test_predecir_da_lo_mismo_que_el_modelo_evaluado(evaluacion, tmp_path):
    tmp, carpeta = evaluacion
    predecir = _script("predecir")
    for clave in VERSIONES:
        salida = tmp_path / f"pred_{clave}.parquet"
        assert predecir.main(["--tabla", clave, "--entrada", str(tmp / "procesado" / f"tabla_modelado_{clave}.parquet"),
                              "--salida", str(salida), "--seleccion", str(carpeta / "modelado_seleccion.json"),
                              "--modelos", str(tmp / "models"), "--desde", "2025-09-03", "--hasta", "2026-09-02"]) == 0
        nuevas = pd.read_parquet(salida)
        evaluadas = pd.read_parquet(carpeta / "predicciones_prueba.parquet")
        evaluadas = evaluadas[(evaluadas["tabla"] == clave) & evaluadas["es_final"]]
        assert len(nuevas) == 8760
        np.testing.assert_allclose(nuevas["prediccion_mwh"].to_numpy(), evaluadas["prediccion_mwh"].to_numpy())


def test_predecir_rechaza_la_tabla_de_la_otra_version_y_un_modelo_alterado(evaluacion):
    tmp, carpeta = evaluacion
    predecir = _script("predecir")
    modelo, _ = predecir.modelo_final("r120", carpeta / "modelado_seleccion.json", tmp / "models")
    tabla_r24 = pd.read_parquet(tmp / "procesado" / "tabla_modelado_r24.parquet")
    with pytest.raises(predecir.ErrorPrediccion, match="otra version"):
        predecir.predecir(modelo, tabla_r24)
    # Sin demanda tambien predice: es lo que se quiere estimar.
    tabla_r120 = pd.read_parquet(tmp / "procesado" / "tabla_modelado_r120.parquet").drop(columns="demanda_mwh")
    assert "demanda_mwh" not in predecir.predecir(modelo, tabla_r120).columns

    seleccion = json.loads((carpeta / "modelado_seleccion.json").read_text(encoding="utf-8"))
    archivo = tmp / "models" / Path(seleccion["tablas"]["r24"]["final"]["modelo"]["ruta"]).name
    copia = tmp / "models_alterados"
    copia.mkdir(exist_ok=True)
    (copia / archivo.name).write_bytes(archivo.read_bytes() + b"x")
    with pytest.raises(Exception, match="SHA-256"):
        predecir.modelo_final("r24", carpeta / "modelado_seleccion.json", copia)


def test_la_prueba_no_se_evalua_dos_veces_sin_forzar(evaluacion):
    tmp, carpeta = evaluacion
    with pytest.raises(SystemExit, match="una vez"):
        _script("evaluar_final").main(_rutas(tmp, carpeta))
