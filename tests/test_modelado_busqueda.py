"""Pruebas de la busqueda de hiperparametros (src/modelado/busqueda.py). Sin red.

Las busquedas reales tardan minutos: aqui se reducen los espacios con
monkeypatch (la logica es la misma) y se usan las tablas sinteticas de
tests/test_modelado_base.py. Los resultados de los pasos anteriores, de los
que salen los candidatos, se fabrican a mano.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelado import busqueda  # noqa: E402
from modelado.busqueda import (  # noqa: E402
    ESPACIOS,
    Candidato,
    ErrorBusqueda,
    Espacio,
    categorica,
    combinaciones,
    comparar_busquedas,
    continua,
    crear_estimador,
    entero,
    evaluar_candidato,
    logaritmica,
    seleccionar_candidatos,
)
from modelado.datos import cargar_tabla  # noqa: E402
from modelado.objetivos import construir_alerta_pico  # noqa: E402
from modelado.particion import particionar  # noqa: E402
from modelado.tiempos import cargar_modelo  # noqa: E402
from test_modelado_base import VERSIONES, _entrenar, guardar_tablas, panel_minimo  # noqa: E402

# Espacios pequenos para que las pruebas tarden segundos.
ESPACIOS_PEQUENOS = {
    ("ensembles", "hist_gradient"): Espacio(
        {"learning_rate": (0.05, 0.2), "max_leaf_nodes": (7, 15)},
        {"learning_rate": logaritmica(0.05, 0.2), "max_leaf_nodes": entero(7, 15)},
    ),
    ("ensembles", "adaboost"): Espacio(
        {"estimator__max_depth": (2, 4), "loss": ("linear", "square")},
        {"estimator__max_depth": entero(2, 4), "loss": categorica("linear", "square")},
    ),
    ("lineal", "ridge"): Espacio({"alpha": (0.1, 10.0, 1000.0)}, {"alpha": logaritmica(0.1, 1000.0)}),
    ("clasificacion", "svm_rbf"): Espacio(
        {"C": (0.3, 3.0), "gamma": (0.03, 0.3)}, {"C": logaritmica(0.3, 3.0), "gamma": logaritmica(0.03, 0.3)}
    ),
}


@pytest.fixture
def espacios_pequenos(monkeypatch):
    for clave, espacio in ESPACIOS_PEQUENOS.items():
        monkeypatch.setitem(busqueda.ESPACIOS, clave, espacio)
    # Los mas lentos de la rejilla real no deben colarse en las pruebas.
    for clave in [c for c in busqueda.ESPACIOS if c not in ESPACIOS_PEQUENOS]:
        monkeypatch.delitem(busqueda.ESPACIOS, clave)


def resultados_previos(claves=("r24", "r120")) -> dict[str, dict]:
    """JSON de lineal, ensembles y clasificacion con solo lo que lee la seleccion."""
    def m(valor, clave="mape_pct"):
        return {"metricas_validacion": {clave: valor}}

    lineal = {"lineal": m(3.0), "polinomial": m(2.0), "ridge": {**m(2.6), "hiperparametros": {"alpha": 10.0}}}
    ensembles = {
        "hist_gradient": {"nivel": m(2.7), "diferenciado": m(2.5)},
        "adaboost": {"nivel": m(2.4), "diferenciado": m(2.9)},
    }
    clasificadores = {
        "logistica": {**m(0.70, "pr_auc"), "hiperparametros": {"C": 0.1}},
        "svm_rbf": {**m(0.77, "pr_auc"), "hiperparametros": {"C": 1.0}},
    }
    return {
        "lineal": {"familia": "lineal", "tablas": {c: {"modelos": lineal} for c in claves}},
        "ensembles": {"familia": "ensembles", "tablas": {c: {"modelos": ensembles} for c in claves}},
        "clasificacion": {"familia": "clasificacion", "tablas": {c: {"modelos": clasificadores} for c in claves}},
    }


# ==========================================================================
# Seleccion y espacios
# ==========================================================================


def test_la_seleccion_toma_los_tres_mejores_por_mape_y_el_mejor_clasificador(espacios_pequenos):
    candidatos, registro = seleccionar_candidatos(resultados_previos(), "r24")

    # polinomial (2.0) es el mejor pero no tiene hiperparametros: queda omitido.
    assert [c.etiqueta for c in candidatos] == [
        "adaboost (nivel)", "hist_gradient (diferenciado)", "ridge", "svm_rbf"
    ]
    assert "polinomial" in registro["detalle"]["omitidos"]
    assert [f["mape_validacion_pct"] for f in registro["detalle"]["ranking_regresion"]] == sorted(
        f["mape_validacion_pct"] for f in registro["detalle"]["ranking_regresion"]
    )
    assert candidatos[2].hiperparametros_previos == {"alpha": 10.0}
    assert candidatos[3].tipo == "clasificacion" and candidatos[3].criterio_seleccion == "pr_auc_validacion"
    with pytest.raises(ErrorBusqueda, match="Faltan"):
        seleccionar_candidatos(resultados_previos(claves=("r24",)), "r120")


@pytest.mark.parametrize("clave", list(ESPACIOS))
def test_las_distribuciones_cubren_el_mismo_rango_que_la_rejilla(clave):
    espacio = ESPACIOS[clave]
    rng = np.random.RandomState(42)
    for parametro, valores in espacio.rejilla.items():
        distribucion = espacio.distribuciones[parametro].crear()
        if isinstance(distribucion, list):
            assert set(distribucion) == set(valores), parametro
            continue
        muestras = distribucion.rvs(size=2000, random_state=rng)
        assert muestras.min() >= min(valores) - 1e-12 and muestras.max() <= max(valores) + 1e-12, parametro
        # Y la distribucion llega cerca de los dos extremos de la rejilla.
        ancho = max(valores) - min(valores)
        assert muestras.min() - min(valores) < 0.1 * ancho and max(valores) - muestras.max() < 0.1 * ancho
    assert espacio.combinaciones == len(combinaciones(espacio.rejilla))


@pytest.mark.parametrize("clave", list(ESPACIOS))
def test_cada_espacio_se_aplica_al_estimador_de_su_candidato(clave):
    familia, nombre = clave
    formas = ("nivel", "diferenciado") if familia == "ensembles" else ("-",)
    for forma in formas:
        candidato = Candidato(familia, nombre, forma, "x", 0.0, {})
        estimador = crear_estimador(candidato)
        primero = combinaciones(ESPACIOS[clave].rejilla)[0]
        estimador.set_params(**{busqueda.prefijo(candidato) + k: v for k, v in primero.items()})
        if nombre == "random_forest":  # la busqueda ya reparte en procesos
            interno = estimador.estimador if forma == "diferenciado" else estimador
            assert interno.n_jobs == 1


def test_espacio_y_distribuciones_tienen_que_coincidir():
    with pytest.raises(ErrorBusqueda):
        Espacio({"a": (1, 2)}, {"b": continua(1, 2)})


def test_comparar_busquedas_cuenta_los_ajustes_hasta_el_mejor_de_grid():
    def registro(puntajes, n_splits=5):
        acumulado = np.maximum.accumulate(puntajes)
        return {
            "mejor_puntaje_cv": float(max(puntajes)),
            "n_ajustes_cv": len(puntajes) * n_splits,
            "segundos_busqueda": 1.0,
            "trayectoria": [
                {"ajustes_cv": (i + 1) * n_splits, "mejor_puntaje_cv_acumulado": float(a)}
                for i, a in enumerate(acumulado)
            ],
        }

    grid = registro([-300.0, -250.0, -200.0, -260.0])
    random = registro([-280.0, -200.5, -210.0, -205.0])
    comparacion = comparar_busquedas(grid, random, tolerancia=0.005)  # umbral -201
    assert comparacion["ajustes_cv_grid_hasta_su_mejor"] == 15
    assert comparacion["ajustes_cv_random_hasta_alcanzar"] == 10 and comparacion["random_alcanza_grid"]
    assert comparacion["diferencia_mejor_puntaje_cv"] == pytest.approx(-0.5)
    peor = comparar_busquedas(grid, registro([-280.0, -260.0]), tolerancia=0.005)
    assert not peor["random_alcanza_grid"] and peor["ajustes_cv_random_hasta_alcanzar"] is None


# ==========================================================================
# Busquedas sobre las tablas sinteticas
# ==========================================================================


@pytest.fixture(scope="module")
def carpeta_tablas(tmp_path_factory) -> Path:
    return guardar_tablas(panel_minimo("2024-06-01", "2026-09-07 23:00"), tmp_path_factory.mktemp("procesado"))


@pytest.fixture(scope="module")
def entradas(carpeta_tablas):
    resultado = {}
    for version in VERSIONES:
        (X, y, fechas), _ = cargar_tabla(version, carpeta_tablas)
        indices, _ = particionar(fechas)
        marco, _ = construir_alerta_pico(fechas, y, version, indices)
        resultado[version] = (X.astype("float64"), y, marco["alerta_pico"].to_numpy(), fechas, indices)
    return resultado


CANDIDATOS = [
    Candidato("ensembles", "hist_gradient", "diferenciado", "mape_validacion_pct", 2.5, {}),
    Candidato("lineal", "ridge", "-", "mape_validacion_pct", 2.6, {"alpha": 10.0}),
    Candidato("clasificacion", "svm_rbf", "-", "pr_auc_validacion", 0.77, {"C": 1.0}),
]


@pytest.mark.parametrize("candidato", CANDIDATOS, ids=lambda c: c.clave)
def test_grid_y_random_recorren_su_espacio_y_reentrenan_con_el_entrenamiento(espacios_pequenos, entradas, candidato):
    X, y, objetivo, fechas, indices = entradas["r24"]
    modelos, registro = evaluar_candidato(candidato, X, y, objetivo, fechas, indices, "r24")
    espacio = busqueda.ESPACIOS[(candidato.familia, candidato.nombre)]
    n_splits = 5

    assert set(modelos) == {"referencia", "grid", "random"}
    grid, random = registro["grid"], registro["random"]
    assert [c["hiperparametros"] for c in grid["candidatos"]] == combinaciones(espacio.rejilla)
    assert random["n_candidatos"] == grid["n_candidatos"] == espacio.combinaciones == random["n_iter"]
    for r in (grid, random):
        assert r["n_ajustes_cv"] == espacio.combinaciones * n_splits and r["n_ajustes"] == r["n_ajustes_cv"] + 1
        assert r["mejor_puntaje_cv"] == max(c["puntaje_cv"] for c in r["candidatos"])
        acumulado = [p["mejor_puntaje_cv_acumulado"] for p in r["trayectoria"]]
        assert acumulado == sorted(acumulado) and acumulado[-1] == r["mejor_puntaje_cv"]
        assert r["segundos_total"] >= r["segundos_busqueda"] > 0
        for parametro, valor in r["mejores"].items():
            valores = espacio.rejilla[parametro]
            if isinstance(valor, str):
                assert valor in valores
            else:
                assert min(valores) <= valor <= max(valores)
    # La mejora en validacion es la diferencia con la referencia.
    if candidato.tipo == "regresion":
        assert grid["scoring"] == "neg_mean_absolute_error"
        ref, buscado = registro["referencia"]["metricas_validacion"], grid["metricas_validacion"]
        assert grid["mejora_validacion_frente_a_referencia"]["mae_mwh"] == pytest.approx(ref["mae_mwh"] - buscado["mae_mwh"])
        assert buscado["n"] == 8760
    else:
        assert grid["scoring"] == "f1" and registro["submuestra"]["paso_h"] == 5
        assert registro["validacion_cruzada"]["detalle"]["gap_filas"] == 5  # ceil(24 / 5)
    json.dumps(registro)


def test_alterar_la_validacion_no_cambia_ninguna_busqueda(espacios_pequenos, entradas):
    """Buscar y reentrenar solo ven el entrenamiento; Random con la misma semilla repite sus candidatos."""
    X, y, objetivo, fechas, indices = entradas["r120"]
    valida = indices["validacion"]
    X_alt, y_alt, objetivo_alt = X.copy(), y.copy(), objetivo.copy()
    X_alt.iloc[valida] = X_alt.iloc[valida] * 10 + 1000
    y_alt.iloc[valida] = y_alt.iloc[valida] * 3
    objetivo_alt[valida] = 1 - objetivo_alt[valida]

    for candidato in (CANDIDATOS[0], CANDIDATOS[2]):
        modelos, registro = evaluar_candidato(candidato, X, y, objetivo, fechas, indices, "r120")
        otros, registro_otro = evaluar_candidato(candidato, X_alt, y_alt, objetivo_alt, fechas, indices, "r120")
        entrena = indices["entrenamiento"]
        for metodo in ("grid", "random"):
            assert registro[metodo]["mejores"] == registro_otro[metodo]["mejores"]
            assert registro[metodo]["candidatos"] == registro_otro[metodo]["candidatos"]
            np.testing.assert_array_equal(
                modelos[metodo].predict(X.iloc[entrena[-500:]]), otros[metodo].predict(X.iloc[entrena[-500:]])
            )
            assert registro[metodo]["metricas_validacion"] != registro_otro[metodo]["metricas_validacion"]
        assert registro["referencia"]["puntaje_cv"] == registro_otro["referencia"]["puntaje_cv"]


# ==========================================================================
# scripts/entrenar.py --familia busqueda
# ==========================================================================


def test_entrenar_familia_busqueda_de_punta_a_punta(espacios_pequenos, carpeta_tablas, tmp_path):
    carpeta = tmp_path / "resultados"
    carpeta.mkdir()
    for familia, contenido in resultados_previos().items():
        archivo = {"lineal": "modelado_regresion.json", "ensembles": "modelado_ensembles.json",
                   "clasificacion": "modelado_clasificacion.json"}[familia]
        (carpeta / archivo).write_text(json.dumps(contenido), encoding="utf-8")

    for version in VERSIONES:
        assert _entrenar(carpeta_tablas, tmp_path, version, familia="busqueda") == 0
    resultados = json.loads((carpeta / "modelado_busqueda.json").read_text(encoding="utf-8"))

    assert resultados["familia"] == "busqueda" and list(resultados["tablas"]) == ["r24", "r120"]
    for version, entrada in resultados["tablas"].items():
        assert list(entrada["modelos"]) == ["adaboost (nivel)", "hist_gradient (diferenciado)", "ridge", "svm_rbf"]
        assert (tmp_path / "figuras" / f"busqueda_puntaje_tiempo_{version}.png").exists()
        for registro in entrada["modelos"].values():
            rezago = 24 if version == "r24" else 120
            gap_h = registro["validacion_cruzada"]["detalle"]["gap_h"]
            # La SVC RBF va submuestrada: gap = ceil(rezago / 5) filas de 5 h, >= rezago.
            assert gap_h == rezago if registro["tipo"] == "regresion" else rezago <= gap_h < rezago + 5
            for metodo in ("grid", "random"):
                archivo = tmp_path / "models" / Path(registro[metodo]["modelo"]["ruta"]).name
                assert cargar_modelo(archivo, registro[metodo]["modelo"]["sha256"]) is not None


def test_entrenar_busqueda_sin_resultados_previos_falla(carpeta_tablas, tmp_path):
    with pytest.raises(FileNotFoundError, match="corre antes"):
        _entrenar(carpeta_tablas, tmp_path, "r24", familia="busqueda")
