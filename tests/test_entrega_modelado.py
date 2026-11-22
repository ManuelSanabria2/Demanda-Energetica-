"""Pruebas de la integracion del modelado en la entrega. Sin red.

- notebooks/05 reutiliza src/modelado: no importa sklearn ni define modelos.
- scripts/generar_entrega.py encadena las familias en orden y no repite la
  evaluacion en prueba sin --reevaluar-prueba.
- docs/informe/valores_modelado.py: nombres de macro validos en LaTeX, \\pendiente
  si falta un resultado y cifras que salen de los JSON.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import nbformat
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "docs" / "informe"))

from valores_modelado import SUFIJOS, macros_modelado, tabla_seleccion  # noqa: E402

NOTEBOOK = RAIZ / "notebooks" / "05_modelado_supervisado.ipynb"


def _script(ruta: Path, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _num(valor, decimales: int = 0) -> str:
    return r"\pendiente" if valor is None else f"{valor:.{decimales}f}"


# ==========================================================================
# Notebook 05
# ==========================================================================


def _codigo_notebook() -> list[str]:
    generador = _script(RAIZ / "scripts" / "generar_notebooks.py", "generar_notebooks")
    _, libro = generador.NOTEBOOKS["05_modelado_supervisado"]
    return [celda.source for celda in libro.cells if celda.cell_type == "code"]


def test_el_notebook_05_reutiliza_src_modelado_sin_reimplementarlo():
    codigo = _codigo_notebook()
    importados = set()
    for fuente in codigo:
        for nodo in ast.walk(ast.parse(fuente)):
            if isinstance(nodo, ast.ImportFrom) and nodo.module:
                importados.add(nodo.module)
            elif isinstance(nodo, ast.Import):
                importados |= {alias.name for alias in nodo.names}
    assert not any(m.split(".")[0] == "sklearn" for m in importados), "el notebook no ajusta modelos propios"
    usados = {m for m in importados if m.startswith("modelado.")}
    assert {"modelado.datos", "modelado.particion", "modelado.baselines", "modelado.metricas",
            "modelado.regresion", "modelado.objetivos", "modelado.tiempos"} <= usados
    # Las unicas funciones son auxiliares de lectura y de figuras, no de modelado.
    definidas = {n.name for f in codigo for n in ast.walk(ast.parse(f)) if isinstance(n, ast.FunctionDef)}
    assert definidas <= {"resultado", "figura"}
    texto = "\n".join(codigo)
    for archivo in ("modelado_regresion.json", "modelado_clasificacion.json", "modelado_ensembles.json",
                    "modelado_busqueda.json", "modelado_interpretabilidad.json", "modelado_seleccion.json"):
        assert archivo in texto


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook sin generar")
def test_el_notebook_05_esta_publicado_ejecutado_y_sin_errores():
    libro = nbformat.read(NOTEBOOK, as_version=4)
    celdas = [c for c in libro.cells if c.cell_type == "code"]
    assert [c.source for c in celdas] == _codigo_notebook(), "regenera con scripts/generar_notebooks.py"
    assert all(c.execution_count for c in celdas), "publicalo ejecutado"
    assert not any(o.output_type == "error" for c in celdas for o in c.outputs)


# ==========================================================================
# generar_entrega.py
# ==========================================================================


@pytest.fixture
def entrega(monkeypatch, tmp_path):
    modulo = _script(RAIZ / "scripts" / "generar_entrega.py", "generar_entrega")
    llamadas = []
    monkeypatch.setattr(modulo, "paso", lambda nombre, comando, cwd=None: llamadas.append(comando[1:]))
    monkeypatch.setattr(modulo, "SELECCION", tmp_path / "modelado_seleccion.json")
    return modulo, llamadas


def test_generar_entrega_entrena_cada_familia_en_r24_y_r120_y_evalua_al_final(entrega):
    modulo, llamadas = entrega
    modulo.modelado()
    familias = [(c[c.index("--familia") + 1], c[c.index("--tabla") + 1]) for c in llamadas if "--familia" in c]
    assert familias == [(f, t) for f in modulo.FAMILIAS for t in ("r24", "r120")]
    assert modulo.FAMILIAS.index("busqueda") > modulo.FAMILIAS.index("ensembles")
    assert modulo.FAMILIAS.index("interpretabilidad") > modulo.FAMILIAS.index("busqueda")
    assert llamadas[-1] == ["scripts/evaluar_final.py"]


def test_generar_entrega_no_repite_la_prueba_sin_pedirlo(entrega):
    modulo, llamadas = entrega
    modulo.SELECCION.write_text("{}", encoding="utf-8")
    modulo.modelado(sin_busqueda=True)
    assert not any("evaluar_final" in " ".join(c) for c in llamadas)
    assert not any("busqueda" in c for c in llamadas)
    modulo.modelado(reevaluar_prueba=True)
    assert llamadas[-1] == ["scripts/evaluar_final.py", "--forzar"]


# ==========================================================================
# Macros del informe
# ==========================================================================


def test_sin_resultados_todas_las_macros_quedan_pendientes(tmp_path):
    macros, tablas = macros_modelado(tmp_path, _num)
    assert all(re.fullmatch(r"[A-Za-z]+", nombre) for nombre in macros), "LaTeX no admite digitos en macros"
    assert set(macros.values()) == {r"\pendiente"}
    assert r"\pendiente" in tablas["tabla_seleccion.tex"]
    for sufijo in SUFIJOS.values():
        assert f"modFinalMape{sufijo}" in macros


def test_las_macros_salen_de_los_json(tmp_path):
    seleccion = {
        "bloque_prueba": ["2025-09-03 00:00:00-05:00", "2026-09-02 23:00:00-05:00"],
        "criterio_exito": {"mejora_mape_minima_pct": 20.0},
        "costo_rezago": {"final": {"aumento_puntos": 1.234, "aumento_relativo_pct": 48.76}},
        "tablas": {"r24": {
            "final": {"metricas_prueba": {"mape_pct": 2.214, "mape_pico_pct": 2.87},
                      "criterio_exito_prueba": {"mejora_mape_pct": 37.7, "cumple": True}},
            "tabla_seleccion": [{"rol": "ensemble", "nombre": "hist_gradient", "forma": "nivel", "es_final": True,
                                 "interpretabilidad": {"calificacion": "baja"},
                                 "rendimiento": {"validacion": {"mape_pct": 1.86},
                                                 "prueba": {"mape_pct": 2.214, "mape_pico_pct": 2.87,
                                                            "mejora_mape_pct": 37.7}},
                                 "eficiencia": {"prediccion_ms_por_1000_filas": 8.1, "tamano_bytes": 1_090_000}}],
        }},
    }
    (tmp_path / "modelado_seleccion.json").write_text(json.dumps(seleccion), encoding="utf-8")
    macros, tablas = macros_modelado(tmp_path, _num)
    assert macros["modFinalMapeRxxiv"] == "2.21" and macros["modFinalCumpleRxxiv"] == "meets"
    assert macros["modFinalRxxiv"] == "HistGradientBoosting (level)" and macros["modFinalInterpRxxiv"] == "low"
    assert macros["modRezagoPuntos"] == "1.23" and macros["modInicioPrueba"] == "2025-09-03"
    assert macros["modFinalMapeRcxx"] == r"\pendiente"
    assert r"\textbf{HistGradientBoosting (level)}" in tablas["tabla_seleccion.tex"]
    assert tabla_seleccion({}, _num).count(r"\pendiente") == 1


@pytest.mark.skipif(not (RAIZ / "docs" / "resultados" / "modelado_seleccion.json").exists(),
                    reason="sin evaluacion final")
def test_con_los_resultados_del_repo_no_queda_ninguna_macro_pendiente():
    macros, _ = macros_modelado(RAIZ / "docs" / "resultados", _num)
    pendientes = [nombre for nombre, valor in macros.items() if valor == r"\pendiente"]
    assert not pendientes, f"macros sin resultado: {pendientes}"
