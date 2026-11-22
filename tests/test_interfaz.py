"""Interfaz de Streamlit (app/) y su exportador (scripts/exportar_interfaz.py). Sin red.

Dos garantias de la interfaz:

- cada pagina se ejecuta sin excepciones con los artefactos reales del repo;
- con un directorio vacio ninguna falla: muestra el aviso de pendiente, que
  dice que archivo falta y que comando lo genera.

Y del exportador: sobre un panel sintetico produce las tablas que la app lee,
anota en el manifiesto lo que falta en vez de fallar y no deja temporales.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import socket
import sys
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "app"
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(APP))

import conftest  # noqa: E402
from test_scripts_entrega import panel_dos_anios  # noqa: E402

PAGINAS = sorted(p.stem for p in (APP / "paginas").glob("*.py"))
PAGINAS_MODELADO = {"modelos", "prediccion", "alertas", "interpretabilidad", "seleccion"}


@pytest.fixture(autouse=True)
def socketpair_local(monkeypatch):
    """AppTest crea un bucle de asyncio, y en Windows ese bucle abre un par de sockets
    *locales* (socket.socketpair conecta con 127.0.0.1). No es red: se permite solo
    esa conexion; cualquier otra sigue bloqueada por conftest.sin_red."""
    original = socket.socketpair

    def par(*args, **kwargs):
        bloqueado = socket.socket.connect
        socket.socket.connect = conftest._conectar_real
        try:
            return original(*args, **kwargs)
        finally:
            socket.socket.connect = bloqueado

    monkeypatch.setattr(socket, "socketpair", par)


def _cargar_exportador():
    spec = importlib.util.spec_from_file_location("exportar_interfaz", RAIZ / "scripts" / "exportar_interfaz.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _ejecutar(pagina: str) -> AppTest:
    app = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=120)
    app.run()
    app.switch_page(f"paginas/{pagina}.py")
    app.run()
    return app


# --------------------------------------------------------------------------
# Estructura
# --------------------------------------------------------------------------


def test_cada_pagina_esta_en_la_navegacion():
    """Una pagina en paginas/ que no figura en st.navigation no se puede abrir desde la app."""
    inicio = (APP / "Inicio.py").read_text(encoding="utf-8")
    registradas = set(re.findall(r'_pagina\("(\w+)\.py"', inicio))
    assert registradas == set(PAGINAS)
    assert len(PAGINAS) == 10


def test_la_paleta_de_la_app_es_la_del_informe():
    from componentes import estilo as estilo_app
    from mineria import estilo as estilo_informe

    assert estilo_app.CATEGORICA == estilo_informe.CATEGORICA
    assert (estilo_app.TINTA, estilo_app.REJILLA) == (estilo_informe.TINTA, estilo_informe.REJILLA)
    tema = tomllib.loads((RAIZ / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    assert tema["chartCategoricalColors"] == estilo_informe.CATEGORICA
    assert tema["primaryColor"] == estilo_informe.CATEGORICA[0]


def test_la_app_no_entrena_ni_descarga():
    """La regla central: app/ solo lee artefactos. Ni sklearn, ni clientes de las APIs, ni red."""
    prohibidos = {"sklearn", "requests", "urllib", "http", "socket", "joblib", "ingesta", "modelado", "variables"}
    for ruta in APP.rglob("*.py"):
        for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
            if isinstance(nodo, ast.Import):
                nombres = {a.name.split(".")[0] for a in nodo.names}
            elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
                nombres = {nodo.module.split(".")[0]}
            else:
                continue
            assert not nombres & prohibidos, f"{ruta.relative_to(RAIZ)} importa {nombres & prohibidos}"


def test_cada_artefacto_tiene_un_generador_conocido():
    """El aviso de pendiente no debe caer en el comando generico para lo que la app lee."""
    from componentes import pendiente

    leidos = set()
    for ruta in APP.rglob("*.py"):
        leidos |= set(re.findall(r'"((?:app/datos|docs|data|infra)/[^"*]+\.\w+)"', ruta.read_text(encoding="utf-8")))
    assert leidos
    sin_generador = sorted(r for r in leidos if pendiente.generador(r) == pendiente.POR_DEFECTO)
    assert not sin_generador, sin_generador


def test_formato_espanol():
    from componentes.formato import num, pct

    assert num(49824) == "49 824"
    assert pct(2.2050064, 2) == "2,21 %"
    assert num(None) == "—"


# --------------------------------------------------------------------------
# Paginas
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_con_los_artefactos_reales(pagina, monkeypatch):
    monkeypatch.delenv("SIEM_INTERFAZ_RAIZ", raising=False)
    app = _ejecutar(pagina)
    assert not app.exception, [e.value for e in app.exception]
    assert app.title, f"{pagina}: sin titulo"
    if pagina not in PAGINAS_MODELADO:
        avisos = [w.value for w in app.warning if "Pendiente" in w.value]
        assert not avisos, f"{pagina} tiene artefactos pendientes con el repo completo: {avisos}"


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_sin_artefactos_muestra_pendiente(pagina, tmp_path, monkeypatch):
    monkeypatch.setenv("SIEM_INTERFAZ_RAIZ", str(tmp_path))
    app = _ejecutar(pagina)
    assert not app.exception, [e.value for e in app.exception]
    avisos = [w.value for w in app.warning if "Pendiente" in w.value]
    assert avisos, f"{pagina} no avisa de que faltan sus artefactos"
    assert all(re.search(r"Se genera con `python ", a) for a in avisos), avisos


def test_inicio_lee_las_cifras_del_json(monkeypatch):
    """Las cifras clave salen de modelado_seleccion.json, no estan escritas en la pagina."""
    monkeypatch.delenv("SIEM_INTERFAZ_RAIZ", raising=False)
    seleccion = json.loads((RAIZ / "docs" / "resultados" / "modelado_seleccion.json").read_text(encoding="utf-8"))
    from componentes.formato import pct

    app = _ejecutar("inicio")
    valores = {m.label: m.value for m in app.metric}
    mape = seleccion["tablas"]["r24"]["final"]["metricas_prueba"]["mape_pct"]
    assert valores["MAPE en prueba"] == pct(mape, 2)

    app.radio[0].set_value("r120").run()
    valores = {m.label: m.value for m in app.metric}
    mape = seleccion["tablas"]["r120"]["final"]["metricas_prueba"]["mape_pct"]
    assert valores["MAPE en prueba"] == pct(mape, 2)


def test_filtro_de_reglas(monkeypatch):
    monkeypatch.delenv("SIEM_INTERFAZ_RAIZ", raising=False)
    reglas = pd.read_csv(RAIZ / "docs" / "resultados" / "mineria_reglas.csv")
    app = _ejecutar("mineria")
    lift = app.slider[2]
    lift.set_value(float(reglas["lift"].quantile(0.5))).run()
    esperadas = int((reglas["lift"] >= lift.value).sum())
    assert any(c.value.startswith(f"{esperadas} de {len(reglas)} reglas") for c in app.caption)


# --------------------------------------------------------------------------
# Exportador
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exportado(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("interfaz")
    ruta_panel = carpeta / "panel.parquet"
    panel_dos_anios().to_parquet(ruta_panel, index=False)
    resultados = carpeta / "resultados"
    resultados.mkdir()
    destino = carpeta / "datos"
    manifiesto = _cargar_exportador().exportar(ruta_panel, resultados, destino)
    return manifiesto, destino


def test_exportador_tablas(exportado):
    manifiesto, destino = exportado
    serie = pd.read_parquet(destino / "serie_diaria.parquet")
    perfil = pd.read_parquet(destino / "perfil_horario.parquet")
    acf = pd.read_parquet(destino / "autocorrelacion.parquet")

    assert serie["fecha"].is_monotonic_increasing and serie["fecha"].is_unique
    assert set(serie["tipo_dia"]) <= {"laboral", "sabado", "domingo", "festivo", "semana_santa"}
    assert {"semana_santa", "festivo", "domingo", "sabado", "laboral"} <= set(serie["tipo_dia"])
    # el festivo manda sobre el dia de la semana, y la Semana Santa sobre todo
    fuera_de_santa = serie[serie["tipo_dia"] != "semana_santa"]
    assert (fuera_de_santa["es_festivo"] == (fuera_de_santa["tipo_dia"] == "festivo")).all()
    assert (serie["demanda_mwh"] > 0).all()
    # los ultimos 5 dias del panel (publicacion parcial) no entran
    assert serie["fecha"].max() == pd.Timestamp("2025-12-26")

    assert len(perfil) == 168 and not perfil.duplicated(["dia_semana", "hora"]).any()
    assert acf["rezago_h"].tolist() == list(range(337)) and acf["acf"].iloc[0] == pytest.approx(1.0)
    assert acf["pacf"].notna().sum() == 49


def test_exportador_manifiesto(exportado):
    manifiesto, destino = exportado
    exportador = _cargar_exportador()
    escrito = json.loads((destino / "manifiesto_interfaz.json").read_text(encoding="utf-8"))
    assert escrito["version_formato"] == 1
    for nombre, info in escrito["archivos"].items():
        assert exportador.sha256(destino / nombre) == info["sha256"]
        assert info["registro"]["operacion"]
    # sin etiquetas de la mineria ni predicciones: se anota y se sigue
    faltan = {f["archivo"]: f["script"] for f in escrito["faltantes"]}
    assert set(faltan) == {"centroides_grupos.parquet", "predicciones_prueba.parquet", "probabilidades_alerta.parquet"}
    assert faltan["predicciones_prueba.parquet"] == "python scripts/evaluar_final.py"
    assert not list(destino.glob("*.tmp"))


def test_exportador_centroides_y_copias(tmp_path):
    exportador = _cargar_exportador()
    panel = panel_dos_anios()
    ruta_panel = tmp_path / "panel.parquet"
    panel.to_parquet(ruta_panel, index=False)
    resultados = tmp_path / "resultados"
    resultados.mkdir()
    fechas = pd.date_range("2024-01-01", "2025-12-26", freq="D")
    pd.DataFrame({"fecha": fechas.strftime("%Y-%m-%d"),
                  "grupo": ["G2" if d.dayofweek == 6 else "G1" for d in fechas]}).to_csv(
        resultados / "mineria_etiquetas_dia.csv", index=False)
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(resultados / "predicciones_prueba.parquet")

    manifiesto = exportador.exportar(ruta_panel, resultados, tmp_path / "datos")
    centroides = pd.read_parquet(tmp_path / "datos" / "centroides_grupos.parquet")
    assert set(centroides["grupo"]) == {"G1", "G2"} and len(centroides) == 48
    # la forma de cada dia tiene media 1, y la media de formas tambien
    assert centroides.groupby("grupo")["forma_media"].mean().to_numpy() == pytest.approx([1.0, 1.0])
    assert (centroides["p10"] <= centroides["forma_media"]).all() and (centroides["forma_media"] <= centroides["p90"]).all()
    assert manifiesto["archivos"]["predicciones_prueba.parquet"]["filas"] == 3
    assert [f["archivo"] for f in manifiesto["faltantes"]] == ["probabilidades_alerta.parquet"]


def test_exportador_sin_panel_falla_con_el_comando(tmp_path):
    with pytest.raises(FileNotFoundError, match="construir_panel.py"):
        _cargar_exportador().exportar(tmp_path / "no_existe.parquet", tmp_path, tmp_path / "datos")


def test_app_datos_versionado_cuadra_con_su_manifiesto():
    """Lo que esta en app/datos/ es lo que dice el manifiesto (nadie lo edito a mano)."""
    exportador = _cargar_exportador()
    destino = APP / "datos"
    manifiesto = json.loads((destino / "manifiesto_interfaz.json").read_text(encoding="utf-8"))
    for nombre, info in manifiesto["archivos"].items():
        assert exportador.sha256(destino / nombre) == info["sha256"], nombre
    for nombre in ("predicciones_prueba.parquet", "probabilidades_alerta.parquet"):
        origen = RAIZ / "docs" / "resultados" / nombre
        if origen.exists() and nombre in manifiesto["archivos"]:
            assert exportador.sha256(origen) == manifiesto["archivos"][nombre]["sha256"], (
                f"{nombre} cambio en docs/resultados: vuelve a correr python scripts/exportar_interfaz.py"
            )
