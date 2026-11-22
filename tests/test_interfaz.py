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
from componentes.formato import num, pct  # noqa: E402
from test_scripts_entrega import panel_dos_anios  # noqa: E402

PAGINAS = sorted(p.stem for p in (APP / "paginas").glob("*.py"))


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


@pytest.mark.parametrize("pagina", ["modelos", "prediccion", "alertas", "interpretabilidad", "seleccion"])
def test_pagina_de_modelado_en_r120(pagina, monkeypatch):
    """Cada pagina de modelado tiene su selector de version; con r120 tampoco falla."""
    monkeypatch.delenv("SIEM_INTERFAZ_RAIZ", raising=False)
    app = _ejecutar(pagina)
    app.radio[0].set_value("r120").run()
    assert not app.exception, [e.value for e in app.exception]


def test_alertas_con_el_umbral_del_modelo_reproduce_la_evaluacion(monkeypatch):
    monkeypatch.delenv("SIEM_INTERFAZ_RAIZ", raising=False)
    seleccion = _leer_json("docs/resultados/modelado_seleccion.json")
    app = _ejecutar("alertas")
    registrada = seleccion["tablas"]["r24"]["clasificador_prueba"]["matriz_confusion"]
    valores = {m.label: m.value for m in app.metric}
    assert valores["Alertas emitidas"] == num(registrada["vp"] + registrada["fp"])
    app.slider[0].set_value(0.9).run()
    assert {m.label: m.value for m in app.metric}["Alertas emitidas"] != valores["Alertas emitidas"]


# --------------------------------------------------------------------------
# componentes.modelado: lo que la app calcula coincide con los JSON
# --------------------------------------------------------------------------


def _leer_json(relativa: str) -> dict:
    return json.loads((RAIZ / relativa).read_text(encoding="utf-8"))


@pytest.mark.parametrize("version", ["r24", "r120"])
def test_matriz_y_pr_auc_reproducen_la_evaluacion_final(version):
    from componentes import modelado

    seleccion = _leer_json("docs/resultados/modelado_seleccion.json")["tablas"][version]
    datos = pd.read_parquet(APP / "datos" / "probabilidades_alerta.parquet")
    datos = datos[datos["tabla"] == version]
    real = datos["alerta_real"].to_numpy()
    umbral = modelado.probabilidad_de_puntaje(0.0, seleccion["calibracion_alerta"])
    m = modelado.matriz_confusion(real, datos["probabilidad"].to_numpy(), umbral)
    registro = seleccion["clasificador_prueba"]
    assert {k: m[k] for k in ("vn", "fp", "fn", "vp")} == {k: registro["matriz_confusion"][k] for k in ("vn", "fp", "fn", "vp")}
    assert m["f1"] == pytest.approx(registro["f1"])
    curva = modelado.curva_precision_recall(real, datos["puntaje"].to_numpy())
    precision_media = float((curva["recall"].diff().fillna(curva["recall"].iloc[0]) * curva["precision"]).sum())
    assert precision_media == pytest.approx(registro["pr_auc"])


def test_mape_de_las_predicciones_coincide_con_el_registro():
    """El formato ancho de app/datos conserva cada prediccion: su MAPE es el registrado en la evaluacion final."""
    from componentes import modelado

    seleccion = _leer_json("docs/resultados/modelado_seleccion.json")["tablas"]
    datos = pd.read_parquet(APP / "datos" / "predicciones_prueba.parquet")
    for version, tabla in seleccion.items():
        filas = datos[datos["tabla"] == version]
        assert len(filas) == tabla["metricas_prueba"]["baseline"]["n"]
        for rol, metricas in tabla["metricas_prueba"].items():
            assert modelado.mape(filas["real_mwh"], filas[f"{rol}_mwh"]) == pytest.approx(metricas["mape_pct"]), (version, rol)
        assert modelado.mape(filas["real_mwh"], filas["final_mwh"]) == pytest.approx(
            tabla["final"]["metricas_prueba"]["mape_pct"])
    # las banderas del calendario vienen del panel: hay festivos y Semana Santa en el ano de prueba
    assert datos["es_festivo"].any() and datos["es_semana_santa"].any()
    assert datos.loc[datos["es_festivo"], "nombre_festivo"].notna().all()


def test_prediccion_atajo_a_las_peores_semanas(monkeypatch):
    monkeypatch.delenv("SIEM_INTERFAZ_RAIZ", raising=False)
    app = _ejecutar("prediccion")
    atajos = [b for b in app.button if "MAPE" in str(b.label)]
    assert len(atajos) == 3
    atajos[1].click().run()
    assert not app.exception
    assert str(app.select_slider[0].value)[:10] == str(atajos[1].label)[:10]


def test_seleccion_e_interpretabilidad_muestran_los_textos(monkeypatch):
    monkeypatch.delenv("SIEM_INTERFAZ_RAIZ", raising=False)
    app = _ejecutar("seleccion")
    assert any("Modelo final en r24" in m.value for m in app.markdown), "falta la justificacion de seleccion.md"
    app = _ejecutar("interpretabilidad")
    lecturas = [m.value for m in app.markdown if "R² local" in m.value]
    assert len(lecturas) == 4, lecturas


def test_lectura_lime():
    from componentes import modelado

    frase = modelado.lectura_lime({"demanda_real_mwh": 1000.0, "prediccion_mwh": 950.0, "r2_local": 0.6,
                                   "pesos": [{"regla": "a > 1", "peso_mwh": 30.0}, {"regla": "b=1", "peso_mwh": -80.0}]})
    assert "subestima 50 MWh" in frase and "`a > 1` (+30 MWh)" in frase and "`b=1` (-80 MWh)" in frase
    assert "buena" in frase


def test_tablas_de_modelos():
    from componentes import modelado

    resultados = {f: _leer_json(f"docs/resultados/{a}") for f, a in (
        ("baseline", "modelado_baselines.json"), ("lineal", "modelado_regresion.json"),
        ("ensembles", "modelado_ensembles.json"), ("busqueda", "modelado_busqueda.json"))}
    tabla = modelado.tabla_regresion(resultados, "r24")
    estacional = tabla[(tabla["familia"] == "Baseline") & (tabla["modelo"] == "estacional")]
    assert estacional["mape_pct"].iloc[0] == pytest.approx(
        resultados["baseline"]["tablas"]["r24"]["modelos"]["estacional"]["metricas_validacion"]["mape_pct"])
    assert {"Baseline", "Lineal", "Ensemble", "Búsqueda"} <= set(tabla["familia"])
    assert tabla["mape_pct"].notna().all()
    # r120 no tiene lag_24: no hay baseline diario
    assert "diario" not in set(modelado.tabla_regresion(resultados, "r120")["modelo"])

    pliegues = modelado.pliegues(resultados, "r24")
    assert (pliegues.groupby("modelo")["pliegue"].max() == 5).all()
    trayectorias = modelado.trayectorias_busqueda(resultados["busqueda"], "r24")
    # el mejor acumulado nunca empeora (MAE: nunca sube)
    for (_, metodo), serie in trayectorias[trayectorias["candidato"].str.contains("nivel")].groupby(["candidato", "metodo"]):
        assert serie["mejor_cv"].is_monotonic_decreasing, metodo


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


def _predicciones_largas(panel: pd.DataFrame, desde: str = "2025-12-01") -> pd.DataFrame:
    """Predicciones de prueba sinteticas en el formato largo que escribe evaluar_final.py."""
    horas = pd.to_datetime(panel["fecha_hora"])
    horas = horas[horas >= pd.Timestamp(desde, tz="America/Bogota")]
    real = panel.set_index("fecha_hora").loc[horas, "valor_kwh"].to_numpy() / 1000
    partes = [pd.DataFrame({"fecha_hora": horas.to_numpy(), "tabla": "r24", "rol": rol, "modelo": f"modelo {rol}",
                            "es_final": rol == "ensemble", "real_mwh": real, "prediccion_mwh": real * factor})
              for rol, factor in (("baseline", 1.03), ("lineal", 0.98), ("ensemble", 1.01))]
    return pd.concat(partes, ignore_index=True)


def test_exportador_centroides_y_predicciones(tmp_path):
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
    largas = _predicciones_largas(panel)
    largas.to_parquet(resultados / "predicciones_prueba.parquet")

    manifiesto = exportador.exportar(ruta_panel, resultados, tmp_path / "datos")
    centroides = pd.read_parquet(tmp_path / "datos" / "centroides_grupos.parquet")
    assert set(centroides["grupo"]) == {"G1", "G2"} and len(centroides) == 48
    # la forma de cada dia tiene media 1, y la media de formas tambien
    assert centroides.groupby("grupo")["forma_media"].mean().to_numpy() == pytest.approx([1.0, 1.0])
    assert (centroides["p10"] <= centroides["forma_media"]).all() and (centroides["forma_media"] <= centroides["p90"]).all()

    ancho = pd.read_parquet(tmp_path / "datos" / "predicciones_prueba.parquet")
    assert len(ancho) == len(largas) // 3
    assert {"real_mwh", "baseline_mwh", "lineal_mwh", "ensemble_mwh", "final_mwh", "es_festivo", "nombre_festivo"} <= set(ancho)
    assert (ancho["final_mwh"] == ancho["ensemble_mwh"]).all()
    assert ancho["baseline_mwh"].to_numpy() == pytest.approx(ancho["real_mwh"].to_numpy() * 1.03)
    # 25 de diciembre: festivo con nombre, tomado del panel
    navidad = ancho[ancho["fecha_hora"].dt.strftime("%m-%d") == "12-25"]
    assert len(navidad) == 24 and navidad["es_festivo"].all() and navidad["nombre_festivo"].notna().all()
    registro = manifiesto["archivos"]["predicciones_prueba.parquet"]["registro"]
    assert registro["final"] == {"r24": "ensemble"} and registro["horas_sin_calendario"] == 0
    assert [f["archivo"] for f in manifiesto["faltantes"]] == ["probabilidades_alerta.parquet"]


def test_exportador_sin_predicciones_las_calcula_con_los_modelos_guardados(tmp_path):
    """Si evaluar_final.py no dejo las predicciones, se cargan los modelos guardados (con su SHA-256) y no se reentrena."""
    import joblib
    from sklearn.linear_model import LinearRegression
    from variables.construir import construir_tabla

    exportador = _cargar_exportador()
    panel = panel_dos_anios()
    ruta_panel = tmp_path / "panel.parquet"
    panel.to_parquet(ruta_panel, index=False)
    procesado, modelos, resultados = tmp_path / "procesado", tmp_path / "models", tmp_path / "resultados"
    for carpeta in (procesado, modelos, resultados):
        carpeta.mkdir()
    tabla, _ = construir_tabla(panel, 24)
    tabla.to_parquet(procesado / "tabla_modelado_r24.parquet", index=False)
    variables = [c for c in tabla.columns if c not in ("fecha_hora", "demanda_mwh")]
    (procesado / "registro_variables.json").write_text(json.dumps({"versiones": {"r24": {
        "rezago_minimo_h": 24, "archivo": "tabla_modelado_r24.parquet", "variables": variables,
        "sha256": exportador.sha256(procesado / "tabla_modelado_r24.parquet")}}}), encoding="utf-8")

    filas_ajuste = tabla["fecha_hora"] < pd.Timestamp("2025-06-01", tz="America/Bogota")
    entradas = []
    for rol in ("baseline", "lineal", "ensemble"):
        columnas = ["lag_168"] if rol == "baseline" else variables
        modelo = LinearRegression().fit(tabla.loc[filas_ajuste, variables].assign(
            **{c: 0.0 for c in variables if c not in columnas}), tabla.loc[filas_ajuste, "demanda_mwh"])
        ruta = modelos / f"prueba_{rol}_r24.joblib"
        joblib.dump(modelo, ruta)
        entradas.append({"rol": rol, "modelo": f"lr {rol}", "modelo_guardado": {
            "ruta": f"models/{ruta.name}", "sha256": exportador.sha256(ruta)}})
    entradas.append({"rol": "clasificador", "modelo": "svm", "modelo_guardado": {"ruta": "models/no_existe", "sha256": "x"}})
    seleccion = {"bloque_prueba": ["2025-09-03 00:00:00-05:00", "2025-12-26 23:00:00-05:00"],
                 "tablas": {"r24": {"final": {"rol": "lineal"}, "tabla_seleccion": entradas}}}
    (resultados / "modelado_seleccion.json").write_text(json.dumps(seleccion), encoding="utf-8")

    manifiesto = exportador.exportar(ruta_panel, resultados, tmp_path / "datos", procesado=procesado, modelos=modelos)
    ancho = pd.read_parquet(tmp_path / "datos" / "predicciones_prueba.parquet")
    prueba = tabla[tabla["fecha_hora"] >= pd.Timestamp("2025-09-03", tz="America/Bogota")]
    assert len(ancho) == len(prueba)
    esperado = joblib.load(modelos / "prueba_lineal_r24.joblib").predict(prueba[variables])
    assert ancho["lineal_mwh"].to_numpy() == pytest.approx(esperado)
    assert (ancho["final_mwh"] == ancho["lineal_mwh"]).all()
    assert "predecir_con_modelos_guardados" in manifiesto["archivos"]["predicciones_prueba.parquet"]["origen"]

    # un modelo que no es el anotado no se usa: se anota como pendiente
    joblib.dump(LinearRegression().fit([[0.0]], [0.0]), modelos / "prueba_lineal_r24.joblib")
    manifiesto = exportador.exportar(ruta_panel, resultados, tmp_path / "datos2", procesado=procesado, modelos=modelos)
    faltan = {f["archivo"]: f["falta"] for f in manifiesto["faltantes"]}
    assert "SHA-256" in faltan["predicciones_prueba.parquet"]


def test_exportador_sin_panel_falla_con_el_comando(tmp_path):
    with pytest.raises(FileNotFoundError, match="construir_panel.py"):
        _cargar_exportador().exportar(tmp_path / "no_existe.parquet", tmp_path, tmp_path / "datos")


def test_app_datos_versionado_cuadra_con_su_manifiesto():
    """Lo que esta en app/datos/ es lo que dice el manifiesto (nadie lo edito a mano) y sigue a docs/resultados."""
    exportador = _cargar_exportador()
    destino = APP / "datos"
    manifiesto = json.loads((destino / "manifiesto_interfaz.json").read_text(encoding="utf-8"))
    for nombre, info in manifiesto["archivos"].items():
        assert exportador.sha256(destino / nombre) == info["sha256"], nombre
    regenera = "vuelve a correr python scripts/exportar_interfaz.py"
    origen = RAIZ / "docs" / "resultados" / "probabilidades_alerta.parquet"
    if origen.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(destino / "probabilidades_alerta.parquet"), pd.read_parquet(origen),
                                      obj=f"probabilidades_alerta.parquet cambio en docs/resultados: {regenera}")
    origen = RAIZ / "docs" / "resultados" / "predicciones_prueba.parquet"
    if origen.exists():
        largas = pd.read_parquet(origen)
        ancho = pd.read_parquet(destino / "predicciones_prueba.parquet").set_index(["fecha_hora", "tabla"])
        for rol, filas in largas.groupby("rol"):
            filas = filas.set_index(["fecha_hora", "tabla"])
            assert ancho.loc[filas.index, f"{rol}_mwh"].to_numpy() == pytest.approx(filas["prediccion_mwh"].to_numpy()), (
                f"predicciones_prueba.parquet cambio en docs/resultados: {regenera}")
