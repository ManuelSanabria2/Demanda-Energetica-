"""Pruebas de la mineria no supervisada (src/mineria). Sin red.

Se construye un panel sintetico con tres formas de dia conocidas (laboral,
sabado, domingo/festivo), tendencia anual y ruido. Lo que se comprueba:

- los perfiles solo admiten dias completos y descartan la publicacion parcial;
- la forma quita el nivel, asi que la tendencia no decide los grupos;
- K-means recupera los tres tipos de dia y Ward coincide;
- las reglas encuentran domingo/festivo -> su grupo con lift > 1;
- el perfil sectorial distingue proceso continuo de jornada laboral.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from mineria.clustering import (  # noqa: E402
    agrupar,
    contingencia,
    describir,
    dias_atipicos,
    elegir_k,
    evaluar_k,
    v_cramer,
)
from mineria.perfiles import (  # noqa: E402
    ErrorPerfiles,
    atributos_dia,
    forma_y_nivel,
    perfiles_diarios,
)
from mineria.reglas import podar_redundantes, reglas_asociacion, reglas_hacia, transacciones  # noqa: E402
from mineria.sectores import (  # noqa: E402
    describir_sectores,
    indicadores_forma,
    ordenar_por_tamano,
    perfiles_sectoriales,
)

HORAS = np.arange(24)
LABORAL = 1 + 0.25 * np.exp(-((HORAS - 11) ** 2) / 8) + 0.30 * np.exp(-((HORAS - 19) ** 2) / 4) - 0.25 * np.exp(-((HORAS - 3) ** 2) / 6)
SABADO = 1 + 0.15 * np.exp(-((HORAS - 12) ** 2) / 10) + 0.20 * np.exp(-((HORAS - 19) ** 2) / 4) - 0.20 * np.exp(-((HORAS - 4) ** 2) / 6)
DOMINGO = 1 + 0.05 * np.exp(-((HORAS - 13) ** 2) / 12) + 0.30 * np.exp(-((HORAS - 20) ** 2) / 3) - 0.15 * np.exp(-((HORAS - 5) ** 2) / 8)
FESTIVOS = {"2024-01-01", "2024-03-25", "2024-05-01", "2024-07-20", "2024-08-07", "2024-12-25"}


def _panel(dias: int = 364, ruido: float = 0.01, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2024-01-01", periods=dias, freq="D")
    filas = []
    for i, dia in enumerate(fechas):
        festivo = dia.strftime("%Y-%m-%d") in FESTIVOS
        forma = DOMINGO if (festivo or dia.dayofweek == 6) else SABADO if dia.dayofweek == 5 else LABORAL
        nivel = 9_000_000 * (1 + 0.10 * i / dias)          # tendencia: +10 % en el ano
        valores = nivel * forma * (1 + rng.normal(0, ruido, 24))
        for h in HORAS:
            filas.append((dia + pd.Timedelta(hours=int(h)), valores[h], festivo, dia.month == 12 and dia.day >= 25))
    marco = pd.DataFrame(filas, columns=["fecha_hora", "valor_kwh", "es_festivo", "es_ultima_semana_diciembre"])
    marco["fecha_hora"] = marco["fecha_hora"].dt.tz_localize("America/Bogota")
    marco["imputado"] = False
    marco["es_vispera_festivo"] = False
    marco["es_puente"] = False
    marco["es_semana_santa"] = False
    marco["temperatura_c"] = 20 + 3 * np.sin(2 * np.pi * marco["fecha_hora"].dt.hour / 24)
    marco["nombre_festivo"] = np.where(marco["es_festivo"], "festivo", None)
    return marco


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


@pytest.fixture(scope="module")
def datos(panel):
    perfiles, registro = perfiles_diarios(panel)
    forma, nivel = forma_y_nivel(perfiles)
    atributos = atributos_dia(panel, perfiles.index)
    return perfiles, registro, forma, nivel, atributos


# --------------------------------------------------------------------------
# Perfiles
# --------------------------------------------------------------------------


def test_perfiles_descartan_dias_parciales_e_incompletos(panel):
    roto = panel.copy()
    roto.loc[5 * 24 + 3, "valor_kwh"] = np.nan                 # un hueco el 6 de enero
    roto.loc[10 * 24 + 7, "imputado"] = True                  # un imputado el 11 de enero
    perfiles, registro = perfiles_diarios(roto, dias_parciales=5)
    assert registro["dias_totales"] == 364
    assert registro["dias_incompletos"] == 2
    assert registro["dias_publicacion_parcial"] == 5
    assert len(perfiles) == 364 - 2 - 5
    assert pd.Timestamp("2024-01-06").date() not in perfiles.index
    assert max(perfiles.index) == (pd.Timestamp("2024-12-29") - pd.Timedelta(days=5)).date()
    assert list(perfiles.columns) == [f"h{h:02d}" for h in range(24)]


def test_perfiles_en_mwh_y_hora_local(panel):
    perfiles, _ = perfiles_diarios(panel, dias_parciales=0)
    primero = panel.iloc[0]["valor_kwh"] / 1000
    assert perfiles.iloc[0]["h00"] == pytest.approx(primero)


def test_perfiles_rechazan_marcas_repetidas(panel):
    with pytest.raises(ErrorPerfiles):
        perfiles_diarios(pd.concat([panel, panel.iloc[[0]]]))


def test_forma_quita_el_nivel(datos):
    _, _, forma, nivel, _ = datos
    assert np.allclose(forma.mean(axis=1), 1.0)
    # la tendencia del +10 % esta en el nivel, no en la forma
    assert nivel.iloc[-30:].mean() > nivel.iloc[:30].mean() * 1.05


def test_atributos_tipo_de_dia(datos):
    _, _, _, _, atributos = datos
    assert atributos.loc[pd.Timestamp("2024-01-01").date(), "tipo_dia"] == "festivo"
    assert atributos.loc[pd.Timestamp("2024-01-06").date(), "tipo_dia"] == "sabado"
    assert atributos.loc[pd.Timestamp("2024-01-07").date(), "tipo_dia"] == "domingo"
    assert atributos.loc[pd.Timestamp("2024-01-09").date(), "tipo_dia"] == "laboral"


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resultado(datos):
    _, _, forma, _, _ = datos
    evaluacion = evaluar_k(forma.to_numpy(), ks=range(2, 6))
    return evaluacion, agrupar(forma, elegir_k(evaluacion))


def test_k_elegido_es_el_numero_real_de_formas(resultado):
    evaluacion, final = resultado
    assert final.k == 3
    fila = evaluacion.set_index("k").loc[3]
    assert fila["silueta"] == evaluacion["silueta"].max()
    assert fila["estabilidad_ari"] > 0.95


def test_grupos_recuperan_el_tipo_de_dia(resultado, datos):
    _, final = resultado
    _, _, _, _, atributos = datos
    tabla = contingencia(final, atributos)
    assert v_cramer(tabla) > 0.95
    # G1 es el mayor: los laborales
    assert tabla.loc["G1", "laboral"] == tabla["laboral"].sum()
    assert final.ari_ward > 0.95


def test_etiquetas_deterministas(datos):
    _, _, forma, _, _ = datos
    a = agrupar(forma, 3).etiquetas
    b = agrupar(forma, 3).etiquetas
    assert (a == b).all()
    tamanos = a.value_counts()
    assert tamanos["G1"] >= tamanos["G2"] >= tamanos["G3"]


def test_describir_y_atipicos(resultado, datos):
    _, final = resultado
    _, _, forma, nivel, atributos = datos
    tabla = describir(final, forma, nivel, atributos)
    assert tabla["dias"].sum() == len(forma)
    assert tabla.loc["G1", "pct_laboral"] == 100.0
    assert tabla.loc["G1", "hora_pico"] == 19
    lejos = dias_atipicos(final, atributos, cuantil=0.95)
    assert 0 < len(lejos) <= int(0.06 * len(forma)) + 1


def test_tendencia_no_decide_los_grupos(datos):
    """Con MWh en vez de forma, un grupo seria 'fin de ano'; con forma, no."""
    _, _, forma, nivel, atributos = datos
    final = agrupar(forma, 3)
    meses = atributos[["mes"]].join(final.etiquetas)
    g1 = meses[meses["grupo"] == "G1"]["mes"]
    assert g1.nunique() == 12


# --------------------------------------------------------------------------
# Reglas de asociacion
# --------------------------------------------------------------------------


def test_reglas_encuentran_calendario_hacia_grupo(resultado, datos):
    _, final = resultado
    _, _, _, nivel, atributos = datos
    canasta = transacciones(atributos, final.etiquetas, nivel)
    assert canasta.dtypes.eq(bool).all()
    assert canasta.filter(like="grupo=").sum(axis=1).eq(1).all()
    reglas = reglas_asociacion(canasta, soporte_minimo=0.01)
    hacia = reglas_hacia(reglas, "grupo=")
    domingo = hacia[hacia["antecedente"] == "dia=domingo"]
    assert len(domingo) == 1
    assert domingo.iloc[0]["confianza"] == pytest.approx(1.0)
    assert domingo.iloc[0]["lift"] > 3
    assert (reglas["lift"] >= 1.2).all()


def test_poda_de_reglas_redundantes():
    reglas = pd.DataFrame({
        "antecedente": ["dia=domingo", "dia=domingo & temp=fria", "dia=sabado & temp=fria"],
        "consecuente": ["grupo=G2", "grupo=G2", "grupo=G3"],
        "soporte": [0.14, 0.05, 0.04], "confianza": [1.0, 1.0, 0.9], "lift": [5.0, 5.0, 4.0], "n_items": [2, 3, 3],
    })
    podadas = podar_redundantes(reglas)
    assert podadas["antecedente"].tolist() == ["dia=domingo", "dia=sabado & temp=fria"]


def test_terciles_de_nivel_son_por_anio(resultado, datos):
    _, final = resultado
    _, _, _, nivel, atributos = datos
    canasta = transacciones(atributos, final.etiquetas, nivel)
    # con tendencia y terciles globales, casi todo 'nivel=alto' caeria en el ultimo tercio
    altos = canasta.index[canasta["nivel=alto"]]
    assert canasta["nivel=alto"].mean() == pytest.approx(1 / 3, abs=0.01)
    assert len(altos) > 0


# --------------------------------------------------------------------------
# Sectores CIIU
# --------------------------------------------------------------------------


def _ciiu(semanas: int = 10) -> pd.DataFrame:
    tiempo = pd.date_range("2024-01-01", periods=24 * 7 * semanas, freq="h", tz="America/Bogota")
    rng = np.random.default_rng(1)
    hora, dia = tiempo.hour.to_numpy(), tiempo.dayofweek.to_numpy()
    jornada = np.where((dia < 5) & (hora >= 8) & (hora < 18), 3.0, 0.5)
    filas = []
    for i in range(4):
        filas.append(pd.DataFrame({"timestamp": tiempo, "activity": "INDUSTRIAS MANUFACTURERAS",
                                   "subactivity": f"continua {i}", "valor": 100 * (1 + rng.normal(0, 0.01, len(tiempo))),
                                   "origen_valor": "observado"}))
        filas.append(pd.DataFrame({"timestamp": tiempo, "activity": "EDUCACIÓN",
                                   "subactivity": f"jornada {i}", "valor": 10 * jornada * (1 + rng.normal(0, 0.01, len(tiempo))),
                                   "origen_valor": "observado"}))
    corto = pd.DataFrame({"timestamp": tiempo[:100], "activity": "X", "subactivity": "corto",
                          "valor": 1.0, "origen_valor": "observado"})
    return pd.concat(filas + [corto], ignore_index=True)


def test_perfiles_sectoriales_y_grupos():
    from sklearn.cluster import KMeans

    perfiles, energia, registro = perfiles_sectoriales(_ciiu())
    assert registro["sectores_entrada"] == 9
    assert registro["sectores_pocos_datos"] == 1
    assert perfiles.shape == (8, 168)
    assert np.allclose(perfiles.mean(axis=1), 1.0)

    rasgos = indicadores_forma(perfiles)
    continuas = rasgos.xs("INDUSTRIAS MANUFACTURERAS", level=0)
    assert (continuas["factor_carga"] > 0.95).all()
    assert (continuas["domingo_sobre_laboral"].sub(1).abs() < 0.05).all()

    etiquetas = ordenar_por_tamano(KMeans(2, n_init=10, random_state=0).fit_predict(perfiles.to_numpy()))
    tabla = describir_sectores(perfiles, energia, pd.Series(etiquetas, index=perfiles.index))
    assert sorted(tabla["pct_sectores_manufactura"].tolist()) == [0.0, 100.0]
    assert tabla["pct_energia"].sum() == pytest.approx(100.0, abs=0.05)


def test_perfiles_sectoriales_ignoran_no_observados():
    ciiu = _ciiu()
    ciiu.loc[ciiu["subactivity"] == "continua 0", "origen_valor"] = "interpolado"
    perfiles, _, registro = perfiles_sectoriales(ciiu)
    assert ("INDUSTRIAS MANUFACTURERAS", "continua 0") not in perfiles.index
    assert registro["sectores_resultantes"] == 7
