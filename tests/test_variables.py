"""Pruebas de fuga temporal de la tabla de modelado (src/variables). Sin red.

El riesgo que cubren no levanta ninguna excepcion: un rezago contado por
posicion sobre una rejilla con huecos, una media movil que incluye la hora que
se predice, una marca calculada con la demanda de la propia fila. Todo eso da
una tabla plausible y un error de validacion demasiado bueno para ser verdad.

El panel sintetico se construye con las mismas funciones que el real (limpiar,
calendario_horario, unir), asi que tiene exactamente sus columnas: si el panel
gana una columna, estas pruebas obligan a decidir si es variable.

Todas las pruebas de fuga corren para las dos versiones: rezago_minimo_h=24
(backtesting academico) y rezago_minimo_h=120 (disponibilidad real).
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

from ingesta.complementarias import calendario_horario, unir  # noqa: E402
from limpieza.limpiar import limpiar  # noqa: E402
from variables.construir import (  # noqa: E402
    BANDERAS_CALENDARIO,
    ErrorVariables,
    codificar_ciclicas,
    construir_tabla,
)

REZAGOS_MINIMOS = [24, 120]
VENTANA = 168
DIAS = 45
COLUMNAS_FUGA = ("atipico_global", "atipico_iqr_global", "atipico", "atipico_iqr", "z_estacional")


def panel_sintetico(dias: int = DIAS, semilla: int = 7) -> pd.DataFrame:
    """Panel con el esquema real: demanda limpia + clima + calendario colombiano.

    Del 10 de marzo al 23 de abril de 2025: incluye San Jose trasladado al
    lunes 24 (puente) y la Semana Santa, asi que las banderas no son todo 0.
    Cada hora tiene un valor distinto, para que un rezago desplazado una hora
    no pueda coincidir por casualidad.
    """
    horas = pd.date_range("2025-03-10", periods=dias * 24, freq="h", tz="America/Bogota")
    rng = np.random.default_rng(semilla)
    posicion = np.arange(len(horas))
    valor = (
        7.0e6
        + 1.5e6 * np.sin(2 * np.pi * horas.hour / 24)
        + 2.0e3 * posicion
        + rng.normal(0, 5.0e4, len(horas))
    )
    demanda = pd.DataFrame(
        {
            "fecha_hora": horas,
            "fuente": "xm",
            "metrica": "DemaReal",
            "entidad": "Sistema",
            "valor_kwh": valor,
            "version": None,
        }
    )
    limpio, _ = limpiar(demanda, metodo_imputacion="causal")
    clima = pd.DataFrame(
        {
            "fecha_hora": limpio["fecha_hora"],
            "temperatura_c": 20.0 + 5.0 * np.sin(2 * np.pi * (posicion - 6) / 24),
            "ciudades_disponibles": 4,
        }
    )
    calendario, _ = calendario_horario(horas[0].date(), horas[-1].date())
    panel, _ = unir(limpio, clima, calendario, exigir_cobertura_total=True)
    return panel


def demanda_mwh(panel: pd.DataFrame) -> pd.Series:
    """La demanda del panel en MWh, indexada por marca de tiempo."""
    return (panel.set_index("fecha_hora")["valor_kwh"] / 1000).sort_index()


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return panel_sintetico()


@pytest.fixture(scope="module", params=REZAGOS_MINIMOS, ids=lambda r: f"r{r}")
def version(request: pytest.FixtureRequest, panel: pd.DataFrame):
    tabla, registro = construir_tabla(panel, request.param)
    return request.param, tabla, registro


# ==========================================================================
# Rezagos: por marca de tiempo y nunca por debajo del minimo
# ==========================================================================


def test_los_rezagos_son_la_demanda_de_esa_marca_de_tiempo(panel, version):
    r, tabla, registro = version
    y = demanda_mwh(panel)
    marcas = pd.DatetimeIndex(tabla["fecha_hora"])

    for k in (24, 168):
        nombre = f"lag_{k}"
        if k < r:
            assert nombre not in tabla.columns
            assert nombre in [o["columna"] for o in registro["rezagos_omitidos"]]
            continue
        esperado = y.reindex(marcas - pd.Timedelta(hours=k))
        assert esperado.notna().all()
        np.testing.assert_array_equal(tabla[nombre].to_numpy(), esperado.to_numpy())


def test_ningun_rezago_por_debajo_del_minimo(version):
    r, tabla, _ = version
    rezagos = [int(c.split("_")[1]) for c in tabla.columns if c.startswith("lag_")]

    assert rezagos, "al menos lag_168 debe existir en las dos versiones"
    assert all(k >= r for k in rezagos)
    assert "lag_1" not in tabla.columns


def test_r120_omite_lag_24_y_lo_anota(panel):
    tabla, registro = construir_tabla(panel, 120)

    assert "lag_24" not in tabla.columns
    assert "lag_168" in tabla.columns
    [omitido] = registro["rezagos_omitidos"]
    assert omitido["columna"] == "lag_24"
    assert "120" in omitido["motivo"]


@pytest.mark.parametrize("rezago_minimo", [1, 12, 23])
def test_un_rezago_minimo_por_debajo_del_horizonte_falla(panel, rezago_minimo):
    with pytest.raises(ErrorVariables, match="horizonte"):
        construir_tabla(panel, rezago_minimo)


def test_el_orden_de_las_filas_no_cambia_el_resultado(panel):
    """Por marca de tiempo, no por posicion: barajar el panel no cambia nada."""
    ordenada, _ = construir_tabla(panel, 24)
    barajada, _ = construir_tabla(panel.sample(frac=1, random_state=3), 24)

    pd.testing.assert_frame_equal(ordenada, barajada)


def test_una_hora_faltante_hace_fallar_en_vez_de_contar_filas(panel):
    """Con un hueco, 168 filas atras ya no son 168 horas atras."""
    with pytest.raises(ErrorVariables, match="rejilla"):
        construir_tabla(panel.drop(index=500), 24)


def test_una_hora_repetida_hace_fallar(panel):
    with pytest.raises(ErrorVariables, match="repetidas"):
        construir_tabla(pd.concat([panel, panel.iloc[[500]]]), 24)


# ==========================================================================
# Media y desviacion moviles
# ==========================================================================


def test_media_y_desviacion_movil_recalculadas_a_mano(panel, version):
    """Solo con la demanda hasta t - rezago_minimo_h, las ultimas 168 horas."""
    r, tabla, _ = version
    y = demanda_mwh(panel)

    for t, media, desviacion in zip(tabla["fecha_hora"], tabla["ma_168"], tabla["std_168"]):
        limite = t - pd.Timedelta(hours=r)
        conocida = y[y.index <= limite]
        ventana = conocida.iloc[-VENTANA:].to_numpy()

        assert conocida.index[-1] == limite
        assert conocida.index[-VENTANA] == limite - pd.Timedelta(hours=VENTANA - 1)
        assert media == pytest.approx(np.mean(ventana), rel=1e-10)
        assert desviacion == pytest.approx(np.std(ventana, ddof=1), rel=1e-8)


# ==========================================================================
# Perturbacion: el futuro no cambia el pasado
# ==========================================================================


@pytest.mark.parametrize("r", REZAGOS_MINIMOS, ids=lambda r: f"r{r}")
def test_alterar_la_demanda_futura_no_cambia_las_filas_anteriores(panel, r):
    base, _ = construir_tabla(panel, r)
    corte = base["fecha_hora"].iloc[len(base) // 3]

    alterado = panel.copy()
    alterado.loc[alterado["fecha_hora"] == corte, "valor_kwh"] *= 10
    alterado.loc[alterado["fecha_hora"] > corte, "valor_kwh"] += 1.0e9
    perturbada, _ = construir_tabla(alterado, r)

    # Las filas anteriores al corte, identicas en todas sus columnas.
    antes_base = base[base["fecha_hora"] < corte]
    antes_pert = perturbada[perturbada["fecha_hora"] < corte]
    pd.testing.assert_frame_equal(antes_base, antes_pert, check_exact=True)

    # Mas fuerte: hasta corte + r las variables tampoco cambian, porque la fila
    # t solo usa demanda de t - r o antes. Solo cambia el objetivo.
    variables = [c for c in base.columns if c not in ("fecha_hora", "demanda_mwh")]
    tramo = (base["fecha_hora"] >= corte) & (base["fecha_hora"] < corte + pd.Timedelta(hours=r))
    assert tramo.sum() == r
    pd.testing.assert_frame_equal(
        base.loc[tramo, variables], perturbada.loc[tramo, variables], check_exact=True
    )

    # Y la perturbacion se ve donde debe: no es una prueba que pase sola.
    en = base["fecha_hora"] == corte + pd.Timedelta(hours=r)
    assert (perturbada.loc[en, "ma_168"] != base.loc[en, "ma_168"]).all()
    en_168 = base["fecha_hora"] == corte + pd.Timedelta(hours=168)
    assert (perturbada.loc[en_168, "lag_168"] != base.loc[en_168, "lag_168"]).all()


# ==========================================================================
# Codificacion ciclica
# ==========================================================================


def _distancia(tabla: pd.DataFrame, nombre: str, i: int, j: int) -> float:
    a = tabla.loc[i, [f"{nombre}_sin", f"{nombre}_cos"]].to_numpy(dtype=float)
    b = tabla.loc[j, [f"{nombre}_sin", f"{nombre}_cos"]].to_numpy(dtype=float)
    return float(np.hypot(*(a - b)))


def test_la_hora_23_y_la_hora_0_quedan_contiguas(version):
    _, tabla, _ = version
    horas = tabla["fecha_hora"].dt.hour
    i23 = horas[horas == 23].index[0]
    i0, i1, i12 = i23 + 1, i23 + 2, i23 + 13
    assert (horas[i0], horas[i1], horas[i12]) == (0, 1, 12)

    paso = 2 * np.sin(np.pi / 24)
    assert _distancia(tabla, "hora", i23, i0) == pytest.approx(paso)
    assert _distancia(tabla, "hora", i0, i1) == pytest.approx(paso)
    assert _distancia(tabla, "hora", i0, i12) == pytest.approx(2.0)


def test_domingo_y_lunes_y_diciembre_y_enero_quedan_contiguos():
    marcas = pd.to_datetime(
        ["2025-12-28 12:00", "2025-12-29 12:00", "2026-01-01 12:00", "2025-11-30 12:00"]
    ).tz_localize("America/Bogota")
    tabla, _ = codificar_ciclicas(pd.DataFrame({"fecha_hora": marcas}))

    # 2025-12-28 es domingo y 2025-12-29 lunes.
    assert _distancia(tabla, "dia_semana", 0, 1) == pytest.approx(2 * np.sin(np.pi / 7))
    # Diciembre -> enero, igual de cerca que noviembre -> diciembre.
    assert _distancia(tabla, "mes", 1, 2) == pytest.approx(2 * np.sin(np.pi / 12))
    assert _distancia(tabla, "mes", 3, 1) == pytest.approx(2 * np.sin(np.pi / 12))


def test_seno_y_coseno_estan_en_el_circulo_unidad(version):
    _, tabla, _ = version
    for nombre in ("hora", "dia_semana", "mes"):
        radio = tabla[f"{nombre}_sin"] ** 2 + tabla[f"{nombre}_cos"] ** 2
        np.testing.assert_allclose(radio, 1.0, rtol=0, atol=1e-12)


# ==========================================================================
# Columnas
# ==========================================================================


def test_la_tabla_no_contiene_las_marcas_con_fuga(version):
    _, tabla, registro = version
    for columna in COLUMNAS_FUGA:
        assert columna not in tabla.columns
        assert columna in registro["columnas_excluidas"]


def test_solo_quedan_la_marca_de_tiempo_y_columnas_numericas(version):
    _, tabla, _ = version
    resto = tabla.drop(columns="fecha_hora")
    assert all(pd.api.types.is_numeric_dtype(t) for t in resto.dtypes)
    assert not tabla.isna().any().any()


def test_una_columna_nueva_sin_clasificar_hace_fallar(panel):
    with pytest.raises(ErrorVariables, match="sin clasificar"):
        construir_tabla(panel.assign(columna_nueva=1.0), 24)


def test_el_objetivo_es_la_demanda_en_mwh_sin_escalar(panel, version):
    _, tabla, _ = version
    original = panel.set_index("fecha_hora").loc[tabla["fecha_hora"]]

    np.testing.assert_array_equal(tabla["demanda_mwh"], original["valor_kwh"].to_numpy() / 1000)
    np.testing.assert_array_equal(tabla["temperatura_c"], original["temperatura_c"].to_numpy())
    assert "valor_kwh" not in tabla.columns


def test_las_banderas_pasan_a_0_1_sin_cambiar_su_valor(panel, version):
    _, tabla, _ = version
    original = panel.set_index("fecha_hora").loc[tabla["fecha_hora"]]

    for bandera in BANDERAS_CALENDARIO:
        assert tabla[bandera].dtype == "int8"
        np.testing.assert_array_equal(tabla[bandera], original[bandera].astype(int).to_numpy())
    assert tabla["es_festivo"].sum() > 0 and tabla["es_puente"].sum() > 0


# ==========================================================================
# Filas: solo cambian por las eliminaciones documentadas
# ==========================================================================


def test_las_filas_cambian_solo_por_las_eliminaciones_documentadas(panel, version):
    r, tabla, registro = version
    balance = registro["balance_filas"]

    assert balance["panel"] == len(panel)
    assert len(tabla) == (
        len(panel) - balance["arranque_sin_historico"] - balance["publicacion_parcial"]
    )
    # La ventana movil sobre la serie desplazada es lo que mas historico pide.
    assert balance["arranque_sin_historico"] == r + VENTANA - 1
    assert balance["publicacion_parcial"] == 5 * 24

    # Lo que queda es exactamente el tramo del panel entre las dos eliminaciones.
    marcas = pd.DatetimeIndex(panel["fecha_hora"]).sort_values()
    esperadas = marcas[balance["arranque_sin_historico"] : len(marcas) - balance["publicacion_parcial"]]
    np.testing.assert_array_equal(tabla["fecha_hora"].to_numpy(), esperadas.to_numpy())


def test_se_descartan_los_ultimos_cinco_dias_de_calendario(panel, version):
    _, tabla, registro = version
    ultimo_panel = panel["fecha_hora"].max().date()

    assert tabla["fecha_hora"].max().date() == ultimo_panel - pd.Timedelta(days=5)
    paso = next(p for p in registro["operaciones"] if p["operacion"] == "descartar_publicacion_parcial")
    assert len(paso["detalle"]["dias_eliminados"]) == 5


def test_un_nan_en_mitad_de_la_serie_no_se_elimina_se_reporta(panel):
    """Solo se quita el arranque: un hueco largo es decision del modelado."""
    con_hueco = panel.copy()
    con_hueco.loc[600, "valor_kwh"] = np.nan

    tabla, registro = construir_tabla(con_hueco, 24)

    balance = registro["balance_filas"]
    assert len(tabla) == len(panel) - balance["arranque_sin_historico"] - balance["publicacion_parcial"]
    paso = next(p for p in registro["operaciones"] if p["operacion"] == "eliminar_arranque")
    nan = paso["detalle"]["nan_fuera_del_arranque"]
    assert nan["demanda_mwh"] == 1 and nan["lag_24"] == 1 and nan["ma_168"] == VENTANA


# ==========================================================================
# Script: escribe las dos versiones y el registro
# ==========================================================================


def test_el_script_escribe_las_dos_versiones_y_su_registro(panel, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "construir_variables", RAIZ / "scripts" / "construir_variables.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    ruta_panel = tmp_path / "panel_modelado.parquet"
    panel.to_parquet(ruta_panel, index=False)

    assert script.main(["--panel", str(ruta_panel), "--salida", str(tmp_path)]) == 0

    with (tmp_path / "registro_variables.json").open(encoding="utf-8") as origen:
        registro = json.load(origen)
    assert set(registro["versiones"]) == {"r24", "r120"}
    for r in REZAGOS_MINIMOS:
        leida = pd.read_parquet(tmp_path / f"tabla_modelado_r{r}.parquet")
        esperada, _ = construir_tabla(panel, r)
        pd.testing.assert_frame_equal(leida, esperada)
        assert len(registro["versiones"][f"r{r}"]["sha256"]) == 64
