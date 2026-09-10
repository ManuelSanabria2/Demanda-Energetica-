"""Tests del diagnostico de calidad, sobre series sinteticas con defectos conocidos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calidad.diagnostico import diagnosticar, resumen  # noqa: E402


# --- series de prueba -----------------------------------------------------


def serie_limpia(dias: int = 60, inicio: str = "2025-01-06") -> pd.DataFrame:
    """Serie horaria sin defectos, con forma diaria y semanal realista.

    Empieza un lunes para que los dias de la semana queden alineados. Lleva un
    ruido pequeno y determinista: sin el, cada grupo (dia de semana, hora)
    seria constante y no representaria a una serie real.
    """
    import random

    aleatorio = random.Random(42)
    momentos = pd.date_range(inicio, periods=dias * 24, freq="h")
    base = 7_000_000
    # Perfil horario: valle de madrugada, pico de tarde-noche.
    perfil = [0.85, 0.82, 0.80, 0.79, 0.79, 0.81, 0.86, 0.92, 0.97, 1.00,
              1.02, 1.03, 1.03, 1.04, 1.03, 1.02, 1.02, 1.05, 1.15, 1.18,
              1.16, 1.10, 1.02, 0.93]
    valores = [
        base
        * perfil[m.hour]
        * (0.93 if m.dayofweek >= 5 else 1.0)
        * (1 + aleatorio.gauss(0, 0.02))
        for m in momentos
    ]
    return pd.DataFrame(
        {
            "fecha_hora": momentos,
            "fuente": "prueba",
            "entidad": "Sistema",
            "valor_kwh": valores,
        }
    )


# --- completitud ----------------------------------------------------------


def test_serie_limpia_no_reporta_huecos():
    informe = diagnosticar(serie_limpia())

    comp = informe["completitud"]
    assert comp["horas_faltantes"] == 0
    assert comp["huecos"]["n_tramos"] == 0
    assert comp["huecos"]["forma"] == "sin huecos"
    assert comp["completitud_pct"] == 100.0


def test_distingue_huecos_dispersos_de_agrupados():
    """Tres horas sueltas y dos semanas seguidas no son el mismo problema."""
    marco = serie_limpia(dias=90)

    sueltas = [
        pd.Timestamp("2025-01-10 03:00"),
        pd.Timestamp("2025-01-20 14:00"),
        pd.Timestamp("2025-02-01 07:00"),
    ]
    bloque = pd.date_range("2025-03-01", periods=14 * 24, freq="h")
    fuera = set(sueltas) | set(bloque)
    marco = marco[~marco["fecha_hora"].isin(fuera)]

    huecos = diagnosticar(marco)["completitud"]["huecos"]

    assert huecos["n_tramos"] == 4
    assert huecos["tramos_por_clase"]["aislado"] == 3
    assert huecos["tramos_por_clase"]["largo"] == 1
    assert huecos["tramo_mas_largo"]["n_horas"] == 14 * 24
    assert huecos["forma"] == "agrupados"  # domina el bloque largo


def test_huecos_solo_dispersos_se_marcan_como_dispersos():
    marco = serie_limpia(dias=60)
    sueltas = pd.to_datetime(
        ["2025-01-10 03:00", "2025-01-20 14:00", "2025-02-01 07:00"]
    )
    marco = marco[~marco["fecha_hora"].isin(sueltas)]

    huecos = diagnosticar(marco)["completitud"]["huecos"]
    assert huecos["forma"] == "dispersos"
    assert huecos["horas_por_tramo_media"] == 1.0


def test_clasifica_los_tramos_por_duracion():
    marco = serie_limpia(dias=90)
    fuera = set()
    fuera |= {pd.Timestamp("2025-01-10 03:00")}                                   # aislado
    fuera |= set(pd.date_range("2025-01-15 02:00", periods=3, freq="h"))          # corto
    fuera |= set(pd.date_range("2025-02-01 00:00", periods=12, freq="h"))         # medio
    fuera |= set(pd.date_range("2025-03-01 00:00", periods=72, freq="h"))         # largo
    marco = marco[~marco["fecha_hora"].isin(fuera)]

    clases = diagnosticar(marco)["completitud"]["huecos"]["tramos_por_clase"]
    assert clases == {"aislado": 1, "corto": 1, "medio": 1, "largo": 1}


def test_detecta_timestamps_duplicados():
    marco = serie_limpia(dias=10)
    marco = pd.concat([marco, marco.iloc[[5, 6]]], ignore_index=True)

    dup = diagnosticar(marco)["completitud"]["duplicados"]
    assert dup["timestamps_repetidos"] == 2
    assert len(dup["ejemplos"]) == 2


def test_detecta_dias_sin_24_horas():
    """Sin horario de verano, un dia de 23 horas es un error de parseo."""
    marco = serie_limpia(dias=30)
    marco = marco[marco["fecha_hora"] != pd.Timestamp("2025-01-15 05:00")]

    dias = diagnosticar(marco)["completitud"]["dias_sin_24_horas"]
    assert dias["n_dias_interiores"] == 1
    assert dias["dias"][0]["dia"] == "2025-01-15"
    assert dias["dias"][0]["horas_distintas"] == 23
    assert dias["dias"][0]["es_borde_del_rango"] is False


def test_los_dias_borde_no_cuentan_como_defecto():
    """Un rango que empieza a media tarde trunca el primer dia: no es un fallo."""
    marco = serie_limpia(dias=10)
    marco = marco[marco["fecha_hora"] >= pd.Timestamp("2025-01-06 10:00")]

    dias = diagnosticar(marco)["completitud"]["dias_sin_24_horas"]
    assert dias["n_dias"] == 1
    assert dias["n_dias_interiores"] == 0
    assert dias["dias"][0]["es_borde_del_rango"] is True


# --- valores --------------------------------------------------------------


def test_cuenta_nulos_por_columna():
    marco = serie_limpia(dias=10)
    marco.loc[3:7, "valor_kwh"] = None
    marco.loc[1:2, "entidad"] = None

    nulos = diagnosticar(marco)["valores"]["nulos_por_columna"]
    assert nulos["valor_kwh"]["n_nulos"] == 5
    assert nulos["entidad"]["n_nulos"] == 2
    assert nulos["fuente"]["n_nulos"] == 0


def test_detecta_ceros_y_negativos():
    marco = serie_limpia(dias=10)
    marco.loc[10, "valor_kwh"] = 0.0
    marco.loc[11, "valor_kwh"] = 0.0
    marco.loc[12, "valor_kwh"] = -500.0

    fis = diagnosticar(marco)["valores"]["sospechosos_fisicos"]
    assert fis["n_ceros"] == 2
    assert fis["n_negativos"] == 1
    assert fis["ejemplos_negativos"][0]["valor"] == -500.0


def test_iqr_y_estacional_discrepan_en_el_caso_que_importa():
    """Un valor de hora punta puesto a las 3 a.m.

    El criterio global no lo ve raro, porque ese valor existe a diario a otra
    hora. El estacional si, porque a las 3 a.m. nunca se alcanza. Este es el
    motivo por el que el segundo criterio pesa mas que el primero.
    """
    marco = serie_limpia(dias=730)
    pico = marco["valor_kwh"].max()
    marco.loc[marco["fecha_hora"] == pd.Timestamp("2025-02-10 03:00"), "valor_kwh"] = pico

    valores = diagnosticar(marco)["valores"]
    momentos_iqr = {e["momento"] for e in valores["outliers_iqr"]["ejemplos"]}
    momentos_est = {e["momento"] for e in valores["outliers_estacionales"]["ejemplos"]}

    assert "2025-02-10 03:00:00" not in momentos_iqr
    assert "2025-02-10 03:00:00" in momentos_est


def test_el_estacional_apenas_marca_nada_en_una_serie_regular():
    est = diagnosticar(serie_limpia(dias=730))["valores"]["outliers_estacionales"]

    # Con historico suficiente la tasa de falsos positivos se mantiene baja.
    assert est["fiabilidad"] == "alta"
    assert est["pct"] < 0.2


def test_declina_juzgar_si_hay_poco_historico():
    """Con dos meses de datos la MAD no es fiable: mejor decirlo que inventarlo."""
    informe = diagnosticar(serie_limpia(dias=60))
    est = informe["valores"]["outliers_estacionales"]

    assert est["fiabilidad"] == "insuficiente"
    assert est["n_outliers"] == 0
    assert est["n_evaluables"] == 0
    assert any("cinco meses" in a for a in informe["avisos"])


def test_avisa_cuando_la_base_estadistica_es_corta():
    informe = diagnosticar(serie_limpia(dias=180))
    est = informe["valores"]["outliers_estacionales"]

    assert est["fiabilidad"] == "limitada"
    assert any("inflados" in a for a in informe["avisos"])


def test_la_mad_cero_no_esconde_un_atipico_solitario():
    """Un grupo constante con un unico valor extremo: la MAD vale 0 y lo taparia.

    Por eso, cuando la MAD es cero pero hay alguna dispersion, se usa la
    desviacion absoluta media, que si reacciona a un solo valor.
    """
    momentos = pd.date_range("2025-01-06", periods=365 * 24, freq="h")
    marco = pd.DataFrame({"fecha_hora": momentos, "valor_kwh": 1000.0})
    marco.loc[marco["fecha_hora"] == pd.Timestamp("2025-06-09 04:00"), "valor_kwh"] = 9e6

    est = diagnosticar(marco)["valores"]["outliers_estacionales"]

    assert est["n_outliers"] == 1
    assert est["n_por_desviacion_media"] > 0
    assert est["ejemplos"][0]["momento"] == "2025-06-09 04:00:00"


def test_el_estacional_reporta_la_hora_del_dia():
    marco = serie_limpia(dias=730)
    for dia in ("2025-01-13", "2025-01-27"):
        marco.loc[
            marco["fecha_hora"] == pd.Timestamp(f"{dia} 04:00"), "valor_kwh"
        ] = 20_000_000

    est = diagnosticar(marco)["valores"]["outliers_estacionales"]
    # Las claves son cadenas: JSON no admite claves numericas.
    assert est["por_hora_del_dia"].get("4", 0) >= 2
    assert est["ejemplos"][0]["dia_semana"] == "lunes"
    assert est["ejemplos"][0]["hora"] == 4
    assert est["ejemplos"][0]["valor"] == 20_000_000


def test_grupos_constantes_no_generan_falsos_positivos():
    """Una serie del todo constante no tiene dispersion: no se juzga, no se inventa."""
    momentos = pd.date_range("2025-01-06", periods=365 * 24, freq="h")
    marco = pd.DataFrame({"fecha_hora": momentos, "valor_kwh": 1000.0})

    est = diagnosticar(marco)["valores"]["outliers_estacionales"]
    assert est["n_outliers"] == 0
    assert est["n_no_evaluables"] == len(marco)


# --- estructura -----------------------------------------------------------


def test_reporta_cardinalidad_de_categoricas():
    marco = serie_limpia(dias=10)
    marco["mercado"] = ["Regulado", "No Regulado"] * (len(marco) // 2)

    card = diagnosticar(marco)["estructura"]["cardinalidad_categoricas"]
    assert card["mercado"]["n_distintos"] == 2
    assert set(card["mercado"]["todos_los_valores"]) == {"Regulado", "No Regulado"}
    assert card["entidad"]["n_distintos"] == 1


def test_reporta_rango_y_continuidad():
    marco = serie_limpia(dias=10)
    est = diagnosticar(marco)["estructura"]

    assert est["rango_temporal"]["dias_cubiertos"] == 10
    assert est["continuidad_pct"] == 100.0
    assert est["n_filas"] == 240


# --- tablas desagregadas --------------------------------------------------


def test_avisa_cuando_la_tabla_esta_desagregada():
    marco = serie_limpia(dias=5)
    duplicada = pd.concat(
        [marco.assign(Version="TX2"), marco.assign(Version="TXR")], ignore_index=True
    )

    informe = diagnosticar(duplicada)
    assert informe["meta"]["forma"] == "desagregada"
    assert informe["meta"]["filas_por_timestamp"] == 2.0
    assert any("desagregada" in a for a in informe["avisos"])


def test_con_clave_distingue_duplicado_real_de_desagregacion():
    marco = serie_limpia(dias=5)
    duplicada = pd.concat(
        [marco.assign(Version="TX2"), marco.assign(Version="TXR")], ignore_index=True
    )

    informe = diagnosticar(duplicada, columnas_clave=["fecha_hora", "Version"])
    dup = informe["completitud"]["duplicados"]

    assert dup["timestamps_repetidos"] == 120  # esperable: hay dos versiones
    assert dup["clave_repetida"] == 0  # pero la clave completa no se repite


# --- contrato de salida ---------------------------------------------------


def test_el_informe_es_serializable_a_json():
    marco = serie_limpia(dias=30)
    marco.loc[5, "valor_kwh"] = None
    marco.loc[6, "valor_kwh"] = -1.0
    marco = marco[marco["fecha_hora"] != pd.Timestamp("2025-01-15 05:00")]

    texto = json.dumps(diagnosticar(marco), ensure_ascii=False)
    assert json.loads(texto)["completitud"]["horas_faltantes"] == 1
    assert "NaN" not in texto  # NaN no es JSON valido


def test_el_resumen_es_texto_legible():
    texto = resumen(diagnosticar(serie_limpia(dias=30)))

    for encabezado in ("COMPLETITUD TEMPORAL", "VALORES", "ESTRUCTURA"):
        assert encabezado in texto
    assert "no corrige nada" in texto


def test_no_modifica_el_marco_de_entrada():
    marco = serie_limpia(dias=10)
    copia = marco.copy(deep=True)

    diagnosticar(marco)

    pd.testing.assert_frame_equal(marco, copia)


def test_marco_vacio_no_revienta():
    informe = diagnosticar(pd.DataFrame())
    assert informe["meta"]["vacio"] is True
    assert "vacio" in resumen(informe)


def test_columna_inexistente_da_un_error_claro():
    with pytest.raises(KeyError, match="valor"):
        diagnosticar(serie_limpia(dias=2), columna_valor="no_existe")


def test_detecta_los_nombres_de_columna_del_proyecto():
    """El proyecto usa fecha_hora/valor_kwh, timestamp/valor y timestamp/Valor."""
    marco = serie_limpia(dias=5)

    renombrado = marco.rename(columns={"fecha_hora": "timestamp", "valor_kwh": "Valor"})
    informe = diagnosticar(renombrado)

    assert informe["meta"]["columna_tiempo"] == "timestamp"
    assert informe["meta"]["columna_valor"] == "Valor"


def test_las_claves_numericas_del_informe_son_cadenas():
    """JSON no admite claves numericas, asi que el informe las serializa a texto."""
    marco = serie_limpia(dias=730)
    marco.loc[marco["fecha_hora"] == pd.Timestamp("2025-06-09 04:00"), "valor_kwh"] = 9e7

    est = diagnosticar(marco)["valores"]["outliers_estacionales"]
    assert all(isinstance(k, str) for k in est["por_hora_del_dia"])
