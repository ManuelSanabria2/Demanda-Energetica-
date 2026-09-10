"""Tests de la limpieza trazable."""

from __future__ import annotations

import datetime as dt
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from limpieza import limpiar as L  # noqa: E402
from limpieza.limpiar import ErrorLimpieza, limpiar, procedencia, resumen  # noqa: E402


# --- series de prueba -----------------------------------------------------


def serie(dias: int = 400, inicio: str = "2025-01-06") -> pd.DataFrame:
    """Serie horaria realista, con ruido determinista."""
    aleatorio = random.Random(7)
    momentos = pd.date_range(inicio, periods=dias * 24, freq="h")
    perfil = [0.85, 0.82, 0.80, 0.79, 0.79, 0.81, 0.86, 0.92, 0.97, 1.00,
              1.02, 1.03, 1.03, 1.04, 1.03, 1.02, 1.02, 1.05, 1.15, 1.18,
              1.16, 1.10, 1.02, 0.93]
    return pd.DataFrame(
        {
            "fecha_hora": momentos,
            "entidad": "Sistema",
            "valor_kwh": [
                7e6 * perfil[m.hour] * (0.93 if m.dayofweek >= 5 else 1.0)
                * (1 + aleatorio.gauss(0, 0.02))
                for m in momentos
            ],
        }
    )


def quitar(marco: pd.DataFrame, inicio: str, horas: int) -> pd.DataFrame:
    """Elimina un tramo de horas consecutivas."""
    fuera = pd.date_range(inicio, periods=horas, freq="h")
    return marco[~marco["fecha_hora"].isin(fuera)].reset_index(drop=True)


# --- 1. normalizacion de esquema ------------------------------------------


def test_los_nombres_pasan_a_snake_case():
    marco = serie(dias=30).rename(
        columns={"fecha_hora": "FechaHora", "valor_kwh": "Valor"}
    )
    marco["CodigoSICAgente"] = "ENIC"

    limpio, registro = L.normalizar_esquema(marco)

    assert "fecha_hora" in limpio.columns
    assert "valor" in limpio.columns
    assert "codigo_sic_agente" in limpio.columns
    assert registro["detalle"]["renombradas"]["CodigoSICAgente"] == "codigo_sic_agente"


def test_la_marca_de_tiempo_queda_en_utc_menos_5():
    limpio, registro = L.normalizar_esquema(serie(dias=10))

    # America/Bogota es exactamente -05:00 todo el ano: Colombia no aplica DST.
    assert str(limpio["fecha_hora"].dt.tz) == "America/Bogota"
    assert registro["detalle"]["zona_horaria"] == "America/Bogota"
    assert limpio["fecha_hora"].iloc[0].utcoffset() == dt.timedelta(hours=-5)
    # Localizar, no convertir: la hora de reloj no debe moverse.
    assert limpio["fecha_hora"].iloc[0].hour == 0


def test_localizar_no_desplaza_las_horas():
    marco = serie(dias=3)
    original = marco["fecha_hora"].dt.hour.tolist()

    limpio, _ = L.normalizar_esquema(marco)

    assert limpio["fecha_hora"].dt.hour.tolist() == original


def test_los_valores_no_numericos_pasan_a_nan_y_se_reportan():
    marco = serie(dias=5)
    marco["valor_kwh"] = marco["valor_kwh"].astype(object)
    marco.loc[3, "valor_kwh"] = "no es un numero"

    limpio, registro = L.normalizar_esquema(marco)

    assert pd.isna(limpio["valor_kwh"].iloc[3])
    assert registro["detalle"]["valores_no_numericos_a_nan"] == 1


def test_una_colision_de_nombres_falla_en_vez_de_perder_una_columna():
    marco = serie(dias=2)
    marco["Entidad"] = "otra"  # colisiona con "entidad"

    with pytest.raises(ErrorLimpieza, match="colisiona"):
        L.normalizar_esquema(marco)


# --- 2. deduplicacion -----------------------------------------------------


def test_deduplica_conservando_el_mas_reciente_y_lo_reporta():
    marco = serie(dias=5)
    repetida = marco.iloc[[10]].copy()
    repetida["valor_kwh"] = 999.0
    marco = pd.concat([marco, repetida], ignore_index=True)

    limpio, registro = L.deduplicar(marco)

    assert len(limpio) == len(marco) - 1
    assert registro["detalle"]["eliminadas"] == 1
    # Se conserva la ultima, que es la que se anadio.
    fila = limpio[limpio["fecha_hora"] == marco["fecha_hora"].iloc[10]]
    assert fila["valor_kwh"].iloc[0] == 999.0


def test_la_recencia_puede_venir_de_una_columna():
    """Con dos descargas de la misma hora, gana la descargada mas tarde."""
    marco = serie(dias=30)
    marco["descargado"] = pd.Timestamp("2026-01-01")

    vieja = marco.iloc[[100]].copy()
    vieja["valor_kwh"] = 111.0
    vieja["descargado"] = pd.Timestamp("2025-12-01")  # anterior
    nueva = marco.iloc[[100]].copy()
    nueva["valor_kwh"] = 999.0
    nueva["descargado"] = pd.Timestamp("2026-02-01")  # posterior

    # Se anaden desordenadas a proposito: manda la columna, no el orden.
    revuelto = pd.concat([nueva, marco, vieja], ignore_index=True)

    limpio, registro = L.deduplicar(revuelto, columna_recencia="descargado")

    fila = limpio[limpio["fecha_hora"] == marco["fecha_hora"].iloc[100]]
    assert len(fila) == 1
    assert fila["valor_kwh"].iloc[0] == 999.0
    assert registro["detalle"]["columna_recencia"] == "descargado"
    assert registro["detalle"]["eliminadas"] == 2


def test_se_niega_a_deduplicar_una_tabla_desagregada():
    """En SIMEM hay cientos de filas legitimas por hora: esto borraria el 99%."""
    marco = serie(dias=5)
    desagregada = pd.concat(
        [marco.assign(version=v) for v in ("TX2", "TX3", "TXR", "TXF")],
        ignore_index=True,
    )

    with pytest.raises(ErrorLimpieza, match="desagregado"):
        L.deduplicar(desagregada)


def test_la_desagregada_se_puede_forzar_si_se_pide_explicitamente():
    marco = serie(dias=5)
    desagregada = pd.concat([marco, marco], ignore_index=True)

    limpio, _ = L.deduplicar(desagregada, permitir_desagregada=True)
    assert len(limpio) == len(marco)


# --- 3. huecos ------------------------------------------------------------


def test_interpola_solo_los_huecos_cortos():
    marco = quitar(serie(dias=60), "2025-02-01 02:00", horas=3)

    limpio, registro = L.completar_rejilla(marco)

    assert registro["detalle"]["horas_interpoladas"] == 3
    assert registro["detalle"]["horas_dejadas_como_nan"] == 0
    tramo = limpio[
        limpio["fecha_hora"].between("2025-02-01 02:00", "2025-02-01 04:00")
    ]
    assert tramo["valor_kwh"].notna().all()
    assert (tramo["origen_valor"] == "interpolado").all()
    assert tramo["imputado"].all()


def test_no_inventa_dias_completos():
    """Dos semanas seguidas se quedan como NaN, no se rellenan."""
    marco = quitar(serie(dias=90), "2025-03-01 00:00", horas=14 * 24)

    limpio, registro = L.completar_rejilla(marco)

    assert registro["detalle"]["horas_interpoladas"] == 0
    assert registro["detalle"]["horas_dejadas_como_nan"] == 14 * 24
    tramo = limpio[limpio["fecha_hora"].between("2025-03-01", "2025-03-14 23:00")]
    assert tramo["valor_kwh"].isna().all()
    assert (tramo["origen_valor"] == "faltante").all()
    assert not tramo["imputado"].any()


def test_el_limite_de_interpolacion_es_exactamente_tres_horas():
    corto = quitar(serie(dias=60), "2025-02-01 02:00", horas=3)
    largo = quitar(serie(dias=60), "2025-02-01 02:00", horas=4)

    _, reg_corto = L.completar_rejilla(corto)
    _, reg_largo = L.completar_rejilla(largo)

    assert reg_corto["detalle"]["horas_interpoladas"] == 3
    assert reg_largo["detalle"]["horas_interpoladas"] == 0
    assert reg_largo["detalle"]["horas_dejadas_como_nan"] == 4


def test_registra_la_longitud_del_hueco_de_cada_fila():
    marco = quitar(serie(dias=60), "2025-02-01 02:00", horas=2)

    limpio, _ = L.completar_rejilla(marco)

    fila = limpio[limpio["fecha_hora"] == pd.Timestamp("2025-02-01 02:00")]
    assert fila["hueco_horas"].iloc[0] == 2
    observada = limpio[limpio["fecha_hora"] == pd.Timestamp("2025-02-01 06:00")]
    assert observada["hueco_horas"].iloc[0] == 0


def test_los_tramos_no_imputados_quedan_listados_para_auditar():
    marco = quitar(serie(dias=90), "2025-03-01 00:00", horas=48)

    _, registro = L.completar_rejilla(marco)
    tramos = registro["detalle"]["tramos_no_imputados"]

    assert len(tramos) == 1
    assert tramos[0]["n_horas"] == 48
    assert tramos[0]["inicio"].startswith("2025-03-01")


def test_no_extrapola_en_los_extremos():
    """Un hueco corto al principio no tiene con que interpolarse."""
    marco = serie(dias=30)
    completo, _ = L.completar_rejilla(marco)
    sin_inicio = completo.iloc[2:].copy()  # se quitan las dos primeras horas

    limpio, registro = L.completar_rejilla(sin_inicio)

    # La rejilla arranca en la primera hora presente: no se inventa hacia atras.
    assert registro["detalle"]["filas_creadas_para_completar_rejilla"] == 0
    assert limpio["valor_kwh"].notna().all()


# --- 4. atipicos ----------------------------------------------------------


def test_marca_atipicos_sin_eliminarlos():
    marco = serie(dias=400)
    objetivo = pd.Timestamp("2025-06-09 04:00")
    marco.loc[marco["fecha_hora"] == objetivo, "valor_kwh"] = 20_000_000

    limpio, registro = L.marcar_atipicos(marco)

    assert len(limpio) == len(marco)  # no se elimino nada
    fila = limpio[limpio["fecha_hora"] == objetivo]
    assert fila["atipico"].iloc[0]
    assert fila["valor_kwh"].iloc[0] == 20_000_000  # el valor sigue intacto
    assert registro["detalle"]["n_atipicos_estacionales"] >= 1
    assert "no a la limpieza" in registro["detalle"]["nota"]


def test_el_criterio_estacional_y_el_iqr_van_en_columnas_distintas():
    marco = serie(dias=400)
    limpio, _ = L.marcar_atipicos(marco)

    assert "atipico" in limpio.columns
    assert "atipico_iqr" in limpio.columns
    assert "z_estacional" in limpio.columns


def test_los_no_evaluables_no_se_marcan_como_normales():
    marco = serie(dias=30)  # poco historico: grupos pequenos
    limpio, registro = L.marcar_atipicos(marco)

    assert registro["detalle"]["fiabilidad"] == "insuficiente"
    assert not limpio["atipico_evaluable"].any()
    assert limpio["atipico"].sum() == 0


# --- 5. periodo atipico ---------------------------------------------------


def test_marca_la_pandemia():
    marco = serie(dias=400, inicio="2020-01-06")

    limpio, registro = L.marcar_periodo_atipico(marco)

    marcadas = limpio[limpio["periodo_atipico"]]
    assert marcadas["fecha_hora"].min().date() >= dt.date(2020, 3, 1)
    assert marcadas["fecha_hora"].max().date() <= dt.date(2020, 12, 31)
    assert (marcadas["etiqueta_periodo"] == "pandemia_2020").all()
    assert registro["detalle"]["n_filas_marcadas"] == len(marcadas)
    assert len(limpio) == len(marco)  # no se elimino nada


def test_fuera_del_periodo_no_se_marca_nada_y_se_dice():
    marco = serie(dias=30, inicio="2025-01-06")

    limpio, registro = L.marcar_periodo_atipico(marco)

    assert not limpio["periodo_atipico"].any()
    assert registro["detalle"]["cubre_el_periodo"] is False


# --- orquestacion y registro ---------------------------------------------


def test_el_registro_cubre_todas_las_operaciones():
    marco = quitar(serie(dias=400), "2025-02-01 02:00", horas=2)
    _, registro = limpiar(marco)

    operaciones = [o["operacion"] for o in registro["operaciones"]]
    assert operaciones == [
        "normalizar_esquema",
        "deduplicar",
        "completar_rejilla",
        "marcar_atipicos",
        "marcar_periodo_atipico",
    ]
    for paso in registro["operaciones"]:
        assert paso["criterio"]
        assert "filas_afectadas" in paso
        assert "momento" in paso


def test_el_registro_es_serializable_a_json():
    marco = quitar(serie(dias=400), "2025-03-01 00:00", horas=48)
    _, registro = limpiar(marco)

    texto = json.dumps(registro, ensure_ascii=False, default=str)
    assert json.loads(texto)["filas_finales"] > 0


def test_el_marco_final_lleva_las_columnas_de_procedencia():
    marco = quitar(serie(dias=400), "2025-02-01 02:00", horas=2)
    limpio, _ = limpiar(marco)

    for columna in ("origen_valor", "imputado", "hueco_horas", "atipico",
                    "periodo_atipico"):
        assert columna in limpio.columns


def test_la_limpieza_no_pierde_horas():
    marco = quitar(serie(dias=200), "2025-03-01 00:00", horas=48)
    limpio, registro = limpiar(marco)

    esperadas = pd.date_range(
        limpio["fecha_hora"].min(), limpio["fecha_hora"].max(), freq="h"
    )
    assert len(limpio) == len(esperadas)
    assert registro["procedencia"]["pct_observado"] > 95


# --- trazabilidad de una celda -------------------------------------------


def test_procedencia_explica_un_valor_observado():
    limpio, _ = limpiar(serie(dias=400))
    detalle = procedencia(limpio, "2025-02-01 10:00")

    assert detalle["encontrado"]
    assert detalle["origen_valor"] == "observado"
    assert detalle["imputado"] is False
    assert "no se modifico" in detalle["explicacion"]


def test_procedencia_explica_un_valor_interpolado():
    marco = quitar(serie(dias=400), "2025-02-01 02:00", horas=2)
    limpio, _ = limpiar(marco)

    detalle = procedencia(limpio, "2025-02-01 02:00")

    assert detalle["origen_valor"] == "interpolado"
    assert detalle["imputado"] is True
    assert detalle["hueco_horas"] == 2
    assert detalle["valor"] is not None
    assert "interpolacion temporal" in detalle["explicacion"]


def test_procedencia_explica_un_hueco_no_imputado():
    marco = quitar(serie(dias=400), "2025-03-01 00:00", horas=48)
    limpio, _ = limpiar(marco)

    detalle = procedencia(limpio, "2025-03-01 05:00")

    assert detalle["origen_valor"] == "faltante"
    assert detalle["valor"] is None
    assert detalle["hueco_horas"] == 48
    assert "en vez de inventarlo" in detalle["explicacion"]


def test_procedencia_de_una_hora_inexistente():
    limpio, _ = limpiar(serie(dias=30))
    detalle = procedencia(limpio, "1999-01-01 00:00")

    assert detalle["encontrado"] is False


# --- persistencia ---------------------------------------------------------


def test_guardar_escribe_parquet_y_registro(tmp_path: Path):
    marco = quitar(serie(dias=200), "2025-02-01 02:00", horas=2)
    limpio, registro = limpiar(marco)

    rutas = L.guardar(limpio, registro, "prueba", directorio=tmp_path)

    releido = pd.read_parquet(rutas["datos"])
    assert len(releido) == len(limpio)
    # La zona sobrevive al Parquet, nombre incluido.
    assert str(releido["fecha_hora"].dt.tz) == "America/Bogota"
    assert releido["fecha_hora"].iloc[0].utcoffset() == dt.timedelta(hours=-5)
    assert releido["fecha_hora"].iloc[0] == limpio["fecha_hora"].iloc[0]

    with open(rutas["registro"], encoding="utf-8") as origen:
        bitacora = json.load(origen)
    assert bitacora["limpiezas"][0]["conjunto"] == "prueba"


def test_el_registro_de_limpieza_se_acumula(tmp_path: Path):
    limpio, registro = limpiar(serie(dias=200))

    L.guardar(limpio, registro, "uno", directorio=tmp_path)
    L.guardar(limpio, registro, "dos", directorio=tmp_path)

    with open(tmp_path / L.NOMBRE_REGISTRO, encoding="utf-8") as origen:
        bitacora = json.load(origen)
    assert [e["conjunto"] for e in bitacora["limpiezas"]] == ["uno", "dos"]


def test_el_resumen_es_legible():
    marco = quitar(serie(dias=400), "2025-02-01 02:00", horas=2)
    _, registro = limpiar(marco)
    texto = resumen(registro)

    assert "REGISTRO DE LIMPIEZA" in texto
    assert "PROCEDENCIA DE LOS VALORES" in texto
    assert "solo marcado" in texto


def test_marco_vacio_falla_con_mensaje_claro():
    with pytest.raises(ErrorLimpieza, match="vacio"):
        limpiar(pd.DataFrame())
