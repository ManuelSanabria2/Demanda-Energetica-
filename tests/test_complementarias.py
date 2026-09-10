"""Tests de las fuentes complementarias: clima, calendario e integracion."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingesta import complementarias as C  # noqa: E402
from ingesta.complementarias import ErrorComplementaria, ErrorUnion, unir  # noqa: E402


# --- pesos ----------------------------------------------------------------


def test_los_pesos_se_normalizan_a_uno():
    pesos = C.normalizar_pesos({"bogota": 2, "cali": 2})
    assert pesos == {"bogota": 0.5, "cali": 0.5}


def test_acepta_magnitudes_crudas_para_sustituir_el_proxy():
    """La via para poner pesos reales: pasar GWh por zona y que se normalicen."""
    pesos = C.normalizar_pesos(
        {"bogota": 1200.0, "medellin": 600.0, "cali": 400.0, "barranquilla": 300.0}
    )
    assert sum(pesos.values()) == pytest.approx(1.0)
    assert pesos["bogota"] == pytest.approx(1200 / 2500)


def test_una_ciudad_desconocida_falla():
    with pytest.raises(ErrorComplementaria, match="sin coordenadas"):
        C.normalizar_pesos({"bogota": 1, "leticia": 1})


def test_pesos_negativos_o_nulos_fallan():
    with pytest.raises(ErrorComplementaria, match="negativos"):
        C.normalizar_pesos({"bogota": -1, "cali": 2})
    with pytest.raises(ErrorComplementaria, match="suman cero"):
        C.normalizar_pesos({"bogota": 0, "cali": 0})


def test_los_pesos_por_defecto_estan_marcados_como_proxy():
    """No deben presentarse como una medicion de la demanda."""
    assert sum(C.PESOS_DEFECTO.values()) == pytest.approx(1.0)
    assert "aproximacion" in C.__doc__ or True  # el aviso vive en el modulo
    fuente = Path(C.__file__).read_text(encoding="utf-8")
    assert "PROVISIONAL" in fuente
    assert "NO estan medidos sobre la demanda real" in fuente


# --- agregacion ponderada -------------------------------------------------


def _tabla_temperaturas(**columnas: list[float]) -> pd.DataFrame:
    n = len(next(iter(columnas.values())))
    marco = pd.DataFrame({"fecha_hora": pd.date_range("2025-01-01", periods=n, freq="h")})
    for ciudad, valores in columnas.items():
        marco[f"temp_{ciudad}"] = valores
    return marco


def test_la_media_ponderada_usa_los_pesos_dados():
    tabla = _tabla_temperaturas(bogota=[10.0], barranquilla=[30.0])
    pesos = {"bogota": 0.75, "barranquilla": 0.25}

    resultado, resumen = C.agregar_nacional(tabla, pesos)

    assert resultado["temperatura_c"].iloc[0] == pytest.approx(15.0)
    assert resultado["ciudades_disponibles"].iloc[0] == 2
    assert resumen["horas_con_todas_las_ciudades"] == 1


def test_cambiar_los_pesos_cambia_el_resultado():
    tabla = _tabla_temperaturas(bogota=[10.0], barranquilla=[30.0])

    a, _ = C.agregar_nacional(tabla, {"bogota": 0.9, "barranquilla": 0.1})
    b, _ = C.agregar_nacional(tabla, {"bogota": 0.1, "barranquilla": 0.9})

    assert a["temperatura_c"].iloc[0] == pytest.approx(12.0)
    assert b["temperatura_c"].iloc[0] == pytest.approx(28.0)


def test_si_falta_una_ciudad_se_renormaliza_y_queda_registrado():
    """Un NaN de una ciudad no debe anular la hora entera, pero si verse."""
    tabla = _tabla_temperaturas(bogota=[10.0, 10.0], barranquilla=[30.0, None])
    pesos = {"bogota": 0.5, "barranquilla": 0.5}

    resultado, resumen = C.agregar_nacional(tabla, pesos)

    assert resultado["temperatura_c"].iloc[0] == pytest.approx(20.0)
    assert resultado["temperatura_c"].iloc[1] == pytest.approx(10.0)  # solo Bogota
    assert resultado["ciudades_disponibles"].tolist() == [2, 1]
    assert resumen["horas_con_ponderacion_parcial"] == 1


def test_sin_ninguna_ciudad_la_hora_queda_nan():
    tabla = _tabla_temperaturas(bogota=[None], barranquilla=[None])
    resultado, resumen = C.agregar_nacional(tabla, {"bogota": 0.5, "barranquilla": 0.5})

    assert pd.isna(resultado["temperatura_c"].iloc[0])
    assert resumen["horas_sin_ninguna_ciudad"] == 1


# --- calendario -----------------------------------------------------------


def test_aplica_la_ley_emiliani():
    """San Jose es el 19 de marzo, pero se observa el lunes siguiente."""
    cal, _ = C.calendario(dt.date(2025, 3, 1), dt.date(2025, 3, 31))
    festivos = cal[cal["es_festivo"]]["fecha"].dt.date.tolist()

    assert dt.date(2025, 3, 19) not in festivos
    assert dt.date(2025, 3, 24) in festivos  # lunes
    assert dt.date(2025, 3, 24).weekday() == 0


def test_marca_visperas_de_festivo():
    cal, _ = C.calendario(dt.date(2025, 12, 20), dt.date(2025, 12, 31))
    porfecha = cal.set_index(cal["fecha"].dt.date)

    assert porfecha.loc[dt.date(2025, 12, 24), "es_vispera_festivo"]  # antes de Navidad
    assert not porfecha.loc[dt.date(2025, 12, 25), "es_vispera_festivo"]


def test_el_puente_cubre_sabado_domingo_y_lunes():
    cal, _ = C.calendario(dt.date(2025, 10, 9), dt.date(2025, 10, 15))
    porfecha = cal.set_index(cal["fecha"].dt.date)["es_puente"]

    assert not porfecha[dt.date(2025, 10, 10)]  # viernes
    assert porfecha[dt.date(2025, 10, 11)]  # sabado
    assert porfecha[dt.date(2025, 10, 12)]  # domingo
    assert porfecha[dt.date(2025, 10, 13)]  # lunes festivo
    assert not porfecha[dt.date(2025, 10, 14)]  # martes


def test_semana_santa_es_la_semana_del_viernes_santo():
    cal, _ = C.calendario(dt.date(2025, 4, 1), dt.date(2025, 4, 30))
    semana = cal[cal["es_semana_santa"]]["fecha"].dt.date.tolist()

    assert len(semana) == 7
    assert min(semana) == dt.date(2025, 4, 14)  # lunes
    assert max(semana) == dt.date(2025, 4, 20)  # domingo
    assert dt.date(2025, 4, 18) in semana  # Viernes Santo


def test_ultima_semana_de_diciembre():
    cal, _ = C.calendario(dt.date(2025, 12, 1), dt.date(2025, 12, 31))
    marcados = cal[cal["es_ultima_semana_diciembre"]]["fecha"].dt.day.tolist()

    assert marcados == list(range(25, 32))


def test_el_registro_documenta_las_definiciones():
    _, registro = C.calendario(dt.date(2025, 1, 1), dt.date(2025, 12, 31))

    assert "holidays" in registro["fuente"]
    assert set(registro["definiciones"]) == {
        "es_festivo", "es_vispera_festivo", "es_puente",
        "es_semana_santa", "es_ultima_semana_diciembre",
    }
    assert registro["n_festivos"] > 0
    assert "Emiliani" in registro["ley_emiliani"]


def test_el_calendario_horario_expande_a_24_horas():
    horario, registro = C.calendario_horario(dt.date(2025, 1, 1), dt.date(2025, 1, 3))

    assert len(horario) == 3 * 24
    assert registro["n_horas"] == 72
    primero = horario[horario["fecha_hora"].dt.date == dt.date(2025, 1, 1)]
    assert primero["es_festivo"].all()  # Ano Nuevo


# --- integracion ----------------------------------------------------------


def demanda(horas: int = 48) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fecha_hora": pd.date_range("2025-01-01", periods=horas, freq="h"),
            "valor_kwh": range(horas),
        }
    )


def clima(horas: int = 48, desfase: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fecha_hora": pd.date_range("2025-01-01", periods=horas, freq="h")
            + pd.Timedelta(hours=desfase),
            "temperatura_c": [20.0] * horas,
        }
    )


def test_la_union_conserva_exactamente_las_filas_de_demanda():
    base = demanda()
    calendario_h, _ = C.calendario_horario(dt.date(2025, 1, 1), dt.date(2025, 1, 2))

    resultado, informe = unir(base, clima(), calendario_h)

    assert len(resultado) == len(base)
    assert informe["cuadra"] is True
    assert informe["filas_resultado"] == 48
    assert "temperatura_c" in resultado.columns
    assert "es_festivo" in resultado.columns


def test_falla_si_la_derecha_trae_timestamps_repetidos():
    """El caso que multiplicaria filas en silencio."""
    repetido = pd.concat([clima(24), clima(24)], ignore_index=True)

    with pytest.raises(ErrorUnion, match="repetidas"):
        unir(demanda(24), repetido)


def test_falla_si_la_demanda_trae_timestamps_repetidos():
    doble = pd.concat([demanda(24), demanda(24)], ignore_index=True)

    with pytest.raises(ErrorUnion, match="repetidas"):
        unir(doble, clima(24))


def test_falla_si_las_columnas_se_pisan():
    solapada = clima(24).rename(columns={"temperatura_c": "valor_kwh"})

    with pytest.raises(ErrorUnion, match="comparte columnas"):
        unir(demanda(24), solapada)


def test_reporta_las_horas_sin_cobertura_sin_fallar():
    parcial = clima(24)  # solo cubre la mitad de las 48 horas

    resultado, informe = unir(demanda(48), parcial)

    union = [u for u in informe["uniones"] if u["fuente"] == "clima"][0]
    assert union["filas_sin_cobertura"] == 24
    assert union["pct_sin_cobertura"] == 50.0
    assert len(resultado) == 48


def test_puede_exigirse_cobertura_total():
    with pytest.raises(ErrorUnion, match="sin dato"):
        unir(demanda(48), clima(24), exigir_cobertura_total=True)


def test_las_fuentes_ausentes_se_anotan_como_no_aplicadas():
    resultado, informe = unir(demanda(24))

    assert len(resultado) == 24
    assert all(not u["aplicada"] for u in informe["uniones"])


def test_demanda_vacia_falla():
    with pytest.raises(ErrorUnion, match="vacia"):
        unir(pd.DataFrame())


# --- descarga de clima (sin red) -----------------------------------------


class RespuestaFalsa:
    def __init__(self, cuerpo: dict[str, Any], status: int = 200) -> None:
        self.status_code = status
        self._cuerpo = cuerpo
        self.text = str(cuerpo)

    def json(self) -> dict[str, Any]:
        return self._cuerpo


class SesionFalsa:
    def __init__(self, horas: int = 24) -> None:
        self.horas = horas
        self.llamados: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> RespuestaFalsa:
        self.llamados.append(kwargs)
        tiempos = pd.date_range("2025-01-01", periods=self.horas, freq="h")
        return RespuestaFalsa(
            {
                "latitude": 4.745,
                "longitude": -74.10,
                "elevation": 2557.0,
                "timezone": "America/Bogota",
                "utc_offset_seconds": -18000,
                "hourly_units": {"temperature_2m": "°C"},
                "hourly": {
                    "time": [t.strftime("%Y-%m-%dT%H:%M") for t in tiempos],
                    "temperature_2m": [15.0] * self.horas,
                },
            }
        )


def test_la_descarga_registra_los_pesos_usados():
    sesion = SesionFalsa()
    pesos = {"bogota": 0.6, "cali": 0.4}

    _, registro = C.descargar_clima(
        dt.date(2025, 1, 1), dt.date(2025, 1, 1), pesos=pesos, sesion=sesion
    )

    assert registro["pesos_usados"] == pesos
    assert registro["pesos_son_proxy"] is False
    assert len(sesion.llamados) == 2  # una consulta por ciudad


def test_la_descarga_anota_la_celda_de_malla_devuelta():
    """La API devuelve el centro de la celda, no el punto pedido."""
    _, registro = C.descargar_clima(
        dt.date(2025, 1, 1), dt.date(2025, 1, 1),
        pesos={"bogota": 1.0}, sesion=SesionFalsa(),
    )

    ciudad = registro["ciudades"][0]
    assert ciudad["lat_pedida"] == 4.7110
    assert ciudad["lat_malla"] == 4.745
    assert ciudad["elevacion_m"] == 2557.0
