"""Tests de la capa de persistencia. Sin red: el cliente esta simulado."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingesta import config, descarga  # noqa: E402
from ingesta.descarga import ErrorDescarga  # noqa: E402


# --- doble de prueba ------------------------------------------------------


class ClienteFalso:
    """Devuelve un dia de datos por cada fecha del rango pedido."""

    instancias: list[ClienteFalso] = []

    def __init__(self, usar_cache: bool = True) -> None:
        self.usar_cache = usar_cache
        self.consultas: list[tuple[str, dt.date, dt.date]] = []
        ClienteFalso.instancias.append(self)

    def consultar(
        self, identificador: str, desde: dt.date, hasta: dt.date, **kwargs: Any
    ) -> pd.DataFrame:
        self.consultas.append((identificador, desde, hasta))
        horas = pd.date_range(desde, dt.datetime.combine(hasta, dt.time(23)), freq="h")
        return pd.DataFrame(
            {
                "timestamp": horas,
                "fuente": "prueba",
                "identificador": identificador,
                "entidad": "Sistema",
                "valor": range(len(horas)),
            }
        )


@pytest.fixture(autouse=True)
def entorno(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Aisla la capa cruda y el manifiesto en un directorio temporal."""
    monkeypatch.setattr(config, "DIR_CRUDO", tmp_path / "raw")
    monkeypatch.setattr(config, "RUTA_MANIFIESTO_CRUDO", tmp_path / "raw" / "manifiesto.json")
    monkeypatch.setattr(config, "FECHA_INICIO_DEFECTO", dt.date(2024, 1, 1))
    monkeypatch.setitem(descarga.CLIENTES, "prueba", ClienteFalso)
    ClienteFalso.instancias = []
    yield


def _descargar(**kwargs: Any) -> dict[str, Any]:
    base = {"fuente": "prueba", "identificador": "X1"}
    return descarga.descargar(**{**base, **kwargs})


# --- particionado ---------------------------------------------------------


def test_escribe_en_la_ruta_particionada_esperada():
    _descargar(desde=dt.date(2024, 3, 1), hasta=dt.date(2024, 3, 2))

    esperada = config.DIR_CRUDO / "prueba" / "X1" / "anio=2024" / "mes=03" / "datos.parquet"
    assert esperada.exists(), f"No se escribio en {esperada}"


def test_el_mes_va_a_dos_digitos():
    _descargar(desde=dt.date(2024, 1, 1), hasta=dt.date(2024, 1, 1))
    _descargar(desde=dt.date(2024, 11, 1), hasta=dt.date(2024, 11, 1))

    meses = sorted(p.name for p in (config.DIR_CRUDO / "prueba" / "X1" / "anio=2024").iterdir())
    assert meses == ["mes=01", "mes=11"]


def test_un_rango_de_varios_meses_genera_varias_particiones():
    _descargar(desde=dt.date(2024, 1, 15), hasta=dt.date(2024, 3, 10))

    particiones = sorted(
        p.parent.name for p in (config.DIR_CRUDO / "prueba" / "X1").rglob("datos.parquet")
    )
    assert particiones == ["mes=01", "mes=02", "mes=03"]


def test_anio_y_mes_se_leen_como_columnas_del_particionado():
    _descargar(desde=dt.date(2024, 3, 1), hasta=dt.date(2024, 3, 2))
    marco = descarga.leer("prueba", "X1")

    assert "anio" in marco.columns and "mes" in marco.columns
    assert int(marco["anio"].iloc[0]) == 2024


# --- idempotencia ---------------------------------------------------------


def test_ejecutar_dos_veces_no_duplica():
    rango = {"desde": dt.date(2024, 3, 1), "hasta": dt.date(2024, 3, 5)}

    primera = _descargar(**rango)
    contenido_1 = descarga.leer("prueba", "X1")
    segunda = _descargar(**rango)
    contenido_2 = descarga.leer("prueba", "X1")

    assert len(contenido_1) == 5 * 24
    assert len(contenido_2) == len(contenido_1)
    assert primera["hash_resultado"] == segunda["hash_resultado"]
    assert descarga.hash_marco(contenido_1) == descarga.hash_marco(contenido_2)


def test_rangos_solapados_no_duplican():
    _descargar(desde=dt.date(2024, 3, 1), hasta=dt.date(2024, 3, 10))
    _descargar(desde=dt.date(2024, 3, 5), hasta=dt.date(2024, 3, 15))

    marco = descarga.leer("prueba", "X1")
    assert len(marco) == 15 * 24
    assert not marco.duplicated(subset=["timestamp", "entidad"]).any()


def test_un_valor_revisado_sustituye_al_anterior():
    """Si la fuente corrige un dato, gana el recien descargado."""
    _descargar(desde=dt.date(2024, 3, 1), hasta=dt.date(2024, 3, 1))

    revisado = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-03-01 00:00:00")],
            "fuente": ["prueba"],
            "identificador": ["X1"],
            "entidad": ["Sistema"],
            "valor": [999.0],
        }
    )
    descarga.escribir(revisado, "prueba", "X1")

    marco = descarga.leer("prueba", "X1")
    fila = marco[marco["timestamp"] == pd.Timestamp("2024-03-01 00:00:00")]
    assert len(fila) == 1
    assert fila["valor"].iloc[0] == 999.0
    assert len(marco) == 24


def test_la_version_forma_parte_de_la_identidad():
    """En SIMEM conviven varias versiones sobre la misma hora: no se colapsan aqui."""
    filas = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-03-01")] * 2,
            "fuente": ["simem"] * 2,
            "identificador": ["14fabb"] * 2,
            "Version": ["TX2", "TXR"],
            "Valor": [100.0, 101.0],
        }
    )
    descarga.escribir(filas, "prueba", "S1")
    descarga.escribir(filas, "prueba", "S1")  # idempotente

    marco = descarga.leer("prueba", "S1")
    assert len(marco) == 2
    assert sorted(marco["Version"]) == ["TX2", "TXR"]


# --- descarga incremental -------------------------------------------------


def test_la_primera_descarga_arranca_en_la_fecha_por_defecto():
    entrada = _descargar(hasta=dt.date(2024, 1, 3))

    assert entrada["modo"] == "primera descarga"
    assert entrada["rango_solicitado"][0] == "2024-01-01"


def test_la_segunda_descarga_es_incremental_con_solape():
    _descargar(desde=dt.date(2024, 1, 1), hasta=dt.date(2024, 6, 30))
    entrada = _descargar(hasta=dt.date(2024, 7, 5))

    assert entrada["modo"] == "incremental"
    # Arranca VENTANA_REFRESCO_DIAS antes del ultimo dato, no justo despues.
    esperado = dt.date(2024, 6, 30) - dt.timedelta(days=config.VENTANA_REFRESCO_DIAS)
    assert entrada["rango_solicitado"][0] == esperado.isoformat()


def test_el_solape_recoge_revisiones_sin_duplicar():
    _descargar(desde=dt.date(2024, 1, 1), hasta=dt.date(2024, 6, 30))
    antes = len(descarga.leer("prueba", "X1"))

    _descargar(hasta=dt.date(2024, 6, 30))  # mismo fin: no hay dias nuevos
    despues = descarga.leer("prueba", "X1")

    assert len(despues) == antes
    assert not despues.duplicated(subset=["timestamp", "entidad"]).any()


def test_forzar_ignora_el_estado_y_la_cache():
    _descargar(desde=dt.date(2024, 1, 1), hasta=dt.date(2024, 6, 30))
    entrada = _descargar(hasta=dt.date(2024, 7, 5), forzar=True)

    assert entrada["modo"] == "forzado"
    assert entrada["rango_solicitado"][0] == "2024-01-01"
    assert ClienteFalso.instancias[-1].usar_cache is False


def test_desde_explicito_manda_sobre_el_incremental():
    _descargar(desde=dt.date(2024, 1, 1), hasta=dt.date(2024, 6, 30))
    entrada = _descargar(desde=dt.date(2024, 2, 1), hasta=dt.date(2024, 2, 5))

    assert entrada["modo"] == "rango explicito"
    assert entrada["rango_solicitado"] == ["2024-02-01", "2024-02-05"]


def test_ultima_fecha_almacenada_sin_datos_es_none():
    assert descarga.ultima_fecha_almacenada("prueba", "inexistente") is None


# --- manifiesto -----------------------------------------------------------


def test_el_manifiesto_registra_lo_pedido():
    entrada = _descargar(desde=dt.date(2024, 3, 1), hasta=dt.date(2024, 3, 2))

    for campo in (
        "identificador",
        "rango_solicitado",
        "rango_obtenido",
        "n_registros",
        "timestamp_ejecucion",
        "hash_resultado",
    ):
        assert campo in entrada, f"falta {campo} en el manifiesto"

    assert entrada["rango_solicitado"] == ["2024-03-01", "2024-03-02"]
    assert entrada["rango_obtenido"] == ["2024-03-01", "2024-03-02"]
    assert entrada["n_registros"] == 48
    assert entrada["hash_resultado"].startswith("sha256:")


def test_el_manifiesto_es_append_only():
    _descargar(desde=dt.date(2024, 3, 1), hasta=dt.date(2024, 3, 1))
    _descargar(desde=dt.date(2024, 3, 2), hasta=dt.date(2024, 3, 2))
    _descargar(desde=dt.date(2024, 3, 3), hasta=dt.date(2024, 3, 3))

    with config.RUTA_MANIFIESTO_CRUDO.open(encoding="utf-8") as origen:
        manifiesto = json.load(origen)

    assert len(manifiesto["descargas"]) == 3
    momentos = [d["timestamp_ejecucion"] for d in manifiesto["descargas"]]
    assert momentos == sorted(momentos)


def test_el_manifiesto_anota_el_entorno():
    entrada = _descargar(desde=dt.date(2024, 3, 1), hasta=dt.date(2024, 3, 1))

    # El hash depende de la version de pandas, asi que queda registrada.
    assert entrada["entorno"]["pandas"] == pd.__version__
    assert "python" in entrada["entorno"]


def test_el_manifiesto_anota_hash_por_particion():
    entrada = _descargar(desde=dt.date(2024, 1, 15), hasta=dt.date(2024, 2, 10))

    particiones = entrada["particiones_escritas"]
    assert set(particiones) == {"anio=2024/mes=01", "anio=2024/mes=02"}
    for detalle in particiones.values():
        assert detalle["hash"].startswith("sha256:")
        assert detalle["n_registros"] > 0


def test_el_historial_se_puede_leer_como_tabla():
    _descargar(desde=dt.date(2024, 3, 1), hasta=dt.date(2024, 3, 1))
    _descargar(desde=dt.date(2024, 3, 2), hasta=dt.date(2024, 3, 2))

    tabla = descarga.historial(fuente="prueba")
    assert len(tabla) == 2
    assert list(tabla["identificador"].unique()) == ["X1"]


# --- errores --------------------------------------------------------------


def test_fuente_desconocida_falla_con_las_conocidas():
    with pytest.raises(ErrorDescarga, match="Fuente desconocida"):
        descarga.descargar("inexistente", "X1")


def test_un_marco_sin_timestamp_falla():
    with pytest.raises(ErrorDescarga, match="timestamp"):
        descarga.escribir(pd.DataFrame({"otra": [1]}), "prueba", "X1")


def test_hash_estable_ante_el_orden_de_las_filas():
    marco = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=5, freq="h"), "valor": range(5)}
    )
    assert descarga.hash_marco(marco) == descarga.hash_marco(marco.iloc[::-1])
