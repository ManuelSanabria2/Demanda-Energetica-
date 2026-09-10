"""Tests de la capa de clientes. Sin red: la sesion HTTP esta simulada."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingesta import clientes, config  # noqa: E402
from ingesta.clientes import ClienteSIMEM, ClienteXM, ErrorCliente, ErrorHTTP  # noqa: E402


# --- dobles de prueba -----------------------------------------------------


class RespuestaFalsa:
    """Imita lo justo de requests.Response que usan los clientes."""

    def __init__(self, status_code: int = 200, cuerpo: Any = None, texto: str = "") -> None:
        self.status_code = status_code
        self._cuerpo = cuerpo
        self.text = texto or json.dumps(cuerpo, ensure_ascii=False)

    def json(self) -> Any:
        if self._cuerpo is None:
            raise json.JSONDecodeError("sin cuerpo", self.text or "", 0)
        return self._cuerpo


class SesionFalsa:
    """Devuelve respuestas en cola y registra cada llamado recibido."""

    def __init__(self, respuestas: list[Any]) -> None:
        self.respuestas = list(respuestas)
        self.llamados: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}

    def _siguiente(self, metodo: str, url: str, **kwargs: Any) -> RespuestaFalsa:
        self.llamados.append({"metodo": metodo, "url": url, **kwargs})
        if not self.respuestas:
            raise AssertionError("La sesion falsa recibio mas llamados de los previstos")
        siguiente = self.respuestas.pop(0)
        if isinstance(siguiente, Exception):
            raise siguiente
        return siguiente

    def post(self, url: str, **kwargs: Any) -> RespuestaFalsa:
        return self._siguiente("POST", url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> RespuestaFalsa:
        return self._siguiente("GET", url, **kwargs)


def respuesta_xm(fechas: list[str], entidad: str = "Sistema") -> RespuestaFalsa:
    """Respuesta horaria de XM con 24 horas por fecha."""
    items = [
        {
            "Date": fecha,
            "HourlyEntities": [
                {
                    "Id": entidad,
                    "Values": {
                        "code": entidad,
                        **{f"Hour{n:02d}": f"{1000 + n}.5" for n in range(1, 25)},
                    },
                }
            ],
        }
        for fecha in fechas
    ]
    return RespuestaFalsa(200, {"Metric": {"Id": "DemaReal"}, "Items": items})


def respuesta_simem(marcas: list[str], version: str = "TX2") -> RespuestaFalsa:
    """Respuesta de SIMEM con un registro por marca de tiempo."""
    registros = [
        {
            "CodigoVariable": "DdaReal",
            "FechaHora": marca,
            "CodigoSICAgente": "AAAA",
            "Version": version,
            "Valor": 100.0,
            "UnidadMedida": "kWh",
        }
        for marca in marcas
    ]
    return RespuestaFalsa(200, {"success": True, "result": {"records": registros}})


CATALOGO = pd.DataFrame(
    [
        {"MetricId": "DemaReal", "Entity": "Sistema", "MaxDays": 31},
        {"MetricId": "DemaReal", "Entity": "Agente", "MaxDays": 31},
    ]
)


@pytest.fixture(autouse=True)
def sin_esperas(monkeypatch: pytest.MonkeyPatch) -> None:
    """El backoff no debe hacer esperar a la suite de tests."""
    monkeypatch.setattr(clientes.time, "sleep", lambda _: None)


def cliente_xm(sesion: SesionFalsa, tmp_path: Path, **kwargs: Any) -> ClienteXM:
    cliente = ClienteXM(sesion=sesion, dir_cache=tmp_path, **kwargs)
    cliente._catalogo = CATALOGO  # evita el llamado al catalogo en los tests
    return cliente


# --- formato ancho a largo ------------------------------------------------


def test_xm_convierte_ancho_a_largo_con_timestamp_correcto(tmp_path: Path):
    sesion = SesionFalsa([respuesta_xm(["2024-01-01"])])
    marco = cliente_xm(sesion, tmp_path).consultar(
        "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
    )

    assert len(marco) == 24
    assert list(marco.columns) == ClienteXM.COLUMNAS
    # Hour01 es la franja 00:00-01:00 (config.DESFASE_HORA_XM = 1).
    assert marco["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="America/Bogota")
    assert marco["timestamp"].iloc[-1] == pd.Timestamp("2024-01-01 23:00:00", tz="America/Bogota")
    assert marco["valor"].iloc[0] == 1001.5
    assert marco["valor"].iloc[-1] == 1024.5
    assert set(marco["fuente"]) == {"xm"}
    assert set(marco["identificador"]) == {"DemaReal"}


def test_xm_respeta_la_constante_de_desfase(tmp_path: Path, monkeypatch):
    """Si la convencion horaria cambiara, basta con tocar config."""
    monkeypatch.setattr(config, "DESFASE_HORA_XM", 0)
    sesion = SesionFalsa([respuesta_xm(["2024-01-01"])])
    marco = cliente_xm(sesion, tmp_path).consultar(
        "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
    )

    # Con desfase 0, Hour01 pasa a ser la 01:00 y Hour24 se sale del dia.
    assert marco["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 01:00:00", tz="America/Bogota")
    assert len(marco) == 23


def test_xm_valores_vacios_son_nan(tmp_path: Path):
    respuesta = respuesta_xm(["2024-01-01"])
    respuesta._cuerpo["Items"][0]["HourlyEntities"][0]["Values"]["Hour03"] = ""
    respuesta.text = json.dumps(respuesta._cuerpo)

    sesion = SesionFalsa([respuesta])
    marco = cliente_xm(sesion, tmp_path).consultar(
        "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
    )

    assert marco["valor"].isna().sum() == 1
    assert (marco["valor"] == 0).sum() == 0


# --- fragmentacion --------------------------------------------------------


def test_fragmenta_el_rango_y_concatena(tmp_path: Path):
    sesion = SesionFalsa(
        [respuesta_xm(["2024-01-01"]), respuesta_xm(["2024-02-01"])]
    )
    marco = cliente_xm(sesion, tmp_path, usar_cache=False).consultar(
        "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 2, 5)
    )

    assert len(sesion.llamados) == 2  # 36 dias con MaxDays 31 -> dos tramos
    assert len(marco) == 48
    assert marco["timestamp"].is_monotonic_increasing

    primero, segundo = sesion.llamados
    assert primero["json"]["StartDate"] == "2024-01-01"
    assert primero["json"]["EndDate"] == "2024-01-31"
    assert segundo["json"]["StartDate"] == "2024-02-01"
    assert segundo["json"]["EndDate"] == "2024-02-05"


def test_simem_fragmenta_a_31_dias(tmp_path: Path):
    sesion = SesionFalsa(
        [respuesta_simem(["2024-01-01 00:00:00"]), respuesta_simem(["2024-02-01 00:00:00"])]
    )
    ClienteSIMEM(sesion=sesion, dir_cache=tmp_path, usar_cache=False).consultar(
        "14fabb", dt.date(2024, 1, 1), dt.date(2024, 2, 10)
    )

    assert len(sesion.llamados) == 2
    assert sesion.llamados[0]["params"]["endDate"] == "2024-01-31"


def test_tramo_vacio_no_rompe_la_consulta(tmp_path: Path, caplog):
    sesion = SesionFalsa([RespuestaFalsa(200, {"Metric": {}, "Items": []})])
    with caplog.at_level("WARNING"):
        marco = cliente_xm(sesion, tmp_path, usar_cache=False).consultar(
            "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 2)
        )

    assert marco.empty
    assert list(marco.columns) == ClienteXM.COLUMNAS
    assert any("vino vacio" in r.getMessage() for r in caplog.records)


# --- reintentos -----------------------------------------------------------


def test_reintenta_ante_5xx_y_acaba_bien(tmp_path: Path):
    sesion = SesionFalsa(
        [
            RespuestaFalsa(503, {"error": "no disponible"}),
            RespuestaFalsa(502, {"error": "gateway"}),
            respuesta_xm(["2024-01-01"]),
        ]
    )
    marco = cliente_xm(sesion, tmp_path, usar_cache=False).consultar(
        "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
    )

    assert len(sesion.llamados) == 3
    assert len(marco) == 24


def test_no_reintenta_ante_4xx(tmp_path: Path):
    sesion = SesionFalsa([RespuestaFalsa(400, None, texto='"error: rango excedido"')])

    with pytest.raises(ErrorHTTP) as excinfo:
        cliente_xm(sesion, tmp_path, usar_cache=False).consultar(
            "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
        )

    assert len(sesion.llamados) == 1  # un solo intento
    assert excinfo.value.status == 400
    assert "rango excedido" in excinfo.value.cuerpo


def test_agota_tres_intentos_y_falla(tmp_path: Path):
    sesion = SesionFalsa([RespuestaFalsa(500, {"e": 1})] * 3)

    with pytest.raises(ErrorCliente, match="3 intentos"):
        cliente_xm(sesion, tmp_path, usar_cache=False).consultar(
            "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
        )

    assert len(sesion.llamados) == clientes.MAX_INTENTOS == 3


def test_reintenta_ante_timeout(tmp_path: Path):
    sesion = SesionFalsa(
        [requests.Timeout("se agoto"), requests.ConnectionError("caida"), respuesta_xm(["2024-01-01"])]
    )
    marco = cliente_xm(sesion, tmp_path, usar_cache=False).consultar(
        "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
    )

    assert len(sesion.llamados) == 3
    assert len(marco) == 24


def test_backoff_es_exponencial(tmp_path: Path, monkeypatch):
    esperas: list[float] = []
    monkeypatch.setattr(clientes.time, "sleep", esperas.append)

    sesion = SesionFalsa([RespuestaFalsa(500, {"e": 1})] * 3)
    with pytest.raises(ErrorCliente):
        cliente_xm(sesion, tmp_path, usar_cache=False).consultar(
            "DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
        )

    assert esperas == [2.0, 4.0]  # tras el tercer fallo ya no espera


# --- cache ----------------------------------------------------------------


def test_segunda_consulta_se_sirve_de_cache(tmp_path: Path):
    sesion = SesionFalsa([respuesta_xm(["2024-01-01"])])
    cliente = cliente_xm(sesion, tmp_path)
    rango = (dt.date(2024, 1, 1), dt.date(2024, 1, 1))

    primero = cliente.consultar("DemaReal", *rango)
    segundo = cliente.consultar("DemaReal", *rango)

    assert len(sesion.llamados) == 1  # el segundo no toco la red
    pd.testing.assert_frame_equal(primero, segundo)


def test_la_clave_de_cache_incluye_la_entidad(tmp_path: Path):
    """Sistema y Agente no pueden compartir archivo de cache."""
    sesion = SesionFalsa(
        [respuesta_xm(["2024-01-01"], "Sistema"), respuesta_xm(["2024-01-01"], "EEEE")]
    )
    cliente = cliente_xm(sesion, tmp_path)
    rango = (dt.date(2024, 1, 1), dt.date(2024, 1, 1))

    sistema = cliente.consultar("DemaReal", *rango, entidad="Sistema")
    agente = cliente.consultar("DemaReal", *rango, entidad="Agente")

    assert len(sesion.llamados) == 2  # no hubo colision de cache
    assert set(sistema["entidad"]) == {"Sistema"}
    assert set(agente["entidad"]) == {"EEEE"}


def test_los_tramos_recientes_no_se_cachean(tmp_path: Path):
    """Un tramo reciente cacheado se congelaria en datos parciales."""
    hoy = dt.date.today()
    sesion = SesionFalsa([respuesta_xm([hoy.isoformat()])] * 2)
    cliente = cliente_xm(sesion, tmp_path)

    cliente.consultar("DemaReal", hoy, hoy)
    cliente.consultar("DemaReal", hoy, hoy)

    assert len(sesion.llamados) == 2
    assert not list(tmp_path.rglob("*.parquet"))


def test_usar_cache_false_siempre_pide(tmp_path: Path):
    sesion = SesionFalsa([respuesta_xm(["2024-01-01"])] * 2)
    cliente = cliente_xm(sesion, tmp_path, usar_cache=False)

    cliente.consultar("DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    cliente.consultar("DemaReal", dt.date(2024, 1, 1), dt.date(2024, 1, 1))

    assert len(sesion.llamados) == 2


# --- SIMEM ----------------------------------------------------------------


def test_simem_conserva_las_dimensiones_y_la_version(tmp_path: Path):
    sesion = SesionFalsa([respuesta_simem(["2024-01-01 00:00:00", "2024-01-01 01:00:00"])])
    marco = ClienteSIMEM(sesion=sesion, dir_cache=tmp_path).consultar(
        "14fabb", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
    )

    assert list(marco.columns[:3]) == ClienteSIMEM.COLUMNAS_BASE
    # La capa de acceso no colapsa versiones: eso es trabajo de normalizar.
    assert "Version" in marco.columns
    assert marco["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="America/Bogota")
    assert set(marco["identificador"]) == {"14fabb"}


def test_simem_success_false_es_error(tmp_path: Path):
    sesion = SesionFalsa([RespuestaFalsa(200, {"success": False, "message": "vaya"})])

    with pytest.raises(ErrorCliente, match="success=False"):
        ClienteSIMEM(sesion=sesion, dir_cache=tmp_path, usar_cache=False).consultar(
            "14fabb", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
        )


def test_simem_sin_fechahora_falla_con_las_columnas_recibidas(tmp_path: Path):
    sesion = SesionFalsa(
        [RespuestaFalsa(200, {"success": True, "result": {"records": [{"Otra": 1}]}})]
    )

    with pytest.raises(ErrorCliente, match="FechaHora"):
        ClienteSIMEM(sesion=sesion, dir_cache=tmp_path, usar_cache=False).consultar(
            "14fabb", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
        )
