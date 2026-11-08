"""Tests del panel de modelado (scripts/construir_panel.py). Sin red.

El caso que los motiva: ningun comando refrescaba el clima, y el panel se
construia igual, con la temperatura en NaN en todas las horas posteriores a la
ultima descarga y solo un aviso en el log.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from ingesta import cli, config  # noqa: E402
from ingesta.complementarias import ErrorUnion, guardar_clima  # noqa: E402
from limpieza import limpiar as modulo_limpiar  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "construir_panel", RAIZ / "scripts" / "construir_panel.py"
)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)


@pytest.fixture(autouse=True)
def datos_aislados(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Lee y escribe en un temporal, no en data/processed ni en data/procesado."""
    monkeypatch.setattr(config, "DIR_PROCESADO", tmp_path / "processed")
    monkeypatch.setattr(modulo_limpiar, "DIR_SALIDA", tmp_path / "procesado")
    yield


def _horas(dias: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-03-01", periods=dias * 24, freq="h", tz="America/Bogota")


def guardar_demanda(dias: int) -> None:
    demanda = pd.DataFrame(
        {
            "fecha_hora": _horas(dias),
            "fuente": "xm",
            "metrica": "DemaReal",
            "entidad": "Sistema",
            "valor_kwh": 1000.0,
            "version": None,
        }
    )
    cli.escribir_parquet(demanda, "xm_demanda_real_sistema")


def guardar_temperatura(dias: int) -> None:
    clima = pd.DataFrame(
        {"fecha_hora": _horas(dias), "temperatura_c": 20.0, "ciudades_disponibles": 4}
    )
    guardar_clima(clima, {})


def test_falla_si_el_clima_es_mas_antiguo_que_la_demanda():
    guardar_demanda(10)
    guardar_temperatura(5)  # la ultima descarga de clima acaba cinco dias antes

    with pytest.raises(ErrorUnion, match="--fuente clima"):
        panel.construir()


def test_falla_si_no_hay_clima():
    guardar_demanda(10)

    with pytest.raises(FileNotFoundError, match="--fuente clima"):
        panel.construir()


def test_un_directorio_de_clima_vacio_cuenta_como_sin_clima():
    """Pasa si la primera descarga falla tras crear la carpeta: unir() se salta
    las tablas vacias y el panel salia sin temperatura y sin error."""
    guardar_demanda(10)
    (config.DIR_PROCESADO / "clima_nacional").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="--fuente clima"):
        panel.construir()


def test_sin_clima_explicito_construye_sin_temperatura():
    guardar_demanda(10)

    resultado, _ = panel.construir(incluir_clima=False)

    assert len(resultado) == 240
    assert "temperatura_c" not in resultado.columns


def test_con_clima_al_dia_todas_las_horas_tienen_temperatura():
    guardar_demanda(10)
    guardar_temperatura(10)

    resultado, informe = panel.construir()

    assert len(resultado) == 240
    assert resultado["temperatura_c"].notna().all()
    assert informe["union"]["cuadra"]
