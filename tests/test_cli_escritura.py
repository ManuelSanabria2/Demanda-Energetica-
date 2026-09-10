"""Tests de la escritura de la capa procesada.

El caso que importa es la regresion: una ingesta acotada a unos pocos dias
llego a borrar los 26 restantes de marzo de 2025 porque
`existing_data_behavior="delete_matching"` elimina la particion entera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingesta import cli, config  # noqa: E402


@pytest.fixture(autouse=True)
def procesado_aislado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Escribe en un directorio temporal, no en data/processed."""
    monkeypatch.setattr(config, "DIR_PROCESADO", tmp_path)
    yield


def serie(inicio: str, horas: int, valor: float = 1000.0) -> pd.DataFrame:
    """Serie en el esquema comun de la capa procesada."""
    return pd.DataFrame(
        {
            "fecha_hora": pd.date_range(inicio, periods=horas, freq="h"),
            "fuente": "xm",
            "metrica": "DemaReal",
            "entidad": "Sistema",
            "valor_kwh": valor,
            "version": None,
        }
    )


def leer(subcarpeta: str = "prueba") -> pd.DataFrame:
    return pd.read_parquet(config.DIR_PROCESADO / subcarpeta).sort_values("fecha_hora")


# --- la regresion ---------------------------------------------------------


def test_una_ingesta_parcial_no_borra_el_resto_del_mes():
    """El bug original: escribir 5 dias de marzo se llevaba los otros 26."""
    cli.escribir_parquet(serie("2025-03-01", 31 * 24), "prueba")
    assert len(leer()) == 744

    # Reingesta acotada a los cinco primeros dias, como la prueba que rompio.
    cli.escribir_parquet(serie("2025-03-01", 5 * 24, valor=2000.0), "prueba")

    resultado = leer()
    assert len(resultado) == 744, "se perdieron horas del resto del mes"
    assert resultado["fecha_hora"].max() == pd.Timestamp("2025-03-31 23:00:00", tz="America/Bogota")


def test_lo_reingestado_sustituye_al_dato_previo():
    cli.escribir_parquet(serie("2025-03-01", 31 * 24, valor=1000.0), "prueba")
    cli.escribir_parquet(serie("2025-03-01", 5 * 24, valor=2000.0), "prueba")

    resultado = leer().set_index("fecha_hora")["valor_kwh"]

    assert resultado.loc["2025-03-01 00:00:00"] == 2000.0  # revisado
    assert resultado.loc["2025-03-10 00:00:00"] == 1000.0  # intacto


def test_no_deja_filas_duplicadas():
    cli.escribir_parquet(serie("2025-03-01", 31 * 24), "prueba")
    cli.escribir_parquet(serie("2025-03-01", 31 * 24), "prueba")
    cli.escribir_parquet(serie("2025-03-10", 5 * 24), "prueba")

    resultado = leer()
    assert len(resultado) == 744
    assert not resultado.duplicated(subset=cli.CLAVE_PROCESADA).any()


def test_las_particiones_no_tocadas_quedan_intactas():
    cli.escribir_parquet(serie("2025-01-01", 31 * 24), "prueba")
    cli.escribir_parquet(serie("2025-03-01", 31 * 24), "prueba")

    # Se reescribe solo marzo; enero no debe cambiar.
    cli.escribir_parquet(serie("2025-03-01", 24, valor=9999.0), "prueba")

    resultado = leer()
    enero = resultado[resultado["fecha_hora"].dt.month == 1]
    assert len(enero) == 744
    assert (enero["valor_kwh"] == 1000.0).all()
    assert len(resultado) == 744 * 2


def test_un_rango_que_cruza_dos_meses_conserva_ambos():
    cli.escribir_parquet(serie("2025-01-01", 31 * 24), "prueba")
    cli.escribir_parquet(serie("2025-02-01", 28 * 24), "prueba")

    cli.escribir_parquet(serie("2025-01-30", 4 * 24, valor=7.0), "prueba")

    resultado = leer()
    assert len(resultado) == (31 + 28) * 24
    assert not resultado["valor_kwh"].isna().any()


# --- comportamiento base --------------------------------------------------


def test_la_primera_escritura_crea_las_particiones():
    escritas = cli.escribir_parquet(serie("2025-03-01", 48), "prueba")

    assert escritas == 48
    assert (config.DIR_PROCESADO / "prueba" / "anio=2025" / "mes=3").exists()


def test_un_marco_vacio_no_escribe_nada():
    assert cli.escribir_parquet(pd.DataFrame(), "prueba") == 0
    assert not (config.DIR_PROCESADO / "prueba").exists()


def test_series_distintas_conviven_en_la_misma_particion():
    """La clave incluye la entidad: dos entidades no se pisan."""
    cli.escribir_parquet(serie("2025-03-01", 24), "prueba")
    otra = serie("2025-03-01", 24, valor=5.0).assign(entidad="Agente")
    cli.escribir_parquet(otra, "prueba")

    resultado = leer()
    assert len(resultado) == 48
    assert set(resultado["entidad"]) == {"Sistema", "Agente"}
