"""Tests del generador de datasets/ (scripts/descargar_datasets.py). Sin red.

El caso que motiva estas pruebas ocurrio de verdad: una descarga interrumpida
dejo un .csv.gz con nombre definitivo, truncado a mitad del anio, que cualquiera
habria podido abrir como si fuera el dataset completo.
"""

from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

_spec = importlib.util.spec_from_file_location(
    "descargar_datasets", RAIZ / "scripts" / "descargar_datasets.py"
)
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)


@pytest.fixture(autouse=True)
def raiz_temporal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """El manifiesto guarda rutas relativas a RAIZ: se apunta al temporal."""
    monkeypatch.setattr(dd, "RAIZ", tmp_path)
    yield


def tramo(inicio: str, horas: int, desfase: int = 0) -> pd.DataFrame:
    momentos = pd.date_range(inicio, periods=horas, freq="h", tz="America/Bogota")
    return pd.DataFrame(
        {
            "timestamp": momentos,
            "Activity": "A",
            "Subactivity": [f"S{(i + desfase) % 3}" for i in range(horas)],
            "valor": [float(i) for i in range(horas)],
        }
    )


# --- escritura atomica ----------------------------------------------------


def test_no_publica_el_nombre_final_hasta_cerrar(tmp_path: Path):
    ruta = tmp_path / "datos.csv"
    escritor = dd.EscritorCSV(ruta)
    escritor.escribir(tramo("2025-01-01", 24))

    assert not ruta.exists(), "un archivo a medio escribir ya tenia el nombre final"
    assert (tmp_path / "datos.csv.tmp").exists()

    escritor.cerrar()

    assert ruta.exists()
    assert not (tmp_path / "datos.csv.tmp").exists()


def test_descartar_no_deja_ningun_archivo(tmp_path: Path):
    ruta = tmp_path / "datos.csv.gz"
    escritor = dd.EscritorCSV(ruta, comprimir=True)
    escritor.escribir(tramo("2025-01-01", 24))
    escritor.descartar()

    assert list(tmp_path.iterdir()) == []


def test_sin_filas_no_se_crea_el_archivo(tmp_path: Path):
    escritor = dd.EscritorCSV(tmp_path / "vacio.csv")
    escritor.escribir(pd.DataFrame())

    assert escritor.cerrar() is None
    assert list(tmp_path.iterdir()) == []


# --- contenido ------------------------------------------------------------


def test_gzip_por_tramos_se_lee_entero_con_una_sola_cabecera(tmp_path: Path):
    escritor = dd.EscritorCSV(tmp_path / "ciiu.csv", comprimir=True)
    escritor.escribir(tramo("2025-01-01", 48))
    escritor.escribir(tramo("2025-01-03", 48))
    entrada = escritor.cerrar()

    ruta = tmp_path / "ciiu.csv.gz"
    assert entrada["archivo"] == "ciiu.csv.gz"
    assert entrada["comprimido"] is True

    leido = pd.read_csv(ruta)
    assert len(leido) == 96 == entrada["n_filas"]
    with gzip.open(ruta, "rt", encoding="utf-8") as origen:
        cabeceras = sum(1 for linea in origen if linea.startswith("timestamp,"))
    assert cabeceras == 1


def test_el_manifiesto_recoge_filas_y_rango(tmp_path: Path):
    escritor = dd.EscritorCSV(tmp_path / "x.csv")
    escritor.escribir(tramo("2025-01-03", 24))
    escritor.escribir(tramo("2025-01-01", 24))  # llega desordenado a proposito
    entrada = escritor.cerrar()

    assert entrada["n_filas"] == 48
    assert entrada["rango"][0].startswith("2025-01-01 00:00:00")
    assert entrada["rango"][1].startswith("2025-01-03 23:00:00")
    assert len(entrada["sha256_archivo"]) == 64


def test_el_mismo_contenido_da_el_mismo_hash_aunque_llegue_desordenado(tmp_path: Path):
    """Sin orden fijo, el hash dependeria del orden en que responde la API."""
    base = tramo("2025-01-01", 72)
    revuelto = base.sample(frac=1, random_state=3)

    a = dd.EscritorCSV(tmp_path / "a.csv", orden=dd.ORDEN_CIIU)
    a.escribir(base)
    b = dd.EscritorCSV(tmp_path / "b.csv", orden=dd.ORDEN_CIIU)
    b.escribir(revuelto)

    assert a.cerrar()["sha256_archivo"] == b.cerrar()["sha256_archivo"]


def test_un_tramo_con_columnas_distintas_falla(tmp_path: Path):
    """Un tramo desalineado corromperia el CSV sin que nada lo notara."""
    escritor = dd.EscritorCSV(tmp_path / "x.csv")
    escritor.escribir(tramo("2025-01-01", 24))

    with pytest.raises(RuntimeError, match="columnas distintas"):
        escritor.escribir(tramo("2025-01-02", 24).drop(columns="Subactivity"))
    escritor.descartar()


# --- troceo por anios -----------------------------------------------------


def test_los_anios_se_recortan_al_rango_pedido():
    assert dd._anios(dt.date(2025, 3, 10), dt.date(2026, 2, 1)) == [
        (2025, dt.date(2025, 3, 10), dt.date(2025, 12, 31)),
        (2026, dt.date(2026, 1, 1), dt.date(2026, 2, 1)),
    ]


def test_ciiu_se_pide_en_tramos_cortos():
    """31 dias de CIIU rozan el timeout del cliente; 7 dejan margen."""
    assert dd.DIAS_POR_TRAMO_CIIU <= 7
