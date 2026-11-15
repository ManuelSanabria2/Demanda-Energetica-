"""Humo de los scripts de la entrega (eda.py, mineria_no_supervisada.py). Sin red.

Ejecutan de punta a punta sobre un panel sintetico de dos anos construido con
las mismas funciones que el real (limpiar, calendario_horario, unir), y escriben
en una carpeta temporal: nunca tocan docs/ del repositorio.
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
from variables.construir import construir_tabla  # noqa: E402


def _cargar(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def panel_dos_anios(semilla: int = 3) -> pd.DataFrame:
    horas = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="America/Bogota")
    rng = np.random.default_rng(semilla)
    h, d = horas.hour.to_numpy(), horas.dayofweek.to_numpy()
    calendario, _ = calendario_horario(horas[0].date(), horas[-1].date())
    festivo = calendario.set_index("fecha_hora")["es_festivo"].reindex(horas).fillna(False).to_numpy(bool)
    forma = 1 + 0.2 * np.sin(2 * np.pi * (h - 9) / 24) + 0.1 * np.exp(-((h - 19) ** 2) / 4)
    domingo = (d == 6) | festivo
    forma = np.where(domingo, 0.85 + 0.08 * np.exp(-((h - 20) ** 2) / 4), np.where(d == 5, forma * 0.94, forma))
    temperatura = 20 + 4 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.5, len(horas))
    valor = 9.0e6 * (1 + 0.03 * np.arange(len(horas)) / 8760) * forma * (1 + 0.004 * (temperatura - 20))
    valor = valor * (1 + rng.normal(0, 0.01, len(horas)))
    demanda = pd.DataFrame({"fecha_hora": horas, "fuente": "xm", "metrica": "DemaReal", "entidad": "Sistema",
                            "valor_kwh": valor, "version": None})
    limpio, _ = limpiar(demanda, metodo_imputacion="causal")
    clima = pd.DataFrame({"fecha_hora": limpio["fecha_hora"], "temperatura_c": temperatura, "ciudades_disponibles": 4})
    panel, _ = unir(limpio, clima, calendario, exigir_cobertura_total=True)
    return panel


@pytest.fixture(scope="module")
def entradas(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("entrega")
    panel = panel_dos_anios()
    ruta_panel = carpeta / "panel.parquet"
    panel.to_parquet(ruta_panel, index=False)
    tabla, _ = construir_tabla(panel, 24)
    ruta_tabla = carpeta / "r24.parquet"
    tabla.to_parquet(ruta_tabla, index=False)
    return carpeta, ruta_panel, ruta_tabla


def test_eda_de_punta_a_punta(entradas):
    carpeta, ruta_panel, ruta_tabla = entradas
    salida = carpeta / "docs"
    assert _cargar("eda").main(["--panel", str(ruta_panel), "--tabla", str(ruta_tabla), "--salida", str(salida)]) == 0
    resultados = json.loads((salida / "resultados" / "eda.json").read_text(encoding="utf-8"))
    assert len(list((salida / "figuras" / "eda").glob("*.png"))) == 9
    assert resultados["descriptivos_mwh"]["nulos"] == 0
    assert resultados["acf"]["24"] > 0.5
    assert resultados["calendario"]["kruskal_p"] < 1e-6
    assert resultados["calendario"]["festivo_vs_laboral_pct"] < -5
    assert resultados["temperatura"]["spearman_horario"] > 0
    assert "lag_24" in resultados["correlacion_con_objetivo_r24"]


def test_mineria_de_punta_a_punta(entradas):
    carpeta, ruta_panel, _ = entradas
    salida = carpeta / "docs"
    codigo = _cargar("mineria_no_supervisada").main(
        ["--panel", str(ruta_panel), "--salida", str(salida), "--k-max", "6", "--ciiu", str(carpeta / "no_existe.parquet")]
    )
    assert codigo == 0
    resultados = json.loads((salida / "resultados" / "mineria.json").read_text(encoding="utf-8"))
    dias = resultados["dias"]
    assert dias["k"] >= 2
    assert dias["v_cramer_tipo_dia"] > 0.5
    assert dias["reglas"]["n_reglas"] > 0
    assert "omitido" in resultados["sectores"]
    assert len(list((salida / "figuras" / "mineria").glob("*.png"))) == 5
