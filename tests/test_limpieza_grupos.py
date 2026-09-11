"""Tests de la limpieza por grupos y de la etiqueta en las filas creadas."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from limpieza import grupos  # noqa: E402
from limpieza.limpiar import ErrorLimpieza, completar_rejilla  # noqa: E402

ZONA = "America/Bogota"


def serie(sector: str, nivel: float, dias: int = 200, semilla: int = 1) -> pd.DataFrame:
    """Una subactividad con perfil horario y ruido pequeno, en el formato de CIIU."""
    aleatorio = random.Random(semilla)
    momentos = pd.date_range("2025-01-06", periods=dias * 24, freq="h", tz=ZONA)
    return pd.DataFrame(
        {
            "timestamp": momentos,
            "valor": [
                nivel * (0.8 + 0.4 * (m.hour in range(8, 20))) * (1 + aleatorio.gauss(0, 0.02))
                for m in momentos
            ],
            "Activity": "ACT",
            "Subactivity": sector,
        }
    )


def quitar(marco: pd.DataFrame, desde: str, horas: int) -> pd.DataFrame:
    fuera = pd.date_range(desde, periods=horas, freq="h", tz=ZONA)
    return marco[~marco["timestamp"].isin(fuera)].reset_index(drop=True)


# --- la etiqueta en las filas creadas -------------------------------------


def test_las_filas_creadas_conservan_las_columnas_constantes():
    """El fallo que se encontro: las filas nuevas quedaban con subactivity = NaN."""
    marco = pd.DataFrame(
        {
            "fecha_hora": pd.date_range("2025-01-01", periods=10, freq="h", tz=ZONA),
            "valor_kwh": [float(i) for i in range(10)],
            "subactivity": "S1",
            "fuente": "xm",
        }
    ).drop(index=[4, 5, 6, 7])

    completo, _ = completar_rejilla(marco)

    assert len(completo) == 10
    assert completo["subactivity"].eq("S1").all()
    assert completo["fuente"].eq("xm").all()


def test_una_columna_que_varia_no_se_inventa_en_las_filas_creadas():
    marco = pd.DataFrame(
        {
            "fecha_hora": pd.date_range("2025-01-01", periods=10, freq="h", tz=ZONA),
            "valor_kwh": [float(i) for i in range(10)],
            "nota": [f"n{i}" for i in range(10)],
        }
    ).drop(index=[5])

    completo, _ = completar_rejilla(marco)
    creada = completo[completo["fecha_hora"] == pd.Timestamp("2025-01-01 05:00", tz=ZONA)]

    assert pd.isna(creada["nota"].iloc[0])


# --- limpieza por grupos --------------------------------------------------


def test_cada_grupo_se_limpia_por_separado():
    tabla = pd.concat([serie("BIBLIOTECAS", 100.0), serie("INDUSTRIA", 10_000.0, semilla=2)])

    resultados = list(grupos.limpiar_por_grupos(tabla, ["Activity", "Subactivity"]))

    assert [e["Subactivity"] for e, _, _ in resultados] == ["BIBLIOTECAS", "INDUSTRIA"]
    assert sum(len(limpio) for _, limpio, _ in resultados) == len(tabla)
    for etiqueta, limpio, registro in resultados:
        assert limpio["subactivity"].eq(etiqueta["Subactivity"]).all()
        assert not limpio["timestamp"].duplicated().any()
        assert registro["grupo"] == etiqueta


def test_los_estadisticos_de_un_grupo_no_contaminan_a_otro():
    """10 000 es normal en la industria y absurdo en una biblioteca."""
    biblioteca = serie("BIBLIOTECAS", 100.0)
    momento = pd.Timestamp("2025-06-09 03:00", tz=ZONA)
    biblioteca.loc[biblioteca["timestamp"] == momento, "valor"] = 10_000.0
    tabla = pd.concat([biblioteca, serie("INDUSTRIA", 10_000.0, semilla=2)])

    resultado = {
        e["Subactivity"]: limpio
        for e, limpio, _ in grupos.limpiar_por_grupos(tabla, ["Activity", "Subactivity"])
    }

    bib = resultado["BIBLIOTECAS"]
    assert bib.loc[bib["timestamp"] == momento, "atipico"].iloc[0]
    assert resultado["INDUSTRIA"]["atipico"].mean() < 0.01


def test_un_hueco_dentro_de_un_grupo_se_trata_sin_perder_la_etiqueta():
    corto = quitar(serie("CORTO", 100.0), "2025-03-01 02:00", 2)
    largo = quitar(serie("LARGO", 100.0, semilla=3), "2025-03-01 00:00", 48)
    tabla = pd.concat([corto, largo])

    resultado = {
        e["Subactivity"]: limpio
        for e, limpio, _ in grupos.limpiar_por_grupos(tabla, ["Activity", "Subactivity"])
    }

    assert (resultado["CORTO"]["origen_valor"] == "interpolado").sum() == 2
    assert (resultado["LARGO"]["origen_valor"] == "faltante").sum() == 48
    for limpio in resultado.values():
        assert limpio["subactivity"].notna().all()
        assert limpio["activity"].eq("ACT").all()


def test_la_rejilla_de_un_grupo_no_se_extiende_al_rango_de_otro():
    """Un sector que solo existio unos meses no gana horas inventadas antes o despues."""
    efimero = serie("EFIMERO", 100.0, dias=30)
    tabla = pd.concat([efimero, serie("LONGEVO", 100.0, dias=200, semilla=4)])

    resultado = {
        e["Subactivity"]: limpio
        for e, limpio, _ in grupos.limpiar_por_grupos(tabla, ["Activity", "Subactivity"])
    }

    assert len(resultado["EFIMERO"]) == 30 * 24


def test_las_columnas_categoricas_funcionan_como_grupo():
    tabla = pd.concat([serie("A", 100.0), serie("B", 200.0, semilla=5)])
    tabla["Activity"] = tabla["Activity"].astype("category")
    tabla["Subactivity"] = tabla["Subactivity"].astype("category")

    etiquetas = [e for e, _, _ in grupos.limpiar_por_grupos(tabla, ["Activity", "Subactivity"])]

    assert etiquetas == [
        {"Activity": "ACT", "Subactivity": "A"},
        {"Activity": "ACT", "Subactivity": "B"},
    ]


def test_una_columna_de_grupo_ausente_falla():
    with pytest.raises(ErrorLimpieza, match="ausentes"):
        next(grupos.limpiar_por_grupos(serie("A", 1.0, dias=10), ["NoExiste"]))


def test_el_resumen_suma_todos_los_grupos():
    tabla = pd.concat([
        quitar(serie("A", 100.0), "2025-03-01 02:00", 2),
        serie("B", 200.0, semilla=6),
    ])

    filas = [
        grupos.fila_de_resumen(e, limpio, reg)
        for e, limpio, reg in grupos.limpiar_por_grupos(tabla, ["Activity", "Subactivity"])
    ]
    total = grupos.resumir(filas)

    assert total["n_grupos"] == 2
    assert total["filas_finales"] == 2 * 200 * 24
    assert total["interpolados"] == 2
    assert total["grupos_con_interpolados"] == 1
    assert total["observados"] + total["interpolados"] + total["faltantes"] == total["filas_finales"]
