"""Paleta y plantilla de Plotly de la interfaz.

Los colores son los de src/mineria/estilo.py (las figuras del PDF): el mismo
grupo tiene el mismo color en el informe y en la app. No se importa de ahi
porque ese modulo arrastra matplotlib, que la app no necesita;
tests/test_interfaz.py comprueba que las dos paletas no se separen.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
REJILLA = "#e4e3df"
SECUENCIAL = [[0.0, "#f3f7fd"], [0.33, "#9cc3ef"], [0.66, "#2a78d6"], [1.0, "#0d3a75"]]
DIVERGENTE = [[0.0, "#2a78d6"], [0.5, "#f1f0ee"], [1.0, "#eb6834"]]

# Tipo de dia -> color fijo (mismo orden que las figuras del EDA).
COLOR_TIPO_DIA = {
    "laboral": CATEGORICA[0],
    "sabado": CATEGORICA[1],
    "domingo": CATEGORICA[2],
    "festivo": CATEGORICA[3],
    "semana_santa": CATEGORICA[6],
}
NOMBRE_TIPO_DIA = {
    "laboral": "Laboral",
    "sabado": "Sábado",
    "domingo": "Domingo",
    "festivo": "Festivo",
    "semana_santa": "Semana Santa",
}
DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

PLANTILLA = "siem"


def color_grupo(grupo: str) -> str:
    """'G1' -> slot 1, 'S2' -> slot 2: igual que src/mineria/estilo.color_grupo."""
    return CATEGORICA[(int(grupo[1:]) - 1) % len(CATEGORICA)]


def titulo(figura: go.Figure, texto: str) -> go.Figure:
    """Titulo arriba a la izquierda, por encima de la leyenda horizontal (sin solaparse)."""
    return figura.update_layout(
        title=dict(text=texto, x=0, xanchor="left", yref="container", y=0.98, yanchor="top", font=dict(size=15)),
        margin=dict(t=90),
    )


def _registrar() -> None:
    plantilla = go.layout.Template()
    plantilla.layout = go.Layout(
        colorway=CATEGORICA,
        font=dict(color=TINTA, size=13),
        xaxis=dict(gridcolor=REJILLA, linecolor=TINTA_SECUNDARIA, zeroline=False),
        yaxis=dict(gridcolor=REJILLA, linecolor=TINTA_SECUNDARIA, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(font_size=13),
        separators=", ",  # coma decimal, espacio de miles
    )
    pio.templates[PLANTILLA] = plantilla
    pio.templates.default = "plotly_white+" + PLANTILLA


_registrar()
