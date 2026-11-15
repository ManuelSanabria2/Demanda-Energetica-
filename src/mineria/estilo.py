"""Estilo comun de las figuras del informe (matplotlib, salida PNG para el PDF).

Paleta categorica en orden fijo (nunca ciclada): las figuras con grupos usan
siempre el mismo color para el mismo grupo. Las de dispersion con mas de tres
grupos llevan ademas un marcador distinto por grupo, para no depender del color.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*"]
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
REJILLA = "#e4e3df"
SECUENCIAL = LinearSegmentedColormap.from_list("azul", ["#f3f7fd", "#9cc3ef", "#2a78d6", "#0d3a75"])
DIVERGENTE = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f1f0ee", "#eb6834"])


def aplicar() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": TINTA_SECUNDARIA,
            "axes.labelcolor": TINTA,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": REJILLA,
            "grid.linewidth": 0.6,
            "axes.prop_cycle": plt.cycler(color=CATEGORICA),
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "xtick.color": TINTA_SECUNDARIA,
            "ytick.color": TINTA_SECUNDARIA,
        }
    )


def color_grupo(grupo: str) -> str:
    """'G1' -> slot 1, 'S2' -> slot 2: el color sigue al grupo, no a su rango en la figura."""
    return CATEGORICA[(int(grupo[1:]) - 1) % len(CATEGORICA)]


def marcador_grupo(grupo: str) -> str:
    return MARCADORES[(int(grupo[1:]) - 1) % len(MARCADORES)]


def guardar(figura, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta)
    plt.close(figura)
    return ruta
