"""Interfaz del proyecto para la sustentacion: punto de entrada.

    streamlit run app/Inicio.py

Recorre el proyecto en el orden de CRISP-DM, del problema al despliegue. La app
no entrena ni descarga datos: solo lee artefactos que ya genera el pipeline
(docs/resultados/, docs/figuras/, infra/costos/ y app/datos/). Si falta uno, la
pagina que lo usa muestra que archivo falta y que comando lo genera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from componentes import estilo  # noqa: E402,F401  (registra la plantilla de Plotly)

st.set_page_config(
    page_title="Demanda horaria del SIN",
    page_icon=":material/bolt:",
    layout="wide",
)


def _pagina(archivo: str, titulo: str, icono: str, **opciones) -> st.Page:
    return st.Page(APP / "paginas" / archivo, title=titulo, icon=icono, **opciones)


# Secciones en el orden de CRISP-DM.
PAGINAS = {
    "Comprensión del negocio": [
        _pagina("inicio.py", "Inicio", ":material/home:", default=True),
    ],
    "Comprensión de los datos": [
        _pagina("datos.py", "Datos", ":material/database:"),
        _pagina("exploracion.py", "Exploración", ":material/query_stats:"),
    ],
    "Modelado": [
        _pagina("mineria.py", "Minería no supervisada", ":material/hub:"),
        _pagina("modelos.py", "Modelos", ":material/model_training:"),
    ],
    "Evaluación": [
        _pagina("prediccion.py", "Predicción frente a real", ":material/show_chart:"),
        _pagina("alertas.py", "Alertas de pico", ":material/notifications_active:"),
        _pagina("interpretabilidad.py", "Interpretabilidad", ":material/psychology:"),
        _pagina("seleccion.py", "Selección y costo", ":material/fact_check:"),
    ],
    "Despliegue": [
        _pagina("nube.py", "Nube", ":material/cloud:"),
    ],
}

st.navigation(PAGINAS).run()
