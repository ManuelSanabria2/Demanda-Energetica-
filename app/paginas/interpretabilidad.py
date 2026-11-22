"""8. Interpretabilidad. Pagina creada; su contenido llega con el prompt 11 de la cadena."""

from __future__ import annotations

import streamlit as st

from componentes import pendiente

ARTEFACTOS = [
    "docs/resultados/modelado_interpretabilidad.json",
    "docs/figuras/modelado/shap_temperatura_r24.png",
    "docs/figuras/modelado/lime_r24.png",
]

st.title("Interpretabilidad")
st.caption("Importancias, SHAP (resumen y dependencia de la temperatura) y los cuatro casos de LIME.")
pendiente.en_construccion("prompt 11", ARTEFACTOS)
