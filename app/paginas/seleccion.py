"""9. Selección y costo. Pagina creada; su contenido llega con el prompt 11 de la cadena."""

from __future__ import annotations

import streamlit as st

from componentes import pendiente

ARTEFACTOS = [
    "docs/resultados/modelado_seleccion.json",
    "infra/costos/costos.json",
]

st.title("Selección y costo")
st.caption("Tabla de los tres ejes, justificación de la elección, r24 frente a r120 y evaluación económica.")
pendiente.en_construccion("prompt 11", ARTEFACTOS)
