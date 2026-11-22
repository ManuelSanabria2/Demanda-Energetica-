"""5. Modelos. Pagina creada; su contenido llega con el prompt 11 de la cadena."""

from __future__ import annotations

import streamlit as st

from componentes import pendiente

ARTEFACTOS = [
    "docs/resultados/modelado_baselines.json",
    "docs/resultados/modelado_regresion.json",
    "docs/resultados/modelado_clasificacion.json",
    "docs/resultados/modelado_ensembles.json",
    "docs/resultados/modelado_busqueda.json",
]

st.title("Modelos")
st.caption("Tabla filtrable de todos los modelos, MAPE frente al baseline, error por pliegue y Grid frente a Random Search.")
pendiente.en_construccion("prompt 11", ARTEFACTOS)
