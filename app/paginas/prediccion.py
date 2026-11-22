"""6. Predicción frente a real. Pagina creada; su contenido llega con el prompt 11 de la cadena."""

from __future__ import annotations

import streamlit as st

from componentes import pendiente

ARTEFACTOS = [
    "app/datos/predicciones_prueba.parquet",
    "app/datos/serie_diaria.parquet",
    "docs/resultados/modelado_seleccion.json",
]

st.title("Predicción frente a real")
st.caption("Selector de semana del año de prueba con las curvas real, baseline y modelo, festivos marcados y mapa de calor del error por hora y día.")
pendiente.en_construccion("prompt 11", ARTEFACTOS)
