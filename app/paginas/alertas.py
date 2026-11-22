"""7. Alertas de pico. Pagina creada; su contenido llega con el prompt 11 de la cadena."""

from __future__ import annotations

import streamlit as st

from componentes import pendiente

ARTEFACTOS = [
    "app/datos/probabilidades_alerta.parquet",
    "docs/resultados/modelado_clasificacion.json",
    "docs/resultados/modelado_seleccion.json",
]

st.title("Alertas de pico")
st.caption("Matriz de confusión y curva precisión-recall con un deslizador de umbral que recalcula ambas.")
pendiente.en_construccion("prompt 11", ARTEFACTOS)
