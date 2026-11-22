"""1. Inicio: problema, ODS 9, pregunta de mineria y cuatro cifras clave."""

from __future__ import annotations

import streamlit as st

from componentes import carga, pendiente
from componentes.formato import fecha, num, pct

VARIABLES = "data/procesado/registro_variables.json"
SELECCION = "docs/resultados/modelado_seleccion.json"

st.title("Pronóstico de la demanda horaria del SIN a 24 h")
st.caption(
    "Sistema Interconectado Nacional de Colombia · datos abiertos de XM, clima de Open-Meteo y "
    "calendario de festivos · metodología CRISP-DM"
)

izquierda, derecha = st.columns([3, 2], gap="large")
with izquierda:
    st.subheader("El problema")
    st.markdown(
        "La planeación de la infraestructura eléctrica colombiana es sobre todo **reactiva**: la "
        "capacidad se ajusta después de que la demanda ya creció. No anticipar los picos de consumo "
        "se traduce en sobrecostos de operación, riesgo de racionamiento y decisiones de inversión "
        "tomadas con información incompleta."
    )
    st.subheader("Pregunta de minería")
    st.markdown(
        "> ¿Se puede pronosticar la demanda horaria del SIN con 24 horas de anticipación, a partir "
        "del histórico de consumo, el clima y el calendario, con un error en las horas pico lo "
        "bastante bajo para apoyar decisiones de planeación operativa?"
    )
    st.markdown(
        "Es una **regresión supervisada sobre una serie de tiempo**: el objetivo (demanda horaria, "
        "MWh) es continuo y las observaciones dependen del tiempo, lo que condiciona la construcción "
        "de variables y la forma de validar."
    )
with derecha:
    with st.container(border=True):
        st.subheader(":material/public: ODS 9 · meta 9.1")
        st.markdown(
            "*Desarrollar infraestructuras fiables, sostenibles, resilientes y de calidad.* La "
            "fiabilidad de la red depende de la calidad de su planeación, y la planeación depende de "
            "poder anticipar la demanda: el proyecto interviene justo en ese eslabón."
        )
    with st.container(border=True):
        st.subheader(":material/groups: Beneficiarios")
        st.markdown(
            "- **Operador y planeadores**: anticipar picos para dimensionar capacidad.\n"
            "- **Industria**: menos exposición a interrupciones y sobrecostos por restricciones.\n"
            "- **UPME y CREG**: insumo cuantitativo para proyectar infraestructura.\n"
            "- **Ciudadanía**: menos costos de restricción trasladados a la tarifa."
        )

st.divider()
st.subheader("Cifras clave")

variables = carga.json_(VARIABLES)
seleccion = carga.json_(SELECCION)

version = st.radio(
    "Versión de la tabla de modelado",
    ["r24", "r120"],
    horizontal=True,
    help="r24: demanda conocida hasta t − 24 h (backtesting). r120: hasta t − 120 h (disponibilidad real).",
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if variables:
        panel = variables.get("panel", {})
        rango = panel.get("rango", [None, None])
        st.metric("Horas de datos", num(panel.get("filas")), border=True,
                  help=f"Panel de modelado, {fecha(rango[0])} a {fecha(rango[1])}, sin horas faltantes.")
    else:
        pendiente.aviso(VARIABLES)

final = carga.obtener(seleccion, "tablas", version, "final")
baseline = carga.obtener(seleccion, "tablas", version, "metricas_prueba", "baseline")
if final and baseline:
    metricas = final["metricas_prueba"]
    criterio = final.get("criterio_exito_prueba", {})
    meta = carga.obtener(seleccion, "criterio_exito", "mejora_mape_minima_pct")
    with c2:
        st.metric("MAPE en prueba", pct(metricas["mape_pct"], 2), border=True,
                  delta=f"{num(metricas['mape_pct'] - baseline['mape_pct'], 2)} pts frente al baseline",
                  delta_color="inverse",
                  help=f"Modelo final: {final.get('etiqueta')}. Año de prueba completo ({num(metricas['n'])} h).")
    with c3:
        st.metric("Mejora frente al baseline", pct(criterio.get("mejora_mape_pct"), 1), border=True,
                  delta=f"meta ≥ {pct(meta, 0)}" if meta is not None else None, delta_color="off", delta_arrow="off",
                  help="Reducción relativa del MAPE del baseline estacional (misma hora, una semana antes).")
    with c4:
        horas = ", ".join(f"{h}:00" for h in metricas.get("horas_pico", []))
        st.metric("MAPE en horas pico", pct(metricas["mape_pico_pct"], 2), border=True,
                  delta=f"{num(metricas['mape_pico_pct'] - baseline['mape_pico_pct'], 2)} pts frente al baseline",
                  delta_color="inverse", help=f"Horas pico: {horas}.")
    st.caption(
        f"Modelo final {version}: **{final.get('etiqueta')}**, reentrenado con entrenamiento + validación "
        f"y evaluado una sola vez en el año de prueba "
        f"({fecha(carga.obtener(seleccion, 'bloque_prueba', 0))} a {fecha(carga.obtener(seleccion, 'bloque_prueba', 1))})."
    )
else:
    with c2:
        pendiente.aviso(SELECCION)

st.divider()
st.subheader("Recorrido")
st.markdown(
    "El menú lateral sigue las fases de CRISP-DM: **datos** (fuentes, limpieza y las dos versiones "
    "de la tabla), **exploración**, **minería no supervisada**, **modelos**, **predicción frente a "
    "real** en el año de prueba, **alertas de pico**, **interpretabilidad**, **selección del modelo "
    "y costo** y la **arquitectura en la nube**."
)
