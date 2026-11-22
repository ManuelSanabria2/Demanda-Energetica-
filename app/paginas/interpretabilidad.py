"""8. Interpretabilidad: importancias, SHAP (resumen y temperatura) y los casos de LIME."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes import carga, modelado, pendiente
from componentes.estilo import CATEGORICA, TINTA_SECUNDARIA, titulo
from componentes.formato import num

INTERPRETABILIDAD = "docs/resultados/modelado_interpretabilidad.json"
NOMBRE_CASO = {
    "pico_laboral": "Pico de un día laboral",
    "festivo": "Festivo",
    "semana_santa": "Semana Santa",
    "mayor_error_ensemble": "Mayor error del ensemble",
    "mayor_error_lineal": "Mayor error del lineal",
}

st.title("Interpretabilidad")
st.caption("Qué variables usa cada modelo candidato y por qué predice lo que predice. Todo se mide en validación.")

datos = carga.json_(INTERPRETABILIDAD)
if datos is None:
    pendiente.aviso(INTERPRETABILIDAD)
    st.stop()

version = st.radio("Versión de la tabla", list(datos.get("tablas", {})), horizontal=True, key="interp_version")
tabla = datos["tablas"][version]
modelos = tabla.get("modelos", {})
rol = st.radio("Modelo", list(modelos), horizontal=True, key="interp_rol",
               format_func=lambda r: f"{r}: {carga.obtener(modelos, r, 'candidato', 'nombre')} "
                                     f"({carga.obtener(modelos, r, 'calificacion', 'calificacion')})")
modelo = modelos[rol]
calificacion = modelo.get("calificacion", {})
st.info(f"**Interpretabilidad {calificacion.get('calificacion')}**: {calificacion.get('motivo')}", icon=":material/psychology:")

pestanas = st.tabs(["Importancias", "SHAP", "Temperatura", "LIME", "¿Confirma el EDA?"])

# --- Importancias -----------------------------------------------------------
with pestanas[0]:
    importancias = modelo.get("importancias", {})
    permutacion = carga.obtener(importancias, "permutacion", "por_variable") or {}
    impureza = carga.obtener(importancias, "impureza", "por_variable") or {}
    columnas = st.columns(2 if impureza else 1)
    if permutacion:
        serie = pd.DataFrame(permutacion).T.sort_values("media")
        figura = go.Figure(go.Bar(x=serie["media"], y=serie.index, orientation="h", marker_color=CATEGORICA[0],
                                  error_x=dict(type="data", array=serie["desviacion"], color=TINTA_SECUNDARIA),
                                  hovertemplate="%{y}: +%{x:.1f} MWh de MAE<extra></extra>"))
        figura.update_layout(height=max(380, 24 * len(serie) + 100), xaxis_title=carga.obtener(importancias, "permutacion", "unidad"))
        titulo(figura, "Permutación (en validación)")
        columnas[0].plotly_chart(figura)
    if impureza:
        serie = pd.Series(impureza).sort_values()
        figura = go.Figure(go.Bar(x=serie.to_numpy(), y=serie.index, orientation="h", marker_color=CATEGORICA[1],
                                  hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        figura.update_layout(height=max(380, 24 * len(serie) + 100), xaxis_title="fracción de la ganancia")
        titulo(figura, "Impureza (en entrenamiento)")
        columnas[1].plotly_chart(figura)
    motivos = tabla.get("motivos_diferencia_impureza_permutacion") or []
    if motivos and impureza:
        with st.expander("Por qué las dos medidas no coinciden"):
            st.markdown("\n".join(f"- {m}" for m in motivos))

# --- SHAP -------------------------------------------------------------------
with pestanas[1]:
    shap = modelo.get("shap", {})
    media_abs = shap.get("media_abs_por_variable_mwh") or {}
    if media_abs:
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor base", f"{num(shap.get('valor_base_mwh'))} MWh", border=True,
                  help="Predicción media; cada variable suma o resta a partir de aquí.")
        c2.metric("Filas explicadas", num(shap.get("filas_explicadas")), border=True, help=shap.get("motivo_muestra"))
        c3.metric("Error de aditividad máximo", f"{shap.get('error_aditividad_max_mwh'):.1e} MWh", border=True,
                  help="Valor base + suma de SHAP − predicción: cero salvo redondeo, así que la explicación es exacta.")
        izquierda, derecha = st.columns(2)
        serie = pd.Series(media_abs).sort_values()
        figura = go.Figure(go.Bar(x=serie.to_numpy(), y=serie.index, orientation="h", marker_color=CATEGORICA[2],
                                  hovertemplate="%{y}: %{x:.1f} MWh<extra></extra>"))
        figura.update_layout(height=max(420, 24 * len(serie) + 100), xaxis_title="|SHAP| medio (MWh)")
        titulo(figura, "Cuánto mueve cada variable la predicción, en promedio")
        izquierda.plotly_chart(figura)
        with derecha:
            resumen = carga.imagen(shap["figura_resumen"]) if shap.get("figura_resumen") else None
            if resumen:
                st.markdown("**Resumen de SHAP** (un punto por hora de validación; color: valor de la variable)")
                st.image(resumen, width="stretch")
            elif shap.get("figura_resumen"):
                pendiente.aviso(shap["figura_resumen"])
        st.caption(f"Método: `{shap.get('metodo')}`.")

# --- Temperatura ------------------------------------------------------------
with pestanas[2]:
    temperatura = shap.get("temperatura") or {}
    franjas = temperatura.get("pendiente_por_franja_mwh_por_grado") or {}
    if temperatura.get("disponible") and franjas:
        st.metric("Efecto de 1 °C más, en promedio", f"{num(temperatura.get('pendiente_global_mwh_por_grado'), 1)} MWh",
                  help="Pendiente del SHAP de temperatura_c frente a la temperatura.")
        serie = pd.Series(franjas)
        figura = go.Figure(go.Bar(x=serie.index, y=serie.to_numpy(), marker_color=CATEGORICA[1],
                                  text=[num(v, 1) for v in serie], textposition="outside",
                                  hovertemplate="%{x}: %{y:.1f} MWh por °C<extra></extra>"))
        figura.update_layout(height=340, yaxis_title="MWh por °C")
        titulo(figura, "Efecto de la temperatura por franja horaria")
        st.plotly_chart(figura)
        if serie.max() - serie.min() <= 0.01 * abs(serie.max()):
            st.caption("La pendiente es la misma a toda hora: este modelo no puede ver la interacción entre "
                       "temperatura y hora que el EDA encontró en la correlación por hora.")
        else:
            st.caption(f"La pendiente va de {num(serie.min(), 1)} MWh/°C ({serie.idxmin()}) a "
                       f"{num(serie.max(), 1)} MWh/°C ({serie.idxmax()}): el modelo capta que el efecto de la "
                       "temperatura cambia con la hora, como anticipaba la correlación por hora del EDA.")
    figura_temp = carga.obtener(tabla, "figuras", "shap_temperatura")
    if figura_temp:
        imagen = carga.imagen(figura_temp)
        if imagen:
            st.markdown("**Dependencia de SHAP con la temperatura** (figura del informe, los dos candidatos)")
            st.image(imagen, width="stretch")
        else:
            pendiente.aviso(figura_temp)

# --- LIME -------------------------------------------------------------------
with pestanas[3]:
    casos = carga.obtener(tabla, "lime", "casos") or {}
    configuracion = carga.obtener(tabla, "lime", "configuracion") or {}
    disponibles = {clave: caso[rol] for clave, caso in casos.items() if rol in caso}
    if disponibles:
        st.markdown("LIME ajusta un modelo lineal **local** alrededor de una hora concreta para explicar esa predicción. "
                    "Cada barra es una regla sobre una variable y lo que suma (naranja) o resta (azul) en MWh.")
        claves = list(disponibles)
        for inicio in range(0, len(claves), 2):
            columnas = st.columns(2)
            for columna, clave in zip(columnas, claves[inicio:inicio + 2]):
                caso = disponibles[clave]
                with columna, st.container(border=True):
                    st.markdown(f"#### {NOMBRE_CASO.get(clave, clave)}")
                    st.caption(f"{str(caso.get('fecha_hora'))[:16]} · {caso.get('regla')}")
                    st.markdown(modelado.lectura_lime(caso))
                    pesos = pd.DataFrame(caso.get("pesos") or [])
                    if not pesos.empty:
                        pesos = pesos.iloc[pesos["peso_mwh"].abs().argsort()]
                        figura = go.Figure(go.Bar(
                            x=pesos["peso_mwh"], y=pesos["regla"], orientation="h",
                            marker_color=[CATEGORICA[1] if v > 0 else CATEGORICA[0] for v in pesos["peso_mwh"]],
                            hovertemplate="%{y}: %{x:+.0f} MWh<extra></extra>"))
                        figura.update_layout(height=300, xaxis_title="aporte (MWh)", margin=dict(t=10))
                        st.plotly_chart(figura, key=f"lime_{version}_{rol}_{clave}")
        if configuracion:
            st.caption(f"`{configuracion.get('explicador')}`, {configuracion.get('num_samples')} muestras, "
                       f"{configuracion.get('num_features')} variables, semilla {configuracion.get('semilla')}. "
                       "R² local: qué tan bien el modelo lineal local imita al modelo cerca de esa hora.")
    else:
        st.info("No hay casos de LIME para este modelo.")
    figura_lime = carga.obtener(tabla, "figuras", "lime")
    if figura_lime and carga.imagen(figura_lime):
        with st.expander("Figura de LIME del informe"):
            st.image(carga.imagen(figura_lime), width="stretch")

# --- Contraste con el EDA ---------------------------------------------------
with pestanas[4]:
    contraste = tabla.get("contraste_eda") or {}
    if contraste:
        st.markdown(f"**Hipótesis del EDA:** {contraste.get('hipotesis')} "
                    f"(ACF 24 h = {num(contraste.get('acf_24'), 3)}, ACF 168 h = {num(contraste.get('acf_168'), 3)}).")
        st.markdown(f"**Resultado:** {contraste.get('conclusion')}")
        medidas = pd.DataFrame(contraste.get("medidas") or [])
        if not medidas.empty:
            medidas["tres_primeras"] = medidas["tres_primeras"].map(", ".join)
            st.dataframe(medidas, hide_index=True)
    calificaciones = tabla.get("calificaciones") or {}
    if calificaciones:
        with st.expander("Calificación de interpretabilidad de todos los modelos"):
            st.caption(tabla.get("criterio_calificacion", ""))
            st.dataframe(pd.DataFrame(calificaciones).T.rename_axis("modelo").reset_index(), hide_index=True)
