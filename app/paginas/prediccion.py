"""6. Prediccion frente a real: una semana del ano de prueba y el error por hora y dia."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes import carga, modelado, pendiente
from componentes.estilo import CATEGORICA, DIAS_SEMANA, SECUENCIAL, TINTA, TINTA_SECUNDARIA, titulo
from componentes.formato import fecha, pct

PREDICCIONES = "app/datos/predicciones_prueba.parquet"
SERIE = "app/datos/serie_diaria.parquet"
SELECCION = "docs/resultados/modelado_seleccion.json"
ESTILO_ROL = {
    "baseline": dict(color=TINTA_SECUNDARIA, width=1.5, dash="dash"),
    "lineal": dict(color=CATEGORICA[0], width=1.8),
    "ensemble": dict(color=CATEGORICA[1], width=1.8),
}
NOMBRE_ROL = {"baseline": "Baseline", "lineal": "Lineal", "ensemble": "Ensemble"}

st.title("Predicción frente a real")
st.caption("El año de prueba se usó una sola vez, al final, con los modelos ya elegidos en validación.")

predicciones = carga.parquet(PREDICCIONES)
seleccion = carga.json_(SELECCION)
if predicciones is None:
    pendiente.aviso(PREDICCIONES)
    st.stop()

version = st.radio("Versión de la tabla", sorted(predicciones["tabla"].unique(), key=lambda v: int(v[1:])),
                   horizontal=True, key="prediccion_version")
datos = predicciones[predicciones["tabla"] == version].copy()
datos["fecha_hora"] = pd.to_datetime(datos["fecha_hora"]).dt.tz_localize(None)
roles = [r for r in ("baseline", "lineal", "ensemble") if r in set(datos["rol"])]
nombre_modelo = datos.groupby("rol")["modelo"].first().to_dict()
final = datos.loc[datos["es_final"], "rol"].iloc[0] if datos["es_final"].any() else roles[-1]

# --- Semana -----------------------------------------------------------------
st.subheader("Una semana")
lunes = pd.date_range(datos["fecha_hora"].min().normalize(), datos["fecha_hora"].max(), freq="W-MON")
inicio_figura = carga.obtener(seleccion, "archivos", "semana_figura_desde")
por_defecto = pd.Timestamp(inicio_figura) if inicio_figura else lunes[len(lunes) // 2]
por_defecto = lunes[lunes.get_indexer([por_defecto], method="nearest")[0]]
desde = st.select_slider("Semana (lunes)", options=list(lunes), value=por_defecto,
                         format_func=lambda d: d.strftime("%Y-%m-%d"))
visibles = st.multiselect("Curvas", roles, default=roles, format_func=lambda r: f"{NOMBRE_ROL[r]}: {nombre_modelo[r]}")

semana = datos[(datos["fecha_hora"] >= desde) & (datos["fecha_hora"] < desde + pd.Timedelta(days=7))]
figura = go.Figure()
real = semana[semana["rol"] == final]
figura.add_trace(go.Scatter(x=real["fecha_hora"], y=real["real_mwh"], name="Real", line=dict(color=TINTA, width=2.5),
                            hovertemplate="%{x|%a %d %H:00} · real %{y:.0f} MWh<extra></extra>"))
for rol in visibles:
    serie = semana[semana["rol"] == rol]
    estilo = dict(ESTILO_ROL[rol]) | ({"width": 3} if rol == final else {})
    figura.add_trace(go.Scatter(x=serie["fecha_hora"], y=serie["prediccion_mwh"], line=estilo,
                                name=f"{NOMBRE_ROL[rol]}{' (final)' if rol == final else ''}",
                                hovertemplate=f"%{{x|%a %d %H:00}} · {NOMBRE_ROL[rol]} %{{y:.0f}} MWh<extra></extra>"))

serie_diaria = carga.parquet(SERIE)
especiales = []
if serie_diaria is not None:
    dias = serie_diaria.assign(fecha=pd.to_datetime(serie_diaria["fecha"]))
    dias = dias[(dias["fecha"] >= desde) & (dias["fecha"] < desde + pd.Timedelta(days=7))]
    for _, dia in dias[dias["es_festivo"] | (dias["tipo_dia"] == "semana_santa")].iterrows():
        texto = dia["nombre_festivo"] if isinstance(dia["nombre_festivo"], str) else "Semana Santa"
        especiales.append(f"{dia['fecha']:%a %d}: {texto}")
        figura.add_vrect(x0=dia["fecha"], x1=dia["fecha"] + pd.Timedelta(days=1), fillcolor=CATEGORICA[3], opacity=0.12,
                         line_width=0, annotation_text=texto[:28], annotation_position="top left",
                         annotation_font_size=11)
else:
    pendiente.aviso(SERIE)
figura.update_layout(height=440, yaxis_title="MWh por hora", xaxis=dict(dtick=86400000, tickformat="%a %d"))
st.plotly_chart(figura)

columnas = st.columns(len(roles))
for columna, rol in zip(columnas, roles):
    serie = semana[semana["rol"] == rol]
    columna.metric(f"MAPE de la semana · {NOMBRE_ROL[rol]}", pct(modelado.mape(serie["real_mwh"], serie["prediccion_mwh"]), 2),
                   border=True, help=nombre_modelo[rol])
st.caption("Festivos y Semana Santa sombreados." + (" Esta semana: " + "; ".join(especiales) + "." if especiales else ""))

# --- Error por hora y dia ---------------------------------------------------
st.subheader("Dónde se equivoca: error por hora y día de la semana")
rol_mapa = st.radio("Modelo", roles, index=roles.index(final), horizontal=True, key="prediccion_rol_mapa",
                    format_func=lambda r: f"{NOMBRE_ROL[r]}: {nombre_modelo[r]}")
matriz = modelado.error_hora_dia(datos[datos["rol"] == rol_mapa])
figura = go.Figure(go.Heatmap(
    z=matriz.to_numpy(), x=list(matriz.columns), y=[DIAS_SEMANA[i] for i in matriz.index], colorscale=SECUENCIAL,
    colorbar=dict(title="MAPE %"), hovertemplate="%{y} %{x}:00 · MAPE %{z:.2f} %<extra></extra>",
))
horas_pico = carga.obtener(seleccion, "horas_pico") or []
for hora in horas_pico:
    figura.add_vline(x=hora, line_color=CATEGORICA[7], line_width=1, opacity=0.4)
figura.update_layout(height=340, xaxis=dict(title="hora", dtick=1), yaxis=dict(autorange="reversed"))
st.plotly_chart(figura)
st.caption("MAPE medio del año de prueba en cada celda." + (f" Líneas rojas: horas pico ({', '.join(map(str, horas_pico))} h)."
                                                           if horas_pico else ""))

# --- Error diario a lo largo del ano ----------------------------------------
st.subheader("Error diario a lo largo del año de prueba")
datos["fecha"] = datos["fecha_hora"].dt.normalize()
datos["ape"] = 100 * (datos["prediccion_mwh"] - datos["real_mwh"]).abs() / datos["real_mwh"]
diario = datos.groupby(["rol", "fecha"])["ape"].mean().reset_index()
figura = go.Figure()
for rol in roles:
    serie = diario[diario["rol"] == rol]
    figura.add_trace(go.Scatter(x=serie["fecha"], y=serie["ape"], name=NOMBRE_ROL[rol],
                                line=dict(ESTILO_ROL[rol]) | {"width": 1.2}, opacity=1 if rol in (final, "baseline") else 0.6,
                                visible=True if rol in (final, "baseline") else "legendonly",
                                hovertemplate=f"%{{x|%Y-%m-%d %a}} · {NOMBRE_ROL[rol]} %{{y:.2f}} %<extra></extra>"))
figura.update_layout(height=340, yaxis_title="MAPE del día (%)")
titulo(figura, "MAPE de cada día: baseline frente al modelo final")
st.plotly_chart(figura)

with st.expander("Los diez peores días de cada modelo"):
    tipos = (serie_diaria.assign(fecha=pd.to_datetime(serie_diaria["fecha"]))[["fecha", "tipo_dia", "nombre_festivo"]]
             if serie_diaria is not None else pd.DataFrame(columns=["fecha", "tipo_dia", "nombre_festivo"]))
    columnas = st.columns(2)
    for columna, rol in zip(columnas, ["baseline", final]):
        peores = diario[diario["rol"] == rol].nlargest(10, "ape").merge(tipos, on="fecha", how="left")
        with columna:
            st.markdown(f"**{NOMBRE_ROL[rol]}** · {nombre_modelo[rol]}")
            st.dataframe(
                peores.assign(fecha=peores["fecha"].dt.strftime("%Y-%m-%d %a"))[["fecha", "ape", "tipo_dia", "nombre_festivo"]],
                hide_index=True,
                column_config={"ape": st.column_config.NumberColumn("MAPE %", format="%.2f"),
                               "tipo_dia": "tipo de día", "nombre_festivo": "festivo"},
            )
    st.caption("El baseline repite la semana anterior, así que falla en los festivos y en los días que siguen a uno.")
bloque = carga.obtener(seleccion, "bloque_prueba") or []
if len(bloque) == 2:
    st.caption(f"Año de prueba: {fecha(bloque[0])} a {fecha(bloque[1])}. Haz clic en la leyenda para mostrar u ocultar modelos.")
