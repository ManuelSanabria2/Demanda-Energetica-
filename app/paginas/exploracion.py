"""3. Exploracion: serie diaria, perfil hora x dia, autocorrelacion, temperatura y calendario."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from componentes import carga, pendiente
from componentes.estilo import (CATEGORICA, COLOR_TIPO_DIA, DIAS_SEMANA, NOMBRE_TIPO_DIA, REJILLA, SECUENCIAL,
                                TINTA_SECUNDARIA, titulo)
from componentes.formato import num, pct

EDA = "docs/resultados/eda.json"
SERIE = "app/datos/serie_diaria.parquet"
PERFIL = "app/datos/perfil_horario.parquet"
ACF = "app/datos/autocorrelacion.parquet"
MANIFIESTO = "app/datos/manifiesto_interfaz.json"

st.title("Exploración")
st.caption("Qué patrones tiene la serie y qué variable justifica cada uno.")

eda = carga.json_(EDA)
serie = carga.parquet(SERIE)

if eda:
    d = eda.get("descriptivos_mwh", {})
    acf = eda.get("acf", {})
    cal = eda.get("calendario", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demanda media horaria", f"{num(d.get('media'))} MWh", border=True,
              help=f"{num(d.get('n'))} horas · mínimo {num(d.get('minimo'))} · máximo {num(d.get('maximo'))} MWh")
    c2.metric("Coeficiente de variación", pct(d.get("coef_variacion_pct"), 1), border=True)
    c3.metric("Autocorrelación a 24 h / 168 h", f"{num(acf.get('24'), 2)} / {num(acf.get('168'), 2)}", border=True,
              help="La semana pesa más que el día anterior: justifica lag_168 además de lag_24.")
    c4.metric("Festivo frente a laboral", pct(cal.get("festivo_vs_laboral_pct"), 1), border=True,
              help="Diferencia de medianas de la demanda diaria relativa a la media del año.")
else:
    pendiente.aviso(EDA)

# --- Serie diaria -----------------------------------------------------------
st.subheader("Serie diaria")
if serie is not None:
    serie = serie.assign(fecha=pd.to_datetime(serie["fecha"]), gwh=serie["demanda_mwh"] / 1000)
    opciones = st.columns([1, 1, 2])
    por_tipo = opciones[0].toggle("Colorear por tipo de día", value=False)
    con_temperatura = opciones[1].toggle("Superponer temperatura", value=False)

    figura = go.Figure()
    if por_tipo:
        for tipo, grupo in serie.groupby("tipo_dia", sort=False):
            figura.add_trace(go.Scattergl(
                x=grupo["fecha"], y=grupo["gwh"], mode="markers", name=NOMBRE_TIPO_DIA.get(tipo, tipo),
                marker=dict(color=COLOR_TIPO_DIA.get(tipo), size=4),
                customdata=grupo[["nombre_festivo"]].fillna(""),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} GWh %{customdata[0]}<extra></extra>",
            ))
    else:
        figura.add_trace(go.Scatter(x=serie["fecha"], y=serie["gwh"], name="Energía diaria",
                                    line=dict(color=CATEGORICA[0], width=1),
                                    hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} GWh<extra></extra>"))
    media = serie.set_index("fecha")["gwh"].rolling("28D").mean()
    figura.add_trace(go.Scatter(x=media.index, y=media.to_numpy(), name="Media móvil de 28 días",
                                line=dict(color=TINTA_SECUNDARIA, width=2),
                                hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} GWh<extra></extra>"))
    if con_temperatura:
        figura.add_trace(go.Scatter(x=serie["fecha"], y=serie["temperatura_media_c"], name="Temperatura media (°C)",
                                    yaxis="y2", line=dict(color=CATEGORICA[1], width=1), opacity=0.6))
        figura.update_layout(yaxis2=dict(title="°C", overlaying="y", side="right", showgrid=False))
    figura.update_layout(
        height=430, yaxis_title="GWh por día", legend=dict(x=1, xanchor="right"),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(buttons=[
                dict(count=1, label="1 mes", step="month", stepmode="backward"),
                dict(count=6, label="6 meses", step="month", stepmode="backward"),
                dict(count=1, label="1 año", step="year", stepmode="backward"),
                dict(step="all", label="Todo"),
            ]),
        ),
    )
    st.plotly_chart(figura)
    st.caption("Tendencia creciente, ciclo semanal marcado y caídas en festivos, Semana Santa y fin de año. "
               "Arrastra la barra inferior para hacer zoom.")
else:
    pendiente.aviso(SERIE)

if eda and eda.get("crecimiento_anual"):
    anual = pd.DataFrame.from_dict(eda["crecimiento_anual"], orient="index").rename_axis("año").reset_index()
    with st.expander("Crecimiento anual"):
        figura = px.bar(anual, x="año", y="energia_gwh", text=anual["variacion_media_diaria_pct"].map(
            lambda v: "" if pd.isna(v) else f"{v:+.1f} %".replace(".", ",")),
            labels={"energia_gwh": "GWh en el año", "año": ""},
            color=anual["completo"].map({True: "año completo", False: "año incompleto"}),
            color_discrete_sequence=[CATEGORICA[0], REJILLA])
        figura.update_layout(height=320, legend_title_text="")
        st.plotly_chart(figura)
        st.caption("La etiqueta es la variación de la media diaria frente al año anterior; el último año está incompleto.")

# --- Perfil hora x dia ------------------------------------------------------
st.subheader("Perfil hora × día de la semana")
perfil = carga.parquet(PERFIL)
if perfil is not None:
    estadistico = st.radio("Estadístico", ["media", "mediana"], horizontal=True, key="perfil_estadistico")
    columna = "demanda_media_mwh" if estadistico == "media" else "demanda_mediana_mwh"
    matriz = perfil.pivot(index="dia_semana", columns="hora", values=columna).sort_index()
    figura = go.Figure(go.Heatmap(
        z=matriz.to_numpy(), x=list(matriz.columns), y=[DIAS_SEMANA[i] for i in matriz.index],
        colorscale=SECUENCIAL, colorbar=dict(title="MWh"),
        hovertemplate="%{y} %{x}:00 · %{z:.0f} MWh<extra></extra>",
    ))
    figura.update_layout(height=340, xaxis=dict(title="hora", dtick=1), yaxis=dict(autorange="reversed"))
    st.plotly_chart(figura)
    if eda:
        p = eda.get("perfil_horario", {})
        st.caption(f"Pico típico a las {p.get('hora_pico_mediana')}:00 y valle a las {p.get('hora_valle_mediana')}:00; "
                   f"el valle es el {pct(100 * p.get('valle_sobre_pico', 0), 0)} del pico. El domingo es más bajo a toda hora.")
else:
    pendiente.aviso(PERFIL)

# --- Autocorrelacion --------------------------------------------------------
st.subheader("Autocorrelación")
acf = carga.parquet(ACF)
if acf is not None:
    acf = acf[acf["rezago_h"] > 0]
    colores = [CATEGORICA[1] if r % 168 == 0 else (CATEGORICA[0] if r % 24 == 0 else REJILLA) for r in acf["rezago_h"]]
    figura = go.Figure(go.Bar(x=acf["rezago_h"], y=acf["acf"], marker_color=colores, marker_line_width=0,
                              hovertemplate="rezago %{x} h · ACF %{y:.3f}<extra></extra>"))
    banda = carga.obtener(carga.json_(MANIFIESTO), "archivos", "autocorrelacion.parquet", "registro", "banda_95")
    if banda:
        figura.add_hrect(y0=-banda, y1=banda, fillcolor=TINTA_SECUNDARIA, opacity=0.15, line_width=0)
    for rezago, texto in ((24, "24 h"), (168, "168 h (1 semana)"), (336, "336 h")):
        fila = acf[acf["rezago_h"] == rezago]
        if not fila.empty:
            figura.add_annotation(x=rezago, y=float(fila["acf"].iloc[0]), text=texto, showarrow=True, arrowhead=0, ay=-30)
    figura.update_layout(height=340, xaxis=dict(title="rezago (horas)", dtick=24), yaxis_title="ACF", showlegend=False)
    st.plotly_chart(figura)
    st.caption("Picos en múltiplos de 24 h (azul) y más altos en múltiplos de 168 h (naranja): la semana anterior "
               "a la misma hora es el mejor predictor simple. Por eso el baseline es lag_168.")
    if acf["pacf"].notna().any():
        with st.expander("Autocorrelación parcial (hasta 48 h)"):
            parcial = acf.dropna(subset=["pacf"])
            st.plotly_chart(go.Figure(go.Bar(x=parcial["rezago_h"], y=parcial["pacf"], marker_color=CATEGORICA[0]))
                            .update_layout(height=280, xaxis_title="rezago (horas)", yaxis_title="PACF"))
elif eda and eda.get("acf"):
    pendiente.aviso(ACF)
    st.dataframe(pd.Series(eda["acf"], name="ACF").rename_axis("rezago (h)"))
else:
    pendiente.aviso(ACF)

# --- Temperatura ------------------------------------------------------------
st.subheader("Temperatura frente a demanda")
izquierda, derecha = st.columns(2)
with izquierda:
    por_hora = carga.obtener(eda, "temperatura", "spearman_por_hora")
    if por_hora:
        datos = pd.Series(por_hora).rename(index=int).sort_index()
        figura = go.Figure(go.Bar(x=datos.index, y=datos.to_numpy(), marker_color=CATEGORICA[0],
                                  hovertemplate="%{x}:00 · Spearman %{y:.2f}<extra></extra>"))
        figura.update_layout(height=360, xaxis=dict(title="hora", dtick=2), yaxis=dict(title="ρ", range=[0, 1]))
        titulo(figura, "Correlación de Spearman por hora")
        st.plotly_chart(figura)
        t = eda["temperatura"]
        st.caption(f"Global horaria ρ = {num(t.get('spearman_horario'), 2)}, pero gran parte es el ciclo diurno. "
                   f"Sobre las anomalías diarias queda ρ = {num(t.get('spearman_anomalia_diaria'), 2)}. "
                   "El efecto cambia con la hora: una interacción que un modelo lineal no ve por sí solo.")
    else:
        pendiente.aviso(EDA)
with derecha:
    if serie is not None:
        figura = go.Figure()
        for tipo, grupo in serie.groupby("tipo_dia", sort=False):
            figura.add_trace(go.Scattergl(
                x=grupo["temperatura_media_c"], y=grupo["gwh"], mode="markers", name=NOMBRE_TIPO_DIA.get(tipo, tipo),
                marker=dict(color=COLOR_TIPO_DIA.get(tipo), size=5, opacity=0.6),
                customdata=grupo["fecha"].dt.strftime("%Y-%m-%d"),
                hovertemplate="%{customdata}: %{x:.1f} °C · %{y:.1f} GWh<extra></extra>",
            ))
        figura.update_layout(height=360, xaxis_title="temperatura media (°C)", yaxis_title="GWh por día")
        titulo(figura, "Un punto por día")
        st.plotly_chart(figura)
    else:
        pendiente.aviso(SERIE)

# --- Calendario -------------------------------------------------------------
st.subheader("Tipo de día")
if serie is not None:
    relativa = serie["gwh"] / serie.groupby(serie["fecha"].dt.year)["gwh"].transform("mean")
    orden = [t for t in NOMBRE_TIPO_DIA if t in set(serie["tipo_dia"])]
    figura = go.Figure()
    for tipo in orden:
        valores = relativa[serie["tipo_dia"] == tipo]
        figura.add_trace(go.Box(y=valores, name=NOMBRE_TIPO_DIA[tipo], marker_color=COLOR_TIPO_DIA[tipo], boxpoints=False))
    figura.update_layout(height=340, showlegend=False, yaxis_title="demanda diaria / media del año")
    st.plotly_chart(figura)
cal = carga.obtener(eda, "calendario")
if cal:
    st.caption(f"Kruskal-Wallis entre laboral, sábado, domingo y festivo: H = {num(cal.get('kruskal_h'), 0)}, "
               f"p = {cal.get('kruskal_p'):.1e}, ε² = {num(cal.get('epsilon_cuadrado'), 2)}. El calendario separa la "
               "demanda con un efecto grande: justifica las banderas de festivo, víspera, puente y Semana Santa.")
