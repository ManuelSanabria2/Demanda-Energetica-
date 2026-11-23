"""4. Mineria no supervisada: grupos de dias, reglas de asociacion y sectores CIIU."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes import carga, pendiente
from componentes.estilo import SECUENCIAL, color_grupo, titulo
from componentes.formato import num, pct

MINERIA = "docs/resultados/mineria.json"
GRUPOS = "docs/resultados/mineria_grupos.csv"
REGLAS = "docs/resultados/mineria_reglas.csv"
SECTORES = "docs/resultados/mineria_sectores.csv"
ETIQUETAS = "docs/resultados/mineria_etiquetas_dia.csv"
CENTROIDES = "app/datos/centroides_grupos.parquet"
TIPOS = ["laboral", "sabado", "domingo", "festivo"]
NOMBRE_TIPO = {"laboral": "Laboral", "sabado": "Sábado", "domingo": "Domingo", "festivo": "Festivo"}

st.title("Minería no supervisada")
st.caption("Qué tipos de día existen según la forma de su curva, qué reglas los describen y qué sectores mueven la carga.")

mineria = carga.json_(MINERIA)
dias = carga.obtener(mineria, "dias") or {}

pestanas = st.tabs(["Grupos de días", "Reglas de asociación", "Sectores CIIU"])

# --- Grupos de dias ---------------------------------------------------------
with pestanas[0]:
    if mineria:
        kmeans = dias.get("kmeans", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Grupos (k)", dias.get("k"), border=True, help=dias.get("k_criterio"))
        c2.metric("Silueta", num(kmeans.get("silueta"), 2), border=True)
        c3.metric("ARI frente a Ward", num(kmeans.get("ari_ward"), 2), border=True,
                  help="Acuerdo entre K-means y un jerárquico de Ward: el agrupamiento no depende del algoritmo.")
        c4.metric("V de Cramér con el tipo de día", num(dias.get("v_cramer_tipo_dia"), 2), border=True,
                  help="Asociación entre el grupo y el calendario, que no entró al K-means.")
        st.markdown(
            "K-means sobre la **forma** de cada día (cada curva dividida por su media), no sobre los MWh: "
            "así la tendencia de crecimiento no separa los días por año. El calendario no entra al "
            "agrupamiento; sirve después para leer qué tienen en común los días de cada grupo."
        )
    else:
        pendiente.aviso(MINERIA)

    centroides = carga.parquet(CENTROIDES)
    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        if centroides is not None:
            figura = go.Figure()
            for grupo, datos in centroides.groupby("grupo"):
                color = color_grupo(grupo)
                figura.add_trace(go.Scatter(x=list(datos["hora"]) + list(datos["hora"])[::-1],
                                            y=list(datos["p90"]) + list(datos["p10"])[::-1], fill="toself",
                                            fillcolor=color, opacity=0.15, line=dict(width=0), hoverinfo="skip",
                                            showlegend=False))
                figura.add_trace(go.Scatter(x=datos["hora"], y=datos["forma_media"], name=f"{grupo} ({num(datos['dias'].iloc[0])} días)",
                                            line=dict(color=color, width=3),
                                            hovertemplate=f"{grupo} %{{x}}:00 · %{{y:.3f}}<extra></extra>"))
            figura.add_hline(y=1, line_dash="dot", line_color="#999")
            figura.update_layout(height=400, xaxis=dict(title="hora", dtick=2), yaxis_title="demanda / media del día")
            titulo(figura, "Centroides: forma media del día (banda p10–p90)")
            st.plotly_chart(figura)
        else:
            pendiente.aviso(CENTROIDES)
    with derecha:
        contingencia = dias.get("contingencia_tipo_dia")
        if contingencia:
            tabla = pd.DataFrame(contingencia).T.reindex(columns=TIPOS).fillna(0)
            figura = go.Figure(go.Heatmap(z=tabla.to_numpy(), x=[NOMBRE_TIPO[t] for t in tabla.columns], y=list(tabla.index),
                                          colorscale=SECUENCIAL, text=tabla.to_numpy().astype(int), texttemplate="%{text}",
                                          showscale=False, hovertemplate="%{y} · %{x}: %{z} días<extra></extra>"))
            figura.update_layout(height=400, yaxis=dict(autorange="reversed"))
            titulo(figura, "Días de cada tipo en cada grupo")
            st.plotly_chart(figura)

    grupos = carga.csv(GRUPOS)
    if grupos is not None:
        columnas = {"grupo": "grupo", "dias": "días", "pct_dias": "% días", "nivel_medio_mwh": "nivel medio (MWh)",
                    "hora_pico": "hora pico", "hora_valle": "hora valle", "factor_carga": "factor de carga",
                    "pct_laboral": "% laboral", "pct_sabado": "% sábado", "pct_domingo": "% domingo",
                    "pct_festivo": "% festivo", "pct_es_puente": "% puente", "dia_mas_comun": "día más común"}
        st.dataframe(grupos[[c for c in columnas if c in grupos]].rename(columns=columnas), hide_index=True)
    else:
        pendiente.aviso(GRUPOS)

    etiquetas = carga.csv(ETIQUETAS)
    if etiquetas is not None:
        with st.expander("Calendario de grupos"):
            etiquetas = etiquetas.assign(fecha=pd.to_datetime(etiquetas["fecha"]))
            figura = go.Figure()
            for grupo, datos in etiquetas.groupby("grupo"):
                figura.add_trace(go.Scattergl(x=datos["fecha"], y=datos["nivel_mwh"], mode="markers", name=grupo,
                                              marker=dict(color=color_grupo(grupo), size=4),
                                              hovertemplate="%{x|%Y-%m-%d %a} · %{y:.0f} MWh<extra></extra>"))
            figura.update_layout(height=320, yaxis_title="nivel medio del día (MWh)")
            st.plotly_chart(figura)
            st.caption("Los días del grupo de fin de semana y festivos se ven como la franja inferior durante todo el periodo.")

# --- Reglas -----------------------------------------------------------------
with pestanas[1]:
    reglas = carga.csv(REGLAS)
    parametros = carga.obtener(dias, "reglas", "parametros") or {}
    if reglas is not None:
        if parametros:
            st.markdown(
                f"Apriori sobre {num(carga.obtener(dias, 'reglas', 'transacciones'))} días, cada uno una canasta con su "
                f"grupo, calendario, nivel y clima. Mínimos del minado: soporte {num(parametros.get('soporte_minimo'), 2)}, "
                f"confianza {num(parametros.get('confianza_minima'), 2)}, lift {num(parametros.get('lift_minimo'), 2)}; "
                f"hasta {parametros.get('max_items')} ítems por regla."
            )
        c1, c2, c3 = st.columns(3)
        soporte = c1.slider("Soporte mínimo", float(reglas["soporte"].min()), float(reglas["soporte"].max()),
                            float(reglas["soporte"].min()), step=0.005, format="%.3f")
        confianza = c2.slider("Confianza mínima", float(reglas["confianza"].min()), 1.0,
                              float(reglas["confianza"].min()), step=0.01, format="%.2f")
        lift = c3.slider("Lift mínimo", float(reglas["lift"].min()), float(reglas["lift"].max()),
                         float(reglas["lift"].min()), step=0.1, format="%.1f")
        texto = st.text_input("Contiene (antecedente o consecuente)", placeholder="por ejemplo: grupo=G2, festivo, calor")
        filtro = (reglas["soporte"] >= soporte) & (reglas["confianza"] >= confianza) & (reglas["lift"] >= lift)
        if texto:
            filtro &= (reglas["antecedente"].str.contains(texto, case=False, regex=False)
                       | reglas["consecuente"].str.contains(texto, case=False, regex=False))
        visibles = reglas[filtro].sort_values("lift", ascending=False)
        st.caption(f"{len(visibles)} de {len(reglas)} reglas")
        st.dataframe(
            visibles, hide_index=True,
            column_config={
                "antecedente": st.column_config.TextColumn("si…", width="large"),
                "consecuente": st.column_config.TextColumn("entonces…"),
                "soporte": st.column_config.ProgressColumn("soporte", format="%.3f", min_value=0,
                                                            max_value=float(reglas["soporte"].max())),
                "confianza": st.column_config.NumberColumn("confianza", format="%.2f"),
                "lift": st.column_config.NumberColumn("lift", format="%.2f"),
                "n_items": st.column_config.NumberColumn("ítems"),
            },
        )
        if mineria:
            st.caption(f"{carga.obtener(dias, 'reglas', 'n_reglas')} reglas minadas, "
                       f"{carga.obtener(dias, 'reglas', 'n_reglas_no_redundantes')} sin redundancia "
                       "(una regla es redundante si otra más corta con el mismo consecuente tiene al menos su confianza).")
    else:
        pendiente.aviso(REGLAS)

# --- Sectores ---------------------------------------------------------------
with pestanas[2]:
    sectores_json = carga.obtener(mineria, "sectores") or {}
    resumen = sectores_json.get("grupos")
    if resumen:
        columnas = st.columns(len(resumen))
        for columna, (grupo, datos) in zip(columnas, sorted(resumen.items())):
            with columna, st.container(border=True):
                st.markdown(f"#### :material/factory: {grupo}")
                a, b = st.columns(2)
                a.metric("Sectores", num(datos.get("sectores")))
                b.metric("Energía", pct(datos.get("pct_energia"), 1))
                a.metric("Domingo / laboral", num(datos.get("domingo_sobre_laboral"), 2))
                b.metric("Noche / día", num(datos.get("noche_sobre_dia"), 2))
                st.caption(f"Factor de carga {num(datos.get('factor_carga'), 2)} · manufactura: "
                           f"{pct(datos.get('pct_sectores_manufactura'), 1)} de los sectores y "
                           f"{pct(datos.get('pct_energia_manufactura'), 1)} de su energía.")
        st.markdown(
            "Sectores de la demanda no regulada agrupados por su perfil semanal de 168 h. Un grupo opera "
            "casi continuo (proceso, minería, petróleo) y concentra casi toda la energía; el otro sigue "
            "el horario laboral y cae de noche y en domingo."
        )
    elif mineria is None:
        pendiente.aviso(MINERIA)

    sectores = carga.csv(SECTORES)
    if sectores is not None:
        figura = go.Figure()
        for grupo, datos in sectores.groupby("grupo"):
            figura.add_trace(go.Scatter(
                x=datos["domingo_sobre_laboral"], y=datos["noche_sobre_dia"], mode="markers", name=grupo,
                marker=dict(color=color_grupo(grupo), size=(datos["energia_kwh"] ** 0.5) / (sectores["energia_kwh"].max() ** 0.5) * 40 + 4,
                            opacity=0.6, line=dict(width=0)),
                customdata=datos[["subactivity", "activity"]],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>domingo/laboral %{x:.2f} · noche/día %{y:.2f}<extra></extra>",
            ))
        figura.update_layout(height=460, xaxis_title="demanda del domingo / día laboral",
                             yaxis=dict(title="demanda de noche / de día (escala log)", type="log"))
        titulo(figura, "Un punto por sector (tamaño: energía)")
        st.plotly_chart(figura)
        st.caption("El eje vertical es logarítmico: unos pocos sectores (sobre todo de la administración pública) "
                   "consumen mucho más de noche que de día y aplastarían al resto en escala lineal.")
        with st.expander("Sectores con más energía de cada grupo"):
            mayores = sectores.sort_values("energia_kwh", ascending=False).groupby("grupo").head(5)
            st.dataframe(
                mayores.assign(energia_gwh=mayores["energia_kwh"] / 1e6)[["grupo", "activity", "subactivity", "energia_gwh"]]
                .rename(columns={"activity": "actividad", "subactivity": "subactividad", "energia_gwh": "energía (GWh)"}),
                hide_index=True, column_config={"energía (GWh)": st.column_config.NumberColumn(format="%.0f")},
            )
    else:
        pendiente.aviso(SECTORES)
