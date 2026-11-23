"""2. Datos: fuentes, flujo ingesta -> limpieza -> panel -> tabla, calidad, r24 frente a r120."""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes import carga, pendiente
from componentes.estilo import CATEGORICA, REJILLA, TINTA, TINTA_SECUNDARIA
from componentes.formato import fecha, num, pct

LIMPIEZA = "data/procesado/registro_limpieza.json"
VARIABLES = "data/procesado/registro_variables.json"

st.title("Datos")
st.caption("De dónde sale cada valor y cuánta demanda se conoce al momento de predecir.")

st.subheader("Fuentes")
st.dataframe(
    pd.DataFrame(
        [
            ["XM · API SINERGOX", "DemaReal / Sistema", "Demanda real horaria del SIN (kWh). Serie objetivo.",
             "POST servapibi.xm.com.co"],
            ["XM · API SIMEM", "Dataset 14fabb / Nacional", "Demanda real nacional; contraste y reconciliación con SINERGOX.",
             "GET simem.co/backend-files/api/PublicData"],
            ["Open-Meteo (archivo)", "Temperatura horaria", "Temperatura de las principales ciudades, ponderada por demanda.",
             "archive-api.open-meteo.com"],
            ["Calendario", "Festivos de Colombia", "Festivos (Ley Emiliani), vísperas, puentes, Semana Santa y fin de año.",
             "paquete holidays, country_holidays('CO')"],
        ],
        columns=["Fuente", "Serie", "Para qué", "Acceso"],
    ),
    hide_index=True,
)

limpieza = carga.json_(LIMPIEZA)
variables = carga.json_(VARIABLES)

st.subheader("Flujo")
if limpieza and variables:
    ultima = limpieza["limpiezas"][-1]
    panel = variables["panel"]
    versiones = variables["versiones"]
    procedencia = ultima.get("procedencia", {})
    etiqueta_tabla = {
        v: f"tabla_modelado_{v}\\n{num(d['filas_finales'])} filas × {d['n_columnas']} columnas\\nrezago mínimo {d['rezago_minimo_h']} h"
        for v, d in versiones.items()
    }
    dot = f"""
    digraph {{
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#f3f7fd", color="{CATEGORICA[0]}", fontname="Helvetica", fontsize=11];
      edge [color="{TINTA_SECUNDARIA}"];
      ingesta [label="Ingesta\\nXM, SIMEM, Open-Meteo\\ncapa cruda particionada"];
      limpieza [label="Limpieza trazable\\n{num(ultima['filas_finales'])} horas\\n{pct(procedencia.get('pct_observado'), 1)} observadas"];
      panel [label="Panel de modelado\\n{num(panel['filas'])} filas × {panel['columnas']} columnas\\ndemanda + clima + calendario"];
      {" ".join(f'{v} [label="{etiqueta_tabla[v]}", fillcolor="#ffffff"];' for v in versiones)}
      ingesta -> limpieza -> panel;
      {" ".join(f"panel -> {v};" for v in versiones)}
    }}"""
    st.graphviz_chart(dot)
    st.caption(f"Cifras leídas de `{LIMPIEZA}` y `{VARIABLES}`. Imputación del panel: {panel.get('imputacion')}.")
else:
    pendiente.requerir(LIMPIEZA, VARIABLES)

st.subheader("Calidad de la serie objetivo")
if limpieza:
    ultima = limpieza["limpiezas"][-1]
    procedencia = ultima.get("procedencia", {})
    operaciones = {o["operacion"]: o.get("detalle", {}) for o in ultima.get("operaciones", [])}
    rejilla = operaciones.get("completar_rejilla", {})
    atipicos = operaciones.get("marcar_atipicos", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Horas observadas", pct(procedencia.get("pct_observado"), 1), border=True,
              help=f"Interpoladas: {pct(procedencia.get('pct_interpolado'), 1)} · faltantes: {pct(procedencia.get('pct_faltante'), 1)}")
    c2.metric("Horas faltantes", num(rejilla.get("horas_faltantes_totales")), border=True,
              help="Huecos en la rejilla horaria antes de imputar (solo se imputan huecos de hasta 3 h).")
    c3.metric("Atípicos marcados (causal)", num(atipicos.get("n_atipicos_estacionales", procedencia.get("n_atipicos"))),
              border=True, help="Cada hora se juzga solo contra las anteriores de su mismo día de la semana y hora. "
                                "Se usan para filtrar, nunca como variable.")
    c4.metric("Atípicos con la serie entera", num(atipicos.get("n_atipicos_estacionales_global")), border=True,
              help="El mismo criterio mirando también el futuro: solo diagnóstico, es fuga temporal.")
    st.markdown(
        "Los atípicos **se marcan, nunca se eliminan ni se corrigen**: decidir qué hacer con ellos le "
        "corresponde al modelado. Se prefiere un `NaN` honesto a un valor inventado."
    )
    with st.expander("Historial del registro de limpieza"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "momento": l.get("momento"),
                        "filas iniciales": l.get("filas_iniciales"),
                        "filas finales": l.get("filas_finales"),
                        "% observado": l.get("procedencia", {}).get("pct_observado"),
                        "atípicos": l.get("procedencia", {}).get("n_atipicos"),
                    }
                    for l in limpieza["limpiezas"]
                ]
            ),
            hide_index=True,
        )
        st.caption("La primera limpieza tenía un hueco de marzo de 2025 que luego se reingestó.")
else:
    pendiente.aviso(LIMPIEZA)

st.subheader("r24 frente a r120: cuánta demanda se conoce al predecir")
if variables:
    versiones = variables["versiones"]
    st.markdown(
        "La demanda real se publica con varios días de rezago y los dos últimos días llegan parciales. "
        "Por eso la tabla de modelado existe en dos versiones: **r24** supone conocida la demanda hasta "
        "`t − 24 h` (backtesting académico) y **r120** solo hasta `t − 120 h` (lo que de verdad está "
        "publicado al predecir). La diferencia de error entre las dos es lo que cuesta el rezago."
    )

    figura = go.Figure()
    nombres = list(versiones)
    inicio = -(max(d["rezago_minimo_h"] for d in versiones.values()) + 168 + 48)
    for fila, (nombre, datos) in enumerate(versiones.items()):
        rezago = datos["rezago_minimo_h"]
        lags = sorted(int(m.group(1)) for v in datos["variables"] if (m := re.fullmatch(r"lag_(\d+)", v)))
        figura.add_trace(go.Bar(
            y=[nombre], x=[-rezago - inicio], base=[inicio], orientation="h",
            width=0.7, marker_color=CATEGORICA[0], opacity=0.25, name="Demanda conocida", showlegend=fila == 0,
            hovertemplate=f"{nombre}: conocida hasta t − {rezago} h<extra></extra>",
        ))
        figura.add_trace(go.Bar(
            y=[nombre], x=[168], base=[-(rezago + 168)], orientation="h", width=0.25,
            marker_color=CATEGORICA[0], name="Ventana de ma_168 y std_168", showlegend=fila == 0,
            hovertemplate=f"{nombre}: media y desviación de t − {rezago + 168} h a t − {rezago} h<extra></extra>",
        ))
        figura.add_trace(go.Bar(
            y=[nombre], x=[rezago], base=[-rezago], orientation="h",
            width=0.7, marker_color=REJILLA, name="Aún no publicada", showlegend=fila == 0,
            hovertemplate=f"{nombre}: de t − {rezago} h a t no se conoce<extra></extra>",
        ))
        figura.add_trace(go.Scatter(
            y=[nombre] * len(lags), x=[-l for l in lags], mode="markers+text", text=[f"lag_{l}" for l in lags],
            textposition="top center", textfont=dict(color=TINTA, size=13),
            marker=dict(color=CATEGORICA[1], size=13, symbol="diamond", line=dict(color="white", width=1)),
            name="Rezagos usados", showlegend=fila == 0, hovertemplate="%{text}: demanda de t − %{customdata} h<extra></extra>",
            customdata=lags,
        ))
    figura.add_vline(x=0, line_color=CATEGORICA[7], line_width=2,
                     annotation_text="t: hora a predecir", annotation_position="top")
    figura.update_layout(barmode="overlay", height=280, xaxis_title="horas respecto a la hora que se predice",
                         yaxis=dict(categoryorder="array", categoryarray=nombres[::-1]))
    st.plotly_chart(figura)

    filas = []
    for nombre, datos in versiones.items():
        balance = datos.get("balance_filas", {})
        filas.append({
            "versión": nombre,
            "rezago mínimo (h)": datos["rezago_minimo_h"],
            "filas del panel": balance.get("panel"),
            "arranque sin histórico": balance.get("arranque_sin_historico"),
            "publicación parcial": balance.get("publicacion_parcial"),
            "filas finales": balance.get("final", datos.get("filas_finales")),
            "variables": len(datos["variables"]),
            "desde": fecha(datos["rango"][0]),
            "hasta": fecha(datos["rango"][1]),
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True)

    columnas = st.columns(len(versiones))
    for columna, (nombre, datos) in zip(columnas, versiones.items()):
        with columna:
            st.markdown(f"**Variables de {nombre}**")
            st.markdown(" ".join(f"`{v}`" for v in datos["variables"]))
            for omitido in datos.get("rezagos_omitidos", []):
                st.caption(f"Sin `{omitido['columna']}`: {omitido['motivo']}.")
    st.info(
        "**Sin fuga temporal.** Ninguna variable de la fila `t` usa demanda posterior a "
        "`t − rezago mínimo`. Las marcas de atípicos quedan fuera de la tabla porque se calculan con la "
        "demanda de la propia hora, y nada se escala ni imputa con estadísticos de la tabla completa.",
        icon=":material/lock_clock:",
    )
    temperatura = carga.obtener(variables, "versiones", nombres[0], "politica", "temperatura")
    if temperatura:
        st.caption(f"Sobre la temperatura: {temperatura}")
else:
    pendiente.aviso(VARIABLES)
