"""7. Alertas de pico: matriz de confusion y curva precision-recall con umbral ajustable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes import carga, modelado, pendiente
from componentes.estilo import CATEGORICA, TINTA, titulo
from componentes.formato import num, pct

PROBABILIDADES = "app/datos/probabilidades_alerta.parquet"
SELECCION = "docs/resultados/modelado_seleccion.json"
CLASIFICACION = "docs/resultados/modelado_clasificacion.json"

st.title("Alertas de pico")
st.caption("Además de pronosticar la demanda, el proyecto avisa con 24 h de antelación las horas que serán "
           "excepcionalmente altas para su semana.")

datos = carga.parquet(PROBABILIDADES)
seleccion = carga.json_(SELECCION)
clasificacion = carga.json_(CLASIFICACION)
if datos is None:
    pendiente.aviso(PROBABILIDADES)
    st.stop()

version = st.radio("Versión de la tabla", sorted(datos["tabla"].unique(), key=lambda v: int(v[1:])), horizontal=True,
                   key="alertas_version")
datos = datos[datos["tabla"] == version]
real = datos["alerta_real"].to_numpy()
probabilidad = datos["probabilidad"].to_numpy()

criterio = carga.obtener(clasificacion, "tablas", version, "objetivo", "criterio")
st.markdown(
    "**Qué es una alerta.** "
    + (f"`{criterio}`. " if criterio else "")
    + f"En el año de prueba hay {num(real.sum())} horas de alerta de {num(len(real))} "
      f"({pct(100 * real.mean(), 1)}). El clasificador es **{datos['modelo'].iloc[0]}**."
)

calibracion = carga.obtener(seleccion, "tablas", version, "calibracion_alerta")
umbral_modelo = modelado.probabilidad_de_puntaje(0.0, calibracion)
if calibracion:
    st.caption(f"Probabilidad: {calibracion.get('metodo')}, {calibracion.get('ajustada_con')}. "
               f"Nota del registro: {calibracion.get('nota')}.")
if umbral_modelo is None:
    pendiente.aviso(SELECCION)
    umbral_modelo = 0.5

umbral = st.slider("Umbral de probabilidad para emitir la alerta", 0.01, 0.99, float(round(umbral_modelo, 3)), 0.001,
                   format="%.3f",
                   help=f"El umbral del modelo (puntaje 0 de la SVM) equivale a una probabilidad de {num(umbral_modelo, 3)}.")
m = modelado.matriz_confusion(real, probabilidad, umbral)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precisión", num(m["precision"], 3), border=True, help="De las alertas emitidas, cuántas eran picos reales.")
c2.metric("Recall", num(m["recall"], 3), border=True, help="De los picos reales, cuántos se avisaron.")
c3.metric("F1", num(m["f1"], 3), border=True)
c4.metric("Alertas emitidas", num(m["vp"] + m["fp"]), border=True,
          help=f"{num(m['fp'])} falsas alarmas y {num(m['fn'])} picos sin avisar.")

izquierda, derecha = st.columns(2)
with izquierda:
    z = np.array([[m["vn"], m["fp"]], [m["fn"], m["vp"]]])
    texto = [[f"Verdaderos negativos<br>{num(m['vn'])}", f"Falsas alarmas<br>{num(m['fp'])}"],
             [f"Picos sin avisar<br>{num(m['fn'])}", f"Picos avisados<br>{num(m['vp'])}"]]
    figura = go.Figure(go.Heatmap(z=np.log1p(z), x=["sin alerta", "alerta"], y=["no pico", "pico"], text=texto,
                                  texttemplate="%{text}", textfont=dict(color=TINTA, size=14),
                                  colorscale=[[0, "#f3f7fd"], [1, "#9cc3ef"]], showscale=False, hoverinfo="skip"))
    figura.update_layout(height=380, xaxis_title="predicho", yaxis=dict(title="real", autorange="reversed"))
    titulo(figura, f"Matriz de confusión con umbral {num(umbral, 3)}")
    st.plotly_chart(figura)
with derecha:
    curva = modelado.curva_precision_recall(real, probabilidad)
    figura = go.Figure()
    figura.add_trace(go.Scatter(x=curva["recall"], y=curva["precision"], mode="lines", name="curva PR",
                                line=dict(color=CATEGORICA[0], width=2.5, shape="hv"),
                                customdata=curva["umbral"], hovertemplate="umbral %{customdata:.3f}<br>recall %{x:.3f} · precisión %{y:.3f}<extra></extra>"))
    figura.add_hline(y=real.mean(), line_dash="dot", line_color="#999", annotation_text="azar (prevalencia)",
                     annotation_position="bottom right")
    figura.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", name="umbral elegido",
                                marker=dict(color=CATEGORICA[1], size=14, line=dict(color=TINTA, width=1))))
    pr_auc = carga.obtener(seleccion, "tablas", version, "clasificador_prueba", "pr_auc")
    figura.update_layout(height=380, xaxis=dict(title="recall", range=[0, 1.02]), yaxis=dict(title="precisión", range=[0, 1.02]))
    titulo(figura, "Curva precisión-recall" + (f" (PR-AUC {num(pr_auc, 3)})" if pr_auc is not None else ""))
    st.plotly_chart(figura)

st.markdown(
    "Bajar el umbral avisa más picos a costa de más falsas alarmas. Para el operador, un pico sin avisar "
    "(riesgo de racionamiento) cuesta mucho más que una falsa alarma (reserva ociosa), así que conviene un "
    "umbral que priorice el recall."
)

referencia = carga.obtener(seleccion, "tablas", version, "clasificador_prueba")
if referencia:
    with st.expander("Comprobación con el registro de la evaluación final"):
        registrada = referencia.get("matriz_confusion", {})
        en_umbral = modelado.matriz_confusion(real, probabilidad, umbral_modelo)
        st.dataframe(
            pd.DataFrame({
                "registro (modelado_seleccion.json)": [registrada.get(k) for k in ("vn", "fp", "fn", "vp")]
                                                      + [referencia.get("precision"), referencia.get("recall"), referencia.get("f1")],
                "recalculado aquí con el umbral del modelo": [en_umbral[k] for k in ("vn", "fp", "fn", "vp", "precision", "recall", "f1")],
            }, index=["verdaderos negativos", "falsas alarmas", "picos sin avisar", "picos avisados", "precisión", "recall", "F1"]),
        )
        st.caption("Con el umbral del modelo la página reproduce la evaluación registrada; el deslizador solo explora "
                   "otros puntos de la misma curva.")

semana = st.toggle("Ver las alertas sobre la demanda de una semana", value=False)
if semana:
    tiempo = pd.to_datetime(datos["fecha_hora"]).dt.tz_localize(None)
    lunes = pd.date_range(tiempo.min().normalize(), tiempo.max(), freq="W-MON")
    desde = st.select_slider("Semana (lunes)", options=list(lunes), value=lunes[len(lunes) // 2],
                             format_func=lambda d: d.strftime("%Y-%m-%d"), key="alertas_semana")
    tramo = datos.assign(t=tiempo)[(tiempo >= desde) & (tiempo < desde + pd.Timedelta(days=7))]
    emitida = tramo["probabilidad"] >= umbral
    figura = go.Figure()
    figura.add_trace(go.Scatter(x=tramo["t"], y=tramo["real_mwh"], name="demanda real", line=dict(color=TINTA)))
    figura.add_trace(go.Scatter(x=tramo["t"], y=tramo["umbral_alerta_mwh"], name="umbral de pico",
                                line=dict(color="#999", dash="dot")))
    figura.add_trace(go.Scatter(x=tramo.loc[emitida, "t"], y=tramo.loc[emitida, "real_mwh"], mode="markers", name="alerta emitida",
                                marker=dict(color=CATEGORICA[1], size=9, symbol="triangle-up")))
    figura.add_trace(go.Scatter(x=tramo.loc[tramo["alerta_real"] == 1, "t"], y=tramo.loc[tramo["alerta_real"] == 1, "real_mwh"],
                                mode="markers", name="pico real", marker=dict(color=CATEGORICA[7], size=13, symbol="circle-open")))
    figura.update_layout(height=380, yaxis_title="MWh por hora", xaxis=dict(dtick=86400000, tickformat="%a %d"))
    st.plotly_chart(figura)
