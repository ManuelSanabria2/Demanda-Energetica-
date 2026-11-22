"""11. Probar el modelo: prediccion, real y SHAP de una hora de prueba, y sensibilidad a temperatura y calendario."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes import carga, modelo, pendiente
from componentes.estilo import CATEGORICA, DIAS_SEMANA, TINTA, TINTA_SECUNDARIA, titulo
from componentes.formato import num, pct

VARIABLES = "app/datos/variables_prueba.parquet"
PREDICCIONES = "app/datos/predicciones_prueba.parquet"
SELECCION = "docs/resultados/modelado_seleccion.json"
MAXIMO_BARRAS = 9
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]

st.title("Probar el modelo")
st.caption("Elige una hora del año de prueba: el modelo final la predice aquí mismo, y SHAP explica de dónde sale el número.")

version = st.radio("Versión de la tabla", ["r24", "r120"], horizontal=True, key="probar_version",
                   help="r24: demanda conocida hasta t − 24 h. r120: hasta t − 120 h (disponibilidad real).")
descripcion = modelo.descripcion(version)
datos = carga.parquet(VARIABLES)
if descripcion is None or datos is None:
    if descripcion is None:
        pendiente.aviso(modelo.relativa(version))
    if datos is None:
        pendiente.aviso(VARIABLES)
    st.stop()

registro = descripcion.get("registro", {})
estimador = modelo.cargar(version)
st.markdown(f"**Modelo final {version}:** {registro.get('modelo')} · {num(descripcion['bytes'] / 1e6, 1)} MB "
            f"({registro.get('guardado')}).")
if registro.get("variante"):
    st.info(f"**Esta página usa una variante.** {registro.get('motivo_variante')}, así que aquí se usa "
            f"{registro.get('modelo')}, ya entrenada (MAPE de validación {pct(registro.get('mape_validacion_pct'), 2)}). "
            "Las cifras de las demás páginas son las del modelo final.", icon=":material/swap_horiz:")

datos = datos[datos["tabla"] == version].copy()
datos["local"] = pd.to_datetime(datos["fecha_hora"]).dt.tz_localize(None)

# --- Elegir la hora ---------------------------------------------------------
if "probar_dia" not in st.session_state:
    # Por defecto, la hora de mayor demanda del año de prueba: el caso que más importa al operador.
    pico = datos.loc[datos["real_mwh"].idxmax(), "local"]
    st.session_state["probar_dia"], st.session_state["probar_hora"] = pico.date(), pico.hour
izquierda, derecha = st.columns([2, 3])
dia = izquierda.date_input("Día del año de prueba", min_value=datos["local"].min().date(),
                           max_value=datos["local"].max().date(), key="probar_dia",
                           help="Al abrir la página: la hora de mayor demanda del año de prueba.")
hora = derecha.select_slider("Hora", options=list(range(24)), format_func=lambda h: f"{h:02d}:00", key="probar_hora")
momento = pd.Timestamp(dia) + pd.Timedelta(hours=hora)
fila = datos[datos["local"] == momento]
if fila.empty:
    st.warning(f"No hay datos para {momento:%Y-%m-%d %H:00} en {version}.")
    st.stop()
X = modelo.variables(estimador, fila)
prediccion = float(estimador.predict(X)[0])
real = float(fila["real_mwh"].iloc[0])

c1, c2, c3 = st.columns(3)
c1.metric("Predicción del modelo", f"{num(prediccion)} MWh", border=True)
c2.metric("Demanda real", f"{num(real)} MWh", border=True)
c3.metric("Error", f"{num(prediccion - real)} MWh", border=True, delta=pct(100 * (prediccion - real) / real, 2),
          delta_color="off", delta_arrow="off", help="Predicción − real: negativo, el modelo se quedó corto.")

maximo = carga.obtener(carga.json_(SELECCION), "tablas", version, "extrapolacion_prueba", registro.get("rol") or "ensemble",
                       "maximo_entrenamiento_mwh")
if maximo is not None and real > maximo:
    st.info(f"La demanda real de esta hora ({num(real)} MWh) supera la mayor que el modelo vio al entrenar "
            f"({num(maximo)} MWh). Un modelo de árboles no predice por encima de lo que vio: por eso aquí se queda corto.",
            icon=":material/vertical_align_top:")

registradas = carga.parquet(PREDICCIONES)
if registradas is not None and not registro.get("variante"):
    registrada = registradas[(registradas["tabla"] == version)
                             & (pd.to_datetime(registradas["fecha_hora"]).dt.tz_localize(None) == momento)]
    if not registrada.empty:
        coincide = np.isclose(registrada["final_mwh"].iloc[0], prediccion, atol=1e-6)
        st.caption(("Coincide con" if coincide else "**No coincide** con")
                   + f" la predicción registrada en la evaluación final ({num(registrada['final_mwh'].iloc[0], 1)} MWh).")

calendario = [n for n, c in (("festivo", "es_festivo"), ("víspera de festivo", "es_vispera_festivo"), ("puente", "es_puente"),
                             ("Semana Santa", "es_semana_santa"), ("última semana de diciembre", "es_ultima_semana_diciembre"))
              if c in fila and fila[c].iloc[0] == 1]
st.caption(f"{DIAS_SEMANA[momento.dayofweek]} {momento.day} de {MESES[momento.month - 1]} de {momento.year}, "
           f"{momento.hour:02d}:00 · temperatura {num(fila['temperatura_c'].iloc[0], 1)} °C"
           + (f" · {', '.join(calendario)}" if calendario else ""))

# --- SHAP -------------------------------------------------------------------
st.subheader("Por qué predice eso: SHAP de esta hora")
base, contribuciones = modelo.explicar(version, X)
orden = contribuciones.abs().sort_values(ascending=False).index
principales = list(orden[:MAXIMO_BARRAS])
resto = contribuciones.drop(principales).sum()


def etiqueta(variable: str) -> str:
    valor = X[variable].iloc[0]
    return f"{variable} = {valor:.0f}" if abs(valor) >= 100 else f"{variable} = {valor:.2f}"


nombres = ["valor base"] + [etiqueta(v) for v in principales]
valores = [base] + [float(contribuciones[v]) for v in principales]
medidas = ["absolute"] + ["relative"] * len(principales)
if len(orden) > MAXIMO_BARRAS:
    nombres.append(f"otras {len(orden) - MAXIMO_BARRAS} variables")
    valores.append(float(resto))
    medidas.append("relative")
nombres.append("predicción")
valores.append(0.0)
medidas.append("total")
figura = go.Figure(go.Waterfall(
    orientation="h", y=nombres, x=valores, measure=medidas,
    increasing=dict(marker=dict(color=CATEGORICA[1])), decreasing=dict(marker=dict(color=CATEGORICA[0])),
    totals=dict(marker=dict(color=TINTA_SECUNDARIA)), connector=dict(line=dict(color="#bbb", width=1)),
    text=[f"{num(base)}"] + [f"{v:+,.0f}".replace(",", " ") for v in valores[1:-1]] + [f"{num(prediccion)}"],
    textposition="outside", hovertemplate="%{y}: %{x:,.0f} MWh<extra></extra>",
))
inicio = min(np.cumsum(valores[:-1]).min(), base) - 300
figura.update_layout(height=60 + 34 * len(nombres), yaxis=dict(autorange="reversed"), xaxis_title="MWh",
                     xaxis=dict(range=[inicio, max(np.cumsum(valores[:-1]).max(), prediccion) + 400]), showlegend=False)
titulo(figura, "Del valor base a la predicción: lo que suma (naranja) y lo que resta (azul) cada variable")
st.plotly_chart(figura)
st.caption(f"El valor base ({num(base)} MWh) es la predicción media del modelo; cada barra es la contribución SHAP "
           f"exacta de una variable en esta hora, y su suma da la predicción (diferencia "
           f"{abs(base + contribuciones.sum() - prediccion):.1e} MWh).")

# --- Que pasaria si ---------------------------------------------------------
st.subheader("¿Qué pasaría si…?")
st.warning(
    "**Esto muestra la sensibilidad del modelo, no una relación causal.** Se cambia una variable y se deja todo lo "
    "demás igual, aunque en la realidad no lo esté: un festivo de verdad también cambia los rezagos y el resto de la "
    "semana. Y el modelo solo sabe lo que vio en los datos; fuera de ese rango, extrapola mal.",
    icon=":material/science:",
)
temperaturas = datos["temperatura_c"]
clave = f"{version}_{momento:%Y%m%d%H}"
actual_t = float(fila["temperatura_c"].iloc[0])
columnas = st.columns([2, 1, 1])
temperatura = columnas[0].slider("Temperatura (°C)", float(np.floor(temperaturas.min())), float(np.ceil(temperaturas.max())),
                                 round(actual_t, 1), 0.1, key=f"probar_temp_{clave}",
                                 help="Rango de temperaturas del año de prueba.")
festivo = columnas[1].checkbox("Festivo", value=bool(fila["es_festivo"].iloc[0]), key=f"probar_festivo_{clave}")
vispera = columnas[2].checkbox("Víspera de festivo", value=bool(fila["es_vispera_festivo"].iloc[0]), key=f"probar_vispera_{clave}")

escenario = X.copy()
escenario["temperatura_c"] = temperatura
escenario["es_festivo"] = float(festivo)
escenario["es_vispera_festivo"] = float(vispera)
nueva = float(estimador.predict(escenario)[0])
c1, c2 = st.columns(2)
c1.metric("Predicción con estos cambios", f"{num(nueva)} MWh", border=True,
          delta=f"{nueva - prediccion:+,.0f} MWh frente a la original".replace(",", " "), delta_color="off",
          delta_arrow="off")
c2.metric("Predicción original", f"{num(prediccion)} MWh", border=True)

rejilla = np.round(np.linspace(temperaturas.min(), temperaturas.max(), 60), 2)
barrido = pd.concat([escenario] * len(rejilla), ignore_index=True)
barrido["temperatura_c"] = rejilla
curva = estimador.predict(barrido)
figura = go.Figure()
figura.add_trace(go.Scatter(x=rejilla, y=curva, mode="lines", line=dict(color=CATEGORICA[1], width=2.5, shape="hv"),
                            name="predicción", hovertemplate="%{x:.1f} °C · %{y:,.0f} MWh<extra></extra>"))
figura.add_trace(go.Scatter(x=[temperatura], y=[nueva], mode="markers", name="escenario elegido",
                            marker=dict(color=TINTA, size=11)))
figura.add_vline(x=actual_t, line_dash="dot", line_color=TINTA_SECUNDARIA, annotation_text="temperatura real",
                 annotation_position="top")
figura.update_layout(height=340, xaxis_title="temperatura (°C)", yaxis_title="MWh", showlegend=False)
titulo(figura, "Predicción de esta hora según la temperatura, con el calendario elegido")
st.plotly_chart(figura)
st.caption("La curva es escalonada porque el modelo es de árboles: solo cambia cuando la temperatura cruza uno de los "
           "cortes que aprendió.")
