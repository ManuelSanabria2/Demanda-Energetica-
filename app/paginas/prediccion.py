"""6. Prediccion frente a real: una semana del ano de prueba y el error por hora y dia."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes import carga, modelado, pendiente
from componentes.estilo import CATEGORICA, DIAS_SEMANA, SECUENCIAL, TINTA, TINTA_SECUNDARIA, titulo
from componentes.formato import fecha, num, pct

PREDICCIONES = "app/datos/predicciones_prueba.parquet"
MANIFIESTO = "app/datos/manifiesto_interfaz.json"
SELECCION = "docs/resultados/modelado_seleccion.json"
NOMBRE_ROL = {"baseline": "Baseline", "lineal": "Lineal", "ensemble": "Ensemble"}
COLOR_ROL = {"baseline": TINTA_SECUNDARIA, "lineal": CATEGORICA[0], "ensemble": CATEGORICA[1]}

st.title("Predicción frente a real")
st.caption("El año de prueba se usó una sola vez, al final, con los modelos ya elegidos en validación.")

predicciones = carga.parquet(PREDICCIONES)
if predicciones is None:
    pendiente.aviso(PREDICCIONES)
    st.stop()
seleccion = carga.json_(SELECCION)
registro = carga.obtener(carga.json_(MANIFIESTO), "archivos", "predicciones_prueba.parquet", "registro") or {}

version = st.radio("Versión de la tabla", sorted(predicciones["tabla"].unique(), key=lambda v: int(v[1:])),
                   horizontal=True, key="prediccion_version",
                   help="r24: demanda conocida hasta t − 24 h. r120: hasta t − 120 h (disponibilidad real).")
datos = predicciones[predicciones["tabla"] == version].copy()
datos["fecha_hora"] = pd.to_datetime(datos["fecha_hora"]).dt.tz_localize(None)
roles = [r for r in NOMBRE_ROL if f"{r}_mwh" in datos]
final = (carga.obtener(registro, "final", version) or carga.obtener(seleccion, "tablas", version, "final", "rol")
         or roles[-1])
modelos = carga.obtener(registro, "modelos", version) or {}


def etiqueta(rol: str) -> str:
    nombre = f"{NOMBRE_ROL[rol]}{' (final)' if rol == final else ''}"
    return f"{nombre}: {modelos[rol]}" if rol in modelos else nombre


# --- Semana -----------------------------------------------------------------
st.subheader("Una semana del año de prueba")
lunes = list(pd.date_range(datos["fecha_hora"].min().normalize(), datos["fecha_hora"].max(), freq="W-MON"))
clave_semana = f"prediccion_semana_{version}"
if clave_semana not in st.session_state:
    inicio = carga.obtener(seleccion, "archivos", "semana_figura_desde")
    objetivo = pd.Timestamp(inicio) if inicio else lunes[len(lunes) // 2]
    st.session_state[clave_semana] = min(lunes, key=lambda d: abs(d - objetivo))

peores = modelado.peores_semanas(datos["fecha_hora"], datos["real_mwh"], datos["final_mwh"])
if not peores.empty:
    st.markdown("**Atajo: las 3 semanas con mayor error del modelo final**")
    botones = st.columns(len(peores))
    for boton, (_, semana) in zip(botones, peores.iterrows()):
        boton.button(f"{semana['lunes']:%Y-%m-%d} · MAPE {pct(semana['mape_pct'], 2)}", width="stretch",
                     key=f"peor_{version}_{semana['lunes']:%Y%m%d}",
                     on_click=lambda d=semana["lunes"]: st.session_state.__setitem__(clave_semana, d))

desde = st.select_slider("Semana (lunes)", options=lunes, key=clave_semana, format_func=lambda d: d.strftime("%Y-%m-%d"))
otros = st.toggle("Mostrar también el candidato que no se eligió", value=False)

semana = datos[(datos["fecha_hora"] >= desde) & (datos["fecha_hora"] < desde + pd.Timedelta(days=7))]
figura = go.Figure()
figura.add_trace(go.Scatter(x=semana["fecha_hora"], y=semana["real_mwh"], name="Real", line=dict(color=TINTA, width=2.5),
                            hovertemplate="%{x|%a %d %H:00} · real %{y:.0f} MWh<extra></extra>"))
visibles = [r for r in roles if r in ("baseline", final) or otros]
for rol in visibles:
    figura.add_trace(go.Scatter(
        x=semana["fecha_hora"], y=semana[f"{rol}_mwh"], name=etiqueta(rol),
        line=dict(color=COLOR_ROL[rol], width=3 if rol == final else 1.5, dash="dash" if rol == "baseline" else None),
        hovertemplate=f"%{{x|%a %d %H:00}} · {NOMBRE_ROL[rol]} %{{y:.0f}} MWh<extra></extra>"))

dias = semana.assign(dia=semana["fecha_hora"].dt.normalize()).groupby("dia").agg(
    festivo=("es_festivo", "max"), santa=("es_semana_santa", "max"), fin_ano=("es_ultima_semana_diciembre", "max"),
    nombre=("nombre_festivo", "first"))
especiales = []
for dia, fila in dias.iterrows():
    if not (fila["festivo"] or fila["santa"]):
        continue
    texto = fila["nombre"] if isinstance(fila["nombre"], str) else ("Semana Santa" if fila["santa"] else "Festivo")
    especiales.append(f"{dia:%a %d}: {texto}")
    figura.add_vrect(x0=dia, x1=dia + pd.Timedelta(days=1), line_width=0, opacity=0.14,
                     fillcolor=CATEGORICA[3] if fila["festivo"] else CATEGORICA[6],
                     annotation_text=texto[:26], annotation_position="top left", annotation_font_size=11)
figura.update_layout(height=440, yaxis_title="MWh por hora", xaxis=dict(dtick=86400000, tickformat="%a %d"))
st.plotly_chart(figura)
st.caption("Sombreado amarillo: festivo; violeta: Semana Santa." + (" Esta semana: " + "; ".join(especiales) + "." if especiales else ""))

columnas = st.columns(len(visibles))
for columna, rol in zip(columnas, visibles):
    columna.metric(f"MAPE de la semana · {NOMBRE_ROL[rol]}", pct(modelado.mape(semana["real_mwh"], semana[f"{rol}_mwh"]), 2),
                   border=True, help=modelos.get(rol))

# --- Error por hora y dia ---------------------------------------------------
st.subheader("Dónde se equivoca: error por hora y día de la semana")
opciones = st.columns(2)
rol_mapa = opciones[0].radio("Modelo", roles, index=roles.index(final) if final in roles else 0, horizontal=True,
                             key="prediccion_rol_mapa", format_func=lambda r: NOMBRE_ROL[r] + (" (final)" if r == final else ""))
medida = opciones[1].radio("Medida", ["mae", "mape"], horizontal=True, key="prediccion_medida",
                           format_func=lambda m: "Error absoluto medio (MWh)" if m == "mae" else "MAPE (%)")
matriz = modelado.error_hora_dia(datos["fecha_hora"], datos["real_mwh"], datos[f"{rol_mapa}_mwh"], medida)
unidad = "MWh" if medida == "mae" else "%"
figura = go.Figure(go.Heatmap(
    z=matriz.to_numpy(), x=list(matriz.columns), y=[DIAS_SEMANA[i] for i in matriz.index], colorscale=SECUENCIAL,
    colorbar=dict(title=unidad), hovertemplate=f"%{{y}} %{{x}}:00 · %{{z:.{0 if medida == 'mae' else 2}f}} {unidad}<extra></extra>",
))
horas_pico = carga.obtener(seleccion, "horas_pico") or []
for hora in horas_pico:
    figura.add_vline(x=hora, line_color=CATEGORICA[7], line_width=1, opacity=0.4)
figura.update_layout(height=340, xaxis=dict(title="hora", dtick=1), yaxis=dict(autorange="reversed"))
st.plotly_chart(figura)
peor = matriz.stack().idxmax()
st.caption(f"Promedio sobre todo el año de prueba. La celda con más error es {DIAS_SEMANA[peor[0]].lower()} a las "
           f"{peor[1]}:00 ({num(matriz.stack().max(), 0 if medida == 'mae' else 2)} {unidad})."
           + (f" Líneas rojas: horas pico ({', '.join(map(str, horas_pico))} h)." if horas_pico else ""))

# --- Error diario a lo largo del ano ----------------------------------------
st.subheader("Error diario a lo largo del año de prueba")
datos["dia"] = datos["fecha_hora"].dt.normalize()
diario = pd.DataFrame({
    rol: (100 * (datos[f"{rol}_mwh"] - datos["real_mwh"]).abs() / datos["real_mwh"]).groupby(datos["dia"]).mean()
    for rol in roles
})
figura = go.Figure()
for rol in roles:
    figura.add_trace(go.Scatter(x=diario.index, y=diario[rol], name=NOMBRE_ROL[rol],
                                line=dict(color=COLOR_ROL[rol], width=1.2, dash="dash" if rol == "baseline" else None),
                                visible=True if rol in (final, "baseline") else "legendonly",
                                hovertemplate=f"%{{x|%Y-%m-%d %a}} · {NOMBRE_ROL[rol]} %{{y:.2f}} %<extra></extra>"))
figura.update_layout(height=340, yaxis_title="MAPE del día (%)")
titulo(figura, "MAPE de cada día: baseline frente al modelo final")
st.plotly_chart(figura)

with st.expander("Los diez peores días de cada modelo"):
    calendario = datos.groupby("dia").agg(festivo=("es_festivo", "max"), santa=("es_semana_santa", "max"),
                                          nombre=("nombre_festivo", "first"))
    calendario["tipo"] = calendario.apply(lambda f: "Semana Santa" if f["santa"] else ("festivo" if f["festivo"] else ""), axis=1)
    columnas = st.columns(2)
    for columna, rol in zip(columnas, ["baseline", final]):
        peores_dias = diario[rol].nlargest(10).rename("mape").to_frame().join(calendario[["tipo", "nombre"]])
        with columna:
            st.markdown(f"**{etiqueta(rol)}**")
            st.dataframe(
                peores_dias.reset_index().assign(dia=lambda d: d["dia"].dt.strftime("%Y-%m-%d %a")),
                hide_index=True, placeholder="",
                column_config={"dia": "día", "mape": st.column_config.NumberColumn("MAPE %", format="%.2f"),
                               "tipo": "calendario", "nombre": "festivo"},
            )
    st.caption("El baseline repite la semana anterior, así que falla en los festivos y en los días que siguen a uno.")
bloque = carga.obtener(seleccion, "bloque_prueba") or []
if len(bloque) == 2:
    st.caption(f"Año de prueba: {fecha(bloque[0])} a {fecha(bloque[1])}. Haz clic en la leyenda para mostrar u ocultar modelos.")
