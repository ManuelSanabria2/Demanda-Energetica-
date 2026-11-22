"""9. Seleccion y costo: los tres ejes, la eleccion, r24 frente a r120 y la evaluacion economica."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes import carga, pendiente
from componentes.estilo import CATEGORICA, TINTA_SECUNDARIA, titulo
from componentes.formato import num, pct

SELECCION = "docs/resultados/modelado_seleccion.json"
COSTOS = "infra/costos/costos.json"
PREDICCIONES = "app/datos/predicciones_prueba.parquet"
NOMBRE_ROL = {"baseline": "Baseline", "lineal": "Lineal", "ensemble": "Ensemble", "clasificador": "Clasificador", "final": "Final"}
NOMBRE_TIPO = {"laboral": "Laboral", "fin_de_semana": "Fin de semana", "festivo": "Festivo", "semana_santa": "Semana Santa"}

st.title("Selección y costo")
st.caption("Cómo se eligió el modelo final, qué costó el rezago de publicación y qué significa en dinero.")

seleccion = carga.json_(SELECCION)
if seleccion is None:
    pendiente.aviso(SELECCION)
    st.stop()

version = st.radio("Versión de la tabla", list(seleccion.get("tablas", {})), horizontal=True, key="seleccion_version")
tabla = seleccion["tablas"][version]

# --- Regla y tres ejes ------------------------------------------------------
st.subheader("La regla, fijada antes de mirar la prueba")
st.markdown(f"> {seleccion.get('regla_eleccion')}")
eleccion = tabla.get("eleccion", {})
st.caption(f"Bloque de la elección: {eleccion.get('bloque')}. Desempate: {eleccion.get('desempate_tiempo')}.")

filas = []
for fila in tabla.get("tabla_seleccion", []):
    validacion = carga.obtener(fila, "rendimiento", "validacion") or {}
    prueba = carga.obtener(fila, "rendimiento", "prueba") or {}
    eficiencia = fila.get("eficiencia") or {}
    filas.append({
        "rol": NOMBRE_ROL.get(fila.get("rol"), fila.get("rol")),
        "modelo": fila.get("modelo"),
        "final": "✔" if fila.get("es_final") else "",
        "MAPE validación %": validacion.get("mape_pct"),
        "MAPE prueba %": prueba.get("mape_pct"),
        "MAPE pico prueba %": prueba.get("mape_pico_pct"),
        "PR-AUC validación": validacion.get("pr_auc"),
        "PR-AUC prueba": prueba.get("pr_auc"),
        "interpretabilidad": carga.obtener(fila, "interpretabilidad", "calificacion"),
        "ms / 1000 filas": eficiencia.get("prediccion_ms_por_1000_filas"),
        "ajuste (s)": eficiencia.get("ajuste_s"),
        "tamaño (KB)": (eficiencia.get("tamano_bytes") or 0) / 1024,
    })
ejes = pd.DataFrame(filas)
if not ejes.empty:
    numericas = [c for c in ejes.columns if c not in ("rol", "modelo", "final", "interpretabilidad")]
    ejes[numericas] = ejes[numericas].apply(pd.to_numeric, errors="coerce")
    ejes = ejes.dropna(axis=1, how="all")
    st.dataframe(ejes, hide_index=True, placeholder="—", column_config={
        c: st.column_config.NumberColumn(format="%.2f") for c in numericas if c in ejes
    })
    st.caption("Tres ejes: **rendimiento** (MAPE), **interpretabilidad** (calificación con el criterio de la página "
               "Interpretabilidad) y **eficiencia** (tiempo de predicción, de ajuste y tamaño del modelo guardado).")

evaluados = eleccion.get("evaluados") or {}
if evaluados:
    st.subheader(f"Elegido en validación: {NOMBRE_ROL.get(eleccion.get('elegido'), eleccion.get('elegido'))}")
    columnas = st.columns(len(evaluados))
    for columna, (rol, datos) in zip(columnas, evaluados.items()):
        criterio = datos.get("criterio_validacion") or {}
        with columna, st.container(border=True):
            marca = ":material/check_circle:" if rol == eleccion.get("elegido") else ":material/radio_button_unchecked:"
            st.markdown(f"#### {marca} {NOMBRE_ROL.get(rol, rol)}")
            st.markdown(
                f"- MAPE de validación: **{pct(datos.get('mape_validacion_pct'), 2)}**\n"
                f"- mejora frente al baseline: {pct(criterio.get('mejora_mape_pct'), 1)} "
                f"{'✔' if criterio.get('mejora_suficiente') else '✘'}\n"
                f"- MAPE en pico: {pct(criterio.get('mape_pico_pct'), 2)} frente a "
                f"{pct(criterio.get('mape_pico_baseline_pct'), 2)} del baseline {'✔' if criterio.get('ventaja_en_pico') else '✘'}\n"
                f"- interpretabilidad {datos.get('interpretabilidad')} · "
                f"{num(datos.get('prediccion_ms_por_1000_filas'), 2)} ms por 1000 filas"
            )
    if eleccion.get("nota"):
        st.caption(eleccion["nota"])

# --- Prueba -----------------------------------------------------------------
st.subheader("Resultado en el año de prueba")
metricas = tabla.get("metricas_prueba") or {}
if metricas:
    roles = list(metricas)
    figura = go.Figure()
    for i, (medida, nombre) in enumerate([("mape_pct", "MAPE"), ("mape_pico_pct", "MAPE en horas pico")]):
        figura.add_trace(go.Bar(x=[NOMBRE_ROL.get(r, r) for r in roles], y=[metricas[r].get(medida) for r in roles],
                                name=nombre, marker_color=CATEGORICA[i],
                                text=[pct(metricas[r].get(medida), 2) for r in roles], textposition="outside"))
    figura.update_layout(barmode="group", height=360, yaxis_title="%")
    st.plotly_chart(figura)
    exito = carga.obtener(tabla, "final", "criterio_exito_prueba") or {}
    if exito:
        texto = (f"El modelo final mejora el MAPE del baseline un **{pct(exito.get('mejora_mape_pct'), 1)}** y su MAPE en "
                 f"horas pico es {pct(exito.get('mape_pico_pct'), 2)} frente a {pct(exito.get('mape_pico_baseline_pct'), 2)}.")
        (st.success if exito.get("cumple") else st.warning)(
            ("**Cumple el criterio de éxito en la prueba.** " if exito.get("cumple") else "**No cumple el criterio en la prueba.** ") + texto)

residuales = carga.obtener(tabla, "residuales_final", "tipo_dia") or {}
if residuales:
    tipos = list(residuales)
    figura = go.Figure([
        go.Bar(x=[NOMBRE_TIPO.get(t, t) for t in tipos], y=[residuales[t].get("mape_baseline_pct") for t in tipos],
               name="baseline", marker_color=TINTA_SECUNDARIA),
        go.Bar(x=[NOMBRE_TIPO.get(t, t) for t in tipos], y=[residuales[t].get("mape_pct") for t in tipos],
               name="modelo final", marker_color=CATEGORICA[1]),
    ])
    figura.update_layout(barmode="group", height=340, yaxis_title="MAPE en prueba (%)")
    titulo(figura, "MAPE por tipo de día: baseline frente al modelo final")
    st.plotly_chart(figura)
    ganancia = {t: residuales[t]["mape_baseline_pct"] - residuales[t]["mape_pct"] for t in tipos
                if residuales[t].get("mape_baseline_pct") is not None and residuales[t].get("mape_pct") is not None}
    if ganancia:
        mejor = max(ganancia, key=ganancia.get)
        texto = (f"La mayor ventaja está en **{NOMBRE_TIPO.get(mejor, mejor).lower()}** "
                 f"({num(ganancia[mejor], 2)} puntos menos que el baseline), donde repetir la semana anterior falla.")
        peores = [NOMBRE_TIPO.get(t, t).lower() for t, g in ganancia.items() if g < 0]
        if peores:
            texto += f" En **{', '.join(peores)}** el modelo final queda por encima del baseline."
        st.caption(texto)

extrapolacion = carga.obtener(tabla, "extrapolacion_prueba", carga.obtener(tabla, "final", "rol") or "ensemble") or {}
if extrapolacion:
    with st.expander("El techo de los árboles"):
        st.markdown(
            f"En la prueba hay **{num(extrapolacion.get('reales_sobre_maximo'))} horas** con demanda por encima del máximo "
            f"del entrenamiento ({num(extrapolacion.get('maximo_entrenamiento_mwh'))} MWh); el modelo final predijo por "
            f"encima de ese máximo en {num(extrapolacion.get('predicciones_sobre_maximo'))}. En esas horas su sesgo medio es "
            f"{num(extrapolacion.get('sesgo_medio_reales_sobre_maximo_mwh'))} MWh, frente a "
            f"{num(extrapolacion.get('sesgo_medio_resto_mwh'))} MWh en el resto: un modelo de árboles no extrapola una "
            "demanda que sigue creciendo, y por eso conviene reentrenarlo con frecuencia."
        )

# --- r24 frente a r120 ------------------------------------------------------
st.subheader("Lo que cuesta el rezago de publicación: r24 frente a r120")
costo = seleccion.get("costo_rezago") or {}
if costo:
    roles = list(costo)
    figura = go.Figure([
        go.Bar(x=[NOMBRE_ROL.get(r, r) for r in roles], y=[costo[r].get("mape_r24_pct") for r in roles], name="r24",
               marker_color=CATEGORICA[0], text=[pct(costo[r].get("mape_r24_pct"), 2) for r in roles], textposition="outside"),
        go.Bar(x=[NOMBRE_ROL.get(r, r) for r in roles], y=[costo[r].get("mape_r120_pct") for r in roles], name="r120",
               marker_color=CATEGORICA[1], text=[pct(costo[r].get("mape_r120_pct"), 2) for r in roles], textposition="outside"),
    ])
    figura.update_layout(barmode="group", height=360, yaxis_title="MAPE en prueba (%)")
    st.plotly_chart(figura)
    final = costo.get("final") or {}
    if final:
        st.markdown(
            f"Con la disponibilidad real de los datos (r120) el modelo final pasa de {pct(final.get('mape_r24_pct'), 2)} a "
            f"{pct(final.get('mape_r120_pct'), 2)}: **{num(final.get('aumento_puntos'), 2)} puntos más** "
            f"({pct(final.get('aumento_relativo_pct'), 1)} relativo). El baseline no cambia porque solo usa `lag_168`, "
            "disponible en las dos versiones."
        )
    st.dataframe(
        pd.DataFrame(costo).T.rename_axis("rol").reset_index().assign(rol=lambda d: d["rol"].map(lambda r: NOMBRE_ROL.get(r, r))),
        hide_index=True,
        column_config={
            "modelo_r24": "modelo en r24", "modelo_r120": "modelo en r120",
            "mape_r24_pct": st.column_config.NumberColumn("MAPE r24 %", format="%.2f"),
            "mape_r120_pct": st.column_config.NumberColumn("MAPE r120 %", format="%.2f"),
            "aumento_puntos": st.column_config.NumberColumn("aumento (puntos)", format="%.2f"),
            "aumento_relativo_pct": st.column_config.NumberColumn("aumento relativo %", format="%.1f"),
        },
    )

# --- Evaluacion economica ---------------------------------------------------
st.subheader("Evaluación económica")
izquierda, derecha = st.columns(2)
with izquierda:
    st.markdown("**Costo de operar la solución en la nube**")
    costos = carga.json_(COSTOS)
    if costos:
        escenarios = costos.get("escenarios", {})
        st.dataframe(
            pd.DataFrame([{"escenario": e, "USD/mes": d.get("total_usd_mes"), "USD/año": d.get("total_usd_anio"),
                           "COP/año": d.get("total_cop_anio")} for e, d in escenarios.items()]),
            hide_index=True,
            column_config={"USD/mes": st.column_config.NumberColumn(format="%.2f"),
                           "USD/año": st.column_config.NumberColumn(format="%.2f"),
                           "COP/año": st.column_config.NumberColumn(format="%d")},
        )
        st.caption("Término de infraestructura del costo; las horas del equipo se suman con la metodología del curso. "
                   "Detalle en la página Nube.")
    else:
        pendiente.aviso(COSTOS)
with derecha:
    st.markdown("**El error no cuesta lo mismo en las dos direcciones**")
    predicciones = carga.parquet(PREDICCIONES)
    if predicciones is not None:
        datos = predicciones[(predicciones["tabla"] == version) & predicciones["es_final"]]
        horas = pd.to_datetime(datos["fecha_hora"]).dt.hour
        pico = horas.isin(seleccion.get("horas_pico") or [])
        subestima = datos["prediccion_mwh"] < datos["real_mwh"]
        deficit = (datos["real_mwh"] - datos["prediccion_mwh"]).clip(lower=0)
        c1, c2 = st.columns(2)
        c1.metric("Horas subestimadas", pct(100 * subestima.mean(), 1), border=True,
                  help="Horas del año de prueba en que el modelo final predijo menos que la demanda real.")
        c2.metric("… en horas pico", pct(100 * subestima[pico].mean(), 1), border=True)
        st.metric("Energía subestimada en el año", f"{num(deficit.sum() / 1000)} GWh", border=True,
                  help="Suma de (real − predicción) en las horas subestimadas.")
    else:
        pendiente.aviso(PREDICCIONES)
st.markdown(
    "Subestimar expone al sistema a racionamiento, cuyo costo publica la UPME (*Costos de Racionamiento de Energía*) "
    "y es muy superior al de sobrestimar, que solo deja reserva ociosa. Esa asimetría es el centro del análisis "
    "costo-beneficio: el valor del pronóstico se concentra en las horas pico y justifica evaluar una pérdida "
    "asimétrica (por cuantiles) que castigue más la subestimación."
)
