"""5. Modelos: todos los evaluados en validacion, error por pliegue y Grid frente a Random Search."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from componentes import carga, modelado, pendiente
from componentes.estilo import CATEGORICA, REJILLA, TINTA_SECUNDARIA, titulo
from componentes.formato import num, pct

ARCHIVOS = {
    "baseline": "docs/resultados/modelado_baselines.json",
    "lineal": "docs/resultados/modelado_regresion.json",
    "ensembles": "docs/resultados/modelado_ensembles.json",
    "busqueda": "docs/resultados/modelado_busqueda.json",
}
CLASIFICACION = "docs/resultados/modelado_clasificacion.json"
SELECCION = "docs/resultados/modelado_seleccion.json"
COLOR_FAMILIA = {"Baseline": TINTA_SECUNDARIA, "Lineal": CATEGORICA[0], "Ensemble": CATEGORICA[1], "Búsqueda": CATEGORICA[2]}

st.title("Modelos")
st.caption("Todo lo que se entrenó, medido en el bloque de validación (un año antes de la prueba). "
           "La prueba no interviene en nada de esta página.")

resultados = {familia: carga.json_(ruta) for familia, ruta in ARCHIVOS.items()}
version = st.radio("Versión de la tabla", ["r24", "r120"], horizontal=True, key="modelos_version",
                   help="r24: demanda conocida hasta t − 24 h. r120: hasta t − 120 h.")

pestanas = st.tabs(["Regresión", "Error por pliegue", "Grid frente a Random", "Clasificación de picos"])

# --- Regresion --------------------------------------------------------------
with pestanas[0]:
    for familia, ruta in ARCHIVOS.items():
        if resultados[familia] is None:
            pendiente.aviso(ruta)
    tabla = modelado.tabla_regresion(resultados, version)
    if not tabla.empty:
        referencia = carga.obtener(resultados["lineal"], "tablas", version, "baseline_referencia", "mape_validacion_pct")
        if referencia is None:
            fila = tabla[(tabla["familia"] == "Baseline") & (tabla["modelo"] == "estacional")]
            referencia = None if fila.empty else float(fila["mape_pct"].iloc[0])

        familias = st.multiselect("Familias", list(dict.fromkeys(tabla["familia"])), default=list(dict.fromkeys(tabla["familia"])))
        visibles = tabla[tabla["familia"].isin(familias)].sort_values("mape_pct")
        etiqueta = visibles["modelo"] + visibles["forma"].map(lambda f: "" if f == "-" else f" ({f})")

        figura = go.Figure(go.Bar(
            x=visibles["mape_pct"], y=etiqueta, orientation="h",
            marker_color=[COLOR_FAMILIA.get(f, CATEGORICA[3]) for f in visibles["familia"]],
            customdata=visibles[["familia", "mape_pico_pct", "mejora_mape_pct"]].fillna(0),
            hovertemplate="%{y} · %{customdata[0]}<br>MAPE %{x:.2f} %<br>MAPE en pico %{customdata[1]:.2f} %"
                          "<br>mejora frente al baseline %{customdata[2]:.1f} %<extra></extra>",
        ))
        if referencia is not None:
            figura.add_vline(x=referencia, line_dash="dash", line_color=TINTA_SECUNDARIA,
                             annotation_text=f"baseline estacional {pct(referencia, 2)}", annotation_position="top")
            meta = carga.obtener(carga.json_(SELECCION), "criterio_exito", "mejora_mape_minima_pct")
            if meta is not None:
                figura.add_vrect(x0=0, x1=referencia * (1 - meta / 100), fillcolor=CATEGORICA[2], opacity=0.08,
                                 line_width=0, annotation_text=f"zona de mejora ≥ {pct(meta, 0)}",
                                 annotation_position="top left")
        figura.update_layout(height=max(360, 26 * len(visibles) + 120), xaxis_title="MAPE en validación (%)",
                             yaxis=dict(autorange="reversed"), showlegend=False)
        titulo(figura, "MAPE de validación por modelo (color: familia)")
        st.plotly_chart(figura)

        st.dataframe(
            visibles, hide_index=True, placeholder="—",
            column_config={
                "familia": "familia", "modelo": "modelo", "forma": "forma",
                "mape_pct": st.column_config.NumberColumn("MAPE %", format="%.2f"),
                "mape_pico_pct": st.column_config.NumberColumn("MAPE pico %", format="%.2f"),
                "mejora_mape_pct": st.column_config.NumberColumn("mejora %", format="%.1f"),
                "mae_mwh": st.column_config.NumberColumn("MAE (MWh)", format="%.0f"),
                "rmse_mwh": st.column_config.NumberColumn("RMSE (MWh)", format="%.0f"),
                "sesgo_medio_mwh": st.column_config.NumberColumn("sesgo (MWh)", format="%.0f"),
                "error_maximo_diario_pct": st.column_config.NumberColumn("error máx. diario %", format="%.2f"),
                "mape_cv_media_pct": st.column_config.NumberColumn("MAPE CV %", format="%.2f",
                                                                   help="Media de los pliegues de TimeSeriesSplit"),
                "mape_cv_desviacion_pct": st.column_config.NumberColumn("± CV", format="%.2f"),
                "ajuste_s": st.column_config.NumberColumn("ajuste (s)", format="%.2f"),
                "prediccion_ms_por_1000_filas": st.column_config.NumberColumn("ms / 1000 filas", format="%.2f"),
            },
        )
        st.caption("Baselines: la misma hora de la semana anterior (estacional, `lag_168`) y del día anterior "
                   "(diario, `lag_24`; no existe en r120). Ensembles en dos formas: sobre el nivel de la demanda y "
                   "sobre `y − lag_168` (diferenciado). Sesgo negativo: el modelo subestima.")

        comparacion = carga.obtener(resultados["ensembles"], "tablas", version, "comparacion_formas")
        maximo = carga.obtener(resultados["ensembles"], "tablas", version, "maximo_entrenamiento_mwh")
        if comparacion:
            with st.expander("Nivel frente a diferenciado: el límite de los árboles"):
                st.markdown(
                    f"Un árbol no predice por encima del máximo que vio al entrenar ({num(maximo)} MWh). "
                    "Como la demanda crece, en validación hay horas por encima de ese techo; modelar "
                    "`y − lag_168` lo evita. La tabla dice si en esta serie eso compensó."
                )
                st.dataframe(
                    [{"modelo": n,
                      "MAPE nivel %": carga.obtener(c, "nivel", "mape_pct"),
                      "MAPE diferenciado %": carga.obtener(c, "diferenciado", "mape_pct"),
                      "reales sobre el máximo": c.get("reales_sobre_maximo"),
                      "sesgo en esas horas, nivel (MWh)": carga.obtener(c, "nivel", "sesgo_medio_reales_sobre_maximo_mwh"),
                      "diferenciado reduce el sesgo": c.get("diferenciado_reduce_sesgo"),
                      "diferenciado reduce el MAPE": c.get("diferenciado_reduce_mape")}
                     for n, c in comparacion.items()],
                    hide_index=True, placeholder="—",
                )

# --- Pliegues ---------------------------------------------------------------
with pestanas[1]:
    datos = modelado.pliegues(resultados, version)
    if datos.empty:
        pendiente.requerir(ARCHIVOS["lineal"], ARCHIVOS["ensembles"])
    else:
        criterio = carga.obtener(resultados["lineal"], "tablas", version, "validacion_cruzada", "criterio")
        cortes = carga.obtener(resultados["lineal"], "tablas", version, "validacion_cruzada", "detalle", "pliegues") or []
        st.markdown(
            "Cada modelo se valida con **TimeSeriesSplit**: el entrenamiento siempre es anterior a su pliegue de "
            "validación, con 24 h de separación, y el preprocesado se ajusta dentro de cada pliegue."
            + (f" Criterio: `{criterio}`." if criterio else "")
        )
        modelos = sorted(datos["modelo"].unique())
        por_defecto = [m for m in modelos if m in ("ridge", "polinomial_ridge", "random_forest (nivel)", "hist_gradient (nivel)")] or modelos[:4]
        elegidos = st.multiselect("Modelos", modelos, default=por_defecto)
        figura = go.Figure()
        for i, modelo in enumerate(elegidos):
            serie = datos[datos["modelo"] == modelo]
            figura.add_trace(go.Scatter(x=serie["pliegue"], y=serie["mape_pct"], mode="lines+markers", name=modelo,
                                        line=dict(color=CATEGORICA[i % len(CATEGORICA)])))
        figura.update_layout(height=380, xaxis=dict(title="pliegue", dtick=1), yaxis_title="MAPE (%)")
        st.plotly_chart(figura)
        if cortes:
            st.dataframe(
                [{"pliegue": c.get("pliegue"),
                  "entrena desde": str(carga.obtener(c, "entrenamiento", "desde"))[:10],
                  "entrena hasta": str(carga.obtener(c, "entrenamiento", "hasta"))[:16],
                  "valida desde": str(carga.obtener(c, "validacion", "desde"))[:16],
                  "valida hasta": str(carga.obtener(c, "validacion", "hasta"))[:10],
                  "filas de entrenamiento": carga.obtener(c, "entrenamiento", "filas"),
                  "filas de validación": carga.obtener(c, "validacion", "filas")} for c in cortes],
                hide_index=True,
            )
            st.caption("El hueco entre el fin del entrenamiento y el inicio de la validación es el `gap` de 24 h.")

# --- Busqueda ---------------------------------------------------------------
with pestanas[2]:
    if resultados["busqueda"] is None:
        pendiente.aviso(ARCHIVOS["busqueda"])
    else:
        trayectorias = modelado.trayectorias_busqueda(resultados["busqueda"], version)
        resumen = modelado.resumen_busqueda(resultados["busqueda"], version)
        tolerancia = carga.obtener(resultados["busqueda"], "tablas", version, "tolerancia_relativa")
        st.markdown(
            "Los mejores candidatos de cada paso se vuelven a ajustar con **GridSearchCV** (toda la rejilla) y con "
            "**RandomizedSearchCV** (el mismo presupuesto de ajustes, muestreando un espacio continuo), con la misma "
            "validación cruzada temporal."
            + (f" Random «alcanza» a Grid si su mejor puntaje queda a menos del {pct(100 * tolerancia, 1)}." if tolerancia else "")
        )
        if not trayectorias.empty:
            candidato = st.selectbox("Candidato", list(dict.fromkeys(trayectorias["candidato"])))
            datos = trayectorias[trayectorias["candidato"] == candidato]
            es_regresion = carga.obtener(resultados["busqueda"], "tablas", version, "modelos", candidato, "tipo") == "regresion"
            figura = go.Figure()
            for i, (metodo, serie) in enumerate(datos.groupby("metodo")):
                figura.add_trace(go.Scatter(x=serie["computo_s"], y=serie["mejor_cv"], mode="lines+markers", name=metodo,
                                            line=dict(shape="hv", color=CATEGORICA[i]),
                                            customdata=serie["ajustes_cv"],
                                            hovertemplate="%{x:.0f} s · %{customdata} ajustes · mejor %{y:.3f}<extra></extra>"))
            figura.update_layout(height=380, xaxis_title="cómputo acumulado (s)",
                                 yaxis_title="mejor MAE de CV (MWh)" if es_regresion else "mejor puntaje de CV")
            titulo(figura, "Mejor puntaje encontrado frente al cómputo gastado")
            st.plotly_chart(figura)
        if not resumen.empty:
            st.dataframe(resumen, hide_index=True,
                         column_config={"s de reloj (Grid)": st.column_config.NumberColumn(format="%.1f"),
                                        "s de reloj (Random)": st.column_config.NumberColumn(format="%.1f")})
            st.caption("El eje del gráfico suma el tiempo de cada ajuste de la validación cruzada; las búsquedas "
                       "corren en paralelo, así que en reloj tardan lo que dice la tabla.")

# --- Clasificacion ----------------------------------------------------------
with pestanas[3]:
    clasificacion = carga.json_(CLASIFICACION)
    if clasificacion is None:
        pendiente.aviso(CLASIFICACION)
    else:
        objetivo = carga.obtener(clasificacion, "tablas", version, "objetivo", "criterio")
        if objetivo:
            st.markdown(f"**Objetivo:** `{objetivo}`")
        tabla = modelado.tabla_clasificacion(clasificacion, version)
        if not tabla.empty:
            figura = go.Figure()
            for i, medida in enumerate(["pr_auc", "f1", "precision", "recall"]):
                figura.add_trace(go.Bar(x=tabla["modelo"], y=tabla[medida], name=medida.replace("_", "-").upper()
                                        if medida == "pr_auc" else medida, marker_color=CATEGORICA[i]))
            figura.update_layout(barmode="group", height=360, yaxis=dict(range=[0, 1], gridcolor=REJILLA))
            st.plotly_chart(figura)
            st.dataframe(tabla, hide_index=True,
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ("pr_auc", "roc_auc", "f1", "precision", "recall")})
            prevalencia = tabla["positivos"].iloc[0] / tabla["n"].iloc[0] if tabla["n"].iloc[0] else None
            st.caption(f"Las clases están desbalanceadas ({pct(100 * prevalencia, 1)} de las horas de validación son "
                       "alerta), así que se compara por PR-AUC y no por exactitud. El baseline alerta si `lag_168` ya "
                       "superaba el umbral.")
