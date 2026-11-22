"""10. Nube: diagrama de arquitectura en AWS y estimacion de costos."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from componentes import carga, pendiente
from componentes.estilo import CATEGORICA
from componentes.formato import num

ARQUITECTURA = "docs/figuras/cloud/arquitectura.png"
COSTOS = "infra/costos/costos.json"
NOMBRE_ESCENARIO = {
    "academico": "Académico (con capa gratuita)",
    "sin_capa_gratuita": "Académico sin capa gratuita",
    "operacion": "Operación",
}

st.title("Nube")
st.caption("Cómo operaría el pipeline en AWS, definido como código con Terraform, y cuánto costaría.")

st.subheader("Arquitectura")
imagen = carga.imagen(ARQUITECTURA)
if imagen:
    st.image(imagen, width="stretch")
else:
    pendiente.aviso(ARQUITECTURA)

st.dataframe(
    pd.DataFrame(
        [
            ["Ingesta diaria (06:00)", "Lambda (imagen de contenedor)", "ingesta.cli --fuente todas, incremental"],
            ["Panel y variables", "ECS Fargate Spot · tarea procesar", "construir_panel.py y construir_variables.py"],
            ["Pronóstico 24 h", "ECS Fargate Spot · tarea predecir", "scripts/predecir.py"],
            ["Reentrenamiento semanal", "ECS Fargate Spot · tarea entrenar", "scripts/entrenar.py"],
            ["Orquestación", "Step Functions + EventBridge Scheduler", "reintentos y aviso por SNS ante fallos"],
            ["Data lake", "S3 versionado: raw/, staging/, processed/, models/, predictions/", "la estructura de data/"],
            ["Consumo", "Glue Data Catalog + Athena", "partition projection, sin crawler"],
            ["Monitoreo", "CloudWatch + SNS + AWS Budgets", "fallos, ausencia de pronóstico, MAPE, presupuesto"],
        ],
        columns=["Componente", "Servicio", "Qué ejecuta"],
    ),
    hide_index=True,
)

st.subheader("Estimación de costos")
costos = carga.json_(COSTOS)
if costos:
    escenarios = costos.get("escenarios", {})
    columnas = st.columns(len(escenarios))
    for columna, (clave, datos) in zip(columnas, escenarios.items()):
        columna.metric(NOMBRE_ESCENARIO.get(clave, clave), f"{num(datos.get('total_usd_mes'), 2)} USD/mes", border=True,
                       delta=f"{num(datos.get('total_cop_mes'))} COP/mes", delta_color="off", delta_arrow="off",
                       help=f"Al año: {num(datos.get('total_usd_anio'), 2)} USD · {num(datos.get('total_cop_anio'))} COP")

    elegido = st.selectbox("Escenario", list(escenarios), format_func=lambda c: NOMBRE_ESCENARIO.get(c, c))
    detalle = pd.Series(escenarios[elegido].get("detalle_usd_mes", {}), name="usd_mes").sort_values()
    figura = px.bar(detalle, orientation="h", labels={"value": "USD por mes", "index": ""},
                    text=detalle.map(lambda v: f"{v:.2f}".replace(".", ",")), color_discrete_sequence=[CATEGORICA[0]])
    figura.update_layout(height=360, showlegend=False)
    st.plotly_chart(figura)

    supuestos = escenarios[elegido].get("supuestos")
    if supuestos:
        with st.expander("Supuestos del escenario"):
            st.dataframe(pd.Series(supuestos, name="valor").astype(str).rename_axis("supuesto"))
    ahorro = costos.get("decision_sin_nat_gateway_ahorro_usd_mes")
    st.markdown(
        f"Región `{costos.get('region')}`, precios consultados en {costos.get('precios_consultados')}, "
        f"TRM {num(costos.get('trm_cop_usd'), 2)} COP/USD."
        + (f" Prescindir del NAT Gateway ahorra **{num(ahorro, 2)} USD/mes** fijos: las tareas corren en subredes "
           "públicas con un grupo de seguridad sin reglas de entrada." if ahorro else "")
    )
    if costos.get("nota"):
        st.caption(costos["nota"])
else:
    pendiente.aviso(COSTOS)

evidencia = carga.imagenes("docs/figuras/cloud", "consola_*.png")
if evidencia:
    with st.expander("Evidencia del despliegue (capturas de la consola)"):
        st.image(evidencia, width="stretch")
