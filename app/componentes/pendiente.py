"""Aviso estandar cuando falta un artefacto: que archivo falta y que lo genera.

Es el equivalente del `[pending]` del PDF: la pagina se dibuja igual, con un
recuadro que dice exactamente que comando produce lo que falta.

    if pendiente.requerir("docs/resultados/eda.json"):
        ...  # dibujar
"""

from __future__ import annotations

import streamlit as st

from componentes import carga

# Prefijo de la ruta -> comando que lo genera. Gana el prefijo mas largo.
GENERADORES = {
    "app/datos/": "python scripts/exportar_interfaz.py",
    "data/procesado/registro_limpieza.json": "python scripts/construir_panel.py --imputacion causal",
    "data/procesado/registro_variables.json": "python scripts/construir_variables.py",
    "docs/resultados/eda.json": "python scripts/eda.py",
    "docs/figuras/eda/": "python scripts/eda.py",
    "docs/resultados/mineria": "python scripts/mineria_no_supervisada.py",
    "docs/figuras/mineria/": "python scripts/mineria_no_supervisada.py",
    "docs/resultados/modelado_baselines.json": "python scripts/entrenar.py --tabla r24 --familia baseline (y r120)",
    "docs/resultados/modelado_regresion.json": "python scripts/entrenar.py --tabla r24 --familia lineal (y r120)",
    "docs/resultados/modelado_clasificacion.json": "python scripts/entrenar.py --tabla r24 --familia clasificacion (y r120)",
    "docs/resultados/modelado_ensembles.json": "python scripts/entrenar.py --tabla r24 --familia ensembles (y r120)",
    "docs/resultados/modelado_busqueda.json": "python scripts/entrenar.py --tabla r24 --familia busqueda (y r120)",
    "docs/resultados/modelado_interpretabilidad.json": "python scripts/entrenar.py --tabla r24 --familia interpretabilidad (y r120)",
    "docs/resultados/modelado_seleccion.json": "python scripts/evaluar_final.py",
    "docs/modelado/": "python scripts/evaluar_final.py",
    # La evaluacion economica la produce otro frente del proyecto; todavia no tiene script.
    "docs/resultados/evaluacion_economica.json": "el frente de evaluación económica (todavía sin script)",
    "docs/figuras/modelado/": "python scripts/entrenar.py / python scripts/evaluar_final.py",
    "docs/figuras/cloud/arquitectura.png": "python infra/diagrama/arquitectura.py",
    "infra/costos/": "python infra/costos/estimar_costos.py",
}
POR_DEFECTO = "python scripts/generar_entrega.py"


def generador(relativa: str) -> str:
    coincidencias = [p for p in GENERADORES if relativa.startswith(p)]
    return GENERADORES[max(coincidencias, key=len)] if coincidencias else POR_DEFECTO


def aviso(relativa: str, script: str | None = None) -> None:
    st.warning(
        f"**Pendiente.** Falta `{relativa}`.\n\nSe genera con `{script or generador(relativa)}`.",
        icon=":material/hourglass_empty:",
    )


def requerir(*relativas: str) -> bool:
    """True si existen todos; si no, un aviso por cada uno que falta."""
    faltan = [r for r in relativas if not carga.existe(r)]
    for relativa in faltan:
        aviso(relativa)
    return not faltan

