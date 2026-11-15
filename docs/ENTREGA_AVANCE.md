# Avance del proyecto integrador: Minería de Datos y Cloud Computing

Universidad Santo Tomás, Tunja. Ingeniería de Datos e IA, 5.º semestre.
Integrantes: Manuel José Sanabria Gil, Luna Gabriela Guampe Niño, Astrid Vannessa Rincón Oros.
ODS 9, meta 9.1: infraestructura fiable y resiliente.

**Reporte PDF (formato IEEE, en inglés como recomiendan los lineamientos):**
[`docs/informe/avance_proyecto_integrador.pdf`](informe/avance_proyecto_integrador.pdf),
con su fuente LaTeX en la misma carpeta.

## Dónde está cada entregable

| Lineamiento | Evidencia |
|---|---|
| **A.** Documento técnico IEEE: introducción, problema vinculado al ODS, estructura del sistema | `docs/informe/` (secciones I a IV del PDF) |
| **B.** Dataset estructurado y limpio | `src/limpieza/`, `scripts/construir_panel.py`, `scripts/construir_variables.py`; registros en `data/procesado/registro_*.json` y `datasets/limpios/registro_*.json` |
| **B.** Manejo de nulos y atípicos | `src/calidad/`, notebooks `02_diagnostico_calidad` y `limpieza/*`, `notas/hallazgos_calidad.md` |
| **B.** EDA profundo con visualizaciones estadísticas | `scripts/eda.py`, `src/mineria/eda.py`, `notebooks/03_eda.ipynb`, figuras en `docs/figuras/eda/`, cifras en `docs/resultados/eda.json` |
| **B.** Web scraping o consumo de fuentes externas | `src/ingesta/clientes.py` (APIs de XM y SIMEM), `src/ingesta/complementarias.py` (Open-Meteo y calendario), `notas/hallazgos_apis.md` |
| **B.** Minería no supervisada (clustering) y reglas de asociación | `src/mineria/{perfiles,clustering,reglas,sectores}.py`, `scripts/mineria_no_supervisada.py`, `notebooks/04_mineria_no_supervisada.ipynb`, `docs/figuras/mineria/`, `docs/resultados/mineria*.{json,csv}` |
| **C.** Diagrama formal de la arquitectura en la nube | `docs/figuras/cloud/arquitectura.png`, generado por `infra/diagrama/arquitectura.py` |
| **C.** Scripts de IaC (Terraform) | `infra/terraform/` (10 archivos), `infra/contenedor/` (imagen, handler de Lambda, envoltorio de tareas) |
| **C.** Estimación preliminar de costos | `infra/costos/` (`estimar_costos.py`, `costos.md`, `costos.json`) |
| **D.** Integración con Evaluación Económica de Soluciones de IA | Sección VII del PDF |
| Reproducibilidad | `python -m pytest tests` (sin red), `python scripts/generar_entrega.py` |

## Qué se agregó en este avance

- **`src/mineria/`**: perfiles diarios de carga (forma y nivel por separado),
  K-means con selección de k por silueta y estabilidad (ARI entre semillas y
  submuestras), contraste con Ward, V de Cramér frente al tipo de día, días
  atípicos por distancia al centroide, reglas de asociación (Apriori) con poda
  de redundantes, y agrupamiento de los sectores CIIU por perfil semanal.
- **`scripts/eda.py`**: nueve figuras estadísticas y pruebas no paramétricas
  (ADF y KPSS, STL robusta, ACF y PACF, Kruskal-Wallis, Mann-Whitney, Spearman
  por hora y sobre anomalías diarias).
- **`infra/`**: arquitectura AWS como código (S3, ECR, Lambda, ECS Fargate,
  Step Functions, EventBridge Scheduler, Glue y Athena, CloudWatch, SNS, Budgets,
  IAM de mínimo privilegio). Checkov: 171 comprobaciones aprobadas, 0 fallidas,
  22 omitidas con su justificación escrita en el recurso.
- **Pruebas nuevas**: `tests/test_mineria.py`, `tests/test_scripts_entrega.py`
  (EDA y minería de punta a punta sobre un panel sintético construido con las
  funciones de producción) y `tests/test_infra.py`.

## Regenerar todo

```bash
pip install -r requirements.txt
python -m ingesta.cli --fuente todas --desde 2021-01-01    # con PYTHONPATH=src; minutos
python scripts/generar_entrega.py                          # panel, EDA, minería, costos, PDF
```

Ningún número del informe se escribe a mano: `docs/informe/generar_valores.py`
los toma de los JSON de resultados. Si un resultado no se ha calculado, el PDF
lo muestra en rojo como `[pending]` en vez de inventarlo.
