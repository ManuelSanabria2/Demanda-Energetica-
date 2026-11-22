# Entrega final: modelado supervisado

Universidad Santo Tomás, Tunja. Ingeniería de Datos e IA, 5.º semestre.
Integrantes: Manuel José Sanabria Gil, Luna Gabriela Guampe Niño, Astrid Vannessa Rincón Oros.
ODS 9, meta 9.1: infraestructura fiable y resiliente.

Pronóstico de la demanda horaria del SIN colombiano a 24 h, en dos escenarios:
**r24** (la demanda se conoce hasta `t − 24 h`) y **r120** (hasta `t − 120 h`, el
que respeta el rezago de publicación). Los entregables del avance (EDA, minería
no supervisada y nube) siguen en [`ENTREGA_AVANCE.md`](ENTREGA_AVANCE.md).

Este documento no repite cifras: cada número está en los JSON de
`docs/resultados/` y en [`modelado/seleccion.md`](modelado/seleccion.md), que se
genera a partir de ellos. El informe los toma como macros
(`docs/informe/valores_modelado.py`).

## Requisito de Machine Learning → evidencia en el repo

| # | Requisito | Código | Resultados y figuras | Pruebas |
|---|---|---|---|---|
| 1 | **Preparación y partición temporal sin fuga.** Entrenamiento, validación y prueba por fecha; validación cruzada temporal con `gap` igual al rezago; preprocesado dentro de `Pipeline`; métricas del problema (MAPE, MAPE en horas pico, error sobre el máximo diario, sesgo) y baselines ingenuos | `src/modelado/{datos,particion,metricas,baselines,tiempos}.py`, `scripts/entrenar.py --familia baseline`, reglas en `CLAUDE.md` | `docs/resultados/modelado_baselines.json` | `tests/test_modelado_base.py` (bloques en orden, ningún pliegue valida antes de entrenar, métricas a mano, la prueba alterada no cambia nada) |
| 2 | **Regresión: modelos lineales, polinomiales y regularización** (Ridge, Lasso, ElasticNet) con hiperparámetros elegidos por `GridSearchCV` temporal sobre el `Pipeline` completo | `src/modelado/regresion.py`, `--familia lineal` | `docs/resultados/modelado_regresion.json` (coeficientes, los que anula Lasso, número de condición) | `tests/test_modelado_regresion.py` (la validación no entra en el ajuste; reproducibilidad) |
| 3 | **Clasificación** de `alerta_pico` (logística, KNN, LinearSVC, SVC RBF) con métricas para clases desbalanceadas (PR-AUC, F1, matriz de confusión) y un objetivo construido sin fuga | `src/modelado/{objetivos,clasificacion}.py`, `--familia clasificacion`; definición en `docs/diccionario_variables.md` | `docs/resultados/modelado_clasificacion.json`, `docs/figuras/modelado/precision_recall_{r24,r120}.png` | `tests/test_modelado_clasificacion.py` (alterar la demanda posterior a `t − r` no cambia el umbral de `t`) |
| 4 | **Métodos de ensamble**: Random Forest, AdaBoost, Gradient Boosting e HistGradientBoosting, sobre el nivel y sobre la serie diferenciada (`y − lag_168`), con el diagnóstico de extrapolación | `src/modelado/ensembles.py`, `--familia ensembles` | `docs/resultados/modelado_ensembles.json` | `tests/test_modelado_ensembles.py` (un bosque sobre el nivel no pasa del máximo; diferenciado sí) |
| 5 | **Optimización de hiperparámetros**: `GridSearchCV` frente a `RandomizedSearchCV` sobre el mismo espacio y presupuesto, con `TimeSeriesSplit` | `src/modelado/busqueda.py`, `--familia busqueda` | `docs/resultados/modelado_busqueda.json`, `docs/figuras/modelado/busqueda_puntaje_tiempo_{r24,r120}.png` | `tests/test_modelado_busqueda.py` (mismo espacio en los dos métodos; la validación no cambia la búsqueda) |
| 6 | **Interpretabilidad**: importancia por impureza y por permutación, SHAP (`TreeExplainer`, `LinearExplainer`), LIME en casos concretos, contraste con el EDA y calificación de interpretabilidad por modelo | `src/modelado/interpretabilidad.py`, `--familia interpretabilidad` | `docs/resultados/modelado_interpretabilidad.json`, `docs/figuras/modelado/{importancias,shap_*,lime}_*.png` | `tests/test_modelado_interpretabilidad.py` (SHAP suma la predicción; LinearExplainer = coef · (x − media); casos de LIME) |
| 7 | **Evaluación final, selección y despliegue**: reentreno con entrenamiento + validación, evaluación única en prueba, tabla de selección (rendimiento, interpretabilidad, eficiencia), criterio de éxito, costo del rezago, residuales y modelo listo para predecir | `src/modelado/final.py`, `scripts/evaluar_final.py`, `scripts/predecir.py` | `docs/modelado/seleccion.md`, `docs/resultados/modelado_seleccion.json`, `predicciones_prueba.parquet`, `probabilidades_alerta.parquet`, `docs/figuras/modelado/{semana_prueba,mape_prueba,residuales_*}.png`, `models/final_{r24,r120}.joblib` (SHA-256 en el JSON) | `tests/test_modelado_final.py` (el ajuste final no ve la prueba; regla de elección; un MAPE < 1 % detiene todo antes de escribir; `predecir.py` reproduce lo evaluado) |

El recorrido completo, ejecutado, está en
[`notebooks/05_modelado_supervisado.ipynb`](../notebooks/05_modelado_supervisado.ipynb):
llama a `src/modelado` cuando el cálculo es rápido y lee los JSON cuando es
largo.

## Garantías transversales

- **Sin fuga temporal.** Nunca hay particiones aleatorias ni `shuffle`. La
  validación cruzada es solo `TimeSeriesSplit` con `gap` igual al rezago. Todo
  escalado o expansión va dentro de un `Pipeline` ajustado en cada pliegue. La
  parada temprana de HistGradientBoosting está desactivada (haría una partición
  aleatoria) y el error *out-of-bag* de Random Forest no se usa. Un MAPE < 1 %
  lanza `AlertaFuga`.
- **Prueba intacta hasta el final.** El bloque 2025-09-03 → 2026-09-02 solo lo
  toca `scripts/evaluar_final.py`. El modelo final se elige con una regla fijada
  sobre validación, la prueba se evalúa una vez y el script no la repite sin
  `--forzar`.
- **Reproducibilidad.** `random_state=42`. Cada experimento se corre en r24 y en
  r120. Los modelos de `models/` se cargan comprobando su SHA-256, y la suite
  corre sin red (`python -m pytest tests -q`).
- **Cifras trazables.** Ningún número del informe ni de `seleccion.md` se
  escribe a mano.

## Regenerar

```bash
python scripts/generar_entrega.py                  # todo, modelado incluido (~1 h)
python scripts/generar_entrega.py --sin-busqueda   # reutiliza la búsqueda guardada
python scripts/predecir.py --tabla r120            # predicciones del modelo final
```
