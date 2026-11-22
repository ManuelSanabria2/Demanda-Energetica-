# Selección del modelo final

> Documento generado por `scripts/evaluar_final.py` a partir de `docs/resultados/modelado_seleccion.json`. Ninguna cifra está escrita a mano.

Este es el único paso que usa el bloque de prueba (2025-09-03 → 2026-09-02). Los candidatos se reentrenaron con entrenamiento + validación y los hiperparámetros ya elegidos, sin volver a buscar, y se evaluaron una sola vez en prueba.

## Regla de elección

La elección se hizo **con validación, antes de mirar la prueba**: elegibles: mejora de MAPE >= 20 % y MAPE en horas pico menor que el del baseline, en validacion; gana el menor MAPE de validacion; empate a menos de 0.05 puntos: mas interpretable y despues mas rapido al predecir. La prueba solo dice si el criterio de éxito se sostiene fuera de la muestra con la que se eligió.

Criterio de éxito del proyecto: reducir al menos un 20 % el MAPE del baseline estacional (`lag_168`) y mantener la ventaja en las horas pico (18:00, 19:00, 20:00).

## r24

### Tabla de selección

| rol | modelo | MAPE val. | MAPE prueba | MAPE pico prueba | mejora prueba | sesgo prueba | interpretabilidad | ajuste | predicción / 1000 filas | tamaño |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | baseline estacional (lag_168) | 3,35 % | 3,54 % | 3,12 % | — | -15 MWh | alta | 0,0 s | 0,00 ms | 0,7 KB |
| lineal | polinomial_ridge | 2,18 % | 2,28 % | 2,40 % | +35,7 % | -126 MWh | media | 0,1 s | 0,90 ms | 3,3 KB |
| ensemble | hist_gradient (nivel) **(final)** | 1,86 % | 2,21 % | 2,87 % | +37,7 % | -115 MWh | baja | 1,0 s | 8,19 ms | 1,1 MB |

MAE, RMSE y error sobre el máximo diario de cada candidato están en el JSON (`metricas_prueba`).

### Por qué este modelo

**Modelo final en r24: hist_gradient (nivel)** (rol `ensemble`).

- `lineal` (polinomial_ridge): MAPE de validación 2,18 %, cumple el criterio en validación; interpretabilidad media.
- `ensemble` (hist_gradient (nivel)): MAPE de validación 1,86 %, cumple el criterio en validación; interpretabilidad baja.

Frente a polinomial_ridge: el final tiene 1,86 % de MAPE de validación contra 2,18 %; a cambio, su interpretabilidad es baja (la del otro, media) y predice en 8,19 ms por 1000 filas frente a 0,90 ms.

El baseline estacional se queda como referencia: no tiene parámetros y es la cota que cualquier modelo tiene que superar.

### Criterio de éxito en prueba

- Mejora de MAPE frente al baseline: **+37,7 %** (cumple el mínimo del 20 %).
- MAPE en horas pico: 2,87 % frente a 3,12 % del baseline (mantiene la ventaja).
- **Cumple el criterio de éxito en prueba.**
- Como contraste, `lineal` (polinomial_ridge) mejora un +35,7 % en prueba, con 2,40 % en horas pico (mantiene la ventaja).

### Por encima de lo visto

276 de las 8760 horas de prueba (3,2 %) superan el máximo de entrenamiento + validación (11 736 MWh). En esas horas:

- `baseline` (baseline estacional (lag_168)): 242 predicciones por encima del máximo; sesgo medio -318 MWh (en el resto de horas, -5 MWh).
- `lineal` (polinomial_ridge): 81 predicciones por encima del máximo; sesgo medio -399 MWh (en el resto de horas, -117 MWh).
- `ensemble` (hist_gradient (nivel)): 0 predicciones por encima del máximo; sesgo medio -691 MWh (en el resto de horas, -97 MWh).

Un árbol predice con valores que vio y no pasa de ese máximo; la demanda de prueba sí lo pasa. El ensemble elegido trabaja sobre el nivel: en validación solo 3 horas superaban el máximo de entrenamiento y la forma diferenciada (`y - lag_168`), la que permitiría pasar ese techo, no ganó (`docs/resultados/modelado_ensembles.json`).

### Residuales del modelo final en prueba

Figura: `docs/figuras/modelado/residuales_r24.png`. Residual = predicción − real.

- Por hora: el MAPE va de 1,83 % (22:00) a 3,19 % (19:00).
- Por tipo de día:
  - laboral (n = 5736): MAPE 2,29 % frente a 3,26 % del baseline; sesgo -145 MWh.
  - fin de semana (n = 2448): MAPE 1,90 % frente a 2,63 % del baseline; sesgo -64 MWh.
  - festivo (n = 408): MAPE 3,03 % frente a 10,37 % del baseline; sesgo -17 MWh.
  - semana santa (n = 168): MAPE 1,86 % frente a 9,74 % del baseline; sesgo -94 MWh.
- Por día de la semana, el peor es el lunes (MAPE 2,96 %, sesgo -139 MWh).

### Alerta de pico (`alerta_pico`) en prueba

svm_rbf: PR-AUC 0,800 (prevalencia 11,4 %), ROC-AUC 0,972, F1 0,725, precisión 0,609, recall 0,896. Brier de la probabilidad calibrada: 0,0463.

## r120

### Tabla de selección

| rol | modelo | MAPE val. | MAPE prueba | MAPE pico prueba | mejora prueba | sesgo prueba | interpretabilidad | ajuste | predicción / 1000 filas | tamaño |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | baseline estacional (lag_168) | 3,35 % | 3,54 % | 3,12 % | — | -15 MWh | alta | 0,0 s | 0,00 ms | 0,7 KB |
| lineal | polinomial_ridge | 2,87 % | 3,10 % | 3,64 % | +12,4 % | -208 MWh | media | 0,1 s | 0,87 ms | 3,1 KB |
| ensemble | random_forest (nivel) **(final)** | 2,66 % | 3,28 % | 3,83 % | +7,4 % | -237 MWh | baja | 3,1 s | 14,32 ms | 57,6 MB |

MAE, RMSE y error sobre el máximo diario de cada candidato están en el JSON (`metricas_prueba`).

### Por qué este modelo

**Modelo final en r120: random_forest (nivel)** (rol `ensemble`).

- `lineal` (polinomial_ridge): MAPE de validación 2,87 %, no cumple el criterio en validación (mejora de +14,3 %, por debajo del 20 %; MAPE pico de 3,11 % frente a 2,89 % del baseline); interpretabilidad media.
- `ensemble` (random_forest (nivel)): MAPE de validación 2,66 %, cumple el criterio en validación; interpretabilidad baja.

Frente a polinomial_ridge: el final tiene 2,66 % de MAPE de validación contra 2,87 %; a cambio, su interpretabilidad es baja (la del otro, media) y predice en 14,32 ms por 1000 filas frente a 0,87 ms.

El baseline estacional se queda como referencia: no tiene parámetros y es la cota que cualquier modelo tiene que superar.

### Criterio de éxito en prueba

- Mejora de MAPE frente al baseline: **+7,4 %** (no cumple el mínimo del 20 %).
- MAPE en horas pico: 3,83 % frente a 3,12 % del baseline (pierde la ventaja).
- **Hallazgo: no cumple el criterio de éxito en prueba**, aunque lo cumplía en validación. Se reporta tal cual: la elección no se rehízo mirando la prueba.
- Como contraste, `lineal` (polinomial_ridge) mejora un +12,4 % en prueba, con 3,64 % en horas pico (no mantiene la ventaja).
- **Ningún candidato cumple el criterio de éxito en prueba en esta tabla.**

### Por encima de lo visto

276 de las 8760 horas de prueba (3,2 %) superan el máximo de entrenamiento + validación (11 736 MWh). En esas horas:

- `baseline` (baseline estacional (lag_168)): 242 predicciones por encima del máximo; sesgo medio -318 MWh (en el resto de horas, -5 MWh).
- `lineal` (polinomial_ridge): 14 predicciones por encima del máximo; sesgo medio -618 MWh (en el resto de horas, -195 MWh).
- `ensemble` (random_forest (nivel)): 0 predicciones por encima del máximo; sesgo medio -846 MWh (en el resto de horas, -218 MWh).

Un árbol predice con valores que vio y no pasa de ese máximo; la demanda de prueba sí lo pasa. El ensemble elegido trabaja sobre el nivel: en validación solo 3 horas superaban el máximo de entrenamiento y la forma diferenciada (`y - lag_168`), la que permitiría pasar ese techo, no ganó (`docs/resultados/modelado_ensembles.json`).

### Residuales del modelo final en prueba

Figura: `docs/figuras/modelado/residuales_r120.png`. Residual = predicción − real.

- Por hora: el MAPE va de 2,90 % (3:00) a 4,20 % (19:00).
- Por tipo de día:
  - laboral (n = 5736): MAPE 3,44 % frente a 3,26 % del baseline; sesgo -279 MWh.
  - fin de semana (n = 2448): MAPE 2,66 % frente a 2,63 % del baseline; sesgo -150 MWh.
  - festivo (n = 408): MAPE 4,81 % frente a 10,37 % del baseline; sesgo -284 MWh.
  - semana santa (n = 168): MAPE 3,19 % frente a 9,74 % del baseline; sesgo 20 MWh.
- Por día de la semana, el peor es el lunes (MAPE 4,09 %, sesgo -318 MWh).

### Alerta de pico (`alerta_pico`) en prueba

svm_rbf: PR-AUC 0,709 (prevalencia 12,2 %), ROC-AUC 0,954, F1 0,674, precisión 0,591, recall 0,784. Brier de la probabilidad calibrada: 0,0644.

## Lo que cuesta el rezago de publicación (r24 → r120)

r120 es el escenario operativo: la demanda real se publica con días de rezago (README, «Limitación importante para el modelado»), y r24 supone que se conoce la de ayer.

- Modelo final: MAPE de 2,21 % en r24 (hist_gradient (nivel)) a 3,28 % en r120 (random_forest (nivel)): +1,08 puntos, un +48,8 %.
- `baseline`: 3,54 % → 3,54 % (+0,00 puntos).
- `lineal`: 2,28 % → 3,10 % (+0,83 puntos).
- `ensemble`: 2,21 % → 3,28 % (+1,08 puntos).

## Limitaciones

- La probabilidad de alerta es una sigmoide de Platt ajustada con el clasificador entrenado solo con entrenamiento y aplicada al modelo final: es aproximada. La decisión de alerta usa el puntaje del modelo y su umbral natural, 0.
- Los tiempos dependen de la máquina (ver `entorno` en el JSON) y solo se comparan entre sí.
- La prueba se evaluó una vez; un nuevo ajuste a partir de estos resultados ya no tendría una estimación limpia del error.
