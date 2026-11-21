"""Modelado supervisado de la demanda horaria del SIN a 24 h.

Sobre las tablas de modelado r24 y r120 (src/variables), con las reglas de
CLAUDE.md: particion cronologica sin barajar, preprocesado dentro de un
Pipeline ajustado solo con entrenamiento, variables leidas de la tabla y el
bloque de prueba reservado para la evaluacion final.

- `datos`: carga una tabla, comprueba su SHA-256 y la separa en X, y, fechas.
- `particion`: bloques de entrenamiento, validacion y prueba, y TimeSeriesSplit.
- `metricas`: regresion (MAE, RMSE, MAPE, pico, maximo diario, sesgo) y
  clasificacion; alarma de fuga si el MAPE baja del 1 %.
- `baselines`: y_pred = lag_168 y, donde existe, y_pred = lag_24.
- `tiempos`: tiempo de ajuste, de prediccion y tamano serializado del modelo.

Como en la limpieza y las variables, cada operacion devuelve su resultado y un
registro serializable a JSON.
"""

__all__ = ["datos", "particion", "metricas", "baselines", "tiempos"]
