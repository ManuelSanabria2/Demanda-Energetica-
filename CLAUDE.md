# Reglas del proyecto

Pronóstico de la demanda horaria del SIN colombiano a 24 h. Antes de tocar nada,
lee `README.md` y `docs/diccionario_variables.md`.

## Entorno

- Usa siempre el intérprete del proyecto: `.venv/Scripts/python.exe` (Python
  3.12). El Python global de la máquina trae numpy 2 y no tiene pyarrow: con él
  no se puede leer ningún Parquet.
- `numpy==1.26.4` está fijado porque pyarrow 14 no importa con numpy 2
  (`tests/test_requisitos.py` lo vigila). Un paquete nuevo se añade a
  `requirements.txt` con versión fijada y se verifica importándolo en el
  `.venv`, no por suposición. shap no puede pasar de 0.49.1 mientras numpy sea 1.x.
- Todo lo que se importe tiene que estar en `requirements.txt`; los paquetes de
  `src/` van en `PAQUETES_PROPIOS` de `tests/test_requisitos.py`.

## Convenciones del repo

- Cada operación devuelve `(marco, registro)`: el registro dice qué criterio se
  aplicó, a cuántas filas y con qué parámetros (ver `src/variables/construir.py`).
- Los registros son JSON con `version_formato`, escritos de forma atómica
  (`.tmp` y después `replace`).
- `tests/conftest.py` bloquea la red para toda la suite. Una prueba que necesite
  red se marca con `@pytest.mark.red` y no corre por defecto.
- Entradas del modelado: `data/procesado/tabla_modelado_r24.parquet` y
  `tabla_modelado_r120.parquet`. Antes de usarlas, comprueba que su SHA-256
  coincide con el de `data/procesado/registro_variables.json`; si no, regenéralas
  con los comandos del README.

## Reglas del modelado supervisado

### Partición y validación

- **Nunca `shuffle=True` ni particiones aleatorias.** La validación cruzada se
  hace solo con `TimeSeriesSplit`, con `gap` de al menos 24 h. `gap` cuenta
  filas: comprueba que la tabla es una rejilla horaria sin huecos (diferencias de
  `fecha_hora` de exactamente 1 h) para que 24 filas sean 24 h.
- **El conjunto de prueba, de 2025-09-03 00:00 a 2026-09-02 23:00
  (`America/Bogota`), no se usa hasta el prompt de evaluación final.** Ni para
  mirarlo, ni para elegir variables, ni para escalar. La selección de modelos y
  de hiperparámetros se hace siempre en validación, dentro del tramo anterior a
  2025-09-03.

### Sin fuga en el preprocesado

- **Todo escalado, expansión polinomial o imputación va dentro de un
  `sklearn.pipeline.Pipeline`**, ajustado solo con los datos de entrenamiento de
  cada pliegue. Nada se ajusta con la tabla completa.
- **`X` nunca incluye `demanda_mwh`, `fecha_hora` ni columnas que el diccionario
  marca como no aptas.** La lista de variables sale de la tabla (o de
  `variables` en `registro_variables.json`), no se escribe a mano en cada script.
  `lag_24` existe en r24 y no en r120: un código que la nombre a mano se rompe
  en una de las dos.
- **Si una métrica de regresión da MAPE < 1 %, detente y audita la fuga antes
  de seguir.** Con lo que se sabe de esta serie, un error así significa que el
  objetivo se coló en `X`, no que el modelo sea bueno.

### Reproducibilidad

- `random_state=42` en todo lo que tenga azar.
- **Cada experimento se hace para r24 y para r120.** La diferencia entre las
  dos es lo que el rezago de publicación le cuesta al modelo.
- Resultados en `docs/resultados/modelado_*.json`; figuras en
  `docs/figuras/modelado/`; modelos entrenados en `models/` (fuera de git), con
  su SHA-256 anotado en el registro del experimento.
- **Ningún número se escribe a mano en el informe.** Las cifras llegan por
  `docs/informe/generar_valores.py`, que las lee de `docs/resultados/*.json`.

### Pruebas y commits

- Cada módulo nuevo lleva pruebas sin red. La suite completa
  (`.venv/Scripts/python.exe -m pytest tests -q`) pasa antes de cada commit.
- Identificadores y comentarios en español, como el resto del repo.
- Commits pequeños, uno por paso. No hacer push sin que lo pidan.
