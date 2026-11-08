# Diccionario de variables — tabla de modelado

Columnas de `data/procesado/tabla_modelado_r24.parquet` y
`tabla_modelado_r120.parquet`, las tablas de entrada del modelo de demanda
horaria del SIN a 24 h. Se generan desde el panel con:

```bash
python scripts/construir_panel.py --imputacion causal   # si el panel no existe
python scripts/construir_variables.py
```

El código está en `src/variables/construir.py`. Cada operación, su criterio, las
filas afectadas y cada columna excluida con su motivo quedan en
`data/procesado/registro_variables.json`, junto con el SHA-256 de cada tabla.

Todas las marcas de tiempo están en **hora local de Colombia**
(`America/Bogota`, siempre −05:00: no hay horario de verano).

---

## Las dos versiones: r24 y r120

Las dos tablas difieren en un solo parámetro, `rezago_minimo_h`: cuántas horas
separan la hora `t` que se predice de la **última hora de demanda disponible**
en el momento de predecir. Ninguna variable de la fila `t` usa demanda
posterior a `t − rezago_minimo_h`.

| | `r24` | `r120` |
|---|---|---|
| supuesto | al predecir `t` se conoce la demanda hasta `t − 24 h` | se conoce hasta `t − 120 h` |
| para qué sirve | backtesting académico | estimar lo que daría el modelo en operación |
| por qué ese valor | es el horizonte del modelo: con 24 h de antelación, lo más reciente posible es `t − 24` | la demanda se publica con 3 días de rezago y los 2 últimos días publicados llegan parciales (~24 % de lo normal): el último dato fiable tiene ~5 días ([`notas/hallazgos_calidad.md`](../notas/hallazgos_calidad.md), §1) |
| rezagos | `lag_24`, `lag_168` | `lag_168` (`lag_24` no se genera: 24 h < 120 h) |
| ventana de `ma_168`, `std_168` | `t − 191 h` … `t − 24 h` | `t − 287 h` … `t − 120 h` |
| filas × columnas | 49 513 × 18 | 49 417 × 17 |
| rango | 2021-01-08 23:00 → 2026-09-02 23:00 | 2021-01-12 23:00 → 2026-09-02 23:00 |
| filas de arranque eliminadas | 191 | 287 |

`r24` es la versión del README («para el trabajo académico la vía razonable es
backtesting histórico, declarando el rezago como limitación»). `r120` es la que
respeta la disponibilidad real: si las dos dan errores muy distintos, la
diferencia es lo que el rezago de publicación le cuesta al modelo.

Un rezago menor que `rezago_minimo_h` no se genera, y el registro lo anota en
`rezagos_omitidos`. Por debajo de 24 h la construcción falla: `lag_1` no puede
aparecer.

---

## Columnas de la tabla

«Disponible desde» se refiere al momento de predecir la hora `t`.

| columna | descripción | unidad | fuente | cálculo | disponible desde | ¿apta como variable? |
|---|---|---|---|---|---|---|
| `fecha_hora` | hora que se predice (inicio de la hora) | marca de tiempo, `America/Bogota` | panel | sin cambios | — | **no**: es la clave de la fila; para la partición temporal |
| `demanda_mwh` | **variable objetivo**: demanda real del SIN en la hora | MWh por hora (= MW medios) | XM `DemaReal` / `Sistema` | `valor_kwh / 1000` | se publica ~3 días después de `t`; fiable a los ~5 | **no**: es el objetivo |
| `lag_24` | demanda de la misma hora del día anterior | MWh | XM | `demanda_mwh` en la marca `t − 24 h`, buscada por marca de tiempo | `t − 24 h` | sí, **solo en r24** (no existe en r120) |
| `lag_168` | demanda de la misma hora de la semana anterior | MWh | XM | `demanda_mwh` en la marca `t − 168 h` | `t − 168 h` | sí, en las dos |
| `ma_168` | nivel reciente de la demanda | MWh | XM | media de las 168 horas `t − r − 167 h` … `t − r` (`r` = `rezago_minimo_h`); se desplaza la serie `r` horas **antes** de la ventana | `t − r` | sí, en las dos (ventana distinta en cada una) |
| `std_168` | variabilidad reciente de la demanda | MWh | XM | desviación estándar muestral (`ddof=1`) de la misma ventana que `ma_168` | `t − r` | sí, en las dos |
| `temperatura_c` | temperatura nacional ponderada en la hora `t` | °C | Open-Meteo Archive: Bogotá, Medellín, Cali y Barranquilla | media ponderada del panel, sin cambios; pesos proxy 0.45 / 0.22 / 0.18 / 0.15 por tamaño de área metropolitana | es la **observada** en `t`: solo se conoce después | sí en backtesting, con el supuesto de pronóstico perfecto (ver abajo) |
| `hora_sin`, `hora_cos` | hora del día en el círculo | adimensional, [−1, 1] | `fecha_hora` | `sin`/`cos` de `2π·hora/24`, hora local 0…23 | siempre (determinista) | sí |
| `dia_semana_sin`, `dia_semana_cos` | día de la semana en el círculo | adimensional, [−1, 1] | `fecha_hora` | `sin`/`cos` de `2π·día/7`, lunes = 0 … domingo = 6 | siempre | sí |
| `mes_sin`, `mes_cos` | mes del año en el círculo | adimensional, [−1, 1] | `fecha_hora` | `sin`/`cos` de `2π·(mes − 1)/12`, enero = 0 … diciembre = 11 | siempre | sí |
| `es_festivo` | el día es festivo en Colombia | 0/1 (`int8`) | `holidays` `country_holidays('CO')`, con la Ley Emiliani | bandera del panel pasada a entero; definición sin cambios | con antelación (calendario) | sí |
| `es_vispera_festivo` | el día siguiente es festivo | 0/1 | ídem | ídem | con antelación | sí |
| `es_puente` | sábado, domingo o lunes de un fin de semana largo por lunes festivo | 0/1 | ídem | ídem (definición corregida: no marca el fin de semana siguiente) | con antelación | sí |
| `es_semana_santa` | semana de lunes a domingo que contiene el Viernes Santo | 0/1 | ídem | ídem | con antelación | sí |
| `es_ultima_semana_diciembre` | del 25 al 31 de diciembre | 0/1 | ídem | ídem | con antelación | sí |

Las definiciones de las banderas son decisiones, no hechos: están en
`ingesta.complementarias.DEFINICIONES_CALENDARIO` y en `informe_panel.json`.

Las cíclicas se codifican con seno **y** coseno porque con un entero la hora 23
y la 0 quedan a 23 unidades; en el círculo quedan tan cerca como la 0 y la 1.
Ninguna de las dos funciones basta sola: el seno de las 2 y el de las 10 son
iguales.

---

## Columnas del panel que no entran

Ninguna se pierde sin dejar rastro: el registro las lista con este mismo motivo.
Si siguen haciendo falta (por ejemplo, para filtrar filas atípicas del
entrenamiento), se pueden volver a unir desde `panel_modelado.parquet` por
`fecha_hora`.

| columna | motivo |
|---|---|
| `atipico_global`, `atipico_iqr_global` | **fuga temporal**: se calculan con la serie entera, futuro incluido (lo advierte el README) |
| `z_estacional` | **fuga del objetivo**: `(demanda de t − mediana pasada) / MAD`; contiene el valor que se quiere predecir |
| `atipico`, `atipico_iqr` | **fuga del objetivo**: son causales respecto al pasado, pero comparan la demanda de la propia hora `t` con ese pasado. Sirven para filtrar o ponderar filas, no como variables de la misma hora |
| `atipico_evaluable` | procedencia de la marca de atípicos: dice si el grupo tenía histórico suficiente, no describe la demanda |
| `origen_valor`, `imputado`, `hueco_horas` | procedencia del objetivo. Sobre los datos actuales: 100 % observado, 0 imputados |
| `periodo_atipico`, `etiqueta_periodo` | marca de limpieza del periodo de pandemia (2020-03 … 2020-12), no una bandera del calendario. Vale 0 en todo el panel, que empieza en 2021 |
| `fuente`, `metrica`, `entidad` | texto de procedencia, constante |
| `version` | versión de liquidación: solo existe en SIMEM, nula en XM |
| `ciudades_disponibles` | procedencia del clima: cuántas ciudades entraron en la media ponderada (4 en todas las horas) |
| `nombre_festivo` | texto; la información está en `es_festivo` |
| `valor_kwh` | sustituida por `demanda_mwh`; ninguna variable la necesita en kWh |

Una columna nueva del panel que no sea variable ni figure en esta lista
(`COLUMNAS_EXCLUIDAS` en el código) hace fallar la construcción: decidir si es
variable no debe ocurrir en silencio.

---

## Filas eliminadas

El número de filas solo cambia por dos motivos, y el registro los cuadra en
`balance_filas` (panel − arranque − publicación parcial = final):

1. **Arranque sin histórico.** Las primeras filas no tienen 168 horas completas
   de ventana: 191 en r24 (`24 + 168 − 1`) y 287 en r120 (`120 + 168 − 1`). Solo
   se elimina ese bloque inicial; un NaN posterior se conservaría y se
   reportaría en `nan_fuera_del_arranque` (hoy no hay ninguno).
2. **Publicación parcial.** Los últimos 5 días de calendario del panel,
   2026-09-03 00:00 → 2026-09-07 23:00 (120 filas). Los dos últimos están al
   21 % de la mediana de los 28 días anteriores; el registro guarda el total de
   cada día descartado.

---

## Qué no se hace aquí

- **No se normaliza, escala ni imputa** con estadísticos del conjunto completo.
  Hacerlo antes de la partición temporal metería la media y la varianza del
  periodo de prueba en el entrenamiento. Va en el modelado, ajustado solo sobre
  el tramo de entrenamiento.
- **No se excluyen filas atípicas.** Es decisión del modelado; las marcas están
  en el panel.

---

## Supuestos

- **`temperatura_c` es la observada en `t`**, en las dos versiones. En
  backtesting equivale a suponer un pronóstico meteorológico perfecto, lo que
  favorece al modelo frente a la operación real. Para operar habría que
  sustituirla por un pronóstico emitido antes del momento de predicción.
- **Los pesos del clima son un proxy** por tamaño de área metropolitana, no la
  participación medida de cada zona en la demanda (README, «Fuentes
  complementarias»).
- **Los 5 días se cuentan como días de calendario locales**: un último día
  incompleto contaría igual como uno de los cinco. Hoy coincide con las últimas
  120 horas, porque el panel termina a las 23:00.
- **`std_168` es la desviación muestral** (`ddof=1`), y la ventana se exige
  completa (`min_periods=168`): una media de 3 horas no es una media de 168.
- **`r24` supone que la demanda se conoce en tiempo real.** Con la previsión
  emitida al cierre del día `d` para las 24 horas de `d+1`, `t − 24` es
  exactamente la última hora conocida para la hora 23, y queda por detrás de
  ella en las demás: es lo más reciente que vale para las 24 horas a la vez. Es
  el supuesto académico porque XM la publica días después.
