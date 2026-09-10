# Hallazgos de calidad de datos

Resultados de `src/calidad/diagnostico.py` sobre los datos reales del proyecto.
Fecha: **2026-09-10**. Reproducible con:

```python
import pandas as pd
from calidad.diagnostico import diagnosticar, resumen
print(resumen(diagnosticar(pd.read_parquet("data/processed/xm_demanda_real_sistema"))))
```

---

## 1. ⚠️ Los últimos días publicados traen valores parciales

**El hallazgo más importante, y el que ninguna comprobación anterior detectó.**

Totales diarios de `DemaReal`/Sistema, frente al día típico (218.4 GWh):

```
2026-09-03    261.9 GWh   119.9%
2026-09-04    265.2 GWh   121.4%
2026-09-05    249.9 GWh   114.4%
2026-09-06     53.2 GWh    24.3%   <-- parcial
2026-09-07     54.3 GWh    24.9%   <-- parcial
```

Los dos últimos días publicados están al **~24 % de su valor normal**. En 5.7
años de historia (2 076 días) esos son los **únicos dos días** por debajo del
50 % del típico: no es variabilidad, es un artefacto de publicación.

### Por qué no se había visto

Esos días tienen **las 24 horas presentes y ningún nulo**. La comprobación de
cobertura que hace la ingesta los da por perfectos:

```
horas faltantes : 0
horas con NaN   : 0
duplicados      : 0
```

Y lo son, en completitud temporal. El defecto está en los **valores**, no en la
rejilla de tiempo. Por eso hacía falta un diagnóstico que mirase las dos cosas.

Quien lo detecta es el criterio de atípicos **estacional**, no el global: a las
19:00 de un domingo la mediana histórica es 9.46 GWh y ese día marca 1.72 GWh,
lo que da z = −11.2. El criterio IQR global no lo señala, porque 1.72 GWh es un
valor que la serie alcanza de madrugada con normalidad.

### Consecuencia para el modelado

**El rezago útil no es de 3 días, es de ~5.** Documentar "la demanda se publica
con 3 días de rezago" es correcto pero insuficiente: los dos días más recientes
existen, parecen completos y están mal. Entrenar o evaluar con ellos mete un
error del 75 % en las observaciones más recientes, justo las que más pesan en un
modelo a 24 horas.

Recomendación: descartar los últimos **5 días** del histórico en cualquier
partición de entrenamiento o validación, o filtrar por total diario contra la
mediana móvil antes de usarlos.

Pendiente de confirmar: si el valor se completa al republicarse (probable, por
el ciclo de liquidación) o si queda así de forma permanente.

---

## 2. ⚠️ Falta marzo de 2025 en la capa procesada — causado por el propio código

`data/processed/xm_demanda_real_sistema` tenía **624 horas faltantes**, un único
tramo continuo:

```
2025-03-06 00:00 .. 2025-03-31 23:00   (624 h, 26 días)
```

No es un hueco de la API: XM publica ese periodo sin problema. Lo borró
`cli.py`. Su función `escribir_parquet` usa:

```python
pq.write_to_dataset(..., existing_data_behavior="delete_matching")
```

`delete_matching` **elimina la partición entera** antes de escribir. Al ejecutar
una ingesta de prueba acotada a `--desde 2025-03-01 --hasta 2025-03-05`, la
partición `anio=2025/mes=3` se borró completa y se reescribió con solo esos 5
días. Los otros 26 desaparecieron sin ningún aviso.

Es exactamente el fallo que `descarga.py` evita en la capa cruda leyendo,
fusionando y reescribiendo la partición. `cli.py` no hace eso todavía.

- **Alcance**: solo la capa procesada. La capa cruda (`data/raw/`) y la caché no
  se vieron afectadas.
- **Recuperable**: sí, reingestando el rango. Los datos siguen en la API.
- **Sin corregir**: el `delete_matching` de `cli.py` sigue ahí. Cualquier
  ingesta parcial vuelve a destruir el resto de sus particiones.

---

## 3. Observaciones menores

- **`version` es 100 % nula** en la serie de XM. Esperado: las versiones de
  liquidación son un concepto de SIMEM. La columna existe para que ambas fuentes
  compartan esquema.
- **Sin ceros ni negativos** en toda la serie de XM, y **ningún día con 23 o 25
  horas**, lo que confirma que el parseo de `HourNN` es correcto (Colombia no
  aplica horario de verano, así que cualquiera de esos casos habría sido un
  error de parseo).
- **Atípicos estacionales: 151 de 49 200 (0.31 %)** con fiabilidad alta
  (≈290 observaciones por grupo). Las horas 07–11 concentran la mayoría, lo que
  es consistente con el hallazgo 1: los días parciales tienen su mayor
  desviación relativa por la mañana.
- **SIMEM desagregado** (`14fabb`): 44 520 timestamps repetidos y **0 claves
  repetidas** sobre (timestamp, agente, mercado, versión). Correcto: la
  repetición es la desagregación por 65 agentes × 2 mercados × 4 versiones.

---

## Nota metodológica: fiabilidad del criterio estacional

La z modificada se apoya en la MAD, que subestima la dispersión en muestras
pequeñas e infla la tasa de atípicos. Medido sobre serie sintética limpia con
ruido gaussiano del 2 % (tasa teórica para |z|>3.5: ~0.05 %):

| días | obs/grupo | falsos positivos |
|---|---|---|
| 60 | 8.6 | 2.50 % |
| 120 | 17.1 | 0.83 % |
| 180 | 25.7 | 0.58 % |
| 365 | 52.1 | 0.24 % |
| 730 | 104.3 | 0.07 % |

Por eso el módulo exige **20 observaciones por grupo** (unos 5 meses) para
emitir un juicio, y declara la `fiabilidad` del resultado en el propio informe.
Con menos, dice que no puede juzgar en vez de dar un número engañoso.

La MAD tiene además un punto ciego: si un grupo es constante salvo por un único
valor extremo, su MAD es 0 y el atípico se vuelve invisible. Para ese caso el
módulo recurre a la desviación absoluta media, siguiendo la recomendación de
Iglewicz y Hoaglin.
