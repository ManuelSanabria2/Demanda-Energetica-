# Ingesta de demanda energética horaria — SIN Colombia

Capa de ingesta para un modelo predictivo de demanda energética horaria del
Sistema Interconectado Nacional colombiano, horizonte 24 h.

Dos fuentes, ambas de XM S.A. E.S.P.:

- **API XM (SINERGOX)** — `POST https://servapibi.xm.com.co/{granularidad}`
- **API SIMEM** — `GET https://www.simem.co/backend-files/api/PublicData`

La serie objetivo es **XM `DemaReal` / `Entity: Sistema`** (kWh, ya agregada a
nivel nacional). SIMEM `14fabb` se ingesta como fuente complementaria y para
reconciliación cruzada.

Todo lo que el código asume sobre estas APIs está verificado contra el servidor
y documentado en **[`notas/hallazgos_apis.md`](notas/hallazgos_apis.md)**.
Léelo antes de tocar el parseo: hay al menos una trampa que produce datos
plausibles pero mal escalados por un factor entero que además varía con el año
(hasta 4× en mayo de 2026).

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Histórico completo de ambas fuentes
PYTHONPATH=src python -m ingesta.cli --fuente todas --desde 2021-01-01

# Solo XM: ~73 s la primera vez, ~2 s con la caché ya poblada
PYTHONPATH=src python -m ingesta.cli --fuente xm --desde 2021-01-01 --hasta hoy

# Ignorar la caché y volver a pedirlo todo
PYTHONPATH=src python -m ingesta.cli --fuente simem --desde 2021-01-01 --sin-cache
```

En PowerShell: `$env:PYTHONPATH = "src"` antes del comando.

Scripts de verificación:

```bash
python scripts/explorar_apis.py               # JSON crudo sin intermediarios
python scripts/verificar_apis.py              # comprueba los invariantes del código
python scripts/reconciliar.py --mes 2026-08   # XM vs SIMEM, resuelve HourNN
python -m pytest tests -q
```

## Capa de acceso a datos

`src/ingesta/clientes.py` expone un cliente por API con la misma interfaz:

```python
import datetime as dt
from ingesta.clientes import ClienteXM, ClienteSIMEM

xm = ClienteXM()
df = xm.consultar("DemaReal", dt.date(2025, 1, 1), dt.date(2025, 2, 9), entidad="Sistema")
# -> 960 filas: timestamp, fuente, identificador, entidad, valor

simem = ClienteSIMEM()
df = simem.consultar("14fabb", dt.date(2025, 1, 1), dt.date(2025, 1, 3))
# -> timestamp, fuente, identificador + las columnas propias del dataset
```

Los dos clientes fragmentan el rango solo (XM lee `MaxDays` del catálogo;
SIMEM usa 31 días), concatenan los tramos y devuelven formato largo ya
ordenado por `timestamp`. `ClienteXM` convierte `Hour01…Hour24` a una fila por
hora usando `config.DESFASE_HORA_XM`, la única constante del proyecto que
codifica esa convención.

**Reintentos:** hasta 3 intentos, backoff exponencial de 2 s y 4 s, solo ante
5xx, timeouts y caídas de conexión. Un 4xx no se reintenta nunca — en estas dos
APIs significa consulta mal formada — y se propaga como `ErrorHTTP` con el
status y el cuerpo, que es donde ambas ponen el diagnóstico.

**Caché:** cada tramo se guarda en `data/cache/{fuente}/` como Parquet, con
clave `fuente + identificador + rango + kwargs que cambian la consulta`. Los
kwargs son parte de la clave a propósito: sin ellos `DemaReal/Sistema` y
`DemaReal/Agente` compartirían archivo. Un rango de 40 días baja de 2.6 s a
0.04 s en la segunda llamada.

Los tramos que cierran hace menos de `config.VENTANA_REFRESCO_DIAS` (45) **no
se cachean ni se leen de caché**: la demanda llega con 3 días de rezago y las
versiones de liquidación de SIMEM se revisan durante semanas, así que un tramo
reciente cacheado se quedaría congelado en datos parciales.

La capa de acceso **no** colapsa las versiones de liquidación de SIMEM: las
devuelve tal como llegan. Eso es trabajo de `normalizar`, y hacerlo aquí
esconderría la decisión más delicada del proyecto dentro de una descarga.

## Capa cruda: descarga incremental

`src/ingesta/descarga.py` persiste lo que devuelven los clientes, y
`descargar.py` es su CLI:

```bash
python descargar.py --fuente xm --id DemaReal                      # incremental
python descargar.py --fuente xm --id DemaReal --desde 2021-01-01   # rango explícito
python descargar.py --fuente simem --id 14fabb --hasta 2025-03-31
python descargar.py --fuente xm --id DemaReal --forzar             # ignora estado y caché

python descargar.py --historial                                    # bitácora de descargas
python descargar.py --estado --fuente xm --id DemaReal             # qué hay almacenado
```

Ruta en disco: `data/raw/{fuente}/{identificador}/anio={AAAA}/mes={MM}/datos.parquet`.
`anio` y `mes` viven solo en la ruta; `pandas.read_parquet` sobre el directorio
raíz los reconstruye como columnas.

**Incremental.** Sin `--desde`, mira la fecha máxima almacenada y pide desde
`VENTANA_REFRESCO_DIAS` (45) **antes** de esa fecha, no justo después. No es un
descuido: la demanda de XM llega con 3 días de rezago y las versiones de
liquidación de SIMEM se revisan durante semanas, así que los últimos días
guardados pueden estar incompletos o desactualizados.

**Idempotente.** Escribir no es añadir. Para cada partición afectada se lee lo
que había, se combina, se deduplica por una clave explícita y se reescribe la
partición entera. Verificado contra las APIs reales: dos ejecuciones idénticas
dan el mismo número de filas y el mismo hash.

La clave de deduplicación es *todas las columnas menos el valor*. Esto importa
en SIMEM: `Version` forma parte de la identidad, así que las cuatro versiones
que conviven sobre una misma hora se conservan las cuatro. La capa cruda guarda
lo que la fuente publicó; colapsarlas es trabajo de `normalizar`.

**Manifiesto.** `data/raw/manifiesto.json` es una bitácora *append-only*: cada
descarga anota rango solicitado, rango realmente obtenido, número de registros,
hash del resultado, hash por partición, duración y versiones de Python, pandas
y pyarrow. Esas versiones están ahí porque el hash depende de ellas.

**Es el único archivo bajo `data/` que se versiona en git.** Sin eso, el
registro de qué datos había en cada momento viviría solo en un disco local.

## Diagnóstico de calidad

`src/calidad/diagnostico.py` describe qué está mal en una serie. **No corrige
nada**: no rellena huecos, no elimina duplicados, no recorta extremos.

```python
from calidad.diagnostico import diagnosticar, resumen

informe = diagnosticar(marco)          # dict serializable a JSON
print(resumen(informe))                # versión legible
```

Cubre completitud temporal (huecos clasificados por duración, duplicados, días
que no tienen 24 horas), valores (nulos por columna, ceros y negativos,
atípicos por IQR global y por desviación frente a la misma hora del mismo día
de la semana) y estructura (rango, continuidad, cardinalidad de las
categóricas).

Distingue tablas **agregadas** de **desagregadas** por el número de filas por
marca de tiempo. En SIMEM los timestamps repetidos son normales, así que pasarle
`columnas_clave` permite separar un duplicado real de la desagregación
legítima, y `columnas_grupo` evita comparar agentes distintos entre sí.

Los hallazgos sobre los datos reales están en
**[`notas/hallazgos_calidad.md`](notas/hallazgos_calidad.md)**. Dos importan
para el modelado:

1. **Los dos últimos días publicados traen valores parciales** (~24 % de lo
   normal) aunque tengan las 24 horas y ningún nulo. El rezago útil es de ~5
   días, no de 3.
2. **`cli.py` puede destruir datos**: su `existing_data_behavior="delete_matching"`
   borra la partición entera, así que una ingesta parcial elimina el resto del
   mes. Ya ocurrió con marzo de 2025. Sin corregir.

## Salida

```
data/
  raw_samples/     muestras crudas de cada API (scripts/explorar_apis.py)
  cache/xm/        un Parquet por tramo descargado (caché de los clientes)
  raw/
    manifiesto.json                            bitácora de descargas (en git)
    xm/DemaReal/anio=2025/mes=01/datos.parquet capa cruda particionada
    simem/14fabb/anio=2026/mes=05/datos.parquet
  processed/
    xm_demanda_real_sistema/anio=2025/mes=1/*.parquet
    simem_demanda_real_nacional/anio=2025/mes=1/*.parquet
  manifiesto.json  cobertura de la capa procesada (huecos, rezago)
```

Hay **dos manifiestos, con dos propósitos**: `data/raw/manifiesto.json` es la
bitácora de descargas con hashes (reproducibilidad), y `data/manifiesto.json`
registra la cobertura de la capa procesada (huecos y rezago).

Esquema común de las tablas procesadas:

| columna | tipo | nota |
|---|---|---|
| `fecha_hora` | `datetime64[ns]` | hora local de Colombia, naive (no hay DST) |
| `fuente` | `str` | `xm` o `simem` |
| `metrica` | `str` | `DemaReal`, `DdaReal` |
| `entidad` | `str` | `Sistema`, `Nacional`, o código SIC del agente |
| `valor_kwh` | `float64` | `NaN` para huecos, nunca 0 |
| `version` | `str \| None` | versión de liquidación (solo SIMEM) |

Leer así:

```python
import pandas as pd
df = pd.read_parquet("data/processed/xm_demanda_real_sistema")
serie = df.set_index("fecha_hora")["valor_kwh"].sort_index()
```

## Estado actual de los datos

Ingesta completa ejecutada el 2026-09-09:

| | XM `DemaReal`/Sistema | SIMEM `14fabb`/Nacional |
|---|---|---|
| rango | 2021-01-01 → 2026-09-06 | 2021-01-01 → 2026-09-04 |
| filas horarias | 49 800 | 49 752 |
| horas faltantes | 0 | 0 |
| horas con NaN | 0 | 0 |
| duplicados | 0 | 0 |
| rezago de publicación | 3 días | 5 días |

Contraste entre ambas sobre las 49 752 horas comunes:

```
correlación   : 0.992516
dif. media    : +1.438 %   (XM sistemáticamente por encima)
```

El sesgo es de signo estable en los seis años, así que es diferencia de
**alcance** entre las dos series, no de alineamiento horario. Ver
`notas/hallazgos_apis.md` §3.

## Limitación importante para el modelado

**La demanda real se publica con ~3 días de rezago.** Un modelo a 24 h
entrenado sobre este dato no es operativo en tiempo real: en el momento de
predecir el día `d+1`, el último valor observado disponible es el de `d-3`, no
el de `d`.

Esto no se oculta: `data/manifiesto.json` registra `ultima_fecha_con_datos` y
`rezago_dias` en cada ingesta. Para el trabajo académico la vía razonable es
**backtesting histórico**, declarando el rezago como limitación. Si se quisiera
un uso operativo real habría que construir los rezagos de las features
respetando esa disponibilidad (nada de `lag_1` sobre la demanda real), o pasar
a las métricas de pronóstico del CND, que sí se publican por adelantado.

## Estructura

```
src/calidad/
  diagnostico.py  informe de calidad: completitud, valores, estructura
src/ingesta/
  config.py       rutas, constantes, precedencia de versiones, desfase horario
  clientes.py     ClienteXM y ClienteSIMEM: descarga, troceo, reintentos, caché
  descarga.py     capa cruda: Parquet particionado, incremental, manifiesto
  ventanas.py     partición del rango en llamados según MaxDays
  normalizar.py   esquema común, colapso de versiones (SIMEM), cobertura
  manifiesto.py   qué se descargó, hasta dónde llega, qué falta
  cli.py          orquestación y escritura a Parquet
scripts/
  explorar_apis.py    vuelca el JSON crudo y describe su estructura
  verificar_apis.py   comprueba los invariantes de los que depende el código
  reconciliar.py      XM vs SIMEM; resuelve empíricamente la convención HourNN
tests/
  test_clientes.py    troceo, reintentos, caché, ancho→largo (sin red)
  test_descarga.py    particionado, idempotencia, incremental, manifiesto
  test_diagnostico.py huecos, atípicos por dos criterios, fiabilidad
  test_ingesta.py     ventanas, colapso de versiones, cobertura
descargar.py          CLI de la capa cruda
```
