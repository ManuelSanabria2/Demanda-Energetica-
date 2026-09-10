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
python -m pytest tests -q          # 186 pruebas, ninguna toca la red
```

`tests/conftest.py` **bloquea la red para toda la suite**: una prueba que
intente salir falla con un mensaje que dice dónde mirar, en vez de tardar treinta
segundos y depender de que XM y SIMEM estén disponibles. Para una prueba que sí
la necesite: `@pytest.mark.red` y `pytest -m red`.

`tests/test_criticos.py` agrupa lo que puede fallar **en silencio** — troceo de
fechas, conversión `HourNN`, idempotencia, límite de interpolación e integridad
del merge — organizado por riesgo y no por módulo.

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
2. ~~**`cli.py` puede destruir datos**~~ — **corregido**. Su
   `existing_data_behavior="delete_matching"` borraba la partición entera, así
   que una ingesta parcial eliminaba el resto del mes; ocurrió con marzo de
   2025. Ahora lee y fusiona antes de reescribir, y marzo está reingestado.

## Fuentes complementarias: clima y calendario

`src/ingesta/complementarias.py` añade las dos fuentes exógenas y las une con la
demanda.

```python
from ingesta.complementarias import descargar_clima, calendario_horario, unir

clima, reg = descargar_clima(inicio, fin, pesos={"bogota": 0.45, ...})
cal, _     = calendario_horario(inicio, fin)
panel, informe = unir(demanda, clima, cal)   # falla si el merge no cuadra
```

**Clima** (Open-Meteo Archive): temperatura horaria de Bogotá, Medellín, Cali y
Barranquilla, agregada a una serie nacional por media ponderada. **Los pesos son
un parámetro**, se validan y se anotan en `data/processed/registro_clima.json`
en cada descarga, así que siempre consta con qué ponderación se construyó cada
serie.

> ⚠️ `PESOS_DEFECTO` es un **proxy provisional** por tamaño de área
> metropolitana, no una medición de la demanda por zona. La base correcta sería
> la participación de cada zona, calculable con los conjuntos de SIMEM
> `d91840` (demanda por área operativa) o `38FF5B` (por STR), pero el mapeo de
> área a ciudad no está hecho. `normalizar_pesos()` acepta magnitudes crudas
> (GWh por zona) para sustituir el proxy en cuanto tengas los datos.

Si a una hora le falta una ciudad se renormalizan los pesos de las presentes, y
`ciudades_disponibles` deja constancia de cuántas entraron.

**Calendario**: `holidays` con `country='CO'`, que ya aplica la **Ley Emiliani**
(traslado de festivos al lunes siguiente). Banderas: `es_festivo`,
`es_vispera_festivo`, `es_puente`, `es_semana_santa`,
`es_ultima_semana_diciembre`. Las definiciones de `es_puente` y
`es_semana_santa` son decisiones, no hechos, y quedan escritas en el registro.

**Integración**: `unir()` conserva exactamente las filas de la demanda y **falla
ruidosamente** si no. Aborta si la demanda o cualquier fuente traen marcas de
tiempo repetidas (multiplicaría filas en silencio), si dos tablas comparten
columnas, o —con `exigir_cobertura_total=True`— si alguna fuente deja huecos.

Resultado sobre los datos reales: 49 824 filas de demanda → 49 824 del panel,
**0 % sin cobertura** en clima y calendario. El clima diagnostica 100 % de
completitud, sin huecos ni nulos.

## Limpieza trazable

`src/limpieza/limpiar.py` aplica la limpieza y **registra cada transformación**.
Cada operación devuelve `(marco, registro)` con el criterio aplicado y las filas
afectadas; el registro completo va a `data/procesado/registro_limpieza.json`.

```python
from limpieza.limpiar import limpiar, guardar, procedencia, resumen

limpio, registro = limpiar(crudo)
guardar(limpio, registro, "demanda_horaria_sin")
print(resumen(registro))
```

Operaciones, en orden: normalización de esquema (snake_case, marca de tiempo
localizada en UTC-5, tipos fijos) → deduplicación por marca de tiempo → rejilla
horaria completa con interpolación acotada → marcado de atípicos → marcado del
periodo de pandemia.

**Política de imputación, deliberadamente conservadora.** Solo se interpolan
huecos de hasta 3 horas, y sin extrapolar en los extremos. Un hueco mayor se
queda como `NaN` y se marca. Un `NaN` se distingue del dato real; un valor
inventado, no.

**Nada se elimina.** Los atípicos y el periodo de pandemia se *marcan*.
Excluirlos o no es decisión del modelado.

### De dónde salió cada valor

El marco resultante lleva la procedencia fila a fila, y `procedencia()` la
explica para cualquier hora:

```python
>>> procedencia(limpio, "2025-03-10 12:00")
{'valor': None, 'origen_valor': 'faltante', 'hueco_horas': 624,
 'explicacion': 'Sin valor. Pertenece a un hueco de 624 h, por encima del
                 limite de 3 h, asi que se dejo como NaN en vez de inventarlo.'}
```

Columnas de procedencia: `origen_valor` (`observado` / `interpolado` /
`faltante`), `imputado`, `hueco_horas`, `atipico`, `atipico_iqr`,
`atipico_evaluable`, `z_estacional`, `periodo_atipico`, `etiqueta_periodo`.

Sobre los datos reales, tras reparar marzo de 2025: **49 824 observados
(100 %)**, 0 interpolados, 0 faltantes, 157 atípicos marcados y 0 valores
inventados.

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
src/limpieza/
  limpiar.py      limpieza trazable: cada operación devuelve marco + registro
src/calidad/
  diagnostico.py  informe de calidad: completitud, valores, estructura
src/ingesta/
  complementarias.py  clima (Open-Meteo) + calendario CO + unión validada
  particiones.py  escritura Parquet que fusiona en vez de sobrescribir
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
  test_limpieza.py    esquema, dedup, interpolación acotada, procedencia
  test_cli_escritura.py  regresión: una ingesta parcial no borra el resto del mes
  test_complementarias.py pesos, ponderación, Ley Emiliani, validación del merge
  test_criticos.py    lo que falla en silencio (JSON real fijado, casos borde)
  conftest.py         bloqueo de red para toda la suite
  test_ingesta.py     ventanas, colapso de versiones, cobertura
descargar.py          CLI de la capa cruda
```
