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
(hasta 5× en enero de 2026).

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Histórico completo de XM, SIMEM y el clima del panel
PYTHONPATH=src python -m ingesta.cli --fuente todas --desde 2021-01-01

# Solo el clima (Open-Meteo), por ejemplo tras refrescar la demanda
PYTHONPATH=src python -m ingesta.cli --fuente clima --desde 2021-01-01

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
python -m pytest tests -q          # 270 pruebas, ninguna toca la red
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

## Datasets en CSV y notebooks

Para trabajo exploratorio, fuera del pipeline de Parquet:

```bash
python scripts/descargar_datasets.py             # genera datasets/ entero
python scripts/descargar_datasets.py --sin-ciiu  # sin la descarga larga
python scripts/generar_notebooks.py              # regenera notebooks/
```

`datasets/` contiene la demanda de SIMEM desagregada (un CSV por año), la serie
nacional agregada, el catálogo de las 193 métricas de XM, la demanda real y
comercial del SIN, y la demanda comercial por CIIU (un `.csv.gz` por año). Qué
es cada archivo, sus columnas y sus trampas: [`datasets/README.md`](datasets/README.md).

`notebooks/` trae cuadernos que **reutilizan el paquete**, no lo reimplementan,
y se publican ya ejecutados. Los de limpieza van **en una carpeta por dataset**,
con la fuente como prefijo para no confundir la demanda real de XM con la de SIMEM:

```
notebooks/
  00_configuracion.ipynb          comprueba entorno, paquete y datasets
  01_carga_y_exploracion.ipynb    columnas, perfiles, la trampa de las versiones de SIMEM
  02_diagnostico_calidad.ipynb    huecos, días incompletos, atípicos por dos criterios
  limpieza/
    xm_demanda_real/        DemaReal/Sistema, la serie objetivo
    xm_demanda_comercial/   DemaCome/Sistema y coherencia con la real
    xm_catalogo_metricas/   catálogo de 193 métricas; corrige la URL /list -> /lists
    xm_ciiu/                demanda por CIIU, sector por sector
    simem_nacional/         serie nacional de SIMEM y diferencia con XM
    simem_desagregado/      detalle por empresa: colapso de versiones + limpieza por grupos
```

Las salidas limpias y sus registros van a `datasets/limpios/`. Para regenerar
solo algunos notebooks: `python scripts/generar_notebooks.py xm_ciiu simem_nacional`.

> La métrica de CIIU (`DemaComeNoReg`/Entity=CIIU) trae `Activity` y
> `Subactivity` **dentro de `Values`**, no en el `Id` de la entidad. El cliente
> las descartaba en silencio y dejaba miles de filas con la misma marca de
> tiempo; ahora las conserva como columnas.

## Panel de modelado

**Esta es la base para modelar**: de ella sale la
[tabla de modelado](#tabla-de-modelado). Se construye siempre en el mismo orden:
demanda cruda → limpieza (procedencia y marcas) → unión con clima y calendario.

```bash
python scripts/construir_panel.py                      # 49 824 × 25
python scripts/construir_panel.py --imputacion causal   # sin mirar al futuro
```

Sale en `data/procesado/panel_modelado.parquet` con su informe al lado. Lleva
la procedencia de cada valor **y** las exógenas; antes había dos tablas finales
que no se hablaban y ninguna servía.

**Falla si el clima no cubre todas las horas de la demanda**, o si no hay clima:
antes se construía igual y dejaba la temperatura en `NaN` en las horas nuevas.
El mensaje dice qué ejecutar (`--fuente clima`); `--sin-clima` lo omite a
propósito.

`atipico_global` y `atipico_iqr_global` se calculan sobre la serie entera:
**no las uses como variables**, son fuga temporal. `atipico` y `atipico_iqr`
son causales respecto al pasado, pero se calculan con la demanda de la propia
hora: sirven para filtrar filas, no como variables de esa hora (ver
[Tabla de modelado](#tabla-de-modelado)).

## Tabla de modelado

**Esta es la entrada del modelo**: el panel con las variables construidas, en
dos versiones según cuánta demanda se conoce al predecir.

```bash
python scripts/construir_panel.py --imputacion causal   # antes, si no hay panel
python scripts/construir_variables.py                   # r24 y r120
```

Sale en `data/procesado/tabla_modelado_r24.parquet` y
`tabla_modelado_r120.parquet`, con `registro_variables.json` al lado (en git):
cada operación con su criterio y sus filas, cada columna excluida con su motivo
y el SHA-256 de cada tabla. Qué es cada columna, de dónde sale y desde cuándo se
conoce: **[`docs/diccionario_variables.md`](docs/diccionario_variables.md)**.

| | `r24` | `r120` |
|---|---|---|
| supuesto | demanda conocida hasta `t − 24 h` | hasta `t − 120 h` |
| uso | backtesting académico | disponibilidad real |
| filas × columnas | 49 513 × 18 | 49 417 × 17 |
| rango | 2021-01-08 23:00 → 2026-09-02 23:00 | 2021-01-12 23:00 → 2026-09-02 23:00 |
| rezagos | `lag_24`, `lag_168` | `lag_168` |
| ventana de `ma_168`, `std_168` | acaba en `t − 24 h` | acaba en `t − 120 h` |

**`rezago_minimo_h` es el único parámetro de la fuga temporal**: ninguna
variable de la fila `t` usa demanda posterior a `t − rezago_minimo_h`. Un rezago
menor no se genera y queda anotado (`lag_24` en r120); por debajo de 24 h, el
horizonte, la construcción falla, así que `lag_1` no puede aparecer. Rezagos y
ventanas se calculan **por marca de tiempo**, después de comprobar que la
rejilla horaria está completa: con un hueco, «168 filas atrás» dejaría de ser
«168 horas atrás» sin avisar.

Objetivo: `demanda_mwh = valor_kwh / 1000`. Variables: los rezagos, la media y
la desviación de 168 h sobre la serie ya desplazada, `temperatura_c`, seno y
coseno de la hora, el día de la semana y el mes, y las cinco banderas del
calendario en 0/1, con sus definiciones intactas.

**Fuera de la tabla**, con el motivo en el registro: las dos marcas `_global`
y también `atipico`, `atipico_iqr` y `z_estacional`, porque se calculan con la
demanda de la propia hora `t` y como variables de `t` contienen el objetivo. Ni
texto ni procedencia (`origen_valor`, `nombre_festivo`,
`ciudades_disponibles`…). Una columna nueva del panel sin clasificar hace
fallar la construcción en vez de colarse.

Las filas solo cambian por dos eliminaciones, que el registro cuadra:

- **el arranque sin histórico**: 191 h en r24 y 287 h en r120, lo que pide la
  ventana de 168 h desplazada;
- **los últimos 5 días** (2026-09-03 → 2026-09-07) por publicación parcial: los
  dos últimos están al 21 % de la mediana de 28 días.

**No se normaliza, escala ni imputa nada** con estadísticos del conjunto
completo. Eso va en el modelado, después de la partición temporal.

> `temperatura_c` es la temperatura **observada** en la propia hora. En
> backtesting equivale a suponer un pronóstico meteorológico perfecto; para
> operar habría que sustituirla por un pronóstico emitido antes del momento de
> predicción.

`tests/test_variables.py` lo comprueba para las dos versiones: cada rezago es
exactamente la demanda de esa marca de tiempo, la media y la desviación
recalculadas a mano coinciden, y alterar la demanda futura no cambia ninguna
variable anterior.

## Ausencia de fuga temporal

Las marcas de atípicos se calculan con estadísticos **causales**: cada fila se
juzga solo contra las observaciones anteriores de su grupo (día de semana,
hora). Medido antes de corregirlo, sobre la serie real y cortando en
2023-12-31, el criterio global marcaba 90 atípicos y el causal 255: **165
banderas cambiaban** según se incluyera o no el futuro. Tras la corrección, 0.

La zona horaria es **`America/Bogota` en todas las capas** — Colombia no aplica
DST, así que equivale siempre a −05:00. Antes la capa limpia era tz-aware y el
resto naive, y unirlas lanzaba un `ValueError` de pandas.

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

> `PESOS_DEFECTO` es un **proxy provisional** por tamaño de área
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
src/variables/
  construir.py    tabla de modelado: rezagos por marca de tiempo, cíclicas, selección
src/limpieza/
  limpiar.py      limpieza trazable: cada operación devuelve marco + registro
  grupos.py       limpieza de tablas desagregadas, un grupo cada vez
  catalogo.py     limpieza del catálogo de métricas de XM
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
  descargar_datasets.py  genera datasets/ en CSV, tramo a tramo
  generar_notebooks.py   define y regenera los notebooks
  construir_panel.py  demanda limpia + clima + calendario -> panel de modelado
  construir_variables.py  panel -> tablas de modelado r24 y r120 + registro
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
  test_criticos.py    lo que falla en silencio + ausencia de fuga temporal
  test_variables.py   fuga temporal de la tabla de modelado, en r24 y en r120
  conftest.py         bloqueo de red para toda la suite
  test_ingesta.py     ventanas, colapso de versiones, cobertura
  test_datasets.py    escritor de CSV: escritura atómica, orden fijo, gzip
  test_limpieza_catalogo.py  limpieza del catálogo de métricas
  test_limpieza_grupos.py    limpieza por grupos y etiqueta en filas creadas
descargar.py          CLI de la capa cruda
docs/
  diccionario_variables.md  una fila por columna de la tabla de modelado
datasets/             CSV de trabajo (fuera de git salvo README, manifiesto y registros)
  limpios/            salidas de los notebooks de limpieza
notebooks/            00–02 exploración y diagnóstico
  limpieza/           una carpeta por dataset: xm_* y simem_*
```
