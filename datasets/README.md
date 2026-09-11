# datasets/

CSV de trabajo para exploración y limpieza, generados con:

```bash
python scripts/descargar_datasets.py                     # todo
python scripts/descargar_datasets.py --fuente xm --sin-ciiu
python scripts/descargar_datasets.py --fuente xm --solo-ciiu
```

Los datos no están en git (son cientos de MB y se regeneran en minutos desde la
caché). Sí se versionan este README y `manifiesto_datasets.json`, que registra
cada generación: qué archivos, cuántas filas, qué rango y el **SHA-256 de cada
archivo**, para poder comprobar después que un CSV es el que se generó.

Todas las marcas de tiempo están en **hora local de Colombia** con el
desplazamiento explícito (`2025-01-01 00:00:00-05:00`). Colombia no aplica
horario de verano: todos los días tienen 24 horas.

---

## `simem/` — demanda real nacional de SIMEM (`14fabb`)

| archivo | filas | tamaño |
|---|---|---|
| `simem_14fabb_2021.csv` | 755 640 | 79 MB |
| `simem_14fabb_2022.csv` | 769 416 | 81 MB |
| `simem_14fabb_2023.csv` | 784 680 | 82 MB |
| `simem_14fabb_2024.csv` | 772 632 | 81 MB |
| `simem_14fabb_2025.csv` | 2 226 960 | 234 MB |
| `simem_14fabb_2026.csv` | 2 093 856 | 220 MB |
| **total desagregado** | **7 403 184** | **777 MB** |
| `simem_14fabb_nacional.csv` | 49 776 | 3 MB |

Columnas del desagregado: `timestamp, fuente, identificador, CodigoVariable,
FechaHora, CodigoSICAgente, TipoMercado, Version, Valor, UnidadMedida,
CodigoDuracion`. Una fila por hora × agente × tipo de mercado × **versión de
liquidación**.

> ⚠️ **No sumes `Valor` por hora sin más.** Sobre una misma `FechaHora`
> conviven varias versiones de liquidación (`TX2`, `TX3`, `TXR`, `TXF`…), y la
> suma ingenua multiplica la demanda por el número de versiones — hasta ×5 en
> enero de 2026, y el factor cambia de un mes a otro. Es la razón de que 2025 y 2026
> pesen el triple que los años anteriores: están aún en revisión y traen varias
> versiones por hora. Usa `ingesta.normalizar.simem_agregar_nacional()`, que
> colapsa a una versión por hora antes de sumar, o directamente
> `simem_14fabb_nacional.csv`, que ya viene así.

`simem_14fabb_nacional.csv` está en el esquema común del proyecto:
`fecha_hora, fuente, metrica, entidad, valor_kwh, version`.

Verificado al generarlos: ninguna fila cae fuera del año de su archivo, la suma
de los seis coincide con el total del histórico, y el nacional no tiene horas
duplicadas.

---

## `xm/` — API de XM (SINERGOX)

| archivo | filas | contenido |
|---|---|---|
| `xm_catalogo_metricas.csv` | 193 | inventario de métricas: `MetricId, MetricName, Entity, MaxDays, Type, Url, Filter, MetricUnits, MetricDescription` |
| `xm_demareal_sistema.csv` | 49 824 | demanda real del SIN, kWh — **la serie objetivo del modelo** |
| `xm_demacome_sistema.csv` | 49 824 | demanda comercial del SIN, kWh |

Columnas de las dos series: `timestamp, fuente, identificador, entidad, valor`.
Comprobación rápida: `xm_demareal_sistema.csv` vale `7306335.34` el
`2025-01-01 00:00:00-05:00`, el valor verificado contra el servidor.

> ⚠️ **Los dos últimos días publicados llegan parciales** (~24 % de lo normal)
> aunque tengan las 24 horas y ningún nulo. El rezago útil es de ~5 días, no de
> 3: descarta los últimos 5 días de cualquier partición de entrenamiento o
> validación. Ver `notas/hallazgos_calidad.md`.

---

## `xm/ciiu/` — demanda comercial no regulada por CIIU (`DemaComeNoReg`/`CIIU`)

| archivo | filas | horas | subactividades/hora | tamaño |
|---|---|---|---|---|
| `xm_demacomenoreg_ciiu_2021.csv.gz` | 3 091 296 | 8 760 | 353 | 71 MB |
| `xm_demacomenoreg_ciiu_2022.csv.gz` | 3 098 112 | 8 760 | 354 | 71 MB |
| `xm_demacomenoreg_ciiu_2023.csv.gz` | 3 105 720 | 8 760 | 355 | 71 MB |
| `xm_demacomenoreg_ciiu_2024.csv.gz` | 3 064 080 | 8 784 | 349 | 70 MB |
| `xm_demacomenoreg_ciiu_2025.csv.gz` | 3 073 080 | 8 760 | 351 | 70 MB |
| `xm_demacomenoreg_ciiu_2026.csv.gz` | 2 104 392 | 6 000 | 351 | 48 MB |
| **total** | **17 536 680** | | | **401 MB** |

Verificado al generarlos: ninguna fila fuera del año de su archivo, ningún nulo
en `Activity`/`Subactivity`, **cero duplicados** sobre `(timestamp, Activity,
Subactivity)` y todas las horas del año presentes. El número de subactividades
por hora varía entre 349 y 355 porque XM añade y retira códigos con el tiempo.

Columnas: `timestamp, fuente, identificador, entidad, valor, Activity,
Subactivity`. Una fila por hora × subactividad económica (349–355 por hora, en 20–21
actividades).

> ⚠️ **La dimensión está en `Activity` y `Subactivity`, no en `entidad`**, que
> vale siempre `"CIIU"`. En esta métrica la API mete esas dos claves dentro de
> `Values`; el cliente del proyecto las descartaba en silencio y dejaba miles
> de filas con la misma marca de tiempo. La clave de una fila es
> `(timestamp, Activity, Subactivity)`.

Van en `.csv.gz` porque las descripciones de actividad son cadenas largas que
se repiten en cada fila: en CSV plano el histórico pasaría de 2,5 GB. Se leen
igual que un CSV:

```python
pd.read_csv("datasets/xm/ciiu/xm_demacomenoreg_ciiu_2025.csv.gz", parse_dates=["timestamp"])
```

Se descargan en tramos de 7 días. Con el servidor libre una semana tarda 2–3 s
y el histórico completo baja en unos 15 minutos; pero con el servidor cargado
una semana llegó a tardar 34 s y un tramo de 3 días agotó 300 s, así que un
tramo de 31 días podría rebasar el timeout de 180 s del cliente.

---

## Cómo se generan

- Cada archivo se escribe **tramo a tramo** (ventanas de 31 días, 7 para CIIU):
  en memoria nunca hay más de un tramo. La primera versión cargaba un año entero
  de SIMEM (2,1 M filas) y se quedó sin memoria.
- La escritura es **atómica**: se escribe a `*.tmp` y se renombra al terminar.
  Una descarga interrumpida nunca deja un archivo con nombre definitivo. (Pasó:
  un `.csv.gz` truncado a mitad de año con su nombre final.)
- Las filas se escriben en un **orden fijo**, así que dos generaciones sobre los
  mismos datos producen archivos idénticos byte a byte — comprobado con los CSV
  de XM: mismos SHA-256 en dos pasadas seguidas.
- Lo ya descargado se sirve desde la caché de los clientes (`data/cache/`), así
  que regenerar es rápido salvo los últimos 45 días, que se piden siempre de
  nuevo porque aún pueden cambiar.

---

## `limpios/` — salidas de los notebooks de limpieza

Un notebook por dataset (`notebooks/03`–`06`). Cada salida va con su
`registro_*.json`, que dice qué operación se aplicó, con qué criterio y a cuántas
filas; esos registros sí están en git.

| archivo | notebook | filas | qué se hizo |
|---|---|---|---|
| `demanda_real.parquet` | `03_limpieza_demanda_real` | 49 824 | limpieza completa; 100 % observados |
| `catalogo_metricas.csv` | `04_limpieza_catalogo_metricas` | 193 | texto normalizado, `/list` → `/lists`, `granularidad`, `entidad_normalizada` |
| `demanda_comercial.parquet` | `05_limpieza_demanda_comercial` | 49 824 | limpieza completa + `menor_que_real` (4 horas) |
| `ciiu_limpio.parquet` | `06_limpieza_ciiu` | 17 631 984 | limpieza sector por sector (367 grupos) |

Las series llevan la procedencia de cada valor: `origen_valor` (`observado`,
`interpolado`, `faltante`), `imputado`, `hueco_horas` y las marcas de atípicos.
**Nada se elimina**: los atípicos se marcan y los huecos de más de 3 horas se
quedan como NaN.

**Catálogo.** `entity` conserva el literal que espera la API (`"SubArea"` y
`"Subarea"` siguen distintas); para agrupar está `entidad_normalizada`. La URL
de las 7 métricas de listado se corrigió de `/list` (404) a `/lists` (200), con
`url_corregida = True`.

**Demanda comercial.** Queda un 1,5 % por encima de la real (incluye pérdidas),
salvo en 4 horas donde es menor —2021-02-12 12h, 2021-03-05 09h, 2024-06-09 09h y
2024-11-30 10h—, marcadas en `menor_que_real`. No se corrigen: no hay forma de
saber cuál de las dos series está mal.

**CIIU.** De 17 536 680 filas de entrada salen 17 631 984: las 95 304 horas que
faltaban dentro de la vida de 16 sectores se añaden a la rejilla, con su
sector y marcadas. Resultado: 98,95 % observados, 8 511 interpolados (huecos de
hasta 3 h, en 102 sectores) y 177 095 faltantes (huecos largos, en 90 sectores).
La rejilla de cada sector va de *su* primera a *su* última hora: a un sector
que vivió unos meses no se le inventan horas fuera de ellos.

> ⚠️ **En CIIU la marca de atípicos sirve bastante menos.** Marca 901 528 horas
> (5,1 %), frente al 0,3–0,6 % de las series nacionales: los sectores pequeños
> son mucho más ruidosos y el umbral pensado para la demanda agregada salta con
> frecuencia. Además, 1 401 758 filas no se pueden evaluar: son el arranque de
> cada sector, mientras acumula el histórico mínimo, y los sectores de vida
> corta. Antes de usar `atipico` en CIIU conviene revisar el umbral por sector.

Leerlo sin cargar las 17,6 M de filas:

```python
import pyarrow.parquet as pq
educacion = pq.read_table("datasets/limpios/ciiu_limpio.parquet",
                          filters=[("activity", "==", "EDUCACIÓN")]).to_pandas()
```
