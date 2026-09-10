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
