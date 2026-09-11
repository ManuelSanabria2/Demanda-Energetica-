"""Genera los notebooks de `notebooks/` a partir de este script.

Tenerlos definidos en codigo permite regenerarlos si cambia el paquete, en vez
de editar JSON a mano. Despues se ejecutan para que queden con las salidas.

    python scripts/generar_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DIR = Path(__file__).resolve().parents[1] / "notebooks"

ARRANQUE = '''import sys, warnings
from pathlib import Path

RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)
warnings.filterwarnings("ignore", category=FutureWarning)

DATASETS = RAIZ / "datasets"
print("raiz     :", RAIZ)
print("datasets :", DATASETS, "->", "existe" if DATASETS.exists() else "FALTA")'''

# La demanda de XM se pasa al esquema comun del proyecto (fecha_hora,
# valor_kwh...) con la misma funcion que usa la ingesta, en vez de renombrar a
# mano: asi el notebook no se desalinea si el esquema cambia.
CARGA_XM = '''from ingesta.normalizar import xm_cliente_a_esquema_comun

crudo_xm = pd.read_csv(DATASETS / "xm" / "xm_demareal_sistema.csv", parse_dates=["timestamp"])
xm = xm_cliente_a_esquema_comun(crudo_xm)
print(f"{len(xm):,} filas · {xm.fecha_hora.min()} .. {xm.fecha_hora.max()}")'''


def cuaderno(celdas: list[tuple[str, str]]) -> nbf.NotebookNode:
    libro = nbf.v4.new_notebook()
    libro.cells = [
        nbf.v4.new_markdown_cell(texto) if tipo == "md" else nbf.v4.new_code_cell(texto)
        for tipo, texto in celdas
    ]
    libro.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return libro


NB00 = cuaderno([
    ("md", """# 00 · Configuración

Comprueba que el entorno está listo: el paquete `src/` importable, las librerías
instaladas y los datasets en su sitio. La celda de arranque se repite al
principio de los demás notebooks."""),
    ("code", ARRANQUE),
    ("md", "## Versiones\n\nEl hash de los datos depende de la versión de pandas, así que conviene dejarla anotada."),
    ("code", '''import platform, pyarrow, holidays
print(f"python   {platform.python_version()}")
print(f"pandas   {pd.__version__}")
print(f"pyarrow  {pyarrow.__version__}")
print(f"holidays {holidays.__version__}")'''),
    ("md", """## El paquete del proyecto

Los notebooks **no reimplementan** la limpieza: importan lo que ya está escrito y
probado (colapso de versiones, atípicos causales, procedencia de cada valor)."""),
    ("code", '''from calidad.diagnostico import diagnosticar, resumen
from limpieza.limpiar import limpiar, procedencia
from ingesta import config

print("zona horaria del proyecto :", config.ZONA_COLOMBIA)
print("desfase HourNN            :", config.DESFASE_HORA_XM, "(Hour01 = 00:00-01:00)")
print("precedencia de versiones  :", config.PRECEDENCIA_VERSIONES)'''),
    ("md", "## Qué hay descargado"),
    ("code", '''# Los .tmp son archivos a medio escribir por una descarga en curso: no cuentan.
archivos = sorted(p for p in DATASETS.rglob("*.csv*") if p.suffix != ".tmp") if DATASETS.exists() else []
for a in archivos:
    print(f"{a.stat().st_size/1e6:>8.1f} MB  {a.relative_to(DATASETS)}")
if not archivos:
    print("No hay CSV. Ejecuta: python scripts/descargar_datasets.py")'''),
])


NB01 = cuaderno([
    ("md", """# 01 · Carga y exploración

Primer contacto con los CSV: columnas, tipos, cardinalidad de las categóricas y
forma de la demanda a lo largo del día y de la semana."""),
    ("code", ARRANQUE),
    ("md", "## Demanda real del SIN (XM)\n\nUna fila por hora. Es la serie objetivo del modelo."),
    ("code", CARGA_XM + "\nxm.head()"),
    ("code", '''print(xm.dtypes.to_string())
print()
print("filas por marca de tiempo:", round(len(xm) / xm.fecha_hora.nunique(), 2))
print("nulos:", int(xm.valor_kwh.isna().sum()))'''),
    ("md", """### Perfil horario y semanal

La demanda tiene dos estacionalidades muy marcadas. Verlas ahora ahorra
sorpresas: un criterio de atípicos que ignore la hora del día marcará como raros
los picos normales de la tarde."""),
    ("code", '''hora = xm.fecha_hora.dt.hour
dia = xm.fecha_hora.dt.dayofweek

fig, ejes = plt.subplots(1, 2, figsize=(13, 4))
perfil = xm.groupby(hora).valor_kwh.median() / 1e6
ejes[0].plot(perfil.index, perfil.values, marker="o")
ejes[0].set(title="Perfil horario (mediana)", xlabel="hora", ylabel="GWh")
ejes[0].grid(alpha=.3)

semanal = xm.groupby(dia).valor_kwh.median() / 1e6
ejes[1].bar(["lun", "mar", "mié", "jue", "vie", "sáb", "dom"], semanal.values)
ejes[1].set(title="Perfil semanal (mediana)", ylabel="GWh")
ejes[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()'''),
    ("md", """## Catálogo de métricas de XM

Las 193 métricas disponibles. `MaxDays` es el límite de días por llamado."""),
    ("code", '''cat = pd.read_csv(DATASETS / "xm" / "xm_catalogo_metricas.csv")
print(f"{len(cat)} métricas · MaxDays presentes: {sorted(cat.MaxDays.unique())}")
print(cat.Type.value_counts().to_string())
cat[cat.MetricName.str.contains("Demanda", case=False, na=False)][
    ["MetricId", "Entity", "Type", "MaxDays", "MetricUnits"]
].head(15)'''),
    ("md", """## SIMEM desagregado

⚠️ **La trampa del proyecto.** Sobre una misma `FechaHora` conviven varias
versiones de liquidación. Sumar sin filtrarlas multiplica la demanda por el
número de versiones, y el resultado sigue pareciendo una curva plausible.

Los CSV anuales pesan decenas o cientos de MB, así que aquí se lee una muestra."""),
    ("code", '''anuales = sorted(p for p in DATASETS.glob("simem/simem_14fabb_*.csv") if "nacional" not in p.name)
simem = None
if anuales:
    ultimo = anuales[-1]
    simem = pd.read_csv(ultimo, parse_dates=["timestamp"], nrows=300_000)
    print(f"{ultimo.name}: {len(simem):,} filas leídas (muestra)")
    print("columnas:", list(simem.columns))
    print("versiones presentes:", sorted(simem.Version.unique()))
    print("filas por marca de tiempo:", round(len(simem) / simem.timestamp.nunique(), 1))
else:
    print("Aún no hay CSV anuales de SIMEM.")'''),
    ("code", '''from ingesta.normalizar import simem_agregar_nacional

if simem is not None:
    ingenuo = simem.groupby("timestamp").Valor.sum()
    correcto = simem_agregar_nacional(simem).set_index("fecha_hora").valor_kwh
    factor = (ingenuo.values / correcto.values)
    print("versiones por hora     :", sorted(simem.groupby("timestamp").Version.nunique().unique()))
    print(f"suma ingenua (mediana) : {ingenuo.median()/1e6:,.2f} GWh")
    print(f"suma correcta (mediana): {correcto.median()/1e6:,.2f} GWh")
    print(f"factor de inflación    : {factor.min():.1f}x .. {factor.max():.1f}x")'''),
    ("md", """## Demanda comercial por CIIU

La métrica con dimensiones extra: `Activity` y `Subactivity` vienen dentro de
`Values`, no en el `Id` de la entidad, así que por cada hora hay cientos de
filas legítimas, una por subactividad."""),
    ("code", '''ciiu_archivos = sorted(p for p in DATASETS.glob("xm/ciiu/*.csv*") if p.suffix != ".tmp")
if ciiu_archivos:
    ciiu = pd.read_csv(ciiu_archivos[-1], parse_dates=["timestamp"], nrows=400_000)
    print(f"{ciiu_archivos[-1].name}: {len(ciiu):,} filas leídas (muestra)")
    print("actividades   :", ciiu.Activity.nunique())
    print("subactividades:", ciiu.Subactivity.nunique())
    clave = ["timestamp", "Activity", "Subactivity"]
    print("duplicados sobre la clave completa:", int(ciiu.duplicated(subset=clave).sum()))
    top = ciiu.groupby("Activity").valor.sum().sort_values().tail(8) / 1e6
    top.plot.barh(figsize=(9, 4), title="Consumo por actividad (GWh, muestra)")
    plt.tight_layout(); plt.show()
else:
    print("Aún no hay CSV de CIIU. Ejecuta el script sin --sin-ciiu.")'''),
])


NB02 = cuaderno([
    ("md", """# 02 · Diagnóstico de calidad

Antes de limpiar hay que saber qué está mal. Este notebook **no corrige nada**:
llama a `calidad.diagnostico.diagnosticar()`, que solo describe completitud
temporal, valores y estructura."""),
    ("code", ARRANQUE),
    ("code", CARGA_XM + '''

from calidad.diagnostico import diagnosticar, resumen
informe = diagnosticar(xm)
print(resumen(informe))'''),
    ("md", """## Los huecos, uno a uno

Tres horas sueltas y dos semanas seguidas no admiten el mismo tratamiento, así
que el informe los separa por duración."""),
    ("code", '''huecos = informe["completitud"]["huecos"]
print("tramos:", huecos["n_tramos"], "· forma:", huecos["forma"])
print("por clase:", huecos["tramos_por_clase"])
pd.DataFrame(huecos["tramos"]) if huecos["tramos"] else print("Sin huecos: la serie está completa.")'''),
    ("md", """## Por qué el criterio estacional importa más que el global

Un valor alto a las 2 de la tarde es normal; el mismo a las 3 de la madrugada no
lo es. El IQR global no distingue la hora, así que marca picos normales."""),
    ("code", '''iqr = informe["valores"]["outliers_iqr"]
est = informe["valores"]["outliers_estacionales"]
print(f"IQR global : {iqr['n_outliers']:>5} atípicos ({iqr['pct']}%)  "
      f"límites [{iqr['limite_inferior']:,.0f}, {iqr['limite_superior']:,.0f}]")
print(f"Estacional : {est['n_outliers']:>5} atípicos ({est['pct']}%)  "
      f"fiabilidad {est['fiabilidad']}, {est['observaciones_por_grupo_mediana']} obs/grupo")
pd.DataFrame(est["ejemplos"])'''),
    ("code", '''por_hora = pd.Series({int(h): n for h, n in est["por_hora_del_dia"].items()}).sort_index()
if len(por_hora):
    por_hora.plot.bar(figsize=(11, 3.5), title="Atípicos estacionales por hora del día")
    plt.xlabel("hora"); plt.tight_layout(); plt.show()'''),
    ("md", """## Los últimos días publicados

Los dos últimos días publicados suelen llegar con valores parciales aunque
tengan las 24 horas y ningún nulo. Ninguna comprobación de completitud lo ve:
el defecto está en los valores, no en la rejilla de tiempo."""),
    ("code", '''diario = xm.set_index("fecha_hora").valor_kwh.resample("D").sum() / 1e6
tipico = diario.iloc[:-30].median()

ax = diario.tail(30).plot(figsize=(12, 4), marker="o", title="Total diario, últimos 30 días (GWh)")
ax.axhline(tipico, ls="--", c="gray", label=f"día típico ({tipico:.0f} GWh)")
ax.axhline(tipico * .5, ls=":", c="crimson", label="50 % del típico")
ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); plt.show()

parciales = diario[diario < tipico * .5]
print(f"días por debajo del 50 % del típico: {len(parciales)}")
print(parciales.round(1).to_string())'''),
    ("md", """> **Para el modelado:** el rezago útil no es de 3 días sino de ~5. Conviene
> descartar los últimos 5 días de cualquier partición de entrenamiento o validación."""),
])


NB03 = cuaderno([
    ("md", """# 03 · Limpieza · demanda real del SIN (XM)

Toda transformación queda registrada: `limpieza.limpiar.limpiar()` devuelve el
DataFrame y un registro de qué hizo, con qué criterio y a cuántas filas afectó.

- **Se prefiere un NaN honesto a un valor inventado.** Solo se interpolan huecos
  de hasta 3 horas; los mayores se quedan vacíos y marcados.
- **Nada se elimina.** Atípicos y periodos estructurales se *marcan*; excluirlos
  es decisión del modelado."""),
    ("code", ARRANQUE),
    ("code", CARGA_XM + '''

from limpieza.limpiar import limpiar, procedencia, resumen as resumen_limpieza
limpio, registro = limpiar(xm)
print(resumen_limpieza(registro))'''),
    ("md", "## Qué hizo cada paso\n\nEl registro responde a «¿qué le pasó a estos datos?»."),
    ("code", '''pd.DataFrame([
    {"operación": o["operacion"], "filas afectadas": o["filas_afectadas"], "criterio": o["criterio"]}
    for o in registro["operaciones"]
])'''),
    ("md", "## De dónde salió cada valor\n\nLa pregunta que la limpieza tiene que poder responder para cualquier celda."),
    ("code", '''print(limpio.origen_valor.value_counts().to_string())
print()
marcado = limpio[limpio.atipico].fecha_hora
muestras = [limpio.fecha_hora.iloc[0]] + ([marcado.iloc[-1]] if len(marcado) else [])
for momento in muestras:
    d = procedencia(limpio, momento)
    valor = f"{d['valor']:,.2f}" if d["valor"] is not None else "None"
    print(d["momento"])
    print(f"   valor {valor} · origen {d['origen_valor']} · atípico {d['atipico']} (z={d['z_estacional']})")
    print(f"   {d['explicacion']}")
    print()'''),
    ("md", """## ⚠️ Qué columnas NO usar como variables

`atipico_global` y `atipico_iqr_global` se calculan mirando **la serie entera**,
futuro incluido. Meterlas en un modelo es fuga temporal: el pasado quedaría
etiquetado con conocimiento del futuro y la validación saldría optimista.

Para eso están `atipico` y `atipico_iqr`, **causales**: cada fila se juzga solo
contra las observaciones anteriores de su grupo (día de semana, hora)."""),
    ("code", '''comparacion = pd.DataFrame({
    "causal (usar)":    [int(limpio.atipico.sum()), int(limpio.atipico_iqr.sum())],
    "global (no usar)": [int(limpio.atipico_global.sum()), int(limpio.atipico_iqr_global.sum())],
}, index=["estacional", "IQR"])
print(comparacion.to_string())
print()
print(f"filas aún no evaluables (grupo sin histórico): {int((~limpio.atipico_evaluable).sum()):,}")
print("-> son el arranque de la serie; tenlo en cuenta al partir train/test.")'''),
    ("md", "## La serie limpia"),
    ("code", '''# Para graficar se pasa a hora local sin zona: matplotlib no acepta mezclar
# una linea de pandas con un scatter de fechas tz-aware en el mismo eje.
local = limpio.assign(hora_local=limpio.fecha_hora.dt.tz_localize(None))
marcados = local[local.atipico]

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(local.hora_local, local.valor_kwh / 1e6, lw=.4, label="demanda")
ax.scatter(marcados.hora_local, marcados.valor_kwh / 1e6, c="crimson", s=8, zorder=5,
           label=f"atípicos causales ({len(marcados)})")
ax.set(title="Demanda real del SIN (GWh)", ylabel="GWh")
ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); plt.show()'''),
    ("md", "## Guardar\n\nLa serie limpia y su registro van a `datasets/limpios/`."),
    ("code", '''import json
SALIDA = DATASETS / "limpios"
SALIDA.mkdir(parents=True, exist_ok=True)
limpio.to_parquet(SALIDA / "demanda_real.parquet", index=False)
with open(SALIDA / "registro_demanda_real.json", "w", encoding="utf-8") as f:
    json.dump(registro, f, ensure_ascii=False, indent=2, default=str)
print(f"{len(limpio):,} filas -> {SALIDA / 'demanda_real.parquet'}")'''),
    ("md", """## Siguiente paso

El panel de modelado —demanda limpia + clima + calendario, validando que el merge
no altera filas— se construye con `python scripts/construir_panel.py`."""),
])


NB04 = cuaderno([
    ("md", """# 04 · Limpieza · catálogo de métricas (XM)

El catálogo es la lista de las 193 métricas que publica la API de XM. No es una
serie temporal, así que no tiene huecos ni atípicos: sus problemas son de texto
y de coherencia. Todos se encontraron mirando el archivo real:

- el mismo centinela escrito de dos formas: `"No aplica"` y `"No Aplica"`;
- la misma entidad con dos grafías: `"SubArea"` y `"Subarea"`;
- 7 unidades vacías, 13 descripciones con espacios sobrantes y 5 nombres con doble espacio;
- **una URL equivocada**: el catálogo anuncia `/list` para las métricas de listado,
  pero ese endpoint responde 404; el que funciona es `/lists`.

La limpieza la hace `limpieza.catalogo.limpiar_catalogo()`: normaliza lo que se
puede normalizar, corrige solo lo que está verificado y marca lo demás."""),
    ("code", ARRANQUE),
    ("code", '''# keep_default_na=False: sin esto pandas convierte las cadenas vacías en NaN al
# leer y el problema de los vacíos quedaría escondido antes de verlo.
cat = pd.read_csv(DATASETS / "xm" / "xm_catalogo_metricas.csv", keep_default_na=False)
print(f"{len(cat)} filas · {len(cat.columns)} columnas")
cat.head()'''),
    ("md", "## Qué está mal, antes de tocar nada"),
    ("code", '''texto = cat.select_dtypes("object")
print("duplicados (MetricId, Entity):", int(cat.duplicated(["MetricId", "Entity"]).sum()))
print()
print("centinela de 'sin filtro' escrito de varias formas:")
print(cat.Filter[cat.Filter.str.casefold() == "no aplica"].value_counts().to_string())
print()
grafias = cat.Entity.groupby(cat.Entity.str.casefold()).unique()
print("entidades que solo difieren en mayúsculas:", grafias[grafias.str.len() > 1].tolist())
print()
vacios = {c: int((texto[c].str.strip() == "").sum()) for c in texto}
print("celdas vacías   :", {c: n for c, n in vacios.items() if n})
sobrantes = {c: int((texto[c] != texto[c].str.strip()).sum()) for c in texto}
print("espacios de más :", {c: n for c, n in sobrantes.items() if n})
dobles = cat[cat.MetricName.str.contains("  ", regex=False)].MetricName.tolist()
print("dobles espacios :", dobles)'''),
    ("code", '''print("URL anunciada por tipo de métrica:")
pd.crosstab(cat.Type, cat.Url)'''),
    ("md", """> **Sobre `/list`.** Comprobado contra el servidor el 2026-09-10: `POST /list`
> responde **404**, `POST /lists` responde 200 con las 193 métricas. Este notebook
> no llama a la API —se ejecuta igual sin red—, así que la evidencia queda escrita
> en el registro de la limpieza."""),
    ("md", "## Limpieza"),
    ("code", '''from limpieza.catalogo import limpiar_catalogo, resumen
limpio, registro = limpiar_catalogo(cat)
print(resumen(registro))'''),
    ("md", """## Qué cambió en las filas tocadas

`entity` se conserva **tal cual**: es el literal que se envía a la API en el
campo `Entity`, y no está verificado que la API acepte las dos grafías. Para
agrupar y contar está `entidad_normalizada`."""),
    ("code", '''tocadas = limpio[
    limpio.url_corregida | (limpio.entity != limpio.entidad_normalizada)
]
tocadas[["metric_id", "entity", "entidad_normalizada", "granularidad", "url", "url_corregida"]]'''),
    ("code", '''print("filas con filtro real:", int(limpio.tiene_filtro.sum()), "de", len(limpio))
print(limpio["filter"].value_counts().to_string())
print()
print("unidades (tras convertir los vacíos a nulo):")
print(limpio.metric_units.value_counts(dropna=False).to_string())'''),
    ("md", "## Comprobaciones"),
    ("code", '''assert len(limpio) == len(cat), "se perdieron filas"
assert not limpio.duplicated(["metric_id", "entity"]).any()
assert limpio.url_coherente.all(), "hay URLs que no apuntan a su granularidad"
assert limpio.granularidad.notna().all()
assert (limpio.max_days > 0).all()
print("OK: 193 métricas, sin duplicados, todas las URL coherentes con su granularidad")'''),
    ("md", "## Guardar\n\nEs una tabla de referencia pequeña, así que va en CSV para poder abrirla en cualquier parte."),
    ("code", '''import json
SALIDA = DATASETS / "limpios"
SALIDA.mkdir(parents=True, exist_ok=True)
limpio.to_csv(SALIDA / "catalogo_metricas.csv", index=False, encoding="utf-8")
with open(SALIDA / "registro_catalogo_metricas.json", "w", encoding="utf-8") as f:
    json.dump(registro, f, ensure_ascii=False, indent=2, default=str)
print(f"{len(limpio)} filas -> {SALIDA / 'catalogo_metricas.csv'}")'''),
])


NB05 = cuaderno([
    ("md", """# 05 · Limpieza · demanda comercial del SIN (XM)

`DemaCome` es la demanda **comercial**: la que se liquida en el mercado, que
incluye las pérdidas de la red. Por eso debería quedar siempre algo por encima
de la demanda real (`DemaReal`).

Lo que se encontró al diagnosticarla:

- serie completa, sin huecos ni duplicados;
- ~156 atípicos estacionales, casi todos en los dos últimos días publicados,
  que llegan parciales (el mismo defecto que la demanda real);
- **4 horas en las que la comercial queda por debajo de la real**, lo que no
  cuadra entre dos series de la misma fuente.

Se limpia con el mismo `limpiar()` que la demanda real, y además se marca la
incoherencia con la real. Marcar, no corregir: no hay forma de saber cuál de
las dos está mal."""),
    ("code", ARRANQUE),
    ("code", '''from ingesta.normalizar import xm_cliente_a_esquema_comun
com = xm_cliente_a_esquema_comun(pd.read_csv(DATASETS / "xm" / "xm_demacome_sistema.csv", parse_dates=["timestamp"]))
real = xm_cliente_a_esquema_comun(pd.read_csv(DATASETS / "xm" / "xm_demareal_sistema.csv", parse_dates=["timestamp"]))
print(f"comercial: {len(com):,} filas · {com.fecha_hora.min()} .. {com.fecha_hora.max()}")
print(f"real     : {len(real):,} filas")'''),
    ("md", "## Diagnóstico"),
    ("code", '''from calidad.diagnostico import diagnosticar, resumen
informe = diagnosticar(com)
print(resumen(informe))'''),
    ("md", "## Limpieza"),
    ("code", '''from limpieza.limpiar import limpiar, procedencia, resumen as resumen_limpieza
limpio, registro = limpiar(com)
print(resumen_limpieza(registro))'''),
    ("md", """## Coherencia con la demanda real

La comercial incluye pérdidas, así que el cociente comercial / real debería
estar siempre por encima de 1. Las horas en que no lo está se marcan en
`menor_que_real`."""),
    ("code", '''par = limpio.merge(
    real[["fecha_hora", "valor_kwh"]].rename(columns={"valor_kwh": "real_kwh"}),
    on="fecha_hora", how="left", validate="one_to_one",
)
assert len(par) == len(limpio), "el merge cambió el número de filas"

cociente = par.valor_kwh / par.real_kwh
limpio["menor_que_real"] = (par.valor_kwh < par.real_kwh).to_numpy()

registro["operaciones"].append({
    "operacion": "marcar_incoherencia_con_real",
    "criterio": "menor_que_real = demanda comercial < demanda real en la misma hora; marcado, sin corregir",
    "filas_afectadas": int(limpio.menor_que_real.sum()),
    "detalle": {
        "cociente_mediana": round(float(cociente.median()), 4),
        "cociente_p1": round(float(cociente.quantile(.01)), 4),
        "cociente_p99": round(float(cociente.quantile(.99)), 4),
    },
})

print("cociente comercial / real:")
print(cociente.describe(percentiles=[.01, .5, .99]).round(4).to_string())
print()
print("horas marcadas como menor_que_real:", int(limpio.menor_que_real.sum()))
par.assign(cociente=cociente.round(4))[limpio.menor_que_real.to_numpy()][["fecha_hora", "valor_kwh", "real_kwh", "cociente"]]'''),
    ("md", """## Los últimos días publicados

Igual que en la demanda real, los dos últimos días llegan parciales. No hace
falta una marca nueva: el criterio causal de atípicos ya los señala."""),
    ("code", '''diario = limpio.set_index("fecha_hora").resample("D").agg(
    gwh=("valor_kwh", lambda s: s.sum() / 1e6), atipicos=("atipico", "sum")
)
diario.tail(7).round(1)'''),
    ("md", "## Guardar"),
    ("code", '''import json
SALIDA = DATASETS / "limpios"
SALIDA.mkdir(parents=True, exist_ok=True)
limpio.to_parquet(SALIDA / "demanda_comercial.parquet", index=False)
with open(SALIDA / "registro_demanda_comercial.json", "w", encoding="utf-8") as f:
    json.dump(registro, f, ensure_ascii=False, indent=2, default=str)
print(f"{len(limpio):,} filas -> {SALIDA / 'demanda_comercial.parquet'}")'''),
])


NB06 = cuaderno([
    ("md", """# 06 · Limpieza · demanda comercial por CIIU (XM)

La demanda comercial no regulada, desagregada por actividad económica (CIIU):
**17,5 millones de filas**, una por hora y por subactividad, con 367
combinaciones `(Activity, Subactivity)` a lo largo del histórico y entre 349 y
355 activas en cada hora.

Lo que se encontró al mirarla:

- **90 302 valores vacíos** dentro de filas que sí existen;
- sectores que aparecen y desaparecen: alguno vive solo unas semanas;
- ni ceros ni negativos.

**Por qué por grupos.** `limpiar()` está pensada para una serie con una fila por
hora, y se niega a tratar una tabla así: deduplicar por hora borraría el 99 % de
los datos. Hay que partirla en 367 series y limpiar cada una por separado, para
que los estadísticos de un sector no contaminen a otro —10 MWh es normal en la
industria y disparatado en una biblioteca—. Lo hace
`limpieza.grupos.limpiar_por_grupos()`.

La rejilla horaria de cada sector va de **su** primera a **su** última hora: a
un sector que solo existió unos meses no se le inventan horas antes ni después."""),
    ("code", ARRANQUE + '''

import json, time
import pyarrow as pa
import pyarrow.parquet as pq'''),
    ("md", """## Carga con memoria acotada

Leído tal cual, el CSV ocupa más de 1,6 GB en memoria por las descripciones de
actividad repetidas en cada fila. Se lee por trozos, con `Activity` y
`Subactivity` como categorías y la fecha ya convertida."""),
    ("code", '''from pandas.api.types import union_categoricals

t0 = time.time()
partes = []
for ruta in sorted(p for p in DATASETS.glob("xm/ciiu/*.csv.gz")):
    for trozo in pd.read_csv(ruta, usecols=["timestamp", "valor", "Activity", "Subactivity"], chunksize=2_000_000):
        trozo["timestamp"] = pd.to_datetime(trozo["timestamp"], format="ISO8601")
        trozo["Activity"] = trozo["Activity"].astype("category")
        trozo["Subactivity"] = trozo["Subactivity"].astype("category")
        partes.append(trozo)

ciiu = pd.DataFrame({
    "timestamp": pd.concat([p.timestamp for p in partes], ignore_index=True),
    "valor": pd.concat([p.valor for p in partes], ignore_index=True),
    "Activity": union_categoricals([p.Activity for p in partes]),
    "Subactivity": union_categoricals([p.Subactivity for p in partes]),
})
del partes
print(f"{len(ciiu):,} filas en {time.time() - t0:.0f} s · {ciiu.memory_usage(deep=True).sum() / 1e6:.0f} MB en memoria")'''),
    ("md", "## Diagnóstico"),
    ("code", '''g = ciiu.groupby(["Activity", "Subactivity"], observed=True)
filas = g.size()
vida = g.timestamp.agg(["min", "max"])
esperadas = ((vida["max"] - vida["min"]) / pd.Timedelta(hours=1) + 1).astype(int)
nulos = g.valor.apply(lambda s: int(s.isna().sum()))

print(f"actividades                   : {ciiu.Activity.nunique()}")
print(f"grupos (actividad, subactividad): {len(filas)}")
print(f"filas por grupo               : min {filas.min():,} · mediana {int(filas.median()):,} · máx {filas.max():,}")
print(f"grupos con horas sin fila     : {int((esperadas > filas).sum())} ({int((esperadas - filas).sum()):,} horas)")
print(f"valores vacíos                : {int(ciiu.valor.isna().sum()):,} en {int((nulos > 0).sum())} grupos")
print(f"ceros / negativos             : {int((ciiu.valor == 0).sum())} / {int((ciiu.valor < 0).sum())}")
por_hora = ciiu.groupby("timestamp").size()
print(f"subactividades por hora       : {por_hora.min()} .. {por_hora.max()}")'''),
    ("code", '''print("Los sectores de vida más corta:")
vida.assign(filas=filas).sort_values("filas").head(6)'''),
    ("md", """## Limpieza por grupos

Cada sector se limpia como una serie propia y se escribe en cuanto está listo,
sin juntar el resultado en memoria. El Parquet final se escribe primero como
`.tmp` y se renombra al acabar, para que una ejecución interrumpida nunca deje
un archivo que parezca completo.

Tarda unos minutos: son 367 series de hasta 49 824 horas."""),
    ("code", '''from limpieza.grupos import limpiar_por_grupos, fila_de_resumen, resumir

SALIDA = DATASETS / "limpios"
SALIDA.mkdir(parents=True, exist_ok=True)
destino = SALIDA / "ciiu_limpio.parquet"
temporal = destino.with_name(destino.name + ".tmp")

ESQUEMA = pa.schema([
    ("timestamp", pa.timestamp("ns", tz="America/Bogota")),
    ("activity", pa.string()), ("subactivity", pa.string()),
    ("valor", pa.float64()), ("origen_valor", pa.string()),
    ("imputado", pa.bool_()), ("hueco_horas", pa.int64()),
    ("z_estacional", pa.float64()), ("atipico", pa.bool_()),
    ("atipico_evaluable", pa.bool_()), ("atipico_iqr", pa.bool_()),
    ("atipico_global", pa.bool_()), ("atipico_iqr_global", pa.bool_()),
    ("periodo_atipico", pa.bool_()),
])

filas_resumen = []
t0 = time.time()
with pq.ParquetWriter(temporal, ESQUEMA, compression="zstd") as escritor:
    for n, (etiqueta, limpio, registro) in enumerate(limpiar_por_grupos(ciiu, ["Activity", "Subactivity"]), 1):
        assert not limpio.timestamp.duplicated().any(), f"horas repetidas en {etiqueta}"
        tabla = limpio[ESQUEMA.names].astype({"hueco_horas": "int64"})
        escritor.write_table(pa.Table.from_pandas(tabla, schema=ESQUEMA, preserve_index=False))
        filas_resumen.append(fila_de_resumen(etiqueta, limpio, registro))
        if n % 60 == 0:
            print(f"  {n:>3} grupos · {time.time() - t0:.0f} s")
temporal.replace(destino)

total = resumir(filas_resumen)
print(f"{total['n_grupos']} grupos limpiados en {time.time() - t0:.0f} s")
total'''),
    ("md", "## Resultado"),
    ("code", '''grupos = pd.DataFrame(filas_resumen)
print(f"observados  : {total['observados']:>12,}  ({total['pct_observado']} %)")
print(f"interpolados: {total['interpolados']:>12,}  en {total['grupos_con_interpolados']} grupos (huecos de hasta 3 h)")
print(f"faltantes   : {total['faltantes']:>12,}  en {total['grupos_con_faltantes']} grupos (huecos largos, se quedan NaN)")
print(f"atípicos    : {total['atipicos']:>12,}  (causales, dentro de cada sector)")
print(f"horas creadas para completar la rejilla de cada sector: {total['horas_creadas']:,}")
print()
print("Sectores con más horas sin valor:")
grupos.sort_values("faltantes", ascending=False).head(8)[
    ["Activity", "Subactivity", "desde", "hasta", "faltantes", "tramos_no_imputados", "interpolados"]
]'''),
    ("md", "## Comprobaciones"),
    ("code", '''meta = pq.ParquetFile(destino).metadata
assert meta.num_rows == total["filas_finales"], "el Parquet no tiene las filas esperadas"
assert total["filas_iniciales"] == len(ciiu), "no se procesaron todas las filas de entrada"
assert total["observados"] + total["interpolados"] + total["faltantes"] == total["filas_finales"]
assert total["filas_finales"] - total["filas_iniciales"] == total["horas_creadas"]
print(f"OK: {len(ciiu):,} filas de entrada -> {meta.num_rows:,} en el Parquet")
print(f"    (+{total['horas_creadas']:,} horas que faltaban en la rejilla de su sector, marcadas)")'''),
    ("md", "## Registro"),
    ("code", '''registro_ciiu = {
    "version_formato": 1,
    "conjunto": "ciiu_demacomenoreg",
    "momento": pd.Timestamp.now().isoformat(timespec="seconds"),
    "archivo": destino.name,
    "politica": ("limpieza por (Activity, Subactivity); rejilla de cada sector entre su primera "
                 "y su ultima hora; interpolacion solo en huecos de hasta 3 h; atipicos causales "
                 "dentro de cada sector; nada se elimina"),
    "resumen": total,
    "grupos": filas_resumen,
}
with open(SALIDA / "registro_ciiu.json", "w", encoding="utf-8") as f:
    json.dump(registro_ciiu, f, ensure_ascii=False, indent=2, default=str)
print("registro ->", SALIDA / "registro_ciiu.json")'''),
    ("md", """## Cómo leerlo sin cargar las 17,5 M de filas

El Parquet se escribió sector a sector, así que se puede leer solo una parte."""),
    ("code", '''educacion = pq.read_table(
    destino,
    columns=["timestamp", "subactivity", "valor", "origen_valor", "atipico"],
    filters=[("activity", "==", "EDUCACIÓN")],
).to_pandas()
print(f"EDUCACIÓN: {len(educacion):,} filas · {educacion.subactivity.nunique()} subactividades")
educacion.head()'''),
])


NOTEBOOKS = {
    "00_configuracion": NB00,
    "01_carga_y_exploracion": NB01,
    "02_diagnostico_calidad": NB02,
    "03_limpieza_demanda_real": NB03,
    "04_limpieza_catalogo_metricas": NB04,
    "05_limpieza_demanda_comercial": NB05,
    "06_limpieza_ciiu": NB06,
}


def main(argv: list[str] | None = None) -> None:
    import argparse

    analizador = argparse.ArgumentParser(description="Genera los notebooks")
    analizador.add_argument(
        "nombres", nargs="*",
        help="solo estos notebooks (por defecto todos); regenerar borra sus salidas",
    )
    args = analizador.parse_args(argv)

    desconocidos = set(args.nombres) - set(NOTEBOOKS)
    if desconocidos:
        raise SystemExit(f"Notebooks desconocidos: {sorted(desconocidos)}. Validos: {list(NOTEBOOKS)}")

    DIR.mkdir(exist_ok=True)
    for nombre in args.nombres or NOTEBOOKS:
        ruta = DIR / f"{nombre}.ipynb"
        nbf.write(NOTEBOOKS[nombre], ruta)
        print("escrito", ruta.name)


if __name__ == "__main__":
    main()
