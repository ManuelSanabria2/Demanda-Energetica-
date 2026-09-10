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
    ("md", """# 03 · Limpieza

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
    ("md", """## Siguiente paso

El panel de modelado —demanda limpia + clima + calendario, validando que el merge
no altera filas— se construye con `python scripts/construir_panel.py`."""),
])


def main() -> None:
    DIR.mkdir(exist_ok=True)
    for nombre, libro in [
        ("00_configuracion", NB00),
        ("01_carga_y_exploracion", NB01),
        ("02_diagnostico_calidad", NB02),
        ("03_limpieza", NB03),
    ]:
        ruta = DIR / f"{nombre}.ipynb"
        nbf.write(libro, ruta)
        print("escrito", ruta.name)


if __name__ == "__main__":
    main()
