"""Mineria no supervisada: tipos de dia (K-means), reglas de asociacion y sectores CIIU.

    python scripts/mineria_no_supervisada.py
    python scripts/mineria_no_supervisada.py --k 4            # fijar k a mano
    python scripts/mineria_no_supervisada.py --sin-sectores   # sin el CIIU (mas rapido)

Entradas:
    data/procesado/panel_modelado.parquet       demanda + calendario + clima
    datasets/limpios/ciiu_limpio.parquet        opcional: demanda por CIIU limpia

Salidas, versionadas porque son la evidencia de la entrega:
    docs/figuras/mineria/*.png
    docs/resultados/mineria.json                k elegido, metricas, grupos, reglas
    docs/resultados/mineria_*.csv               tablas completas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from mineria.clustering import (  # noqa: E402
    agrupar,
    contingencia,
    describir,
    dias_atipicos,
    elegir_k,
    evaluar_k,
    v_cramer,
)
from mineria.estilo import (  # noqa: E402
    CATEGORICA,
    SECUENCIAL,
    TINTA_SECUNDARIA,
    aplicar,
    color_grupo,
    guardar,
    marcador_grupo,
    plt,
)
from mineria.perfiles import atributos_dia, forma_y_nivel, perfiles_diarios  # noqa: E402
from mineria.reglas import podar_redundantes, reglas_asociacion, reglas_hacia, transacciones  # noqa: E402

PROCESADO = RAIZ / "data" / "procesado"
CIIU = RAIZ / "datasets" / "limpios" / "ciiu_limpio.parquet"
TIPOS = ["laboral", "sabado", "domingo", "festivo"]
TIPOS_EN = {"laboral": "Weekday", "sabado": "Saturday", "domingo": "Sunday", "festivo": "Holiday"}


def _relativa(ruta: Path) -> str:
    try:
        return ruta.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return ruta.as_posix()


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------


def fig_seleccion_k(evaluacion: pd.DataFrame, k: int, destino: Path) -> Path:
    figura, ejes = plt.subplots(1, 4, figsize=(7.4, 2.0))
    for eje, columna, titulo in zip(
        ejes,
        ("inercia", "silueta", "davies_bouldin", "estabilidad_ari"),
        ("Inertia (elbow)", "Silhouette ↑", "Davies-Bouldin ↓", "Stability (ARI) ↑"),
    ):
        eje.plot(evaluacion["k"], evaluacion[columna], marker="o", markersize=4, color=CATEGORICA[0])
        fila = evaluacion[evaluacion["k"] == k].iloc[0]
        eje.scatter([k], [fila[columna]], s=60, facecolor="none", edgecolor=CATEGORICA[1], linewidth=1.6, zorder=3)
        eje.set_title(titulo, fontsize=8.5)
        eje.set_xticks(evaluacion["k"])
        eje.tick_params(labelsize=7)
    ejes[0].set_ylabel("value")
    figura.suptitle(f"Choice of k for K-means on daily load shapes (chosen k = {k})", fontsize=9.5, fontweight="bold", y=1.06)
    figura.tight_layout()
    return guardar(figura, destino / "01_seleccion_k.png")


def fig_centroides(resultado, forma: pd.DataFrame, destino: Path) -> Path:
    figura, eje = plt.subplots(figsize=(7.2, 2.8))
    horas = np.arange(24)
    for grupo, centroide in resultado.centroides.iterrows():
        miembros = forma[resultado.etiquetas == grupo]
        bajo, alto = miembros.quantile(0.1).values, miembros.quantile(0.9).values
        color = color_grupo(grupo)
        eje.fill_between(horas, bajo, alto, color=color, alpha=0.12, linewidth=0)
        n = int((resultado.etiquetas == grupo).sum())
        eje.plot(horas, centroide.values, color=color, marker=marcador_grupo(grupo), markersize=3.5, label=f"{grupo} (n={n})")
        eje.annotate(grupo, (23, centroide.values[-1]), xytext=(4, 0), textcoords="offset points",
                     fontsize=7.5, color=TINTA_SECUNDARIA, va="center")
    eje.set_xticks(range(0, 24, 2))
    eje.set_xlabel("Hour of day")
    eje.set_ylabel("Demand / daily mean")
    eje.set_title("Cluster centroids of daily load shape (band: 10th-90th percentile)")
    eje.legend(ncol=min(4, resultado.k), fontsize=7.5, loc="upper left")
    return guardar(figura, destino / "02_centroides.png")


def fig_pca(resultado, destino: Path) -> Path:
    figura, eje = plt.subplots(figsize=(4.2, 3.2))
    for grupo in resultado.centroides.index:
        puntos = resultado.pca[resultado.etiquetas == grupo]
        eje.scatter(puntos["pc1"], puntos["pc2"], s=7, alpha=0.65, color=color_grupo(grupo),
                    marker=marcador_grupo(grupo), label=grupo, linewidths=0)
    v = resultado.varianza_pca
    eje.set_xlabel(f"PC1 ({100 * v[0]:.1f} % var.)")
    eje.set_ylabel(f"PC2 ({100 * v[1]:.1f} % var.)")
    eje.set_title("Days in PCA space, coloured by cluster")
    eje.legend(fontsize=7, markerscale=2)
    return guardar(figura, destino / "03_pca.png")


def fig_contingencia(tabla: pd.DataFrame, cramer: float, destino: Path) -> Path:
    columnas = [t for t in TIPOS if t in tabla.columns]
    tabla = tabla[columnas]
    proporcion = tabla.div(tabla.sum(axis=0), axis=1)          # que % de cada tipo cae en cada grupo
    figura, eje = plt.subplots(figsize=(4.4, 0.55 * len(tabla) + 1.0))
    imagen = eje.imshow(proporcion.values, cmap=SECUENCIAL, vmin=0, vmax=1, aspect="auto")
    eje.set_xticks(range(len(columnas)), [TIPOS_EN[c] for c in columnas])
    eje.set_yticks(range(len(tabla)), tabla.index)
    eje.grid(False)
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            eje.text(j, i, f"{tabla.values[i, j]}\n{100 * proporcion.values[i, j]:.0f} %", ha="center", va="center",
                     fontsize=6.5, color="white" if proporcion.values[i, j] > 0.6 else "#0b0b0b")
    eje.set_title(f"Day type vs cluster (Cramér's V = {cramer:.2f})", fontsize=9)
    figura.colorbar(imagen, ax=eje, fraction=0.05, pad=0.02, label="share of day type")
    return guardar(figura, destino / "04_contingencia_tipo_dia.png")


def fig_calendario(resultado, destino: Path) -> Path:
    etiquetas = resultado.etiquetas.copy()
    fechas = pd.to_datetime(pd.Series(etiquetas.index))
    anios = sorted(fechas.dt.year.unique())
    grupos = list(resultado.centroides.index)
    codigo = {g: i for i, g in enumerate(grupos)}
    matriz = np.full((len(anios), 366), np.nan)
    for fecha, grupo in zip(fechas, etiquetas.values):
        matriz[anios.index(fecha.year), fecha.dayofyear - 1] = codigo[grupo]
    from matplotlib.colors import ListedColormap

    figura, eje = plt.subplots(figsize=(7.2, 0.32 * len(anios) + 0.9))
    eje.imshow(matriz, aspect="auto", interpolation="nearest", cmap=ListedColormap([color_grupo(g) for g in grupos]),
               vmin=-0.5, vmax=len(grupos) - 0.5)
    eje.set_yticks(range(len(anios)), anios)
    meses = pd.date_range("2024-01-01", periods=12, freq="MS")
    eje.set_xticks(meses.dayofyear - 1, meses.strftime("%b"))
    eje.grid(False)
    from matplotlib.patches import Patch

    eje.legend(handles=[Patch(color=color_grupo(g), label=g) for g in grupos], ncol=len(grupos), fontsize=7,
               loc="upper center", bbox_to_anchor=(0.5, -0.25))
    eje.set_title("Cluster of each day across the calendar")
    return guardar(figura, destino / "05_calendario_grupos.png")


def fig_sectores(perfiles: pd.DataFrame, etiquetas: pd.Series, rasgos: pd.DataFrame, energia: pd.Series, destino: Path) -> list[Path]:
    rutas = []
    figura, eje = plt.subplots(figsize=(7.2, 2.6))
    for grupo in sorted(etiquetas.unique()):
        miembros = perfiles[etiquetas.values == grupo]
        eje.plot(range(168), miembros.median().values, color=color_grupo(grupo), marker=None,
                 label=f"{grupo} (n={len(miembros)})", linewidth=1.4)
    for d in range(1, 7):
        eje.axvline(24 * d, color="#c3c2b7", linewidth=0.6)
    eje.set_xticks([12 + 24 * d for d in range(7)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    eje.set_ylabel("Demand / sector mean")
    eje.set_title("Median weekly profile of each CIIU sector cluster (non-regulated market)")
    eje.legend(fontsize=7.5, ncol=4)
    rutas.append(guardar(figura, destino / "06_sectores_perfil_semanal.png"))

    figura, eje = plt.subplots(figsize=(4.6, 3.4))
    tamanos = 15 + 120 * np.sqrt(energia / energia.max()).to_numpy()
    for grupo in sorted(etiquetas.unique()):
        m = etiquetas.values == grupo
        eje.scatter(rasgos["noche_sobre_dia"].to_numpy()[m], rasgos["domingo_sobre_laboral"].to_numpy()[m], s=tamanos[m],
                    color=color_grupo(grupo), marker=marcador_grupo(grupo), alpha=0.6, edgecolor="white", linewidth=0.5,
                    label=grupo)
    eje.axhline(1, color="#c3c2b7", linewidth=0.8)
    eje.axvline(1, color="#c3c2b7", linewidth=0.8)
    eje.set_xlabel("Night (00-05 h) / day (09-17 h), weekdays")
    eje.set_ylabel("Sunday / weekday")
    eje.set_title("CIIU sub-activities: continuous vs\nworking-hours consumption (size = energy)", fontsize=9)
    eje.legend(fontsize=7, markerscale=0.6)
    rutas.append(guardar(figura, destino / "07_sectores_rasgos.png"))
    return rutas


# --------------------------------------------------------------------------
# Analisis
# --------------------------------------------------------------------------


def analisis_dias(panel: pd.DataFrame, k_fijo: int | None, ks: range, figuras: Path, resultados: Path) -> dict:
    perfiles, registro_perfiles = perfiles_diarios(panel)
    forma, nivel = forma_y_nivel(perfiles)
    atributos = atributos_dia(panel, perfiles.index)

    evaluacion = evaluar_k(forma.to_numpy(), ks=ks)
    k = k_fijo or elegir_k(evaluacion)
    resultado = agrupar(forma, k)
    tabla = describir(resultado, forma, nivel, atributos)
    cont = contingencia(resultado, atributos)
    cramer = v_cramer(cont)
    lejos = dias_atipicos(resultado, atributos, cuantil=0.99)

    canasta = transacciones(atributos, resultado.etiquetas, nivel)
    reglas = reglas_asociacion(canasta)
    no_redundantes = podar_redundantes(reglas)
    hacia_grupo = reglas_hacia(no_redundantes, "grupo=")

    rutas = [
        fig_seleccion_k(evaluacion, k, figuras),
        fig_centroides(resultado, forma, figuras),
        fig_pca(resultado, figuras),
        fig_contingencia(cont, cramer, figuras),
        fig_calendario(resultado, figuras),
    ]

    resultados.mkdir(parents=True, exist_ok=True)
    evaluacion.to_csv(resultados / "mineria_seleccion_k.csv", index=False)
    tabla.to_csv(resultados / "mineria_grupos.csv")
    reglas.to_csv(resultados / "mineria_reglas.csv", index=False)
    lejos.to_csv(resultados / "mineria_dias_atipicos.csv")
    pd.concat([resultado.etiquetas, resultado.distancia, nivel], axis=1).to_csv(resultados / "mineria_etiquetas_dia.csv")

    return {
        "perfiles": registro_perfiles,
        "seleccion_k": json.loads(evaluacion.round(4).to_json(orient="records")),
        "k": k,
        "k_criterio": "fijado a mano" if k_fijo else "mayor silueta entre k con estabilidad ARI >= 0.80 y grupo minimo >= 10 dias",
        "kmeans": resultado.registro,
        "grupos": json.loads(tabla.to_json(orient="index")),
        "contingencia_tipo_dia": json.loads(cont.to_json(orient="index")),
        "v_cramer_tipo_dia": cramer,
        "dias_atipicos_top": json.loads(lejos.head(15).reset_index().astype(str).to_json(orient="records")),
        "reglas": {
            "parametros": {"soporte_minimo": 0.02, "confianza_minima": 0.60, "lift_minimo": 1.20, "max_items": 3},
            "transacciones": int(len(canasta)),
            "items": list(canasta.columns),
            "n_reglas": int(len(reglas)),
            "n_reglas_no_redundantes": int(len(no_redundantes)),
            "top_hacia_grupo": json.loads(hacia_grupo.head(15).to_json(orient="records")),
            "top_general": json.loads(no_redundantes.head(15).to_json(orient="records")),
        },
        "figuras": [_relativa(r) for r in rutas],
    }


def analisis_sectores(ruta: Path, ks: range, figuras: Path, resultados: Path) -> dict:
    import pyarrow.parquet as pq

    from mineria.clustering import elegir_k as elegir, evaluar_k as evaluar
    from mineria.sectores import (
        describir_sectores,
        indicadores_forma,
        mayores_por_grupo,
        ordenar_por_tamano,
        perfiles_sectoriales,
    )
    from sklearn.cluster import KMeans

    columnas = [c for c in ("timestamp", "activity", "subactivity", "valor", "origen_valor")
                if c in pq.ParquetFile(ruta).schema.names]
    ciiu = pd.read_parquet(ruta, columns=columnas)
    perfiles, energia, registro = perfiles_sectoriales(ciiu)
    del ciiu
    x = perfiles.to_numpy()
    evaluacion = evaluar(x, ks=range(2, min(9, len(x) - 1)))
    k = elegir(evaluacion, tamano_minimo=5)
    etiquetas = pd.Series(ordenar_por_tamano(KMeans(k, n_init=20, random_state=42).fit_predict(x)), index=perfiles.index,
                          name="grupo")
    rasgos = indicadores_forma(perfiles)
    tabla = describir_sectores(perfiles, energia, etiquetas)
    mayores = mayores_por_grupo(energia, etiquetas, n=5)

    rutas = fig_sectores(perfiles, etiquetas, rasgos, energia, figuras)
    rasgos.join(energia).assign(grupo=etiquetas.to_numpy()).to_csv(resultados / "mineria_sectores.csv")
    evaluacion.to_csv(resultados / "mineria_sectores_seleccion_k.csv", index=False)
    return {
        "perfiles": registro,
        "seleccion_k": json.loads(evaluacion.round(4).to_json(orient="records")),
        "k": k,
        "grupos": json.loads(tabla.to_json(orient="index")),
        "mayores_por_grupo": json.loads(mayores.astype({"energia_kwh": float}).to_json(orient="records")),
        "figuras": [_relativa(r) for r in rutas],
    }


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description="Mineria no supervisada")
    analizador.add_argument("--panel", type=Path, default=PROCESADO / "panel_modelado.parquet")
    analizador.add_argument("--ciiu", type=Path, default=CIIU)
    analizador.add_argument("--k", type=int, default=None, help="fijar el numero de grupos")
    analizador.add_argument("--k-max", type=int, default=10)
    analizador.add_argument("--sin-sectores", action="store_true")
    analizador.add_argument("--salida", type=Path, default=RAIZ / "docs", help="carpeta base de figuras/ y resultados/")
    args = analizador.parse_args(argv)

    if not args.panel.exists():
        raise FileNotFoundError(f"No existe {args.panel}. Ejecuta antes: python scripts/construir_panel.py --imputacion causal")
    aplicar()
    figuras = args.salida / "figuras" / "mineria"
    resultados = args.salida / "resultados"
    panel = pd.read_parquet(args.panel)

    salida = {"dias": analisis_dias(panel, args.k, range(2, args.k_max + 1), figuras, resultados)}
    if args.sin_sectores:
        salida["sectores"] = {"omitido": "--sin-sectores"}
    elif args.ciiu.exists():
        salida["sectores"] = analisis_sectores(args.ciiu, range(2, 9), figuras, resultados)
    else:
        salida["sectores"] = {"omitido": f"no existe {_relativa(args.ciiu)}; se genera con notebooks/limpieza/xm_ciiu"}

    resultados.mkdir(parents=True, exist_ok=True)
    with (resultados / "mineria.json").open("w", encoding="utf-8") as destino:
        json.dump(salida, destino, ensure_ascii=False, indent=2, default=str)

    dias = salida["dias"]
    print(f"\n=== MINERIA NO SUPERVISADA === {dias['perfiles']['dias_resultantes']:,} dias")
    print(f"  k = {dias['k']} ({dias['k_criterio']})")
    print(f"  silueta {dias['kmeans']['silueta']:.3f} · ARI vs Ward {dias['kmeans']['ari_ward']:.3f} · "
          f"V de Cramer tipo de dia {dias['v_cramer_tipo_dia']:.3f}")
    print(f"  reglas: {dias['reglas']['n_reglas']}")
    if "k" in salida["sectores"]:
        print(f"  sectores CIIU: {salida['sectores']['perfiles']['sectores_resultantes']} en {salida['sectores']['k']} grupos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
