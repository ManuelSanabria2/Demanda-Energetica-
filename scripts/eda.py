"""Analisis exploratorio (EDA) con visualizaciones estadisticas.

    python scripts/eda.py
    python scripts/eda.py --panel data/procesado/panel_modelado.parquet

Entradas (se construyen antes con construir_panel.py y construir_variables.py):
    data/procesado/panel_modelado.parquet
    data/procesado/tabla_modelado_r24.parquet   (opcional: matriz de correlacion)

Salidas, versionadas en git porque son la evidencia de la entrega:
    docs/figuras/eda/*.png
    docs/resultados/eda.json
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

from mineria import eda  # noqa: E402
from mineria.estilo import (  # noqa: E402
    CATEGORICA,
    DIVERGENTE,
    SECUENCIAL,
    TINTA_SECUNDARIA,
    aplicar,
    guardar,
    plt,
)

PROCESADO = RAIZ / "data" / "procesado"
FIGURAS = RAIZ / "docs" / "figuras" / "eda"
RESULTADOS = RAIZ / "docs" / "resultados"
DIAS_SEMANA = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TIPOS = ["laboral", "sabado", "domingo", "festivo"]
TIPOS_EN = {"laboral": "Weekday", "sabado": "Saturday", "domingo": "Sunday", "festivo": "Holiday"}


def _relativa(ruta: Path) -> str:
    try:
        return ruta.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return ruta.as_posix()


def fig_serie(serie: pd.Series, atipico: pd.Series | None) -> Path:
    diario = serie.resample("D").sum(min_count=24) / 1000
    figura, eje = plt.subplots(figsize=(7.2, 2.8))
    eje.plot(diario.index, diario.values, color="#9cc3ef", linewidth=0.7, label="Daily energy")
    eje.plot(diario.index, diario.rolling(30, center=True).mean(), color=CATEGORICA[0], linewidth=2, label="30-day mean")
    if atipico is not None and atipico.any():
        dias = sorted(set(atipico[atipico].index.normalize()))
        valores = diario.reindex(dias)
        eje.scatter(valores.index, valores.values, s=9, color=CATEGORICA[1], zorder=3, label="Day with causal outlier hours")
    eje.set_ylabel("GWh/day")
    eje.set_title("Daily electricity demand of the Colombian SIN (XM DemaReal)")
    eje.legend(loc="lower right", ncol=3, fontsize=7.5)
    return guardar(figura, FIGURAS / "01_serie_diaria.png")


def fig_heatmap(serie: pd.Series) -> Path:
    tabla = serie.groupby([serie.index.dayofweek, serie.index.hour]).median().unstack() / 1000
    figura, eje = plt.subplots(figsize=(7.2, 2.4))
    imagen = eje.imshow(tabla.values, aspect="auto", cmap=SECUENCIAL)
    eje.set_yticks(range(7), DIAS_SEMANA)
    eje.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    eje.set_xlabel("Hour of day")
    eje.grid(False)
    eje.set_title("Median hourly demand by weekday and hour (GWh)")
    figura.colorbar(imagen, ax=eje, fraction=0.025, pad=0.01)
    return guardar(figura, FIGURAS / "02_heatmap_hora_dia.png")


def fig_cajas(serie: pd.Series) -> Path:
    relativa = serie / serie.groupby(serie.index.year).transform("mean")
    figura, ejes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [2, 1.2]})
    for eje, clave, etiquetas, titulo in (
        (ejes[0], relativa.index.hour, [f"{h}" for h in range(24)], "By hour of day"),
        (ejes[1], relativa.index.month, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], "By month"),
    ):
        grupos = [g.values for _, g in relativa.groupby(clave)]
        caja = eje.boxplot(grupos, widths=0.6, patch_artist=True, showfliers=True,
                           flierprops={"markersize": 1.5, "markerfacecolor": TINTA_SECUNDARIA, "markeredgewidth": 0},
                           medianprops={"color": CATEGORICA[1], "linewidth": 1.4})
        for parche in caja["boxes"]:
            parche.set(facecolor="#d6e6f8", edgecolor=CATEGORICA[0], linewidth=0.8)
        eje.set_xticks(range(1, len(etiquetas) + 1), etiquetas, fontsize=7)
        eje.set_title(titulo)
    ejes[0].set_ylabel("Demand / annual mean")
    figura.suptitle("Hourly demand relative to its year's mean (trend removed)", fontsize=10, fontweight="bold", y=1.03)
    return guardar(figura, FIGURAS / "03_cajas_hora_mes.png")


def fig_distribucion(serie: pd.Series) -> Path:
    from scipy.stats import gaussian_kde

    figura, ejes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    valores = serie.dropna().values / 1000
    ejes[0].hist(valores, bins=80, density=True, color="#9cc3ef", edgecolor="white", linewidth=0.3)
    malla = np.linspace(valores.min(), valores.max(), 300)
    ejes[0].plot(malla, gaussian_kde(valores)(malla), color=CATEGORICA[0])
    ejes[0].set_xlabel("Hourly demand (GWh)")
    ejes[0].set_ylabel("Density")
    ejes[0].set_title("Distribution (bimodal: night vs day)")
    anios = sorted(set(serie.index.year))
    grupos = [serie[serie.index.year == a].values / 1000 for a in anios]
    caja = ejes[1].boxplot(grupos, widths=0.6, patch_artist=True, showfliers=False,
                           medianprops={"color": CATEGORICA[1], "linewidth": 1.4})
    for parche in caja["boxes"]:
        parche.set(facecolor="#d6e6f8", edgecolor=CATEGORICA[0])
    ejes[1].set_xticks(range(1, len(anios) + 1), [str(a) for a in anios], fontsize=7.5)
    ejes[1].set_title("By year (upward trend)")
    ejes[1].set_ylabel("GWh")
    return guardar(figura, FIGURAS / "04_distribucion.png")


def fig_acf(acf: pd.Series, pacf: pd.Series, n: int) -> Path:
    banda = 1.96 / np.sqrt(n)
    figura, ejes = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [2, 1]})
    ejes[0].vlines(acf.index, 0, acf.values, color=CATEGORICA[0], linewidth=0.8)
    for rezago in (24, 168, 336):
        ejes[0].axvline(rezago, color=CATEGORICA[1], linestyle="--", linewidth=0.8)
        ejes[0].annotate(f"lag {rezago}\n{acf[rezago]:.2f}", (rezago, acf[rezago]), xytext=(5, 4),
                         textcoords="offset points", fontsize=7, color=TINTA_SECUNDARIA)
    ejes[0].set_title("ACF of hourly demand (2 weeks)")
    ejes[0].set_xlabel("Lag (h)")
    ejes[1].vlines(pacf.index[1:], 0, pacf.values[1:], color=CATEGORICA[0], linewidth=1)
    ejes[1].axhspan(-banda, banda, color="#e4e3df")
    ejes[1].set_title("PACF (48 h)")
    ejes[1].set_xlabel("Lag (h)")
    return guardar(figura, FIGURAS / "05_acf_pacf.png")


def fig_stl(serie: pd.Series) -> Path:
    from statsmodels.tsa.seasonal import STL

    diario = serie.resample("D").sum(min_count=24).dropna() / 1000
    ajuste = STL(diario, period=7, robust=True).fit()
    figura, ejes = plt.subplots(3, 1, figsize=(7.2, 3.8), sharex=True)
    for eje, datos, titulo in (
        (ejes[0], ajuste.trend, "Trend (GWh/day)"),
        (ejes[1], ajuste.seasonal, "Weekly seasonality"),
        (ejes[2], ajuste.resid, "Remainder"),
    ):
        eje.plot(datos.index, datos.values, color=CATEGORICA[0], linewidth=0.8)
        eje.set_ylabel(titulo, fontsize=7.5)
    ejes[0].set_title("Robust STL decomposition of daily energy (period = 7 days)")
    return guardar(figura, FIGURAS / "06_stl.png"), ajuste


def fig_temperatura(serie: pd.Series, temp: pd.Series, por_hora: dict[int, float]) -> Path:
    datos = pd.DataFrame({"d": serie / 1000, "t": temp}).dropna()
    figura, ejes = plt.subplots(1, 2, figsize=(7.4, 2.7), layout="constrained")
    hb = ejes[0].hexbin(datos["t"], datos["d"], gridsize=45, cmap=SECUENCIAL, mincnt=1, linewidths=0)
    figura.colorbar(hb, ax=ejes[0], fraction=0.04, pad=0.01, label="hours")
    ejes[0].set_xlabel("Weighted national temperature (°C)")
    ejes[0].set_ylabel("Hourly demand (GWh)")
    ejes[0].set_title("Demand vs temperature")
    horas = sorted(por_hora)
    valores = [por_hora[h] for h in horas]
    ejes[1].bar(horas, valores, color=[CATEGORICA[0] if v >= 0 else CATEGORICA[1] for v in valores], width=0.8)
    ejes[1].axhline(0, color=TINTA_SECUNDARIA, linewidth=0.6)
    ejes[1].set_xlabel("Hour of day")
    ejes[1].set_ylabel("Spearman ρ")
    ejes[1].set_title("Correlation within each hour")
    return guardar(figura, FIGURAS / "07_temperatura.png")


def fig_calendario(diario: pd.Series, tipos: pd.Series) -> Path:
    relativa = diario / diario.groupby(diario.index.year).transform("mean")
    grupos = [relativa[tipos == t].dropna().values for t in TIPOS]
    figura, eje = plt.subplots(figsize=(7.2, 2.5))
    partes = eje.violinplot(grupos, showmedians=True, widths=0.8)
    for i, cuerpo in enumerate(partes["bodies"]):
        cuerpo.set_facecolor(CATEGORICA[i])
        cuerpo.set_edgecolor(CATEGORICA[i])
        cuerpo.set_alpha(0.55)
    for clave in ("cbars", "cmins", "cmaxes", "cmedians"):
        partes[clave].set_color(TINTA_SECUNDARIA)
        partes[clave].set_linewidth(0.8)
    eje.set_xticks(range(1, 5), [f"{TIPOS_EN[t]}\n(n={len(g)})" for t, g in zip(TIPOS, grupos)])
    eje.set_ylabel("Daily demand / annual mean")
    eje.set_title("Daily demand by type of day (Colombian calendar, Emiliani law)")
    return guardar(figura, FIGURAS / "08_tipo_de_dia.png")


def fig_correlacion(tabla: pd.DataFrame) -> Path:
    columnas = [c for c in tabla.columns if c != "fecha_hora"]
    matriz = tabla[columnas].astype(float).corr(method="spearman")
    figura, eje = plt.subplots(figsize=(6.4, 5.2))
    imagen = eje.imshow(matriz.values, cmap=DIVERGENTE, vmin=-1, vmax=1)
    eje.set_xticks(range(len(columnas)), columnas, rotation=60, ha="right", fontsize=7)
    eje.set_yticks(range(len(columnas)), columnas, fontsize=7)
    eje.grid(False)
    for i in range(len(columnas)):
        for j in range(len(columnas)):
            valor = matriz.values[i, j]
            if abs(valor) >= 0.3 and i != j:
                eje.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=5.5,
                         color="white" if abs(valor) > 0.75 else "#0b0b0b")
    figura.colorbar(imagen, ax=eje, fraction=0.04, pad=0.02, label="Spearman ρ")
    eje.set_title("Spearman correlation, modelling table r24 (|ρ| ≥ 0.3 labelled)")
    return guardar(figura, FIGURAS / "09_correlacion_r24.png"), matriz


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description="EDA de la demanda horaria")
    analizador.add_argument("--panel", type=Path, default=PROCESADO / "panel_modelado.parquet")
    analizador.add_argument("--tabla", type=Path, default=PROCESADO / "tabla_modelado_r24.parquet")
    analizador.add_argument("--dias-parciales", type=int, default=5)
    analizador.add_argument("--salida", type=Path, default=RAIZ / "docs", help="carpeta base de figuras/ y resultados/")
    args = analizador.parse_args(argv)

    global FIGURAS, RESULTADOS
    FIGURAS, RESULTADOS = args.salida / "figuras" / "eda", args.salida / "resultados"

    if not args.panel.exists():
        raise FileNotFoundError(f"No existe {args.panel}. Ejecuta antes: python scripts/construir_panel.py --imputacion causal")
    aplicar()
    panel = pd.read_parquet(args.panel)
    serie = eda.recortar_parcial(eda.serie_mwh(panel), args.dias_parciales)
    indice = serie.index
    panel_local = panel.set_index(eda.serie_mwh(panel).index).reindex(indice)

    atipico = panel_local["atipico"].fillna(False).astype(bool) if "atipico" in panel_local else None
    temp = panel_local["temperatura_c"] if "temperatura_c" in panel_local else None

    diario = serie.resample("D").sum(min_count=24).dropna()
    festivo_dia = panel_local["es_festivo"].fillna(False).astype(bool).resample("D").max().reindex(diario.index)
    tipos = eda.tipo_dia(diario.index, festivo_dia)

    acf, pacf = eda.autocorrelacion(serie)
    figuras = [fig_serie(serie, atipico), fig_heatmap(serie), fig_cajas(serie), fig_distribucion(serie),
               fig_acf(acf, pacf, len(serie))]
    ruta_stl, stl = fig_stl(serie)
    figuras.append(ruta_stl)

    resultados = {
        "fuente": "XM DemaReal/Sistema via panel_modelado.parquet",
        "dias_parciales_excluidos": args.dias_parciales,
        "descriptivos_mwh": eda.descriptivos(serie),
        "perfil_horario": eda.perfil_horario(serie),
        "crecimiento_anual": json.loads(eda.crecimiento_anual(serie).to_json(orient="index")),
        "estacionariedad": eda.estacionariedad(serie),
        "acf": {str(k): float(acf[k]) for k in (1, 24, 48, 168, 336)},
        "stl": {
            "fuerza_tendencia": float(max(0.0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())),
            "fuerza_estacional_semanal": float(max(0.0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())),
        },
        "calendario": eda.pruebas_calendario(diario, tipos),
        "atipicos_causales_horas": int(atipico.sum()) if atipico is not None else None,
    }
    if temp is not None and temp.notna().any():
        resultados["temperatura"] = eda.temperatura(serie, temp)
        figuras.append(fig_temperatura(serie, temp, resultados["temperatura"]["spearman_por_hora"]))
    figuras.append(fig_calendario(diario, tipos))
    if args.tabla.exists():
        tabla = pd.read_parquet(args.tabla)
        ruta, matriz = fig_correlacion(tabla)
        figuras.append(ruta)
        objetivo = matriz["demanda_mwh"].drop("demanda_mwh").sort_values(key=abs, ascending=False)
        resultados["correlacion_con_objetivo_r24"] = {k: round(float(v), 4) for k, v in objetivo.items()}

    resultados["figuras"] = [_relativa(p) for p in figuras]
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    with (RESULTADOS / "eda.json").open("w", encoding="utf-8") as destino:
        json.dump(resultados, destino, ensure_ascii=False, indent=2, default=str)

    d = resultados["descriptivos_mwh"]
    c = resultados["calendario"]
    print(f"\n=== EDA === {d['n']:,} horas · {d['rango'][0]} .. {d['rango'][1]}")
    print(f"  media {d['media']:.0f} MWh · CV {d['coef_variacion_pct']:.1f} % · asimetria {d['asimetria']:.2f}")
    print(f"  ACF 24h {resultados['acf']['24']:.3f} · ACF 168h {resultados['acf']['168']:.3f}")
    print(f"  Kruskal-Wallis tipo de dia H={c['kruskal_h']:.1f} p={c['kruskal_p']:.2e}")
    print(f"  {len(figuras)} figuras -> {_relativa(FIGURAS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
