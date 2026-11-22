"""Exporta a app/datos/ lo que la interfaz necesita y no esta en git.

La interfaz (app/) no entrena ni descarga: solo lee artefactos. La mayoria ya
estan versionados (docs/resultados/*.json, docs/figuras/, infra/costos/), pero
la serie diaria y los perfiles salen del panel de modelado, que vive fuera de
git. Este script los resume en tablas pequenas y las deja en app/datos/ con un
manifiesto (fecha y SHA-256 de cada archivo):

    python scripts/exportar_interfaz.py

| archivo                       | contenido                                                    |
|-------------------------------|--------------------------------------------------------------|
| serie_diaria.parquet          | energia diaria (MWh), temperatura media y tipo de dia        |
| perfil_horario.parquet        | demanda media por dia de la semana y hora (168 filas)        |
| autocorrelacion.parquet       | ACF hasta 336 h y PACF hasta 48 h de la serie horaria        |
| centroides_grupos.parquet     | forma media de cada grupo de dias del K-means, hora a hora   |
| predicciones_prueba.parquet   | copia de docs/resultados/ (si existe: evaluar_final.py)      |
| probabilidades_alerta.parquet | copia de docs/resultados/ (si existe: evaluar_final.py)      |
| manifiesto_interfaz.json      | momento, origen y SHA-256 de cada archivo; lo que falto      |

Si falta una entrada opcional (las etiquetas de la mineria o las predicciones
del modelado), se anota en `faltantes` del manifiesto y se sigue: la pagina que
la usa muestra el aviso de pendiente. Sin panel no hay nada que exportar y el
script falla con el comando que lo genera.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from mineria import eda  # noqa: E402
from mineria.perfiles import NOMBRES_DIA, forma_y_nivel, perfiles_diarios  # noqa: E402

VERSION_FORMATO = 1
DIAS_PARCIALES = 5
TIPOS_DIA = ("laboral", "sabado", "domingo", "festivo", "semana_santa")

# Copias directas de artefactos del modelado: archivo -> script que lo genera.
COPIAS = {
    "predicciones_prueba.parquet": "python scripts/evaluar_final.py",
    "probabilidades_alerta.parquet": "python scripts/evaluar_final.py",
}


# --------------------------------------------------------------------------
# Escritura atomica y huellas
# --------------------------------------------------------------------------


def sha256(ruta: Path) -> str:
    huella = hashlib.sha256()
    with ruta.open("rb") as origen:
        for bloque in iter(lambda: origen.read(1 << 20), b""):
            huella.update(bloque)
    return huella.hexdigest()


def _escribir_parquet(marco: pd.DataFrame, ruta: Path) -> None:
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    marco.to_parquet(temporal, index=False)
    temporal.replace(ruta)


def _escribir_json(datos: dict, ruta: Path) -> None:
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporal.replace(ruta)


def _copiar(origen: Path, destino: Path) -> None:
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    shutil.copyfile(origen, temporal)
    temporal.replace(destino)


def _relativa(ruta: Path) -> str:
    try:
        return ruta.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return ruta.as_posix()


# --------------------------------------------------------------------------
# Tablas: cada una devuelve (marco, registro)
# --------------------------------------------------------------------------


def _serie_y_panel_local(panel: pd.DataFrame, dias_parciales: int) -> tuple[pd.Series, pd.DataFrame]:
    """Serie horaria en MWh sin los dias parciales, y el panel con el mismo indice local."""
    completa = eda.serie_mwh(panel)
    serie = eda.recortar_parcial(completa, dias_parciales)
    panel_local = panel.set_index(completa.index).reindex(serie.index)
    return serie, panel_local


def tipo_dia(indice: pd.DatetimeIndex, festivo: pd.Series, semana_santa: pd.Series) -> pd.Series:
    """laboral / sabado / domingo / festivo / semana_santa.

    Igual que `mineria.eda.tipo_dia` (el festivo manda sobre el dia de la
    semana), y la Semana Santa manda sobre todo: es una semana entera con un
    perfil propio, festivos del jueves y el viernes incluidos.
    """
    base = eda.tipo_dia(indice, festivo)
    return base.mask(semana_santa.to_numpy(bool), "semana_santa")


def serie_diaria(panel: pd.DataFrame, dias_parciales: int = DIAS_PARCIALES) -> tuple[pd.DataFrame, dict[str, Any]]:
    serie, local = _serie_y_panel_local(panel, dias_parciales)
    energia = serie.resample("D").sum(min_count=24)
    completos = energia.notna()
    dias = energia.index[completos]

    def por_dia(columna: str, como: str) -> pd.Series:
        if columna not in local:
            return pd.Series(np.nan, index=dias)
        datos = local[columna]
        if como == "max":
            datos = datos.fillna(False).astype(bool)
        return getattr(datos.resample("D"), como)().reindex(dias)

    festivo = por_dia("es_festivo", "max")
    santa = por_dia("es_semana_santa", "max")
    nombres = (
        local["nombre_festivo"].resample("D").first().reindex(dias)
        if "nombre_festivo" in local
        else pd.Series(None, index=dias, dtype=object)
    )
    marco = pd.DataFrame(
        {
            "fecha": dias,
            "demanda_mwh": energia[completos].to_numpy(),
            "demanda_media_mwh": serie.resample("D").mean().reindex(dias).to_numpy(),
            "temperatura_media_c": por_dia("temperatura_c", "mean").to_numpy(),
            "tipo_dia": tipo_dia(dias, festivo, santa).to_numpy(),
            "es_festivo": festivo.to_numpy(bool),
            "nombre_festivo": nombres.to_numpy(),
            "horas_atipicas": por_dia("atipico", "sum").fillna(0).astype(int).to_numpy(),
        }
    )
    registro = {
        "operacion": "serie_diaria",
        "criterio": (
            "energia diaria = suma de las 24 h en MWh; solo dias con las 24 h; se descartan los "
            f"ultimos {dias_parciales} dias por publicacion parcial; tipo de dia: semana_santa > "
            "festivo > domingo > sabado > laboral; horas_atipicas = marca causal `atipico`"
        ),
        "dias": int(len(marco)),
        "dias_incompletos": int((~completos).sum()),
        "rango": [str(dias.min().date()), str(dias.max().date())],
        "tipos": {t: int((marco["tipo_dia"] == t).sum()) for t in TIPOS_DIA},
    }
    return marco, registro


def perfil_horario(panel: pd.DataFrame, dias_parciales: int = DIAS_PARCIALES) -> tuple[pd.DataFrame, dict[str, Any]]:
    serie, _ = _serie_y_panel_local(panel, dias_parciales)
    datos = pd.DataFrame({"dia_semana": serie.index.dayofweek, "hora": serie.index.hour, "mwh": serie.to_numpy()})
    marco = (
        datos.groupby(["dia_semana", "hora"])["mwh"]
        .agg(demanda_media_mwh="mean", demanda_mediana_mwh="median", n="count")
        .reset_index()
    )
    marco.insert(1, "nombre_dia", marco["dia_semana"].map(dict(enumerate(NOMBRES_DIA))))
    registro = {
        "operacion": "perfil_horario",
        "criterio": "media y mediana de la demanda horaria (MWh) por dia de la semana (0 = lunes) y hora local",
        "filas": int(len(marco)),
        "horas": int(datos["mwh"].notna().sum()),
    }
    return marco, registro


def autocorrelacion(panel: pd.DataFrame, dias_parciales: int = DIAS_PARCIALES) -> tuple[pd.DataFrame, dict[str, Any]]:
    """La misma ACF/PACF que scripts/eda.py: eda.json solo guarda cinco rezagos."""
    serie, _ = _serie_y_panel_local(panel, dias_parciales)
    acf, pacf = eda.autocorrelacion(serie)
    marco = pd.DataFrame({"rezago_h": np.arange(len(acf)), "acf": acf.to_numpy()})
    marco["pacf"] = pacf.reindex(marco.index).to_numpy()
    registro = {
        "operacion": "autocorrelacion",
        "criterio": "mineria.eda.autocorrelacion sobre la serie horaria sin dias parciales: ACF hasta 336 h, PACF hasta 48 h",
        "rezagos": int(len(marco) - 1),
        "banda_95": float(1.96 / np.sqrt(serie.notna().sum())),
    }
    return marco, registro


def centroides_grupos(
    panel: pd.DataFrame, etiquetas: pd.DataFrame, dias_parciales: int = DIAS_PARCIALES
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Forma media (media del dia = 1) de cada grupo, con su banda p10-p90.

    Los grupos son los de docs/resultados/mineria_etiquetas_dia.csv: aqui no se
    vuelve a agrupar, solo se promedia la forma de los dias de cada etiqueta.
    """
    perfiles, _ = perfiles_diarios(panel, dias_parciales=dias_parciales)
    forma, _ = forma_y_nivel(perfiles)
    forma.index = pd.to_datetime(forma.index)
    grupo = etiquetas.assign(fecha=pd.to_datetime(etiquetas["fecha"])).set_index("fecha")["grupo"]
    comunes = forma.index.intersection(grupo.index)
    largo = (
        forma.loc[comunes]
        .assign(grupo=grupo.loc[comunes])
        .melt(id_vars="grupo", var_name="hora", value_name="forma")
    )
    largo["hora"] = largo["hora"].str[1:].astype(int)
    marco = (
        largo.groupby(["grupo", "hora"])["forma"]
        .agg(forma_media="mean", p10=lambda x: x.quantile(0.10), p90=lambda x: x.quantile(0.90), dias="count")
        .reset_index()
    )
    registro = {
        "operacion": "centroides_grupos",
        "criterio": "media y percentiles 10/90 de la forma diaria (dia / su media) de los dias de cada grupo",
        "dias_etiquetados": int(len(grupo)),
        "dias_con_perfil": int(len(comunes)),
        "dias_sin_perfil": int(len(grupo.index.difference(forma.index))),
        "grupos": sorted(marco["grupo"].unique().tolist()),
    }
    return marco, registro


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------


def exportar(panel_ruta: Path, resultados: Path, destino: Path, dias_parciales: int = DIAS_PARCIALES) -> dict[str, Any]:
    if not panel_ruta.exists():
        raise FileNotFoundError(
            f"No existe {panel_ruta}. Ejecuta antes: python scripts/construir_panel.py --imputacion causal"
        )
    destino.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_ruta)

    tablas: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {
        "serie_diaria.parquet": serie_diaria(panel, dias_parciales),
        "perfil_horario.parquet": perfil_horario(panel, dias_parciales),
        "autocorrelacion.parquet": autocorrelacion(panel, dias_parciales),
    }
    faltantes: list[dict[str, str]] = []

    ruta_etiquetas = resultados / "mineria_etiquetas_dia.csv"
    if ruta_etiquetas.exists():
        tablas["centroides_grupos.parquet"] = centroides_grupos(panel, pd.read_csv(ruta_etiquetas), dias_parciales)
    else:
        faltantes.append({"archivo": "centroides_grupos.parquet", "falta": _relativa(ruta_etiquetas),
                          "script": "python scripts/mineria_no_supervisada.py"})

    archivos: dict[str, dict[str, Any]] = {}
    for nombre, (marco, registro) in tablas.items():
        ruta = destino / nombre
        _escribir_parquet(marco, ruta)
        archivos[nombre] = {"sha256": sha256(ruta), "bytes": ruta.stat().st_size, "filas": int(len(marco)),
                            "columnas": list(marco.columns), "origen": _relativa(panel_ruta), "registro": registro}

    for nombre, script in COPIAS.items():
        origen = resultados / nombre
        if not origen.exists():
            faltantes.append({"archivo": nombre, "falta": _relativa(origen), "script": script})
            continue
        ruta = destino / nombre
        _copiar(origen, ruta)
        huella = sha256(ruta)
        if huella != sha256(origen):
            raise RuntimeError(f"La copia de {origen} no coincide con el original")
        archivos[nombre] = {"sha256": huella, "bytes": ruta.stat().st_size,
                            "filas": int(pq.ParquetFile(ruta).metadata.num_rows),
                            "origen": _relativa(origen), "registro": {"operacion": "copiar", "criterio": "copia byte a byte"}}

    manifiesto = {
        "version_formato": VERSION_FORMATO,
        "momento": dt.datetime.now().isoformat(timespec="seconds"),
        "panel": {"ruta": _relativa(panel_ruta), "sha256": sha256(panel_ruta), "filas": int(len(panel))},
        "dias_parciales_excluidos": dias_parciales,
        "archivos": archivos,
        "faltantes": faltantes,
    }
    _escribir_json(manifiesto, destino / "manifiesto_interfaz.json")
    return manifiesto


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description="Exporta a app/datos/ los artefactos de la interfaz")
    analizador.add_argument("--panel", type=Path, default=RAIZ / "data" / "procesado" / "panel_modelado.parquet")
    analizador.add_argument("--resultados", type=Path, default=RAIZ / "docs" / "resultados")
    analizador.add_argument("--destino", type=Path, default=RAIZ / "app" / "datos")
    analizador.add_argument("--dias-parciales", type=int, default=DIAS_PARCIALES)
    args = analizador.parse_args(argv)

    manifiesto = exportar(args.panel, args.resultados, args.destino, args.dias_parciales)
    print(f"\n=== Interfaz === {_relativa(args.destino)}")
    for nombre, info in manifiesto["archivos"].items():
        print(f"  {nombre:<32} {info['filas']:>7,} filas  {info['bytes'] / 1024:>7.0f} KB  {info['sha256'][:12]}")
    for falta in manifiesto["faltantes"]:
        print(f"  PENDIENTE {falta['archivo']}: falta {falta['falta']} ({falta['script']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
