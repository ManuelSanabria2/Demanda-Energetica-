"""Estadisticos del analisis exploratorio (EDA) de la demanda horaria.

Todo lo que el informe afirma con un numero sale de aqui, para que sea
reproducible y probado; `scripts/eda.py` solo dibuja y guarda.

    from mineria.eda import serie_mwh, descriptivos, pruebas, autocorrelacion

Cada prueba estadistica responde a una decision del modelado:

- ADF y KPSS sobre la serie diaria: si la serie tiene tendencia (no
  estacionaria), un arbol no puede extrapolarla; motiva modelar diferencias o
  un componente lineal (seccion de modelos del documento del proyecto).
- Kruskal-Wallis entre tipos de dia y Mann-Whitney festivo vs laboral: si el
  calendario separa la demanda, las banderas del calendario estan justificadas.
- Autocorrelacion en 24 y 168 h: justifica `lag_24` y `lag_168`.
- Spearman temperatura-demanda, global y por hora: justifica la temperatura
  y muestra si su efecto depende de la hora (interaccion que un lineal no ve).

Se usan pruebas no parametricas porque la demanda horaria es bimodal (valle
nocturno y meseta diurna): la normalidad no se sostiene y no hace falta.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


def serie_mwh(panel: pd.DataFrame) -> pd.Series:
    """Serie horaria en MWh con indice en hora local sin zona (Colombia no tiene DST)."""
    tiempo = pd.to_datetime(panel["fecha_hora"])
    if tiempo.dt.tz is not None:
        tiempo = tiempo.dt.tz_convert("America/Bogota").dt.tz_localize(None)
    serie = pd.Series(panel["valor_kwh"].to_numpy(float) / 1000.0, index=tiempo, name="demanda_mwh")
    return serie.sort_index()


def recortar_parcial(serie: pd.Series, dias: int = 5) -> pd.Series:
    """Quita los ultimos `dias` dias (publicacion parcial, notas/hallazgos_calidad.md s.1)."""
    corte = serie.index.max().normalize() - pd.Timedelta(days=dias - 1)
    return serie[serie.index < corte]


def descriptivos(serie: pd.Series) -> dict[str, Any]:
    valores = serie.dropna().to_numpy()
    q1, q2, q3 = np.percentile(valores, [25, 50, 75])
    return {
        "n": int(len(valores)),
        "nulos": int(serie.isna().sum()),
        "media": float(valores.mean()),
        "desviacion": float(valores.std(ddof=1)),
        "coef_variacion_pct": float(100 * valores.std(ddof=1) / valores.mean()),
        "minimo": float(valores.min()),
        "q1": float(q1),
        "mediana": float(q2),
        "q3": float(q3),
        "maximo": float(valores.max()),
        "asimetria": float(stats.skew(valores)),
        "curtosis_exceso": float(stats.kurtosis(valores)),
        "rango": [str(serie.index.min()), str(serie.index.max())],
    }


def crecimiento_anual(serie: pd.Series) -> pd.DataFrame:
    """Energia anual (GWh) y su variacion; los anos incompletos se anotan, no se comparan."""
    diario = serie.resample("D").sum(min_count=24)
    tabla = diario.groupby(diario.index.year).agg(["sum", "count", "mean"])
    tabla.columns = ["energia_mwh", "dias", "media_diaria_mwh"]
    tabla["completo"] = tabla["dias"] >= 365
    tabla["variacion_media_diaria_pct"] = 100 * tabla["media_diaria_mwh"].pct_change()
    tabla["energia_gwh"] = tabla["energia_mwh"] / 1000
    return tabla


def estacionariedad(serie: pd.Series) -> dict[str, Any]:
    """ADF (H0: raiz unitaria) y KPSS (H0: estacionaria en nivel) sobre la serie diaria."""
    diaria = serie.resample("D").mean().dropna()
    adf = adfuller(diaria, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = kpss(diaria, regression="c", nlags="auto")
    diff = diaria.diff(7).dropna()
    adf_diff = adfuller(diff, autolag="AIC")
    return {
        "serie": "media diaria (MWh)",
        "adf_estadistico": float(adf[0]),
        "adf_p": float(adf[1]),
        "kpss_estadistico": float(kp[0]),
        "kpss_p": float(kp[1]),
        "adf_diferencia_semanal_estadistico": float(adf_diff[0]),
        "adf_diferencia_semanal_p": float(adf_diff[1]),
    }


def autocorrelacion(serie: pd.Series, rezagos: int = 336) -> tuple[pd.Series, pd.Series]:
    """ACF hasta 2 semanas y PACF hasta 48 h de la serie horaria."""
    limpia = serie.dropna()
    valores_acf = acf(limpia, nlags=rezagos, fft=True)
    valores_pacf = pacf(limpia, nlags=48, method="ywm")
    return pd.Series(valores_acf, name="acf"), pd.Series(valores_pacf, name="pacf")


def tipo_dia(indice: pd.DatetimeIndex, festivo: pd.Series) -> pd.Series:
    """laboral / sabado / domingo / festivo (el festivo manda sobre el dia de la semana)."""
    dia = indice.dayofweek
    return pd.Series(
        np.select([festivo.to_numpy(bool), dia == 6, dia == 5], ["festivo", "domingo", "sabado"], "laboral"),
        index=indice,
    )


def pruebas_calendario(diario: pd.Series, tipos: pd.Series) -> dict[str, Any]:
    """Kruskal-Wallis entre tipos de dia y Mann-Whitney festivo vs laboral.

    Se aplican a la demanda diaria **relativa a la media de su ano**, para que la
    tendencia no se confunda con el efecto del calendario.
    """
    relativa = diario / diario.groupby(diario.index.year).transform("mean")
    grupos = {t: relativa[tipos == t].dropna() for t in ("laboral", "sabado", "domingo", "festivo")}
    h, p = stats.kruskal(*[g for g in grupos.values() if len(g)])
    u, p_mw = stats.mannwhitneyu(grupos["festivo"], grupos["laboral"], alternative="less")
    n1, n2 = len(grupos["festivo"]), len(grupos["laboral"])
    n_total = sum(len(g) for g in grupos.values())
    return {
        "variable": "demanda diaria / media del ano",
        "medianas": {t: float(g.median()) for t, g in grupos.items()},
        "n": {t: int(len(g)) for t, g in grupos.items()},
        "kruskal_h": float(h),
        "kruskal_p": float(p),
        "epsilon_cuadrado": float(h / (n_total - 1)),
        "mannwhitney_festivo_menor_que_laboral_u": float(u),
        "mannwhitney_p": float(p_mw),
        "efecto_r_rank_biserial": float(1 - 2 * u / (n1 * n2)),
        "festivo_vs_laboral_pct": float(100 * (grupos["festivo"].median() / grupos["laboral"].median() - 1)),
    }


def temperatura(serie: pd.Series, temp: pd.Series) -> dict[str, Any]:
    """Spearman demanda-temperatura, global y por hora del dia."""
    datos = pd.DataFrame({"d": serie, "t": temp}).dropna()
    global_ = stats.spearmanr(datos["d"], datos["t"])
    por_hora = {
        int(h): float(stats.spearmanr(g["d"], g["t"])[0]) for h, g in datos.groupby(datos.index.hour)
    }
    # dentro del dia, temperatura y demanda suben juntas por el ciclo diurno; la
    # correlacion de los residuos diarios quita esa coincidencia
    diario = datos.resample("D").mean().dropna()
    anomalia_d = diario["d"] - diario["d"].rolling(28, center=True, min_periods=7).median()
    anomalia_t = diario["t"] - diario["t"].rolling(28, center=True, min_periods=7).median()
    anom = stats.spearmanr(anomalia_d, anomalia_t, nan_policy="omit")
    return {
        "spearman_horario": float(global_[0]),
        "spearman_horario_p": float(global_[1]),
        "spearman_por_hora": por_hora,
        "spearman_anomalia_diaria": float(anom[0]),
        "spearman_anomalia_diaria_p": float(anom[1]),
        "temperatura_media": float(datos["t"].mean()),
        "temperatura_min": float(datos["t"].min()),
        "temperatura_max": float(datos["t"].max()),
    }


def perfil_horario(serie: pd.Series) -> dict[str, Any]:
    mediana = serie.groupby(serie.index.hour).median()
    return {
        "hora_pico_mediana": int(mediana.idxmax()),
        "hora_valle_mediana": int(mediana.idxmin()),
        "pico_mwh": float(mediana.max()),
        "valle_mwh": float(mediana.min()),
        "valle_sobre_pico": float(mediana.min() / mediana.max()),
        "pico_diario_hora_frecuencia": {
            int(h): int(n) for h, n in serie.groupby(serie.index.date).idxmax().map(lambda t: t.hour).value_counts().items()
        },
    }
