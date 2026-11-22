"""Interpretabilidad de los candidatos finales, sobre el bloque de validacion.

Candidatos, por tabla, leidos de los resultados ya escritos (nada a mano):

- **lineal**: el de menor MAPE de validacion entre los modelos lineales *en las
  variables originales* (`LINEALES_EN_VARIABLES`). Las variantes polinomiales
  quedan fuera aunque tengan menos MAPE: `LinearExplainer` explica un modelo
  lineal en las variables que recibe, y la polinomial es lineal en 152
  terminos con interacciones, no en las 16 variables. El registro las anota.
- **ensemble**: el de menor MAPE de validacion entre los ensembles (en sus dos
  formas) y, si existe, las versiones reentrenadas de la busqueda de
  hiperparametros (modelado_busqueda.json).

Cada modelo se lee de models/ comprobando su SHA-256 contra el registro: se
explica exactamente el modelo evaluado, sin reentrenar.

1. **Importancias.** `feature_importances_` del ensemble (reduccion media de
   impureza, medida en entrenamiento; HistGradientBoosting no la expone y se
   calcula igual, sumando la ganancia de cada division por variable) y
   `permutation_importance` en validacion para los dos candidatos (aumento
   del MAE al permutar cada columna).
2. **SHAP.** `TreeExplainer` (tree_path_dependent) para el ensemble y
   `LinearExplainer` para el lineal (sobre las variables estandarizadas: el
   escalado es una por una, asi que cada valor SHAP sigue siendo de su
   variable). Si un piloto de `PILOTO_SHAP` filas estima mas de
   `LIMITE_SHAP_S` segundos para toda la validacion, se usa una muestra
   aleatoria de `MUESTRA_SHAP` filas (semilla 42) y se anota.
3. **LIME** en modo regresion para cuatro casos de validacion: una hora pico
   laboral, un festivo, un dia de Semana Santa y la hora de mayor error
   absoluto de cada modelo. La regla de cada caso va en el registro.

Por ultimo contrasta con el EDA (autocorrelacion a 168 h frente a 24 h) y
califica la interpretabilidad de cada modelo (`CALIFICACIONES`).

El bloque de prueba no se toca: todo se calcula en validacion, y el fondo de
las explicaciones sale del entrenamiento.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from modelado.datos import registro_operacion
from modelado.ensembles import RegresorDiferenciado, estimador_interno
from modelado.metricas import HORAS_PICO
from modelado.tiempos import cargar_modelo

SEMILLA = 42
LINEALES_EN_VARIABLES = ("lineal", "ridge", "lasso", "elasticnet")
REPETICIONES_PERMUTACION = 10
PILOTO_SHAP = 200
LIMITE_SHAP_S = 60.0
MUESTRA_SHAP = 2000
FONDO_LINEAL = 1000
MUESTRAS_LIME = 5000
VARIABLES_LIME = 8
HORA_CASOS = 19  # hora pico (HORAS_PICO) para los casos de calendario
COLUMNA_ESTACIONAL = "lag_168"


class ErrorInterpretabilidad(Exception):
    """No se puede explicar el modelo con seguridad."""


# --------------------------------------------------------------------------
# Calificacion de interpretabilidad
# --------------------------------------------------------------------------

CRITERIO_CALIFICACION = (
    "alta: la prediccion se lee directamente del modelo (una columna o una suma de "
    "coeficientes por variable); media: el modelo tiene una estructura legible pero "
    "con muchos terminos, interacciones o explicaciones solo locales; baja: cientos "
    "de arboles o un kernel, sin forma cerrada; solo se explica a posteriori."
)

CALIFICACIONES: dict[str, dict[str, str]] = {
    "baseline_estacional": {"calificacion": "alta", "motivo": "la prediccion es lag_168, sin parametros."},
    "baseline_diario": {"calificacion": "alta", "motivo": "la prediccion es lag_24, sin parametros."},
    "lineal": {"calificacion": "alta", "motivo": "un coeficiente por variable; SHAP coincide con coef * (x - media)."},
    "ridge": {"calificacion": "alta", "motivo": "como la lineal, con coeficientes contraidos y mas estables."},
    "lasso": {"calificacion": "alta", "motivo": "coeficientes por variable y ademas anula los irrelevantes."},
    "elasticnet": {"calificacion": "alta", "motivo": "coeficientes por variable con contraccion mixta L1/L2."},
    "polinomial": {
        "calificacion": "media",
        "motivo": "152 terminos con cuadrados e interacciones; matriz singular, coeficientes inestables.",
    },
    "polinomial_ridge": {
        "calificacion": "media",
        "motivo": "152 terminos con interacciones; Ridge los estabiliza pero no se leen por variable.",
    },
    "random_forest": {
        "calificacion": "baja",
        "motivo": "200 arboles profundos; se explica a posteriori con TreeSHAP exacto, pero es lento.",
    },
    "adaboost": {
        "calificacion": "baja",
        "motivo": "100 arboles con mediana ponderada; TreeExplainer no lo soporta, solo metodos agnosticos.",
    },
    "gradient_boosting": {
        "calificacion": "baja",
        "motivo": "cientos de arboles en serie; se explica a posteriori con TreeSHAP exacto y rapido.",
    },
    "hist_gradient": {
        "calificacion": "baja",
        "motivo": "cientos de arboles sobre variables discretizadas; TreeSHAP exacto y rapido.",
    },
    "logistica": {"calificacion": "alta", "motivo": "un coeficiente por variable en la escala de log-odds."},
    "svm_lineal": {"calificacion": "alta", "motivo": "hiperplano con un peso por variable."},
    "knn": {
        "calificacion": "media",
        "motivo": "cada prediccion se explica por sus vecinos, pero no hay una explicacion global.",
    },
    "svm_rbf": {
        "calificacion": "baja",
        "motivo": "kernel RBF sobre ~1 800 vectores de soporte; sin pesos por variable.",
    },
}
NOTA_DIFERENCIADO = (
    "En la forma diferenciada el nivel lo pone lag_168 de forma explicita (y = lag_168 + "
    "arboles); la calificacion no cambia, pero esa parte se lee directamente."
)


# --------------------------------------------------------------------------
# Candidatos
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidato:
    rol: str  # "lineal" o "ensemble"
    familia: str
    nombre: str
    forma: str
    origen: str
    mape_validacion_pct: float
    ruta: str
    sha256: str

    @property
    def etiqueta(self) -> str:
        return self.nombre if self.forma == "-" else f"{self.nombre} ({self.forma}, {self.origen})"


def seleccionar_candidatos(
    resultados: dict[str, dict[str, Any]], clave: str
) -> tuple[dict[str, Candidato], dict[str, Any]]:
    """El mejor lineal en las variables originales y el mejor ensemble, por MAPE de validacion.

    `resultados`: {"lineal": ..., "ensembles": ..., "busqueda": ... (opcional)}.
    """
    for familia in ("lineal", "ensembles"):
        if clave not in (resultados.get(familia) or {}).get("tablas", {}):
            raise ErrorInterpretabilidad(f"Faltan los resultados de la familia {familia!r} para {clave}.")

    lineales, excluidos = [], {}
    for nombre, registro in resultados["lineal"]["tablas"][clave]["modelos"].items():
        mape_pct = registro["metricas_validacion"]["mape_pct"]
        if nombre not in LINEALES_EN_VARIABLES:
            excluidos[nombre] = {
                "mape_validacion_pct": mape_pct,
                "motivo": "no es lineal en las variables originales: LinearExplainer no la explica por variable",
            }
            continue
        lineales.append(Candidato("lineal", "lineal", nombre, "-", "modelado_regresion.json", mape_pct,
                                  registro["modelo"]["ruta"], registro["modelo"]["sha256"]))

    arboles = []
    for nombre, formas in resultados["ensembles"]["tablas"][clave]["modelos"].items():
        for forma, registro in formas.items():
            arboles.append(Candidato("ensemble", "ensembles", nombre, forma, "modelado_ensembles.json",
                                     registro["metricas_validacion"]["mape_pct"],
                                     registro["modelo"]["ruta"], registro["modelo"]["sha256"]))
    busqueda = (resultados.get("busqueda") or {}).get("tablas", {}).get(clave)
    for registro in (busqueda or {}).get("modelos", {}).values():
        if registro.get("familia") != "ensembles":
            continue
        for metodo in ("grid", "random"):
            r = registro[metodo]
            arboles.append(Candidato("ensemble", "ensembles", registro["nombre"], registro["forma"],
                                     f"modelado_busqueda.json ({metodo})", r["metricas_validacion"]["mape_pct"],
                                     r["modelo"]["ruta"], r["modelo"]["sha256"]))
    if not lineales or not arboles:
        raise ErrorInterpretabilidad("No hay candidatos lineales o ensembles que explicar.")

    lineal = min(lineales, key=lambda c: c.mape_validacion_pct)
    ensemble = min(arboles, key=lambda c: c.mape_validacion_pct)
    registro = registro_operacion(
        "seleccionar_candidatos",
        "menor MAPE de validacion: entre los lineales en las variables originales, y entre los "
        "ensembles (dos formas) y sus versiones de la busqueda de hiperparametros",
        lineal=asdict(lineal),
        ensemble=asdict(ensemble),
        lineales_excluidos=excluidos,
        ensembles_considerados=len(arboles),
        busqueda_incluida=busqueda is not None,
    )
    return {"lineal": lineal, "ensemble": ensemble}, registro


def cargar(candidato: Candidato, dir_modelos: Path):
    """El modelo del registro, leido de `dir_modelos` y comprobando su SHA-256."""
    ruta = Path(dir_modelos) / Path(candidato.ruta).name
    if not ruta.exists():
        raise ErrorInterpretabilidad(f"No esta {candidato.ruta}: vuelve a entrenar su familia.")
    return cargar_modelo(ruta, candidato.sha256)


# --------------------------------------------------------------------------
# Importancias
# --------------------------------------------------------------------------


def _ordenadas(valores: np.ndarray, variables: list[str]) -> dict[str, float]:
    orden = np.argsort(valores)[::-1]
    return {variables[i]: float(valores[i]) for i in orden}


def importancia_impureza(modelo, variables: list[str]) -> dict[str, Any]:
    """Reduccion media de impureza por variable, normalizada a suma 1.

    Random Forest, AdaBoost y Gradient Boosting la exponen como
    `feature_importances_`. HistGradientBoosting no: se suma la ganancia
    (reduccion de la perdida) de cada division de cada arbol por variable,
    que es la misma idea (lo que LightGBM llama importancia por ganancia).
    En la forma diferenciada es la del ensemble que aprende y - lag_168.
    """
    interno = estimador_interno(modelo)
    if hasattr(interno, "feature_importances_"):
        valores = np.asarray(interno.feature_importances_, dtype="float64")
        metodo = "feature_importances_ de sklearn"
    elif hasattr(interno, "_predictors"):
        valores = np.zeros(len(variables))
        for iteracion in interno._predictors:
            for arbol in iteracion:
                nodos = arbol.nodes
                internos = nodos["is_leaf"] == 0
                np.add.at(valores, nodos["feature_idx"][internos], nodos["gain"][internos])
        valores = valores / valores.sum()
        metodo = "suma de la ganancia de cada division por variable (HistGradientBoosting), normalizada"
    else:
        raise ErrorInterpretabilidad(f"{type(interno).__name__} no tiene importancia por impureza.")
    if len(valores) != len(variables):
        raise ErrorInterpretabilidad("La importancia no tiene una entrada por variable.")
    return {
        "metodo": metodo,
        "bloque": "entrenamiento (se mide en los arboles ajustados)",
        "objetivo": "y - lag_168" if isinstance(modelo, RegresorDiferenciado) else "y",
        "por_variable": _ordenadas(valores, variables),
    }


def importancia_permutacion(modelo, X: pd.DataFrame, y: pd.Series, n_jobs: int = -1) -> dict[str, Any]:
    """Aumento del MAE en validacion al permutar cada columna, media y desviacion de las repeticiones."""
    inicio = time.perf_counter()
    resultado = permutation_importance(
        modelo, X, y, scoring="neg_mean_absolute_error", n_repeats=REPETICIONES_PERMUTACION,
        random_state=SEMILLA, n_jobs=n_jobs,
    )
    segundos = time.perf_counter() - inicio
    variables = list(X.columns)
    orden = np.argsort(resultado.importances_mean)[::-1]
    return {
        "metodo": f"sklearn.inspection.permutation_importance, scoring MAE, n_repeats={REPETICIONES_PERMUTACION}",
        "bloque": "validacion",
        "unidad": "aumento del MAE en MWh",
        "por_variable": {
            variables[i]: {"media": float(resultado.importances_mean[i]), "desviacion": float(resultado.importances_std[i])}
            for i in orden
        },
        "filas": int(len(X)),
        "segundos": segundos,
    }


MOTIVOS_DIFERENCIA = [
    "La importancia por impureza se mide en entrenamiento, sobre las divisiones del arbol; la de "
    "permutacion, en validacion, sobre el error. Una variable que ayuda a ajustar el pasado pero no "
    "generaliza pesa mas en la primera.",
    "La impureza favorece variables continuas con muchos cortes posibles (lag_168, ma_168, "
    "temperatura_c) frente a binarias como es_festivo, que solo ofrecen un corte.",
    "Con variables correlacionadas (lag_24, lag_168 y ma_168) la permutacion de una no rompe la "
    "informacion que siguen dando las otras: su importancia por permutacion se subestima, y el arbol "
    "reparte la impureza entre ellas de forma casi arbitraria.",
    "Permutar crea filas imposibles (un festivo con la demanda de un dia laboral); el modelo las "
    "evalua fuera de lo que vio.",
    "En la forma diferenciada la impureza explica y - lag_168 y la permutacion la prediccion completa.",
]


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------


def _explicador(rol: str, modelo, X_fondo: pd.DataFrame):
    """Funcion filas -> (valores SHAP en las variables originales, valor base)."""
    import shap

    if rol == "ensemble":
        interno = estimador_interno(modelo)
        explicador = shap.TreeExplainer(interno, feature_perturbation="tree_path_dependent")
        base_interna = float(np.ravel(explicador.expected_value)[0])
        diferenciado = isinstance(modelo, RegresorDiferenciado)
        if diferenciado:
            posicion = list(X_fondo.columns).index(modelo.columna)
            media = float(X_fondo[modelo.columna].mean())

        def explicar(X: pd.DataFrame) -> tuple[np.ndarray, float]:
            valores = np.asarray(explicador.shap_values(X), dtype="float64")
            if not diferenciado:
                return valores, base_interna
            # y = arboles + lag_168: el termino aditivo aporta x - media a su variable.
            valores[:, posicion] += X[modelo.columna].to_numpy(dtype="float64") - media
            return valores, base_interna + media

        descripcion = "shap.TreeExplainer(feature_perturbation='tree_path_dependent')"
        if diferenciado:
            descripcion += f" sobre y - {modelo.columna}, mas el termino aditivo {modelo.columna}"
        return explicar, descripcion

    if rol == "lineal":
        escalar, lineal = modelo[:-1], modelo[-1]
        fondo = escalar.transform(X_fondo)
        masker = shap.maskers.Independent(fondo, max_samples=len(fondo))
        explicador = shap.LinearExplainer(lineal, masker)

        def explicar(X: pd.DataFrame) -> tuple[np.ndarray, float]:
            valores = np.asarray(explicador.shap_values(escalar.transform(X)), dtype="float64")
            return valores, float(np.ravel(explicador.expected_value)[0])

        return explicar, "shap.LinearExplainer sobre las variables estandarizadas, fondo Independent"
    raise ErrorInterpretabilidad(f"Rol desconocido: {rol!r}")


def muestra_fondo(X_entrena: pd.DataFrame, n: int = FONDO_LINEAL) -> pd.DataFrame:
    """Muestra aleatoria del entrenamiento (semilla 42) como distribucion de referencia."""
    if len(X_entrena) <= n:
        return X_entrena
    rng = np.random.default_rng(SEMILLA)
    return X_entrena.iloc[np.sort(rng.choice(len(X_entrena), size=n, replace=False))]


def explicar_shap(
    rol: str, modelo, X_valida: pd.DataFrame, X_entrena: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Valores SHAP en validacion (o en una muestra si es lento).

    Devuelve (posiciones dentro de validacion, valores, valor base, registro).
    """
    explicar, descripcion = _explicador(rol, modelo, muestra_fondo(X_entrena))
    n = len(X_valida)
    piloto = min(PILOTO_SHAP, n)
    inicio = time.perf_counter()
    explicar(X_valida.iloc[:piloto])
    estimado = (time.perf_counter() - inicio) * n / piloto

    if estimado > LIMITE_SHAP_S and n > MUESTRA_SHAP:
        rng = np.random.default_rng(SEMILLA)
        posiciones = np.sort(rng.choice(n, size=MUESTRA_SHAP, replace=False))
        motivo = (f"el piloto de {piloto} filas estima {estimado:.0f} s para las {n} filas "
                  f"(> {LIMITE_SHAP_S:.0f} s): muestra aleatoria de {MUESTRA_SHAP} filas, semilla {SEMILLA}")
    else:
        posiciones = np.arange(n)
        motivo = f"el piloto estima {estimado:.1f} s: toda la validacion"

    inicio = time.perf_counter()
    X_explicar = X_valida.iloc[posiciones]
    valores, base = explicar(X_explicar)
    segundos = time.perf_counter() - inicio
    prediccion = np.asarray(modelo.predict(X_explicar), dtype="float64")
    aditividad = float(np.abs(valores.sum(axis=1) + base - prediccion).max())
    if aditividad > 1e-3 * float(np.abs(prediccion).mean()):
        raise ErrorInterpretabilidad(f"Los valores SHAP no suman la prediccion (error maximo {aditividad:.3g}).")

    variables = list(X_valida.columns)
    media_abs = np.abs(valores).mean(axis=0)
    registro = {
        "metodo": descripcion,
        "filas_validacion": int(n),
        "filas_explicadas": int(len(posiciones)),
        "muestra": len(posiciones) < n,
        "motivo_muestra": motivo,
        "segundos_estimados_completo": float(estimado),
        "segundos": segundos,
        "valor_base_mwh": base,
        "error_aditividad_max_mwh": aditividad,
        "media_abs_por_variable_mwh": _ordenadas(media_abs, variables),
    }
    return posiciones, valores, base, registro


def resumen_temperatura(valores: np.ndarray, X: pd.DataFrame, horas: np.ndarray) -> dict[str, Any]:
    """Efecto SHAP de temperatura_c: pendiente global y por franja horaria (MWh por grado)."""
    if "temperatura_c" not in X.columns:
        return {"disponible": False}
    posicion = list(X.columns).index("temperatura_c")
    t = X["temperatura_c"].to_numpy(dtype="float64")
    s = valores[:, posicion]

    def pendiente(mascara: np.ndarray) -> float | None:
        if mascara.sum() < 3 or np.ptp(t[mascara]) == 0:
            return None
        return float(np.polyfit(t[mascara], s[mascara], 1)[0])

    franjas = {"madrugada 0-5 h": (0, 5), "manana 6-11 h": (6, 11), "tarde 12-17 h": (12, 17),
               "noche 18-23 h": (18, 23)}
    return {
        "disponible": True,
        "pendiente_global_mwh_por_grado": pendiente(np.ones_like(t, dtype=bool)),
        "pendiente_por_franja_mwh_por_grado": {
            nombre: pendiente((horas >= a) & (horas <= b)) for nombre, (a, b) in franjas.items()
        },
        "media_abs_mwh": float(np.abs(s).mean()),
        "rango_shap_mwh": [float(s.min()), float(s.max())],
    }


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------


def _hora(fechas: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(fechas).hour.to_numpy()


def casos_lime(
    fechas: pd.Series, X: pd.DataFrame, y: pd.Series, predicciones: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Posiciones (dentro de validacion) de los casos a explicar, con la regla que los elige."""
    momentos = pd.DatetimeIndex(fechas)
    hora = momentos.hour.to_numpy()
    laborable = momentos.dayofweek.to_numpy() < 5
    marca = {c: X[c].to_numpy() == 1 for c in ("es_festivo", "es_semana_santa", "es_puente", "es_vispera_festivo")
             if c in X.columns}
    for requerida in ("es_festivo", "es_semana_santa"):
        if requerida not in marca:
            raise ErrorInterpretabilidad(f"Falta {requerida} para elegir los casos de LIME.")
    especial = marca["es_festivo"] | marca["es_semana_santa"] | marca.get("es_puente", False) | marca.get(
        "es_vispera_festivo", False)
    real = y.to_numpy(dtype="float64")

    def _caso(posicion: int, regla: str) -> dict[str, Any]:
        return {"posicion": int(posicion), "fecha_hora": str(momentos[posicion]), "regla": regla,
                "demanda_real_mwh": float(real[posicion])}

    casos: dict[str, dict[str, Any]] = {}
    pico = np.flatnonzero(laborable & ~especial & (hora == HORA_CASOS))
    if len(pico):
        mediana = pico[np.argsort(real[pico])[len(pico) // 2]]
        casos["pico_laboral"] = _caso(mediana, f"dia laborable sin marca de calendario, {HORA_CASOS}:00 "
                                               f"(hora pico {list(HORAS_PICO)}), demanda mediana entre esas horas")
    festivo = np.flatnonzero(marca["es_festivo"] & ~marca["es_semana_santa"] & (hora == HORA_CASOS))
    if len(festivo):
        casos["festivo"] = _caso(festivo[0], f"primer festivo de validacion fuera de Semana Santa, {HORA_CASOS}:00")
    santa = np.flatnonzero(marca["es_semana_santa"] & ~marca["es_festivo"] & laborable & (hora == HORA_CASOS))
    if len(santa):
        casos["semana_santa"] = _caso(santa[0], f"primer dia laborable no festivo de Semana Santa, {HORA_CASOS}:00")
    for rol, prediccion in predicciones.items():
        error = np.abs(np.asarray(prediccion, dtype="float64") - real)
        caso = _caso(int(np.argmax(error)), f"hora de validacion con mayor error absoluto del modelo {rol}")
        caso["solo_para"] = rol
        casos[f"mayor_error_{rol}"] = caso
    return casos


def explicador_lime(X_entrena: pd.DataFrame):
    """LimeTabularExplainer de regresion con las estadisticas del entrenamiento; las binarias, categoricas."""
    from lime.lime_tabular import LimeTabularExplainer

    variables = list(X_entrena.columns)
    binarias = [i for i, c in enumerate(variables) if set(np.unique(X_entrena[c])) <= {0.0, 1.0}]
    return LimeTabularExplainer(
        X_entrena.to_numpy(dtype="float64"),
        mode="regression",
        feature_names=variables,
        categorical_features=binarias,
        discretize_continuous=True,
        random_state=SEMILLA,
    ), binarias


def explicar_lime(explicador, modelo, X_valida: pd.DataFrame, caso: dict[str, Any]) -> dict[str, Any]:
    variables = list(X_valida.columns)

    def predecir(filas: np.ndarray) -> np.ndarray:
        return np.asarray(modelo.predict(pd.DataFrame(filas, columns=variables)), dtype="float64")

    fila = X_valida.iloc[caso["posicion"]].to_numpy(dtype="float64")
    explicacion = explicador.explain_instance(fila, predecir, num_features=VARIABLES_LIME, num_samples=MUESTRAS_LIME)
    intercepto = explicacion.intercept
    if isinstance(intercepto, dict):  # lime lo guarda por etiqueta
        intercepto = next(iter(intercepto.values()))
    return {
        "prediccion_mwh": float(predecir(fila[None, :])[0]),
        "intercepto_local_mwh": float(np.ravel(intercepto)[0]),
        "prediccion_local_mwh": float(np.ravel(explicacion.local_pred)[0]),
        "r2_local": float(explicacion.score),
        "pesos": [{"regla": regla, "peso_mwh": float(peso)} for regla, peso in explicacion.as_list()],
    }


# --------------------------------------------------------------------------
# Contraste con el EDA
# --------------------------------------------------------------------------


def _puesto(ranking: list[str], variable: str) -> int | None:
    return ranking.index(variable) + 1 if variable in ranking else None


def contraste_eda(medidas: dict[str, dict[str, list[str]]], acf: dict[str, float]) -> dict[str, Any]:
    """¿Domina lag_168 las importancias, como sugiere su autocorrelacion mayor que la de 24 h?

    `medidas`: {rol: {metodo: variables de mas a menos importante}}.
    """
    filas, primeras = [], 0
    for rol, metodos in medidas.items():
        for metodo, ranking in metodos.items():
            puesto = _puesto(ranking, COLUMNA_ESTACIONAL)
            puesto_24 = _puesto(ranking, "lag_24")
            primeras += puesto == 1
            filas.append({
                "modelo": rol,
                "medida": metodo,
                "primera": ranking[0],
                "tres_primeras": ranking[:3],
                "puesto_lag_168": puesto,
                "puesto_lag_24": puesto_24,
                "lag_168_antes_que_lag_24": None if puesto_24 is None or puesto is None else puesto < puesto_24,
            })
    total = len(filas)
    return {
        "acf_24": acf.get("24"),
        "acf_168": acf.get("168"),
        "hipotesis": "si la autocorrelacion a 168 h es mayor que a 24 h, lag_168 deberia dominar",
        "medidas": filas,
        "lag_168_primera_en": primeras,
        "de": total,
        "lag_168_domina": primeras == total,
        "conclusion": (
            f"lag_168 es la variable mas importante en {primeras} de {total} medidas"
            + ("." if primeras == total else
               "; donde no lo es, la primera es " + ", ".join(sorted({f['primera'] for f in filas if f['puesto_lag_168'] != 1})) + ".")
        ),
    }


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------


def figura_importancias(registro: dict[str, Any], clave: str, ruta: Path) -> Path:
    from mineria import estilo

    estilo.aplicar()
    paneles = [("ensemble", "impureza", "impureza (fraccion)"),
               ("ensemble", "permutacion", "permutacion: +MAE (MWh)"),
               ("lineal", "permutacion", "permutacion: +MAE (MWh)")]
    figura, ejes = estilo.plt.subplots(1, 3, figsize=(11.5, 4.6))
    for eje, (rol, metodo, etiqueta) in zip(ejes, paneles):
        datos = registro[rol]["importancias"][metodo]["por_variable"]
        nombres = list(datos)[::-1]
        if metodo == "impureza":
            valores, errores = [datos[n] for n in nombres], None
        else:
            valores = [datos[n]["media"] for n in nombres]
            errores = [datos[n]["desviacion"] for n in nombres]
        colores = [estilo.CATEGORICA[1] if n == COLUMNA_ESTACIONAL else estilo.CATEGORICA[0] for n in nombres]
        eje.barh(nombres, valores, xerr=errores, color=colores, error_kw={"linewidth": 0.8})
        eje.set(xlabel=etiqueta, title=f"{registro[rol]['candidato']['nombre']} · {metodo}")
        eje.tick_params(axis="y", labelsize=7.5)
    figura.suptitle(f"Importancia de variables en validacion · {clave} (lag_168 en naranja)", fontsize=10)
    figura.tight_layout()
    return estilo.guardar(figura, ruta)


def figura_shap_resumen(valores: np.ndarray, X: pd.DataFrame, base: float, titulo: str, ruta: Path) -> Path:
    import shap

    from mineria import estilo

    estilo.aplicar()
    explicacion = shap.Explanation(values=valores, base_values=np.full(len(X), base), data=X.to_numpy(),
                                   feature_names=list(X.columns))
    estilo.plt.figure(figsize=(7.5, 5.5))
    shap.plots.beeswarm(explicacion, max_display=len(X.columns), show=False)
    figura = estilo.plt.gcf()
    figura.suptitle(titulo, fontsize=10)
    figura.tight_layout()
    return estilo.guardar(figura, ruta)


def figura_shap_temperatura(paneles: dict[str, tuple[np.ndarray, pd.DataFrame, np.ndarray, str]], clave: str,
                            ruta: Path) -> Path:
    from mineria import estilo

    estilo.aplicar()
    figura, ejes = estilo.plt.subplots(1, len(paneles), figsize=(5.2 * len(paneles), 4.2), squeeze=False)
    for eje, (rol, (valores, X, horas, nombre)) in zip(ejes.ravel(), paneles.items()):
        posicion = list(X.columns).index("temperatura_c")
        puntos = eje.scatter(X["temperatura_c"], valores[:, posicion], c=horas, cmap="twilight", vmin=0, vmax=24,
                             s=6, alpha=0.7, linewidths=0)
        eje.axhline(0, color=estilo.TINTA_SECUNDARIA, linewidth=0.7)
        eje.set(xlabel="temperatura_c (°C)", ylabel="SHAP de temperatura_c (MWh)", title=f"{rol}: {nombre}")
        figura.colorbar(puntos, ax=eje, label="hora del dia")
    figura.suptitle(f"Dependencia SHAP de la temperatura, color por hora · {clave}", fontsize=10)
    figura.tight_layout()
    return estilo.guardar(figura, ruta)


def figura_lime(lime: dict[str, Any], clave: str, ruta: Path) -> Path:
    from mineria import estilo

    estilo.aplicar()
    # Una fila por caso; los dos "mayor error" (uno por modelo) comparten la ultima.
    filas: list[tuple[str, dict[str, Any]]] = []
    mayor_error: dict[str, Any] = {}
    for caso, por_rol in lime.items():
        if caso.startswith("mayor_error_"):
            mayor_error.update(por_rol)
        else:
            filas.append((caso, por_rol))
    if mayor_error:
        filas.append(("mayor_error", mayor_error))
    figura, ejes = estilo.plt.subplots(len(filas), 2, figsize=(11.5, 2.5 * len(filas)), squeeze=False)
    roles = ["ensemble", "lineal"]
    for i, (caso, por_rol) in enumerate(filas):
        for j, rol in enumerate(roles):
            eje = ejes[i, j]
            if rol not in por_rol:
                eje.set_visible(False)
                continue
            e = por_rol[rol]
            pesos = e["pesos"][::-1]
            eje.barh([p["regla"] for p in pesos], [p["peso_mwh"] for p in pesos],
                     color=[estilo.CATEGORICA[0] if p["peso_mwh"] >= 0 else estilo.CATEGORICA[1] for p in pesos])
            eje.axvline(0, color=estilo.TINTA_SECUNDARIA, linewidth=0.7)
            eje.tick_params(axis="y", labelsize=7)
            eje.set_title(f"{caso} · {rol} · {e['fecha_hora'][:16]} · real {e['demanda_real_mwh']:.0f}, "
                          f"pred {e['prediccion_mwh']:.0f} MWh · R² local {e['r2_local']:.2f}", fontsize=8)
            eje.set_xlabel("peso LIME (MWh)", fontsize=8)
    figura.suptitle(f"LIME en cuatro casos de validacion · {clave}", fontsize=10)
    figura.tight_layout()
    return estilo.guardar(figura, ruta)


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------


def interpretar(
    modelos: dict[str, Any],
    candidatos: dict[str, Candidato],
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    indices: dict[str, np.ndarray],
    acf: dict[str, float],
    clave: str,
    dir_figuras: Path,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Importancias, SHAP, LIME, contraste con el EDA y calificacion, con sus figuras."""
    entrena, valida = indices["entrenamiento"], indices["validacion"]
    if len(valida) == 0 or entrena.max() >= valida.min():
        raise ErrorInterpretabilidad("El entrenamiento tiene que ser anterior a la validacion.")
    X_entrena = X.iloc[entrena].astype("float64")
    X_valida = X.iloc[valida].astype("float64")
    y_valida = y.iloc[valida]
    fechas_valida = fechas.iloc[valida]
    horas_valida = _hora(fechas_valida)

    registro: dict[str, Any] = {}
    shap_paneles: dict[str, tuple] = {}
    predicciones: dict[str, np.ndarray] = {}
    medidas: dict[str, dict[str, list[str]]] = {}
    for rol in ("ensemble", "lineal"):
        modelo, candidato = modelos[rol], candidatos[rol]
        predicciones[rol] = np.asarray(modelo.predict(X_valida), dtype="float64")
        importancias: dict[str, Any] = {"permutacion": importancia_permutacion(modelo, X_valida, y_valida, n_jobs)}
        if rol == "ensemble":
            importancias["impureza"] = importancia_impureza(modelo, list(X.columns))
        posiciones, valores, base, registro_shap = explicar_shap(rol, modelo, X_valida, X_entrena)
        X_explicada = X_valida.iloc[posiciones]
        registro_shap["temperatura"] = resumen_temperatura(valores, X_explicada, horas_valida[posiciones])
        nombre = candidato.nombre if candidato.forma == "-" else f"{candidato.nombre} ({candidato.forma})"
        figura_resumen = figura_shap_resumen(
            valores, X_explicada, base,
            f"SHAP · {nombre} · {clave} · {len(posiciones)} de {len(X_valida)} filas de validacion",
            dir_figuras / f"shap_resumen_{rol}_{clave}.png",
        )
        registro_shap["figura_resumen"] = figura_resumen
        shap_paneles[rol] = (valores, X_explicada, horas_valida[posiciones], nombre)

        medidas[rol] = {metodo: list(datos["por_variable"]) for metodo, datos in importancias.items()}
        medidas[rol]["shap"] = list(registro_shap["media_abs_por_variable_mwh"])
        registro[rol] = {
            "candidato": asdict(candidato),
            "calificacion": {**CALIFICACIONES[candidato.nombre],
                             **({"nota": NOTA_DIFERENCIADO} if candidato.forma == "diferenciado" else {})},
            "importancias": importancias,
            "shap": registro_shap,
        }

    figura_temperatura = figura_shap_temperatura(shap_paneles, clave, dir_figuras / f"shap_temperatura_{clave}.png")
    figura_imp = figura_importancias(registro, clave, dir_figuras / f"importancias_{clave}.png")

    casos = casos_lime(fechas_valida, X_valida, y_valida, predicciones)
    explicador, binarias = explicador_lime(X_entrena)
    lime: dict[str, Any] = {}
    inicio = time.perf_counter()
    for nombre_caso, caso in casos.items():
        lime[nombre_caso] = {}
        for rol in ("ensemble", "lineal"):
            if caso.get("solo_para", rol) != rol:
                continue
            lime[nombre_caso][rol] = {**caso, **explicar_lime(explicador, modelos[rol], X_valida, caso)}
    segundos_lime = time.perf_counter() - inicio
    figura_casos = figura_lime(lime, clave, dir_figuras / f"lime_{clave}.png")

    return {
        "bloque": "validacion",
        "motivos_diferencia_impureza_permutacion": MOTIVOS_DIFERENCIA,
        "modelos": registro,
        "lime": {
            "configuracion": {
                "explicador": "lime.lime_tabular.LimeTabularExplainer(mode='regression', discretize_continuous=True)",
                "estadisticas": "bloque de entrenamiento",
                "categoricas": [X.columns[i] for i in binarias],
                "num_samples": MUESTRAS_LIME,
                "num_features": VARIABLES_LIME,
                "semilla": SEMILLA,
                "segundos": segundos_lime,
            },
            "casos": lime,
        },
        "contraste_eda": contraste_eda(medidas, acf),
        "figuras": {"importancias": figura_imp, "shap_temperatura": figura_temperatura, "lime": figura_casos},
    }


__all__ = [
    "CALIFICACIONES",
    "Candidato",
    "cargar",
    "casos_lime",
    "contraste_eda",
    "explicar_lime",
    "explicar_shap",
    "importancia_impureza",
    "importancia_permutacion",
    "interpretar",
    "seleccionar_candidatos",
]
