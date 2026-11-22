"""Macros y tabla del modelado supervisado para el informe final (en ingles).

Lo llama docs/informe/generar_valores.py. Todas las cifras salen de los JSON
que escriben scripts/entrenar.py y scripts/evaluar_final.py en docs/resultados;
si un resultado falta, su macro queda como \\pendiente.

Los nombres de macro de LaTeX no admiten digitos: cada tabla de modelado lleva
su sufijo en numeros romanos, r24 -> Rxxiv y r120 -> Rcxx
(p. ej. \\modFinalMapeRxxiv, \\modFinalMapeRcxx).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

SUFIJOS = {"r24": "Rxxiv", "r120": "Rcxx"}
ARCHIVOS = {
    "baseline": "modelado_baselines.json",
    "lineal": "modelado_regresion.json",
    "clasificacion": "modelado_clasificacion.json",
    "ensembles": "modelado_ensembles.json",
    "busqueda": "modelado_busqueda.json",
    "interpretabilidad": "modelado_interpretabilidad.json",
    "seleccion": "modelado_seleccion.json",
}
NOMBRES_EN = {
    "estacional": "Seasonal naive", "diario": "Daily naive",
    "lineal": "Linear regression", "polinomial": "Polynomial OLS", "polinomial_ridge": "Polynomial Ridge",
    "ridge": "Ridge", "lasso": "Lasso", "elasticnet": "Elastic Net",
    "random_forest": "Random Forest", "adaboost": "AdaBoost", "gradient_boosting": "Gradient Boosting",
    "hist_gradient": "HistGradientBoosting",
    "logistica": "Logistic regression", "knn": "k-NN", "svm_lineal": "Linear SVM", "svm_rbf": "RBF SVM",
    "baseline": "Seasonal-lag baseline",
}
FORMAS_EN = {"nivel": "level", "diferenciado": "differenced"}
CALIFICACION_EN = {"alta": "high", "media": "medium", "baja": "low"}
ROLES_EN = {"baseline": "Baseline", "lineal": "Best linear", "ensemble": "Best ensemble"}


def _leer(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def nombre_en(nombre: str | None, forma: str | None = None) -> str:
    if not nombre:
        return r"\pendiente"
    texto = NOMBRES_EN.get(nombre, nombre.replace("_", r"\_"))
    return texto if forma in (None, "-") else f"{texto} ({FORMAS_EN.get(forma, forma)})"


def _get(objeto: Any, *claves: Any) -> Any:
    for clave in claves:
        if not isinstance(objeto, dict) or clave not in objeto:
            return None
        objeto = objeto[clave]
    return objeto


def macros_modelado(
    resultados: Path, num: Callable[..., str]
) -> tuple[dict[str, str], dict[str, str]]:
    """({macro: valor}, {archivo .tex: contenido}) del modelado. `num` formatea cifras (y \\pendiente)."""
    r = {clave: _leer(resultados / archivo) for clave, archivo in ARCHIVOS.items()}
    pendiente = r"\pendiente"
    m: dict[str, str] = {}

    seleccion = r["seleccion"]
    bloque = seleccion.get("bloque_prueba") or [None, None]
    m["modInicioPrueba"] = bloque[0][:10] if bloque[0] else pendiente
    m["modFinPrueba"] = bloque[1][:10] if bloque[1] else pendiente
    m["modCriterioMejora"] = num(_get(seleccion, "criterio_exito", "mejora_mape_minima_pct"))

    for clave, s in SUFIJOS.items():
        # --- particion (del registro de la evaluacion final, que la recalcula igual)
        bloques = _get(seleccion, "tablas", clave, "particion", "detalle", "bloques") or {}
        for bloque_es, nombre in (("entrenamiento", "Entrena"), ("validacion", "Valida"), ("prueba", "Prueba")):
            m[f"modFilas{nombre}{s}"] = num(_get(bloques, bloque_es, "filas"))
        m[f"modGap{s}"] = num(_get(r["lineal"], "tablas", clave, "validacion_cruzada", "detalle", "gap_h"))

        # --- baselines (validacion)
        base = _get(r["baseline"], "tablas", clave, "modelos") or {}
        m[f"modBaseMapeVal{s}"] = num(_get(base, "estacional", "metricas_validacion", "mape_pct"), 2)
        m[f"modBasePicoVal{s}"] = num(_get(base, "estacional", "metricas_validacion", "mape_pico_pct"), 2)

        # --- lineales (validacion)
        lineales = _get(r["lineal"], "tablas", clave, "modelos") or {}
        if lineales:
            mejor = min(lineales, key=lambda n: lineales[n]["metricas_validacion"]["mape_pct"])
            m[f"modLinMejorVal{s}"] = nombre_en(mejor)
            m[f"modLinMapeVal{s}"] = num(lineales[mejor]["metricas_validacion"]["mape_pct"], 2)
            m[f"modLinMejoraVal{s}"] = num(lineales[mejor]["metricas_validacion"]["mejora_mape_pct"], 1)
        else:
            m[f"modLinMejorVal{s}"] = pendiente
            m[f"modLinMapeVal{s}"] = m[f"modLinMejoraVal{s}"] = num(None)
        m[f"modOlsMapeVal{s}"] = num(_get(lineales, "lineal", "metricas_validacion", "mape_pct"), 2)
        m[f"modPolRango{s}"] = num(_get(lineales, "polinomial", "condicionamiento", "rango"))
        m[f"modPolColumnas{s}"] = num(_get(lineales, "polinomial", "condicionamiento", "columnas"))
        m[f"modLassoNulos{s}"] = num(_get(lineales, "lasso", "coeficientes", "n_nulos"))

        # --- clasificacion (validacion)
        clasif = _get(r["clasificacion"], "tablas", clave) or {}
        modelos_clf = clasif.get("modelos") or {}
        if modelos_clf:
            mejor = max(modelos_clf, key=lambda n: modelos_clf[n]["metricas_validacion"]["pr_auc"])
            m[f"modClfMejorVal{s}"] = nombre_en(mejor)
            m[f"modClfPrAucVal{s}"] = num(modelos_clf[mejor]["metricas_validacion"]["pr_auc"], 3)
            m[f"modClfFUnoVal{s}"] = num(modelos_clf[mejor]["metricas_validacion"]["f1"], 3)
        else:
            m[f"modClfMejorVal{s}"] = pendiente
            m[f"modClfPrAucVal{s}"] = m[f"modClfFUnoVal{s}"] = num(None)
        m[f"modClfBasePrAucVal{s}"] = num(_get(clasif, "baseline", "metricas_validacion", "pr_auc"), 3)
        prevalencia = _get(clasif, "objetivo", "detalle", "por_bloque", "validacion", "proporcion_positivos")
        m[f"modClfPrevalenciaVal{s}"] = num(None if prevalencia is None else 100 * prevalencia, 1)

        # --- ensembles (validacion)
        ens = _get(r["ensembles"], "tablas", clave, "modelos") or {}
        filas = [(n, f, reg) for n, formas in ens.items() for f, reg in formas.items()]
        if filas:
            n, f, reg = min(filas, key=lambda fila: fila[2]["metricas_validacion"]["mape_pct"])
            m[f"modEnsMejorVal{s}"] = nombre_en(n, f)
            m[f"modEnsMapeVal{s}"] = num(reg["metricas_validacion"]["mape_pct"], 2)
            m[f"modEnsSobreMaxVal{s}"] = num(_get(reg, "extrapolacion_validacion", "reales_sobre_maximo"))
            m[f"modEnsMaxEntrena{s}"] = num(_get(reg, "extrapolacion_validacion", "maximo_entrenamiento_mwh"))
        else:
            m[f"modEnsMejorVal{s}"] = pendiente
            for nombre in ("MapeVal", "SobreMaxVal", "MaxEntrena"):
                m[f"modEns{nombre}{s}"] = num(None)

        # --- busqueda
        busq = _get(r["busqueda"], "tablas", clave, "modelos") or {}
        comparaciones = [reg["comparacion"] for reg in busq.values()]
        m[f"modBusqCandidatos{s}"] = num(len(busq) if busq else None)
        m[f"modBusqRandomAlcanza{s}"] = num(sum(c["random_alcanza_grid"] for c in comparaciones) if busq else None)
        menos = sum(1 for c in comparaciones if c["ajustes_cv_random_hasta_alcanzar"] is not None
                    and c["ajustes_cv_grid_hasta_su_mejor"] is not None
                    and c["ajustes_cv_random_hasta_alcanzar"] < c["ajustes_cv_grid_hasta_su_mejor"])
        m[f"modBusqRandomMenos{s}"] = num(menos if busq else None)
        m[f"modBusqAjustes{s}"] = num(_get(next(iter(busq.values()), {}), "grid", "n_ajustes_cv"))

        # --- interpretabilidad
        contraste = _get(r["interpretabilidad"], "tablas", clave, "contraste_eda") or {}
        m[f"modLagPrimera{s}"] = num(contraste.get("lag_168_primera_en"))
        m[f"modLagMedidas{s}"] = num(contraste.get("de"))
        m["modAcfDia"] = num(contraste.get("acf_24"), 3) if contraste else m.get("modAcfDia", num(None))
        m["modAcfSemana"] = num(contraste.get("acf_168"), 3) if contraste else m.get("modAcfSemana", num(None))

        # --- seleccion y prueba
        t = _get(seleccion, "tablas", clave) or {}
        final = t.get("final") or {}
        filas_sel = {f["rol"]: f for f in t.get("tabla_seleccion", [])}
        fila_final = next((f for f in t.get("tabla_seleccion", []) if f.get("es_final")), {})
        m[f"modFinal{s}"] = nombre_en(fila_final.get("nombre"), fila_final.get("forma")) if fila_final else pendiente
        m[f"modFinalInterp{s}"] = CALIFICACION_EN.get(_get(fila_final, "interpretabilidad", "calificacion"), pendiente)
        m[f"modFinalMape{s}"] = num(_get(final, "metricas_prueba", "mape_pct"), 2)
        m[f"modFinalPico{s}"] = num(_get(final, "metricas_prueba", "mape_pico_pct"), 2)
        m[f"modFinalMae{s}"] = num(_get(final, "metricas_prueba", "mae_mwh"))
        m[f"modFinalSesgo{s}"] = num(_get(final, "metricas_prueba", "sesgo_medio_mwh"))
        m[f"modFinalMejora{s}"] = num(_get(final, "criterio_exito_prueba", "mejora_mape_pct"), 1)
        cumple = _get(final, "criterio_exito_prueba", "cumple")
        m[f"modFinalCumple{s}"] = pendiente if cumple is None else ("meets" if cumple else "does not meet")
        for rol, nombre in (("baseline", "Base"), ("lineal", "Lin"), ("ensemble", "Ens")):
            m[f"mod{nombre}MapePrueba{s}"] = num(_get(t, "metricas_prueba", rol, "mape_pct"), 2)
            m[f"mod{nombre}PicoPrueba{s}"] = num(_get(t, "metricas_prueba", rol, "mape_pico_pct"), 2)
        m[f"modLinPrueba{s}"] = nombre_en(_get(filas_sel, "lineal", "nombre"))
        m[f"modLinMejoraPrueba{s}"] = num(_get(t, "metricas_prueba", "lineal", "mejora_mape_pct"), 1)
        clf = t.get("clasificador_prueba") or {}
        m[f"modClfPrueba{s}"] = nombre_en(_get(filas_sel, "clasificador", "nombre"))
        m[f"modClfPrAucPrueba{s}"] = num(clf.get("pr_auc"), 3)
        m[f"modClfFUnoPrueba{s}"] = num(clf.get("f1"), 3)
        m[f"modClfRecallPrueba{s}"] = num(clf.get("recall"), 3)
        m[f"modClfBrier{s}"] = num(clf.get("brier_probabilidad"), 4)
        extra = _get(t, "extrapolacion_prueba", fila_final.get("rol") if fila_final else None) or {}
        m[f"modSobreMaxPrueba{s}"] = num(extra.get("reales_sobre_maximo"))
        m[f"modSobreMaxSesgo{s}"] = num(extra.get("sesgo_medio_reales_sobre_maximo_mwh"))
        m[f"modSobreMaxPred{s}"] = num(extra.get("predicciones_sobre_maximo"))

    costo = _get(seleccion, "costo_rezago", "final") or {}
    m["modRezagoPuntos"] = num(costo.get("aumento_puntos"), 2)
    m["modRezagoPct"] = num(costo.get("aumento_relativo_pct"), 1)

    return m, {"tabla_seleccion.tex": tabla_seleccion(seleccion, num)}


def _kb(bytes_: float, num: Callable[..., str]) -> str:
    return "$<$1" if bytes_ < 1e3 else num(bytes_ / 1e3)


def tabla_seleccion(seleccion: dict, num: Callable[..., str]) -> str:
    """Tabla de seleccion en ingles: rendimiento en prueba, interpretabilidad y eficiencia."""
    lineas = [r"\begin{tabular}{@{}llrrrrlrr@{}}", r"\toprule",
              r"Lag & Model & Val. MAPE & Test MAPE & Peak MAPE & Impr. \% & Interp. & ms/1k & KB \\", r"\midrule"]
    tablas = seleccion.get("tablas") or {}
    for clave, t in tablas.items():
        for fila in t.get("tabla_seleccion", []):
            if fila["rol"] == "clasificador":
                continue
            p, v, e = fila["rendimiento"]["prueba"], fila["rendimiento"]["validacion"], fila["eficiencia"]
            modelo = nombre_en(fila["nombre"], fila["forma"])
            if fila["es_final"]:
                modelo = rf"\textbf{{{modelo}}}"
            lineas.append(
                f"{clave} & {modelo} & {num(v.get('mape_pct'), 2)} & {num(p.get('mape_pct'), 2)} & "
                f"{num(p.get('mape_pico_pct'), 2)} & {num(p.get('mejora_mape_pct'), 1) if p.get('mejora_mape_pct') is not None else '--'} & "
                f"{CALIFICACION_EN.get(fila['interpretabilidad']['calificacion'], '--')} & "
                f"{num(e['prediccion_ms_por_1000_filas'], 2)} & {_kb(e['tamano_bytes'], num)}" + r" \\"
            )
        if clave != list(tablas)[-1]:
            lineas.append(r"\midrule")
    if not tablas:
        lineas.append(r"\multicolumn{9}{c}{\pendiente} \\")
    lineas += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lineas) + "\n"


__all__ = ["SUFIJOS", "macros_modelado", "nombre_en", "tabla_seleccion"]
