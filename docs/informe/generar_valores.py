"""Vuelca los resultados (JSON) a macros y tablas de LaTeX para el informe IEEE.

    python docs/informe/generar_valores.py
    cd docs/informe && latexmk -pdf avance_proyecto_integrador.tex

Ningun numero del informe se escribe a mano: sale de docs/resultados/*.json,
de infra/costos/costos.json o de los registros versionados (data/procesado,
datasets/limpios). Si un resultado aun no existe (p. ej. no se ha ejecutado
scripts/eda.py con los datos reales), la macro queda como \\pendiente y el PDF lo
muestra en rojo: el informe nunca presenta como resultado algo que no se calculo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = Path(__file__).resolve().parent / "generado"
TIPOS_EN = {"laboral": "Weekday", "sabado": "Saturday", "domingo": "Sunday", "festivo": "Holiday"}
# Plural dentro de una frase: los dias de la semana van con mayuscula en ingles.
TIPOS_EN_PLURAL = {"laboral": "weekdays", "sabado": "Saturdays", "domingo": "Sundays", "festivo": "holidays"}


def _leer(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def _tex(texto: str) -> str:
    return (str(texto).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("$", r"\$"))


def _num(valor, decimales: int = 0) -> str:
    if valor is None:
        return r"\pendiente"
    texto = f"{abs(valor):,.{decimales}f}".replace(",", "{,}")
    # Signo menos tipografico (no guion), valido en texto y en modo matematico, y sin
    # punto de corte entre el signo y la cifra.
    return r"\ensuremath{-}" + texto if valor < 0 and float(texto.replace("{,}", "")) != 0 else texto


def _p(valor) -> str:
    if valor is None:
        return r"\pendiente"
    if valor == 0:
        return r"$<10^{-300}$"
    if valor < 1e-3:
        base, exp = f"{valor:.1e}".split("e")
        return rf"${base}\times10^{{{int(exp)}}}$"
    return f"{valor:.3f}"


def _interpretar_grupos(dias: dict) -> str:
    """Una frase por grupo, derivada de los datos (que tipos de dia concentra y su forma)."""
    if not dias.get("grupos"):
        return r"\pendiente{} (run \nolinkurl{scripts/mineria_no_supervisada.py} on the full data)."
    cont = dias.get("contingencia_tipo_dia", {})
    totales: dict[str, int] = {}
    for fila in cont.values():
        for tipo, n in fila.items():
            totales[tipo] = totales.get(tipo, 0) + n
    frases = []
    for grupo, g in dias["grupos"].items():
        fila = cont.get(grupo, {})
        capta = [f"{100 * fila.get(t, 0) / totales[t]:.0f}\\% of {TIPOS_EN_PLURAL[t]}"
                 for t in ("laboral", "sabado", "domingo", "festivo") if totales.get(t) and fila.get(t, 0) / totales[t] >= 0.3]
        especiales = []
        if g.get("pct_es_semana_santa", 0) >= 10:
            especiales.append(f"{g['pct_es_semana_santa']:.0f}\\% of its days are in Holy Week")
        if g.get("pct_es_ultima_semana_diciembre", 0) >= 10:
            especiales.append(f"{g['pct_es_ultima_semana_diciembre']:.0f}\\% fall on Dec 25--31")
        texto = (f"\\textbf{{{grupo}}} ({g['dias']:,} days, {g['pct_dias']:.1f}\\%) contains "
                 + (", ".join(capta) if capta else "a mix of day types")
                 + f"; its centroid peaks at {g['hora_pico']}~h with a valley at {100 * g['valle_sobre_pico']:.0f}\\% of the peak"
                 + (" and " + "; ".join(especiales) if especiales else "") + ".")
        frases.append(texto)
    return " ".join(frases)


def _interpretar_sectores(sect: dict) -> str:
    if not sect.get("grupos"):
        return (r"\pendiente{} (requires \nolinkurl{datasets/limpios/ciiu_limpio.parquet}, produced by the CIIU "
                r"cleaning notebook).")
    frases = []
    for grupo, g in sect["grupos"].items():
        tipo = ("continuous-process" if g["domingo_sobre_laboral"] > 0.9 and g["noche_sobre_dia"] > 0.85
                else "working-hours" if g["domingo_sobre_laboral"] < 0.75 or g["noche_sobre_dia"] < 0.7 else "intermediate")
        frases.append(
            f"\\textbf{{{grupo}}} ({g['sectores']} sectors, {g['pct_energia']:.1f}\\% of the energy) has "
            f"{'an' if tipo[0] in 'aeiou' else 'a'} {tipo} profile "
            f"(Sunday/weekday {g['domingo_sobre_laboral']:.2f}, night/day {g['noche_sobre_dia']:.2f}); "
            f"{g['pct_energia_manufactura']:.0f}\\% of its energy is manufacturing."
        )
    return " ".join(frases)


def main() -> int:
    SALIDA.mkdir(parents=True, exist_ok=True)
    eda = _leer(RAIZ / "docs" / "resultados" / "eda.json")
    min_ = _leer(RAIZ / "docs" / "resultados" / "mineria.json")
    costos = _leer(RAIZ / "infra" / "costos" / "costos.json")
    variables = _leer(RAIZ / "data" / "procesado" / "registro_variables.json")
    ciiu = _leer(RAIZ / "datasets" / "limpios" / "registro_ciiu.json")

    m: dict[str, str] = {}
    hay_eda = bool(eda)
    hay_min = bool(min_)
    m["hayEDA"] = "1" if hay_eda else "0"
    m["hayMineria"] = "1" if hay_min else "0"
    m["haySectores"] = "1" if hay_min and "k" in min_.get("sectores", {}) else "0"

    # --- registros versionados (siempre disponibles) ------------------------
    r24 = variables.get("versiones", {}).get("r24", {})
    r120 = variables.get("versiones", {}).get("r120", {})
    m["rVeinticuatroFilas"] = _num(r24.get("filas_finales"))
    m["rCientoVeinteFilas"] = _num(r120.get("filas_finales"))
    m["panelFilas"] = _num(variables.get("panel", {}).get("filas"))
    res = ciiu.get("resumen", {})
    m["ciiuFilas"] = _num(res.get("filas_iniciales"))
    m["ciiuGrupos"] = _num(res.get("n_grupos"))
    m["ciiuObservadoPct"] = _num(res.get("pct_observado"), 2)
    m["ciiuInterpolados"] = _num(res.get("interpolados"))
    m["ciiuFaltantes"] = _num(res.get("faltantes"))

    # --- EDA ----------------------------------------------------------------
    d = eda.get("descriptivos_mwh", {})
    m["edaHoras"] = _num(d.get("n"))
    m["edaDesde"] = _tex(d.get("rango", [r"\pendiente"])[0][:10]) if d else r"\pendiente"
    m["edaHasta"] = _tex(d.get("rango", ["", r"\pendiente"])[1][:10]) if d else r"\pendiente"
    m["edaMedia"] = _num(d.get("media"))
    m["edaDesv"] = _num(d.get("desviacion"))
    m["edaCV"] = _num(d.get("coef_variacion_pct"), 1)
    m["edaMin"] = _num(d.get("minimo"))
    m["edaMax"] = _num(d.get("maximo"))
    m["edaMediana"] = _num(d.get("mediana"))
    m["edaAsimetria"] = _num(d.get("asimetria"), 2)
    m["edaCurtosis"] = _num(d.get("curtosis_exceso"), 2)
    m["edaNulos"] = _num(d.get("nulos"))
    ph = eda.get("perfil_horario", {})
    m["edaHoraPico"] = _num(ph.get("hora_pico_mediana"))
    m["edaHoraValle"] = _num(ph.get("hora_valle_mediana"))
    m["edaValleSobrePico"] = _num(100 * ph["valle_sobre_pico"], 1) if ph else r"\pendiente"
    acf = eda.get("acf", {})
    for k, nombre in (("24", "edaAcfVeinticuatro"), ("168", "edaAcfSemana"), ("1", "edaAcfUno"), ("336", "edaAcfDosSemanas")):
        m[nombre] = _num(acf.get(k), 3)
    est = eda.get("estacionariedad", {})
    m["edaAdfP"] = _p(est.get("adf_p"))
    m["edaKpssP"] = _p(est.get("kpss_p"))
    m["edaAdfDiffP"] = _p(est.get("adf_diferencia_semanal_p"))
    stl = eda.get("stl", {})
    m["edaFuerzaTendencia"] = _num(stl.get("fuerza_tendencia"), 2)
    m["edaFuerzaSemanal"] = _num(stl.get("fuerza_estacional_semanal"), 2)
    cal = eda.get("calendario", {})
    m["edaKruskalH"] = _num(cal.get("kruskal_h"), 1)
    m["edaKruskalP"] = _p(cal.get("kruskal_p"))
    m["edaEpsilon"] = _num(cal.get("epsilon_cuadrado"), 2)
    m["edaFestivoPct"] = _num(cal.get("festivo_vs_laboral_pct"), 1)
    m["edaMannWhitneyP"] = _p(cal.get("mannwhitney_p"))
    m["edaRankBiserial"] = _num(cal.get("efecto_r_rank_biserial"), 2)
    for tipo, nombre in (("laboral", "Laboral"), ("sabado", "Sabado"), ("domingo", "Domingo"), ("festivo", "Festivo")):
        m[f"edaMediana{nombre}"] = _num(cal.get("medianas", {}).get(tipo), 3)
        m[f"edaN{nombre}"] = _num(cal.get("n", {}).get(tipo))
    t = eda.get("temperatura", {})
    m["edaRhoHorario"] = _num(t.get("spearman_horario"), 2)
    m["edaRhoAnomalia"] = _num(t.get("spearman_anomalia_diaria"), 2)
    m["edaRhoAnomaliaP"] = _p(t.get("spearman_anomalia_diaria_p"))
    por_hora = t.get("spearman_por_hora", {})
    if por_hora:
        m["edaRhoHoraMin"] = _num(min(por_hora.values()), 2)
        m["edaRhoHoraMax"] = _num(max(por_hora.values()), 2)
    else:
        m["edaRhoHoraMin"] = m["edaRhoHoraMax"] = r"\pendiente"
    m["edaAtipicos"] = _num(eda.get("atipicos_causales_horas"))
    crec = eda.get("crecimiento_anual", {})
    completos = {a: v for a, v in crec.items() if v.get("completo")}
    if len(completos) >= 2:
        anios = sorted(completos)
        primero, ultimo = completos[anios[0]], completos[anios[-1]]
        tasa = (ultimo["media_diaria_mwh"] / primero["media_diaria_mwh"]) ** (1 / (int(anios[-1]) - int(anios[0]))) - 1
        m["edaCrecimientoAnual"] = _num(100 * tasa, 2)
        m["edaAnioIni"], m["edaAnioFin"] = anios[0], anios[-1]
        m["edaEnergiaUltimo"] = _num(ultimo["energia_gwh"], 0)
    else:
        m["edaCrecimientoAnual"] = m["edaAnioIni"] = m["edaAnioFin"] = m["edaEnergiaUltimo"] = r"\pendiente"
    corr = eda.get("correlacion_con_objetivo_r24", {})
    for clave, nombre in (("lag_168", "edaCorrLagSemana"), ("lag_24", "edaCorrLagDia"), ("temperatura_c", "edaCorrTemp"),
                          ("ma_168", "edaCorrMedia"), ("hora_cos", "edaCorrHoraCos")):
        m[nombre] = _num(corr.get(clave), 2)

    # --- Mineria ------------------------------------------------------------
    dias = min_.get("dias", {})
    km = dias.get("kmeans", {})
    m["minDias"] = _num(dias.get("perfiles", {}).get("dias_resultantes"))
    m["minK"] = _num(dias.get("k"))
    m["minSilueta"] = _num(km.get("silueta"), 3)
    m["minAriWard"] = _num(km.get("ari_ward"), 3)
    m["minCramer"] = _num(dias.get("v_cramer_tipo_dia"), 3)
    m["minVarPca"] = _num(100 * sum(km["varianza_pca"]), 1) if km else r"\pendiente"
    sel = {f["k"]: f for f in dias.get("seleccion_k", [])}
    elegido = sel.get(dias.get("k"), {})
    m["minEstabilidad"] = _num(elegido.get("estabilidad_ari"), 3)
    m["minDaviesBouldin"] = _num(elegido.get("davies_bouldin"), 3)
    reglas = dias.get("reglas", {})
    m["minReglas"] = _num(reglas.get("n_reglas"))
    m["minTransacciones"] = _num(reglas.get("transacciones"))
    m["minReglasNoRedundantes"] = _num(reglas.get("n_reglas_no_redundantes"))
    sect = min_.get("sectores", {})
    m["secSectores"] = _num(sect.get("perfiles", {}).get("sectores_resultantes"))
    m["secK"] = _num(sect.get("k"))

    # --- Pruebas: se cuentan, no se escriben ------------------------------
    import subprocess
    import sys as _sys

    salida = subprocess.run([_sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"], cwd=RAIZ,
                            capture_output=True, text=True).stdout
    total = next((linea.split()[0] for linea in salida.splitlines() if " tests collected" in linea or " test collected" in linea), None)
    m["nPruebas"] = total or r"\pendiente"

    # --- Costos -------------------------------------------------------------
    esc = costos.get("escenarios", {})
    for clave, nombre in (("academico", "Academico"), ("sin_capa_gratuita", "SinGratis"), ("operacion", "Operacion")):
        e = esc.get(clave, {})
        m[f"costo{nombre}Usd"] = _num(e.get("total_usd_mes"), 2)
        m[f"costo{nombre}Cop"] = _num(e.get("total_cop_mes"))
        m[f"costo{nombre}Anual"] = _num(e.get("total_usd_anio"), 2)
    m["costoNat"] = _num(costos.get("decision_sin_nat_gateway_ahorro_usd_mes"), 2)
    m["trm"] = _num(costos.get("trm_cop_usd"), 2)

    # --- Modelado supervisado: de docs/resultados/modelado_*.json (ver valores_modelado.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from valores_modelado import macros_modelado

    macros_mod, tablas_mod = macros_modelado(RAIZ / "docs" / "resultados", _num)
    m.update(macros_mod)

    lineas = [r"% Generado por docs/informe/generar_valores.py. No editar a mano."]
    for nombre, valor in sorted(m.items()):
        lineas.append(rf"\newcommand{{\{nombre}}}{{{valor}}}")
    (SALIDA / "valores.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # --- Tablas ---------------------------------------------------------------
    tabla_costos = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
                    r"Service & Academic & No free tier & Operation \\", r"\midrule"]
    if esc:
        nombres_en = {
            "Lambda (ingesta)": "Lambda (ingestion)",
            "ECS Fargate (procesar, predecir, entrenar)": "ECS Fargate (3 tasks)",
            "S3 (almacenamiento y solicitudes)": "S3 (storage + requests)",
            "ECR (imagenes)": "ECR (images)",
            "Step Functions": "Step Functions",
            "CloudWatch (logs, alarmas, metricas)": "CloudWatch",
            "Athena + Glue Data Catalog": "Athena + Glue",
            "X-Ray": "X-Ray",
            "EventBridge Scheduler, SNS, Budgets": "Scheduler, SNS, Budgets",
        }
        for fila in esc["academico"]["detalle_usd_mes"]:
            valores = [esc[e]["detalle_usd_mes"][fila] for e in ("academico", "sin_capa_gratuita", "operacion")]
            tabla_costos.append(f"{nombres_en.get(fila, _tex(fila))} & " + " & ".join(f"{v:.2f}" for v in valores) + r" \\")
        tabla_costos.append(r"\midrule")
        tabla_costos.append(r"\textbf{Total (USD/month)} & " + " & ".join(
            rf"\textbf{{{esc[e]['total_usd_mes']:.2f}}}" for e in ("academico", "sin_capa_gratuita", "operacion")) + r" \\")
        tabla_costos.append(r"Total (COP/month) & " + " & ".join(
            _num(esc[e]["total_cop_mes"]) for e in ("academico", "sin_capa_gratuita", "operacion")) + r" \\")
    tabla_costos += [r"\bottomrule", r"\end{tabular}"]
    (SALIDA / "tabla_costos.tex").write_text("\n".join(tabla_costos) + "\n", encoding="utf-8")

    tabla_grupos = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
                    r"Cl. & Days & Wkd \% & Sat \% & Sun \% & Hol \% & Peak h & Valley/peak & Level (MWh) \\",
                    r"\midrule"]
    for grupo, g in dias.get("grupos", {}).items():
        tabla_grupos.append(
            f"{grupo} & {g['dias']:,} & {g['pct_laboral']:.1f} & {g['pct_sabado']:.1f} & {g['pct_domingo']:.1f} & "
            f"{g['pct_festivo']:.1f} & {g['hora_pico']} & {g['valle_sobre_pico']:.3f} & {g['nivel_medio_mwh']:,.0f}"
            .replace(",", "{,}") + r" \\"
        )
    if not dias.get("grupos"):
        tabla_grupos.append(r"\multicolumn{9}{c}{\pendiente} \\")
    tabla_grupos += [r"\bottomrule", r"\end{tabular}"]
    (SALIDA / "tabla_grupos.tex").write_text("\n".join(tabla_grupos) + "\n", encoding="utf-8")

    def _item(texto: str) -> str:
        reemplazos = {"dia=laboral": "weekday", "dia=sabado": "Saturday", "dia=domingo": "Sunday",
                      "dia=festivo": "holiday", "puente": "long weekend", "semana_santa": "Holy Week",
                      "fin_de_anio": "Dec 25--31", "vispera_festivo": "eve of holiday",
                      "nivel=alto": "high level", "nivel=bajo": "low level", "temp=calida": "warm", "temp=fria": "cool"}
        partes = [p.strip() for p in texto.split("&")]
        return r" $\wedge$ ".join(reemplazos.get(p, p.replace("grupo=", "")) for p in partes)

    tabla_reglas = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
                    r"Antecedent & Consequent & Supp. & Conf. & Lift \\", r"\midrule"]
    vistas = set()
    filas_reglas = reglas.get("top_hacia_grupo", [])[:6] + reglas.get("top_general", [])
    for r in filas_reglas:
        clave = (r["antecedente"], r["consecuente"])
        if clave in vistas:
            continue
        vistas.add(clave)
        tabla_reglas.append(f"{_item(r['antecedente'])} & {_item(r['consecuente'])} & {r['soporte']:.3f} & "
                            f"{r['confianza']:.2f} & {r['lift']:.2f}" + r" \\")
        if len(vistas) >= 9:
            break
    if not filas_reglas:
        tabla_reglas.append(r"\multicolumn{5}{c}{\pendiente} \\")
    tabla_reglas += [r"\bottomrule", r"\end{tabular}"]
    (SALIDA / "tabla_reglas.tex").write_text("\n".join(tabla_reglas) + "\n", encoding="utf-8")

    tabla_sect = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
                  r"Cl. & Sectors & Energy \% & Sun/wkd & Night/day & Load f. & Manuf. energy \% \\", r"\midrule"]
    for grupo, g in sect.get("grupos", {}).items():
        tabla_sect.append(f"{grupo} & {g['sectores']} & {g['pct_energia']:.1f} & {g['domingo_sobre_laboral']:.2f} & "
                          f"{g['noche_sobre_dia']:.2f} & {g['factor_carga']:.2f} & {g['pct_energia_manufactura']:.1f}" + r" \\")
    if not sect.get("grupos"):
        tabla_sect.append(r"\multicolumn{7}{c}{\pendiente} \\")
    tabla_sect += [r"\bottomrule", r"\end{tabular}"]
    (SALIDA / "tabla_sectores.tex").write_text("\n".join(tabla_sect) + "\n", encoding="utf-8")

    (SALIDA / "interpretacion_grupos.tex").write_text(_interpretar_grupos(dias) + "\n", encoding="utf-8")
    (SALIDA / "interpretacion_sectores.tex").write_text(_interpretar_sectores(sect) + "\n", encoding="utf-8")

    for nombre, contenido in tablas_mod.items():
        (SALIDA / nombre).write_text(contenido, encoding="utf-8")

    print(f"{len(m)} macros · EDA {'si' if hay_eda else 'NO'} · mineria {'si' if hay_min else 'NO'} -> {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
