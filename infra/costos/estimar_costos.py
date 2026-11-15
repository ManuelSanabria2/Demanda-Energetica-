"""Estimacion preliminar de costos mensuales de la arquitectura en AWS.

    python infra/costos/estimar_costos.py

Salidas (versionadas): infra/costos/costos.json, infra/costos/costos.md y
docs/figuras/cloud/costos.png.

Todo supuesto esta en este archivo y es un numero que se puede cambiar: precios
unitarios (us-east-1, lista publica de AWS consultada en noviembre de 2026) y
volumenes de uso derivados de lo que el pipeline hace hoy en local. Se calculan
tres escenarios:

- `academico`: el uso previsto, aplicando la capa gratuita permanente de AWS.
- `sin_capa_gratuita`: el mismo uso, pagando todo. Es la cota honesta para una
  cuenta que ya consume su capa gratuita en otros proyectos.
- `operacion`: reentrenamiento diario, dos entornos (dev + prod), mas consultas
  y retencion de logs de un ano. Orden de magnitud de una operacion real.

Lo que NO se incluye: QuickSight u otra herramienta de BI, soporte de AWS,
dominio, y el costo de personas. Ese ultimo entra en la evaluacion economica.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
TRM = 3297.07            # COP por USD, TRM del 2026-11-11 (Superfinanciera)
FECHA_PRECIOS = "2026-11"
HORAS_MES = 730

# Precios unitarios, us-east-1 (USD). Fuentes: paginas de precios de AWS
# (aws.amazon.com/{lambda,fargate,s3,athena,cloudwatch,step-functions,ecr}/pricing).
PRECIOS = {
    "lambda_gb_s": 0.0000166667,
    "lambda_millon_solicitudes": 0.20,
    "fargate_vcpu_h": 0.04048,
    "fargate_gb_h": 0.004445,
    "fargate_spot_vcpu_h": 0.0124419,
    "fargate_spot_gb_h": 0.00136621,
    "s3_gb_mes": 0.023,
    "s3_mil_put": 0.005,
    "s3_mil_get": 0.0004,
    "ecr_gb_mes": 0.10,
    "sfn_mil_transiciones": 0.025,
    "cw_logs_gb_ingesta": 0.50,
    "cw_logs_gb_mes": 0.03,
    "cw_alarma_mes": 0.10,
    "cw_metrica_mes": 0.30,
    "athena_tb": 5.0,
    "xray_millon_trazas": 5.0,
    "nat_gateway_h": 0.045,
}

# Capa gratuita permanente (no la de 12 meses).
GRATIS = {
    "lambda_gb_s": 400_000,
    "lambda_solicitudes": 1_000_000,
    "sfn_transiciones": 4_000,
    "cw_alarmas": 10,
    "cw_metricas": 10,
    "cw_logs_gb": 5,
    "xray_trazas": 100_000,
}


@dataclass
class Uso:
    ejecuciones_diarias_mes: float = 30
    reentrenamientos_mes: float = 4.35
    lambda_segundos: float = 120          # ingesta incremental: ~73 s sin cache, ~2 s con cache (README)
    lambda_gb: float = 1.0
    procesar_min: float = 8                # panel + variables: segundos en local; holgura por arranque de contenedor
    procesar_vcpu: float = 1
    procesar_gb: float = 4
    predecir_min: float = 4
    predecir_vcpu: float = 0.5
    predecir_gb: float = 2
    entrenar_min: float = 45               # escalera de modelos con walk-forward
    entrenar_vcpu: float = 2
    entrenar_gb: float = 8
    spot: bool = True
    s3_gb: float = 6                       # raw + staging + processed + modelos, con versiones
    s3_put_por_ejecucion: float = 250
    s3_get_por_ejecucion: float = 900
    ecr_gb: float = 4                      # ~1.3 GB por imagen, se conservan hasta 10
    transiciones_diario: float = 8
    transiciones_semanal: float = 6
    logs_gb_mes: float = 0.4
    alarmas: int = 6
    metricas: int = 3
    consultas_athena_mes: float = 150
    mb_por_consulta: float = 25            # Parquet particionado: minimo facturable 10 MB
    entornos: int = 1
    capa_gratuita: bool = True


ESCENARIOS = {
    "academico": Uso(),
    "sin_capa_gratuita": Uso(capa_gratuita=False),
    "operacion": Uso(reentrenamientos_mes=30, entornos=2, capa_gratuita=False, logs_gb_mes=1.5,
                     consultas_athena_mes=1500, s3_gb=20, spot=False),
}


def _gratis(cantidad: float, clave: str, uso: Uso) -> float:
    return max(0.0, cantidad - GRATIS[clave]) if uso.capa_gratuita else cantidad


def costos(uso: Uso) -> dict[str, float]:
    p = PRECIOS
    vcpu_h = p["fargate_spot_vcpu_h"] if uso.spot else p["fargate_vcpu_h"]
    gb_h = p["fargate_spot_gb_h"] if uso.spot else p["fargate_gb_h"]

    def tarea(minutos: float, vcpu: float, gb: float, veces: float) -> float:
        horas = minutos / 60 * veces
        return horas * (vcpu * vcpu_h + gb * gb_h)

    ejecuciones = uso.ejecuciones_diarias_mes
    gb_s = ejecuciones * uso.lambda_segundos * uso.lambda_gb
    total_ejecuciones = ejecuciones + uso.reentrenamientos_mes
    transiciones = ejecuciones * uso.transiciones_diario + uso.reentrenamientos_mes * uso.transiciones_semanal

    detalle = {
        "Lambda (ingesta)": _gratis(gb_s, "lambda_gb_s", uso) * p["lambda_gb_s"]
        + _gratis(ejecuciones, "lambda_solicitudes", uso) / 1e6 * p["lambda_millon_solicitudes"],
        "ECS Fargate (procesar, predecir, entrenar)": tarea(uso.procesar_min, uso.procesar_vcpu, uso.procesar_gb, total_ejecuciones)
        + tarea(uso.predecir_min, uso.predecir_vcpu, uso.predecir_gb, ejecuciones)
        + tarea(uso.entrenar_min, uso.entrenar_vcpu, uso.entrenar_gb, uso.reentrenamientos_mes),
        "S3 (almacenamiento y solicitudes)": uso.s3_gb * p["s3_gb_mes"]
        + total_ejecuciones * (uso.s3_put_por_ejecucion / 1000 * p["s3_mil_put"] + uso.s3_get_por_ejecucion / 1000 * p["s3_mil_get"]),
        "ECR (imagenes)": uso.ecr_gb * p["ecr_gb_mes"],
        "Step Functions": _gratis(transiciones, "sfn_transiciones", uso) / 1000 * p["sfn_mil_transiciones"],
        "CloudWatch (logs, alarmas, metricas)": _gratis(uso.logs_gb_mes, "cw_logs_gb", uso) * p["cw_logs_gb_ingesta"]
        + uso.logs_gb_mes * (12 if uso.entornos > 1 else 1) * p["cw_logs_gb_mes"]
        + _gratis(uso.alarmas, "cw_alarmas", uso) * p["cw_alarma_mes"]
        + _gratis(uso.metricas, "cw_metricas", uso) * p["cw_metrica_mes"],
        "Athena + Glue Data Catalog": uso.consultas_athena_mes * max(uso.mb_por_consulta, 10) / 1e6 * p["athena_tb"],
        "X-Ray": _gratis(total_ejecuciones * 10, "xray_trazas", uso) / 1e6 * p["xray_millon_trazas"],
        "EventBridge Scheduler, SNS, Budgets": 0.0,
    }
    return {k: round(v * uso.entornos, 4) for k, v in detalle.items()}


def main() -> int:
    resultados = {}
    for nombre, uso in ESCENARIOS.items():
        detalle = costos(uso)
        total = round(sum(detalle.values()), 2)
        resultados[nombre] = {
            "detalle_usd_mes": detalle,
            "total_usd_mes": total,
            "total_usd_anio": round(12 * total, 2),
            "total_cop_mes": round(total * TRM),
            "total_cop_anio": round(12 * total * TRM),
            "supuestos": uso.__dict__,
        }
    alternativa_nat = round(HORAS_MES * PRECIOS["nat_gateway_h"], 2)
    salida = {
        "region": "us-east-1",
        "precios_consultados": FECHA_PRECIOS,
        "trm_cop_usd": TRM,
        "precios_unitarios_usd": PRECIOS,
        "capa_gratuita_permanente": GRATIS,
        "escenarios": resultados,
        "decision_sin_nat_gateway_ahorro_usd_mes": alternativa_nat,
        "nota": (
            "Estimacion preliminar. Validar con AWS Pricing Calculator antes de desplegar; "
            "sa-east-1 (Sao Paulo) cuesta tipicamente 30-50 % mas que us-east-1."
        ),
    }
    carpeta = RAIZ / "infra" / "costos"
    (carpeta / "costos.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    filas = list(resultados["academico"]["detalle_usd_mes"])
    lineas = [
        f"# Estimacion preliminar de costos (us-east-1, precios de {FECHA_PRECIOS})",
        "",
        f"TRM {TRM:,.2f} COP/USD. Generado por `infra/costos/estimar_costos.py`.",
        "",
        "| Servicio | Academico (USD/mes) | Sin capa gratuita | Operacion (2 entornos) |",
        "|---|---:|---:|---:|",
    ]
    for fila in filas:
        valores = [resultados[e]["detalle_usd_mes"][fila] for e in ESCENARIOS]
        lineas.append(f"| {fila} | " + " | ".join(f"{v:.2f}" for v in valores) + " |")
    lineas.append("| **Total USD/mes** | " + " | ".join(f"**{resultados[e]['total_usd_mes']:.2f}**" for e in ESCENARIOS) + " |")
    lineas.append("| Total COP/mes | " + " | ".join(f"{resultados[e]['total_cop_mes']:,}" for e in ESCENARIOS) + " |")
    lineas.append("| Total USD/ano | " + " | ".join(f"{resultados[e]['total_usd_anio']:.2f}" for e in ESCENARIOS) + " |")
    lineas += [
        "",
        f"Decision de diseno: sin NAT Gateway se evitan {alternativa_nat} USD/mes fijos, mas que todo el resto junto.",
        "",
        salida["nota"],
    ]
    (carpeta / "costos.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    import sys

    sys.path.insert(0, str(RAIZ / "src"))
    from mineria.estilo import CATEGORICA, aplicar, guardar, plt

    aplicar()
    figura, eje = plt.subplots(figsize=(7.2, 2.8))
    nombres = {"academico": "Academic (free tier)", "sin_capa_gratuita": "No free tier", "operacion": "Operation (2 envs, daily retrain)"}
    base = [0.0] * len(ESCENARIOS)
    for i, fila in enumerate(f for f in filas if any(resultados[e]["detalle_usd_mes"][f] > 0 for e in ESCENARIOS)):
        valores = [resultados[e]["detalle_usd_mes"][fila] for e in ESCENARIOS]
        eje.barh(list(nombres.values()), valores, left=base, color=CATEGORICA[i % 8], label=fila, height=0.55,
                 edgecolor="white", linewidth=1)
        base = [b + v for b, v in zip(base, valores)]
    for y, total in enumerate(base):
        eje.text(total + 0.3, y, f"{total:.2f} USD", va="center", fontsize=8)
    eje.set_xlabel("USD per month")
    eje.set_xlim(0, max(base) * 1.25)
    eje.invert_yaxis()
    eje.legend(fontsize=6.5, ncol=2, loc="lower right")
    eje.set_title("Preliminary monthly AWS cost by scenario (us-east-1)")
    guardar(figura, RAIZ / "docs" / "figuras" / "cloud" / "costos.png")

    for e in ESCENARIOS:
        print(f"{e:20s} {resultados[e]['total_usd_mes']:7.2f} USD/mes  {resultados[e]['total_cop_mes']:>10,} COP/mes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
