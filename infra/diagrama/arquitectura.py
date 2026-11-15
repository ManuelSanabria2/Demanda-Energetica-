"""Diagrama formal de la arquitectura en AWS (iconos oficiales via `diagrams`).

    python infra/diagrama/arquitectura.py      # requiere graphviz (dot)

Salida: docs/figuras/cloud/arquitectura.png. El diagrama refleja exactamente los
recursos de infra/terraform; si cambia uno, hay que cambiar el otro.
"""

from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import Athena, Glue
from diagrams.aws.compute import ECR, ElasticContainerServiceContainer, Lambda
from diagrams.aws.integration import Eventbridge, SNS, StepFunctions
from diagrams.aws.management import Cloudwatch, CloudwatchAlarm
from diagrams.aws.cost import Budgets
from diagrams.aws.security import IAMRole
from diagrams.aws.storage import S3
from diagrams.onprem.client import Users
from diagrams.generic.network import Firewall
from diagrams.programming.language import Python

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "docs" / "figuras" / "cloud" / "arquitectura"

AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

# Disposicion compacta: la figura ocupa el ancho de la pagina del informe, asi que
# cuanto mas ancho que alto sea el grafo, mas grandes llegan los textos al PDF.
atributos = {
    "fontsize": "16",
    "fontname": "DejaVu Sans",
    "pad": "0.5",
    "nodesep": "0.35",
    "ranksep": "0.9",
    "splines": "spline",
    "bgcolor": "white",
}
CLUSTER = {"fontsize": "15"}

SALIDA.parent.mkdir(parents=True, exist_ok=True)
with Diagram(
    "Demand forecasting pipeline on AWS (us-east-1) - Terraform-managed",
    filename=str(SALIDA),
    outformat="png",
    show=False,
    direction="LR",
    graph_attr=atributos,
    node_attr={"fontsize": "14", "fontname": "DejaVu Sans"},
):
    with Cluster("External open data (public APIs)", graph_attr=CLUSTER):
        xm = Firewall("XM SINERGOX\nDemaReal (hourly)")
        simem = Firewall("SIMEM\n14fabb")
        clima = Firewall("Open-Meteo\narchive")

    with Cluster("Orchestration", graph_attr=CLUSTER):
        horario = Eventbridge("EventBridge Scheduler\ndaily 06:00 / weekly Mon\n(America/Bogota)")
        diario = StepFunctions("Step Functions\ndaily pipeline")
        semanal = StepFunctions("Step Functions\nweekly retraining")

    with Cluster("Compute (one container image)", graph_attr=CLUSTER):
        repo = ECR("ECR\npipeline image")
        ingesta = Lambda("Lambda\nincremental ingestion")
        with Cluster("ECS Fargate (Spot)", graph_attr=CLUSTER):
            procesar = ElasticContainerServiceContainer("procesar\nclean + panel + features")
            predecir = ElasticContainerServiceContainer("predecir\n24 h forecast")
            entrenar = ElasticContainerServiceContainer("entrenar\nmodel ladder,\nwalk-forward")

    with Cluster("S3 data lake (versioned, SSE, TLS-only)", graph_attr=CLUSTER):
        raw = S3("raw/ + staging/")
        procesado = S3("processed/\npanel, r24, r120")
        modelos = S3("models/")
        pronosticos = S3("predictions/")

    with Cluster("Consumption", graph_attr=CLUSTER):
        catalogo = Glue("Glue Data Catalog\n(partition projection)")
        athena = Athena("Athena workgroup\n1 GB/query cap")
        usuarios = Users("Planners / analysts")

    with Cluster("Monitoring & governance", graph_attr=CLUSTER):
        # Partidas en tres lineas: en dos invadian la etiqueta vecina.
        metricas = Cloudwatch("CloudWatch\nlogs + MAPE\nmetrics")
        alarmas = CloudwatchAlarm("Alarms: failures,\nno forecast,\nMAPE drift")
        aviso = SNS("SNS e-mail\n(also on any\npipeline failure)")
        presupuesto = Budgets("AWS Budgets\n10 USD/month")
        iam = IAMRole("Least-privilege\nIAM roles")

    horario >> Edge(color=GRIS) >> [diario, semanal]
    # Fuentes externas en dos columnas (XM y Open-Meteo en la primera, SIMEM en la segunda)
    # en lugar de una pila de tres sobre la orquestacion.
    [xm, clima] >> Edge(color=GRIS, style="dashed", minlen="2") >> ingesta
    simem >> Edge(color=GRIS, style="dashed") >> ingesta
    xm >> Edge(style="invis") >> simem
    diario >> Edge(color=AZUL, label="1 ingest") >> ingesta
    diario >> Edge(color=AZUL, label="2 process") >> procesar
    diario >> Edge(color=AZUL, label="3 forecast") >> predecir
    semanal >> Edge(color=NARANJA, label="retrain") >> entrenar
    ingesta >> Edge(color=AZUL) >> raw
    procesar >> Edge(color=AZUL) >> procesado
    predecir >> Edge(color=AZUL) >> pronosticos
    entrenar >> Edge(color=NARANJA) >> modelos
    repo >> Edge(color=GRIS, style="dotted", label="image") >> ingesta
    [pronosticos, raw] >> Edge(color=VERDE) >> catalogo
    catalogo >> Edge(color=VERDE) >> athena >> Edge(color=VERDE) >> usuarios
    modelos >> Edge(color=GRIS, style="dashed", label="weekly MAPE") >> metricas
    metricas >> Edge(color=GRIS) >> alarmas >> Edge(color=GRIS) >> aviso
    presupuesto >> Edge(color=GRIS, style="dashed") >> aviso

print(f"{SALIDA}.png")
