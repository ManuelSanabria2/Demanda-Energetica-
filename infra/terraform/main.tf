# Arquitectura de despliegue propuesta (Fase E del proyecto integrador).
#
#   EventBridge Scheduler --(diario)--> Step Functions "pipeline-diario"
#        Lambda ingesta (XM, SIMEM, Open-Meteo) -> S3 raw/ y staging/
#        ECS Fargate "procesar"  (limpieza, panel, variables) staging/ -> processed/
#        ECS Fargate "predecir"  (pronostico 24 h) -> S3 predictions/
#   EventBridge Scheduler --(semanal)--> Step Functions "reentrenamiento"
#        ECS Fargate "entrenar" (escalera de modelos, validacion temporal) -> S3 models/
#        metrica MAPE -> CloudWatch -> alarma -> SNS (correo)
#   Consumo: Glue Data Catalog + Athena sobre S3 (Parquet particionado).
#
# Una sola imagen de contenedor (ECR) con el paquete src/ sirve a la Lambda y a
# las tareas de Fargate; cambia solo el comando. Asi el codigo que corre en la
# nube es el mismo que pasa la suite de pruebas.

data "aws_caller_identity" "actual" {}
data "aws_region" "actual" {}

locals {
  nombre = "${var.proyecto}-${var.entorno}"
  cuenta = data.aws_caller_identity.actual.account_id

  etiquetas = {
    Proyecto    = var.proyecto
    Entorno     = var.entorno
    ODS         = "9"
    Gestionado  = "terraform"
    Repositorio = "github.com/ManuelSanabria2/Demanda-Energetica-"
  }

  prefijos_s3 = ["raw/", "staging/", "processed/", "models/", "predictions/", "athena-results/"]
  imagen      = "${aws_ecr_repository.pipeline.repository_url}:${var.imagen_tag}"
  es_prod     = var.entorno == "prod"
}
