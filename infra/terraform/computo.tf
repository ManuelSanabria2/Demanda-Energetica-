# --------------------------------------------------------------------------
# Lambda de ingesta: incremental diaria contra XM, SIMEM y Open-Meteo
# --------------------------------------------------------------------------
# La ingesta incremental pide desde 45 dias antes del ultimo dato (las
# versiones de liquidacion de SIMEM se revisan durante semanas) y con la cache
# poblada tarda segundos: cabe holgada en los 15 min de Lambda.

resource "aws_cloudwatch_log_group" "ingesta" {
  #checkov:skip=CKV_AWS_158:Logs operativos sin datos personales; cifrado por defecto de CloudWatch.
  #checkov:skip=CKV_AWS_338:Retencion parametrizada (var.dias_retencion_logs); 1 ano solo se justifica en prod.
  name              = "/aws/lambda/${local.nombre}-ingesta"
  retention_in_days = var.dias_retencion_logs
}

resource "aws_lambda_function" "ingesta" {
  #checkov:skip=CKV_AWS_116:La invoca Step Functions de forma sincrona con reintentos y aviso por SNS; una DLQ no aplica.
  #checkov:skip=CKV_AWS_117:Solo sale a APIs publicas y a S3; dentro de una VPC necesitaria NAT Gateway (~32 USD/mes).
  #checkov:skip=CKV_AWS_272:Firma de codigo fuera del alcance del avance; la imagen se escanea en ECR.
  #checkov:skip=CKV_AWS_173:Las variables de entorno no contienen secretos (nombres de bucket y prefijos).
  function_name = "${local.nombre}-ingesta"
  role          = aws_iam_role.lambda_ingesta.arn
  package_type  = "Image"
  image_uri     = local.imagen
  architectures = ["x86_64"]
  memory_size   = 1024
  timeout       = 900

  # Una sola ejecucion a la vez: dos ingestas simultaneas leerian y reescribirian
  # las mismas particiones de S3 y la ultima en subir pisaria a la otra.
  reserved_concurrent_executions = 1

  tracing_config {
    mode = "Active"
  }

  image_config {
    command = ["lambda_ingesta.handler"]
  }

  environment {
    variables = {
      BUCKET_DATOS    = aws_s3_bucket.datalake.bucket
      PREFIJO_CRUDO   = "raw/"
      PREFIJO_STAGING = "staging/"
      DIAS_REFRESCO   = "45"
      ZONA_HORARIA    = "America/Bogota"
    }
  }

  depends_on = [aws_cloudwatch_log_group.ingesta]
}

# --------------------------------------------------------------------------
# ECS Fargate: procesar, predecir y entrenar
# --------------------------------------------------------------------------
# Sin NAT Gateway (~32 USD/mes fijos, mas que todo lo demas junto): las tareas
# corren en subredes publicas de la VPC por defecto con IP publica y un grupo de
# seguridad que no admite NINGUNA entrada. Solo salen a las APIs y a S3/ECR.

data "aws_vpc" "defecto" {
  default = true
}

data "aws_subnets" "defecto" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.defecto.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

resource "aws_security_group" "tareas" {
  #checkov:skip=CKV2_AWS_5:Se adjunta en tiempo de ejecucion a las tareas lanzadas por Step Functions (NetworkConfiguration).
  name        = "${local.nombre}-tareas"
  description = "Tareas del pipeline: sin entrada, salida HTTPS"
  vpc_id      = data.aws_vpc.defecto.id

  egress {
    description = "HTTPS hacia APIs de XM/SIMEM/Open-Meteo, S3 y ECR"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_cluster" "pipeline" {
  #checkov:skip=CKV_AWS_65:Container Insights cobra por metrica; las tareas son lotes de minutos y sus logs bastan.
  name = "${local.nombre}-pipeline"

  setting {
    name  = "containerInsights"
    value = "disabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "pipeline" {
  cluster_name       = aws_ecs_cluster.pipeline.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }
}

resource "aws_cloudwatch_log_group" "tareas" {
  #checkov:skip=CKV_AWS_158:Logs operativos sin datos personales; cifrado por defecto de CloudWatch.
  #checkov:skip=CKV_AWS_338:Retencion parametrizada (var.dias_retencion_logs); 1 ano solo se justifica en prod.
  name              = "/ecs/${local.nombre}"
  retention_in_days = var.dias_retencion_logs
}

# Cada tarea envuelve un comando del repositorio con infra/contenedor/tarea.py,
# que baja de S3 las capas de entrada, ejecuta y sube las de salida.
# `procesar` usa scripts que ya existen. `predecir` y `entrenar` apuntan a los
# scripts de la fase de modelado (pendiente): estan declaradas para que la
# orquestacion quede completa, y se activan cuando esos scripts existan.
locals {
  envoltorio = "python infra/contenedor/tarea.py"
  tareas = {
    procesar = {
      cpu     = 1024
      memoria = 4096
      comando = "${local.envoltorio} --entrada staging/ --salida processed/ -- sh -c 'python scripts/construir_panel.py --imputacion causal && python scripts/construir_variables.py'"
    }
    predecir = {
      cpu     = 512
      memoria = 2048
      comando = "${local.envoltorio} --entrada processed/ models/ --salida predictions/ -- python scripts/predecir.py --horizonte 24"
    }
    entrenar = {
      cpu     = var.entrenamiento_cpu
      memoria = var.entrenamiento_memoria
      comando = "${local.envoltorio} --entrada processed/ --salida models/ -- python scripts/entrenar.py --validacion walk-forward"
    }
  }
}

resource "aws_ecs_task_definition" "tarea" {
  #checkov:skip=CKV_AWS_336:Las tareas escriben data/ en disco antes de subirlo a S3 (infra/contenedor/tarea.py).
  for_each = local.tareas

  family                   = "${local.nombre}-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = each.value.cpu
  memory                   = each.value.memoria
  execution_role_arn       = aws_iam_role.ecs_ejecucion.arn
  task_role_arn            = aws_iam_role.ecs_tarea.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name      = each.key
    image     = local.imagen
    essential        = true
    entryPoint       = ["/bin/sh", "-c"]
    command          = [each.value.comando]
    workingDirectory = "/var/task"
    environment = [
      { name = "BUCKET_DATOS", value = aws_s3_bucket.datalake.bucket },
      { name = "ZONA_HORARIA", value = "America/Bogota" },
      { name = "METRICA_NAMESPACE", value = "DemandaEnergetica/${var.entorno}" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.tareas.name
        awslogs-region        = var.region
        awslogs-stream-prefix = each.key
      }
    }
  }])
}
