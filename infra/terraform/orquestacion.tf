# --------------------------------------------------------------------------
# Step Functions: dos flujos, uno diario y uno semanal
# --------------------------------------------------------------------------
# Cada paso reintenta ante fallos transitorios (las APIs de XM y SIMEM devuelven
# 5xx esporadicos) y cualquier fallo definitivo termina en un aviso por SNS:
# un pipeline que falla en silencio deja el pronostico de ayer publicado como
# si fuera de hoy.

locals {
  red_tareas = {
    AwsvpcConfiguration = {
      Subnets        = data.aws_subnets.defecto.ids
      SecurityGroups = [aws_security_group.tareas.id]
      AssignPublicIp = "ENABLED"
    }
  }
  capacidad = [{ CapacityProvider = "FARGATE_SPOT", Weight = 1 }]
  reintento_tarea = [{
    ErrorEquals     = ["States.TaskFailed", "ECS.AmazonECSException"]
    IntervalSeconds = 60
    MaxAttempts     = 2
    BackoffRate     = 2
  }]
  aviso_fallo = {
    Type     = "Task"
    Resource = "arn:aws:states:::sns:publish"
    Parameters = {
      TopicArn    = aws_sns_topic.alertas.arn
      Subject     = "[${local.nombre}] fallo del pipeline"
      "Message.$" = "States.JsonToString($)"
    }
    Next = "Fallido"
  }

  paso_ecs = { for nombre in keys(local.tareas) : nombre => {
    Type     = "Task"
    Resource = "arn:aws:states:::ecs:runTask.sync"
    Parameters = {
      Cluster                  = aws_ecs_cluster.pipeline.arn
      TaskDefinition           = aws_ecs_task_definition.tarea[nombre].arn
      CapacityProviderStrategy = local.capacidad
      NetworkConfiguration     = local.red_tareas
    }
    ResultPath = null
    Retry      = local.reintento_tarea
    Catch      = [{ ErrorEquals = ["States.ALL"], ResultPath = "$.error", Next = "AvisarFallo" }]
  } }
}

resource "aws_sfn_state_machine" "diario" {
  name     = "${local.nombre}-pipeline-diario"
  role_arn = aws_iam_role.step_functions.arn

  definition = jsonencode({
    Comment = "Ingesta incremental, panel y variables, y pronostico de las proximas 24 h"
    StartAt = "Ingesta"
    States = {
      Ingesta = {
        Type       = "Task"
        Resource   = "arn:aws:states:::lambda:invoke"
        Parameters = { FunctionName = aws_lambda_function.ingesta.arn, Payload = {} }
        ResultPath = "$.ingesta"
        Retry = [{
          ErrorEquals     = ["Lambda.ServiceException", "Lambda.TooManyRequestsException", "States.TaskFailed"]
          IntervalSeconds = 120
          MaxAttempts     = 3
          BackoffRate     = 2
        }]
        Catch = [{ ErrorEquals = ["States.ALL"], ResultPath = "$.error", Next = "AvisarFallo" }]
        Next  = "Procesar"
      }
      Procesar    = merge(local.paso_ecs["procesar"], { Next = "Predecir" })
      Predecir    = merge(local.paso_ecs["predecir"], { Next = "Exito" })
      AvisarFallo = local.aviso_fallo
      Exito       = { Type = "Succeed" }
      Fallido     = { Type = "Fail", Error = "PipelineFallido" }
    }
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.step_functions.arn}:*"
    include_execution_data = true # cargas pequenas y sin secretos
    level                  = "ALL"
  }

  tracing_configuration {
    enabled = true
  }
}

resource "aws_sfn_state_machine" "reentrenamiento" {
  name     = "${local.nombre}-reentrenamiento"
  role_arn = aws_iam_role.step_functions.arn

  definition = jsonencode({
    Comment = "Reentrenamiento semanal con validacion walk-forward; publica el MAPE en CloudWatch"
    StartAt = "Procesar"
    States = {
      Procesar    = merge(local.paso_ecs["procesar"], { Next = "Entrenar" })
      Entrenar    = merge(local.paso_ecs["entrenar"], { Next = "Exito" })
      AvisarFallo = local.aviso_fallo
      Exito       = { Type = "Succeed" }
      Fallido     = { Type = "Fail", Error = "ReentrenamientoFallido" }
    }
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.step_functions.arn}:*"
    include_execution_data = true # cargas pequenas y sin secretos
    level                  = "ALL"
  }

  tracing_configuration {
    enabled = true
  }
}

resource "aws_cloudwatch_log_group" "step_functions" {
  #checkov:skip=CKV_AWS_158:Logs operativos sin datos personales; cifrado por defecto de CloudWatch.
  #checkov:skip=CKV_AWS_338:Retencion parametrizada (var.dias_retencion_logs); 1 ano solo se justifica en prod.
  name              = "/aws/vendedlogs/states/${local.nombre}"
  retention_in_days = var.dias_retencion_logs
}

# --------------------------------------------------------------------------
# EventBridge Scheduler, en hora de Colombia
# --------------------------------------------------------------------------
# La demanda de XM se publica con ~3 dias de rezago: correr a las 06:00 deja
# el dato del dia anterior ya publicado por el operador.

resource "aws_scheduler_schedule" "diario" {
  #checkov:skip=CKV_AWS_297:El horario no contiene datos sensibles; clave administrada por AWS.
  name                         = "${local.nombre}-diario"
  schedule_expression          = var.horario_ingesta
  schedule_expression_timezone = "America/Bogota"

  flexible_time_window {
    mode                      = "FLEXIBLE"
    maximum_window_in_minutes = 15
  }

  target {
    arn      = aws_sfn_state_machine.diario.arn
    role_arn = aws_iam_role.scheduler.arn
    retry_policy {
      maximum_retry_attempts = 2
    }
  }
}

resource "aws_scheduler_schedule" "semanal" {
  #checkov:skip=CKV_AWS_297:El horario no contiene datos sensibles; clave administrada por AWS.
  name                         = "${local.nombre}-reentrenamiento"
  schedule_expression          = var.horario_reentrenamiento
  schedule_expression_timezone = "America/Bogota"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_sfn_state_machine.reentrenamiento.arn
    role_arn = aws_iam_role.scheduler.arn
  }
}
