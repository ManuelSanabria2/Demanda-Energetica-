# --------------------------------------------------------------------------
# Alertas: fallos del pipeline, degradacion del modelo y presupuesto
# --------------------------------------------------------------------------

resource "aws_sns_topic" "alertas" {
  name              = "${local.nombre}-alertas"
  kms_master_key_id = "alias/aws/sns"
}

resource "aws_sns_topic_subscription" "correo" {
  topic_arn = aws_sns_topic.alertas.arn
  protocol  = "email"
  endpoint  = var.correo_alertas
}

resource "aws_cloudwatch_metric_alarm" "ingesta_errores" {
  alarm_name          = "${local.nombre}-ingesta-errores"
  alarm_description   = "La Lambda de ingesta fallo: el panel no tendra los datos del dia."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.ingesta.function_name }
  statistic           = "Sum"
  period              = 86400
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alertas.arn]
}

resource "aws_cloudwatch_metric_alarm" "pipeline_fallido" {
  for_each = {
    diario          = aws_sfn_state_machine.diario.arn
    reentrenamiento = aws_sfn_state_machine.reentrenamiento.arn
  }
  alarm_name          = "${local.nombre}-${each.key}-fallido"
  alarm_description   = "Una ejecucion de Step Functions (${each.key}) termino en fallo."
  namespace           = "AWS/States"
  metric_name         = "ExecutionsFailed"
  dimensions          = { StateMachineArn = each.value }
  statistic           = "Sum"
  period              = 86400
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alertas.arn]
}

# Ningun pronostico emitido en 2 dias: el pipeline puede "tener exito" sin
# producir nada (p. ej. la API no publico). La tarea predecir publica
# PronosticosEmitidos = 1 en cada ejecucion correcta.
resource "aws_cloudwatch_metric_alarm" "sin_pronostico" {
  alarm_name          = "${local.nombre}-sin-pronostico"
  alarm_description   = "No se emitio pronostico en las ultimas 48 h."
  namespace           = "DemandaEnergetica/${var.entorno}"
  metric_name         = "PronosticosEmitidos"
  statistic           = "Sum"
  period              = 86400
  evaluation_periods  = 2
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching"
  alarm_actions       = [aws_sns_topic.alertas.arn]
}

# Degradacion: el reentrenamiento publica el MAPE de la ultima semana (pronostico
# vs real) y el MAPE en horas pico. Por encima del umbral, el modelo ya no
# supera con margen al baseline estacional y hay que revisarlo.
resource "aws_cloudwatch_metric_alarm" "mape_degradado" {
  for_each = { MAPE_semanal = "global", MAPE_pico_semanal = "horas pico" }

  alarm_name          = "${local.nombre}-${lower(each.key)}"
  alarm_description   = "MAPE (${each.value}) por encima de ${var.umbral_mape_pct} %: posible degradacion del modelo."
  namespace           = "DemandaEnergetica/${var.entorno}"
  metric_name         = each.key
  statistic           = "Maximum"
  period              = 604800
  evaluation_periods  = 1
  threshold           = var.umbral_mape_pct
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "missing"
  alarm_actions       = [aws_sns_topic.alertas.arn]
}

resource "aws_budgets_budget" "mensual" {
  name         = "${local.nombre}-mensual"
  budget_type  = "COST"
  limit_amount = tostring(var.presupuesto_mensual_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:Proyecto$%s", var.proyecto)]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.correo_alertas]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.correo_alertas]
  }
}
