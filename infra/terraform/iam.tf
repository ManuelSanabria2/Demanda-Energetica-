# --------------------------------------------------------------------------
# IAM de minimo privilegio: un rol por componente, acotado al bucket y prefijos
# --------------------------------------------------------------------------

data "aws_iam_policy_document" "confianza" {
  for_each = {
    lambda    = "lambda.amazonaws.com"
    ecs       = "ecs-tasks.amazonaws.com"
    states    = "states.amazonaws.com"
    scheduler = "scheduler.amazonaws.com"
  }
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = [each.value]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.cuenta]
    }
  }
}

# Acceso a S3 por prefijos: quien lee una capa no puede escribir otra.
data "aws_iam_policy_document" "s3_prefijos" {
  for_each = {
    ingesta = { leer = ["raw/", "staging/"], escribir = ["raw/", "staging/"] }
    tareas  = { leer = ["staging/", "processed/", "models/"], escribir = ["processed/", "models/", "predictions/"] }
  }

  statement {
    sid       = "Listar"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.datalake.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = [for p in distinct(concat(each.value.leer, each.value.escribir)) : "${p}*"]
    }
  }
  statement {
    sid       = "Leer"
    actions   = ["s3:GetObject"]
    resources = [for p in each.value.leer : "${aws_s3_bucket.datalake.arn}/${p}*"]
  }
  statement {
    sid       = "Escribir"
    actions   = ["s3:PutObject"]
    resources = [for p in each.value.escribir : "${aws_s3_bucket.datalake.arn}/${p}*"]
  }
}

# --- Lambda de ingesta -------------------------------------------------------

resource "aws_iam_role" "lambda_ingesta" {
  name               = "${local.nombre}-lambda-ingesta"
  assume_role_policy = data.aws_iam_policy_document.confianza["lambda"].json
}

resource "aws_iam_role_policy" "lambda_ingesta_s3" {
  role   = aws_iam_role.lambda_ingesta.id
  policy = data.aws_iam_policy_document.s3_prefijos["ingesta"].json
}

data "aws_iam_policy_document" "lambda_logs" {
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.ingesta.arn}:*"]
  }
}

resource "aws_iam_role_policy" "lambda_ingesta_logs" {
  role   = aws_iam_role.lambda_ingesta.id
  policy = data.aws_iam_policy_document.lambda_logs.json
}

# --- ECS: rol de ejecucion (traer imagen, escribir logs) y rol de tarea ------

resource "aws_iam_role" "ecs_ejecucion" {
  name               = "${local.nombre}-ecs-ejecucion"
  assume_role_policy = data.aws_iam_policy_document.confianza["ecs"].json
}

resource "aws_iam_role_policy_attachment" "ecs_ejecucion" {
  role       = aws_iam_role.ecs_ejecucion.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_tarea" {
  name               = "${local.nombre}-ecs-tarea"
  assume_role_policy = data.aws_iam_policy_document.confianza["ecs"].json
}

resource "aws_iam_role_policy" "ecs_tarea_s3" {
  role   = aws_iam_role.ecs_tarea.id
  policy = data.aws_iam_policy_document.s3_prefijos["tareas"].json
}

data "aws_iam_policy_document" "metricas" {
  statement {
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] # PutMetricData no admite ARN; se acota por namespace
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["DemandaEnergetica/${var.entorno}"]
    }
  }
}

resource "aws_iam_role_policy" "ecs_tarea_metricas" {
  role   = aws_iam_role.ecs_tarea.id
  policy = data.aws_iam_policy_document.metricas.json
}

# --- Step Functions -----------------------------------------------------------

data "aws_iam_policy_document" "step_functions" {
  statement {
    sid       = "InvocarIngesta"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.ingesta.arn, "${aws_lambda_function.ingesta.arn}:*"]
  }
  statement {
    sid       = "EjecutarTareas"
    actions   = ["ecs:RunTask"]
    resources = [for t in aws_ecs_task_definition.tarea : replace(t.arn, "/:\\d+$/", ":*")]
    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.pipeline.arn]
    }
  }
  statement {
    sid       = "SeguirTareas"
    actions   = ["ecs:StopTask", "ecs:DescribeTasks"]
    resources = ["*"]
    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.pipeline.arn]
    }
  }
  statement {
    sid       = "PasarRoles"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.ecs_ejecucion.arn, aws_iam_role.ecs_tarea.arn]
  }
  statement {
    sid       = "IntegracionSync"
    actions   = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
    resources = ["arn:aws:events:${var.region}:${local.cuenta}:rule/StepFunctionsGetEventsForECSTaskRule"]
  }
  statement {
    sid       = "Avisar"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.alertas.arn]
  }
  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery", "logs:DeleteLogDelivery",
      "logs:ListLogDeliveries", "logs:PutResourcePolicy", "logs:DescribeResourcePolicies", "logs:DescribeLogGroups",
    ]
    resources = ["*"] # las entregas de logs de Step Functions no admiten ARN de recurso
  }
}

resource "aws_iam_role" "step_functions" {
  name               = "${local.nombre}-step-functions"
  assume_role_policy = data.aws_iam_policy_document.confianza["states"].json
}

resource "aws_iam_role_policy" "step_functions" {
  role   = aws_iam_role.step_functions.id
  policy = data.aws_iam_policy_document.step_functions.json
}

# --- EventBridge Scheduler ----------------------------------------------------

data "aws_iam_policy_document" "scheduler" {
  statement {
    actions   = ["states:StartExecution"]
    resources = [aws_sfn_state_machine.diario.arn, aws_sfn_state_machine.reentrenamiento.arn]
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${local.nombre}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.confianza["scheduler"].json
}

resource "aws_iam_role_policy" "scheduler" {
  role   = aws_iam_role.scheduler.id
  policy = data.aws_iam_policy_document.scheduler.json
}
