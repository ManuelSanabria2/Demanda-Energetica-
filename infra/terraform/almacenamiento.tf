# --------------------------------------------------------------------------
# Data lake en S3: un bucket, seis prefijos (capas)
# --------------------------------------------------------------------------
#   raw/          capa cruda de descarga.py: XM y SIMEM tal como se publican,
#                 particionada por anio/mes, con el manifiesto de descargas
#   staging/      series normalizadas y clima de ingesta.cli (data/processed/)
#   processed/    panel y tablas de modelado r24/r120 (Parquet) + registros JSON
#   models/       modelos serializados y su registro de hiperparametros y metricas
#   predictions/  pronostico de 24 h, particionado por fecha de emision
#   athena-results/  salidas de consultas (expiran)
#
# El repositorio ya trata el manifiesto de descargas como registro de
# reproducibilidad; aqui el versionado de S3 cumple la misma funcion para los
# datos: una reingesta nunca destruye la version anterior.

resource "aws_s3_bucket" "datalake" {
  #checkov:skip=CKV_AWS_18:Un solo bucket de datos publicos; los accesos quedan en CloudTrail. Un bucket de logs aparte duplica costo sin valor para el alcance academico.
  #checkov:skip=CKV_AWS_144:Replicacion entre regiones innecesaria: los datos se regeneran desde las APIs de XM y SIMEM (README, capa cruda).
  #checkov:skip=CKV_AWS_145:Datos publicos del mercado electrico; SSE-S3 basta y evita 1 USD/mes por clave KMS.
  #checkov:skip=CKV2_AWS_62:La orquestacion es por horario (EventBridge Scheduler), no por eventos de S3.
  bucket        = "${local.nombre}-datalake-${local.cuenta}"
  force_destroy = !local.es_prod
}

resource "aws_s3_bucket_versioning" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "datalake" {
  bucket                  = aws_s3_bucket.datalake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "datalake" {
  bucket = aws_s3_bucket.datalake.id

  rule {
    id     = "versiones-antiguas"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = local.es_prod ? 180 : 30
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }

  rule {
    id     = "resultados-athena"
    status = "Enabled"
    filter {
      prefix = "athena-results/"
    }
    expiration {
      days = 7
    }
  }

  # Los pronosticos viejos solo sirven para auditar el error: pasan a acceso
  # infrecuente. raw/ no se mueve: sus objetos son pequenos (< 128 KB) y
  # STANDARD_IA cobra un minimo de 128 KB por objeto.
  rule {
    id     = "pronosticos-historicos"
    status = "Enabled"
    filter {
      prefix = "predictions/"
    }
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }
}

data "aws_iam_policy_document" "solo_tls" {
  statement {
    sid     = "DenegarSinTLS"
    effect  = "Deny"
    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.datalake.arn,
      "${aws_s3_bucket.datalake.arn}/*",
    ]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "datalake" {
  bucket     = aws_s3_bucket.datalake.id
  policy     = data.aws_iam_policy_document.solo_tls.json
  depends_on = [aws_s3_bucket_public_access_block.datalake]
}

resource "aws_s3_object" "prefijos" {
  for_each = toset(local.prefijos_s3)
  bucket   = aws_s3_bucket.datalake.id
  key      = each.value
  content  = ""
}

# --------------------------------------------------------------------------
# Registro de contenedores: una imagen para Lambda y Fargate
# --------------------------------------------------------------------------

resource "aws_ecr_repository" "pipeline" {
  #checkov:skip=CKV_AWS_136:Imagen sin secretos; cifrado AES256 administrado por AWS.
  name                 = "${local.nombre}-pipeline"
  image_tag_mutability = "IMMUTABLE" # cada imagen se etiqueta con el commit: un tag nunca cambia de contenido
  force_delete         = !local.es_prod

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "pipeline" {
  repository = aws_ecr_repository.pipeline.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Conservar las 10 imagenes mas recientes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}
