# --------------------------------------------------------------------------
# Consumo del pronostico: Glue Data Catalog + Athena sobre S3
# --------------------------------------------------------------------------
# Sin crawler (cobra por DPU-hora): el esquema se conoce y se declara aqui, y
# las particiones se resuelven con partition projection, asi que no hay que
# registrar cada mes nuevo.

resource "aws_glue_catalog_database" "demanda" {
  name        = replace("${local.nombre}_db", "-", "_")
  description = "Demanda horaria del SIN (XM) y pronosticos a 24 h - proyecto integrador ODS 9"
}

resource "aws_glue_catalog_table" "demanda_real" {
  name          = "demanda_real_sistema"
  database_name = aws_glue_catalog_database.demanda.name
  table_type    = "EXTERNAL_TABLE"
  description   = "XM DemaReal/Sistema normalizada (ingesta.cli): una fila por hora, kWh"

  parameters = {
    "classification"            = "parquet"
    "projection.enabled"        = "true"
    "projection.anio.type"      = "integer"
    "projection.anio.range"     = "2021,2035"
    "projection.mes.type"       = "integer"
    "projection.mes.range"      = "1,12"
    "storage.location.template" = "s3://${aws_s3_bucket.datalake.bucket}/staging/xm_demanda_real_sistema/anio=$${anio}/mes=$${mes}/"
  }

  partition_keys {
    name = "anio"
    type = "int"
  }
  partition_keys {
    name = "mes"
    type = "int"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.datalake.bucket}/staging/xm_demanda_real_sistema/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }

    columns {
      name = "fecha_hora"
      type = "timestamp"
    }
    columns {
      name = "fuente"
      type = "string"
    }
    columns {
      name = "metrica"
      type = "string"
    }
    columns {
      name = "entidad"
      type = "string"
    }
    columns {
      name = "valor_kwh"
      type = "double"
    }
    columns {
      name = "version"
      type = "string"
    }
  }
}

resource "aws_glue_catalog_table" "pronosticos" {
  name          = "pronosticos_24h"
  database_name = aws_glue_catalog_database.demanda.name
  table_type    = "EXTERNAL_TABLE"
  description   = "Pronostico horario a 24 h, una particion por fecha de emision"

  parameters = {
    "classification"                  = "parquet"
    "projection.enabled"              = "true"
    "projection.fecha_emision.type"   = "date"
    "projection.fecha_emision.range"  = "2026-01-01,NOW"
    "projection.fecha_emision.format" = "yyyy-MM-dd"
    "storage.location.template"       = "s3://${aws_s3_bucket.datalake.bucket}/predictions/fecha_emision=$${fecha_emision}/"
  }

  partition_keys {
    name = "fecha_emision"
    type = "date"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.datalake.bucket}/predictions/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }

    columns {
      name = "fecha_hora"
      type = "timestamp"
    }
    columns {
      name = "demanda_pronosticada_mwh"
      type = "double"
    }
    columns {
      name    = "limite_inferior_mwh"
      type    = "double"
      comment = "intervalo de prediccion (cuantil 10)"
    }
    columns {
      name    = "limite_superior_mwh"
      type    = "double"
      comment = "intervalo de prediccion (cuantil 90)"
    }
    columns {
      name = "modelo"
      type = "string"
    }
    columns {
      name = "version_modelo"
      type = "string"
    }
  }
}

resource "aws_athena_workgroup" "demanda" {
  name          = "${local.nombre}-analitica"
  force_destroy = !local.es_prod

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true
    # Freno de costo: ninguna consulta puede leer mas de 1 GB (~0.005 USD).
    bytes_scanned_cutoff_per_query = 1073741824

    result_configuration {
      output_location = "s3://${aws_s3_bucket.datalake.bucket}/athena-results/"
      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
}

resource "aws_athena_named_query" "pico_diario" {
  name      = "pico-diario-pronosticado-vs-real"
  workgroup = aws_athena_workgroup.demanda.id
  database  = aws_glue_catalog_database.demanda.name
  query     = <<-SQL
    -- Error sobre el maximo diario: la variable que usa un planificador de capacidad.
    WITH real AS (
      SELECT date(fecha_hora) AS dia, max(valor_kwh) / 1000 AS pico_real_mwh
      FROM demanda_real_sistema
      WHERE anio = year(current_date)
      GROUP BY 1
    ), pron AS (
      SELECT date(fecha_hora) AS dia, max(demanda_pronosticada_mwh) AS pico_pron_mwh
      FROM pronosticos_24h
      WHERE fecha_emision >= current_date - INTERVAL '60' DAY
      GROUP BY 1
    )
    SELECT dia, pico_real_mwh, pico_pron_mwh,
           100 * (pico_pron_mwh - pico_real_mwh) / pico_real_mwh AS error_pico_pct
    FROM real JOIN pron USING (dia)
    ORDER BY dia DESC
  SQL
}
