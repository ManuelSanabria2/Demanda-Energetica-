output "bucket_datalake" {
  description = "Bucket del data lake (raw/, staging/, processed/, models/, predictions/)."
  value       = aws_s3_bucket.datalake.bucket
}

output "repositorio_ecr" {
  description = "Destino de `docker push` de la imagen del pipeline."
  value       = aws_ecr_repository.pipeline.repository_url
}

output "lambda_ingesta" {
  value = aws_lambda_function.ingesta.function_name
}

output "maquinas_de_estado" {
  value = {
    diario          = aws_sfn_state_machine.diario.arn
    reentrenamiento = aws_sfn_state_machine.reentrenamiento.arn
  }
}

output "athena" {
  description = "Base de datos y workgroup para consultar demanda y pronosticos."
  value = {
    base_de_datos = aws_glue_catalog_database.demanda.name
    workgroup     = aws_athena_workgroup.demanda.name
  }
}

output "topico_alertas" {
  value = aws_sns_topic.alertas.arn
}
