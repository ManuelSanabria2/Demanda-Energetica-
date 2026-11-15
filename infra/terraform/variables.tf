variable "region" {
  description = "Region de AWS. us-east-1 es la de menor precio; sa-east-1 (Sao Paulo) reduce latencia desde Colombia a mayor costo."
  type        = string
  default     = "us-east-1"
}

variable "proyecto" {
  description = "Prefijo de todos los recursos."
  type        = string
  default     = "demanda-energetica"

  validation {
    condition     = can(regex("^[a-z0-9-]{3,30}$", var.proyecto))
    error_message = "Solo minusculas, digitos y guiones (3-30): el nombre entra en buckets de S3."
  }
}

variable "entorno" {
  description = "dev o prod. Cambia retenciones y la proteccion contra borrado."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "prod"], var.entorno)
    error_message = "entorno debe ser dev o prod."
  }
}

variable "correo_alertas" {
  description = "Correo que recibe alarmas y avisos de presupuesto (hay que confirmar la suscripcion de SNS)."
  type        = string
}

variable "presupuesto_mensual_usd" {
  description = "Tope mensual del presupuesto de AWS Budgets; avisa al 80 % real y al 100 % pronosticado."
  type        = number
  default     = 10
}

variable "imagen_tag" {
  description = "Tag de la imagen del pipeline en ECR: el SHA corto del commit (los tags son inmutables)."
  type        = string
  default     = "v0.1.0"
}

variable "horario_ingesta" {
  description = "Cron de EventBridge Scheduler para la ingesta diaria, en hora de Colombia."
  type        = string
  default     = "cron(0 6 * * ? *)"
}

variable "horario_reentrenamiento" {
  description = "Cron del reentrenamiento semanal (lunes 07:00, hora de Colombia)."
  type        = string
  default     = "cron(0 7 ? * MON *)"
}

variable "entrenamiento_cpu" {
  description = "vCPU de la tarea Fargate de entrenamiento (unidades de ECS: 1024 = 1 vCPU)."
  type        = number
  default     = 2048
}

variable "entrenamiento_memoria" {
  description = "Memoria de la tarea Fargate de entrenamiento, en MiB."
  type        = number
  default     = 8192
}

variable "umbral_mape_pct" {
  description = "MAPE semanal (%) por encima del cual se considera degradado el modelo y se alerta."
  type        = number
  default     = 5
}

variable "dias_retencion_logs" {
  description = "Retencion de los grupos de CloudWatch Logs."
  type        = number
  default     = 30
}
