terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
  }

  # Estado remoto: copiar backend.tf.example a backend.tf y crear antes el
  # bucket y la tabla de bloqueo (ver README). Sin backend.tf, el estado queda
  # local, que basta para un `plan` de revision.
}

provider "aws" {
  region = var.region

  default_tags {
    tags = local.etiquetas
  }
}
