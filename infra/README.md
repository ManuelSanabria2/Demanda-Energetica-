# Infraestructura en la nube (módulo Cloud Computing)

Arquitectura propuesta para operar el pipeline de pronóstico en AWS (Fase E del
proyecto), definida como código con **Terraform**. Diagrama:
[`docs/figuras/cloud/arquitectura.png`](../docs/figuras/cloud/arquitectura.png).

```
infra/
  terraform/     IaC: S3, ECR, Lambda, ECS Fargate, Step Functions, Scheduler,
                 Glue + Athena, CloudWatch, SNS, Budgets, IAM
  contenedor/    imagen única del pipeline + handler de Lambda + envoltorio de tareas
  costos/        estimación preliminar de costos (script, JSON y tabla)
  diagrama/      diagrama de arquitectura generado desde código (iconos oficiales de AWS)
```

## Qué hace cada pieza

| Componente | Recurso | Qué ejecuta del repositorio |
|---|---|---|
| Ingesta diaria (06:00 COT) | Lambda (imagen de contenedor) | `ingesta.descarga.descargar` y `ingesta.cli --fuente todas`, incremental con 45 días de solape |
| Panel y variables | ECS Fargate Spot, tarea `procesar` | `scripts/construir_panel.py --imputacion causal` y `scripts/construir_variables.py` |
| Pronóstico 24 h | ECS Fargate Spot, tarea `predecir` | `scripts/predecir.py` *(fase de modelado, pendiente)* |
| Reentrenamiento semanal | ECS Fargate Spot, tarea `entrenar` | `scripts/entrenar.py` *(fase de modelado, pendiente)* |
| Orquestación | Step Functions (diario y semanal) + EventBridge Scheduler | reintentos y aviso por SNS ante cualquier fallo |
| Data lake | S3 versionado: `raw/`, `staging/`, `processed/`, `models/`, `predictions/` | la misma estructura de `data/` en local |
| Consumo | Glue Data Catalog + Athena (partition projection, tope de 1 GB por consulta) | consulta de ejemplo: error sobre el pico diario |
| Monitoreo | CloudWatch + SNS + AWS Budgets | fallos, ausencia de pronóstico 48 h, MAPE global y de horas pico, presupuesto |

Decisiones con impacto en costo, documentadas en el código:

- **Sin NAT Gateway** (≈ 32.85 USD/mes fijos): las tareas corren en subredes
  públicas de la VPC por defecto con un grupo de seguridad **sin ninguna regla
  de entrada**.
- **Fargate Spot** para lotes de minutos que toleran interrupción (Step
  Functions reintenta).
- **Sin crawler de Glue**: el esquema se declara y las particiones se proyectan.
- **Concurrencia 1 en la Lambda**: dos ingestas simultáneas pisarían las mismas
  particiones de S3.

## Uso

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars      # correo de alertas, entorno
terraform init
terraform fmt -check && terraform validate
terraform plan -out plan.tfplan
terraform apply plan.tfplan

# Imagen del pipeline (desde la raíz del repositorio)
TAG=$(git rev-parse --short HEAD)
REPO=$(terraform -chdir=infra/terraform output -raw repositorio_ecr)
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%/*}"
docker build -f infra/contenedor/Dockerfile -t "$REPO:$TAG" .
docker push "$REPO:$TAG"
terraform -chdir=infra/terraform apply -var imagen_tag=$TAG
```

## Verificación en este avance

| Comprobación | Resultado |
|---|---|
| Sintaxis HCL (python-hcl2, los 10 archivos `.tf`) | OK |
| Checkov 3 (análisis estático de seguridad, `checkov -d infra/terraform`) | **171 aprobadas, 0 fallidas**, 22 omitidas con justificación escrita en el recurso (`#checkov:skip=...`) |
| `terraform validate` / `plan` | Pendiente: requiere el binario de Terraform y credenciales de AWS del equipo |

Las 22 omisiones son decisiones de costo para el alcance académico (claves KMS
propias, retención de logs de un año, replicación entre regiones, Container
Insights), no descuidos: cada una dice por qué en el propio recurso.

## Costos

`python infra/costos/estimar_costos.py` regenera [`costos/costos.md`](costos/costos.md).
Resumen (us-east-1, precios de lista de noviembre de 2026, TRM 3 297,07):

| Escenario | USD/mes | COP/mes |
|---|---:|---:|
| Académico (con capa gratuita permanente) | 0.84 | ≈ 2 770 |
| Mismo uso, sin capa gratuita | 2.61 | ≈ 8 605 |
| Operación (dev + prod, reentrenamiento diario) | 14.31 | ≈ 47 181 |
