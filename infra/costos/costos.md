# Estimacion preliminar de costos (us-east-1, precios de 2026-11)

TRM 3,297.07 COP/USD. Generado por `infra/costos/estimar_costos.py`.

| Servicio | Academico (USD/mes) | Sin capa gratuita | Operacion (2 entornos) |
|---|---:|---:|---:|
| Lambda (ingesta) | 0.00 | 0.06 | 0.12 |
| ECS Fargate (procesar, predecir, entrenar) | 0.22 | 0.22 | 6.29 |
| S3 (almacenamiento y solicitudes) | 0.19 | 0.19 | 1.11 |
| ECR (imagenes) | 0.40 | 0.40 | 0.80 |
| Step Functions | 0.00 | 0.01 | 0.02 |
| CloudWatch (logs, alarmas, metricas) | 0.01 | 1.71 | 5.58 |
| Athena + Glue Data Catalog | 0.02 | 0.02 | 0.38 |
| X-Ray | 0.00 | 0.00 | 0.01 |
| EventBridge Scheduler, SNS, Budgets | 0.00 | 0.00 | 0.00 |
| **Total USD/mes** | **0.84** | **2.61** | **14.31** |
| Total COP/mes | 2,770 | 8,605 | 47,181 |
| Total USD/ano | 10.08 | 31.32 | 171.72 |

Decision de diseno: sin NAT Gateway se evitan 32.85 USD/mes fijos, mas que todo el resto junto.

Estimacion preliminar. Validar con AWS Pricing Calculator antes de desplegar; sa-east-1 (Sao Paulo) cuesta tipicamente 30-50 % mas que us-east-1.
