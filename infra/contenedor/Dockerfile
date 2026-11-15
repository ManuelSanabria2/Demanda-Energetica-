# Imagen unica del pipeline: la usan la Lambda de ingesta y las tareas de Fargate.
#   docker build -f infra/contenedor/Dockerfile -t demanda-energetica-pipeline .
# (desde la raiz del repositorio)
FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt scikit-learn==1.5.2 scipy statsmodels==0.14.4 mlxtend==0.23.4

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY scripts/ ${LAMBDA_TASK_ROOT}/scripts/
COPY descargar.py ${LAMBDA_TASK_ROOT}/
COPY infra/contenedor/ ${LAMBDA_TASK_ROOT}/infra/contenedor/
COPY infra/contenedor/lambda_ingesta.py infra/contenedor/s3sync.py ${LAMBDA_TASK_ROOT}/

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/src:${LAMBDA_TASK_ROOT}
# Lambda usa ENTRYPOINT + el handler; las tareas de ECS sustituyen el
# entryPoint por /bin/sh -c (ver computo.tf).
CMD ["lambda_ingesta.handler"]
