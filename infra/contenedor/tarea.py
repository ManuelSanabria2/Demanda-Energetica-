"""Punto de entrada de las tareas de ECS Fargate: sincroniza, ejecuta, devuelve.

    python infra/contenedor/tarea.py --entrada staging/ --salida processed/ -- \
        python scripts/construir_panel.py --imputacion causal

Trae los prefijos de `--entrada` a data/, ejecuta el comando del repositorio tal
cual y sube los prefijos de `--salida`. Un comando que falla no sube nada: la
salida parcial de una ejecucion rota no debe pisar la ultima buena en S3.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3sync  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
CARPETAS = {"raw/": RAIZ / "data" / "raw", "staging/": RAIZ / "data" / "processed",
            "processed/": RAIZ / "data" / "procesado",
            "models/": RAIZ / "data" / "modelos", "predictions/": RAIZ / "data" / "pronosticos"}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("uso: tarea.py --entrada ... --salida ... -- comando")
    corte = argv.index("--")
    analizador = argparse.ArgumentParser()
    analizador.add_argument("--entrada", nargs="*", default=[])
    analizador.add_argument("--salida", nargs="*", default=[])
    args = analizador.parse_args(argv[:corte])
    comando = argv[corte + 1:]

    bucket = os.environ["BUCKET_DATOS"]
    for prefijo in args.entrada:
        s3sync.bajar(bucket, prefijo, CARPETAS[prefijo])
    codigo = subprocess.call(comando, cwd=RAIZ)
    if codigo != 0:
        logging.error("el comando termino con codigo %d: no se sube nada", codigo)
        return codigo
    for prefijo in args.salida:
        s3sync.subir(CARPETAS[prefijo], bucket, prefijo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
