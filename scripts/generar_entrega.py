"""Regenera toda la evidencia del avance a partir de los datos locales.

    python scripts/generar_entrega.py                 # todo
    python scripts/generar_entrega.py --sin-notebooks # sin ejecutar 03 y 04
    python scripts/generar_entrega.py --sin-sectores  # sin el CIIU (mas rapido)

Orden, con los datos ya descargados y limpios (ver README):

1. panel y tablas de modelado, si faltan;
2. EDA            -> docs/figuras/eda, docs/resultados/eda.json
3. mineria        -> docs/figuras/mineria, docs/resultados/mineria*.{json,csv}
4. costos         -> infra/costos, docs/figuras/cloud/costos.png
5. valores LaTeX  -> docs/informe/generado/
6. notebooks 03 y 04 ejecutados (si estan nbclient e ipykernel)
7. PDF del informe, si hay latexmk o pdflatex en el sistema

Cada paso es un script del repositorio; este solo los encadena y se detiene en
el primero que falle.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PY = sys.executable


def paso(nombre: str, comando: list[str], cwd: Path = RAIZ) -> None:
    print(f"\n>>> {nombre}: {' '.join(comando)}", flush=True)
    codigo = subprocess.call(comando, cwd=cwd)
    if codigo:
        raise SystemExit(f"fallo '{nombre}' (codigo {codigo})")


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analizador.add_argument("--sin-notebooks", action="store_true")
    analizador.add_argument("--sin-sectores", action="store_true")
    args = analizador.parse_args(argv)

    procesado = RAIZ / "data" / "procesado"
    if not (procesado / "panel_modelado.parquet").exists():
        paso("panel", [PY, "scripts/construir_panel.py", "--imputacion", "causal"])
    if not (procesado / "tabla_modelado_r24.parquet").exists():
        paso("variables", [PY, "scripts/construir_variables.py"])

    paso("EDA", [PY, "scripts/eda.py"])
    paso("mineria", [PY, "scripts/mineria_no_supervisada.py"] + (["--sin-sectores"] if args.sin_sectores else []))
    paso("costos", [PY, "infra/costos/estimar_costos.py"])
    paso("valores del informe", [PY, "docs/informe/generar_valores.py"])

    if not args.sin_notebooks:
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            print("\n(sin nbclient/ipykernel: los notebooks 03 y 04 no se ejecutan; pip install nbclient ipykernel)")
        else:
            for nombre in ("03_eda.ipynb", "04_mineria_no_supervisada.ipynb"):
                ruta = RAIZ / "notebooks" / nombre
                print(f"\n>>> notebook {nombre}", flush=True)
                libro = nbformat.read(ruta, as_version=4)
                NotebookClient(libro, timeout=1800, kernel_name="python3",
                               resources={"metadata": {"path": str(ruta.parent)}}).execute()
                nbformat.write(libro, ruta)

    informe = RAIZ / "docs" / "informe"
    if shutil.which("latexmk"):
        paso("PDF", ["latexmk", "-pdf", "-interaction=nonstopmode", "avance_proyecto_integrador.tex"], cwd=informe)
    elif shutil.which("pdflatex"):
        for _ in range(2):
            paso("PDF", ["pdflatex", "-interaction=nonstopmode", "avance_proyecto_integrador.tex"], cwd=informe)
    else:
        print("\n(sin LaTeX: compila docs/informe/avance_proyecto_integrador.tex en Overleaf subiendo la carpeta "
              "docs/informe y docs/figuras)")
    print("\nListo. Revisa docs/ y haz commit de docs/figuras, docs/resultados, docs/informe y notebooks/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
