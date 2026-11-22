"""Regenera toda la evidencia de la entrega a partir de los datos locales.

    python scripts/generar_entrega.py                    # todo (el modelado tarda ~1 h)
    python scripts/generar_entrega.py --sin-notebooks    # sin ejecutar 03, 04 y 05
    python scripts/generar_entrega.py --sin-sectores     # sin el CIIU (mas rapido)
    python scripts/generar_entrega.py --sin-modelado     # sin entrenar ni evaluar
    python scripts/generar_entrega.py --sin-busqueda     # reutiliza modelado_busqueda.json (~30 min menos)
    python scripts/generar_entrega.py --reevaluar-prueba # rehace la evaluacion en prueba

Orden, con los datos ya descargados y limpios (ver README):

1. panel y tablas de modelado, si faltan;
2. EDA            -> docs/figuras/eda, docs/resultados/eda.json
3. mineria        -> docs/figuras/mineria, docs/resultados/mineria*.{json,csv}
4. modelado       -> scripts/entrenar.py por familia, en r24 y r120:
                     baseline, lineal, clasificacion, ensembles, busqueda e
                     interpretabilidad -> docs/resultados/modelado_*.json,
                     docs/figuras/modelado, models/
5. evaluacion final en prueba -> scripts/evaluar_final.py: modelado_seleccion.json,
                     docs/modelado/seleccion.md, predicciones de prueba. Solo si
                     aun no existe o con --reevaluar-prueba: la prueba se evalua una vez.
6. costos         -> infra/costos, docs/figuras/cloud/costos.png
7. valores LaTeX  -> docs/informe/generado/
8. notebooks 03, 04 y 05 ejecutados (si estan nbclient e ipykernel)
9. PDF del informe, si hay latexmk o pdflatex en el sistema

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
TABLAS = ("r24", "r120")
# En este orden: busqueda lee lineal, ensembles y clasificacion; interpretabilidad, la busqueda.
FAMILIAS = ("baseline", "lineal", "clasificacion", "ensembles", "busqueda", "interpretabilidad")
NOTEBOOKS = ("03_eda.ipynb", "04_mineria_no_supervisada.ipynb", "05_modelado_supervisado.ipynb")
SELECCION = RAIZ / "docs" / "resultados" / "modelado_seleccion.json"


def paso(nombre: str, comando: list[str], cwd: Path = RAIZ) -> None:
    print(f"\n>>> {nombre}: {' '.join(comando)}", flush=True)
    codigo = subprocess.call(comando, cwd=cwd)
    if codigo:
        raise SystemExit(f"fallo '{nombre}' (codigo {codigo})")


def modelado(sin_busqueda: bool = False, reevaluar_prueba: bool = False) -> None:
    """Entrena cada familia en r24 y r120 y, si toca, evalua una vez en prueba."""
    for familia in FAMILIAS:
        if familia == "busqueda" and sin_busqueda:
            print("\n(sin busqueda: se reutiliza docs/resultados/modelado_busqueda.json)")
            continue
        for tabla in TABLAS:
            paso(f"modelado {familia} {tabla}", [PY, "scripts/entrenar.py", "--tabla", tabla, "--familia", familia])
    if SELECCION.exists() and not reevaluar_prueba:
        print(f"\n(la prueba ya se evaluo: {SELECCION.name} se conserva; "
              "--reevaluar-prueba para rehacerla)")
    else:
        paso("evaluacion final en prueba", [PY, "scripts/evaluar_final.py"] + (["--forzar"] if SELECCION.exists() else []))


def main(argv: list[str] | None = None) -> int:
    analizador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analizador.add_argument("--sin-notebooks", action="store_true")
    analizador.add_argument("--sin-sectores", action="store_true")
    analizador.add_argument("--sin-modelado", action="store_true", help="no entrena ni evalua")
    analizador.add_argument("--sin-busqueda", action="store_true", help="reutiliza modelado_busqueda.json")
    analizador.add_argument("--reevaluar-prueba", action="store_true", help="rehace la evaluacion en prueba")
    args = analizador.parse_args(argv)

    procesado = RAIZ / "data" / "procesado"
    if not (procesado / "panel_modelado.parquet").exists():
        paso("panel", [PY, "scripts/construir_panel.py", "--imputacion", "causal"])
    if not (procesado / "tabla_modelado_r24.parquet").exists():
        paso("variables", [PY, "scripts/construir_variables.py"])

    paso("EDA", [PY, "scripts/eda.py"])
    paso("mineria", [PY, "scripts/mineria_no_supervisada.py"] + (["--sin-sectores"] if args.sin_sectores else []))
    if not args.sin_modelado:
        modelado(args.sin_busqueda, args.reevaluar_prueba)
    paso("costos", [PY, "infra/costos/estimar_costos.py"])
    paso("valores del informe", [PY, "docs/informe/generar_valores.py"])

    if not args.sin_notebooks:
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            print("\n(sin nbclient/ipykernel: los notebooks no se ejecutan; pip install nbclient ipykernel)")
        else:
            for nombre in NOTEBOOKS:
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
    print("\nListo. Revisa docs/ y haz commit de docs/figuras, docs/resultados, docs/modelado, docs/informe y notebooks/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
