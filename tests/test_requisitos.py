"""Lo que importa el codigo esta en requirements.txt. Sin red.

El caso que lo motiva: faltaban numpy, nbformat y matplotlib, y la suite pasaba
igual porque el entorno de desarrollo ya los tenia instalados. Una instalacion
limpia rompia en el primer Parquet: pip ponia numpy 2 y pyarrow 14 no importa
con numpy 2.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PAQUETES_PROPIOS = {"ingesta", "calidad", "limpieza", "conftest"}


def _requisitos() -> dict[str, str]:
    """Nombre del paquete -> especificador de version, desde requirements.txt."""
    requisitos = {}
    for linea in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if linea:
            nombre = re.split(r"[<>=!~ ]", linea, maxsplit=1)[0]
            requisitos[nombre.lower()] = linea[len(nombre):].strip()
    return requisitos


def _codigo_del_proyecto() -> list[str]:
    """Todo el Python del proyecto, celdas de los notebooks incluidas."""
    codigo = [
        ruta.read_text(encoding="utf-8")
        for carpeta in ("src", "scripts", "tests")
        for ruta in (RAIZ / carpeta).rglob("*.py")
    ]
    codigo.append((RAIZ / "descargar.py").read_text(encoding="utf-8"))
    for cuaderno in (RAIZ / "notebooks").rglob("*.ipynb"):
        for celda in json.loads(cuaderno.read_text(encoding="utf-8"))["cells"]:
            if celda["cell_type"] == "code":
                # Las magias de IPython (%matplotlib, !pip) no son Python.
                lineas = "".join(celda["source"]).splitlines()
                codigo.append("\n".join(l for l in lineas if not l.lstrip().startswith(("%", "!"))))
    return codigo


def _importados(codigo: str) -> set[str]:
    nombres = set()
    for nodo in ast.walk(ast.parse(codigo)):
        if isinstance(nodo, ast.Import):
            nombres |= {alias.name.split(".")[0] for alias in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
            nombres.add(nodo.module.split(".")[0])
    return nombres


def test_todo_lo_que_se_importa_esta_en_requirements():
    importados = set().union(*map(_importados, _codigo_del_proyecto()))
    terceros = importados - set(sys.stdlib_module_names) - PAQUETES_PROPIOS
    faltan = sorted(n for n in terceros if n.lower() not in _requisitos())
    assert not faltan, f"Se importan pero no estan en requirements.txt: {faltan}"


def test_numpy_por_debajo_de_2_mientras_pyarrow_sea_anterior_a_16():
    requisitos = _requisitos()
    pyarrow = int(re.search(r"\d+", requisitos["pyarrow"]).group())
    if pyarrow < 16:
        assert re.fullmatch(r"==\s*1\.[\d.]+|<\s*2(\.0)*", requisitos.get("numpy", "")), (
            f"pyarrow {requisitos['pyarrow']} no importa con numpy 2: fija numpy por debajo de 2"
        )
