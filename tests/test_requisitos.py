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
PAQUETES_PROPIOS = {"ingesta", "calidad", "limpieza", "variables", "mineria", "modelado", "conftest",
                    # infra/contenedor: modulos del handler de Lambda, importados por tests/test_infra.py
                    "lambda_ingesta", "s3sync",
                    # tests/test_modelado_regresion.py reutiliza las tablas sinteticas de este modulo
                    "test_modelado_base",
                    # docs/informe: macros del modelado, importadas por generar_valores.py y sus pruebas
                    "valores_modelado",
                    # tests/test_interfaz.py reutiliza el panel sintetico de este modulo
                    "test_scripts_entrega"}
# app/: la interfaz importa sus propios componentes con `from componentes import ...`.
PAQUETES_APP = {"componentes", "paginas"}


def _requisitos(archivo: Path = RAIZ / "requirements.txt") -> dict[str, str]:
    """Nombre del paquete -> especificador de version, desde un requirements.txt."""
    requisitos = {}
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if linea:
            nombre = re.split(r"[<>=!~ ]", linea, maxsplit=1)[0]
            requisitos[nombre.lower()] = linea[len(nombre):].strip()
    return requisitos


def _codigo_del_proyecto() -> list[str]:
    """Todo el Python del proyecto, celdas de los notebooks incluidas."""
    codigo = [
        ruta.read_text(encoding="utf-8")
        for carpeta in ("src", "scripts", "tests", "app")
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


# Paquetes cuyo nombre de importacion no coincide con el de PyPI.
DISTRIBUCION = {"sklearn": "scikit-learn", "PIL": "pillow", "IPython": "ipython"}


def test_todo_lo_que_se_importa_esta_en_requirements():
    importados = set().union(*map(_importados, _codigo_del_proyecto()))
    terceros = importados - set(sys.stdlib_module_names) - PAQUETES_PROPIOS - PAQUETES_APP
    distribucion = DISTRIBUCION
    faltan = sorted(n for n in terceros if distribucion.get(n, n).lower() not in _requisitos())
    assert not faltan, f"Se importan pero no estan en requirements.txt: {faltan}"


def test_numpy_por_debajo_de_2_mientras_pyarrow_sea_anterior_a_16():
    requisitos = _requisitos()
    pyarrow = int(re.search(r"\d+", requisitos["pyarrow"]).group())
    if pyarrow < 16:
        assert re.fullmatch(r"==\s*1\.[\d.]+|<\s*2(\.0)*", requisitos.get("numpy", "")), (
            f"pyarrow {requisitos['pyarrow']} no importa con numpy 2: fija numpy por debajo de 2"
        )


def test_lo_que_importa_la_app_esta_en_app_requirements():
    """La app desplegada se instala solo con app/requirements.txt: lo que falte ahi rompe en la nube."""
    codigo = [ruta.read_text(encoding="utf-8") for ruta in (RAIZ / "app").rglob("*.py")]
    importados = set().union(*map(_importados, codigo))
    terceros = importados - set(sys.stdlib_module_names) - PAQUETES_APP
    requisitos_app = _requisitos(RAIZ / "app" / "requirements.txt")
    faltan = sorted(n for n in terceros if DISTRIBUCION.get(n, n).lower() not in requisitos_app)
    assert not faltan, f"app/ importa pero no estan en app/requirements.txt: {faltan}"
    # pandas lee Parquet con pyarrow sin importarlo por nombre
    assert "pyarrow" in requisitos_app


def test_app_requirements_fijado_y_con_las_versiones_del_proyecto():
    """Mismas versiones que requirements.txt: la app en la nube es la que pasa las pruebas aqui."""
    raiz = _requisitos()
    for nombre, version in _requisitos(RAIZ / "app" / "requirements.txt").items():
        assert version.startswith("=="), f"{nombre} sin version fijada en app/requirements.txt"
        assert raiz.get(nombre) == version, f"{nombre}{version} en app/ pero {raiz.get(nombre)} en requirements.txt"
