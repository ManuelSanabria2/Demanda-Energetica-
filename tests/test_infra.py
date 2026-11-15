"""Pruebas de la capa de nube que no necesitan AWS (sin red).

- `s3sync` sube solo lo que cambio y baja conservando la estructura.
- `lambda_ingesta.reconfigurar_rutas` lleva TODAS las rutas de datos a /tmp.
- `tarea.py` no sube nada si el comando falla.
- La estimacion de costos cuadra: el total es la suma del detalle.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "infra" / "contenedor"))

import s3sync  # noqa: E402


class S3Falso:
    """Lo minimo de boto3.client('s3') que usa s3sync, en memoria."""

    def __init__(self):
        self.objetos: dict[str, bytes] = {}
        self.subidas: list[str] = []

    def get_paginator(self, _nombre):
        falso = self

        class Paginador:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                yield {"Contents": [
                    {"Key": k, "ETag": f'"{hashlib.md5(v).hexdigest()}"'}  # noqa: S324
                    for k, v in sorted(falso.objetos.items()) if k.startswith(Prefix)
                ]}

        return Paginador()

    def upload_file(self, ruta, _bucket, clave):
        self.objetos[clave] = Path(ruta).read_bytes()
        self.subidas.append(clave)

    def download_file(self, _bucket, clave, ruta):
        Path(ruta).write_bytes(self.objetos[clave])


def test_subir_solo_lo_que_cambio(tmp_path):
    (tmp_path / "xm" / "anio=2025").mkdir(parents=True)
    (tmp_path / "xm" / "anio=2025" / "datos.parquet").write_bytes(b"a")
    (tmp_path / "manifiesto.json").write_bytes(b"{}")
    s3 = S3Falso()
    assert s3sync.subir(tmp_path, "b", "raw/", s3) == 2
    assert s3sync.subir(tmp_path, "b", "raw/", s3) == 0          # idempotente
    (tmp_path / "manifiesto.json").write_bytes(b'{"n": 1}')
    assert s3sync.subir(tmp_path, "b", "raw/", s3) == 1
    assert s3.subidas[-1] == "raw/manifiesto.json"


def test_bajar_conserva_la_estructura(tmp_path):
    s3 = S3Falso()
    s3.objetos = {"raw/": b"", "raw/xm/anio=2025/mes=01/datos.parquet": b"x", "otro/y": b"y"}
    assert s3sync.bajar("b", "raw/", tmp_path, s3) == 1
    assert (tmp_path / "xm" / "anio=2025" / "mes=01" / "datos.parquet").read_bytes() == b"x"
    assert not (tmp_path / "y").exists()


def test_lambda_redirige_todas_las_rutas(tmp_path, monkeypatch):
    from ingesta import config

    originales = {n: getattr(config, n) for n in dir(config) if n.startswith(("DIR_", "RUTA_"))}
    import lambda_ingesta

    try:
        lambda_ingesta.reconfigurar_rutas(tmp_path)
        for nombre in originales:
            if nombre == "DIR_NOTAS":
                continue  # notas/ es documentacion, no datos
            assert str(getattr(config, nombre)).startswith(str(tmp_path)), nombre
    finally:
        for nombre, valor in originales.items():
            setattr(config, nombre, valor)


def test_tarea_no_sube_si_el_comando_falla(monkeypatch):
    spec = importlib.util.spec_from_file_location("tarea", RAIZ / "infra" / "contenedor" / "tarea.py")
    tarea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tarea)
    llamadas = []
    monkeypatch.setenv("BUCKET_DATOS", "b")
    monkeypatch.setattr(tarea.s3sync, "bajar", lambda *a, **k: llamadas.append("bajar"))
    monkeypatch.setattr(tarea.s3sync, "subir", lambda *a, **k: llamadas.append("subir"))
    codigo = tarea.main(["--entrada", "staging/", "--salida", "processed/", "--", sys.executable, "-c", "raise SystemExit(3)"])
    assert codigo == 3
    assert llamadas == ["bajar"]
    assert tarea.main(["--salida", "processed/", "--", sys.executable, "-c", "pass"]) == 0
    assert llamadas == ["bajar", "subir"]


def test_costos_cuadran():
    spec = importlib.util.spec_from_file_location("costos", RAIZ / "infra" / "costos" / "estimar_costos.py")
    costos = importlib.util.module_from_spec(spec)
    sys.modules["costos"] = costos  # dataclasses necesita el modulo registrado
    spec.loader.exec_module(costos)
    academico = costos.costos(costos.ESCENARIOS["academico"])
    sin_gratis = costos.costos(costos.ESCENARIOS["sin_capa_gratuita"])
    assert all(v >= 0 for v in academico.values())
    assert sum(sin_gratis.values()) >= sum(academico.values())
    # la capa gratuita cubre la Lambda de ingesta diaria entera
    assert academico["Lambda (ingesta)"] == 0
    assert sin_gratis["Lambda (ingesta)"] == pytest.approx(30 * 120 * 1.0 * 0.0000166667, abs=1e-3)
