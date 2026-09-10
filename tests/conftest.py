"""Configuracion comun de las pruebas.

Bloquea la red para toda la suite. El proyecto vive de dos APIs publicas, asi
que es facil que una prueba acabe llamandolas sin querer -- basta olvidar un
mock. Cuando eso pasa, la suite se vuelve lenta, intermitente y dependiente de
que XM y SIMEM esten disponibles y devuelvan lo mismo de siempre.

Con este guardia, una prueba que intente salir a la red falla de inmediato y
con un mensaje que dice donde mirar, en vez de tardar treinta segundos y pasar.

Si alguna vez hiciera falta una prueba que si use la red, marcarla con
`@pytest.mark.red` y ejecutarla aparte:

    pytest -m red
"""

from __future__ import annotations

import socket

import pytest

_conectar_real = socket.socket.connect
_conectar_ex_real = socket.socket.connect_ex


class RedProhibida(AssertionError):
    """Una prueba intento abrir una conexion de red."""


def _bloqueado(self, direccion, *args, **kwargs):  # noqa: ANN001
    raise RedProhibida(
        f"Una prueba intento conectarse a {direccion}. Las pruebas no deben usar "
        "la red: simula la API con unittest.mock. Si de verdad necesita red, "
        "marcala con @pytest.mark.red."
    )


def pytest_configure(config: pytest.Config) -> None:
    """Registra la marca que exime del bloqueo."""
    config.addinivalue_line("markers", "red: la prueba necesita acceso real a la red")


@pytest.fixture(autouse=True)
def sin_red(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Impide cualquier conexion saliente, salvo en las pruebas marcadas."""
    if request.node.get_closest_marker("red"):
        return

    monkeypatch.setattr(socket.socket, "connect", _bloqueado)
    monkeypatch.setattr(socket.socket, "connect_ex", _bloqueado)
