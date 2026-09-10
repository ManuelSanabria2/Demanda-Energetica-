"""Pruebas de lo que puede fallar en silencio.

Un fallo ruidoso se arregla el dia que aparece. Los que preocupan son los que
producen datos plausibles pero mal: un desplazamiento de una hora, un dia que se
pierde al trocear un rango, un merge que duplica filas, una interpolacion que
rellena mas de lo que debe. Nada de eso levanta una excepcion; todo eso
contamina el modelo entero.

Este modulo agrupa esas pruebas por riesgo, no por modulo de codigo. Ninguna
toca la red: las APIs se simulan con `unittest.mock`.

Organizadas segun las cinco prioridades:

    1. Fragmentacion de fechas y reensamblado
    2. Conversion ancho a largo de XM   <-- la mas importante
    3. Idempotencia de la descarga
    4. Limite de interpolacion en la limpieza
    5. Integridad del merge
"""

from __future__ import annotations

import datetime as dt
import itertools
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingesta import complementarias, config, descarga  # noqa: E402
from ingesta.clientes import ClienteXM  # noqa: E402
from ingesta.ventanas import partir_rango  # noqa: E402
from limpieza import limpiar as L  # noqa: E402


# ==========================================================================
# 1. Fragmentacion de fechas
# ==========================================================================
#
# El riesgo: perder un dia o pedirlo dos veces al trocear un rango largo. No
# falla nunca de forma visible; simplemente falta un dia en la serie final.


def test_noventa_dias_con_limite_31_dan_tres_tramos():
    """90 dias = 31 + 31 + 28, con los cortes exactamente donde deben caer."""
    tramos = partir_rango(dt.date(2025, 1, 1), dt.date(2025, 3, 31), 31)

    assert tramos == [
        (dt.date(2025, 1, 1), dt.date(2025, 1, 31)),
        (dt.date(2025, 2, 1), dt.date(2025, 3, 3)),
        (dt.date(2025, 3, 4), dt.date(2025, 3, 31)),
    ]
    assert [(b - a).days + 1 for a, b in tramos] == [31, 31, 28]


def test_el_reensamblado_cubre_cada_dia_exactamente_una_vez():
    """La prueba que de verdad importa: ni un dia perdido, ni uno repetido."""
    inicio, fin = dt.date(2025, 1, 1), dt.date(2025, 3, 31)  # 90 dias
    tramos = partir_rango(inicio, fin, 31)

    dias: list[dt.date] = []
    for desde, hasta in tramos:
        dias.extend(
            desde + dt.timedelta(days=n) for n in range((hasta - desde).days + 1)
        )

    esperados = [inicio + dt.timedelta(days=n) for n in range((fin - inicio).days + 1)]
    assert dias == esperados, "el troceado no reconstruye el rango original"
    assert len(dias) == 90
    assert len(set(dias)) == 90, "hay dias repetidos entre tramos"


@pytest.mark.parametrize(
    "dias,limite,tramos_esperados",
    [
        (1, 31, 1),      # un solo dia
        (30, 31, 1),     # por debajo del limite
        (31, 31, 1),     # exactamente el limite
        (32, 31, 2),     # el limite mas uno
        (62, 31, 2),     # dos tramos exactos
        (63, 31, 3),     # dos tramos exactos mas un dia
        (90, 31, 3),     # el caso del enunciado
        (1, 1, 1),       # limite minimo
        (5, 1, 5),       # un tramo por dia
    ],
)
def test_casos_borde_del_troceado(dias: int, limite: int, tramos_esperados: int):
    inicio = dt.date(2025, 1, 1)
    fin = inicio + dt.timedelta(days=dias - 1)

    tramos = partir_rango(inicio, fin, limite)

    assert len(tramos) == tramos_esperados
    assert tramos[0][0] == inicio
    assert tramos[-1][1] == fin
    assert all((b - a).days + 1 <= limite for a, b in tramos)


def test_el_limite_exacto_no_parte_de_mas():
    """31 dias con limite 31 debe ser UN llamado, no dos."""
    tramos = partir_rango(dt.date(2025, 1, 1), dt.date(2025, 1, 31), 31)
    assert len(tramos) == 1
    assert tramos[0] == (dt.date(2025, 1, 1), dt.date(2025, 1, 31))


def test_el_limite_mas_uno_parte_en_dos_desiguales():
    """32 dias con limite 31: un tramo lleno y otro de un solo dia."""
    tramos = partir_rango(dt.date(2025, 1, 1), dt.date(2025, 2, 1), 31)

    assert len(tramos) == 2
    assert tramos[0] == (dt.date(2025, 1, 1), dt.date(2025, 1, 31))
    assert tramos[1] == (dt.date(2025, 2, 1), dt.date(2025, 2, 1))


def test_ninguna_combinacion_pierde_ni_repite_dias():
    """Barrido sobre muchos rangos y limites: propiedad, no ejemplo."""
    inicio = dt.date(2025, 1, 1)

    for dias, limite in itertools.product(range(1, 75), (1, 2, 7, 30, 31, 60)):
        fin = inicio + dt.timedelta(days=dias - 1)
        tramos = partir_rango(inicio, fin, limite)

        cubiertos = [
            desde + dt.timedelta(days=n)
            for desde, hasta in tramos
            for n in range((hasta - desde).days + 1)
        ]
        assert len(cubiertos) == dias, f"{dias} dias, limite {limite}: faltan o sobran"
        assert len(set(cubiertos)) == dias, f"{dias} dias, limite {limite}: repetidos"
        assert min(cubiertos) == inicio and max(cubiertos) == fin


# ==========================================================================
# 2. Conversion ancho a largo de XM   <-- LA MAS IMPORTANTE
# ==========================================================================
#
# Un desplazamiento de una hora aqui contamina el proyecto entero y no produce
# ningun error: la serie sigue teniendo 24 valores por dia y una forma diaria
# plausible, solo que corrida. Por eso se fija contra una respuesta REAL del
# servidor, con sus valores literales.

# Capturada de la API el 2026-09-09 para DemaReal/Sistema del 2025-01-01.
# Copia textual de data/raw_samples/xm_hourly_DemaReal_Sistema_20250101.json.
RESPUESTA_XM_REAL: dict[str, Any] = {
    "Metric": {
        "Id": "DemaReal",
        "Name": "Demanda Real por Sistema",
        "StartDate": "2025-01-01T00:00:00",
        "EndDate": "2025-01-01T00:00:00",
    },
    "Items": [
        {
            "Date": "2025-01-01",
            "HourlyEntities": [
                {
                    "Id": "Sistema",
                    "Values": {
                        "code": "Sistema",
                        "Hour01": "7306335.34000",
                        "Hour02": "7183560.15000",
                        "Hour03": "7059543.59000",
                        "Hour04": "6927948.08000",
                        "Hour05": "6817406.45000",
                        "Hour06": "6681093.93000",
                        "Hour07": "6328032.32000",
                        "Hour08": "6303549.75000",
                        "Hour09": "6475989.57000",
                        "Hour10": "6668547.55000",
                        "Hour11": "6918522.47000",
                        "Hour12": "7128732.29000",
                        "Hour13": "7301342.79000",
                        "Hour14": "7399780.64000",
                        "Hour15": "7359716.56000",
                        "Hour16": "7332181.84000",
                        "Hour17": "7302295.05000",
                        "Hour18": "7470656.91000",
                        "Hour19": "8431773.29000",
                        "Hour20": "8637418.15000",
                        "Hour21": "8555426.93000",
                        "Hour22": "8386926.63000",
                        "Hour23": "8221798.60000",
                        "Hour24": "7925012.28000",
                    },
                }
            ],
        }
    ],
}


ZONA = "America/Bogota"


def convertir(crudo: dict[str, Any] = RESPUESTA_XM_REAL) -> pd.DataFrame:
    """Aplica la conversion del cliente sin tocar la red."""
    return ClienteXM(sesion=mock.MagicMock())._a_dataframe(crudo, "DemaReal")


def marca(texto: str) -> pd.Timestamp:
    """Marca de tiempo en la zona unica del proyecto."""
    return pd.Timestamp(texto, tz=ZONA)


def test_un_dia_se_desdobla_en_24_filas():
    tabla = convertir()

    assert len(tabla) == 24
    assert tabla["timestamp"].nunique() == 24


def test_hour01_es_la_medianoche_con_su_valor_real():
    """El anclaje del proyecto: Hour01 es la franja 00:00-01:00.

    Verificado empiricamente contra SIMEM (correlacion 0.9996 con desfase 0
    sobre un mes ya liquidado). Si esta asercion falla, toda la serie esta
    corrida y hay que revisar config.DESFASE_HORA_XM antes que nada.
    """
    tabla = convertir().set_index("timestamp")

    assert tabla.loc[marca("2025-01-01 00:00:00"), "valor"] == 7306335.34


def test_hour24_es_las_23_del_mismo_dia_no_las_00_del_siguiente():
    """El error clasico: mandar Hour24 al dia siguiente."""
    tabla = convertir().set_index("timestamp")

    assert tabla.loc[marca("2025-01-01 23:00:00"), "valor"] == 7925012.28
    assert marca("2025-01-02 00:00:00") not in tabla.index


def test_cada_hora_cae_donde_debe():
    """Las 24 en bloque, para que un desplazamiento de una hora salte."""
    tabla = convertir().set_index("timestamp")["valor"]
    esperado = [
        7306335.34, 7183560.15, 7059543.59, 6927948.08, 6817406.45, 6681093.93,
        6328032.32, 6303549.75, 6475989.57, 6668547.55, 6918522.47, 7128732.29,
        7301342.79, 7399780.64, 7359716.56, 7332181.84, 7302295.05, 7470656.91,
        8431773.29, 8637418.15, 8555426.93, 8386926.63, 8221798.60, 7925012.28,
    ]
    for hora, valor in enumerate(esperado):
        momento = marca("2025-01-01") + pd.Timedelta(hours=hora)
        assert tabla.loc[momento] == valor, f"la hora {hora:02d} no cuadra"


def test_el_rango_horario_va_de_00_a_23():
    tabla = convertir()

    assert tabla["timestamp"].min() == marca("2025-01-01 00:00:00")
    assert tabla["timestamp"].max() == marca("2025-01-01 23:00:00")
    assert sorted(tabla["timestamp"].dt.hour) == list(range(24))


def test_un_desfase_distinto_desplazaria_la_serie():
    """Comprobacion de que la prueba anterior detectaria de verdad el fallo.

    Con DESFASE_HORA_XM = 0 la serie se corre una hora: Hour01 pasa a las 01:00.
    Si esto no cambiara nada, las aserciones de arriba no estarian probando nada.
    """
    with mock.patch.object(config, "DESFASE_HORA_XM", 0):
        tabla = convertir().set_index("timestamp")

    assert marca("2025-01-01 00:00:00") not in tabla.index
    assert tabla.loc[marca("2025-01-01 01:00:00"), "valor"] == 7306335.34


def test_el_valor_llega_como_numero_no_como_cadena():
    """La API devuelve cadenas; sumarlas concatenaria en vez de sumar."""
    tabla = convertir()

    assert tabla["valor"].dtype == float
    # El total del dia: si los valores siguieran siendo cadenas, esto reventaria.
    assert tabla["valor"].sum() == pytest.approx(176_123_591.16, abs=0.01)


def test_varios_dias_seguidos_no_se_solapan():
    dia2 = {
        "Date": "2025-01-02",
        "HourlyEntities": [
            {"Id": "Sistema", "Values": {"code": "Sistema",
                                         **{f"Hour{n:02d}": "1000.0" for n in range(1, 25)}}}
        ],
    }
    crudo = {"Metric": {}, "Items": [RESPUESTA_XM_REAL["Items"][0], dia2]}

    tabla = ClienteXM(sesion=mock.MagicMock())._a_dataframe(crudo, "DemaReal")

    assert len(tabla) == 48
    assert tabla["timestamp"].nunique() == 48
    esperadas = pd.date_range("2025-01-01", "2025-01-02 23:00", freq="h", tz=ZONA)
    assert sorted(tabla["timestamp"]) == list(esperadas)


# ==========================================================================
# 3. Idempotencia de la descarga
# ==========================================================================
#
# El riesgo: que reejecutar acumule filas en vez de sustituirlas. Se nota tarde
# y mal, cuando la serie tiene el doble de observaciones de las que deberia.


@pytest.fixture
def almacen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Aisla la capa cruda y su manifiesto en un directorio temporal."""
    monkeypatch.setattr(config, "DIR_CRUDO", tmp_path / "raw")
    monkeypatch.setattr(config, "RUTA_MANIFIESTO_CRUDO", tmp_path / "raw" / "m.json")
    monkeypatch.setattr(config, "DIR_CACHE", tmp_path / "cache")
    return tmp_path


def _sesion_simulada(dias: int = 3) -> mock.MagicMock:
    """Sesion HTTP falsa que responde como la API de XM."""
    items = []
    for n in range(dias):
        fecha = (dt.date(2025, 1, 1) + dt.timedelta(days=n)).isoformat()
        items.append(
            {
                "Date": fecha,
                "HourlyEntities": [
                    {
                        "Id": "Sistema",
                        "Values": {
                            "code": "Sistema",
                            **{f"Hour{h:02d}": f"{1000 + h}.5" for h in range(1, 25)},
                        },
                    }
                ],
            }
        )

    respuesta = mock.MagicMock()
    respuesta.status_code = 200
    respuesta.json.return_value = {"Metric": {"Id": "DemaReal"}, "Items": items}

    sesion = mock.MagicMock()
    sesion.post.return_value = respuesta
    return sesion


def test_descargar_dos_veces_no_cambia_el_resultado(almacen: Path):
    """Misma peticion, dos veces: mismas filas y mismo hash."""
    sesion = _sesion_simulada(dias=3)
    cliente = ClienteXM(sesion=sesion, usar_cache=False)
    cliente._catalogo = pd.DataFrame(
        [{"MetricId": "DemaReal", "Entity": "Sistema", "MaxDays": 31}]
    )

    with mock.patch.dict(descarga.CLIENTES, {"xm": lambda **_: cliente}):
        primera = descarga.descargar(
            "xm", "DemaReal", dt.date(2025, 1, 1), dt.date(2025, 1, 3)
        )
        contenido_1 = descarga.leer("xm", "DemaReal")

        segunda = descarga.descargar(
            "xm", "DemaReal", dt.date(2025, 1, 1), dt.date(2025, 1, 3)
        )
        contenido_2 = descarga.leer("xm", "DemaReal")

    assert len(contenido_1) == 3 * 24
    assert len(contenido_2) == len(contenido_1), "la segunda ejecucion acumulo filas"
    assert primera["hash_resultado"] == segunda["hash_resultado"]
    assert descarga.hash_marco(contenido_1) == descarga.hash_marco(contenido_2)


def test_reejecutar_no_deja_duplicados(almacen: Path):
    sesion = _sesion_simulada(dias=3)
    cliente = ClienteXM(sesion=sesion, usar_cache=False)
    cliente._catalogo = pd.DataFrame(
        [{"MetricId": "DemaReal", "Entity": "Sistema", "MaxDays": 31}]
    )

    with mock.patch.dict(descarga.CLIENTES, {"xm": lambda **_: cliente}):
        for _ in range(3):
            descarga.descargar("xm", "DemaReal", dt.date(2025, 1, 1), dt.date(2025, 1, 3))

    contenido = descarga.leer("xm", "DemaReal")
    assert len(contenido) == 3 * 24
    assert not contenido.duplicated(subset=["timestamp", "entidad"]).any()


def test_la_descarga_no_toco_la_red(almacen: Path):
    """Guardia: si alguien quita el mock, esta prueba lo dice."""
    sesion = _sesion_simulada(dias=1)
    cliente = ClienteXM(sesion=sesion, usar_cache=False)
    cliente._catalogo = pd.DataFrame(
        [{"MetricId": "DemaReal", "Entity": "Sistema", "MaxDays": 31}]
    )

    with mock.patch.dict(descarga.CLIENTES, {"xm": lambda **_: cliente}):
        descarga.descargar("xm", "DemaReal", dt.date(2025, 1, 1), dt.date(2025, 1, 1))

    assert sesion.post.called
    assert sesion.post.call_args.kwargs["json"]["MetricId"] == "DemaReal"


# ==========================================================================
# 4. Limite de interpolacion en la limpieza
# ==========================================================================
#
# El riesgo: inventar datos. Un hueco de dos semanas relleno por interpolacion
# produce una serie continua, bonita y falsa, indistinguible del dato real.


def _serie_horaria(dias: int = 30, inicio: str = "2025-01-06") -> pd.DataFrame:
    momentos = pd.date_range(inicio, periods=dias * 24, freq="h")
    return pd.DataFrame(
        {
            "fecha_hora": momentos,
            "valor_kwh": [1000.0 + i for i in range(len(momentos))],
        }
    )


def _quitar(marco: pd.DataFrame, desde: str, horas: int) -> pd.DataFrame:
    fuera = pd.date_range(desde, periods=horas, freq="h")
    return marco[~marco["fecha_hora"].isin(fuera)].reset_index(drop=True)


@pytest.mark.parametrize(
    "horas_hueco,se_interpola",
    [(1, True), (2, True), (3, True), (4, False), (5, False), (24, False), (336, False)],
)
def test_el_limite_de_tres_horas_se_respeta(horas_hueco: int, se_interpola: bool):
    marco = _quitar(_serie_horaria(dias=40), "2025-01-15 02:00", horas_hueco)

    limpio, registro = L.completar_rejilla(marco)
    detalle = registro["detalle"]

    if se_interpola:
        assert detalle["horas_interpoladas"] == horas_hueco
        assert detalle["horas_dejadas_como_nan"] == 0
    else:
        assert detalle["horas_interpoladas"] == 0
        assert detalle["horas_dejadas_como_nan"] == horas_hueco

    hueco = pd.date_range("2025-01-15 02:00", periods=horas_hueco, freq="h")
    tramo = limpio[limpio["fecha_hora"].isin(hueco)]
    assert len(tramo) == horas_hueco
    if se_interpola:
        assert tramo["valor_kwh"].notna().all()
        assert tramo["imputado"].all()
    else:
        assert tramo["valor_kwh"].isna().all()
        assert not tramo["imputado"].any()


def test_un_hueco_largo_permanece_como_nan():
    """Dos semanas: se quedan vacias, marcadas, y nadie las rellena."""
    marco = _quitar(_serie_horaria(dias=60), "2025-02-01 00:00", 14 * 24)

    limpio, _ = L.completar_rejilla(marco)
    tramo = limpio[limpio["fecha_hora"].between("2025-02-01", "2025-02-14 23:00")]

    assert len(tramo) == 14 * 24
    assert tramo["valor_kwh"].isna().all(), "se inventaron dias completos"
    assert (tramo["origen_valor"] == "faltante").all()
    assert not tramo["imputado"].any()
    assert (tramo["hueco_horas"] == 14 * 24).all()


def test_el_valor_interpolado_es_el_que_toca():
    """No basta con que rellene: el valor debe ser la interpolacion lineal."""
    marco = _serie_horaria(dias=10)
    antes = marco.loc[marco["fecha_hora"] == pd.Timestamp("2025-01-08 01:00"), "valor_kwh"].iloc[0]
    despues = marco.loc[marco["fecha_hora"] == pd.Timestamp("2025-01-08 04:00"), "valor_kwh"].iloc[0]
    marco = _quitar(marco, "2025-01-08 02:00", 2)

    limpio, _ = L.completar_rejilla(marco)
    obtenido = limpio.set_index("fecha_hora")["valor_kwh"]

    assert obtenido.loc["2025-01-08 02:00"] == pytest.approx(antes + (despues - antes) / 3)
    assert obtenido.loc["2025-01-08 03:00"] == pytest.approx(antes + 2 * (despues - antes) / 3)


def test_un_hueco_corto_pegado_al_borde_no_se_extrapola():
    """Interpolar sin datos a un lado seria extrapolar, o sea inventar."""
    marco = _serie_horaria(dias=10)
    completo, _ = L.completar_rejilla(marco)
    sin_cabeza = completo.iloc[2:].copy()

    limpio, registro = L.completar_rejilla(sin_cabeza)

    assert registro["detalle"]["filas_creadas_para_completar_rejilla"] == 0
    assert limpio["fecha_hora"].min() == sin_cabeza["fecha_hora"].min()


def test_lo_imputado_queda_distinguible_de_lo_observado():
    """El requisito de fondo: un NaN o un valor imputado siempre identificables."""
    marco = _quitar(_serie_horaria(dias=40), "2025-01-15 02:00", 2)
    marco = _quitar(marco, "2025-01-20 00:00", 48)

    limpio, _ = L.completar_rejilla(marco)
    origenes = limpio["origen_valor"].value_counts()

    assert origenes["interpolado"] == 2
    assert origenes["faltante"] == 48
    assert limpio.loc[limpio["imputado"], "valor_kwh"].notna().all()
    assert limpio.loc[limpio["origen_valor"] == "faltante", "valor_kwh"].isna().all()


# ==========================================================================
# 5. Integridad del merge
# ==========================================================================
#
# El riesgo: unir con una tabla que trae claves repetidas multiplica filas. La
# serie resultante sigue pareciendo una serie, con el doble de observaciones.


def _demanda(horas: int = 72) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fecha_hora": pd.date_range("2025-01-01", periods=horas, freq="h"),
            "valor_kwh": range(horas),
        }
    )


def _clima(horas: int = 72) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fecha_hora": pd.date_range("2025-01-01", periods=horas, freq="h"),
            "temperatura_c": [20.0] * horas,
        }
    )


def test_el_merge_conserva_el_numero_de_filas():
    base = _demanda()
    calendario, _ = complementarias.calendario_horario(
        dt.date(2025, 1, 1), dt.date(2025, 1, 3)
    )

    resultado, informe = complementarias.unir(base, _clima(), calendario)

    assert len(resultado) == len(base) == 72
    assert informe["filas_demanda"] == informe["filas_resultado"]
    assert informe["cuadra"] is True


def test_una_clave_repetida_a_la_derecha_no_pasa_desapercibida():
    """Sin la validacion, esto duplicaria silenciosamente cada fila."""
    duplicado = pd.concat([_clima(24), _clima(24)], ignore_index=True)

    with pytest.raises(complementarias.ErrorUnion, match="repetidas"):
        complementarias.unir(_demanda(24), duplicado)


def test_el_merge_sin_validacion_si_duplicaria():
    """Demuestra que el peligro es real y que la guardia sirve para algo."""
    duplicado = pd.concat([_clima(24), _clima(24)], ignore_index=True)

    ingenuo = _demanda(24).merge(duplicado, on="fecha_hora", how="left")

    assert len(ingenuo) == 48, "el merge ingenuo duplica"


def test_una_fuente_incompleta_no_borra_filas_de_demanda():
    """Un left join con cobertura parcial deja NaN, nunca menos filas."""
    resultado, informe = complementarias.unir(_demanda(72), _clima(24))

    assert len(resultado) == 72
    union = [u for u in informe["uniones"] if u["fuente"] == "clima"][0]
    assert union["filas_sin_cobertura"] == 48
    assert resultado["temperatura_c"].isna().sum() == 48


def test_el_orden_de_las_fuentes_no_altera_el_resultado():
    calendario, _ = complementarias.calendario_horario(
        dt.date(2025, 1, 1), dt.date(2025, 1, 3)
    )

    a, _ = complementarias.unir(_demanda(), _clima(), calendario)
    b, _ = complementarias.unir(_demanda(), None, calendario)
    b, _ = complementarias.unir(b, _clima(), None)

    assert len(a) == len(b) == 72
    pd.testing.assert_series_equal(
        a.set_index("fecha_hora")["temperatura_c"],
        b.set_index("fecha_hora")["temperatura_c"],
    )


# ==========================================================================
# 6. Ausencia de fuga temporal
# ==========================================================================
#
# El riesgo: etiquetar el pasado con conocimiento del futuro. No falla nunca;
# solo hace que la validacion salga mejor de lo que sera en produccion.


def _serie_con_estacionalidad(dias: int = 420) -> pd.DataFrame:
    import random

    aleatorio = random.Random(11)
    momentos = pd.date_range("2025-01-06", periods=dias * 24, freq="h")
    perfil = [0.85, 0.82, 0.80, 0.79, 0.79, 0.81, 0.86, 0.92, 0.97, 1.00,
              1.02, 1.03, 1.03, 1.04, 1.03, 1.02, 1.02, 1.05, 1.15, 1.18,
              1.16, 1.10, 1.02, 0.93]
    return pd.DataFrame(
        {
            "fecha_hora": momentos,
            "valor_kwh": [
                7e6 * perfil[m.hour] * (1 + aleatorio.gauss(0, 0.02)) for m in momentos
            ],
        }
    )


def test_el_criterio_causal_no_cambia_al_recortar_el_futuro():
    """La propiedad que define la ausencia de fuga.

    Si una fila se juzga solo con su pasado, su bandera tiene que ser la misma
    tanto si la serie termina hoy como si sigue dos anos mas. Con el criterio
    global no lo es: sobre los datos reales, 165 banderas cambiaban.
    """
    from calidad.diagnostico import z_estacional

    completa = _serie_con_estacionalidad()
    corte = pd.Timestamp("2025-09-01")
    recortada = completa[completa["fecha_hora"] < corte]

    z_completa, _, _ = z_estacional(completa, "fecha_hora", "valor_kwh", causal=True)
    z_recortada, _, _ = z_estacional(recortada, "fecha_hora", "valor_kwh", causal=True)

    hasta_corte = (z_completa.abs() > 3.5).fillna(False)[: len(recortada)]
    solo_pasado = (z_recortada.abs() > 3.5).fillna(False)

    assert list(hasta_corte) == list(solo_pasado), "el criterio causal mira al futuro"


def test_el_criterio_global_si_cambia_al_recortar():
    """Demuestra que la prueba anterior no es trivial."""
    from calidad.diagnostico import z_estacional

    completa = _serie_con_estacionalidad()
    completa.loc[completa.index[-500:], "valor_kwh"] *= 3  # el futuro cambia mucho
    corte = pd.Timestamp("2025-09-01")
    recortada = completa[completa["fecha_hora"] < corte]

    z_completa, _, _ = z_estacional(completa, "fecha_hora", "valor_kwh", causal=False)
    z_recortada, _, _ = z_estacional(recortada, "fecha_hora", "valor_kwh", causal=False)

    a = (z_completa.abs() > 3.5).fillna(False)[: len(recortada)]
    b = (z_recortada.abs() > 3.5).fillna(False)
    assert list(a) != list(b), "sin fuga, el modo global seria identico al causal"


def test_la_limpieza_marca_con_criterio_causal_por_defecto():
    limpio, registro = L.marcar_atipicos(_serie_con_estacionalidad())
    detalle = registro["detalle"]

    assert detalle["causal"] is True
    assert detalle["usa_informacion_futura"] is False
    # Las columnas globales existen, pero etiquetadas para no usarlas de variable.
    assert "atipico_global" in limpio.columns
    assert "no a la limpieza" in detalle["nota"]


def test_la_imputacion_causal_no_usa_el_valor_posterior():
    marco = _quitar(_serie_horaria(dias=40), "2025-01-15 02:00", 2)

    temporal, reg_t = L.completar_rejilla(marco, metodo="temporal")
    causal, reg_c = L.completar_rejilla(marco, metodo="causal")

    assert reg_t["detalle"]["usa_informacion_futura"] is True
    assert reg_c["detalle"]["usa_informacion_futura"] is False

    # La causal arrastra el ultimo valor observado; la temporal interpola.
    anterior = marco.set_index("fecha_hora")["valor_kwh"].loc["2025-01-15 01:00"]
    obtenido_causal = causal.set_index("fecha_hora")["valor_kwh"]
    assert obtenido_causal.loc["2025-01-15 02:00"] == anterior
    assert obtenido_causal.loc["2025-01-15 03:00"] == anterior
    obtenido_temporal = temporal.set_index("fecha_hora")["valor_kwh"]
    assert obtenido_temporal.loc["2025-01-15 02:00"] != anterior


# ==========================================================================
# 7. Metricas con dimensiones extra (CIIU)
# ==========================================================================
#
# El riesgo: descartar dimensiones que vienen dentro de Values. No falla; deja
# miles de filas con la misma marca de tiempo y sin saber a que pertenecen.

# Recorte textual de la respuesta real de DemaComeNoReg/Entity=CIIU. Las claves
# Activity y Subactivity viven DENTRO de Values, y el Id de la entidad es
# siempre "CIIU": la dimension real no esta donde el resto de metricas la pone.
RESPUESTA_CIIU_REAL: dict[str, Any] = {
    "Metric": {"Id": "DemaComeNoReg", "Name": "Demanda Comercial por CIIU"},
    "Items": [
        {
            "Date": "2025-01-01",
            "HourlyEntities": [
                {
                    "Id": "CIIU",
                    "Values": {
                        "Activity": "ACTIVIDADES ARTÍSTICAS, DE ENTRETENEMIENTO Y RECREACIÓN",
                        "code": "CIIU",
                        "Subactivity": "ARTES PLÁSTICAS Y VISUALES",
                        **{f"Hour{n:02d}": f"{12 + n}.5" for n in range(1, 25)},
                    },
                },
                {
                    "Id": "CIIU",
                    "Values": {
                        "Activity": "ACTIVIDADES ARTÍSTICAS, DE ENTRETENEMIENTO Y RECREACIÓN",
                        "code": "CIIU",
                        "Subactivity": "ACTIVIDADES DE BIBLIOTECAS Y ARCHIVOS",
                        **{f"Hour{n:02d}": f"{99 + n}.5" for n in range(1, 25)},
                    },
                },
            ],
        }
    ],
}


def test_las_dimensiones_de_ciiu_se_conservan():
    """Antes se perdian: Activity y Subactivity no llegaban al DataFrame."""
    tabla = ClienteXM(sesion=mock.MagicMock())._a_dataframe(
        RESPUESTA_CIIU_REAL, "DemaComeNoReg"
    )

    assert "Activity" in tabla.columns
    assert "Subactivity" in tabla.columns
    assert tabla["Subactivity"].nunique() == 2
    assert tabla["Activity"].notna().all()


def test_ciiu_no_deja_duplicados_sobre_su_clave_real():
    """El Id de la entidad no distingue nada: la clave la dan las dimensiones."""
    tabla = ClienteXM(sesion=mock.MagicMock())._a_dataframe(
        RESPUESTA_CIIU_REAL, "DemaComeNoReg"
    )

    assert len(tabla) == 2 * 24
    # Por marca de tiempo hay dos filas, una por subactividad: eso es correcto.
    assert tabla["timestamp"].duplicated().sum() == 24
    # Sobre la clave completa no puede repetirse ninguna.
    clave = ["timestamp", "Activity", "Subactivity"]
    assert not tabla.duplicated(subset=clave).any()


def test_una_metrica_normal_no_gana_columnas():
    """La correccion no debe cambiar el resto de metricas."""
    tabla = convertir()

    assert list(tabla.columns) == ClienteXM.COLUMNAS
    assert len(tabla) == 24
    assert tabla["timestamp"].duplicated().sum() == 0
