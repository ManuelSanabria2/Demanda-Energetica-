"""Reglas de asociacion (Apriori) sobre los dias: grupo, calendario, nivel y clima.

    from mineria.reglas import transacciones, reglas_asociacion

    canasta = transacciones(atributos, resultado.etiquetas, nivel)
    reglas = reglas_asociacion(canasta)

Cada dia es una transaccion y sus items son hechos binarios: a que grupo de
forma pertenece, que tipo de dia es, si es puente, Semana Santa o fin de ano,
si su nivel de demanda es alto o bajo y si hizo calor o frio.

Dos decisiones que importan:

- **Terciles de nivel dentro de cada ano.** La demanda crece de un ano a otro;
  con terciles globales, "nivel alto" seria casi sinonimo de "2025-2026" y las
  reglas hablarian de la tendencia, no del tipo de dia.
- **Lift, no solo confianza.** "laboral -> G1" puede tener confianza alta solo
  porque G1 es el grupo mas grande. El lift compara contra esa base: por encima
  de 1 la asociacion es real; cerca de 1, casualidad de frecuencias.
"""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

SOPORTE_MINIMO = 0.02
CONFIANZA_MINIMA = 0.60
LIFT_MINIMO = 1.20


def _terciles_por_anio(valores: pd.Series, anios: pd.Series, etiquetas: tuple[str, str, str]) -> pd.Series:
    def cortar(grupo: pd.Series) -> pd.Series:
        return pd.qcut(grupo.rank(method="first"), 3, labels=list(etiquetas)).astype(str)

    return valores.groupby(anios, group_keys=False).apply(cortar).reindex(valores.index)


def transacciones(
    atributos: pd.DataFrame, etiquetas: pd.Series, nivel: pd.Series
) -> pd.DataFrame:
    """Matriz booleana dia x item, lista para Apriori."""
    datos = atributos.join(etiquetas.rename("grupo")).join(nivel.rename("nivel_mwh"))
    items: dict[str, pd.Series] = {}

    for grupo in sorted(datos["grupo"].dropna().unique()):
        items[f"grupo={grupo}"] = datos["grupo"] == grupo
    for tipo in ("laboral", "sabado", "domingo", "festivo"):
        items[f"dia={tipo}"] = datos["tipo_dia"] == tipo
    for bandera, nombre in (
        ("es_puente", "puente"),
        ("es_semana_santa", "semana_santa"),
        ("es_ultima_semana_diciembre", "fin_de_anio"),
        ("es_vispera_festivo", "vispera_festivo"),
    ):
        if bandera in datos:
            items[nombre] = datos[bandera].fillna(False).astype(bool)

    niveles = _terciles_por_anio(datos["nivel_mwh"], datos["anio"], ("bajo", "medio", "alto"))
    for etiqueta in ("bajo", "alto"):
        items[f"nivel={etiqueta}"] = niveles == etiqueta

    if "temperatura_c" in datos and datos["temperatura_c"].notna().any():
        temperatura = _terciles_por_anio(
            datos["temperatura_c"].fillna(datos["temperatura_c"].median()), datos["anio"],
            ("fria", "templada", "calida"),
        )
        for etiqueta in ("fria", "calida"):
            items[f"temp={etiqueta}"] = temperatura == etiqueta

    return pd.DataFrame(items, index=datos.index).astype(bool)


def reglas_asociacion(
    canasta: pd.DataFrame,
    soporte_minimo: float = SOPORTE_MINIMO,
    confianza_minima: float = CONFIANZA_MINIMA,
    lift_minimo: float = LIFT_MINIMO,
    max_items: int = 3,
) -> pd.DataFrame:
    """Reglas con un solo consecuente, ordenadas por lift y confianza."""
    frecuentes = apriori(canasta, min_support=soporte_minimo, use_colnames=True, max_len=max_items)
    if frecuentes.empty:
        return pd.DataFrame(columns=["antecedente", "consecuente", "soporte", "confianza", "lift"])
    reglas = association_rules(frecuentes, metric="confidence", min_threshold=confianza_minima)
    reglas = reglas[(reglas["lift"] >= lift_minimo) & (reglas["consequents"].map(len) == 1)]
    tabla = pd.DataFrame(
        {
            "antecedente": reglas["antecedents"].map(lambda s: " & ".join(sorted(s))),
            "consecuente": reglas["consequents"].map(lambda s: next(iter(s))),
            "soporte": reglas["support"].round(4),
            "confianza": reglas["confidence"].round(4),
            "lift": reglas["lift"].round(3),
            "n_items": reglas["antecedents"].map(len) + 1,
        }
    )
    return tabla.sort_values(["lift", "confianza", "soporte"], ascending=False).reset_index(drop=True)


def podar_redundantes(reglas: pd.DataFrame) -> pd.DataFrame:
    """Quita las reglas que no anaden nada a otra mas simple.

    Una regla A & B -> C es redundante si ya existe A -> C con confianza igual o
    mayor: el item B no aporta. Sin esta poda, "domingo -> G2" aparece repetida
    como "domingo & fresco -> G2", "domingo & nivel bajo -> G2", etc.
    """
    if reglas.empty:
        return reglas
    conjuntos = reglas["antecedente"].map(lambda a: frozenset(a.split(" & ")))
    conservar = []
    for i, fila in reglas.iterrows():
        propio = conjuntos[i]
        redundante = any(
            conjuntos[j] < propio and reglas.at[j, "consecuente"] == fila["consecuente"]
            and reglas.at[j, "confianza"] >= fila["confianza"] - 1e-9
            for j in reglas.index if j != i
        )
        conservar.append(not redundante)
    return reglas[conservar].reset_index(drop=True)


def reglas_hacia(reglas: pd.DataFrame, prefijo: str) -> pd.DataFrame:
    """Solo las reglas cuyo consecuente empieza por `prefijo` (p. ej. 'grupo=')."""
    return reglas[reglas["consecuente"].str.startswith(prefijo)].reset_index(drop=True)
