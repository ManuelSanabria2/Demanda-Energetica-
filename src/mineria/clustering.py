"""K-means sobre la forma del dia: que tipos de dia tiene la demanda del SIN.

    from mineria.clustering import evaluar_k, elegir_k, agrupar, describir

    evaluacion = evaluar_k(forma.to_numpy(), ks=range(2, 11))
    k = elegir_k(evaluacion)
    resultado = agrupar(forma, k)
    tabla = describir(resultado, forma, nivel, atributos)

Decisiones, y por que:

- **Solo la forma entra al algoritmo.** El calendario y la temperatura se usan
  despues, para leer los grupos. Si el calendario entrara, los grupos lo
  reproducirian por construccion y no se aprenderia nada de ellos.
- **k por silueta, con la estabilidad como freno.** Se elige el k de mayor
  silueta entre los que reparten de forma estable (ARI medio entre semillas
  y submuestras >= `ARI_ESTABLE`). Un k con buena silueta pero que cambia con
  la semilla no describe una estructura de los datos.
- **Contraste con Ward.** Si un metodo jerarquico con otro criterio llega a una
  particion parecida (ARI alto), los grupos no son un artefacto de K-means.
- **Etiquetas deterministas.** Los grupos se numeran por tamano (G1 el mayor),
  para que las figuras y el texto no cambien al volver a ejecutar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

SEMILLA = 42
ARI_ESTABLE = 0.80
N_INIT = 20


@dataclass
class Resultado:
    """Particion final y todo lo necesario para describirla y graficarla."""

    k: int
    etiquetas: pd.Series                 # indice fecha -> "G1".."Gk"
    centroides: pd.DataFrame             # una fila por grupo, columnas h00..h23
    distancia: pd.Series                 # distancia de cada dia a su centroide
    inercia: float
    silueta: float
    ari_ward: float
    pca: pd.DataFrame                    # componentes 1 y 2 por dia
    varianza_pca: list[float]
    registro: dict[str, Any] = field(default_factory=dict)


def _kmeans(x: np.ndarray, k: int, semilla: int = SEMILLA) -> KMeans:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=semilla).fit(x)


def estabilidad(
    x: np.ndarray, k: int, repeticiones: int = 10, fraccion: float = 0.8, semilla: int = SEMILLA
) -> float:
    """ARI medio entre la particion de referencia y las de submuestras con otra semilla.

    Cada repeticion ajusta K-means sobre el `fraccion` de los dias con otra
    semilla, predice todos los dias y compara con la referencia. Un ARI cercano
    a 1 dice que la particion no depende de la inicializacion ni de que dias
    concretos entraron.
    """
    referencia = _kmeans(x, k, semilla).labels_
    generador = np.random.default_rng(semilla)
    valores = []
    for r in range(repeticiones):
        indices = generador.choice(len(x), size=max(k + 1, int(fraccion * len(x))), replace=False)
        modelo = KMeans(n_clusters=k, n_init=5, random_state=semilla + 1 + r).fit(x[indices])
        valores.append(adjusted_rand_score(referencia, modelo.predict(x)))
    return float(np.mean(valores))


def evaluar_k(
    x: np.ndarray, ks: Iterable[int] = range(2, 11), semilla: int = SEMILLA, con_estabilidad: bool = True
) -> pd.DataFrame:
    """Inercia (codo), silueta, Davies-Bouldin, Calinski-Harabasz y estabilidad por k."""
    filas = []
    for k in ks:
        if k >= len(x):
            break
        modelo = _kmeans(x, k, semilla)
        etiquetas = modelo.labels_
        filas.append(
            {
                "k": k,
                "inercia": float(modelo.inertia_),
                "silueta": float(silhouette_score(x, etiquetas)),
                "davies_bouldin": float(davies_bouldin_score(x, etiquetas)),
                "calinski_harabasz": float(calinski_harabasz_score(x, etiquetas)),
                "tamano_minimo": int(np.bincount(etiquetas).min()),
                "estabilidad_ari": estabilidad(x, k, semilla=semilla) if con_estabilidad else np.nan,
            }
        )
    return pd.DataFrame(filas)


def elegir_k(evaluacion: pd.DataFrame, ari_minimo: float = ARI_ESTABLE, tamano_minimo: int = 10) -> int:
    """Mayor silueta entre los k estables y sin grupos diminutos.

    Si ninguno cumple, el de mayor silueta a secas: la regla no debe impedir
    el analisis, y el registro deja constancia de que no hubo k estable.
    """
    candidatos = evaluacion[
        (evaluacion["estabilidad_ari"].fillna(1.0) >= ari_minimo)
        & (evaluacion["tamano_minimo"] >= tamano_minimo)
    ]
    tabla = candidatos if len(candidatos) else evaluacion
    return int(tabla.sort_values(["silueta", "k"], ascending=[False, True]).iloc[0]["k"])


def agrupar(forma: pd.DataFrame, k: int, semilla: int = SEMILLA) -> Resultado:
    """K-means final con etiquetas ordenadas por tamano, Ward y PCA para graficar."""
    x = forma.to_numpy(dtype=float)
    modelo = _kmeans(x, k, semilla)
    crudas = modelo.labels_

    tamanos = np.bincount(crudas, minlength=k)
    orden = np.argsort(-tamanos, kind="stable")              # mayor primero
    nuevo = {int(viejo): i for i, viejo in enumerate(orden)}
    numeros = np.array([nuevo[int(e)] for e in crudas])
    nombres = np.array([f"G{n + 1}" for n in numeros])

    centroides = pd.DataFrame(
        modelo.cluster_centers_[orden], columns=forma.columns, index=[f"G{i + 1}" for i in range(k)]
    )
    distancia = np.linalg.norm(x - modelo.cluster_centers_[crudas], axis=1)

    ward = fcluster(linkage(x, method="ward"), t=k, criterion="maxclust")
    ari_ward = float(adjusted_rand_score(numeros, ward))

    pca = PCA(n_components=2, random_state=semilla).fit(x)
    componentes = pd.DataFrame(pca.transform(x), index=forma.index, columns=["pc1", "pc2"])

    resultado = Resultado(
        k=k,
        etiquetas=pd.Series(nombres, index=forma.index, name="grupo"),
        centroides=centroides,
        distancia=pd.Series(distancia, index=forma.index, name="distancia"),
        inercia=float(modelo.inertia_),
        silueta=float(silhouette_score(x, numeros)) if k > 1 else float("nan"),
        ari_ward=ari_ward,
        pca=componentes,
        varianza_pca=[float(v) for v in pca.explained_variance_ratio_],
    )
    resultado.registro = {
        "operacion": "kmeans_forma_diaria",
        "criterio": (
            "K-means (n_init=%d, semilla=%d) sobre la forma del dia (cada dia dividido "
            "por su media); calendario y clima no entran al algoritmo" % (N_INIT, semilla)
        ),
        "k": k,
        "dias": int(len(x)),
        "tamanos": {f"G{i + 1}": int(tamanos[o]) for i, o in enumerate(orden)},
        "silueta": resultado.silueta,
        "ari_ward": ari_ward,
        "varianza_pca": resultado.varianza_pca,
    }
    return resultado


def v_cramer(tabla: pd.DataFrame) -> float:
    """Asociacion entre dos categoricas a partir de su tabla de contingencia (0..1)."""
    chi2 = chi2_contingency(tabla.to_numpy(), correction=False)[0]
    n = tabla.to_numpy().sum()
    r, c = tabla.shape
    if n == 0 or min(r, c) < 2:
        return float("nan")
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def describir(
    resultado: Resultado, forma: pd.DataFrame, nivel: pd.Series, atributos: pd.DataFrame
) -> pd.DataFrame:
    """Una fila por grupo: tamano, forma del centroide y composicion de calendario."""
    datos = atributos.join(resultado.etiquetas).join(nivel)
    filas = []
    for grupo, centroide in resultado.centroides.iterrows():
        sub = datos[datos["grupo"] == grupo]
        perfil = centroide.to_numpy()
        fila: dict[str, Any] = {
            "grupo": grupo,
            "dias": int(len(sub)),
            "pct_dias": round(100 * len(sub) / len(datos), 2),
            "nivel_medio_mwh": round(float(sub["nivel_mwh"].mean()), 1),
            "hora_pico": int(np.argmax(perfil)),
            "hora_valle": int(np.argmin(perfil)),
            "pico_sobre_media": round(float(perfil.max()), 3),
            "valle_sobre_pico": round(float(perfil.min() / perfil.max()), 3),
            "factor_carga": round(float(perfil.mean() / perfil.max()), 3),
        }
        for tipo in ("laboral", "sabado", "domingo", "festivo"):
            fila[f"pct_{tipo}"] = round(100 * float((sub["tipo_dia"] == tipo).mean()), 1)
        for bandera in ("es_semana_santa", "es_ultima_semana_diciembre", "es_puente"):
            if bandera in sub:
                fila[f"pct_{bandera}"] = round(100 * float(sub[bandera].astype(bool).mean()), 1)
        if "temperatura_c" in sub:
            fila["temperatura_media_c"] = round(float(sub["temperatura_c"].mean()), 2)
        fila["dia_mas_comun"] = sub["nombre_dia"].mode().iat[0] if len(sub) else None
        filas.append(fila)
    return pd.DataFrame(filas).set_index("grupo")


def contingencia(resultado: Resultado, atributos: pd.DataFrame, columna: str = "tipo_dia") -> pd.DataFrame:
    """Tabla grupo x categoria (por defecto, tipo de dia)."""
    datos = atributos[[columna]].join(resultado.etiquetas)
    return pd.crosstab(datos["grupo"], datos[columna])


def dias_atipicos(resultado: Resultado, atributos: pd.DataFrame, cuantil: float = 0.99) -> pd.DataFrame:
    """Dias cuya forma esta lejos de su propio centroide (cola superior de la distancia).

    No son errores de datos (la limpieza ya los habria marcado) sino dias con
    una curva poco comun para su grupo: el caso tipico es un festivo que cae
    entre semana o un dia de evento.
    """
    umbral = float(resultado.distancia.quantile(cuantil))
    lejos = resultado.distancia[resultado.distancia > umbral].sort_values(ascending=False)
    columnas = [c for c in ("nombre_dia", "tipo_dia", "nombre_festivo") if c in atributos]
    tabla = pd.DataFrame({"grupo": resultado.etiquetas.reindex(lejos.index), "distancia": lejos.round(4)})
    return tabla.join(atributos[columnas])
