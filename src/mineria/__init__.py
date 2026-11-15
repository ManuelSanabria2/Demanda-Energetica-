"""Mineria no supervisada sobre la demanda horaria del SIN.

Tres analisis, cada uno con su registro, igual que la limpieza y las variables:

- `perfiles`: del panel horario a una matriz dia x hora (forma y nivel).
- `clustering`: K-means sobre la forma del dia, con seleccion de k, estabilidad,
  contraste con Ward y lectura de los grupos contra el calendario.
- `reglas`: reglas de asociacion (Apriori) entre grupos, calendario, nivel de
  demanda y temperatura.
- `sectores`: agrupamiento de los sectores CIIU por su perfil semanal.

Es analisis descriptivo: usa la serie completa a proposito. Las etiquetas de
grupo **no** son variables del modelo; si algun dia lo fueran, habria que
recalcularlas solo con el pasado de cada fila.
"""

__all__ = ["perfiles", "clustering", "reglas", "sectores"]
