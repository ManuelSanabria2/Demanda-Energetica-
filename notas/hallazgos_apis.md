# Hallazgos verificados sobre las APIs de XM y SIMEM

Todo lo de este documento está **medido contra el servidor**, no inferido de la
documentación. Fecha del sondeo: **2026-09-09**. Reproducible con
`python scripts/verificar_apis.py`.

Ambas APIs pertenecen a XM S.A. E.S.P., así que comparten fuente primaria pero
**no** comparten formato, convenciones de error ni límites.

---

## 1. API XM (SINERGOX)

`POST https://servapibi.xm.com.co/{hourly|daily|monthly|lists}`
Cuerpo: `{"MetricId", "StartDate", "EndDate", "Entity"}`

### Forma de la respuesta horaria

```json
{"Metric": {"Id": "DemaReal", "Name": "Demanda Real por Sistema",
            "StartDate": "2025-01-01T00:00:00", "EndDate": "2025-01-02T00:00:00"},
 "Items": [{"Date": "2025-01-01",
            "HourlyEntities": [{"Id": "Sistema",
                                "Values": {"code": "Sistema",
                                           "Hour01": "7306335.34000",
                                           "...": "...",
                                           "Hour24": "7925012.28000"}}]}]}
```

### Lo que hay que saber

| Hallazgo | Detalle |
|---|---|
| Tipo de los valores | **Cadenas**, no números (`'7306335.34000'`). Conversión explícita obligatoria. |
| Límite por llamado | **31 días para `DemaReal`, no 30.** Varía por métrica. |
| Descubrimiento del límite | `POST /lists` con `{"MetricId":"ListadoMetricas","Entity":"Sistema"}` → 193 métricas con `MaxDays`, `MetricUnits`, `Type`, `Url`. El código lee `MaxDays` de ahí en vez de asumirlo. |
| Error de rango excedido | `400` con `Content-Type: application/json`, pero el cuerpo es una **cadena JSON suelta**: `"error: Rango de consulta supera..."`. `r.json()` **no** falla — devuelve un `str`, y es el `.get()` posterior el que revienta con `AttributeError`. |
| Fechas sin datos | `200` con `"Items": []`. **No es un error**: el hueco es silencioso y solo se detecta comparando contra el calendario esperado. |
| Rezago de publicación | **3 días.** Al 2026-09-09 el último dato de `DemaReal`, `DemaCome` y `Gene` era 2026-09-06. |

### Métricas horarias de demanda confirmadas

`DemaReal`, `DemaCome`, `DemaRealReg`, `DemaRealNoReg`, `DemaComeReg`,
`DemaComeNoReg` — todas en kWh, `MaxDays 31`, entidades `Sistema` y `Agente`.

`DemaReal` = demanda de usuarios regulados y no regulados del SIN,
**sin incluir alumbrado público** (según su propia `MetricDescription`).

---

## 2. API SIMEM

`GET https://www.simem.co/backend-files/api/PublicData?datasetId=&startDate=&endDate=`

### Forma de la respuesta

```json
{"parameters": {...},
 "success": true,
 "result": {"idDataset": "14fabb", "name": "Demanda real nacional",
            "metadata": {...}, "filterDate": ..., "records": [...],
            "variables": [...], "columns": null, "tags": [...]}}
```

`result.records` es **formato largo**; las columnas de dimensión cambian según
el dataset. Para `14fabb`:

```json
{"CodigoVariable": "DdaReal", "FechaHora": "2026-08-05 01:00:00",
 "CodigoSICAgente": "CHVC", "TipoMercado": "No Regulado", "Version": "TX2",
 "Valor": 62463.18, "UnidadMedida": "kWh", "CodigoDuracion": "PT1H"}
```

### Lo que hay que saber

| Hallazgo | Detalle |
|---|---|
| **Dos formatos de error distintos** | Sin `datasetId` → JSON RFC9110 de ASP.NET con clave `errors`. `datasetId` inválido → `{"status": false, "message": "Error: El conjunto de datos indicado no existe"}`. |
| Catálogo de datasets | Es a su vez un dataset: **`e007fb`** (388 conjuntos, con `idDataset`, `nombreConjuntoDatos`, `inicioDato`, `finDato`). El inventario de variables es `a5a6c4`. |
| Dataset deprecado | **`c1b851` está en deprecación**, finaliza 2026-06-30. Reemplazo nacional: **`14fabb` "Demanda real nacional"**, horaria, desde 2021-01-01. |
| Paginación | No hay. La respuesta llega completa: ~27 MB por mes de `14fabb`, ~24 MB por **2 días** de `c1b851`. Hay que volcar a disco en streaming. |
| Marcas de tiempo | `FechaHora` de `00:00:00` a `23:00:00`. Colombia no aplica DST, así que no hay días de 23 ni de 25 horas. |

### ⚠️ Trampa crítica: versiones de liquidación coexistentes

El número de versiones simultáneas **cambia con la antigüedad del dato**:

| Mes | Versiones sobre la misma `FechaHora` | Factor de duplicación si se suma sin filtrar |
|---|---|---|
| 2021-01 | `TX4` | 1× (sin riesgo) |
| 2023-06 | `TX5` | 1× (sin riesgo) |
| 2026-05 | `TX2`, `TX3`, `TXR`, `TXF` | **4×** |
| 2026-08 | `TX2`, `TXR` | **2×** |

Un `groupby("FechaHora")["Valor"].sum()` ingenuo multiplica la demanda por el
número de versiones presentes, y el resultado **sigue pareciendo una curva de
demanda plausible**: la forma es correcta, solo el nivel está mal, y el factor
ni siquiera es constante a lo largo del histórico. Es el fallo silencioso más
peligroso del proyecto.

Sobre el histórico completo 2021→2026 aparecen **nueve** versiones distintas:
`TX2, TX3, TX4, TX5, TX6, TX7, TX8, TXF, TXR`. La ingesta pasa de 7 400 928
filas crudas a 4 430 304 tras colapsar.

`normalizar.simem_colapsar_versiones()` elige **una sola versión por
`FechaHora`** según `config.PRECEDENCIA_VERSIONES` y **falla con excepción** si
quedan duplicados o si aparece una versión sin precedencia definida. Nunca
continúa en silencio — de hecho fue esa excepción la que reveló las versiones
`TX5`–`TX8` en la primera ingesta histórica.

### ¿Importa cuál versión se elige?

Medido sobre mayo de 2026, donde coexisten las cuatro:

```
Total mensual por version (GWh):   TX2 7.506   TXR 7.506   TX3 7.507   TXF 7.508
```

**Menos de un 0.05 % de diferencia entre versiones.** El orden exacto de la
precedencia es, para efectos del modelo, irrelevante; lo que importa es
colapsar. El orden elegido (`TXF > TXR > TX8 … > TX1`) toma TXR y TXF como
revisiones posteriores a las liquidaciones numeradas. **No está confirmado
contra documentación oficial de XM**: es una decisión del proyecto, registrada
aquí para que sea revisable.

---

## 3. Convención `HourNN` — resuelta empíricamente

La respuesta de XM no documenta a qué hora de reloj corresponde `Hour01`. En
vez de asumirlo, se midió comparando la serie de XM contra la de SIMEM (misma
fuente primaria) para agosto de 2026, probando tres desfases:

```
 desfase      n   correlacion   error medio %
      -1    743      0.948753           3.357
       0    744      0.991236           1.897
       1    743      0.952222           3.185
```

**Desfase óptimo = 0** con `config.DESFASE_HORA_XM = 1`, es decir:
**`Hour01` es el intervalo 00:00–01:00.** La convención queda confirmada.

Reproducible con `python scripts/reconciliar.py --mes 2026-08`.

### Contraste sobre un mes ya liquidado

Repetida la reconciliación sobre **marzo de 2025**, un mes cerrado en el que
`14fabb` trae una única versión (`TX4`):

```
 desfase      n   correlacion   error medio %
      -1    743      0.952317           2.929
       0    744      0.999614           0.470
       1    743      0.953245           2.918
```

**Correlación 0.9996 y 0.47 % de error**, frente a 0.9912 y 1.90 % en agosto de
2026. Esto resuelve buena parte de la duda que quedaba abierta: la discrepancia
entre las dos fuentes se debe en gran medida a las **versiones preliminares**,
no a un desajuste estructural. Sobre datos ya liquidados las dos series
prácticamente coinciden, y el desfase óptimo sigue siendo 0.

Consecuencia práctica para el modelado: los meses recientes traen un objetivo
algo más ruidoso que los históricos, y el ruido se reduce solo cuando XM
publica la liquidación definitiva.

### Contraste sobre el histórico completo

Comparando las dos series ya ingestadas, 49 752 horas de 2021-01-01 a 2026-09-04:

```
correlacion   : 0.992516
dif. media    : +1.438 %    mediana +0.558 %
dif. p5 / p95 : +0.277 %  / +5.185 %

diferencia media por anio:  2021 +0.88   2022 +1.06   2023 +2.15
                            2024 +2.09   2025 +1.25   2026 +1.09
```

XM está **sistemáticamente por encima** de SIMEM, con signo estable en los seis
años. Un desalineamiento horario produciría diferencias de signo alternante, no
un sesgo constante: esto confirma que el alineamiento es correcto y que la
brecha es de **alcance**, no de tiempo.

Causas plausibles, pendientes de confirmar:

- `DemaReal` de XM excluye explícitamente el alumbrado público; no está
  confirmado que `14fabb` lo excluya igual.
- El salto a ~2.1 % en 2023–2024 frente a ~1.0 % en el resto sugiere un cambio
  de perímetro o de agentes reportantes en esos años, no ruido. Como esos meses
  ya están liquidados, la explicación de las versiones preliminares no aplica
  ahí: queda como pregunta abierta.

---

## 4. Pendientes de verificación

- Qué devuelve XM cuando falta **una hora suelta** dentro de un día publicado
  (¿clave ausente, `null`, o `"0"`?). El parser trata las tres como `NaN`, pero
  no se ha observado el caso real.
- Regla **oficial** de precedencia entre versiones de liquidación de XM
  (impacto medido: <0.05 % del total, ver §2).
- Si `14fabb` incluye o no alumbrado público, y a qué se debe el salto de
  discrepancia en 2023–2024 (ver §3).
