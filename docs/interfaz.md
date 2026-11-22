# Interfaz de la sustentación

Aplicación de Streamlit que recorre el proyecto en el orden de CRISP-DM, del
problema al modelo final. No entrena ni descarga datos: lee los artefactos del
pipeline y, en *Probar el modelo*, carga el modelo final ya entrenado.

- **En línea:** <https://demanda-sin-colombia.streamlit.app> (subdominio que se
  fija en el paso 4 del despliegue).
- **En local:** `streamlit run app/Inicio.py` (ver [Arrancarla en local](#arrancarla-en-local)).
- **Código:** `app/` (páginas, componentes y datos exportados), `scripts/exportar_interfaz.py`
  y `tests/test_interfaz.py`. Detalles en el README, sección *Interfaz para la sustentación*.

---

## 1. Despliegue en Streamlit Community Cloud

### Antes de desplegar

1. **La rama tiene que estar en GitHub.** Community Cloud despliega desde una rama del
   repositorio público `ManuelSanabria2/Demanda-Energetica-`. Lo normal es desplegar desde
   `main` después de fusionar el PR de `interfaz`. Si el PR aún no está fusionado, se puede
   desplegar desde la rama `interfaz` (hay que haberla subido con `git push -u origin interfaz`).
2. **Los artefactos tienen que estar al día.** Si cambió el panel o algún resultado del
   modelado, regenera y confirma `app/datos/` antes de subir:

   ```bash
   python scripts/exportar_interfaz.py
   python -m pytest tests/test_interfaz.py -q      # comprueba que app/datos cuadra con docs/resultados
   ```

3. **Nada de lo que la app necesita está fuera de git.** La app lee `docs/resultados/`,
   `docs/figuras/`, `docs/modelado/seleccion.md`, `infra/costos/costos.json`, los registros
   de `data/procesado/` y `app/datos/` (incluidos los dos modelos finales, el mayor de
   19,8 MB). Todo eso está versionado; `data/`, `models/` y `datasets/` no hacen falta.

### Pasos

1. Entra en <https://share.streamlit.io> con la cuenta de GitHub dueña del repositorio y
   autoriza a Streamlit a leer tus repositorios si te lo pide.
2. Pulsa **Create app** → **Deploy a public app from GitHub** (o *Yup, I have an app*).
3. Rellena el formulario:

   | campo | valor |
   |---|---|
   | Repository | `ManuelSanabria2/Demanda-Energetica-` |
   | Branch | `main` (o `interfaz` mientras el PR no esté fusionado) |
   | Main file path | `app/Inicio.py` |

4. **App URL:** escribe el subdominio `demanda-sin-colombia` para que la dirección sea
   `https://demanda-sin-colombia.streamlit.app`. Si ya está ocupado, elige otro y cambia el
   enlace en los tres sitios donde aparece: este documento, el README (sección *Interfaz para
   la sustentación*) y el informe (sección *Reproducibility*).
5. Abre **Advanced settings** y elige **Python 3.12**. Es obligatorio: pyarrow 14 no tiene
   wheels para 3.13, y **la versión de Python no se puede cambiar después** (habría que borrar
   la app y desplegarla otra vez). No hace falta ningún *secret*. Pulsa **Save**.
6. Pulsa **Deploy**. La primera instalación tarda varios minutos (numba, llvmlite y shap son
   pesados); se ve el progreso en el panel de registros de la derecha.

Qué usa Community Cloud y por qué ya está listo:

- **Dependencias:** busca primero el archivo de dependencias en la carpeta del punto de
  entrada, así que usa `app/requirements.txt` (ligero, versiones fijadas) y no el
  `requirements.txt` de la raíz, que es el del pipeline completo.
- **Tema:** solo lee `.streamlit/config.toml` de la **raíz** del repositorio, que es donde está.
- **Directorio de trabajo:** arranca desde la raíz del repositorio; la app calcula sus rutas
  a partir de su propia ubicación, así que no depende de eso.
- **Recursos:** el plan gratuito da entre 690 MB y 2,7 GB de RAM. En local, recorriendo las
  11 páginas con los dos modelos cargados, el proceso llega a unos 500 MB.

### Comprobar que quedó bien

Abre la app y recorre las 11 páginas del menú lateral. Ninguna debe mostrar un recuadro
amarillo **Pendiente** ni un error rojo. En *Probar el modelo*, el texto bajo las cifras debe
decir «Coincide con la predicción registrada en la evaluación final», en r24 y en r120.

### Si el despliegue falla

1. **Lee los registros.** En la app, abajo a la derecha, **Manage app** abre el panel de
   registros (también desde <https://share.streamlit.io>, menú **⋮** de la app → **Logs**).
   El error real suele estar en las últimas líneas de la instalación o en el *traceback*.
2. **`No matching distribution found for pyarrow==14.0.2` (o numpy, numba) o errores de
   compilación al instalar:** la app se creó con una versión de Python distinta de 3.12. No se
   puede cambiar: en el menú **⋮** → **Settings** apunta el subdominio, borra la app
   (**Delete**) y vuelve a desplegarla con el mismo subdominio y **Python 3.12** en
   *Advanced settings*.
3. **`ModuleNotFoundError` al arrancar:** falta un paquete en `app/requirements.txt`.
   `tests/test_requisitos.py` debería haberlo detectado; añádelo con versión fijada (la misma
   del `requirements.txt` de la raíz), confirma, sube y la app se reinstala sola.
4. **`ImportError: numpy.core.multiarray failed to import` o similar:** se coló numpy 2.
   `app/requirements.txt` fija `numpy==1.26.4`; comprueba que nadie lo quitó.
5. **Una página muestra *Pendiente*:** falta un artefacto en la rama desplegada. El recuadro
   dice qué archivo y qué comando lo genera; genéralo en local, confírmalo y súbelo.
6. **`ModeloAlterado` en *Probar el modelo*:** el modelo de `app/datos/` no es el del
   manifiesto. Vuelve a correr `python scripts/exportar_interfaz.py` y confirma los dos archivos.
7. **La app se queda en *Your app is in the oven* o se reinicia sola:** se quedó sin memoria
   o tardó demasiado en arrancar. Pulsa **Reboot app** en *Manage app*; si se repite, revisa
   en los registros si el proceso fue terminado por memoria.
8. **Cambios que no aparecen:** Community Cloud redespliega con cada `push` a la rama; si no,
   **Manage app** → **Reboot app**.

### Arrancarla en local

```bash
python -m venv .venv                   # Python 3.12; con 3.13 pyarrow 14 no se instala
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app/Inicio.py
```

Se abre en <http://localhost:8501>. Con solo `app/requirements.txt` también funciona (es lo
que instala la nube; se verificó en un entorno limpio).

---

## 2. Guion de la sustentación (unos 10 minutos)

Antes de empezar: la app abierta en **Inicio**, en pantalla completa (F11), con el menú
lateral visible. Las cifras se leen de la pantalla; el guion dice dónde mirar, no las repite.

| min | página | qué decir (2–3 frases) | qué hacer |
|---|---|---|---|
| 0:00 | **Inicio** | La planeación eléctrica en Colombia reacciona tarde a la demanda; nuestro objetivo es pronosticar la demanda horaria del SIN con 24 h de antelación (ODS 9.1). La pregunta de minería es si se puede hacer con un error en horas pico útil para planear. | Señalar las cuatro cifras clave. Cambiar de **r24** a **r120** y leer la frase de abajo: lo que cuesta el rezago de publicación. |
| 1:00 | **Datos** | Los datos son públicos: demanda de XM, clima de Open-Meteo y festivos; cada valor es trazable hasta su fuente. La demanda se publica con días de rezago, por eso hay dos versiones de la tabla. | Señalar el diagrama de flujo y luego el gráfico **r24 frente a r120**: pasar el ratón por los rombos de `lag_24` y `lag_168`. |
| 2:00 | **Exploración** | La serie tiene tendencia, ciclo semanal y caídas en festivos. La autocorrelación a 168 h es mayor que a 24 h: la misma hora de la semana pasada es el mejor predictor simple, y por eso es el baseline. | En la serie diaria, botón **1 año**; activar **Colorear por tipo de día**. Bajar a la autocorrelación y señalar el pico naranja de 168 h. |
| 3:00 | **Minería no supervisada** | El K-means sobre la forma del día encuentra dos tipos de día sin que le digamos el calendario, y coinciden con laboral frente a domingo y festivo. Las reglas de asociación y los sectores CIIU describen quién mueve la carga. | Señalar los dos centroides. Pestaña **Reglas de asociación**: escribir `festivo` en el filtro. |
| 4:00 | **Modelos** | Probamos baselines, modelos lineales y ensembles, siempre con validación temporal y sin barajar. La línea discontinua es el baseline; todo lo que queda en la zona verde lo mejora al menos un 20 %. | Señalar la línea del baseline y la zona verde. Pestaña **Error por pliegue**: una caja estrecha es un modelo estable. |
| 5:00 | **Predicción frente a real** | Esta es la prueba de verdad: un año que el modelo nunca vio. La curva naranja sigue a la real; la gris discontinua es el baseline. | Pulsar el atajo de la semana de **abril** (la de después de Semana Santa): el baseline copia la semana festiva y se hunde, el modelo no. Luego el atajo de **agosto**, la peor del modelo: la demanda superó todo lo visto en entrenamiento y un modelo de árboles no extrapola. |
| 6:30 | **Alertas de pico** | Además de pronosticar, avisamos las horas que serán excepcionalmente altas. Un pico sin avisar cuesta más que una falsa alarma. | Mover el **deslizador de umbral** hacia abajo y mostrar cómo sube el recall y bajan la precisión en la matriz y en la curva. |
| 7:30 | **Interpretabilidad** | No es una caja negra: SHAP dice cuánto aporta cada variable y LIME explica horas concretas. La temperatura pesa distinto según la hora, como anticipaba el EDA. | Pestaña **LIME**: leer la frase del caso **Festivo**. |
| 8:15 | **Selección y costo** | El modelo se eligió en validación, con una regla fijada antes de mirar la prueba. Con los datos realmente disponibles (r120) el error sube, y el modelo tiende a quedarse corto, que es el error caro. | Señalar el recuadro verde del criterio de éxito y el gráfico **r24 frente a r120**. |
| 9:00 | **Probar el modelo** | El modelo responde en vivo. Esta es la hora de mayor demanda del año: SHAP muestra de dónde sale la predicción. | Bajar al **¿Qué pasaría si…?**: subir la temperatura y marcar **Festivo**; leer el aviso de que es sensibilidad, no causalidad. |
| 9:45 | **Nube** | Así operaría en AWS, definido con Terraform, por menos de un dólar al mes en el escenario académico. | Señalar el diagrama y las tres tarjetas de costo. Cierre. |

Si sobra tiempo o preguntan: *Modelos* → **Grid frente a Random**; *Selección y costo* →
**Documento de selección completo**; *Datos* → **Historial del registro de limpieza**.

---

## 3. Lista de verificación para el día de la sustentación

**La víspera**

- [ ] La rama desplegada tiene el último commit y la suite pasa
      (`python -m pytest tests -q`).
- [ ] Recorrer las 11 páginas en la app en línea: ninguna con *Pendiente* ni errores, y
      *Probar el modelo* dice «Coincide con la predicción registrada».
- [ ] Probar la versión local **sin internet** (desconectar el wifi y abrir
      `streamlit run app/Inicio.py`): Streamlit sirve todo desde el equipo.
- [ ] Generar el plan B en PDF (ver abajo) y copiarlo a una memoria USB y al correo.
- [ ] Ensayar el guion con cronómetro.

**Una hora antes**

- [ ] **Abrir la app en línea.** Community Cloud la duerme tras 12 h sin visitas; si sale
      *This app has gone to sleep*, pulsar **Yes, get this app back up!** y esperar a que
      cargue (puede tardar unos minutos). Recorrer las páginas una vez para que queden en caché.
- [ ] **Tener la versión local abierta** en otra pestaña:
      `.venv/Scripts/python.exe -m streamlit run app/Inicio.py` (<http://localhost:8501>).
      Si la red de la sala falla, se sigue desde ahí sin cambiar nada del guion.
- [ ] Cerrar notificaciones, subir el zoom del navegador si la sala es grande (Ctrl + +).

**Plan B sin internet ni app**

- **PDF de cada página.** En la app (local o en línea), abrir cada página, desplegar lo que
  se vaya a mostrar (pestañas, desplegables) e imprimir con **Ctrl + P** → *Guardar como
  PDF* (o menú **⋮** de arriba a la derecha → **Print**). Las pestañas que no estén abiertas
  no salen en el PDF: imprime una vez por pestaña si hace falta. Guardar los 11 archivos numerados en el orden del guion (`01_inicio.pdf`, …,
  `11_nube.pdf`).
- **Notebook 05 ejecutado.** `notebooks/05_modelado_supervisado.ipynb` ya está confirmado con
  sus salidas (tablas y figuras del modelado); se abre en VS Code o Jupyter sin ejecutar nada.
  Para regenerarlo: `python scripts/generar_entrega.py`.
- **Informe PDF.** `docs/informe/avance_proyecto_integrador.pdf`, con todas las figuras.
