He revisado el código y las comprobaciones realizadas. **La base funcional está implementada, pero no daría las cuatro fases por completamente cerradas.** Hay extensiones pendientes y he encontrado dos aspectos concretos del diagnóstico que necesitan corregirse.

**En esta revisión no he modificado código, configuración ni documentación.** Para explicar MASE utilizaré nombres descriptivos: *política sin suelo* y *política con suelo ε*.

## 1. Evaluación temporal: qué ocurre realmente

Hay que distinguir tres decisiones independientes:

1. **Cuándo entrenamos y evaluamos:** los cortes temporales.
2. **Qué modelo conservamos entre cortes:** reinicialización o continuidad.
3. **Qué datos utilizamos para actualizarlo:** histórico acumulado, datos nuevos o ventana reciente.

### 1.1. Calendario común

Todas las estaciones utilizan los mismos cortes. Con la configuración principal, los tres folds son:

| Fold | Targets de entrenamiento | Targets de validación | Targets de test |
|---|---|---|---|
| 1 | Anteriores al 04-01-2026 | Del 04-01 al 01-02 | Del 01-02 al 01-03 |
| 2 | Anteriores al 01-02-2026 | Del 01-02 al 01-03 | Del 01-03 al 29-03 |
| 3 | Anteriores al 01-03-2026 | Del 01-03 al 29-03 | Del 29-03 al 26-04 |

Los extremos derechos son exclusivos.

Una ventana utiliza **90 horas de contexto para predecir 12 horas**. Sus targets deben quedar completamente dentro de su partición:

- Un target de entrenamiento no puede atravesar el comienzo de validación.
- Uno de validación no puede atravesar el comienzo de test.
- Uno de test no puede superar el final del periodo.

El contexto sí puede tomar observaciones del periodo anterior. Por ejemplo, la primera predicción de test puede utilizar las últimas 90 horas conocidas de validación.

### 1.2. Folds independientes frente a continuidad

He implementado ambos protocolos:

| Aspecto | `independent` | `streaming` |
|---|---|---|
| Pesos al comenzar otro fold | Se reinicializan | Se recupera el checkpoint seleccionado del fold anterior |
| Información aprendida previamente | No se conserva | Se conserva en los pesos |
| Normalización | Se ajusta para cada fold | Se mantiene la referencia inicial |
| Qué mide | Reentrenamiento por fecha | Actualización progresiva del modelo |

En Granite, reinicializar significa volver al checkpoint preentrenado con una adaptación nueva, no generar un backbone aleatorio.

**La configuración principal utiliza `streaming`.**

### 1.3. Qué datos nuevos entran

Dentro de `streaming` existen tres opciones:

#### `expanding`: volver a entrenar con todo el pasado disponible

En el segundo fold se conserva el modelo anterior, pero se actualiza utilizando todo el entrenamiento disponible antes del 1 de febrero.

```text
Fold 1: pesos iniciales → entrenamiento hasta 4 de enero
Fold 2: pesos anteriores → entrenamiento hasta 1 de febrero
Fold 3: pesos anteriores → entrenamiento hasta 1 de marzo
```

No se empieza desde cero, aunque se vuelven a visitar datos antiguos.

**Es la opción predeterminada del archivo principal.**

#### `new-only`: actualizar únicamente con los nuevos targets

El primer fold utiliza su histórico inicial. Después:

```text
Fold 2: targets entre 4 de enero y 1 de febrero
Fold 3: targets entre 1 de febrero y 1 de marzo
```

Se conservan los pesos anteriores, pero no se vuelven a utilizar como targets los periodos de entrenamiento ya incorporados.

El contexto de una ventana puede empezar antes de ese intervalo. Eso permite predecir el comienzo del periodo nuevo utilizando el pasado que ya conocemos.

**Esta es la opción utilizada en las pruebas temporales pequeñas.**

#### `sliding`: actualizar con una ventana reciente

Se utilizan los targets de las últimas `history-hours` horas anteriores al corte.

Por ejemplo, con 336 horas:

```text
Cada actualización utiliza las últimas dos semanas de targets disponibles.
```

Esto limita los datos usados para los nuevos gradientes. **No borra lo aprendido anteriormente**, porque los pesos siguen conservándose.

---

## 2. Diferencias entre local, centralized y federated

### 2.1. Local: un modelo independiente por estación

Supongamos dos estaciones, A y B.

En el primer fold:

1. Se crea un modelo para A y otro para B.
2. Cada uno entrena únicamente con su estación.
3. Cada estación evalúa sus checkpoints con su propia validación.
4. Cada una conserva su mejor checkpoint.
5. Se evalúan esos checkpoints en test.

A puede seleccionar la etapa 2 y B la etapa 4. No es necesario que ambas elijan la misma.

En el siguiente fold, con continuidad:

```text
Modelo A del fold 2 ← mejor checkpoint anterior de A
Modelo B del fold 2 ← mejor checkpoint anterior de B
```

Cada modelo incorpora los datos que corresponden a la política de histórico elegida.

**No hay intercambio de pesos ni de datos entre estaciones.**

### 2.2. Centralized: un único modelo con ventanas de todas las estaciones

Aquí hay un solo modelo.

1. Se preparan las ventanas de A, B y las demás estaciones.
2. Se reúnen como muestras de entrenamiento.
3. Se entrena el modelo compartido.
4. Se evalúa por estación.
5. Se agregan las métricas de validación para seleccionar un checkpoint común.

No se concatena el final de la serie de A con el principio de B. Se concatenan **ventanas ya construidas**, manteniendo separadas las series.

En el siguiente fold:

```text
Modelo centralizado del fold 2
    ← mejor checkpoint centralizado del fold 1
    + ventanas del histórico correspondiente
```

Aunque exista un único modelo, se mantiene la normalización por estación. Las predicciones se devuelven después a las unidades de cada estación.

### 2.3. Federated: un modelo global que se actualiza mediante los clientes

En cada ronda:

1. El servidor envía el estado global a los clientes.
2. Cada estación entrena con sus ventanas locales.
3. Los clientes devuelven los parámetros que corresponda agregar.
4. El servidor agrega las actualizaciones.
5. Los clientes evalúan el estado global en validación.
6. Se selecciona el mejor checkpoint global.

Para los modelos clásicos se comparten sus parámetros entrenables. Para Granite se comparten los estados de adaptación, no el backbone completo.

En el siguiente fold:

```text
Estado global inicial del fold 2
    ← mejor checkpoint global del fold 1
```

Los clientes reciben ese estado y entrenan con los datos que ya están disponibles para el nuevo corte.

**Los clientes no conservan optimizadores locales persistentes.** Los pesos continúan, pero AdamW se reinicia por etapa/ronda. Por tanto, esto es continuidad del aprendizaje mediante pesos, no una reanudación exacta de todos los estados de ejecución.

Con las ventanas enmascaradas, la agregación se pondera por los **targets de entrenamiento observados**. Por ejemplo:

- A aporta 900 targets observados.
- B aporta 300.
- Sus pesos de agregación son 0,75 y 0,25.

Estos recuentos corresponden a posiciones de targets en las ventanas; con ventanas solapadas, no equivalen necesariamente a horas distintas.

### 2.4. Qué ocurre dentro del test

Esta distinción es importante:

> **Avanzar el origen de predicción no significa actualizar los pesos.**

Con stride de seis horas:

```text
00:00 → predecir las próximas 12 horas
06:00 → actualizar el contexto con lo observado → predecir otras 12 horas
12:00 → volver a actualizar el contexto → predecir otras 12 horas
```

Durante ese test, los pesos permanecen fijos. Se permite incorporar al contexto lo que ya habría sido observado antes del nuevo origen.

**No he implementado entrenamiento cada seis horas.** Las actualizaciones progresivas de pesos ocurren entre folds.

### 2.5. Normalización y selección

En continuidad, la media y la escala se calculan con el pasado anterior al primer corte y se mantienen por estación.

Si una concentración de 20 se normalizaba inicialmente como 1, cambiar después el escalador podría convertirla en otro valor aunque conserváramos los mismos pesos. He evitado ese cambio de coordenadas.

Además:

- Se evalúa el checkpoint inicial del fold.
- Puede conservarse si las actualizaciones posteriores no mejoran la validación.
- El checkpoint heredado se selecciona por validación, nunca por test.
- No hay un reentrenamiento automático adicional con entrenamiento más validación antes de test.

La implementación está principalmente en `src/fd_airquality/backtesting.py`, `baselines.py`, `warmstart.py` y `strategy.py`.

---

## 3. Tratamiento de los NaN

### 3.1. Granite: sí, se utiliza su soporte nativo

Tu interpretación es correcta, con un adaptador para cada interfaz:

| Modelo | Tratamiento |
|---|---|
| TTM-R3 | Valores finitos de relleno junto con `past_observed_mask` |
| PatchTST-FM-R2 | Contexto incompleto y máscara mediante su interfaz |
| FlowState-R1 | Soporte de ausencias de la revisión con `with_missing=true` |

No he añadido un interpolador externo obligatorio para estos modelos. Tampoco he supuesto que sus interfaces sean idénticas.

**El soporte nativo del contexto no elimina nuestra responsabilidad sobre los targets:** la pérdida y las métricas siguen excluyendo las etiquetas ausentes.

### 3.2. Clásicos: imputación del contexto antes del forward

Para MLP, autoencoder y NLinear he mantenido el artefacto con sus ausencias originales. El relleno se realiza **sobre el contexto de cada ventana**, justo antes de pasarlo al modelo.

Las políticas predeterminadas son:

- MLP: `ffill`.
- Autoencoder: `ffill`.
- NLinear: `linear`.

Son decisiones iniciales configurables, **no políticas cuya superioridad haya demostrado experimentalmente**.

#### Ejemplo

Supongamos una media de entrenamiento igual a 8 y este contexto:

```text
Original: [NaN, 10, NaN, 14, NaN]
```

El resultado equivalente en unidades originales es:

| Política | Contexto recibido |
|---|---|
| `ffill` | `[8, 10, 10, 14, 14]` |
| `linear` | `[8, 10, 12, 14, 14]` |
| `mean` | `[12, 10, 12, 14, 12]` |
| `zero` | `[8, 10, 8, 14, 8]` |
| `strict` | Rechaza el contexto incompleto |

Los detalles son:

- **`ffill`:** utiliza el último valor observado dentro del contexto. Un prefijo sin observaciones se rellena con la media de entrenamiento.
- **`linear`:** interpola huecos interiores. Los huecos finales utilizan el último valor observado; el prefijo utiliza la media de entrenamiento.
- **`mean`:** utiliza la media de las observaciones del contexto de esa ventana.
- **`zero`:** utiliza cero **normalizado**, equivalente a la media de entrenamiento, no necesariamente a concentración cero.
- **`strict`:** no imputa.

Interpolar entre 10 y 14 no utiliza información posterior al origen: ambos valores ya pertenecen al contexto disponible.

**No se rellena toda la serie antes de dividirla ni se utiliza el target para completar el contexto.**

### 3.3. Los targets no se rellenan

Por ejemplo:

```text
Target:     [20, NaN, 24]
Predicción: [23, 999, 21]
Máscara:    [ 1,   0,  1]
```

La posición central no contribuye a la pérdida ni al error:

\[
\mathrm{MAE}=\frac{|23-20|+|21-24|}{2}=3
\]

En entrenamiento se aplica esta misma idea a la MSE de los targets normalizados.

En el autoencoder, la reconstrucción auxiliar solo se puntúa sobre las posiciones del contexto que eran observadas. No se obliga a reconstruir un relleno como si fuera una medición real.

### 3.4. Límites de lo implementado

Hay tres límites relevantes:

1. Los clásicos **no reciben actualmente un canal adicional de máscara**. Reciben los valores completados y no pueden distinguir directamente un relleno de un valor real igual a ese relleno.
2. Los mínimos de observaciones por ventana son compartidos en la preparación, no específicos de cada modelo.
3. Por defecto basta con una observación de contexto y un target observado. Eso evita una restricción excesiva, pero puede admitir ventanas poco informativas.

También se rechazan predicciones no finitas, incluso cuando el target correspondiente está ausente. No se permite mejorar una puntuación omitiendo fallos del modelo.

El código está en `src/fd_airquality/missing.py` y `task.py`.

---

## 4. MASE: denominador, políticas y agregación

### 4.1. Fórmula

Para una estación:

\[
\mathrm{MASE}=
\frac{\mathrm{MAE\ sobre\ targets\ observados}}
{s}
\]

El denominador es la media de diferencias estacionales absolutas válidas:

\[
s=\frac{1}{|P|}
\sum_{t\in P}|y_t-y_{t-m}|
\]

donde \(P\) contiene únicamente los pares cuyos dos extremos están observados dentro del periodo de referencia.

Con la configuración principal, \(m=24\) horas.

**Se comparan timestamps separados por 24 horas reales**, no posiciones de una serie a la que se hayan quitado los NaN. Un hueco entre ambos extremos no invalida el par; un extremo ausente sí.

### 4.2. Periodo utilizado para el denominador

Actualmente utilizo las **672 horas anteriores al comienzo del periodo evaluado**.

En el último fold:

| Evaluación | Referencia del denominador |
|---|---|
| Validación: 1–29 de marzo | 1 de febrero–1 de marzo |
| Test: 29 de marzo–26 de abril | 1–29 de marzo |

Esto tiene un matiz importante:

> **El denominador de test puede utilizar validación. No se calcula siempre exclusivamente con el conjunto utilizado para entrenar los pesos.**

No usa targets del propio test y toda la referencia está disponible antes de su primer origen. Sin embargo, es una definición distinta de fijar ambos denominadores únicamente con entrenamiento.

La referencia permanece fija durante cada evaluación y es común a todos los candidatos de esa estación. No cambia con su imputador.

### 4.3. Dos políticas

#### Política sin suelo

- Utiliza el denominador calculado.
- Si es cero, MASE queda indefinido.
- Si faltan pares suficientes, también queda indefinido.
- Un denominador positivo pero diminuto puede producir un MASE muy grande.

#### Política con suelo ε

Cuando existen suficientes pares:

\[
s_{\mathrm{efectivo}}=\max(s,\varepsilon)
\]

El valor configurado es \(\varepsilon=0{,}01\).

Por ejemplo:

```text
Escala histórica: 0
MAE:              0,02
Suelo:            0,01

MASE = 0,02 / 0,01 = 2
```

**El suelo no inventa pares.** Si no hay cobertura suficiente, la métrica sigue indefinida.

El mínimo del TOML principal es **24 pares válidos**. No significa 24 días. En una referencia completa de 672 horas y lag de 24 hay, como máximo, 648 pares.

Para RMSSE se calcula la media de diferencias estacionales cuadráticas y se aplica:

\[
\mathrm{RMSSE}=\sqrt{\frac{\mathrm{MSE}}{q}}
\]

La variante regularizada utiliza un suelo \(\varepsilon^2\) para \(q\).

### 4.4. Ejemplo numérico con ausencias

Para hacerlo corto, utilizamos lag de dos horas y admitimos dos pares:

```text
Referencia: [10, NaN, 12, 14, 16, NaN]
```

Los únicos pares válidos son:

```text
12 − 10 = 2
16 − 12 = 4
```

Por tanto:

\[
s=(2+4)/2=3
\]

Con:

```text
Target:     [20, NaN, 24]
Predicción: [23, 999, 21]
```

el MAE observado es 3 y:

\[
\mathrm{MASE}=3/3=1
\]

No se interpola la referencia ni se puntúa el target ausente.

### 4.5. Agregación entre estaciones

La selección predeterminada utiliza **MASE macro**:

1. Se calcula MASE para cada estación.
2. Se promedian las estaciones con escala definida, dando el mismo peso a cada una.

También se conserva `mean-mase`, que promedia los MASE por origen. **Con targets incompletos no tiene por qué coincidir con el MASE macro**, porque cambian las ponderaciones.

Si una estación no tiene escala válida de validación y se selecciona por MASE, se excluye del panel común antes de entrenar. Alternativamente se puede elegir explícitamente MAE o MAE macro. No hay un cambio automático de métrica.

Dos advertencias de interpretación:

- MASE < 1 compara el error con la variabilidad estacional de la **referencia histórica**; no garantiza superar al Seasonal Naive medido en ese mismo test.
- Los denominadores cambian entre folds. Una mejora temporal de MASE puede deberse parcialmente a ese cambio, por lo que también se reportan MAE/RMSE en unidades originales.

La implementación está en `src/fd_airquality/metrics.py`, `data.py` y `task.py`.

---

## 5. Integridad y diagnóstico

### 5.1. Controles de integridad implementados

**Datos y calendario**

- Alineación a la rejilla de cinco minutos con tolerancia configurable.
- Recuento de timestamps duplicados y lecturas desplazadas.
- Conservación del calendario y de las ausencias.
- Filtrado de negativos, valores no finitos, congelación y cobertura horaria.
- Exigencia de nueve lecturas útiles por hora en la configuración principal.
- Flags del proveedor configurables, sin adivinar su significado.

**Artefactos y ventanas**

- Preparaciones nuevas, sin sobrescribir las anteriores.
- Hashes de los artefactos y del código de preparación.
- Validación de límites, orden y recuentos de ventanas.
- Comprobación de targets observados y mínimos de disponibilidad.
- Comprobación de fronteras temporales.

**Entrenamiento y federación**

- Rechazo de pérdidas, pesos y predicciones no finitos.
- Comprobación exacta de la forma de salida.
- Identidad de clientes y panel completo.
- Orden estable de agregación.
- Validación de compatibilidad y cronología de checkpoints.
- Continuidad de las matrices congeladas y del historial de adaptación.

### 5.2. Diagnósticos implementados

Se generan:

- Inventarios de calidad y exclusiones.
- Cobertura por estación y partición.
- Recuentos de observaciones, slots de targets y ausencias.
- MAE/RMSE por horizonte.
- Sesgo firmado.
- Número y proporción de predicciones negativas, sin recortarlas silenciosamente.
- Comparaciones con persistencia y Seasonal Naive.
- Tiempos, memoria y payload de comunicación.
- Versiones, configuración, hardware, semillas y hashes.

Por ejemplo, un MAE bajo sobre 72 targets observados de 96 slots debe interpretarse con su cobertura del 75 %, no como una evaluación de las 96 posiciones.

### 5.3. Dos pendientes encontrados ahora

**1. `blocks.csv` no representa todas las ventanas enmascaradas.**

Sus recuentos consideran ventanas completamente contenidas en tramos observados. Las ventanas aceptadas que atraviesan NaN no aparecen en esos recuentos.

Lo he comprobado en una preparación: **24 ventanas de test contenidas en bloques completos frente a 190 ventanas enmascaradas retenidas**.

Esto no demuestra pérdida de esas ventanas en entrenamiento/evaluación; demuestra que ese diagnóstico no puede interpretarse como el total.

**2. El indicador de inactividad anterior al corte necesita corregirse.**

Actualmente utiliza la última observación global. Si una estación llevaba semanas sin datos antes del corte, pero reaparece después, puede informar cero días de inactividad.

Debe utilizar la última observación **anterior al corte**.

**No he corregido ninguno de estos dos puntos en esta revisión.**

---

## 6. Qué he hecho en extensiones

### Evaluación probabilística

Cuando el modelo proporciona cuantiles reales se calculan:

- Pérdida de cuantiles —dos veces pinball—.
- WIS cuando existen mediana y pares simétricos.
- Cobertura y anchura de intervalos.
- Cruces de cuantiles.

Por ejemplo, para un intervalo del 80 % `[10, 20]`:

- Un valor real de 18 está cubierto.
- Uno de 25 no lo está.
- La anchura es 10.
- La puntuación de intervalo penaliza tanto amplitud como falta de cobertura.

**Esto es evaluación probabilística**, no un entrenamiento nuevo específico por cuantiles ni una recalibración automática. Los modelos que solo producen puntos no reciben intervalos inventados.

### Variabilidad e incertidumbre

He añadido:

- Media y desviación entre semillas de la misma configuración.
- Bootstrap pareado por bloques temporales de siete días.
- 1.000 réplicas e intervalos exploratorios del 95 % para diferencias de MAE.
- Remuestreo conjunto de todas las estaciones para conservar episodios compartidos.

Con menos de 14 días de origen observados no se publica ese intervalo.

No hay todavía una inferencia conjunta completa que combine variabilidad entre semillas, folds y estaciones, ni corrección confirmatoria por comparaciones múltiples.

---

## 7. Revisión de las cuatro fases: pendientes y plan

**No: no está implementado y validado absolutamente todo.** Mi respuesta anterior debía haber distinguido más claramente entre disponer de la infraestructura y cerrar el experimento completo.

El estado verificable, agrupado en los cuatro bloques de trabajo, es:

| Bloque | Implementado | Qué impide darlo por cerrado |
|---|---|---|
| Datos y ausencias | Preparación enmascarada, políticas por modelo, pérdidas enmascaradas y controles de calidad | Contratos de fuente y estudio de sensibilidad de las decisiones |
| Evaluación temporal y comparación | Folds independientes/continuos, tres históricos, selección por validación y panel común | Decisiones metodológicas finales y ejecución a escala completa |
| Integridad y diagnóstico | Controles, trazabilidad, pruebas y métricas de diagnóstico | Correcciones concretas de informes y endurecimiento adicional |
| Extensiones | Evaluación probabilística, semillas y bootstrap temporal | Varias ampliaciones todavía no implementadas |

### Plan pendiente —sin ejecutarlo ahora—

**Datos y ausencias**

1. Confirmar unidades, límites instrumentales y significado de flags por fuente.
2. Estudiar sensibilidad a cobertura horaria, reglas de congelación y mínimos observados.
3. Comparar las políticas de contexto por modelo utilizando validación.
4. Evaluar máscaras explícitas como entrada de los clásicos y mínimos de disponibilidad específicos por modelo.

**Evaluación temporal**

1. Decidir y documentar si el denominador de MASE debe usar entrenamiento puro o el periodo inmediatamente anterior a cada evaluación.
2. Fijar el procedimiento de selección de hiperparámetros sin utilizar el test final.
3. Ejecutar el experimento completo y sus controles independientes frente a continuidad.
4. Definir un holdout realmente protegido: la etiqueta `final_holdout` por sí sola no impide consultarlo, y ya se han puntuado esos periodos en pruebas de integración.

**Integridad y diagnóstico**

1. Corregir el indicador de inactividad previa al corte.
2. Adaptar los recuentos de `blocks.csv` al protocolo enmascarado.
3. Identificar claramente los recuentos anteriores y posteriores al límite de ventanas de prueba.
4. Añadir una comprobación explícita de identidad de la referencia de normalización entre folds, más allá de comprobar su fecha.
5. Completar diagnósticos de errores por cantidad de ausencias, longitud de huecos y episodios.
6. Validar en GPU y ejecutar CI remotamente.
7. Sustituir las denominaciones actuales de las políticas por nombres funcionales neutrales si la restricción debe aplicarse también al repositorio.

**Extensiones**

1. Búsqueda automática de hiperparámetros con presupuestos comparables.
2. Evaluación de estaciones no vistas durante entrenamiento.
3. Covariables meteorológicas con disponibilidad temporal controlada.
4. Sensibilidad del bootstrap a la longitud del bloque y tratamiento de multiplicidad si se quieren conclusiones confirmatorias.
5. Integración prospectiva real, si se desea pasar de simulación retrospectiva a operación.
6. Evaluación de NeuralForecast como alternativa de modelos.

La validación previa de **374 pruebas**, incluidas 15 con checkpoints reales, y las ejecuciones reales de Flower/Ray respaldan la integración comprobada. **No sustituyen estos pendientes ni una validación científica a escala completa.**

---

## 8. Sobre NeuralForecast

**No lo integré ni lo evalué como backend en esta implementación.** Mantuve MLP/autoencoder propios y NLinear de Darts. Ahora he revisado su documentación y su código actual: **sí merece considerarse**, pero con una precisión importante.

### Qué ofrece para las ausencias

NeuralForecast dispone de:

- `available_mask`.
- Máscaras de contexto y targets en su interfaz de ventanas.
- Pérdidas enmascaradas.
- Umbrales configurables de disponibilidad para seleccionar ventanas de entrenamiento.

La documentación recomienda un valor finito de relleno acompañado de máscara:

```text
Estación   Hora    y     available_mask
A          10:00   10    1
A          11:00    0    0
A          12:00   14    1
```

Ese cero con máscara cero es un **placeholder**, no una observación real utilizada como target.

Fuente: [requisitos de datos de NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/docs/getting-started/datarequirements.html).

### Qué no conviene asumir

Hay tres capacidades distintas:

1. No puntuar targets ausentes.
2. Normalizar y seleccionar ventanas respetando una máscara.
3. Que la arquitectura utilice explícitamente la máscara para interpretar el contexto.

**No todos los modelos hacen las tres cosas de la misma manera.**

En la revisión de código consultada:

- MLP utiliza los valores del contexto sin consumir directamente su máscara en el `forward`.
- NLinear toma el último valor del contexto como ancla.
- NHITS sí utiliza la máscara en su cálculo de residuos.

Por tanto, cambiar a NeuralForecast no elimina automáticamente la decisión de qué valor recibe una arquitectura en los huecos, especialmente si falta la última observación.

Esto puede comprobarse en sus implementaciones de [NLinear](https://github.com/Nixtla/neuralforecast/blob/9d76fb3cb49a9dd3d0d1a617a01f20c4b74c4f03/neuralforecast/models/nlinear.py#L142-L155) y [NHITS](https://github.com/Nixtla/neuralforecast/blob/9d76fb3cb49a9dd3d0d1a617a01f20c4b74c4f03/neuralforecast/models/nhits.py#L411-L434).

### Cómo propondría evaluarlo

1. Fijar una versión.
2. Auditar por arquitectura el tratamiento de huecos interiores, prefijos y última observación ausente.
3. Probar inicialmente MLP/NLinear y un modelo adicional como NHITS.
4. Mantener nuestros cortes, targets, métricas y soporte de evaluación común.
5. Verificar continuidad de pesos, normalización y control del optimizador en Flower.
6. Compararlo como candidato explícito, sin sustituir silenciosamente las implementaciones actuales.

**Mi recomendación es evaluarlo antes de ampliar más los clásicos propios, pero no migrar únicamente porque exista `available_mask`.** Lo importante es el contrato completo de cada modelo y que no cambie inadvertidamente nuestro protocolo de comparación.
