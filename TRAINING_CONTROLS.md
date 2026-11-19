# Folds de evaluación y controles de entrenamiento

## Muestreo con reemplazo y padding: qué cambia y qué no

**Con reemplazo** significa que, después de extraer una ventana, esta vuelve al
conjunto del que se puede extraer. Si hay tres ventanas admisibles `A, B, C` y se
piden cinco, una muestra posible es `A, C, A, B, C`. No crea información nueva.

En la ruta de ventanas de NeuralForecast 3.1.4, si `windows_batch_size=W` y hay
menos de `W` ventanas admisibles, las extrae con reemplazo. Si hay al menos `W`,
muestrea sin reemplazo para ese paso; `None` conserva todas las admisibles. Una
misma ventana puede volver a aparecer en pasos posteriores.

Aquí no se utiliza esa ruta de muestreo: el protocolo externo decide las ventanas.
`max-train-windows` es un **presupuesto por estación y época**, no un subconjunto
fijo por ronda. La preparación conserva todas las ventanas. Si el presupuesto
es 1000 y hay 10 000 ventanas admisibles en una estación, cada época sortea
1000 de las 10 000, sin reemplazo. Dos épocas pueden coincidir en parte; no se
obliga a agotar el conjunto antes de volver a sortear. Si se omite el presupuesto,
cada época visita todas las ventanas admisibles. `min-train-windows` controla
la elegibilidad de una estación y puede ser mayor que la cuota de cada época.

En federado se sortea solo del bloque autorizado de la ronda; en local, del
entrenamiento autorizado del fold. En centralizado se aplica la cuota por estación
y luego se mezclan las selecciones en los lotes. El orden de operaciones es:

1. Corte de llegada y exclusión de la reserva de monitorización.
2. Mínimos de observaciones y umbrales de disponibilidad del modelo.
3. Sorteo sin reemplazo de hasta `max-train-windows` distintas por estación.
4. Formación de batches (`batch-size`), completando el último lote corto con
   reemplazo **solo** de lo sorteado en esa época.

Por ejemplo, con 11 ventanas sorteadas y `batch-size=8`, hay dos lotes de
ocho: 11 originales y 5 repeticiones. Los microbatches físicos pueden ser menores.
Validación y test no repiten ventanas ni cambian sus lotes. Un conjunto vacío no
fabrica muestras: una ronda federada queda inactiva; un ajuste completo sin
ventanas falla. El sorteo y el reemplazo son reproducibles con la semilla.

El parámetro sigue en `[prepare]` o `--max-train-windows` en `prepare` y
`prepare-backtest`. Queda registrado en el manifest y solo se aplica al entrenar.
Para usar esta revisión con datos ya preparados, genera un manifest nuevo.

```toml
[prepare]
max-train-windows = 1000

[run]
batch-size = 100
```

Con al menos 1000 ventanas admisibles, cada estación local o federada hace diez
actualizaciones por época; la selección se renueva sin aumentar ese presupuesto.

Las ventanas solapadas pueden supervisar la misma hora en distintos horizontes:
eso no es una repetición del sampler. Los CSV de entrenamiento distinguen:

- `admissible-windows`: conjunto completo tras cortes y filtros.
- `num-examples`: ventanas distintas sorteadas **por época**.
- `unique-training-windows`: distintas visitadas entre todas las épocas del ajuste.
- `selected-target-occurrences`: targets observados de las selecciones, sumados
  entre épocas y sin relleno. `num-targets` divide esa suma por las épocas
  realizadas: puede ser fraccionario y es el peso federado con ventanas `masked`.
  En formato estricto, ponderar por ventanas es equivalente porque los horizontes
  son completos y de la misma longitud.
- `sampled-window-occurrences`, `sampled-target-occurrences` y
  `replacement-window-occurrences`: trabajo realizado y repeticiones por lote
  corto, que **no** aumentan el peso de agregación.

**Padding** añade posiciones artificiales para completar una ventana. En la ruta
de entrenamiento de la versión examinada, NeuralForecast añade `H` posiciones a
la derecha y, con `start_padding_enabled`, `L-1` a la izquierda. Su máscara vale
cero: no son etiquetas verdaderas ni contribuyen a las estadísticas enmascaradas.
Esto puede permitir ventanas tempranas sin suficiente historia real o targets
parcialmente disponibles. No garantiza que los placeholders sean invisibles para
la arquitectura ni que el resultado sea mejor.

El padding externo del protocolo se configura al **preparar**:

```toml
[prepare]
window-policy = "masked"
window-padding = "both"  # none | right | both
```

- **`none`** lo desactiva: exige las `L+H` posiciones reales y un horizonte
  completo dentro del intervalo autorizado, como antes.
- **`right`** permite horizontes parcialmente disponibles al final del
  entrenamiento o del bloque de la ronda. Sus posiciones posteriores se
  representan como NaN, incluso si el NPZ original contiene observaciones futuras.
- **`both`** añade lo anterior y hasta `L-1` posiciones virtuales de contexto a
  la izquierda del inicio de la serie. Se conserva la rejilla de orígenes de
  `train-stride`; no se desplazan las ventanas completas existentes.

Siempre se requiere al menos una observación real en contexto y otra en targets;
los mínimos configurados y umbrales por modelo pueden exigir más. El padding
cuenta como **ausencia**, nunca como cero observado. Un umbral `[1.0, 1.0]`
descarta todas las ventanas con padding. `missing-policy="strict"` tampoco
acepta contextos incompletos: configura umbrales completos o desactiva padding.

El TOML principal activa `both`; sin esa opción, el valor interno es `none`.
También se puede desactivar desde CLI:

```bash
uv run --frozen fd-airquality prepare --window-padding none
uv run --frozen fd-airquality prepare-backtest --window-padding none
```

Cambiar la política requiere **preparar de nuevo** y usar el nuevo manifest.
Las preparaciones con padding usan esquema 6 y registran política y extremo de
targets autorizado; los arrays originales de valores, timestamps y calidad no
se alteran. Las preparaciones de esquemas anteriores no se reinterpretan con padding.

El padding es **solo de entrenamiento**. Validación, test, sus máscaras y sus firmas
de evaluación no cambian. En federado no se supervisan targets anteriores al
bloque, posteriores al corte ni de la reserva de monitorización. La primera ronda
puede completar un horizonte con posiciones enmascaradas antes de esa reserva;
las siguientes hacen lo mismo en el extremo derecho de su bloque.

`padded-context-slots` y `padded-target-slots` cuentan posiciones virtuales en
las ventanas distintas visitadas, una sola vez aunque se repitan en otras épocas
o al completar un lote. No se confunden con las
ausencias originales. Rehacer la rejilla horaria, imputar un contexto autorizado
y añadir padding fuera del historial siguen siendo operaciones diferentes.
Los modelos, pérdidas y normalizadores de NeuralForecast continúan llamándose
desde el paquete instalado; no se copian. Ninguna máscara permite ignorar una
predicción no finita: se sigue rechazando.

## Folds: evaluación normal, búsqueda solo si se solicita

```toml
[backtest]
stage = "evaluation"
```

Es el valor predeterminado también sin TOML. Evalúa **todos** los folds con
reinicio, no reserva automáticamente el último y no produce `selection.json`.
Comparar varios modelos no hace que el programa elija una configuración ganadora.
La preparación solo identifica un *posible* holdout para el flujo optativo.

Se publican por configuración y semilla:

- `temporal_comparison.csv`: resultados de cada fold.
- **`cv_fold_mean.csv`: media aritmética y desviación descriptiva entre folds**,
  con el mismo peso por fold. Es el resumen principal de `evaluation`.
- `cv_summary.csv`: agregación por sumas de errores y conteos observados.
- `cv_station_summary.csv`: reducción por estación.

No hay una única media universal: con dos folds de MAE `0` y `10`, uno con un
target observado y otro con nueve, la media simple es `5` y el MAE agrupado es
`9`. Para RMSE tampoco es equivalente promediar las raíces y extraer la raíz
después de agrupar cuadrados. Se conservan ambas lecturas con nombres explícitos.
Una métrica indefinida en un fold no se elimina silenciosamente de su media
simple: se registra el número de folds definidos y la media queda indefinida.
Las semillas siguen siendo réplicas, no candidatos que seleccionar.

La validación interna de **cada** fold sigue existiendo: monitoriza el ajuste
y/o selecciona su checkpoint, no es un «fold de desarrollo» externo.

Para buscar configuraciones hay que solicitarlo:

```toml
[backtest]
stage = "development"
```

Ese flujo usa los primeros `N-1` folds para selección y reserva el último. Después:

```bash
uv run --frozen fd-airquality backtest --stage final \
  --manifest <familia.json> --selection <desarrollo>/selection.json
```

`--stage all` sigue siendo una opción explícita de exploración del flujo de
selección, no el valor normal. Haber puntuado todos los folds mediante
`evaluation` deja registrado el acceso: el último no puede presentarse más tarde
como un test final intacto. Ningún registro prueba desconocimiento humano ni
ausencia de contaminación en preentrenamiento.

## Early stopping y reducción del LR

La configuración principal utiliza:

```toml
[run]
local-epochs = 50                 # máximo, no número garantizado de épocas
early-stopping = 1
early-stopping-patience = 5
early-stopping-min-delta = 0.0     # mejora absoluta mínima, en unidades de MAE
lr-scheduler = "plateau"           # ReduceLROnPlateau de PyTorch
lr-scheduler-patience = 2
lr-scheduler-factor = 0.5
min-learning-rate = 0.000001
round-validation-hours = 24
```

Tras cada época se monitoriza **MAE por targets originales observados en
unidades físicas**. Se suman errores y conteos; no se promedian medias de lotes.
En centralizado se invierte la escala correspondiente a cada estación.
No se utiliza el error externo de CV ni el test para parar o ajustar el LR.
La pérdida optimizada puede ser MSE o multicuantil normalizada y no es esa métrica.

- Early stopping para después de `patience` épocas consecutivas sin una mejora
  mayor que `min-delta`; restaura los mejores pesos, incluyendo el estado inicial
  si ningún ajuste lo mejora.
- ReduceLROnPlateau reduce el LR por `factor`, respetando el mínimo. Su paciencia
  es la del scheduler de PyTorch: reduce cuando sus épocas malas **superan**
  `lr-scheduler-patience` (con cero, en la primera).
- En los controles local y centralizado se usa la validación interna ya separada
  del entrenamiento. El límite de etapas sigue siendo `num-server-rounds`; los
  controles de épocas se reinician al comenzar cada etapa.
- En federado se detienen **las épocas locales del bloque actual**, no las rondas.
  El siguiente bloque sigue llegando y parte de los pesos globales agregados.
  AdamW, scheduler, paciencia y LR inicial se reinician por ronda; no se promete
  una continuación del estado del optimizador.

Con un presupuesto de una sola época no hay épocas posteriores que detener ni
cuyo LR reducir; se ejecuta esa época sin crear una reserva adicional. El TOML
principal permite 50; el smoke de integración mantiene una sola. Para desactivar
explícitamente ambos controles: `early-stopping=0` y `lr-scheduler="none"`.

### Reserva causal para monitorizar rondas federadas

La validación completa del fold empieza **después** de su entrenamiento. No se
puede consultar desde rondas anteriores sin adelantar información.

Para el control local de épocas se reserva un intervalo fijo, anterior al corte
de la primera ronda: las últimas `max(round-validation-hours, H)` horas de ese
prefijo. Por ejemplo:

```text
Primera llegada: antes del 7 de enero
  ajuste inicial: targets anteriores al 6 de enero
  monitorización: targets del [6 de enero, 7 de enero)

Ronda 2: ajustar solo [7 de enero, 8 de enero), monitorizar la reserva ya conocida
Ronda 3: ajustar solo [8 de enero, 9 de enero), monitorizar la misma reserva
```

Los targets de la reserva nunca aportan gradientes de pronóstico, ni se ajusta
con ellos el scaler fijo. El autoencoder tampoco los utiliza como etiquetas de
su reconstrucción auxiliar cuando aparecen en contextos posteriores. Pueden
aparecer como entradas conocidas, como permite el protocolo.

Si no existe una reserva con verdad observada, se informa el error: no se
sustituye por validación futura ni por pérdida de entrenamiento. La validación
completa del fold se mantiene al final del flujo federado; la evaluación externa
mantiene los pesos fijos. La reserva, los mejores pesos y todos los controles
reinician entre folds.

`station_metrics.csv`, `round_metrics.csv` y los logs registran épocas realizadas,
paradas, mejor época, LR inicial/final, reducciones y MAE de monitorización.
`num-targets` usado para agregar clientes mide el soporte medio por época; un
cliente no gana peso por ejecutar más épocas o completar un lote corto.

## Verificación del presupuesto por época

En la revisión que introdujo esta cuota, **689 pruebas aprobadas, sin
omisiones**, incluidos los checkpoints reales.
La integración completó **48 ejecuciones acotadas en CPU** en los modos local,
centralizado y federado, con Flower/Ray en las federaciones. Se reprodujeron los
sorteos con sus semillas y se verificaron el soporte observado, los límites de
cada ronda y la exclusión de la reserva de monitorización.

Con una cuota de 13 y lotes de 8, la primera ronda seleccionó 13 ventanas por
estación en cada época y visitó **25–26 distintas en dos épocas**. Los dos lotes
por época se completaron con tres repeticiones, sin contarlas para agregar. Las
firmas y las máscaras de evaluación siguieron siendo las mismas.
Informe: `reports/epoch_window_sampling_validation.json`.

## Evidencia anterior: primer muestreo con reemplazo y padding

Antes del presupuesto por época, la revisión pasó **674 pruebas, sin omisiones**,
con los checkpoints reales habilitados: `reports/window_sampling_padding_tests.xml`.
Esa suite incluía:

- Batches exactos y cortos, conjunto vacío, semillas y varias épocas.
- Reemplazo sin inflar el soporte federado ni el límite de ventanas distintas.
- Los tres modos de padding, ceros observados, ausencias originales, horizontes
  parciales y validación/test sin padding ni reemplazo.
- Exclusión de targets futuros y de monitorización, bloques disjuntos,
  umbrales completos, protección de la reconstrucción e integridad del esquema 6.
- Gradientes equivalentes con microbatches y checkpoints reales Granite con
  contextos casi vacíos, targets parciales y batches completados.

Ruff y `git diff --check` también pasan. Estas comprobaciones de mecanismos no
demuestran superioridad predictiva ni certifican CUDA.

En esa revisión se completaron además **48 casos reales acotados en CPU** con preparaciones
nuevas: cuatro modelos clásicos × tres modos × dos folds × dos políticas
(`both` y `none`). Son dieciséis ejecuciones federadas con Flower/Ray real,
auditadas con SuperLink aislado y hashes de código/versiones comprobados.

Con padding, los tres bloques por estación usaron `[16, 4, 4]` ventanas; sin padding,
`[16, 2, 2]`. Tras dos épocas, las repeticiones son respectivamente `[0, 8, 8]`
y `[0, 12, 12]`, sin inflar el soporte observado de agregación. El límite de
dieciséis ventanas era aún fijo por ronda. El ajuste conservó las tres llegadas
y el control de épocas termina en dos con dos reducciones de LR.

Las firmas de evaluación y máscaras coinciden entre preparaciones: los folds
tienen respectivamente **2251 y 2648 targets de test observados**, sin padding de
evaluación. La auditoría reconstruye los targets realmente autorizados, comprueba
que no invaden el futuro ni la reserva y que los bloques no comparten etiquetas.
`right` está probado en la suite, no en una tercera matriz CLI independiente.

Informe de esa revisión: `reports/window_sampling_padding_final_validation.json`.
Auditorías: `reports/training_controls_validation/20261118_223641/validation.json`
(`both`) y `reports/training_controls_validation/20261118_223849/validation.json`
(`none`). El patch `reports/training_controls_fork.patch` es histórico.

## Evidencia anterior y reproducción

Antes de añadir padding y completado de batches, la revisión pasó **559 pruebas,
cero omisiones**, con checkpoints reales habilitados:
`reports/training_controls_tests.xml`. Se probaron la media
simple frente a la agrupada, evaluación sin búsqueda, selección optativa, paciencia,
scheduler, restauración de pesos, máscaras, escalas por estación, reserva causal
y protección de la reconstrucción auxiliar.

En esa revisión se completaron **24 casos reales acotados en CPU**: cuatro modelos clásicos
por tres modos por dos folds. Flower/Ray es real en las ocho federaciones.
Con un máximo de ocho épocas y una mejora mínima deliberadamente enorme para
forzar meseta, cada ajuste termina en dos épocas y reduce el LR dos veces.
Los clientes siguen recibiendo los tres bloques `[16, 2, 2]`: no se cancelan llegadas.
Los dos folds se evalúan y se publican doce medias por configuración/semilla,
sin `selection.json`.

```bash
FD_AIRQUALITY_REAL_MODELS=1 uv run --frozen pytest -q --tb=short
uv run --frozen python experiments/validate_training_controls.py \
  --raw-root <directorio_de_CSV_originales> --window-padding both
uv run --frozen python experiments/validate_training_controls.py \
  --raw-root <directorio_de_CSV_originales> --window-padding none
```

El verificador arranca y cierra **su propio SuperLink en un puerto libre**, con
instalación dinámica de dependencias desactivada. Verifica los hashes del código
y versiones de Torch, NeuralForecast, Flower y Ray contra el entorno congelado.
Esto evita reutilizar un servicio `:local:` arrancado desde otro worktree con
módulos antiguos; no detiene ese servicio ajeno.

Informe de la revisión anterior: `reports/training_controls_final_validation.json`.
Auditoría anterior: `reports/training_controls_validation/20261117_224747/validation.json`.
La primera tentativa, conservada bajo `20261117_224306`, detectó precisamente
esa reutilización de un runtime antiguo y no se considera evidencia aprobada.
La meseta forzada prueba mecanismos, no precisión ni superioridad predictiva.
