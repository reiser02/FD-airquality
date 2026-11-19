# NeuralForecast, validación temporal y llegada federada

## Alcance actual

El proyecto instala **NeuralForecast 3.1.4**, commit
`69e431413b22cbb1a74d35db2fa7994092ce0ea4`. No contiene copias de sus arquitecturas,
normalizadores ni pérdidas. MLP, NLinear y NHITS son instancias del paquete;
el autoencoder sigue siendo un control propio, porque no tiene un equivalente
directo en esa colección. No existe dependencia ni import de Darts.

La investigación de partida se conserva, sin modificar, en
[neuralforecast-validacion-temporal-datos-faltantes.md](neuralforecast-validacion-temporal-datos-faltantes.md).
[REVISION_EVALUACION_TEMPORAL.md](REVISION_EVALUACION_TEMPORAL.md) es una auditoría
**histórica anterior a esta migración**; no describe las garantías nuevas.

## Datos, máscaras y umbrales

Se conservan los timestamps horarios, los valores originales con NaN y la máscara
de observación. Un cero observado sigue siendo distinto de un cero de relleno.
Los NPZ son inmutables. El relleno ocurre dentro de cada contexto, nunca sobre toda
la serie ni usando sus targets posteriores.

El puente `src/fd_airquality/neural.py` proporciona ventanas con `y` finito y
`available_mask` original a la normalización y al parser del paquete. El forward
recibe `insample_mask`; la pérdida recibe `outsample_mask`. Los objetivos ausentes
solo tienen un placeholder numérico para la operación enmascarada: no se convierten
en etiquetas. El scoring recupera siempre los valores canónicos del artefacto.

`native` usa el placeholder finito y la máscara del modelo. **No significa que
el relleno sea invisible**: MLP/NLinear no consumen explícitamente esa máscara
en el forward; NHITS la usa en sus residuos, pero no en todas sus operaciones.
No se modifica ese comportamiento de las arquitecturas.

También se pueden comparar `ffill`, `linear`, `mean`, `zero` y `strict`. Cuando hay
normalización por ventana, la imputación explícita se aplica después de calcular
estadísticas con la máscara original: un prefijo `ffill` vacío utiliza cero
normalizado, equivalente al centro del contexto. Con normalización fija equivale
al centro fijo de entrenamiento. La interpolación solo utiliza extremos conocidos
dentro del contexto, nunca el horizonte de la muestra.

```toml
[run]
training-data-availability-threshold = [0.5, 0.5]

[availability]
mlp = [0.75, 1.0]
nlinear = 0.5
nhits = [0.5, 0.75]

[missing]
mlp = "native"
nlinear = "ffill"
nhits = "native"
```

Los umbrales usan floats, no enteros. Los mínimos univariados son
`max(1, int(L × α))` y `max(1, int(H × β))`, con truncamiento, como en la versión
fijada del paquete. `0.0` permite una observación de contexto y una de target:
es permisivo, no una recomendación universal. La preparación puede imponer mínimos
comunes adicionales. La configuración efectiva se valida también al construir
el modelo de NeuralForecast.

El filtro por modelo se aplica **solo al entrenamiento**, después del corte de
llegada y antes de muestrear las ventanas de cada época. No elimina fechas del calendario,
no modifica ventanas de validación/test y no permite ocultar predicciones inválidas.
Si no queda entrenamiento en todo el fold se aborta; una ronda temprana sin datos
se registra explícitamente como inactiva, con peso de agregación cero.

### Padding y completado de batches

`[prepare].window-padding = "none" | "right" | "both"` controla exclusivamente el
entrenamiento. `none` conserva el requisito de posiciones reales; `right` permite
un horizonte parcial al final del intervalo autorizado; `both` permite también
contexto virtual a la izquierda. El TOML principal usa `both`; el valor interno
sin configurar es `none`. Requiere ventanas globales `masked` y una nueva preparación.

Los arrays originales no cambian. Cada posición virtual se presenta con máscara
cero; no cuenta como observación, no ajusta el scaler y no se convierte en etiqueta.
Los mínimos observados y umbrales siguen aplicándose. Las ventanas de evaluación
no usan padding y sus firmas permanecen iguales si el panel no cambia.
Esquema 6 registra política y límite autorizado de targets.

Después de los filtros, cada época vuelve a sortear hasta `max-train-windows`
ventanas distintas por estación desde el conjunto admisible completo. Sin cuota
se recorren todas una vez. Solo se completa el último batch corto con reemplazo
de la selección de esa época, sin introducir ventanas distintas adicionales.
No se repiten ventanas al evaluar. Con `masked`, la agregación federada usa el
promedio del soporte observado por época **antes del reemplazo**; las ocurrencias
muestreadas y repetidas se informan aparte. Consulta
[las reglas completas](TRAINING_CONTROLS.md).

## Normalización sin fuga

```toml
[run]
normalization = "window"  # window | fixed | none
scaler-type = "standard"  # identity | standard | robust | minmax
```

- **`window`:** `TemporalNorm` del paquete, por contexto y con máscara original.
  Se excluye todo el horizonte al calcular estadísticas, también en entrenamiento.
  Se recalculan al avanzar el origen; eso no utiliza targets futuros.
- **`fixed`:** media/desviación por estación, solo sobre el prefijo observado
  disponible en la primera ronda **de cada fold**, limitado a su histórico
  autorizado. Se ajusta de nuevo en cada fold, sin heredar estadísticas del
  anterior. Al activar el control de épocas se excluye también la reserva de
  monitorización. Se registra un hash local de su referencia y estadísticas.
  `normalization-reference-end` permite fijar un extremo explícito, nunca
  posterior a la primera llegada ni fuera del historial de ajuste.
- **`none`:** sin escalado externo ni temporal adicional.

No se utiliza `NeuralForecast.local_scaler_type`: evitamos tanto su ajuste previo
al corte de ventanas como la posible inclusión de placeholders finitos.
Granite conserva su soporte nativo de ausencias y su normalización interna; la
capa externa configurada usa también `TemporalNorm` y se invierte al pronosticar.
El autoencoder utiliza el mismo normalizador del paquete.

Pérdidas de entrenamiento en la escala normalizada; evaluación y baselines en
unidades originales. No se agregan ni transmiten estadísticas de escalado. La
normalización por ventana y las capas internas del backbone se registran como
parte del procedimiento, no se confunden con un scaler global aprendido.

## CV externa, validación interna y holdout

Los folds son walk-forward con cortes de calendario comunes entre estaciones:

```text
[ targets de entrenamiento ][ validación interna ] | [ evaluación externa ]
```

El contexto puede usar observaciones anteriores ya disponibles, pero los targets
supervisados de una ventana no cruzan fronteras. Un horizonte de entrenamiento
con padding puede prolongarse nominalmente, con sus posiciones no autorizadas
enmascaradas. El horizonte cuenta horas, no valores válidos.
Los contextos avanzan cada `stride`; los pesos permanecen fijos durante cada
evaluación externa. Los horizontes solapados son predicciones distintas, no
réplicas independientes.

**Todos los folds son ajustes nuevos.** No hay una opción de continuidad entre
folds ni un argumento `--protocol`. Cada fold reinicia pesos, adaptadores e
historia FLoRA; en Granite se parte del mismo backbone preentrenado fijado, con
adaptación nueva. Los optimizadores se reinician por etapa/ronda.

`train-history = expanding | sliding` controla **datos**, no pesos:

```text
Expansivo:   inicio → ajustar A         → evaluar B
             inicio → ajustar A+B       → evaluar C
             inicio → ajustar A+B+C     → evaluar D

Deslizante:  inicio → ajustar A+B       → evaluar C
             inicio → ajustar B+C       → evaluar D
             inicio → ajustar C+D       → evaluar E
```

Compartir histórico entre folds no significa compartir su modelo aprendido.
Local crea un modelo nuevo por estación; centralizado crea una red nueva con
ventanas de todas las estaciones; federado crea un estado global nuevo.
`best_model.pt` evalúa su propio fold y `last_model.pt` es un artefacto de
trazabilidad: **ninguno inicializa otro fold**. Se ha eliminado también la opción
de pasar un checkpoint entrenado como inicialización de una ejecución.

Solo dentro del fold federado se conservan los pesos entre rondas mientras llegan
datos. El único comportamiento **entre rondas**, no entre folds, es utilizar
solo el bloque nuevo de targets en cada ronda,
sin volver a entrenar con los bloques anteriores. La historia de contexto puede
preceder al primer target de ajuste. Esto no cambia `train-history = "expanding"`:
cada fold sigue teniendo un histórico autorizado mayor, pero un modelo nuevo.

```toml
[prepare-backtest]
folds = 3
step-hours = 672
train-history = "expanding"  # opción normal; solo se amplían los datos
# Alternativa: train-history = "sliding"
history-hours = 672         # longitud de targets de ajuste si se usa sliding
```

Dentro de cada fold los checkpoints siguen sirviendo para seleccionar y restaurar
su predictor. Lo eliminado es **cargarlos como punto de partida de otro fold**.
Las pruebas corrompen los checkpoints del fold anterior para comprobar que ni el
siguiente ajuste ni el ajuste final dependen de ellos.

### Desarrollo y evaluación final separados

**El modo normal es `--stage evaluation`:** puntúa todos los folds, publica la
media aritmética en `cv_fold_mean.csv` y no selecciona una configuración. No
reserva automáticamente el último. `cv_summary.csv` conserva además la
agregación por conteos observados, que no siempre coincide con esa media.
La preparación marca un posible holdout; solo el flujo optativo le asigna ese papel.

El flujo de selección siguiente debe solicitarse explícitamente:

```bash
uv run --frozen fd-airquality prepare-backtest --config experiments/smoke.toml
uv run --frozen fd-airquality backtest --config experiments/smoke.toml \
  --manifest <backtest_manifest.json> --stage development

uv run --frozen fd-airquality backtest --config experiments/smoke.toml \
  --manifest <backtest_manifest.json> --stage final \
  --selection <resultado_desarrollo>/selection.json
```

Con `--stage development` se puntúan todos los folds **menos el último**. Se agregan errores
observados y conteos dentro de cada estación, se calculan métricas macro y se
promedian las réplicas de semillas al elegir configuración. No se elige la mejor
semilla como hiperparámetro. `selection.json` congela candidato, presupuesto,
semillas, panel, código y resumen de CV, no pesos entrenados.

`--stage final` utiliza solamente esa configuración y sus semillas. No acepta
reconfigurar modelos/umbrales/presupuesto mediante la matriz del comando. Comprueba
los hashes y registra acceso al holdout antes de ejecutarlo. El modelo final
también se inicializa desde cero (o desde el backbone preentrenado), se ajusta con
su histórico autorizado y no carga checkpoints de desarrollo. Una segunda consulta
final se rechaza. El panel se fija con desarrollo, no intersectando resultados
futuros: si falta una estación en el holdout, se informa el problema en vez de
cambiar silenciosamente la población.

`--stage all` es una opción **exploratoria**, útil para integración. Puntúa también
el último fold para todos los candidatos, lo marca como consultado y no lo llama
holdout sellado. El registro es un control del software, no prueba de que nadie
haya visto antes esos datos, ni de ausencia de contaminación en preentrenamiento.
Los periodos utilizados en pruebas históricas no se convierten en inéditos.
Lo mismo ocurre si se puntuaron previamente con `evaluation`: se registra el
acceso y no se permite relabelar posteriormente el último como holdout intacto.

La CV externa es propia del protocolo común a Flower, Granite y NeuralForecast.
No se utiliza el ajuste de alto nivel de NeuralForecast dentro de Flower, porque
tomaría control de ventanas, sampling, optimizador y validación. Se llama al código
instalado de normalización/parser/modelo/pérdida con las ventanas externas. Las
pruebas comparan sus pronósticos con `NeuralForecast.cross_validation` real.

## Llegada periódica dentro del fold federado

```toml
[run]
num-server-rounds = 3
arrival-period-hours = 24
```

Si el entrenamiento del fold acaba el 9 de enero a las 00:00:

| Ronda | Información disponible, extremo exclusivo | Targets usados para ajustar |
|---|---|---|
| 1 | Antes del 7 de enero, 00:00 | Histórico inicial anterior al 7 de enero |
| 2 | Antes del 8 de enero, 00:00 | Solo `[7 de enero, 8 de enero)` |
| 3 | Antes del 9 de enero, 00:00 | Solo `[8 de enero, 9 de enero)` |

La primera ronda recibe un histórico inicial; las siguientes incorporan 24 horas
de calendario. No se divide todo el histórico en partes iguales: el reloj mantiene
un prefijo inicial y `num-server-rounds - 1` incrementos diarios. No se fijan
cantidades iguales de ventanas por cliente, porque la disponibilidad puede variar.

```text
Ronda 1: pesos iniciales  → ajustar bloque A       → agregar → pesos 1
Ronda 2: pesos 1          → ajustar solo bloque B  → agregar → pesos 2
Ronda 3: pesos 2          → ajustar solo bloque C  → agregar → pesos 3
```

**No se ajusta A+B en la ronda 2 ni A+B+C en la ronda 3.** El servidor envía los
pesos globales agregados de la ronda anterior, no una inicialización nueva. AdamW
sí se reinicia en cada ronda, como antes.

El contexto puede reutilizar observaciones antiguas: no son targets de ajuste
de esa ronda. Con `window-padding="none"`, el horizonte completo debe quedar
dentro de su bloque y haber llegado, incluso si su último target está ausente.
Con `right` o `both`, puede prolongarse a la derecha, pero solo se supervisan
sus posiciones dentro del bloque y anteriores al corte; las demás quedan NaN.
Nunca se retrocede a targets del bloque anterior. Un bloque más corto que el
horizonte puede aportar una ventana parcial si alcanza los mínimos observados;
si no los alcanza queda sin ventanas. No se borran timestamps ni se añaden
etiquetas artificiales. El dataset se trunca físicamente al prefijo llegado y
el extremo autorizado de targets puede ser aún anterior si hay monitorización.

No existe un selector de histórico dentro del fold. Se han eliminado
`arrival-history` y `arrival-history-hours` de Python, CLI, TOML y Flower.
Las configuraciones antiguas que los incluyan se rechazan: deben retirar esas
claves, no sustituirlas por otro valor. `new-only` solo se conserva como etiqueta
descriptiva del protocolo en los informes, no como una opción configurable.
Esto no elimina `train-history = expanding | sliding`, que controla los datos
autorizados **entre folds**.

Se conservan los pesos entre rondas. No se exige que el número de
ventanas crezca: normalmente baja tras el histórico inicial y luego depende de
cada bloque. La cuota de cada época se aplica después del corte y el filtro
de disponibilidad. Se registran ventanas admisibles, sorteadas por época y
distintas visitadas entre todas las épocas de la ronda.
Dentro de un bloque, las ventanas pueden solaparse, `local-epochs` puede hacer
varias pasadas y el sampler puede repetirlas para completar un lote; lo que no
se repite son los bloques de targets entre rondas.
Las ausencias pueden hacer que lleguen horas sin nuevas ventanas admisibles.

El servidor envía los extremos del bloque y comprueba el intervalo devuelto:
`available-from-ns`, cuando hay límite inferior, `available-through-ns`,
`train-first-target-ns` y `train-last-target-ns`. Rechaza targets anteriores al
bloque o todavía no llegados. El último extremo describe la última posición
real autorizada, no el final nominal del horizonte con padding.
Se registran ventanas, targets observados,
exclusiones de disponibilidad y clientes inactivos por ronda. Los clientes
inactivos no influyen en los pesos. La validación completa
del fold no se consulta hasta terminar su flujo de entrenamiento; en este modo
no se hace early stopping con validación futura durante las llegadas.
El control de épocas locales utiliza una reserva causal del prefijo inicial,
excluida de gradientes y del ajuste del scaler fijo. No cancela rondas posteriores.
Consulta [los controles de entrenamiento](TRAINING_CONTROLS.md).

`arrival-period-hours = 0` desactiva expresamente la simulación dentro del fold.
Una sola ronda no permite observar una progresión: usa al menos dos para ese
objetivo. El modo retrospectivo por estación no comparte un reloj global y no
activa esta simulación. Los retrasos reales de transmisión/revisión no se
inventan: el contrato actual supone disponibilidad de la hora completada.

Esta política de bloques se aplica al entrenamiento federado. Los controles
local y centralizado siguen usando todo su entrenamiento autorizado por etapa;
no se ha cambiado su operativa ni los cortes entre folds.

## MASE y RMSSE

```toml
[prepare]
seasonality = 24
error-scale-hours = 672
error-scale-reference = "training"  # o pre-evaluation
error-scale-policy = "unfloored"    # o epsilon
error-scale-epsilon = 0.01
min-scale-pairs = 24
```

Para cada estación y referencia se calculan, solo sobre pares de extremos
observados a distancia real de `m` horas:

```text
a = media |y(t) − y(t−m)|
q = media (y(t) − y(t−m))²
MASE  = media (|predicción − y| / a)
RMSSE = sqrt(media ((predicción − y)² / q))
```

Nunca se compacta el calendario ni se interpolan las escalas. Con `training`,
ambos splits usan las horas anteriores al corte de entrenamiento del fold,
independientemente de las ventanas finalmente seleccionadas para gradientes. Con
`pre-evaluation`, cada split usa las horas anteriores a su propio inicio; el de
test puede incluir validación. No se mezclan las dos definiciones.

La política sin suelo deja indefinida una escala nula; con suelo se usan
`max(a, ε)` y `max(q, ε²)`, **solo si hay suficientes pares**. Una referencia vacía
no se convierte en un denominador artificial. Se comunican los conteos definidos.
El panel de selección se fija usando la disponibilidad de la métrica interna,
no los errores externos.

En CV se suman errores escalados y conteos antes de dividir; para RMSSE se extrae
la raíz después de agrupar, no se promedian raíces de folds. `cv_summary.csv`
incluye resultados por semilla y promedios macro por estación. El cálculo por
estación es interno: no se publica un resumen individual ni MAE/RMSE físicos.

## Cuantiles y CRPS

Con `probabilistic = 1`, los tres modelos de NeuralForecast entrenan con
`MQLoss` del paquete sobre 19 niveles, 0.05 a 0.95. La versión fijada suma sus
pinball entre cuantiles: el puente divide explícitamente por 19 para optimizar
la media. No copia ni modifica esa pérdida. Se selecciona el checkpoint con
nuestras métricas externas, no con `ptl/val_loss`.

Granite utiliza los cuantiles genuinos disponibles. Las revisiones fijadas de
TTM-R3, PatchTST-FM-R2 y FlowState-R1 los han proporcionado en la validación real;
no se fabrican intervalos para modelos con salida exclusivamente puntual.
En unidades originales se calcula:

- Pinball, WIS y cobertura/anchura de intervalos compatibles.
- Cruces de cuantiles, sin ordenarlos silenciosamente.
- **CRPS aproximado** por `2 × integral de pinball en τ`: cuadratura trapezoidal
  en la rejilla proporcionada, con colas cuantiles constantes hasta 0 y 1.
  La rejilla y el método forman parte del registro. No es el CRPS exacto de una
  distribución continua desconocida; los cruces son una advertencia sobre su
  interpretación distribucional.
- `sample_crps` proporciona el CRPS exacto de una distribución empírica si se
  dispone de muestras genuinas; no transforma puntos en muestras ficticias.

El servidor agrega **sumas de CRPS y targets observados**, no medias arbitrarias
de clientes. No hay puntuación en objetivos sin verdad original y se rechaza una
predicción no finita incluso cuando su etiqueta está ausente.

## Integridad y límites

Se corrigen los dos diagnósticos detectados en la revisión histórica:
`blocks.csv` añade las ventanas que cruzan ausencias y sus sumas cuadran con el
manifest; la inactividad previa al corte usa la última observación anterior a
ese corte, no una reaparición futura. `coverage.csv` separa la cobertura de todo
el conjunto antes de muestrear de la cota de ventanas por época. Los umbrales
específicos y los recuentos realizados se registran en cada ejecución.

Las preparaciones anteriores se conservan y no se sobrescriben. Los cambios en
módulos fingerprinted requieren preparar de nuevo. Checkpoints anteriores de
MLP/NLinear no se reinterpretan como pesos de NeuralForecast.
Las familias temporales nuevas usan esquema 2, `protocol = temporal_cv` e
inicialización `fresh`. Las familias anteriores se rechazan con una indicación
de ejecutar `prepare-backtest`; no se reinterpretan ni se sobrescriben.

Pruebas:

```bash
uv run --frozen ruff check src tests
FD_AIRQUALITY_REAL_MODELS=1 uv run --frozen pytest -q --tb=short
```

Estas pruebas verifican integración, máscaras, mínimos, normalización, gradientes,
cronología, agregación y restauración. No prueban superioridad predictiva, no
convierten ventanas solapadas en réplicas independientes y no son una ejecución
del experimento científico completo ni una certificación de CUDA.

### Verificación del muestreo por época

Al introducir la cuota por época se pasaron **689 pruebas, sin omisiones** y **48
ejecuciones acotadas en CPU**. La [revisión de la suite](TEST_REVIEW.md)
posterior consolidó los casos redundantes. Se verificó que las selecciones se renuevan sin
exceder la cuota por época ni el bloque autorizado y que no cambia el soporte
externo. Informe: `reports/epoch_window_sampling_validation.json`.

### Validación anterior: primer muestreo con reemplazo y padding

**674 pruebas aprobadas, cero fallos y cero omisiones**, con checkpoints reales
habilitados. JUnit: `reports/window_sampling_padding_tests.xml`. Se comprueban
reemplazo de lotes cortos, preservación del soporte federado, los tres modos de
padding, límites de llegada/monitorización, esquema 6, evaluación sin padding y
los tres backbones Granite con targets parciales. Los detalles y la reproducción
están en [TRAINING_CONTROLS.md](TRAINING_CONTROLS.md).
La integración CLI completó **48 casos reales acotados** con padding activado y
desactivado, manteniendo idénticas firmas y máscaras externas. Informe de esa revisión:
`reports/window_sampling_padding_final_validation.json`.

### Validación anterior: evaluación normal y controles de épocas

Antes del padding y completado de batches: **559 pruebas aprobadas, sin omisiones**, con checkpoints
reales habilitados. JUnit: `reports/training_controls_tests.xml`.
Se completaron 24 casos reales acotados de MLP/NLinear/NHITS/autoencoder, tres
modos y dos folds, con Flower/Ray en las federaciones. El control de épocas
aplica paciencia, reduce LR y restaura pesos sin omitir nuevas rondas.
Los folds se evalúan sin búsqueda por defecto. Los detalles, límites y comandos
están en [TRAINING_CONTROLS.md](TRAINING_CONTROLS.md).

### Validación anterior: sin selector de histórico entre rondas

Suite completa con modelos reales habilitados: **525 pruebas aprobadas, cero
fallos y cero omisiones**. JUnit: `reports/round_blocks_only_tests.xml`.
La reducción respecto a 535 corresponde a los casos específicos de los modos
acumulativo y deslizante eliminados.

Las pruebas rechazan las opciones eliminadas en CLI y Flower, comprueban que
no quedan en los TOML y mantienen la separación respecto a `train-history`.
La simulación real renovada con Flower/Ray, dos estaciones PM25 y tres rondas
utiliza `[16, 2, 2]` ventanas por estación, sin targets de bloques anteriores.
Se verifican los extremos de cada bloque y la ausencia de las opciones retiradas
en los ajustes registrados. Es una prueba de integración acotada en CPU, no
evidencia de superioridad predictiva.

Ejecución: `reports/round_blocks_only_smoke/PM25_mlp_full_fedavg_20261117_114800_536881/`.

### Validación anterior a la eliminación del selector entre rondas (17-11-2026)

En esa revisión: **535 pruebas aprobadas, sin omisiones**, con los modelos reales
habilitados. Resultado: `reports/round_blocks_tests.xml`. Se comprueba:

- `new-only` predeterminado en Python, CLI, Flower y los TOML principal y smoke,
  sin modificar `train-history = "expanding"` ni el reinicio entre folds.
- Ventanas y targets de bloques anteriores no se repiten en rondas posteriores;
  el contexto anterior sí permanece disponible.
- Cada cliente comienza con los pesos globales agregados de la ronda anterior,
  tanto en modelos clásicos como en adaptadores.
- El servidor rechaza targets anteriores al bloque, todavía no llegados o sin
  un intervalo de entrenamiento acreditado.
- Un bloque nuevo vacío no vuelve a entrenar con el pasado ni modifica los pesos.
- Esa revisión aún incluía alternativas acumulativas explícitas; ahora se han
  eliminado, junto con sus opciones y pruebas específicas.

Validación real renovada en CPU: **29 ejecuciones de CLI** (24 casos clásicos,
dos folds TTM-R3/FLoRA y tres controles probabilísticos Granite), con Flower/Ray
real en los casos federados y **74 checkpoints restaurados**. Se reauditan 20
flujos estación-ejecución, verificando que no repiten ventanas ni targets entre
rondas. En esta prueba acotada, con cap de 16 ventanas, los clientes clásicos
usan `[16, 2, 2]` ventanas y TTM/FLoRA `[16, 2]`: los bloques posteriores no
son históricos acumulados. FLoRA reinicia su historia en `[1, 2]` en ambos folds.

Informe consolidado: `reports/round_blocks_final_validation.json`.
Auditoría y logs: `reports/temporal_cv_validation/20261117_005143/validation.json`.
Las preparaciones y resultados anteriores se conservan.

### Validación del reinicio obligatorio por fold del 16-11-2026

La suite de esa revisión pasó **518 pruebas, sin omisiones**, incluidas las 15
con checkpoints reales. Resultado JUnit: `reports/temporal_cv_tests.xml`.
Se comprueba expresamente:

- Pesos iniciales idénticos para la misma semilla en folds distintos, en local,
  centralizado y federado, incluso corrompiendo sus checkpoints anteriores.
- Evaluación final desde inicialización nueva, también con los checkpoints de
  desarrollo corruptos.
- Históricos expansivos y deslizantes y escaladores fijos ajustados solo con
  el intervalo autorizado de cada fold.
- Historia FLoRA nueva por fold y continuidad entre rondas dentro de él.
- Llegada progresiva, máscaras, métricas y rechazo de opciones eliminadas.

Validación real renovada en CPU: **29 ejecuciones de CLI** (24 casos clásicos,
dos folds TTM-R3/FLoRA y tres controles probabilísticos Granite), con Flower/Ray
real en los casos federados y **74 checkpoints restaurados**. FLoRA termina con
historia `[1, 2]` en el primer fold y vuelve a `[1, 2]` en el segundo: no acumula
rondas entre folds.

Se crearon preparaciones nuevas, sin modificar los resultados anteriores.
Informe consolidado: `reports/temporal_cv_final_validation.json`.
Logs y auditoría: `reports/temporal_cv_validation/20261116_202904/validation.json`.
Estas ejecuciones preceden al cambio de valor predeterminado a bloques
`new-only`: verificaron llegada acumulativa y no deben presentarse como evidencia
de entrenamiento exclusivo con el bloque nuevo.

### Evidencia histórica anterior al reinicio obligatorio por fold

- Suite completa: **506 pruebas aprobadas, cero omisiones**, incluidas las 15 con
  checkpoints reales. 91 avisos de dependencias externas. Resultado JUnit local:
  `reports/neuralforecast_tests.xml`.
- CLI real: **24 candidatos clásicos**, cuatro modelos × tres modos × dos folds,
  con tres rondas y 24 horas de llegada por ronda.
- CLI real: **dos folds TTM-R3/FLoRA**, dos rondas por fold.
- CLI real: **tres controles zero-shot Granite**, todos con cuantiles y CRPS
  disponible. Las rejillas observadas fueron 9, 99 y 9 niveles respectivamente.
- Reauditoría completada: **74 checkpoints restaurados**, con pronósticos finitos
  también desde contextos incompletos. FLoRA conserva las rondas `[1, 2]` y después
  `[1, 2, 3, 4]`; ese comportamiento entre folds se ha eliminado. En la CV actual
  la historia debe reiniciar en `[1, 2]` en cada fold de dos rondas.
- Todas esas ejecuciones usan dos estaciones PM25, CPU y un cap de 16 ventanas
  por estación para entrenamiento. Son comprobaciones de integración, no una
  comparación científica de precisión.

Estos resultados no validan por sí mismos el protocolo actual. Las ejecuciones
y sus logs se conservan bajo
`reports/neuralforecast_validation/20261116_195535/`. El primer runner contenía
una comprobación incorrecta que suponía que TTM-R3 no emitía cuantiles; los
entrenamientos y evaluaciones sí habían terminado. Se corrigió esa comprobación
y se añadió una reauditoría sin repetir ni sobrescribir las ejecuciones.
Esos informes se conservan como evidencia histórica; el verificador actual
rechaza su código y protocolo antiguos.

`verification.json` es el informe de esa reauditoría. `validation.json` conserva
el resultado del runner original, incluida esa aserción incorrecta. Para repetir
todo con preparaciones nuevas:

```bash
uv run --frozen python experiments/validate_neuralforecast.py --foundations
```

Resumen histórico consolidado: `reports/neuralforecast_final_validation.json`.
Los informes y datos de `reports/` no se versionan; la documentación y el
verificador reproducible sí forman parte del repositorio.
