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
`DataLoader(shuffle=True)` hace una permutación **sin reemplazo por época** y
conserva el último minibatch incompleto. Varias épocas vuelven a visitar el mismo
conjunto; eso no es muestreo con reemplazo dentro de una época. Las ventanas pueden
solaparse y contener la misma hora: tampoco equivale a repetir una ventana.
`max-train-windows`, cuando se activa, limita determinísticamente el conjunto de
la ronda después de los cortes y filtros; no duplica ventanas para llenar lotes.

**Padding** añade posiciones artificiales para completar una ventana. En la ruta
de entrenamiento de la versión examinada, NeuralForecast añade `H` posiciones a
la derecha y, con `start_padding_enabled`, `L-1` a la izquierda. Su máscara vale
cero: no son etiquetas verdaderas ni contribuyen a las estadísticas enmascaradas.
Esto puede permitir ventanas tempranas sin suficiente historia real o targets
parcialmente disponibles. No garantiza que los placeholders sean invisibles para
la arquitectura ni que el resultado sea mejor.

Nuestro protocolo exige las `L+H` posiciones reales de calendario: **90 horas de
contexto y 12 de horizonte** en el experimento principal. No inventa horas antes
del inicio o después del final de la serie. Las ausencias *dentro* de esas
posiciones se conservan mediante máscaras. Rehacer la rejilla horaria, rellenar
un contexto con información autorizada y añadir padding fuera del historial son
operaciones diferentes.

**No se han añadido ni muestreo con reemplazo ni padding de ventanas.** Esta
decisión no sustituye las pérdidas, normalizadores ni modelos de NeuralForecast:
seguimos llamando al paquete instalado mediante el puente externo.

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
`num-targets` usado para agregar clientes cuenta el soporte de una pasada, no se
multiplica por el número variable de épocas que cada cliente haya realizado.

## Evidencia y reproducción

La suite de este worktree pasa **559 pruebas, cero omisiones**, con checkpoints
reales habilitados: `reports/training_controls_tests.xml`. Se prueban la media
simple frente a la agrupada, evaluación sin búsqueda, selección optativa, paciencia,
scheduler, restauración de pesos, máscaras, escalas por estación, reserva causal
y protección de la reconstrucción auxiliar.

Se completaron **24 casos reales acotados en CPU**: cuatro modelos clásicos
por tres modos por dos folds. Flower/Ray es real en las ocho federaciones.
Con un máximo de ocho épocas y una mejora mínima deliberadamente enorme para
forzar meseta, cada ajuste termina en dos épocas y reduce el LR dos veces.
Los clientes siguen recibiendo los tres bloques `[16, 2, 2]`: no se cancelan llegadas.
Los dos folds se evalúan y se publican doce medias por configuración/semilla,
sin `selection.json`.

```bash
FD_AIRQUALITY_REAL_MODELS=1 uv run --frozen pytest -q --tb=short
uv run --frozen python experiments/validate_training_controls.py \
  --raw-root <directorio_de_CSV_originales>
```

El verificador arranca y cierra **su propio SuperLink en un puerto libre**, con
instalación dinámica de dependencias desactivada. Verifica los hashes del código
y versiones de Torch, NeuralForecast, Flower y Ray contra el entorno congelado.
Esto evita reutilizar un servicio `:local:` arrancado desde otro worktree con
módulos antiguos; no detiene ese servicio ajeno.

Informe: `reports/training_controls_final_validation.json`.
Auditoría: `reports/training_controls_validation/20261117_224747/validation.json`.
La primera tentativa, conservada bajo `20261117_224306`, detectó precisamente
esa reutilización de un runtime antiguo y no se considera evidencia aprobada.
La meseta forzada prueba mecanismos, no precisión ni superioridad predictiva.
