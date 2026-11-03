# FD-airquality

Simulación de predicción de calidad del aire con Flower (`flwr`): una estación
por cliente y un servidor que agrega pesos, sin recibir las series de los clientes.
Es independiente: contiene su propio tratamiento de datos y lee una copia local
de los CSV, sin importar código ni configuración de otro proyecto.

## Ejecutar

Requiere Python 3.11, `uv` y los datos de cinco minutos bajo
`data/raw/datos_estaciones_5m/` (un subdirectorio por estación).

```bash
uv sync --locked
uv run fd-airquality prepare
uv run fd-airquality inspect
uv run fd-airquality simulate
uv run fd-airquality models
uv run pytest
uv run ruff check src tests
```

Ejecuta los comandos desde la raíz de este repositorio. La CLI carga automáticamente
[`fd-airquality.toml`](fd-airquality.toml): NO₂, contexto común de **90 horas**, horizonte
de 12 y stride de evaluación de 6 horas. Entrenamiento anterior a **2026-03-01**;
validación `[2026-03-01, 2026-03-29)` y test `[2026-03-29, 2026-04-26)`.
Los extremos derechos son exclusivos. Se conserva el calendario sin zona de los
CSV, sin presumir que sea UTC. No se lee configuración de otro proyecto.

`simulate` usa la preparación más reciente, calcula automáticamente cuántas
estaciones hay y lanza **`uv run --frozen flwr run`**, con el simulador oficial
Ray y la API de mensajes de Flower 1.40. Por defecto ejecuta cinco rondas,
una época local y un cliente simultáneo por GPU utilizable, hasta el número de
estaciones; en CPU se ejecuta un cliente cada vez. No emplea la API
obsoleta `start_simulation`. FedAvg/FedProx clásicos son oficiales; las variantes
LoRA tienen su propia agregación de factores auditada.

Con `--device cuda` se comprueba una operación real y su sincronización, no solo
`torch.cuda.is_available()`. Si CUDA no está disponible o no puede ejecutar los
kernels, se avisa y se continúa en CPU; el dispositivo efectivo queda en `run.json`.
El launcher configura automáticamente Ray: una GPU completa por cliente y tantos
clientes simultáneos como GPUs utilizables (sin superar las estaciones del panel).
En CPU/fallback no se reservan GPUs y el entrenamiento de clientes es secuencial.
No hay que configurar `workers` ni `client-gpus`; esas opciones ya no se admiten.

### Configuración por archivo

Edita `fd-airquality.toml` para no repetir argumentos. La prioridad es:

**argumento explícito en CLI > sección del comando > `[run]` > valor interno**.

- `[prepare]`: fuentes, filtrado y geometría temporal.
- `[run]`: opciones compartidas por `simulate`, `baseline` y `benchmark`.
- `[simulate]`, `[baseline]`, `[benchmark]`: opciones específicas y overrides de `[run]`.
- `[inspect]` y `[compare]`: también admiten sus opciones (`manifest`, `runs`, `output`).

Las claves coinciden con los argumentos, sin `--`; las listas son arrays TOML.
Por ejemplo:

```toml
[run]
selection-metric = "macro-mase"
num-server-rounds = 10

[benchmark]
models = ["mlp", "autoencoder", "darts-nlinear"]
modes = ["local", "centralized", "federated"]
```

```bash
# Usa el archivo predeterminado; este argumento prevalece sobre sus valores:
uv run fd-airquality benchmark --num-server-rounds 2
# Usa solo este otro archivo, sin heredar fd-airquality.toml:
uv run fd-airquality benchmark --config experimentos/no2.toml
```

`--config` funciona antes o después del subcomando. Las rutas relativas se resuelven
desde el directorio de ejecución, igual que las de CLI, no desde el archivo TOML.
Si no existe el archivo predeterminado, se usan los valores internos; un archivo
explícito inexistente, una clave desconocida o un tipo inválido producen un error.
Los argumentos de listas sustituyen la lista del archivo, también `--station`.
`--no-print-command` permite desactivar un `print-command = true` del archivo.

Para comparar estrategias, fija **el mismo manifest**:

```bash
uv run fd-airquality simulate --manifest data/federated/<preparacion>/manifest.json \
  --strategy fedavg --num-server-rounds 10
uv run fd-airquality simulate --manifest data/federated/<preparacion>/manifest.json \
  --strategy fedprox --proximal-mu 0.01 --num-server-rounds 10
```

### Prueba pequeña con datos reales

```bash
uv run fd-airquality prepare --pollutant NO2 --max-train-windows 128 \
  --station 'AQN1 - Puerto' --station 'AQN3 - Paseo Alfonso XIII'
uv run fd-airquality simulate --num-server-rounds 2
```

El límite de ventanas es **solo para pruebas rápidas**. Vuelve a ejecutar
`prepare` sin ese límite para preparar el experimento completo. También admite
`--pollutant O3` o `CO`, `--raw-root`, `--context`, `--horizon`,
`--validation-start`, `--test-start`, `--test-end`, `--train-stride` y mínimos
`--min-train-windows`, `--min-validation-windows`, `--min-test-windows` (1/7/15).
`--error-scale-hours 672` fija la referencia de MASE/RMSSE a cuatro semanas;
no recorta el entrenamiento ni cambia el contexto de entrada.
`--zero-run-hours 24` controla el congelado de ceros: rachas cortas se conservan;
al acumular 24 horas observadas a cero se marca el fallo. Los huecos pausan el
contador, no lo reinician ni cuentan como ceros. Solo una hora válida no nula lo reinicia.
No se exige una racha para todo el test: cada ventana necesita 90 + 12 horas.
Consulta `uv run fd-airquality prepare --help` y `uv run fd-airquality simulate --help`.

## Comparar modelos con y sin federación

Modelos disponibles con `--model`:

| Nombre | Implementación y entrenamiento |
| --- | --- |
| `mlp` | MLP residual PyTorch; referencia original |
| `autoencoder` | Encoder con cuello de botella, decoder de reconstrucción y cabeza de forecasting |
| `darts-nlinear` | Módulo **NLinear real de Darts**, entrenado con las ventanas auditadas |
| `ttm-r3` | Granite TTM-R3, checkpoint `90-30-dec-90-r3`, adaptación interna |
| `patchtst-fm-r2` | Granite PatchTST-FM-R2, adaptación interna |
| `flowstate-r1` | Granite FlowState-R1/1.0, adaptación interna experimental |

Los tres Granite cargan **pesos preentrenados reales**, con revisiones SHA fijadas;
no se reemplazan por arquitecturas aleatorias. Se sustituyó el MLP externo por
**LoRA dentro de las lineales**. Los pesos base no reciben gradientes, pero
autograd atraviesa las capas adaptadas. Zero-shot no instala adaptadores ni entrena.
FlowState no publica soporte oficial de fine-tuning: su adaptación es una
integración experimental del proyecto, no una función oficial IBM.

TTM fija la revisión `417bfc83545095031c2a002ddfe73ac6b700ea1a`: contexto y contexto
residual **90**, salida nativa 30. Se usan las primeras 12 salidas, también para la
pérdida. Todos los modelos comparten el mismo manifest; no se inventa historia.
No compares directamente con las preparaciones históricas de 72/512.

```bash
uv run fd-airquality prepare --context 90
# Ejecuta la matriz completa sobre la preparación más reciente:
uv run fd-airquality benchmark --num-server-rounds 5
```

`benchmark` ejecuta para **cada foundation**: zero-shot, cuatro algoritmos federados
(LoRA, FFA-LoRA, FLoRA y LoRA-FAIR), y controles local/centralizado con LoRA y A
congelada. Son **27 configuraciones de foundations**, más nueve controles clásicos:
**36 ejecuciones** con una semilla y rango. Zero-shot no se duplica por semilla/rango;
los clásicos no se duplican por rango. FLoRA y LoRA-FAIR no tienen modalidades
centralizadas nativas ni se cruzan automáticamente con FedProx.

### Modos secuenciales y candidatos paralelos por GPU

La matriz respeta el orden de `--modes` / `[benchmark].modes`; por defecto:
**local → centralizado → federado → zero-shot**. Todos los casos de un modo deben
terminar antes de comenzar el siguiente.

- **Local, centralizado y zero-shot:** con `--device cuda`, se comprueba cada GPU
  visible y se ejecuta como máximo un candidato completo por GPU utilizable, en
  procesos independientes. Un candidato incluye modelo, adaptación, semilla y rango;
  no se reparten las estaciones de un mismo candidato local entre GPUs.
- **Federado:** un candidato cada vez. Flower/Ray reparte sus clientes entre las
  GPUs utilizables, una GPU completa por cliente: dos GPUs permiten dos clientes
  simultáneos, cuatro GPUs permiten cuatro, hasta el número de estaciones.
  No se lanzan federaciones de modelos distintos simultáneamente.
- **CPU o CUDA no utilizable:** los candidatos se ejecutan secuencialmente; el
  fallback y el dispositivo efectivo quedan registrados.

`--max-parallel-models` limita los candidatos no federados simultáneos; si se omite,
el límite es el número de GPUs utilizables. Los clientes federados se dimensionan
automáticamente con una GPU por cliente, sin opciones de concurrencia manuales.
El límite de candidatos **no limita** los clientes de una federación.

```bash
# Dos candidatos como máximo en los modos no federados; en federado, Ray gestiona clientes:
uv run fd-airquality benchmark --device cuda --max-parallel-models 2
```

Para limitar las GPUs visibles se puede usar `CUDA_VISIBLE_DEVICES` antes de iniciar
el comando; cada candidato no federado recibe una única GPU de ese conjunto y Ray
reserva una GPU por cliente federado. Las GPUs se comprueban individualmente y se
excluyen las no utilizables. Esto no reconfigura un SuperLink que ya estuviese
arrancado: debe iniciarse con el mismo conjunto de GPUs visibles.
Cada candidato debe caber por sí mismo en memoria. No se reducen automáticamente
batch, contexto, estaciones ni precisión ante un OOM: el error detiene la matriz.
Más concurrencia puede aumentar el uso de RAM/VRAM y no garantiza menor duración.

La primera ejecución descarga los pesos. PatchTST tiene unos 385 millones de
parámetros (1,54 GB de pesos). LoRA necesita backpropagation dentro del backbone;
el benchmark completo en CPU puede ser costoso y no se lanza al instalar.

PatchTST usa autocast BF16 en CUDA solo si la GPU tiene soporte **nativo**; en GPUs
como la RTX 2080 Ti se ejecuta en FP32, también durante el entrenamiento de LoRA.
La emulación de tensores BF16 no basta para sus convoluciones cuDNN. No se desactiva
cuDNN ni se cambia la precisión de los otros modelos.

En CUDA, PatchTST divide entrenamiento y evaluación en microbatches de hasta 8
ventanas. El DataLoader de entrenamiento conserva los batches originales: se
acumulan gradientes ponderados por el número de targets, con un único clipping y
paso de AdamW por batch efectivo (incluido el último incompleto). FedProx se aplica
una vez por batch efectivo. No se modifica el batch del TOML, el checkpoint ni la
arquitectura; puede haber pequeñas diferencias por el orden de suma o por dropout.
Los logs y las métricas registran batch efectivo, microbatch y pasos del optimizador.
En modo local CUDA solo el modelo de la estación activa permanece en GPU; los
demás conservan sus estados independientes en CPU, también durante evaluación.

Puedes limitar la matriz, sin cambiar el soporte temporal:

```bash
uv run fd-airquality benchmark --manifest data/federated/<preparacion>/manifest.json \
  --models mlp autoencoder darts-nlinear --modes local centralized federated
```

O ejecutar un caso individual:

```bash
uv run fd-airquality simulate --manifest data/federated/<preparacion>/manifest.json --model autoencoder
uv run fd-airquality baseline --manifest data/federated/<preparacion>/manifest.json --model autoencoder --mode local
uv run fd-airquality baseline --manifest data/federated/<preparacion>/manifest.json --model autoencoder --mode centralized
uv run fd-airquality baseline --manifest data/federated/<preparacion>/manifest.json --model ttm-r3 --mode zero-shot
uv run fd-airquality simulate --manifest data/federated/<preparacion>/manifest.json --model ttm-r3 --adaptation flora
uv run fd-airquality simulate --manifest data/federated/<preparacion>/manifest.json --model patchtst-fm-r2 --adaptation lora-fair
uv run fd-airquality compare reports/<ejecucion1> reports/<ejecucion2>
```

- **Local:** un modelo independiente por estación, sin intercambio ni agregación;
  cada estación elige su checkpoint por su propia validación.
- **Centralizado:** un modelo entrenado con las ventanas de todas las estaciones;
  es un control **sin privacidad federada**. Conserva el escalado por estación y
  no concatena series raw a través de huecos o estaciones.
- **Federado:** modelo compartido; LoRA promedia factores, FFA agrega B, FLoRA
  apila actualizaciones y LoRA-FAIR refina la agregación.
- **Zero-shot:** inferencia del checkpoint Granite sin adaptación con estos datos.

Los tres métodos entrenados usan `num-server-rounds × local-epochs` pasadas por
sus ventanas; centralizado y local reinician AdamW en cada etapa, igual que los
clientes FL. Igualar pasadas **no iguala trayectorias de optimización ni tiempo**.
Todos evalúan el test una vez después de seleccionar exclusivamente con validación.

Se selecciona por **MASE macro de validación**. MASE/RMSSE usan diferencias a
24 horas sobre los **28 días naturales anteriores al inicio del periodo evaluado**:
febrero para validación y del 1 al 28 de marzo para test con las fechas por defecto.
El denominador es fijo por estación durante cada periodo y común a todos los modelos.
Solo se exigen los dos extremos observados; un hueco intermedio no invalida el par.
Se conservan las horas ausentes, sin compactar ni ampliar la referencia para compensarlas.
`inspect` muestra los pares disponibles; una escala nula o sin pares queda indefinida.
Antes de entrenar, `simulate`, `baseline` y `benchmark` descartan las estaciones
cuya **métrica de selección de validación** no esté definida. La regla es común a
local, centralizado, federado y zero-shot: no se deja una estación en unos candidatos
y fuera de otros. Cada descarte aparece en el log y en `excluded_stations.csv`, con
estación, motivo, métrica, intervalo de referencia y número de pares. Se distingue
entre ausencia de pares y una escala cero con pares disponibles.
Si todas quedan descartadas, se guarda el informe y se aborta antes de entrenar.
La preparación original y sus NPZ no se alteran; se genera un `runtime_manifest.json`
con el panel efectivo y las identidades originales. La matriz lo fija una sola vez.
Puedes elegir explícitamente `--selection-metric mae` para conservar las estaciones
sin referencia escalada: solo cambia la selección del checkpoint, no el MASE/RMSSE
reportado ni su denominador. No hay cambio automático de métrica ni ampliación de
referencia. Los fallos de integridad o de entrenamiento siguen abortando, sin cambiar
el panel a mitad del experimento.
La normalización de entrada sigue ajustándose únicamente con entrenamiento.
MAE/RMSE se desnormalizan por estación.
`comparison.csv` incluye mejoras frente a persistencia y al zero-shot de la misma
revisión y configuración de inferencia, cuando hay una única referencia compatible;
`station_comparison.csv` conserva el detalle. `pairwise_comparison.csv` y
`station_pairwise_comparison.csv` comparan **todos contra todos**, en ambas direcciones,
para MAE/RMSE y MASE/RMSSE (macro en la tabla global). Delta = A − B: negativo
favorece A; mejora (%) = `100 × (1 − A/B)`: positiva favorece A y queda vacía si B=0.
Cambiar la escala de FlowState no excluye ningún candidato de esas comparaciones;
solo impide atribuir la diferencia exclusivamente a la adaptación si no hay un
zero-shot con la misma escala. La escala y revisión se incluyen en `comparison.csv`.
Los campos `robust-*` resumen estaciones con tres días de validación y siete de
test; no cambian el entrenamiento ni seleccionan el checkpoint.
`compare` rechaza manifests distintos o paneles incompletos y avisa si cambian
entornos o presupuestos. No ejecuta tests estadísticos sobre targets solapados.

### Prueba de integración pequeña para los seis modelos

```bash
uv run fd-airquality prepare --context 90 --max-train-windows 8 \
  --station 'AQN1 - Puerto' --station 'AQN2 - San Francisco' --output data/benchmark_smoke
uv run fd-airquality benchmark --manifest data/benchmark_smoke/<preparacion>/manifest.json \
  --num-server-rounds 1 --batch-size 8
```

Esta prueba valida la integración, **no estima el rendimiento del experimento
completo**. No compares sus puntuaciones con las de un manifest sin límite.

Varias semillas y rangos sobre las mismas ventanas:

```bash
uv run fd-airquality benchmark --manifest data/federated/<preparacion>/manifest.json \
  --models ttm-r3 patchtst-fm-r2 flowstate-r1 \
  --adaptations lora ffa-lora flora lora-fair --seeds 42 43 44 --lora-ranks 4 8
```

Valores iniciales: `--lora-rank 4 --lora-alpha 8 --lora-dropout 0`.
Un rango excesivo falla, no se recorta. FAIR añade refinamiento de servidor:
`--fair-iterations 1000 --fair-learning-rate 0.01 --fair-lambda 1`.
Reducir iteraciones para un smoke cambia el presupuesto, no es un benchmark completo.

Pruebas con los checkpoints reales, incluidos gradientes y restauración:

```bash
FD_AIRQUALITY_REAL_MODELS=1 uv run pytest -q
```

Sin esa variable se omiten explícitamente las 12 pruebas reales; las unitarias e
integraciones de Flower con transporte local sí se ejecutan.
La suite usa CPU y oculta las GPU físicas; las comprobaciones de CUDA usan dobles
de prueba. MLflow solo se activa en sus pruebas específicas, con una base SQLite
aislada por prueba y el esquema migrado una vez por sesión. No se fuerza el modo
determinista de PyTorch; se conservan las semillas del experimento.

Para medir los casos más lentos: `uv run pytest -q --durations=20`.

Validación histórica del protocolo anterior: **164 pruebas**, matriz smoke de **36/36 configuraciones**
con Flower/Ray real, 36 runs MLflow finalizados y **60 checkpoints restaurados**.
El presupuesto smoke es deliberadamente mínimo: dos estaciones, dos ventanas de
entrenamiento por estación, un origen de validación y otro de test, dos rondas y
cinco iteraciones de refinamiento FAIR. No constituye una comparación de precisión.
El resultado y las rutas auditables están en `reports/final_adapter_validation.json`.
Estos resultados históricos no validan el nuevo periodo de referencia de MASE/RMSSE;
para él se deben regenerar las preparaciones y volver a evaluar.

## Logs y progreso en vivo

No hace falta usar `tee`: los comandos largos escriben el log mientras trabajan,
además de mostrarlo en el terminal. Los mensajes incluyen fecha, nivel, PID e
identidad del candidato; entrenamiento y evaluación añaden estación, ronda/época
y lotes cuando corresponde. El log informa también de carga de modelos, selección
de checkpoints, descartes, fallback a CPU, errores con traceback e interrupciones.

En una matriz bajo `reports/benchmarks/<benchmark>/`:

- `benchmark.log`: eventos globales y progreso de candidatos, con escritura coordinada
  mediante una cola; los procesos no escriben concurrentemente en ese mismo archivo.
- `logs/case_0001.log`, etc.: detalle de cada candidato; los casos en subprocesos
  incluyen su salida combinada de stdout/stderr. Sus TOML efectivos quedan al lado.
- `runs/case_0001/<ejecucion>/run.log`, etc.: log propio del baseline o servidor.
- `benchmark.json`: estado por caso (`pending`, `running`, `completed`, `failed`,
  `cancelled`), GPU asignada, ruta del log y resultado. Se reemplaza atómicamente
  para poder leerlo mientras se actualiza.

```bash
tail -F reports/benchmarks/<benchmark>/benchmark.log
tail -F reports/benchmarks/<benchmark>/logs/case_0001.log
```

`--heartbeat-seconds 60` emite un resumen periódico con etapa, contadores, tiempo
transcurrido y tiempo desde el último evento. **Es una señal de actividad del
monitor, no una prueba de avance del entrenamiento ni una estimación de finalización.**
El mismo intervalo limita los mensajes intermedios por lotes. Se puede reducir,
por ejemplo con `--heartbeat-seconds 15`, o ajustar el nivel mediante
`--log-level DEBUG|INFO|WARNING|ERROR` (predeterminado `INFO`). También son opciones
de `[run]` en el TOML; para `prepare` pertenecen a `[prepare]`.

Los baselines independientes guardan `run.log` en su directorio de resultados.
`simulate` conserva el flujo real de Flower en `flower.log`, además del `run.log`
del servidor y `clients/client_000.log`, etc., para el detalle de los clientes.
Su auditoría y logs previos al lanzamiento quedan en `.flwr/launches/<lanzamiento>/`;
el flujo se copia a los resultados al completar la simulación. La CLI de `prepare`
escribe en `<output>/logs/prepare_<sello>/run.log` y, tras finalizar correctamente,
lo copia a `prepare.log` junto al manifest generado.

Un fallo cancela los casos activos y pendientes antes de iniciar otro modo; no
genera una comparación parcial como si estuviese completa. Al interrumpir Flower,
el launcher solicita detener su `run_id` con la CLI nativa, sin apagar SuperLink
ni otras ejecuciones. Si esa solicitud falla, el error se conserva en el log.

## MLflow

MLflow está instalado y **activo por defecto** en simulaciones y baselines.
Con los directorios estándar usa SQLite en `reports/mlflow.db`, sin requerir un
servidor. La ruta efectiva y el ID de cada ejecución quedan en `mlflow.json`.

```bash
uv run mlflow ui --backend-store-uri sqlite:///reports/mlflow.db --host 127.0.0.1 --port 5000
```

Abre `http://127.0.0.1:5000` y el experimento `fd-airquality`; selecciona las
ejecuciones de interés para compararlas. Se registran configuración, geometría,
hash del manifest, revisión del modelo, métricas por etapa/split y estación,
selección del checkpoint, CSV/JSON y pesos. No hay autologging ni se suben NPZ,
series, predicciones individuales o estadísticas de normalización.
También registra capas, rango, parámetros, tiempos, RSS y payload comunicado.
RSS es un máximo de vida del proceso, no memoria exclusiva del modelo; los bytes
no incluyen cabeceras Flower ni tráfico de red. MLflow está fijado a **3.17.2**.
Los contadores de payload corresponden al entrenamiento, no a distribuciones de
evaluación ni descargas Hugging Face. El tiempo del coordinador cubre el bloque
de ejecución, no la preparación ni la carga inicial de metadatos previa.

Para elegir almacenamiento, experimento o servidor existente:

```bash
uv run fd-airquality benchmark --manifest data/federated/<preparacion>/manifest.json \
  --mlflow-tracking-uri sqlite:///reports/mlflow.db --mlflow-experiment comparacion-no2
uv run fd-airquality simulate --mlflow-tracking-uri http://127.0.0.1:5000
```

Con `--output-dir` personalizado, el SQLite predeterminado se coloca en el
directorio padre de esa ruta; fija `--mlflow-tracking-uri` para reunir resultados
en un mismo almacenamiento. Usa `--mlflow-log-models 0` para no subir checkpoints
o `--mlflow-tracking-uri disabled` para desactivar tracking explícitamente.
Los errores de entrenamiento marcan la ejecución como `FAILED`.
El coordinador inicializa el experimento antes de lanzar candidatos concurrentes;
cada candidato mantiene su propio run. `run.log` se adjunta como artefacto al cerrar
el bloque de tracking; no se sincroniza línea a línea. Para seguir el log en vivo,
usa los archivos locales. Los logs detallados de actores Ray se conservan localmente.

## Resultados y auditoría

Cada preparación crea un directorio nuevo bajo `data/federated/`:

- `manifest.json`: identidad, protocolo, hashes y metadatos de cada cliente; sin mediciones ni escaladores.
- `clients/000/series.npz`, etc.: serie e índices de ventanas **locales** a cada estación.
- `stations.csv`, `blocks.csv`, `excluded_stations.csv`: fragmentación, soporte y exclusiones.
- `prepare.log`: log de la preparación cuando se ejecuta mediante la CLI.
- `retention.json`: ventanas y horas únicas retenidas respecto al entrenamiento
  disponible antes del corte, antes de cualquier límite de prueba.

Los manifests nuevos usan `schema_version=4` y hashes del código local de
preparación. Guardan límites y recuentos de pares de referencia para validación y
test; los denominadores permanecen en los NPZ locales, separados por split.
Las preparaciones antiguas deben regenerarse con `prepare`; no se
modifican ni se sobrescriben. Los resultados históricos se conservan.

Preparación histórica verificada el 29-10-2026, contexto 90, calendario global y calidad
causal predeterminados, sin límite de entrenamiento:

| Contaminante | Estaciones elegibles | Ventanas de entrenamiento | Retención de ventanas anteriores al corte |
| --- | ---: | ---: | ---: |
| NO₂ | 24 | 296.211 | 93,20 % |
| O₃ | 23 | 293.528 | 93,72 % |
| CO | 24 | 361.426 | 95,41 % |

Estos son recuentos de soporte, **no resultados de precisión**. El resumen auditable
local queda en `reports/global_context90_preparation.json`; no se distribuyen los
datos mediante Git. El cambio a calidad causal explica las diferencias frente a
estimaciones hechas con el filtrado retrospectivo anterior.

Cada simulación escribe un directorio nuevo bajo `reports/federated/`:

- `run.json`: configuración, manifest y versiones efectivas del entorno de ejecución.
- `run.log`: progreso, avisos, errores e interrupciones del servidor o baseline.
- `flower.log`, `clients/client_000.log`, etc.: flujo de Flower y logs de clientes en simulaciones.
- `excluded_stations.csv`: descartes en ejecución y sus motivos; vacío con cabecera si no hay ninguno.
- `runtime_manifest.json`: panel efectivo cuando hay descartes, sin copiar ni modificar los NPZ.
- `round_metrics.csv`, `station_metrics.csv`: métricas globales y por estación.
- `best_model.pt`: pesos elegidos por la métrica de validación declarada; puede ser la ronda cero.
- `last_model.pt`: pesos de la última ronda de entrenamiento.
- `test_metrics.json`: evaluación final del checkpoint elegido y baseline de persistencia.
- `mlflow.json`: URI, experimento e ID de seguimiento, salvo que esté desactivado.
- `resource_metrics.json`: tiempo del coordinador y máximo RSS de su proceso.

Los baselines generan el mismo esquema CSV/JSON bajo `reports/non_federated/`;
el modo local guarda además un checkpoint por estación. Las matrices se escriben
bajo `reports/benchmarks/`, con los resultados individuales y una comparación final.
La raíz de la matriz conserva también `excluded_stations.csv` y, si hay descartes,
`runtime_manifest.json`; el launcher deja su auditoría previa en `.flwr/launches/`.
`benchmark.json` registra todos los casos y su estado. Un fallo detiene la matriz
y queda marcado; no se omiten combinaciones silenciosamente.
Cada caso usa `runs/case_<numero>/` para aislar sus resultados. El bloque `execution`
de `run.json` registra PID y `CUDA_VISIBLE_DEVICES`; el dispositivo efectivo sigue
en `settings.device`.

Los artefactos locales son necesarios para aplicar la normalización del cliente;
el checkpoint central no contiene sus medias ni desviaciones. Los pesos se pueden
restaurar con `fd_airquality.models.restore_checkpoint(path)`. FFA conserva A
congelada y FLoRA toda la historia acumulada; los factores de la última ronda no
bastan. No se implementa reanudación automática de rondas interrumpidas.

## Implementación independiente

Los cargadores CSV/UPCT, el preprocesado horario y de congelados, la segmentación,
el holdout y la validación, MASE/RMSSE y los helpers de artefactos están en
`src/fd_airquality/`. No hay imports, búsqueda de checkouts ni modificaciones de
`sys.path` para usar otro proyecto. Los hashes del código local utilizado se
guardan en el manifest; los clientes y baselines rechazan preparaciones tras un
cambio de ese código.

Los CSV locales son una copia física, no enlaces a otro directorio. `prepare`
los lee sin alterarlos. `data/` y `reports/` siguen excluidos de Git: para ejecutar
el proyecto en otra máquina hay que transferir los datos además del código y
volver a preparar allí los artefactos, que contienen rutas absolutas.

Se conserva el pin `torch==2.11.0+cu128` y su índice de uv.
Todos los clientes de una ejecución comparten arquitectura y agregan pesos PyTorch.
Para Darts se adapta su módulo NLinear, no el objeto forecaster ni un Trainer de
Lightning; su interfaz interna queda acotada a Darts 0.48.x y cubierta por tests.
No se promedian escaladores ni optimizadores. Para TTM-R3 se instala `granite-tsfm`
desde un commit fijado, porque la versión PyPI 0.3.10 aún no implementa R3.

Flower crea además un entorno aislado por ejecución e instala sus dependencias
con **uv**. Su bundle no transporta `uv.lock`; esa resolución puede diferir del
entorno de desarrollo. Consulta las versiones efectivas en `run.json` antes de
comparar ejecuciones; el lock raíz fija el entorno de desarrollo.
Pandas está fijado también en `pyproject.toml` para evitar que el runtime de Flower
salte a otra versión mayor al evaluar los mismos resultados.

## Restricciones importantes

- **Nunca se juntan bloques separados por `NaN` u horas ausentes.** No hay interpolación automática.
- Todas las estaciones mantienen su identidad y su distribución natural; no hay particionado IID.
- El escalador se ajusta exclusivamente con entrenamiento local.
- El servidor recibe pesos, errores agregados y recuentos, **no series ni predicciones individuales**.
- Es una simulación con aislamiento lógico en un mismo sistema de archivos: **no garantiza privacidad formal**.
- El protocolo predeterminado impide futuro cruzado entre estaciones mediante
  cortes globales. No garantiza ausencia de contaminación en el preentrenamiento externo.
- El filtrado causal marca rachas no nulas desde la sexta lectura repetida. Los
  ceros usan otro umbral: se conservan las primeras 23 horas observadas y se
  descarta desde la 24.ª; los huecos pausan el contador y no se rellenan. No se
  borran retrospectivamente las primeras horas usando lecturas futuras. Cambia la cobertura.
- El modo histórico exige `--split-mode station --quality-mode retrospective`;
  `--validation-hours` y `--test-hours` determinan el holdout solo en ese modo.

El detalle metodológico, incluidos solapamientos y limitaciones, está en
[PROTOCOL.md](PROTOCOL.md).

## Paquetes adicionales útiles

- **Ray** ya viene con `flwr[simulation]`: administra los clientes virtuales y sus recursos.
- **Flower Datasets no hace falta**: los clientes ya están definidos por estaciones reales;
  repartir aleatoriamente estos datos sería incorrecto.
- **Opacus** sería útil si se exige privacidad diferencial, pero requiere definir el
  presupuesto y la unidad protegida. No está instalado ni activado. Tampoco se
  activa agregación segura; no debe confundirse federación con anonimización.
- **MLflow** ya está integrado para registrar y comparar los experimentos, sin
  sustituir los CSV/JSON auditables.
