# Protocolo de forecasting federado

## Datos fragmentados e implementación local

Un experimento trabaja con un único contaminante. Cada CSV de estación genera
un cliente, sin mezclar estaciones ni sus calendarios. Los datos se leen de una
copia física local en `data/raw/datos_estaciones_5m/`; las preparaciones se escriben
sin alterar esos CSV. No se necesitan código, configuración ni datos de otro
repositorio. `--raw-root` permite elegir otra fuente explícitamente.

`fd_airquality.data` coordina las siguientes implementaciones locales:

| Responsabilidad | Implementación en `fd_airquality` |
| --- | --- |
| CSV ordinario y exportación especial UPCT | `loaders.load_pollutant_file` |
| Rejilla, fechas y duplicados | `preprocessing.ensure_datetime_series` |
| Media horaria y sensores congelados | `preprocessing.preprocess` |
| Bloques continuos observados | `preprocessing.observed_blocks` |
| Test retrospectivo y entrenamiento/validación | `splitting.select_holdout_window`, `split_train_val_subseries` |
| Escalas estacionales sobre la referencia previa | `metrics.seasonal_error_scales` |
| Directorios de preparación sin sobrescrituras | `artifacts.create_run_dir` |

El tratamiento conserva la geometría temporal: primero rejilla de
cinco minutos, después media horaria de lecturas útiles. Actualmente `min_run=6`
detecta rachas congeladas **no nulas** desde la sexta lectura de cinco minutos y
`min_useful=1` exige una lectura útil por hora. Se aplican los límites de detección
locales (`0.0` para CO, NO₂ y O₃, con comparación inclusiva `>=`) y se marcan
repeticiones horarias no nulas. Los ceros se tratan por separado: una racha corta
puede ser válida; `zero_run_hours=24` (`prepare --zero-run-hours`) marca como
congelado al acumular 24 horas observadas a cero desde la última hora válida no
nula. Un hueco horario pausa el contador, sin reiniciarlo ni sumar horas a cero;
solo una observación válida distinta de cero lo reinicia. Los huecos y congelados
siguen siendo `NaN`; además se excluyen valores no finitos. No se añaden detectores
de anomalías aprendidos ni imputadores en este experimento inicial.

Cada ventana debe tener contexto y targets completos y consecutivos. Una estación
con muchas observaciones puede seguir siendo inelegible si ninguna racha alcanza
la longitud necesaria. Las exclusiones se guardan, nunca se sustituyen por datos
imputados ni por un test más corto. `blocks.csv` permite comprobar cuántas ventanas
de cada split proceden de cada bloque observado.

La geometría predeterminada se define en `DataConfig` y en el archivo local
`fd-airquality.toml`; las opciones explícitas de `prepare` prevalecen sobre ese archivo.
`--config` permite seleccionar otro TOML, sin heredar el predeterminado ni configuración
de otro proyecto. La prioridad completa está en [README.md](README.md#configuración-por-archivo).
El manifest de
esquema 4 registra `preparation_sources`, con hashes de los módulos locales. Los
manifests anteriores se conservan sin modificar y se regeneran para experimentos nuevos.

## Separación temporal global, esquema 4

El protocolo predeterminado es `global_temporal_forecasting_v3`, contexto 90 y
horizonte 12. Los intervalos de **objetivos** son semiabiertos:

| Split | Intervalo predeterminado |
| --- | --- |
| Entrenamiento | Anterior a 2026-03-01 |
| Validación | [2026-03-01, 2026-03-29) |
| Test | [2026-03-29, 2026-04-26) |

No se permite que los objetivos de una ventana crucen una frontera. Se conserva
toda la historia anterior elegible, sin exigir un inicio común ni recortar las
estaciones largas al tamaño de las cortas. Cada bloque aporta ventanas completas
de 102 horas: stride 1 en entrenamiento y una rejilla global cada 6 horas en
evaluación. Puede haber varias rachas por periodo; no se exige una única racha
de 28 días ni se toma la intersección de horas observadas de todas las estaciones.

Los mínimos son 1/7/15 ventanas de entrenamiento/validación/test. Los recuentos
pueden diferir entre estaciones, nunca entre métodos sobre el mismo manifest.
`validation_days` y `test_days` indican dispersión temporal. `robust_evaluation`
requiere tres días de validación y siete de test: las métricas `robust-*` de ese
subconjunto son sensibilidad, no otro entrenamiento ni criterio de selección.

Los contextos pueden usar observaciones anteriores ya disponibles, incluso
targets de validación/test observados antes de un origen rolling posterior.
Los pesos no se actualizan durante test. No se impone un embargo artificial de
90 horas: la separación corresponde a objetivos, no a historia observable.

El escalador cuenta cada observación de los bloques de entrenamiento una vez.
`max_train_windows` limita ejemplos, no vuelve a ajustar el escalador ni las
escalas estacionales. `retention.json` distingue entrenamiento potencial antes
del corte, retenido tras elegir el panel y usado tras el límite. Informa horas
únicas y ventanas solapadas; estas no son tamaño muestral efectivo. Las exclusiones
conservan CSV y preparaciones históricas; se registra la última fecha raw y útil.

La calidad se filtra causalmente: en valores no nulos se marca una racha desde la
sexta lectura idéntica sin borrar las cinco primeras usando lecturas futuras.
Para ceros se conservan las primeras 23 horas observadas y se marca desde la
24.ª hora observada a cero, incluidas las siguientes hasta detectar un valor no
nulo. No se aplica a los ceros el filtro de media hora ni el de repetición de dos
horas. Los huecos de cinco minutos no reinician el contador si la hora conserva
suficientes lecturas útiles; una hora sin lecturas útiles queda `NaN` y lo pausa.
Las medias son de horas completadas; las repeticiones horarias no nulas solo
consultan la hora anterior. En modo retrospectivo se marca toda la racha observada
si alcanza el umbral, manteniendo sus huecos como `NaN`.
Un prefijo horario no cambia al añadir horas futuras. Se asume disponibilidad de
la hora completada en el siguiente origen; no se modelan retrasos ni revisiones.

El modo antiguo exige `split_mode=station`, `quality_mode=retrospective` y no
elimina futuro cruzado. La disponibilidad condiciona el panel y se examina antes
del modelado. Esto no garantiza ausencia de contaminación en el preentrenamiento
externo ni reproduce la disponibilidad histórica de un checkpoint publicado después.

## Aprendizaje federado

Flower 1.40 ejecuta `ServerApp` y `ClientApp` mediante su Simulation Runtime con
Ray. El lanzador utiliza la CLI actual `flwr run`, no las funciones de simulación
Python obsoletas. El número de SuperNodes coincide con las estaciones elegibles.
Cada `partition-id` identifica exactamente una estación.

El cliente abre únicamente su NPZ y transforma sus ventanas con su media y
desviación de entrenamiento. Todos los modelos producen directamente un horizonte
univariante sobre las mismas ventanas. No se comparten estadísticas de normalización.

### Modelos y parámetros agregados

- **MLP residual:** la última capa inicializada a cero hace que la ronda cero
  equivalga a persistencia.
- **Autoencoder de forecasting:** encoder `context → hidden → latent`, decoder
  `latent → hidden → context` y cabeza `latent → horizon`, residual respecto a la
  última observación. Se minimiza MSE de forecasting más
  `reconstruction-weight × MSE de reconstrucción del contexto centrado` (0,1 por
  defecto). La reconstrucción solo usa el contexto de ventanas de entrenamiento;
  no imputa huecos ni ve targets de validación/test. Flower agrega encoder,
  decoder y cabeza. `latent-size` es 16 por defecto; la selección usa únicamente
  error de forecasting, no reconstrucción. La ronda cero es persistencia.
- **Darts NLinear:** módulo original de Darts 0.48.x, con normalización por última
  observación y pesos compartidos. Se construye mediante su hook `_create_model`
  y se invoca con `PLModuleInput`, sin `fit` sobre series compactadas. El bucle
  común usa AdamW y las ventanas auditadas; no es el Trainer predeterminado de
  Darts. Su inicialización es la de Darts y no se presupone persistencia.
- **Granite TTM-R3, PatchTST-FM-R2 y FlowState-R1:** adaptación interna en lineales
  `W0 + (alpha/r) BA`, no MLP externo ni fine-tuning completo. Pesos preentrenados
  y dropout del backbone permanecen congelados/en evaluación, pero autograd
  atraviesa los adaptadores. El dropout LoRA es configurable aparte. A se inicializa
  con Gaussianas de desviación 0,02 y B a cero: la ronda cero conserva el predictor.
  La semilla por capa no depende del cargador ni de la identidad del cliente.
  No se agregan buffers, escaladores ni parámetros complejos del SSM.

El servidor carga pesos oficiales para inventariar capas y construir factores,
pero no abre mediciones. Las instancias tienen módulos y adaptadores independientes;
comparten solo almacenamiento inmutable de pesos CPU, no objetos Parameter.
Las actualizaciones FLoRA no mutan la caché ni contaminan otro cliente o zero-shot.

TTM adapta `mlp.fc1/fc2` de mezcladores en las dos ramas; PatchTST, `attn.qkv`,
`attn.proj` y las lineales de `mlp` o `ffn1/ffn2` en sus bloques Conformer; FlowState, lineales de salida/gating y decoder,
sin adaptar su núcleo complejo. Las rutas/dimensiones exactas están en `run.json`.
`lora_targets` permite selección explícita; una ruta inexistente o rango excesivo
falla, sin fallback. No se aceptan pesos de checkpoint ausentes o incompatibles.

PatchTST integra cuantiles bajo `no_grad` upstream: se reconstruye la misma
integración puntual de forma diferenciable. La adaptación de FlowState es
**experimental**: se verifican gradientes reales, no se atribuye soporte oficial
de fine-tuning al proveedor ni se presupone una mejora de rendimiento.

Las revisiones SHA oficiales están fijadas en `src/fd_airquality/models.py`.
`--hf-revision` solo acepta otro SHA inmutable, que pasa a formar parte del registro;
no se sustituyen familias/versiones silenciosamente. TTM-R3 usa la clase
`TinyTimeMixerForDecomposedPrediction`, con su descomposición trend/residual;
el código Granite se fija a un commit con soporte R3. La revisión FlowState elegida
es **R1/1.0** del repositorio solicitado, no R1.1.

Se usan predicciones puntuales upstream para TTM/PatchTST y mediana para FlowState,
con `flowstate-scale-factor=1.0` para datos horarios. No se optimiza esa escala en
test. No se evalúa calidad probabilística ni se calculan CRPS o coberturas.

### Contexto común y fragmentación

TTM-R3 `90-30-dec-90-r3`, SHA `417bfc83545095031c2a002ddfe73ac6b700ea1a`,
exige 90 pasos de contexto y predice 30: se conservan
los primeros `horizon <= 30`. El adaptador no inventa ni rellena contexto. La
comparación que incluya TTM necesita preparar **todos** los modelos con contexto
90, manteniendo el mismo manifest y las exclusiones resultantes. No se comparan
directamente puntuaciones entre preparaciones de 72, 90 y 512 horas: pueden cambiar
estaciones elegibles, bloques retenidos y fechas de evaluación.

PatchTST acepta aquí contextos de 16 a 8192 y FlowState hasta 2048 para el
checkpoint/escala horaria utilizados y horizonte <= 24. Con otras escalas o
horizontes mayores, se comprueba además el límite efectivo de FlowState y se
rechaza un recorte implícito del contexto. Cualquier padding interno de patches
pertenece a la implementación del modelo, no añade observaciones al dataset ni
permite cruzar un hueco. Los adaptadores reciben únicamente ventanas continuas
que ya pasaron `WindowDataset`.

Cada ronda distribuye los pesos globales, entrena localmente con AdamW y devuelve
pesos y métricas. El optimizador local se reinicia en cada ronda; no se agrega su
estado. Hay semillas por ronda y estación, clipping de gradiente y orden estable
de agregación. No se fuerzan algoritmos deterministas de PyTorch ni se garantiza
igualdad bit a bit entre ejecuciones, hardware o versiones diferentes.

PatchTST en CUDA usa microbatches de hasta 8 ventanas sin cambiar el batch efectivo
configurado ni el orden de las ventanas. Cada pérdida se pondera por sus targets;
se acumulan los gradientes antes de un único clipping y paso de AdamW por batch
efectivo. El último batch incompleto usa su tamaño real y FedProx se añade una sola
vez. No se recorta el contexto ni se cambia la arquitectura; el orden de suma y el
dropout pueden introducir diferencias numéricas. En el control local CUDA los
modelos siguen siendo independientes, pero solo la estación activa reside en GPU;
las restantes se conservan en CPU.

- **FedAvg:** implementación oficial de Flower, ponderada por el número de ventanas
  de entrenamiento realmente utilizadas, no por filas brutas, horas de calendario
  ni número de épocas.
- **FedProx:** estrategia oficial de Flower, con el término local
  `mu / 2 * sum((w - w_global)^2)`; la referencia es una copia de los pesos recibidos.

`AuditedStrategy` delega en estas estrategias para modelos clásicos y añade
auditoría, métricas y checkpoints. Los foundations usan:

| Algoritmo | Regla |
| --- | --- |
| LoRA / FedIT | Promedia A y B por separado: baseline que no equivale al promedio de productos. |
| FFA-LoRA | A común congelada; solo B se optimiza y transmite. A se guarda completa en checkpoint. |
| FLoRA | Apilado exacto, peso del cliente aplicado solo a A; historia acumulada y reinicialización local entre rondas. |
| LoRA-FAIR | Promedia A/B y productos; refina B con discrepancia coseno por columna y L2 cuadrática. |

Referencias de los algoritmos: arXiv:2403.12313, 2409.05976 y 2411.14961v3.
FLoRA es el algoritmo federado, no el optimizador homónimo. FAIR sigue el objetivo
del código oficial ICCV-2025: SGD, corrección Xavier y semilla fijada, salvo
actualización ideal nula, que permanece nula. No necesita datos de servidor.

Se usan rangos homogéneos para aislar el algoritmo. FLoRA aumenta el rango efectivo
global y la descarga; no se trunca ni aplica SVD. Su historia utiliza **alpha/r
local**, no alpha dividido por el ancho apilado. Los uplinks envían factores locales;
el servidor conserva y redistribuye historia para reconstrucción incluso en un
actor nuevo. Un checkpoint incluye toda esa historia, no solo la última ronda.
FLoRA/FAIR reemplazan la agregación: no se cruzan con FedProx ni tienen modalidades
sin federación. LoRA/FFA admiten la penalización proximal explícita como ablación.

Participan todas las
estaciones en cada ronda; una respuesta fallida, ausente, duplicada, con recuentos
incompatibles o valores no finitos aborta, en vez de cambiar silenciosamente el panel.

El escalado es local: el objetivo de entrenamiento es MSE normalizado por estación,
no MSE global en unidades físicas. FedAvg sigue ponderando por ventanas. Las
estaciones con más ventanas influyen más; las métricas macro permiten observar
también el rendimiento dando igual importancia a cada estación.

## Selección y evaluación

Cada ronda evalúa **validación local**, no un dataset central. La **MASE macro** de
validación selecciona el checkpoint, incluida la ronda cero. Se puede elegir
explícitamente `macro-rmsse` o `mae`, nunca una métrica de test. Al terminar
se envía ese checkpoint a todos los clientes para evaluar el test una sola vez.
El test no decide rondas ni hiperparámetros automáticamente.

Se calculan MAE/RMSE en unidades originales y un baseline de persistencia sobre
los mismos targets. El servidor suma errores absolutos, errores cuadrados y
recuentos; el RMSE global es la raíz del MSE agrupado, no la media de los RMSE
locales. También se reportan MAE/RMSE macro por estación.

MASE/RMSSE usan una **referencia previa de 28 días naturales** (`error_scale_hours=672`,
opción `prepare --error-scale-hours`), separada del contexto de entrada y del
historial usado para ajustar pesos. En modo global se fija por los cortes comunes:

| Evaluación | Referencia predeterminada, extremo derecho exclusivo |
| --- | --- |
| Validación | [2026-02-01, 2026-03-01) |
| Test | [2026-03-01, 2026-03-29) |

En modo por estación se toman las mismas 672 horas anteriores al primer target
del split. Se usan observaciones ya disponibles, incluida validación para la
referencia del test; nunca targets del periodo que se está evaluando. La escala
se calcula una vez por estación y split y se congela para todas sus ventanas,
rondas y modelos. Las escalas de validación y test pueden diferir: sus valores
MASE/RMSSE no deben interpretarse como si compartieran el mismo denominador.

Para `seasonality=24`, ambos extremos de cada diferencia deben estar observados,
pertenecer a la referencia y distar exactamente 24 horas. No se exige observar
las horas intermedias. Se conserva el calendario con sus `NaN`, sin concatenar
fragmentos ni extender la referencia hacia datos antiguos para reemplazar huecos.
MASE divide MAE por la media de diferencias absolutas; RMSSE divide MSE por la
media de diferencias cuadráticas antes de extraer raíz. Es la fórmula estacional
de MASE/RMSSE con una ventana de referencia reciente explícita, no una escala
calculada sobre todo el entrenamiento. La elección de 28 días es de protocolo,
no una duración impuesta por MASE ni un número de observaciones independientes.

Sin huecos hay 648 pares posibles en 672 horas. El manifest registra inicio,
fin, pares válidos y disponibilidad de escala por split; `inspect` muestra los
recuentos. Las métricas locales incluyen `error-scale-pairs`. Escalas nulas o sin
pares se excluyen con cobertura explícita, no se sustituyen por uno ni producen
errores cero ficticios. Los esquemas
históricos 2/3 no se reinterpretan como esta nueva referencia: los experimentos
nuevos requieren `prepare` y el mismo manifest para todos los candidatos.

Con selección por `macro-mase` o `macro-rmsse`, en tiempo de ejecución se descarta
cada estación sin escala válida de **validación**, antes de entrenar. Se aplica la
misma regla a modelos locales, compartidos y zero-shot, incluso si un compartido
podría obtener una métrica macro usando solo parte del panel. No se decide la
elegibilidad usando el error de test ni cambiando sus targets. La selección por
`mae` no exige denominador estacional; permite conservar ese panel sin modificar
el cálculo reportado de MASE/RMSSE.

Cada descarte se avisa en el log y se registra por separado en
`excluded_stations.csv`: estación, partición original, causa, métrica de selección,
intervalo de referencia y pares observados. Se distingue `validation_reference_no_pairs`
de `validation_mase_scale_zero` / `validation_rmsse_scale_zero`. Si no queda ninguna
estación, se escribe ese informe y se aborta antes del entrenamiento.
`runtime_manifest.json` conserva el origen, su hash y las exclusiones; renumera
consecutivamente las particiones conservando `source_partition_id`. No modifica
la preparación ni los NPZ, ventanas, normalización o denominadores. Dos ejecuciones
con la misma preparación y regla generan el mismo contenido y hash del panel efectivo.
`benchmark` fija ese panel una sola vez para todos sus casos. Un panel ya filtrado
por otra métrica debe sustituirse por el manifest original para ejecutar otra selección.
En la simulación, el servidor comunica el manifest efectivo y la partición asignada
a cada nodo; esa asociación permanece fija desde la primera evaluación hasta test.
Los descartes no ocultan corrupción de artefactos, errores de modelo o fallos de
clientes después de fijar el panel: esos errores siguen abortando la ejecución.
No se cambian automáticamente métricas ni ventanas de referencia.

`macro-mase` y `macro-rmsse` dan igual peso a cada estación con escala definida;
`mean-*` conserva la media por origen como complemento. La RMSSE por estación
parte del MSE agrupado, no del promedio de raíces de sus orígenes. Los targets
rolling se solapan; los pares origen-target no son observaciones independientes.

`--max-train-windows` toma un subconjunto determinista repartido sobre las ventanas
elegibles. Modifica tanto el entrenamiento como sus pesos de agregación y debe
reservarse para pruebas de funcionamiento, no confundirse con el experimento completo.

## Alcance y privacidad

El servidor no abre NPZ de estaciones ni CSV raw: solo lee metadatos y recibe
pesos y resúmenes de errores. El bundle enviado a Flower incluye exclusivamente
el código de la aplicación y la configuración de proyecto, no datos, reports,
modelos, skills ni escaladores. Los hashes permiten detectar cambios en los datos
preparados y en el código local de preparación.

Sin embargo, todos los procesos de esta simulación comparten máquina y permisos
del sistema de archivos. El manifest contiene rutas y nombres de estaciones, y
el servidor ve actualizaciones individuales y métricas locales. No hay privacidad
diferencial, agregación segura ni protección frente a inferencia desde gradientes.
El SuperLink local usa comunicación loopback sin TLS; no debe exponerse así a red.

Para despliegue real serían necesarios almacenamiento y configuración por nodo,
autenticación/TLS y, según el riesgo, agregación segura y una definición explícita
de privacidad diferencial. La partición temporal local tampoco demuestra ausencia
de información futura entre estaciones; un estudio prospectivo exigiría otro protocolo.

## Controles sin federación y comparación

Se ofrecen dos controles entrenados, además de la persistencia en todas las métricas:

1. **Local:** un modelo independiente por estación, con la misma inicialización
   y las mismas ventanas locales. No hay agregación ni intercambio entre modelos;
    cada estación selecciona por su MASE de validación predeterminada. Se guardan las
   etapas seleccionadas por estación, que pueden diferir.
2. **Centralizado:** un único modelo accede a todas las ventanas de entrenamiento
   mediante `ConcatDataset`. Se concatena el índice de muestras, **no las series
   raw**; ninguna ventana cruza una estación, hueco o split. Conserva exactamente
   el escalador de entrenamiento de cada estación, por lo que el objetivo sigue
   siendo MSE normalizado por estación, ponderado por ventanas. Selecciona por
    la misma MASE macro de validación que FL. Es un control metodológico que
   deliberadamente elimina la restricción de acceso local a los datos.

Los controles usan las mismas funciones `train_local` y `evaluate_local` que
Flower. Una etapa aplica `local-epochs` a cada ventana; se ejecutan
`num-server-rounds` etapas y se reinicia AdamW en cada una. Esto iguala exposición
a datos entre métodos, pero no número/orden de actualizaciones globales, velocidad
ni costo de comunicación. FedProx aplica su penalización solo al caso federado;
no se añade esa penalización a los controles sin FL.

Para Granite se ejecutan cuatro algoritmos federados por foundation y controles
local/centralizado con LoRA o A congelada. FLoRA/FAIR no tienen modalidades sin
federación nativas. **Zero-shot** no instala adaptadores ni llama al bucle
de entrenamiento. No consume sus
ventanas de entrenamiento para ajustar pesos; sigue empleando el escalador local
existente, ajustado con entrenamiento, por lo que no representa un protocolo sin
ningún acceso a historia de entrenamiento. Se evalúa validación una vez y después
test, sin seleccionar escala, revisión o hiperparámetros con este último.

`benchmark` fija un manifest y ejecuta los modos secuencialmente en el orden
solicitado: por defecto local, centralizado, federado y zero-shot. Entre modos hay
una barrera: no comienza el siguiente hasta completar todos los candidatos del anterior.
Dentro de local/centralizado/zero-shot, los candidatos completos pueden ejecutarse
en procesos independientes, uno por GPU utilizable, sin cambiar ventanas, semillas,
presupuestos ni criterios de selección. `max-parallel-models` limita esa concurrencia;
no distribuye estaciones de un mismo candidato local ni implementa entrenamiento
de un modelo entre varias GPUs. CPU/fallback se mantiene secuencial.

En federado los candidatos permanecen secuenciales: Ray gestiona el paralelismo de
clientes de una sola federación automáticamente, reservando una GPU completa por
cliente. La concurrencia es el mínimo entre las GPUs utilizables y las estaciones
del panel efectivo; en CPU/fallback se ejecuta un cliente cada vez y no se reservan
GPUs. No hay ajustes manuales `workers`/`client-gpus`. No se presupone coordinación
de GPUs entre runtimes Ray independientes ni se reconfigura un SuperLink ya iniciado.
Un fallo detiene/cancela los procesos del benchmark, marca los casos no finalizados
y evita comparar un panel o una matriz incompletos. El orden de terminación de
candidatos no cambia la identidad ni el orden de las filas de resultados.
`compare` requiere el mismo SHA del manifest, clientes completos y recuentos de
targets compatibles. Emite una tabla global y otra por estación, además de
`pairwise_comparison.csv` y `station_pairwise_comparison.csv`: todos los pares
ordenados A/B, sin filtrar familias, adaptaciones ni hiperparámetros. Las diferencias
globales son de MAE, RMSE, macro-MASE y macro-RMSSE; las locales de MAE, RMSE,
MASE y RMSSE. Delta = A − B (negativo favorece A); mejora (%) = `100 × (1 − A/B)`
(positiva favorece A, indefinida si B=0 o falta una métrica).
La mejora atribuible a adaptación frente a zero-shot solo se añade cuando hay una
única referencia del mismo modelo, revisión y configuración de inferencia, incluida
la escala de FlowState. Su ausencia o ambigüedad no bloquea las comparaciones
generales. Se registran escala de FlowState y revisión en la tabla global; avisa cuando las
versiones de ejecución, semillas o presupuestos difieren. La mejora porcentual
frente a persistencia es `100 × (1 − MAE_modelo / MAE_persistencia)`; queda vacía
si la MAE de persistencia es cero. Las comparaciones son descriptivas, sin afirmar
significación estadística ni independencia de los orígenes rolling solapados.

## Trazabilidad con MLflow

El logging operativo es independiente del tracking: `benchmark.log` global,
logs por candidato y `run.log` se actualizan durante la ejecución. Una cola
serializa los registros en el coordinador; los subprocesos envían su salida por
pipes y cada uno recibe una única GPU visible. El flujo externo de Flower se
captura expresamente en `flower.log`, no se presupone que los handlers Python del
coordinador capturen automáticamente procesos externos. Los actores guardan su
detalle por partición en `clients/`. Los logs no incluyen series ni predicciones.
El heartbeat informa de estado y tiempo sin nuevos eventos; no mide avance real,
porcentaje de cómputo o ETA. `benchmark.json` se actualiza con reemplazo atómico.

Cada ejecución registra un run MLflow desde el servidor o coordinador del control,
no desde cada actor Ray. Se guardan parámetros, geometría, hash del manifest,
revisión oficial del checkpoint y métricas globales y por estación para cada
split/etapa. Los JSON, CSV y checkpoints se registran mediante una lista permitida;
no hay autologging ni subida recursiva de directorios de datos. Se conserva además
el resultado local aunque se use un servidor MLflow remoto. Los pesos Granite
registrados son **solo el adaptador**: para reconstruir el predictor se requiere
el backbone en la revisión anotada y el escalador local correspondiente.
FFA guarda A congelada y FLoRA la historia acumulada. `restore_checkpoint` verifica
revisión/capas y reconstruye esos estados. Se registran tiempo, parámetros,
refinamiento, RSS y payload tensorial, no framing de red. El máximo RSS es del
proceso, no memoria exclusiva del modelo. `resource_metrics.json` registra el
tiempo total del coordinador. MLflow se fija a 3.17.2 en ambos entornos.
Los contadores de bytes son de entrenamiento; no incluyen evaluación ni descarga
inicial de pesos. La duración registrada empieza al entrar en el bloque de tracking,
después de preparar los datos y cargar metadatos del modelo.

`run.json` conserva versiones efectivas, procedencia Git de Granite y hashes del
código de esta aplicación. Flower resuelve su entorno aislado con uv y puede
diferir del lock de desarrollo; esos cambios deben revisarse antes de interpretar
comparaciones. MLflow no anonimiza nombres/rutas ni convierte la simulación en
privacidad formal; los runs y checkpoints deben protegerse según su sensibilidad.
El experimento se inicializa en el coordinador antes de los procesos concurrentes.
Los eventos y métricas de cada candidato permanecen en su propio run; el log
adjuntado al terminar tracking es un snapshot, no un servicio remoto de streaming.
