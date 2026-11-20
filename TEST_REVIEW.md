# Revisión de la suite

Se portaron los cambios del worktree `test-review` a la suite vigente. Se adaptaron
a PM25, NeuralForecast y al presupuesto de ventanas por época: no se copiaron las
pruebas antiguas de contaminantes o modelos retirados. No cambió el código de
producción ni el worktree de origen.

## Qué se simplificó

| Antes | Ahora |
|---|---|
| Dos benchmarks reales (con y sin MLflow) y un tercer test solo para comprobar el proceso hijo | Un backtest de un fold con MLflow real verifica PID, configuración aislada, logs y comparación; el modo sin tracking se comprueba en baselines. |
| Entrenar modelos para provocar un error en `compare` | Archivos de resultado sintéticos. El rechazo de recuentos erróneos mantiene intacto el total del panel, de forma que exige validar cada estación. |
| Cruces completos de modo, modelo, adaptador o padding en pruebas de orquestación | Casos representativos: se conservan las rutas que difieren y se prueban las máscaras, las pérdidas y la agregación donde se implementan. El control de épocas pasa de 45 a 5 casos de integración. |
| Un segundo enumerador de ventanas dentro de un test de padding | Calendario pequeño con orígenes esperados explícitos: stride, hueco horario, mínimos observados y cortes de evaluación. |
| Imports entre módulos `test_*` y stubs de Granite y Flower repetidos | Fixtures compartidas. Flower solo sustituye el transporte en esas pruebas: cliente, servidor, mensajes y agregación siguen siendo los reales. |

Las pruebas de selección ahora fuerzan que MAE y MASE elijan rondas distintas y
comprueban qué pesos se evalúan, no solo que exista un checkpoint. También se
comprueba que una reserva de monitorización sin targets observados no use la
validación futura como sustituto, aunque esta tenga datos.

## Qué se mantuvo

- Las 18 pruebas con checkpoints reales de Granite, incluida su matriz de
  adaptadores. No se trasladaron pruebas de la suite habitual a una categoría optativa.
- La cancelación de un proceso real con cierre de logs y MLflow, y la comparación
  con `NeuralForecast.cross_validation` real: no se sustituyen por tests de mocks.
- Regresiones de fugas temporales, reinicio entre folds, histórico FLoRA, ventanas
  por ronda, máscaras, muestreo, pesos de agregación y selección sin consultar test.

## Reproducción

```bash
uv run --frozen --no-sync pytest -q --durations=20
FD_AIRQUALITY_REAL_MODELS=1 uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check src tests experiments
```

## Medición

En el mismo worktree, con pytest y el mismo entorno congelado,
la suite habitual pasó de **68,9 s** a **45,0 s** (≈35 % menos en esa medición).
Se pasó de 671 a 558 casos aprobados, con las mismas 18 omisiones optativas.
Los JUnit están en `reports/test_review_before.xml` y `reports/test_review_after.xml`.

La ejecución completa en CPU, con `FD_AIRQUALITY_REAL_MODELS=1`, aprobó
**576 pruebas en 99,00 s**, incluidas las 18 con checkpoints reales, sin fallos
ni omisiones. El JUnit está en `reports/test_review_with_checkpoints.xml`.

Las mayores reducciones acumuladas por archivo fueron `test_execution.py`
(30,4 → 20,6 s) y `test_training_control.py` (7,8 → 0,9 s). Los tiempos pueden
variar con la máquina, las cachés y la carga. No se midió cobertura de líneas:
el recorte elimina combinaciones redundantes, no pretende ser una equivalencia
exhaustiva de todas ellas.
