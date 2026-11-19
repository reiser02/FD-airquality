# Validación temporal con datos faltantes en NeuralForecast

<blockquote style="border-left: 4px solid var(--vscode-charts-green, #388a34); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-charts-green, #388a34);"><strong>💡 Lectura en VS Code</strong></p>
<p>Abre la vista previa con <strong>Ctrl+Shift+V</strong>, o pulsa <strong>Ctrl+K y después V</strong> para verla al lado del editor. Los recuadros distinguen principios, recomendaciones y advertencias; los apartados desplegables reúnen detalles técnicos. Los colores se adaptan al tema mediante variables de VS Code, con colores alternativos si no están disponibles.</p>
</blockquote>

## Alcance y procedencia

Este documento integra, sin modificar los originales:

- [Uso de `available_mask` en NeuralForecast](neuralforecast-available-mask.md).
- [Validación cruzada temporal con datos faltantes](validacion-cruzada-temporal-con-datos-faltantes.md).

El objetivo es **diseñar una evaluación de pronósticos futuros con datos
faltantes** y explicar qué partes resuelve NeuralForecast y cuáles corresponden
al protocolo de evaluación. No se evalúa la reconstrucción de huecos históricos.

| Elemento | Alcance |
| --- | --- |
| Versión de la implementación | NeuralForecast **3.1.4**, commit [`69e431413b22cbb1a74d35db2fa7994092ce0ea4`][nf-version]. |
| Procedencia temporal | Documentos de partida: consultas del **4 de noviembre de 2026**. Integración y contraste adicional: **5 de noviembre de 2026**. |
| Evidencia | Inspección estática de código, documentación y apartados pertinentes de las fuentes bibliográficas. |
| Modelos y pérdidas | Ruta de modelos basados en ventanas, con NHITS como ejemplo; MAE, MSE y MQLoss en las condiciones indicadas. |
| Supuestos | Calendario de frecuencia definida; máscara binaria; valores finitos en las operaciones numéricas descritas. |
| Trabajo no realizado | No se ejecutaron entrenamientos, notebooks ni pruebas de NeuralForecast; no se analizaron datos reales ni se certificó un entorno instalado. |

La integración distingue tres niveles: **recomendación metodológica**,
**comportamiento comprobado de la biblioteca** e **implicación para su uso**.
Compartir fuentes no convierte esos niveles en equivalentes. Las aclaraciones
que conectan ambos documentos se resumen en la [sección 9](#coherencia).

<a id="indice"></a>

## Índice

1. [Resumen: qué controla cada operación](#resumen)
2. [Conservar calendario, verdad y máscara](#datos)
3. [Diseñar los cortes y separar las validaciones](#cortes)
4. [Seleccionar ventanas de entrenamiento](#ventanas)
5. [Normalizar: referencia por ventana, por serie y riesgo de fuga](#normalizacion)
6. [Contexto, pérdidas y supervisión enmascarada](#perdidas)
7. [Validación interna y early stopping](#validacion-interna)
8. [Puntuación externa, agregación y test final](#evaluacion-externa)
9. [Cómo encajan los dos documentos](#coherencia)
10. [Protocolo integrado y lista de comprobación](#protocolo)
11. [Limitaciones y decisiones pendientes](#limites)
12. [Fuentes y localizadores](#fuentes)

**Lectura práctica:** [cortes](#cortes) → [puntuación](#evaluacion-externa) →
[protocolo por fold](#protocolo).

---

<a id="resumen"></a>

## 1. Resumen: qué controla cada operación

| Operación | Pregunta que resuelve | Lo que no garantiza |
| --- | --- | --- |
| Corte temporal externo | ¿Qué información existía al emitir el pronóstico? | Que cualquier preprocesamiento o estado del modelo respete ese corte. |
| Selección de ventanas | ¿Hay suficientes observaciones para admitir una muestra de entrenamiento? | Que la ventana sea estadísticamente suficiente o que se filtren igual las de validación. |
| Normalización temporal | ¿Con qué centro y escala se representa la ventana? | Que el relleno desaparezca de la entrada. |
| Máscara de pérdida | ¿Qué objetivos aportan error supervisado? | Que el relleno del contexto no influya en la arquitectura. |
| Validación interna | ¿Qué señal se utiliza para seguir el entrenamiento o detenerlo? | Que `ptl/val_loss` sea la métrica externa deseada. |
| Evaluación externa | ¿Qué precisión tiene el procedimiento sobre objetivos originales observados? | Conocer el error en posiciones sin verdad disponible. |

<blockquote style="border-left: 4px solid var(--vscode-editorInfo-foreground, #3794ff); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-editorInfo-foreground, #3794ff);"><strong>🔵 Principio central</strong></p>
<p>Los cortes se construyen por tiempo; las entradas se preparan con información autorizada; los objetivos ausentes no se convierten en etiquetas verdaderas; la evaluación externa utiliza la verdad original observada.</p>
</blockquote>

Las tres funciones internas de disponibilidad son complementarias: **el mínimo
decide si entra una muestra; el scaler establece su referencia numérica; la
máscara de pérdida determina qué objetivos aportan supervisión**. Ninguna
sustituye al diseño temporal externo. [Código: ventanas][nf-windows-train],
[normalización][nf-normalization] y [pérdida][nf-train-step].

---

<a id="datos"></a>

## 2. Conservar calendario, verdad y máscara

Antes de rellenar los `NaN`, conserva tres objetos distintos:

1. **Serie original:** contiene la verdad disponible para puntuar.
2. **Máscara original de observación:** `1` si el valor está observado y `0` si
   está ausente.
3. **Serie preparada para el modelo:** contiene los rellenos y transformaciones
   necesarios, sin sobrescribir la verdad original.

```text
Serie original: [10, NaN, 14, 0]
Relleno cero:   [10,   0, 14, 0]
Máscara:        [ 1,   0,  1, 1]
```

El cero de la última posición está observado; el de la segunda posición es un
relleno. Solo el cero observado aporta verdad para evaluar. **No reconstruyas
la máscara comprobando si el valor preparado es distinto de cero.**

En NeuralForecast hay que preservar esa distinción al proporcionar
`available_mask`. Si la columna no existe, el dataset añade una máscara de unos;
no deduce qué ceros procedían de `NaN`. Por tanto, rellenar primero y omitir la
máscara hace que esas posiciones se presenten como disponibles. [Código:
creación de la máscara][nf-ensure-mask].

**Enmascarar no hace seguros por sí solo los `NaN` o infinitos dentro de la red.**
La comprobación de faltantes del core considera las filas disponibles, pero eso
no equivale a una imputación del resto. La explicación de pérdidas y ejemplos
de este documento presupone rellenos finitos y denominadores válidos.
[Código: comprobación de faltantes][nf-check-nan] y [reductor de pérdida][nf-weighted-mean].

<blockquote style="border-left: 4px solid var(--vscode-editorError-foreground, #f14c4c); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-editorError-foreground, #f14c4c);"><strong>🔴 No comprimas el calendario</strong></p>
<p>En una serie de frecuencia regular, haz explícitas las fechas que deberían existir. No elimines las filas con <code>NaN</code> para concatenar los valores restantes: cambiarías las distancias temporales, los retardos y el significado del horizonte.</p>
</blockquote>

Rellenar con una constante cero **no incorpora por sí mismo información futura**,
pero puede modificar las predicciones. Cero y último valor observado son
alternativas que pueden compararse; ninguna es universalmente óptima. Para
`forward fill`, define también qué hacer si no existe una observación anterior.

Evita imputar contextos históricos con sus propios objetivos posteriores. Una
interpolación o un `bfill` aplicados antes de extraer las ventanas pueden hacerlo,
incluso si toda la serie utilizada pertenece al entrenamiento del fold externo.
La política de preparación debe ser compatible con el origen de cada muestra.
[Fundamento metodológico: fuga en preprocesamiento][sklearn-leakage] y
[evaluación de pronósticos, §3.5][forecast-eval].

[↑ Volver al índice](#indice)

---

<a id="cortes"></a>

## 3. Diseñar los cortes y separar las validaciones

### 3.1. Walk-forward con horizonte fijo

Utiliza **walk-forward** o evaluación con **origen de predicción móvil**
(*rolling origin*). En cada origen `c_k`, se dispone del pasado y se emite un
pronóstico de `H` períodos posteriores. No se barajan aleatoriamente los
timestamps para formar los folds. [FPP3, §5.10][fpp-cv].

Ejemplo didáctico con datos diarios, `H=7` y avance de siete días:

| Fold | Historial disponible hasta el origen | Horizonte externo para evaluar |
| --- | --- | --- |
| 1 | Desde el inicio hasta el 30 de junio | Del 1 al 7 de julio |
| 2 | Desde el inicio hasta el 7 de julio | Del 8 al 14 de julio |
| 3 | Desde el inicio hasta el 14 de julio | Del 15 al 21 de julio |

La tabla describe una ventana de historial **expansiva**. Puede utilizarse una
ventana deslizante de longitud fija si así funcionará el sistema. Si hay
validación interna, una parte del historial de la tabla se reserva para ella:
no todo ese historial aporta necesariamente gradientes. [Evaluación y
particionado, §§2 y 4.1][forecast-eval].

Fija antes de comparar modelos:

- **Horizonte `H`** y frecuencia de la serie.
- **Avance entre orígenes:** reproduce cada cuánto se emitirán pronósticos.
- **Longitud del historial inicial** y, si procede, de la ventana deslizante.
- **Política de reentrenamiento:** en cada origen, periódica o sin reajuste de
  pesos; no es lo mismo actualizar el contexto que reentrenar.
- **Disponibilidad real:** un dato fechado antes del corte puede no haber
  llegado todavía; incorpora esos retrasos si existen.

<blockquote style="border-left: 4px solid var(--vscode-editorWarning-foreground, #bf8803); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-editorWarning-foreground, #bf8803);"><strong>⚠️ El horizonte no se cuenta por valores válidos</strong></p>
<p>Si faltan dos de los siete valores del horizonte, este sigue abarcando siete días. No se alarga hasta reunir siete objetivos observados. Si el avance es menor que <code>H</code>, los horizontes se solapan y sus errores no deben tratarse como muestras independientes.</p>
</blockquote>

Que un valor evaluado pase al historial de un fold posterior es compatible con
walk-forward: en el nuevo origen ya pertenece al pasado, siempre que estuviera
disponible entonces. No lo es reutilizar pesos o estadísticas aprendidos con
datos posteriores al origen que se está simulando. [§§2 y 3.5][forecast-eval].

### 3.2. Validación interna, fold externo y test final

Para evitar ambigüedades, este documento utiliza estos nombres:

| Nombre | Función | Información que puede influir en el modelo |
| --- | --- | --- |
| Ajuste dentro del fold | Optimizar los pesos y ajustar preprocesamiento fijo. | Su tramo autorizado de entrenamiento. |
| Validación interna | Monitorizar el entrenamiento y, si se usa, decidir el early stopping. | Puede influir en la parada; no es el horizonte externo de ese fold. |
| Horizonte externo del fold | Puntuar la configuración candidata en la CV. | Se consulta después de generar el pronóstico; su puntuación sirve para seleccionar configuraciones. |
| Test temporal final | Evaluar la configuración ya seleccionada. | No se utiliza para elegir modelo, imputación, umbral, scaler o hiperparámetros. |

```text
Dentro de un fold de selección:

[ ajuste de pesos ][ validación interna ] | c_k | [ horizonte externo: H ]
<------ historial disponible hasta c_k ------->  <---- puntuación CV ---->

Después de seleccionar la configuración:

[ período de desarrollo: folds de selección ][ test temporal final reservado ]
```

La validación interna es opcional si no se emplea para la parada u otras
decisiones. Si se reserva, sus etiquetas no deben entrar en la pérdida de ajuste
ni en el ajuste de preprocesadores fijos bajo esta separación. Sus valores ya
conocidos sí pueden formar parte del contexto al pronosticar desde `c_k`.

En `NeuralForecast.fit`, esta versión exige `val_size=0` o `val_size>=h`;
activar early stopping exige un `val_size` positivo. Reservar más de `h`
períodos puede generar varias ventanas internas: no convierte su horizonte en
un único pronóstico más largo. Estas comprobaciones de `fit` no deben suponerse
idénticas en todas las rutas de CV. [Comprobaciones del core][nf-fit-validation-size]
y [ventanas de validación][nf-windows-val].

En la construcción interna de ventanas de NeuralForecast, la rama de
entrenamiento retira los últimos `val_size + test_size` períodos antes del
padding. Esto describe el corte de **ventanas**, no una garantía sobre todos los
pasos anteriores del core. [Código][nf-windows-train].

<details>
<summary>🔍 Si se utiliza <code>NeuralForecast.cross_validation</code></summary>

El nombre de un argumento no determina el papel metodológico de los datos:

- `val_size` reserva validación interna del modelo; no debe confundirse con los
  horizontes externos que se puntúan para seleccionar configuraciones.
- `test_size` puede definir la extensión temporal evaluada por la llamada a CV.
  Si esa llamada se usa para seleccionar modelos, ese tramo no es por ello un
  test final independiente.
- En la configuración ordinaria con horizonte del modelo `H`, el código calcula
  `test_size = H + step_size × (n_windows − 1)` cuando se especifica
  `n_windows`. No son necesariamente bloques disjuntos.
- `refit=False`, valor predeterminado, ajusta al comienzo y reutiliza el modelo;
  `refit=True` reajusta en cada ventana; un entero positivo permite reajustes
  periódicos. Debe coincidir con la operativa que se pretende medir.

[Código: parámetros, tamaño y rama sin refit][nf-cv-config];
[bucle con refit][nf-cv-refit]. No se ha ejecutado esta API ni se garantiza con
ello una reinicialización concreta de todos los estados entre ajustes.

</details>

[↑ Volver al índice](#indice)

---

<a id="ventanas"></a>

## 4. Seleccionar ventanas de entrenamiento

### 4.1. Contexto y objetivos dentro del historial

Cada muestra contiene un contexto de longitud `L` y un bloque de objetivos de
longitud `H`:

```text
[ contexto: L posiciones ][ objetivos: H posiciones ]
```

«Objetivos futuros» significa aquí **posteriores al contexto de una muestra
histórica**. No significa utilizar el futuro del horizonte externo. Los
objetivos observados que aportan pérdida de ajuste deben pertenecer al tramo
autorizado de entrenamiento del fold. [Construcción][nf-windows-train].

### 4.2. Umbral de disponibilidad y mínimos efectivos

`training_data_availability_threshold` admite:

- Un `float` entre `0.0` y `1.0`, aplicado a contexto y objetivos.
- Una lista o tupla de dos `float`: `[fracción_contexto, fracción_objetivos]`.

En esta versión, los enteros `0` y `1` no sustituyen a `0.0` y `1.0`: se rechazan
también dentro de la lista o tupla. [Validación del parámetro][nf-threshold].

Para una ventana univariada, con `H>0` y máscara binaria, los mínimos son:

$$
k_{\mathrm{contexto}}=\max\left(1,\operatorname{int}(L\alpha)\right),
\qquad
k_{\mathrm{objetivos}}=\max\left(1,\operatorname{int}(H\beta)\right).
$$

Se exigen **ambas** condiciones:

$$
\sum_{t=1}^{L}m^{\mathrm{contexto}}_t\geq k_{\mathrm{contexto}},
\qquad
\sum_{t=1}^{H}m^{\mathrm{objetivos}}_t\geq k_{\mathrm{objetivos}}.
$$

El código suma la máscara; solo con valores `0` y `1` esa suma puede interpretarse
como un número de observaciones. [Cálculo exacto][nf-availability].

Para `L=24` y `H=6`:

| Configuración | Mínimo en el contexto | Mínimo en los objetivos |
| --- | ---: | ---: |
| `0.0`, valor predeterminado | 1 de 24 | 1 de 6 |
| `0.5` | 12 de 24 | 3 de 6 |
| `[0.75, 1.0]` | 18 de 24 | 6 de 6 |
| `1.0` | 24 de 24 | 6 de 6 |

El valor predeterminado es **muy permisivo**, no una recomendación de suficiencia
estadística. Además, `int` trunca; no redondea hacia arriba. Con `H=3` y `β=0.8`:

$$
\max(1,\operatorname{int}(3\cdot0.8))=2.
$$

Bastan dos observaciones de tres, menos del 80 % nominal. Para exigir todos los
objetivos, `[fracción_contexto, 1.0]` es una opción de configuración, no una
elección universalmente superior. Debe compararse sin cambiar después los
objetivos del scoring externo. [Código][nf-availability].

### 4.3. Padding, filtrado y muestreo

<details>
<summary>🔍 Orden de preparación y selección de ventanas</summary>

La secuencia de la ruta de entrenamiento es:

1. Retirar los tramos internos reservados de validación y test.
2. Aplicar padding y construir ventanas de tamaño `L+H`.
3. Identificar las ventanas que cumplen los mínimos.
4. Muestrear entre las ventanas admisibles para ese paso de entrenamiento.

El padder añade `H` posiciones a la derecha; con `start_padding_enabled`, añade
también `L−1` a la izquierda. La máscara del padding vale cero. El dataset puede
añadir padding izquierdo adicional para igualar longitudes entre series.
[Padder][nf-padder], [dataset][nf-dataset-padding] y [ventanas][nf-windows-train].

Los mínimos se calculan sobre longitudes que incluyen padding. Con `H>0` puede
admitirse un horizonte parcialmente rellenado, pero no uno completamente
indisponible. Si un batch no tiene ninguna ventana admisible, se lanza
`No windows available for training`; no se omite automáticamente.

Para `windows_batch_size=W>0`:

| Ventanas admisibles | Selección en ese paso |
| --- | --- |
| Menos de `W` | `W` extracciones con reemplazo: puede haber repeticiones. |
| Al menos `W` | `W` ventanas sin reemplazo. |
| `windows_batch_size=None` | Todas las ventanas admisibles. |

El muestreo no pondera las ventanas por su número de objetivos disponibles. Los
solapamientos y las repeticiones pueden representar varias veces el mismo
timestamp. [Código de muestreo][nf-sampling].

</details>

Los mínimos no controlan la duración de los huecos, la cercanía de las
observaciones al origen, la identificación de estacionalidades ni la estabilidad
de una escala. Un cero observado cuenta si su máscara es uno; uno imputado no
cuenta si su máscara es cero.

**Compartir pesos entre series no implica contar disponibilidad conjuntamente
entre todas ellas.** En NHITS, `n_series=1` internamente y se filtra cada ventana
serie-origen. En arquitecturas multivariadas, el mínimo es
`max(1, int(longitud × fracción × n_series))` y se suma disponibilidad entre tiempo
y series; no se exige un mínimo separado a cada serie. [Configuración][nf-n-series]
y [conteo][nf-availability].

<blockquote style="border-left: 4px solid var(--vscode-editorWarning-foreground, #bf8803); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-editorWarning-foreground, #bf8803);"><strong>⚠️ El umbral no filtra la evaluación</strong></p>
<p>Las ramas de validación y predicción no aplican estos mínimos de entrenamiento. Conservan las ventanas construidas por sus propias reglas de corte y padding. Subir el umbral no evita automáticamente validar sobre bloques sin objetivos observados.</p>
</blockquote>

[Código de validación y predicción][nf-windows-val]. Descartar una muestra de
entrenamiento no equivale a eliminar timestamps del calendario ni autoriza a
escoger solo períodos fáciles para evaluar.

[↑ Volver al índice](#indice)

---

<a id="normalizacion"></a>

## 5. Normalizar: referencia por ventana, por serie y riesgo de fuga

### 5.1. Tres estrategias diferentes

| Estrategia | Referencia | Configuración o procedimiento |
| --- | --- | --- |
| Compartida por todo el dataset | Observaciones agrupadas de diferentes series. | Preprocesamiento distinto; no equivale a `local_scaler_type`. |
| Fija por serie | Historial autorizado de entrenamiento de cada serie. | `local_scaler_type` en el core, con las salvedades de la sección 5.4. |
| Por ventana | Contexto disponible de cada muestra. | `scaler_type` en el modelo, mediante TemporalNorm. |

Las capas fija por serie y temporal pueden coexistir. Para usar solo la primera,
puede desactivarse la temporal con `scaler_type="identity"`. El escalado por
ventana es una elección, no una obligación de NeuralForecast.
[Documentación de escalado][nf-scaling-doc] y [core][nf-core-scaler].

**Ajustar un preprocesador fijo solo con entrenamiento y recalcular estadísticas
con el contexto de una nueva ventana son operaciones distintas.** La segunda no
incorpora por sí misma los objetivos futuros: utiliza entradas conocidas al
emitir ese pronóstico. No es necesario congelar la media de la última ventana
de entrenamiento para evitar fuga.

### 5.2. Qué calcula TemporalNorm

Se aplica la misma regla a todas las muestras, pero cada ventana tiene sus
estadísticas. Para los scalers afines:

$$
z_t=\frac{x_t-a_w}{s_w},
\qquad
\hat y_t=s_w\hat z_t+a_w.
$$

| `scaler_type` | Centro `a_w` | Escala `s_w`, antes de las protecciones numéricas |
| --- | --- | --- |
| `"standard"` | Media | Desviación estándar poblacional. |
| `"robust"` | Mediana | Mediana de las desviaciones absolutas respecto a la mediana. |
| `"minmax"` | Mínimo | Máximo menos mínimo. |
| `"identity"` | Cero | Uno. |

Las estadísticas son propias de cada ventana, serie y canal temporal, no una
mezcla de ejemplos del batch. La transformación alcanza el target y las
covariables temporales seleccionadas, no las estáticas mediante este método.
[Normalización común][nf-normalization] y [transformación][nf-temporal-transform].

Para estimarlas, la normalización copia `available_mask` y anula **todo el
horizonte** en esa copia. Solo interviene el contexto observado: ni los rellenos
del contexto ni los objetivos futuros observados de la muestra histórica. La
máscara original permanece disponible para calcular después la pérdida.
[Código][nf-normalization].

Para `standard`, con observaciones finitas y `n_w>0`:

$$
\mu_w=\frac{\sum_t m_t x_t}{n_w},
\qquad
\sigma_w=\sqrt{\frac{\sum_t m_t(x_t-\mu_w)^2}{n_w}},
\qquad
n_w=\sum_t m_t.
$$

Estas fórmulas omiten `eps` y las protecciones. `standard` sustituye una
desviación nula por uno y añade `eps`. `robust` calcula una **mediana** de
desviaciones absolutas, pese a una descripción textual inconsistente que habla
de media; si esa escala es cero, usa una alternativa basada en la desviación
estándar y, si sigue siendo cero, uno, antes de añadir `eps`.
[Standard][nf-standard] y [robust][nf-robust].

### 5.3. Excluir de las estadísticas no elimina el relleno

```text
Contexto: [10, 0, 14]
Máscara:  [ 1, 0,  1]
```

Las estadísticas estándar usan `[10, 14]`: `μ=12` y `σ=2`. Ignorando `eps`, el
vector completo se transforma en:

```text
Contexto normalizado: [-1, -6, 1]
```

<blockquote style="border-left: 4px solid var(--vscode-charts-purple, #b180d7); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-charts-purple, #b180d7);"><strong>🔎 La máscara no vuelve invisible el relleno</strong></p>
<p>El cero imputado no alteró las estadísticas, pero se transformó igualmente y pasó a valer aproximadamente <code>−6</code>. No desapareció ni se convirtió automáticamente en cero normalizado. La influencia posterior depende de la arquitectura.</p>
</blockquote>

Los `NaN` temporales utilizados para excluir posiciones de las estadísticas no
imputan la entrada: después se transforma el vector original.
[Estadísticas enmascaradas][nf-masked-stats] y [transformación][nf-temporal-transform].
En inferencia se recalculan centro y escala con el contexto disponible de la
nueva ventana.

### 5.4. Salvedades del scaler fijo por serie

Un scaler fijo ajustado exclusivamente con datos autorizados **no constituye
fuga por definición**. Sin embargo, `local_scaler_type` no debe confundirse con
TemporalNorm por dos razones comprobadas en esta versión:

1. **Máscara:** el ajuste mediante `GroupedArray` no recibe `available_mask`.
   Aunque se excluye la columna de máscara del escalado, no puede suponerse que
   se excluyan los rellenos finitos de las demás columnas. [Código][nf-core-scaler].
2. **Orden de preparación:** `_prepare_fit` construye el dataset y ajusta el
   scaler sobre los datos recibidos **antes** de que el modelo retire los tramos
   reservados de sus ventanas. En `fit(df, val_size=...)`, el `df` recibido puede
   contener también la validación interna. En la ruta de CV sin refit, se prepara
   de ese modo el `df` completo, incluido el tramo reservado para evaluar.
   [Preparación][nf-prepare-fit], [llamada desde `fit`][nf-core-fit] y
   [CV sin refit][nf-cv-no-refit].

<blockquote style="border-left: 4px solid var(--vscode-editorError-foreground, #f14c4c); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-editorError-foreground, #f14c4c);"><strong>❗ Cortar ventanas no protege automáticamente el preprocesamiento fijo</strong></p>
<p>Si se activa <code>local_scaler_type</code> y se entrega un dataframe que incluye datos reservados, esas posiciones pueden entrar en el ajuste de su referencia. La exclusión posterior mediante <code>val_size</code> o <code>test_size</code> no deshace ese ajuste. Esto es incompatible con un protocolo que prohíba usar esos valores para estimar estadísticas.</p>
</blockquote>

Es una advertencia **condicionada a esa configuración y ruta**, no una afirmación
de que toda evaluación de NeuralForecast tenga fuga ni una medición de su
efecto. Para respetar la separación propuesta, una opción es no activar esa capa
fija y utilizar la normalización temporal; otra es controlar externamente un
escalado ajustado solo con el tramo permitido y con tratamiento explícito de
faltantes. Si se escala fuera del core, hay que invertir también esa
transformación al puntuar.

### 5.5. Motivación y costes de la referencia por ventana

La documentación relaciona el escalado con estabilidad numérica, convergencia y
aprendizaje con parámetros compartidos entre series de magnitudes diferentes.
Olivares y colaboradores presentan TemporalNorm y experimentos acotados sobre
sus variantes; esto no demuestra superioridad universal. [Documentación][nf-scaling-doc],
[§3.2][hint-method] y [apéndice C.1][hint-ablation].

Como consecuencia de la transformación, estas ventanas tienen el mismo patrón
estandarizado, salvo detalles numéricos:

```text
Ventana A: [10, 12, 14]
Ventana B: [100, 120, 140]
```

La referencia por ventana se adapta al nivel y variabilidad actuales. Una
referencia compartida por todo el dataset no elimina las diferencias relativas
entre series; una fija por serie sí reduce diferencias entre ellas, pero no se
adapta igual a los cambios entre ventanas de la misma serie.

Los costes son relevantes: ventanas de distinto nivel absoluto pueden resultar
equivalentes para la red; invertir el escalado no garantiza conservar ese nivel
como información predictiva. Con pocas observaciones, las estadísticas pueden
ser inestables aunque existan protecciones numéricas. Cuando los niveles
relativos entre ventanas importan, la documentación contempla preferir una
referencia fija por serie. La elección debe evaluarse dentro del protocolo, no
darse por óptima. [Advertencia de la documentación][nf-scaling-doc].

[↑ Volver al índice](#indice)

---

<a id="perdidas"></a>

## 6. Contexto, pérdidas y supervisión enmascarada

### 6.1. Separación de máscaras e influencia en NHITS

`_parse_windows` separa:

- `insample_mask`: disponibilidad del contexto, entregada al modelo.
- `outsample_mask`: disponibilidad de objetivos, entregada a la pérdida.

La segunda procede de la máscara original, no de la copia anulada sobre todo
el horizonte para estimar el scaler. [Separación][nf-parse] y [entrenamiento][nf-train-step].

**Recibir una máscara no implica utilizarla antes de todas las operaciones.**
En `NHITS.forward`, los residuos se inicializan con `insample_y`; el primer
bloque los procesa antes del enmascarado. Después de cada bloque se actualizan
como `(residuals - backcast) * insample_mask`. Además, el nivel inicial del
pronóstico utiliza el último valor del contexto sin consultar su máscara.
[Código de NHITS][nf-nhits].

Por tanto, aun excluyendo un relleno de las estadísticas y de su error como
objetivo, ese valor puede influir cuando aparece en el contexto. No se extrapola
este comportamiento concreto a todas las arquitecturas.

### 6.2. MAE y MSE: denominador efectivo

Para cada ventana `w` y posición del horizonte `t`, sea:

$$
q_{w,t}=m_{w,t}h_t,
$$

donde `m` es la disponibilidad y `h_t=1` salvo que se configure
`horizon_weight`. En las fórmulas de pérdida, `y` representa el objetivo en la
**escala entregada a esa llamada**, no necesariamente las unidades originales.

Con valores finitos y suma de pesos positiva:

$$
L_{\mathrm{MAE}}=
\frac{\sum_{w,t}q_{w,t}|y_{w,t}-\hat y_{w,t}|}
{\sum_{w,t}q_{w,t}},
\qquad
L_{\mathrm{MSE}}=
\frac{\sum_{w,t}q_{w,t}(y_{w,t}-\hat y_{w,t})^2}
{\sum_{w,t}q_{w,t}}.
$$

Se reducen todas las dimensiones de la llamada, no una media independiente por
ventana seguida de otra media. El denominador es la **masa efectiva de pesos**:
con máscara binaria y sin pesos de horizonte adicionales, cuenta objetivos
disponibles. [Reductor][nf-weighted-mean], [pesos][nf-loss-weights], [MAE][nf-mae]
y [MSE][nf-mse].

Ejemplo didáctico en una misma escala:

| Objetivo | Predicción | Máscara | Error absoluto que cuenta |
| ---: | ---: | ---: | ---: |
| 10 | 12 | 1 | 2 |
| 0 imputado | 999 | 0 | Ninguno |
| 20 | 17 | 1 | 3 |

$$
L_{\mathrm{MAE}}=\frac{2+3}{2}=2.5.
$$

No se divide entre tres ni se enseña que el objetivo ausente vale cero. Una
ventana con más objetivos observados aporta más términos **dentro de esa
llamada**. Se cuentan apariciones en ventanas, no timestamps únicos; además,
cada llamada tiene su propio denominador. No se optimiza acumulando todos los
numeradores y denominadores en una única media global de época. Los pesos del
horizonte no cambian el filtro de disponibilidad del entrenamiento.

Con las condiciones anteriores, en MAE, MSE y MQLoss una posición futura con
máscara cero no aporta **gradiente directo de su propio error**. Esto no congela
su predicción ni los parámetros compartidos: estos reciben gradientes de otros
objetivos, ventanas y series. Tampoco anula la influencia del contexto.

### 6.3. Precisión de versión: MQLoss suma los cuantiles

La máscara de cada objetivo se extiende a todos sus cuantiles. Sin embargo, en
3.1.4 la reducción entre cuantiles implementada no es la media convencional.
El código contiene:

```python
quantiles = self.quantiles[None, None, None, :]
losses = (1 / len(quantiles)) * (
    quantiles * sq + (1 - quantiles) * s1_q
)
```

Después de expandir dimensiones, `quantiles` tiene forma `[1, 1, 1, Q]`, de modo
que `len(quantiles)=1`, no `Q`. Los pesos mantienen una dimensión final
singleton: el numerador se expande entre cuantiles, pero el denominador no
incorpora un factor `Q`. [Implementación][nf-mqloss].

Definiendo la pérdida pinball como:

$$
\rho_{\tau}(r)=\tau\max(r,0)+(1-\tau)\max(-r,0),
$$

el cálculo efectivo, para las formas previstas y denominador positivo, es:

$$
L_{\mathrm{MQ}}=
\frac{
\sum_{w,t}q_{w,t}\sum_{j=1}^{Q}
\rho_{\tau_j}(y_{w,t}-\hat y_{w,t,j})
}{\sum_{w,t}q_{w,t}}.
$$

<blockquote style="border-left: 4px solid var(--vscode-editorWarning-foreground, #bf8803); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-editorWarning-foreground, #bf8803);"><strong>⚠️ No confundir suma con media multicuantil</strong></p>
<p>En esta versión se suman las pérdidas entre cuantiles. El valor es <code>Q</code> veces la media convencional entre puntos y cuantiles; esto no cambia qué objetivos excluye la máscara, pero sí la magnitud de la pérdida y sus gradientes.</p>
</blockquote>

Ejemplo: con `y=0`, cuantiles `0.25` y `0.75`, y predicciones `−2` y `2`, cada
pérdida pinball es `0.5`: la suma es `1` y la media `0.5`. Para `Q` fijo, la
constante preserva el mínimo matemático y la ordenación ideal de puntuaciones;
no garantiza idéntica dinámica de optimización ni igualdad con una métrica
externa que promedie cuantiles. Es un hallazgo estático de esta versión.

### 6.4. Escala durante entrenamiento y validación

En la ruta examinada, MAE, MSE y MQLoss entrenan con objetivos y predicciones
**normalizados por ventana**. Para un scaler afín de escala positiva:

$$
|z-\hat z|=\frac{|y-\hat y|}{s_w},
\qquad
(z-\hat z)^2=\frac{(y-\hat y)^2}{s_w^2}.
$$

El escalado modifica así el peso relativo de errores de ventanas distintas,
además de transformar las entradas. En cambio, la ruta de `DistributionLoss`
utiliza objetivos anteriores a TemporalNorm y parámetros distribucionales
reescalados. No se generaliza la fórmula puntual a toda pérdida probabilística.
[Entrenamiento][nf-train-step].

En validación se conservan los objetivos anteriores a TemporalNorm y se evalúan
predicciones desnormalizadas o distribuciones reescaladas, con
`outsample_mask`. Si se activó `local_scaler_type`, esa escala anterior a
TemporalNorm ya está transformada por el core: **no son necesariamente las
unidades originales**. [Objetivos conservados][nf-validation-step] y
[`valid_loss`][nf-valid-loss]. La documentación también advierte que esa capa
fija afecta a la selección interna de modelos Auto. [Documentación, §3][nf-scaling-doc].

[↑ Volver al índice](#indice)

---

<a id="validacion-interna"></a>

## 7. Validación interna y early stopping

Aunque utiliza `outsample_mask`, la validación interna **no aplica el filtro de
mínimos del entrenamiento**. Para las pérdidas tensoriales puntuales examinadas,
hay varios niveles de agregación:

1. Cada bloque interno calcula una pérdida ponderada por objetivos disponibles.
2. `validation_step` combina los bloques por **número de ventanas**, no por
   número de objetivos disponibles.
3. `on_validation_epoch_end` promedia los escalares de los pasos y registra
   `ptl/val_loss`.

En el caso de un proceso, sin describir la reducción distribuida de Lightning:

$$
V_j=\frac{\sum_c n_{j,c}L_{j,c}}{\sum_c n_{j,c}},
\qquad
\texttt{ptl/val\_loss}=\frac{1}{J}\sum_{j=1}^{J}V_j,
$$

donde `c` identifica bloques internos, `j` pasos de validación y `n` cuenta
ventanas por bloque. [Combinación de bloques][nf-validation-step] y
[media final][nf-validation-epoch].

Ejemplo didáctico: dos bloques con igual número de ventanas, uno con una
observación y pérdida `0`, y otro con nueve observaciones y pérdida `10`, se
combinan como `5`. La media global por observaciones sería `9`.

<blockquote style="border-left: 4px solid var(--vscode-editorError-foreground, #f14c4c); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-editorError-foreground, #f14c4c);"><strong>❗ Cero interno no equivale a predicción perfecta</strong></p>
<p>Un bloque completamente enmascarado puede devolver un valor <em>forward</em> de cero por la protección del cociente y seguir entrando en la agregación. Eso no acredita acierto ni gradientes seguros. El early stopping incorporado monitoriza <code>ptl/val_loss</code> y hereda esta agregación.</p>
</blockquote>

[Protección del cociente][nf-weighted-mean] y [monitor de parada][nf-early-stopping].
Por tanto, **`ptl/val_loss` no es generalmente el MAE global de todos los
objetivos originales observados**: pueden diferir la escala, la pérdida y la
ponderación, además del conjunto evaluado.

Como recomendación de uso, comprueba la disponibilidad de los bloques internos
y la métrica que gobierna la parada. No confíes en subir
`training_data_availability_threshold` para resolver bloques vacíos de
validación. Si se requiere otra agregación de parada, hay que implementarla o
seleccionar los pasos de entrenamiento mediante un procedimiento compatible;
no se ha implementado aquí esa modificación.

[↑ Volver al índice](#indice)

---

<a id="evaluacion-externa"></a>

## 8. Puntuación externa, agregación y test final

### 8.1. Pronosticar todo el horizonte; puntuar solo la verdad observada

Genera una predicción para **cada fecha del horizonte**. Después, cruza las
predicciones con la serie original y su máscara, no con los objetivos rellenados.

```text
Objetivos originales: [10, NaN, 12, NaN, 15, 14, 13]
Predicciones:         [11,   9, 11,  10, 16, 14, 12]
Se puntúan:          [sí,  no, sí,  no, sí, sí, sí]
```

Si `O_k` contiene las fechas con objetivo originalmente observado dentro del
horizonte del fold `k`, en unidades originales:

$$
\operatorname{MAE}_k =
\frac{1}{|O_k|}
\sum_{t \in O_k} |y_t - \hat{y}_{t \mid c_k}|.
$$

Solo se calcula si `|O_k|>0` y las predicciones exigidas son válidas. En el
ejemplo hay cinco objetivos puntuables y la suma de errores es cuatro:
`MAE=0.8`. Es un ejemplo aritmético, no un resultado experimental.

Registra también la **cobertura de etiquetas** del horizonte:

$$
\operatorname{cobertura}_k=\frac{|O_k|}{H}.
$$

Esta cobertura expresa disponibilidad de verdad; no es cobertura de intervalos
de predicción ni porcentaje de pronósticos válidos.

<blockquote style="border-left: 4px solid var(--vscode-charts-green, #388a34); padding: 12px 16px; margin: 16px 0; background: var(--vscode-textBlockQuote-background, transparent); border-radius: 6px;">
<p style="color: var(--vscode-charts-green, #388a34);"><strong>✅ Reglas comunes a todos los modelos</strong></p>
<p>Usa los mismos cortes y objetivos originales observados. Marca los folds sin verdad como <strong>no evaluables</strong>, nunca como error cero. No puntúes contra rellenos ni elimines silenciosamente predicciones inválidas.</p>
</blockquote>

Si se exige una cobertura mínima, fija la regla antes de comparar modelos y
comunica las exclusiones. No escojas solo períodos completos para mejorar la
puntuación. Para calcular el error con una serie que aún contiene `NaN`,
selecciona las posiciones observadas **antes** de operar; multiplicar después
un `NaN` por cero no garantiza eliminarlo.

La salida de `cross_validation` incorpora el target del dataframe recibido;
en la ruta con refit, el resultado une predicciones con target e identificadores,
no con la máscara original. Si ese target estaba rellenado, hay que recuperar
externamente la verdad y la máscara guardadas. No se debe deducir disponibilidad
de que la columna `y` devuelta sea finita. [Salida sin refit][nf-cv-output] y
[salida con refit][nf-cv-refit].

### 8.2. Agregación externa propuesta

Para dar el mismo peso a cada par origen-fecha con objetivo observado, acumula
sumas de errores y conteos:

$$
\operatorname{MAE}_{\mathrm{CV}} =
\frac{
  \sum_k \sum_{t \in O_k} |y_t - \hat{y}_{t \mid c_k}|
}{
  \sum_k |O_k|
}.
$$

El denominador total debe ser positivo; si no lo es, la CV no tiene una
puntuación definida. Tampoco se deben ocultar fallos de predicción mediante
esta agregación. Esta es una **elección explícita del protocolo**, no una
descripción de `ptl/val_loss`.

- No equivale al promedio no ponderado de los MAE de folds con distintos conteos.
- Si los horizontes se solapan, una misma fecha puede aparecer desde varios
  orígenes: son pronósticos diferentes, no timestamps únicos.
- Presenta error y cobertura por fold y, si interesa, por paso del horizonte.
- Si se extiende a varias series, define además cómo ponderarlas: la media por
  apariciones observadas no da el mismo peso a cada serie ni corrige sus
  diferencias de escala. [Discusión de métricas y escalas, §3.3][forecast-eval].

Para puntuaciones probabilísticas, fija también qué cuantiles se evalúan y si
se suman o promedian. No supongas igualdad numérica con `MQLoss` 3.1.4 ni que un
evaluador externo de GIFT-Eval o Impermanent herede automáticamente la máscara
de NeuralForecast; habría que comprobar cada evaluador por separado.

### 8.3. Selección y test temporal final

Utiliza los folds externos de desarrollo para elegir modelo, hiperparámetros,
imputación, umbrales de disponibilidad, scaler, contexto y longitud de historial
si se comparan. La puntuación de esa selección no sustituye al test final.

Reserva un tramo cronológico final que no intervenga en esas decisiones. Una
vez seleccionada la configuración, evalúala siguiendo la política de
actualización fijada previamente. **Congelar la configuración no obliga a
congelar los pesos**: en un test walk-forward puede permitirse reentrenar con
observaciones anteriores que ya hayan llegado, si esa es la operativa declarada.
No se permiten cambios retrospectivos de configuración al ver los errores del
test. [Evaluación y particionado, §§2 y 4.1][forecast-eval].

Las métricas externas deben expresarse en la escala original: comprueba la
inversión de todas las capas de transformación. El core dispone de inversión
del scaler fijo antes de devolver los pronósticos; eso no convierte en cruda
la pérdida interna anterior a esa inversión. [Inversión][nf-core-inverse] y
[documentación][nf-scaling-doc].

[↑ Volver al índice](#indice)

---

<a id="coherencia"></a>

## 9. Cómo encajan los dos documentos

No es correcto trasladar automáticamente una recomendación del protocolo a una
garantía de la biblioteca. Estas distinciones eliminan las aparentes
contradicciones sin ocultar limitaciones de implementación:

| Afirmaciones que pueden parecer incompatibles | Interpretación integrada |
| --- | --- |
| «Ajustar normalización solo con train» / «recalcularla en inferencia». | El preprocesador fijo aprende una referencia con train; TemporalNorm calcula la suya con el contexto conocido de cada nueva ventana, no con su horizonte futuro. |
| «Los faltantes se ignoran» / «el cero imputado influye». | Se excluyen de estadísticas o de términos de pérdida concretos; el valor del contexto puede seguir entrando en la red. |
| «No entrenar con objetivos ausentes» / «admitir horizontes incompletos». | Una ventana puede conservar objetivos observados y enmascarar los demás; no se convierten los ausentes en etiquetas cero. |
| «Filtrar muestras incompletas» / «no eliminar fechas». | Se pueden descartar ventanas de entrenamiento manteniendo intacto el calendario de la serie y el horizonte externo. |
| «Los folds sin verdad no se puntúan» / «un bloque enmascarado devuelve cero». | Lo primero es la regla externa propuesta; lo segundo, un comportamiento interno que hay que vigilar, no imitar en el scoring. |
| «MAE en unidades originales» / «entrenamiento en escala normalizada». | Son métricas de fases distintas. La pérdida de entrenamiento y la de parada no tienen por qué coincidir con el MAE externo. |
| «Agregar por objetivos observados» / «promediar pérdidas internas». | La agregación externa se define acumulando errores y conteos; `ptl/val_loss` emplea los niveles de ponderación de la sección 7. |
| «Retirar validación y test» / «el scaler fijo puede utilizarlos». | El corte del modelo ocurre después de la preparación del core. Con `local_scaler_type` hay que controlar por separado qué datos ajustan la referencia. |
| «`test_size` en CV» / «test final intocable». | El argumento delimita datos de esa llamada, pero su función depende del uso: si sirve para seleccionar configuraciones, no es un test final independiente. |

Las fórmulas del MAE externo y de la pérdida interna pueden tener una forma
parecida y, aun así, representar cantidades diferentes por su escala, conjunto
de objetivos, repeticiones y agregación. Su equivalencia no debe asumirse.

---

<a id="protocolo"></a>

## 10. Protocolo integrado y lista de comprobación

### 10.1. Procedimiento por fold

1. **Fija `c_k` y el horizonte de calendario.** Conserva aparte los objetivos
   posteriores que solo se utilizarán para puntuar.
2. **Construye el historial autorizado.** Respeta el tamaño expansivo o
   deslizante y los retrasos de disponibilidad. Si hay varias series con pesos
   compartidos, no entrenes con información posterior al origen simulado.
3. **Separa validación interna si hace falta.** No utilices el horizonte externo
   del fold como señal de early stopping de ese mismo entrenamiento.
4. **Prepara entradas, máscara y escalado.** Conserva verdad y máscara originales;
   usa rellenos finitos compatibles con cada contexto. Ajusta preprocesadores
   fijos solo en el tramo permitido. Si usas `local_scaler_type`, revisa el orden
   y las salvedades de la sección 5.4.
5. **Forma y filtra ventanas de entrenamiento.** Aplica el umbral elegido a las
   máscaras originales. Registra si quedan ventanas admisibles; subirlo puede
   reducir mucho los datos o impedir un ajuste.
6. **Entrena según la política fijada.** La máscara de objetivos excluye la
   supervisión ausente en las pérdidas examinadas. Comprueba la escala y la
   disponibilidad de la validación interna si controla la parada.
7. **Pronostica las `H` fechas desde `c_k`.** La normalización temporal usa su
   contexto conocido; no sus objetivos externos. Deshaz las transformaciones
   necesarias para obtener las unidades originales.
8. **Puntúa con la verdad guardada.** Usa solo objetivos originales observados,
   registra errores absolutos, conteos, cobertura y fallos. Un fold sin verdad
   es no evaluable.
9. **Avanza al siguiente origen.** Incorpora solo información ya disponible y
   respeta la política de actualización; no reutilices estados aprendidos con
   fechas posteriores a ese origen.

Después, agrega los numeradores y conteos según la sección 8, selecciona la
configuración usando los folds de desarrollo y realiza la evaluación temporal
final sin volver a ajustar las decisiones al resultado del test.

### 10.2. Lista de comprobación

- [ ] Se conservan la serie original y la máscara anterior a toda imputación.
- [ ] Un cero observado y uno imputado mantienen máscaras diferentes.
- [ ] Se mantiene el calendario; no se concatenan observaciones a través de huecos.
- [ ] Horizonte, avance, historial y reentrenamiento reflejan la tarea real.
- [ ] Las etiquetas de ajuste no cruzan el tramo de entrenamiento autorizado.
- [ ] La validación interna y el horizonte externo tienen funciones separadas.
- [ ] Los preprocesadores fijos no se ajustan con los bloques reservados.
- [ ] Se ha revisado `local_scaler_type` si se activa; no se supone que use la máscara como TemporalNorm.
- [ ] El umbral usa flotantes y se han comprobado sus mínimos efectivos y las ventanas admisibles.
- [ ] No se confunde máscara de pérdida con invisibilidad del relleno para la red.
- [ ] Se conoce la escala y agregación de la señal de early stopping.
- [ ] Los bloques internos sin objetivos observados no pasan inadvertidos.
- [ ] El scoring externo usa verdad original, unidades originales y una máscara común.
- [ ] Se comunican cobertura, folds no evaluables, exclusiones y fallos de predicción.
- [ ] Se documenta la ponderación de orígenes, horizontes, series y cuantiles.
- [ ] El test final no interviene en la selección de configuración.

---

<a id="limites"></a>

## 11. Limitaciones y decisiones pendientes

La puntuación sobre objetivos observados mide precisión **condicionada a que
exista verdad disponible**. No permite conocer directamente el error en los
objetivos originalmente ausentes. Si faltan valores durante averías, picos u
otros eventos especiales, los observados pueden no representar esas situaciones.
Hay que examinar la causa de las ausencias antes de generalizar.
[FPP3, §13.9, «Missing values»][fpp-missing].

Además:

- Los ejemplos y cálculos son didácticos, no resultados sobre un dataset real.
- No se establece un umbral, imputador o scaler universalmente superior.
- Un mínimo admitido por el código no garantiza suficiencia estadística.
- Las garantías sobre error y gradiente se limitan a las pérdidas separables
  examinadas, valores finitos y denominadores válidos.
- Las rutas recurrentes, las pérdidas personalizadas y las probabilísticas
  conjuntas requieren un análisis específico; la agregación descrita no
  certifica la reducción distribuida de Lightning.
- No se extrapolan los hallazgos a otras versiones ni a todas las integraciones
  de GIFT-Eval o Impermanent.
- Los colores y estructuras se validan como Markdown y HTML; no sustituyen una
  inspección visual de la vista previa ni modifican la configuración de VS Code.

Para concretar el diseño faltan la frecuencia, longitud y estacionalidades de
la serie; el horizonte operativo; la frecuencia de emisión y actualización; la
proporción, duración y causa de los huecos; el modelo y pérdida elegidos; y los
posibles retrasos de llegada de datos. No se fijan valores arbitrarios para esas
decisiones.

---

<a id="fuentes"></a>

## 12. Fuentes y localizadores

Se conserva la procedencia de ambos documentos y se han recontrastado los puntos
de integración con el código fijado al commit indicado. No se ha realizado una
nueva revisión bibliográfica exhaustiva. Las páginas de documentación web son
mutables; los enlaces al código y al preprint fijan una versión.

### 12.1. Implementación y documentación

1. **Nixtla/neuralforecast, versión 3.1.4**, commit
   `69e431413b22cbb1a74d35db2fa7994092ce0ea4`. Código primario inspeccionado:
   `common/_base_model.py`, `common/_scalers.py`, `losses/pytorch.py`, `core.py`,
   `tsdataset.py` y `models/nhits.py`. Los enlaces de cada afirmación fijan líneas
   relevantes. La integración añade el contraste del orden del escalado fijo,
   la creación de máscara y las rutas de CV; no una prueba empírica.
2. **Nixtla, «Time Series Scaling».** [Documentación oficial][nf-scaling-doc],
   especialmente §§3 y 4 y sus advertencias. Se conserva como referencia el
   [notebook fijado al mismo commit][nf-scaling-notebook]. Se inspeccionó el
   contenido, sin ejecutar el notebook ni sus instrucciones de instalación.

### 12.2. Fundamento metodológico

3. **Hyndman, R. J., y Athanasopoulos, G.** *Forecasting: Principles and Practice*,
   tercera edición, versión web de OTexts. Apartados pertinentes del texto:
   [§5.10, «Time series cross-validation»][fpp-cv] y [§13.9, «Dealing with
   outliers and missing values», apartado «Missing values»][fpp-missing].
4. **Hewamalage, H., Ackermann, K., y Bergmeir, C. (2023).** «Forecast evaluation
   for data scientists: common pitfalls and best practices». *Data Mining and
   Knowledge Discovery*, **37**(2), 788–832. Publicación en línea: 2 de diciembre
   de 2022; edición de 2023. DOI:
   <https://doi.org/10.1007/s10618-022-00894-5>. Artículo publicado; texto completo
   depositado en [PMC][forecast-eval], con atención a §§2, 3.3, 3.5 y 4.1.
5. **Scikit-learn, documentación estable.** «Common pitfalls and recommended
   practices», [§12.2, «Data leakage»][sklearn-leakage]. Texto sobre separación
   antes del ajuste de transformaciones; no evidencia de una versión instalada
   ni de ejecución de `TimeSeriesSplit`.
6. **Olivares, Kin G.; Luo, David; Challu, Cristian; La Vattiata, Stefania;
   Mergenthaler, Max; Dubrawski, Artur (2023).** *Hierarchically Coherent
   Multivariate Mixture Networks*. Preprint [arXiv:2305.07089v2][hint-record],
   revisado el 16 de octubre de 2023. Se comprobaron identidad y versión y los
   apartados [§3.2][hint-method] y [apéndice C.1][hint-ablation] del texto. La
   documentación de NeuralForecast lo cita con un título anterior que comienza
   por «HINT»; aquí se utiliza el de la versión consultada. No se presenta el
   preprint como artículo de revista ni como garantía general sobre faltantes.

[↑ Volver al índice](#indice)

[nf-version]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/__init__.py#L1-L4
[nf-threshold]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L353-L391
[nf-availability]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L735-L758
[nf-windows-train]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L703-L776
[nf-padder]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L326-L333
[nf-dataset-padding]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/tsdataset.py#L187-L199
[nf-sampling]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L1698-L1727
[nf-n-series]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L138-L145
[nf-windows-val]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L778-L841
[nf-normalization]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L845-L872
[nf-standard]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_scalers.py#L163-L200
[nf-robust]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_scalers.py#L203-L254
[nf-masked-stats]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_scalers.py#L12-L55
[nf-temporal-transform]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_scalers.py#L433-L489
[nf-core-scaler]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L220-L273
[nf-scaling-doc]: https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/time_series_scaling.html
[nf-scaling-notebook]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/nbs/docs/capabilities/time_series_scaling.ipynb
[nf-parse]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L905-L924
[nf-train-step]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L1728-L1772
[nf-weighted-mean]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/losses/pytorch.py#L30-L42
[nf-loss-weights]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/losses/pytorch.py#L78-L104
[nf-mae]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/losses/pytorch.py#L127-L147
[nf-mse]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/losses/pytorch.py#L170-L190
[nf-nhits]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/models/nhits.py#L398-L421
[nf-mqloss]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/losses/pytorch.py#L599-L660
[nf-validation-step]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L1818-L1884
[nf-valid-loss]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L991-L1023
[nf-validation-epoch]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L666-L678
[nf-early-stopping]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/common/_base_model.py#L296-L304
[nf-ensure-mask]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/tsdataset.py#L132-L138
[nf-check-nan]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L326-L348
[nf-prepare-fit]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L306-L324
[nf-core-fit]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L491-L501
[nf-fit-validation-size]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L476-L486
[nf-core-inverse]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L296-L304
[nf-cv-config]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L1336-L1468
[nf-cv-no-refit]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L1244-L1289
[nf-cv-refit]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L1471-L1528
[nf-cv-output]: https://github.com/Nixtla/neuralforecast/blob/69e431413b22cbb1a74d35db2fa7994092ce0ea4/neuralforecast/core.py#L1328-L1334
[hint-record]: https://arxiv.org/abs/2305.07089v2
[hint-method]: https://arxiv.org/html/2305.07089v2#S3.SS2
[hint-ablation]: https://arxiv.org/html/2305.07089v2#A3.SS1
[fpp-cv]: https://otexts.com/fpp3/toolbox.html#sec-tscv
[fpp-missing]: https://otexts.com/fpp3/practical.html#sec-missing-outliers
[forecast-eval]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/
[sklearn-leakage]: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
