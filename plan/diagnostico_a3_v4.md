# Diagnóstico de la selección de A3 en v4 y propuesta de diapositivas

**Fecha del análisis:** 10 de octubre de 2026. **Alcance:** los CSV, las 17 gráficas PNG y el notebook de `kaggle/dqn-dummy/v4`; revisión posterior de resultados existentes, sin ejecutar un nuevo entrenamiento.

## 1. Conclusión que se puede defender ante la asesora

**A3 funciona y fue explorada, pero la política seleccionada la utiliza muy poco en prueba. El indicio más sólido es que la representación del estado vuelve indistinguibles situaciones que requieren acciones diferentes según la auditoría inmediata. La inestabilidad de las decisiones y el compromiso entre AUC y costo son otros factores que se deben comprobar con experimentos controlados.**

El avance nuevo consiste en localizar y cuantificar el problema, reconstruir las observaciones que recibió el agente y definir pruebas para sus posibles causas. Los resultados experimentales siguen siendo los de v4.

La pregunta precisa es: **«¿Por qué A3 se selecciona poco cuando ofrece una buena relación entre desempeño y costo?»**. Decir que nunca se selecciona sería incorrecto.

### Qué significa cada acción en este experimento

| Acción | Implementación en v4 |
| --- | --- |
| A1 | Mantener el pipeline actual. |
| A2 | Actualizar medianas e IQR del transformador y su referencia de monitoreo; conservar el clasificador. |
| A3 | Entrenar un nuevo XGBoost con los datos de actualización del día; conservar transformador y referencia de monitoreo. |
| A4 | Actualizar transformador, entrenar un nuevo XGBoost y actualizar la referencia de monitoreo. |

A3 reemplaza el clasificador mediante un nuevo ajuste; no es un entrenamiento incremental del mismo modelo. El target es `y_dummy`, construido para la prueba de concepto. Los resultados no representan todavía una validación con una etiqueta clínica real.

### Cómo leer las comparaciones

La decisión del día `d` actualiza con datos disponibles de ese día; su reward usa el AUC reservado del día `d+1`, menos el costo de la acción. Hay **16 días de métricas y 15 decisiones por escenario**.

`action_audit.csv` prueba las cuatro acciones desde el mismo pipeline que visitó DQN y compara su utilidad del día siguiente. Su **oracle inmediato** sirve para localizar oportunidades; **no demuestra cuál es la mejor política para toda la secuencia**. Además, utiliza resultados futuros reservados para la auditoría, que el agente no conoce al decidir. No se debe sumar sus ventajas y presentar el resultado como el retorno alcanzable de una política oracle: cambiar una acción cambia los estados posteriores.

## 2. Hallazgos cuantitativos

### 2.1. La brecha está en los escenarios con cambio de concepto

| Escenario | Decisiones DQN | Veces que DQN eligió A3 | Veces que A3 fue la mejor opción inmediata | Coincidencias con el oracle | Regret inmediato medio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Estable | 15 | 0 | 0 | 15/15 | 0,000000 |
| Covariate | 15 | 0 | 0 | 15/15 | 0,000000 |
| Concept | 15 | 1 | 6 | 10/15 | 0,010751 |
| Combinado | 15 | 0 | 4 | 9/15 | 0,007673 |
| **Total** | **60** | **1** | **10** | **49/60 (81,7 %)** | **0,004606** |

El DQN elige A1/A2/A3/A4 **38/18/1/3** veces. El oracle inmediato señala **35/14/10/1**. A3 se utiliza en el **1,7 %** de las decisiones y se aprovecha **1 de sus 10 oportunidades inmediatas (10 %)**. Las nueve oportunidades omitidas representan **96,7 % del regret inmediato total**: 0,267228 de 0,276360 unidades de utilidad acumuladas en la auditoría.

La coincidencia global del 81,7 % oculta esta dificultad específica. El 87,5 % reportado como `oracle_near_rate` es otra métrica: **49 de 56 decisiones**, excluyendo fronteras de inicio de drift, tienen regret ≤ 0,01. No es la tasa de selección correcta de A3.

![Selección DQN y mejor utilidad inmediata](C:/Users/jeffr/GitHub/tesis-darl/outputs/figures/diagnostico_a3_v4/01_brecha_a3.png)

### 2.2. Dónde estaban las oportunidades de A3

La utilidad ya descuenta el costo. Un regret de 0,065844 equivale a una diferencia de 0,065844 en **AUC menos costo**, no a 6,58 puntos de AUC puro.

| Escenario | Día de decisión | Elegida | AUC disponible antes de actuar | Utilidad de A3 | Utilidad elegida | Regret inmediato | ID de observación reconstruida |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Concept | 6 | A1 | 0,907010 | 0,878332 | 0,861576 | 0,016756 | 3 |
| Concept | 7 | A1 | 0,861577 | 0,875907 | 0,810063 | **0,065844** | 3 |
| Concept | 8 | **A3** | 0,810063 | 0,879214 | 0,879214 | 0,000000 | 14 |
| Concept | 9 | A1 | 0,921053 | 0,885453 | 0,874488 | 0,010965 | 15 |
| Concept | 14 | A1 | 0,897403 | 0,882089 | 0,856379 | 0,025710 | 3 |
| Concept | 15 | A1 | 0,856380 | 0,894293 | 0,852306 | 0,041987 | 3 |
| Combinado | 9 | A2 | 0,887219 | 0,883169 | 0,837144 | **0,046025** | 8 |
| Combinado | 10 | A4 | 0,845701 | 0,878250 | 0,870282 | 0,007968 | 17 |
| Combinado | 12 | A2 | 0,889590 | 0,880444 | 0,837845 | **0,042599** | 19 |
| Combinado | 13 | A4 | 0,847056 | 0,886176 | 0,876802 | 0,009374 | 20 |

En concepto, A3 se activa cuando el AUC ya cayó a 0,810. En combinado, las dos caídas por debajo de 0,85 llevan a A4. Esto sugiere una respuesta tardía y dependiente de la señal de covariate drift, pero dos episodios no prueban una regla general de la política.

### 2.3. La misma observación representa situaciones diferentes

Se reconstruyó el **vector completo de 30 valores**, mediante la función original `encode_observation` y la misma ventana de tres días de `BedEnv`. No se comparó únicamente el bin del día actual.

Cada bloque contiene KS y PSI en tres categorías, AUC en tres categorías —con cortes en 0,65 y 0,85— y un indicador de AUC válido. El agente recibe los últimos tres bloques. **Un AUC de 0,94 y otro de 0,86 pertenecen a la misma categoría alta.** La magnitud de la caída se pierde.

Las 60 decisiones producen **21 observaciones distintas**. En **cuatro grupos** de observaciones idénticas, la acción con mejor utilidad inmediata cambia:

| ID local | Casos con el mismo vector completo | Mejores acciones inmediatas | Acción que siempre eligió DQN | Interpretación |
| --- | ---: | --- | --- | --- |
| 3 | 27 | A1 en 23; A3 en 4 | A1 | Los días de concepto 6, 7, 14 y 15 se confunden con numerosos días donde conviene mantener. |
| 7 | 3 | A2 en 2; A1 en 1 | A2 | También hay ambigüedad fuera de A3. |
| 8 | 2 | A2 en covariate día 9; A3 en combinado día 9 | A2 | Se pierde la distinción entre corregir features y corregir el clasificador. |
| 19 | 2 | A3 en combinado día 12; A1 en combinado día 15 | A2 | La ambigüedad no explica por sí sola la preferencia por A2: también debe revisarse lo aprendido. |

Los IDs se asignaron durante este análisis; no son estados numerados por el experimento original. El grupo 3 acumula **54,4 % del regret total**. **Seis de las nueve oportunidades omitidas de A3** pertenecen a grupos con acciones oracle distintas.

En el grupo 3, los Q son exactamente iguales en sus 27 casos: A1 = **1,502317**, A2 = **1,458126**, A3 = **1,421757**, A4 = **1,454640**. La elección A1 concuerda con `argmax(Q)`. La red determinista no puede elegir acciones diferentes cuando recibe exactamente el mismo vector.

Esto demuestra una limitación de la representación frente a la auditoría inmediata. No demuestra que cada elección diferente del oracle sea un error del objetivo secuencial, ni que agregar una variable específica vaya a resolverlo sin probarla.

![AUC y decisiones en concepto](C:/Users/jeffr/GitHub/tesis-darl/outputs/figures/diagnostico_a3_v4/02_estado_concepto.png)

### 2.4. A3 sí funciona y sí recibió exploración

- El entrenamiento registró **1 665 ejecuciones de A3 sobre 15 000 decisiones (11,1 %)**. En los últimos 200 episodios fueron 197/3 000 (6,6 %).
- La ejecución de prueba de concepto del día 8 pasa de AUC **0,810063** a **0,921053** al día siguiente después de A3. Es una recuperación observada en esa trayectoria; la auditoría proporciona la comparación inmediata desde el mismo pipeline.
- La calibración del drift de concepto seleccionó tasa 0,15 y registró `repair_gain` **0,298069** para la reparación con A3. Este control pertenece al conjunto de entrenamiento y mide ganancia de AUC, no utilidad descontada.
- No hay acciones solicitadas sustituidas por fallback ni rewards inválidos en las 420 decisiones de las siete políticas evaluadas. Las 60 elecciones DQN coinciden con el máximo Q guardado.

Estos controles debilitan las explicaciones «A3 está deshabilitada», «su índice es incorrecto», «nunca se exploró» o «el reentrenamiento no funciona». Los conteos agregados no permiten saber si hubo suficientes ejemplos de A3 **en los estados críticos**.

### 2.5. Retorno estable no significa política estable

`run_summary.json` registra `needs_improvement` y `validation_converged = false`. Al cierre, la variación de los últimos cinco retornos es **1,47 %**, pero el acuerdo entre las últimas dos políticas es **85 %**, por debajo del criterio del **95 %**.

El modelo utilizado en prueba es el del **episodio 940**, seleccionado por el mayor retorno medio suavizado sobre tres validaciones. En su validación elige A3 **2/60** veces y tiene **75 %** de acuerdo con el checkpoint previo. El 85 % del resumen pertenece al cierre del entrenamiento en el episodio 1000, no al modelo seleccionado.

La cantidad de A3 varía: validación 940 → 2; 960 → 11; 990 → 10; 1000 → 3. Más A3 tampoco garantiza mayor retorno: el checkpoint 960 tiene retorno 13,540448, inferior al 13,757568 del 940. No se deben comparar directamente las dos A3 de validación con la única A3 de prueba como si fueran los mismos datos.

![Estabilidad del retorno y de las decisiones](C:/Users/jeffr/GitHub/tesis-darl/outputs/figures/diagnostico_a3_v4/03_estabilidad_validacion.png)

### 2.6. El costo influye, pero no explica toda la brecha

La utilidad es `AUC del día siguiente − 0,05 × tiempo de la acción / tiempo de referencia de A4`. El tiempo de referencia queda congelado tras su calibración. Las penalizaciones de referencia son A2 = **0,009199**, A3 = **0,040269** y A4 = **0,050000**: A3 cuesta aproximadamente **80,5 % de A4**, con ahorro de 0,009731 unidades de reward frente a A4. Los tiempos reales por acción fluctúan.

A4 puede competir con A3 porque, por poco costo adicional de referencia, actualiza también el transformador. Sin embargo, **A3 supera a A4 después de descontar costo en las diez oportunidades auditadas**; el margen va aproximadamente de 0,0044 a 0,0129. Por tanto, «A3 es demasiado cara» no basta como diagnóstico.

| Escenario y política | AUC media: 16 días | Retorno: 15 decisiones | Costo total descontado |
| --- | ---: | ---: | ---: |
| Concept — DQN | 0,900752 | 13,382348 | 0,094636 |
| Concept — heurística | 0,901680 | 13,405353 | 0,086487 |
| Concept — siempre A3 | 0,927086 | 13,270767 | 0,627564 |
| Combinado — DQN | 0,910343 | 13,466884 | 0,163555 |
| Combinado — siempre A3 | 0,885794 | 12,555756 | 0,681908 |
| Combinado — siempre A4 | 0,925145 | 13,090243 | 0,777039 |

En concepto, siempre A3 obtiene mayor AUC media y menor retorno que DQN. El objetivo premia mantenimiento selectivo; no se busca aumentar A3 indiscriminadamente. Las diferencias pequeñas entre políticas no establecen significancia estadística: esta evidencia proviene de una semilla.

## 3. Posibles problemas y pruebas para distinguirlos

La prioridad indica el orden recomendado de investigación, no una probabilidad causal estimada. Las pruebas de esta tabla **están propuestas, todavía no ejecutadas**.

| Prioridad | Posible problema | Evidencia disponible y límite | Cómo comprobarlo |
| --- | --- | --- | --- |
| 1 | **Discretización demasiado gruesa del AUC.** La red pierde el deterioro que ocurre dentro de una categoría. | Cuatro oportunidades de A3 en concepto comparten el mismo vector que 23 situaciones favorables a A1. Es una limitación comprobada frente al oracle inmediato. | Comparar la observación original con otra que añada AUC continuo y diferencias con uno y dos días anteriores. Mantener algoritmo, costo, calendario y presupuesto de entrenamiento; seleccionar por validación y medir en prueba retorno, regret inmediato y oportunidades de A3 aprovechadas. |
| 2 | **Falta información sobre el clasificador y la historia de mantenimiento.** KS/PSI de features no distinguen por sí solos un cambio de relación con la etiqueta. | En las distribuciones de concepto los KS día 1–16 son bajos, aproximadamente 0,03–0,06, mientras el AUC del modelo inicial cae. La observación no incluye última acción, edad del modelo ni saturación; el combinado tiene observaciones compartidas con covariate. | Añadir gradualmente días desde A2/A3/A4, última acción y saturación. Probar una señal supervisada causal de deterioro si las etiquetas están disponibles con el retraso previsto. No introducir el nombre del escenario, severidad oculta ni métricas del día siguiente. |
| 3 | **Estimación Q y política insuficientemente estables.** El checkpoint se elige por retorno medio, que puede ocultar errores de decisiones. | No se cumple el criterio de estabilidad. A3 cambia entre checkpoints; en el estado 19 DQN escoge A2 aunque el oracle inmediato favorece A3 o A1. Esto no permite separar todavía error de aprendizaje y efecto a largo plazo. | Guardar por checkpoint las oportunidades de A3, regret, márgenes Q y estabilidad en validación. Comparar ajustes de aprendizaje/actualización target de uno en uno; usar estabilidad en varios checkpoints como criterio adicional predefinido, sin seleccionar por prueba. |
| 4 | **Cobertura insuficiente de A3 en los estados críticos.** Explorar A3 globalmente no garantiza explorarla donde conviene. | A3 tiene 11,1 % de las decisiones de entrenamiento, pero falta el registro de transiciones por estado/acción y de muestras usadas en cada minibatch. No se puede afirmar que el replay esté desbalanceado en esos estados. | Registrar observación, acción, escenario solo para auditoría, reward, terminal y error TD. Medir cobertura de estados con deterioro y comparar, si corresponde, exploración más persistente o muestreo equilibrado por tipo de transición. El buffer de 50 000 no se llenó con las 15 000 transiciones; no hay evidencia de pérdida por expulsión FIFO. |
| 5 | **Penalización de costo y costo incremental pequeño de A4.** Puede favorecer esperar o actualizar todo. | A3 tiene penalización de referencia cercana a A4; aun así, gana en las diez oportunidades después del descuento. El CSV de auditoría no guarda AUC y tiempo separados de cada alternativa. | Guardar esos componentes para las cuatro acciones y repetir mediciones de tiempo. Como diagnóstico, recalcular la auditoría a varios pesos de costo; para comparar políticas aprendidas, reentrenar cada configuración bajo el mismo protocolo. No recalcular contrafactuales usando el tiempo de la acción elegida. |
| 6 | **Se evalúa una mejora de un día, pero se aprende retorno secuencial.** Parte de la discrepancia puede ser deliberada. | El oracle usa utilidad del día siguiente; Double DQN usa descuento 0,99 y consecuencias posteriores. Sus Q no tienen la misma escala ni significado que `utility_A3`. | Desde snapshots guardados, ejecutar cada acción candidata y luego una misma política de continuación durante los días restantes. Comparar retorno a varios días y costo total; esto distingue una mala decisión local de una decisión útil a largo plazo. |

**Orden práctico:** instrumentar los estados críticos y el costo contrafactual → comparar AUC continuo/tendencia → añadir historia del pipeline → revisar estabilidad y cobertura → sensibilidad al costo y auditoría secuencial. Cambiar una cosa por experimento permite atribuir los resultados. Usar semilla 42 para el diagnóstico inicial y después el protocolo de múltiples semillas acordado para la validación final.

No se encontró una inconsistencia aritmética del reward: `auc_next − cost` reproduce los 420 rewards con error absoluto máximo de **3,33 × 10⁻¹⁶**. El centrado interno de reward por −0,8 no cambia el orden de las políticas verdaderas en episodios de igual horizonte y no depende de la acción; por sí solo no explica la exclusión de A3. Con estos archivos tampoco puede concluirse que Double DQN sea un algoritmo inadecuado: primero hay que separar información insuficiente, entrenamiento y criterio de evaluación.

## 4. Lectura de las gráficas entregadas

| Gráficas de v4 revisadas | Lectura y uso en el diagnóstico |
| --- | --- |
| `01_ocupacion_recambio` | Replay de 200 camas ocupadas y recambio diario. Da contexto de la simulación; no identifica una causa de la selección de A3. |
| `02_calendario_cambios` | Las severidades de prueba aumentan gradualmente; concepto empieza en día 5 en su escenario y día 7 en combinado. El calendario es información oculta al agente. |
| `03_monitoreo_ks_psi_auc` | En combinado hay alertas de features, caídas de AUC y cambios de referencia tras acciones. No cubre por sí sola el diagnóstico de concepto puro. |
| `04_entrenamiento_retorno_epsilon`, `04a_retorno_entrenamiento`, `04b_acciones_validacion` | Mejorar el retorno medio no prueba estabilización de las decisiones. La cantidad de A3 de validación fluctúa. La figura nueva 03 separa retorno y acuerdo. |
| `05_auc_y_reward_politicas`, `05a_auc_politicas`, `05b_reward_acumulado_politicas` | En combinado DQN mantiene un buen compromiso de AUC y costo frente a las acciones fijas. Las curvas acumuladas contienen una base de AUC común grande que puede hacer poco visibles las diferencias; usar cifras y auditoría de decisiones. |
| `06_acciones_dqn` | Localiza la única A3 en concepto día 8 y las A4 en concepto día 10 y combinado días 10/13. Contrastar con `actions.csv` y la tabla de oportunidades. |
| `07_utilidad_acciones_regret`, `07a_utilidad_acciones`, `07b_regret` | A3 supera a la elegida en varias decisiones de combinado. Usar el CSV para las fechas y magnitudes exactas: la figura 07 combinada ubica visualmente el pico en un día diferente de 07b/CSV y no se genera en `make_figures` del notebook entregado. Puede ser un artefacto conservado de otra generación; no se usa como fuente numérica. |
| `08_saturacion_y_recuperacion` | La recuperación de AUC sigue a las actualizaciones, pero comparar días sucesivos no aísla una acción del cambio de pacientes y de drift. Saturación se registra, aunque no entra en la observación de la red. |
| `09_auc_dqn_vs_a1` | DQN supera mantener sin actuar en los escenarios de drift. Eso valida su utilidad general en esta ejecución, pero no resuelve la selección escasa de A3. |
| `10_distribucion_vitales_covariate`, `11_concept_drift_relacion` | Distinguen desplazamientos grandes en features de cambios en su relación con la etiqueta. Sus comparaciones día 1–16 no usan necesariamente la referencia dinámica del monitor; no igualar esos KS a los de `actions.csv`. |

La figura 07 combinada requiere regeneración desde el CSV antes de reutilizarse en la presentación. La posible mezcla de artefactos es un problema de trazabilidad de las figuras; no demuestra un fallo del agente. Los SVG enumerados en el resumen no están presentes en esta copia de outputs.

## 5. Propuesta de PPT: cuatro diapositivas principales y un respaldo

### Slide 1 — «A3 funciona, pero se aprovecha en 1 de 10 oportunidades»

**Pregunta:** ¿Qué encontramos al revisar las decisiones de v4?

**Diseño:** figura nueva 01 en el centro; a la derecha tres cifras grandes: **1/60 elegidas**, **10 oportunidades**, **9 omitidas**. Pie: «Auditoría inmediata sobre los estados visitados por DQN; seed 42».

**Texto para la diapositiva:**

- La brecha aparece en concepto y combinado; estable y covariate coinciden con el oracle inmediato.
- Las omisiones de A3 concentran el **96,7 % del regret inmediato**.
- A3 se ejecutó 1 665 veces en entrenamiento: no está deshabilitada.

**Guion:** «Después de presentar v4, auditamos cuándo A3 era conveniente. Encontramos diez oportunidades según AUC del día siguiente menos costo, pero solo una fue aprovechada. El avance es el diagnóstico de estas decisiones».

### Slide 2 — «El estado oculta señales útiles para elegir A3»

**Pregunta:** ¿Qué recibió el agente cuando dejó pasar las oportunidades?

**Diseño:** figura nueva 02. Añadir un recuadro corto: «AUC 0,94 y 0,86 → misma categoría». Resaltar concepto días 6, 7, 14 y 15.

**Texto para la diapositiva:**

- La observación guarda categorías de KS, PSI y AUC durante tres días.
- Un mismo vector aparece 27 veces: oracle A1 en 23 y A3 en 4; DQN responde A1 siempre.
- Hipótesis prioritaria: faltan magnitud/tendencia del AUC e historia del modelo.

**Guion:** «No basta con que exista concept drift. El agente debe poder reconocerlo con su observación. Reconstruimos el vector completo y encontramos casos diferentes que llegan a la red exactamente iguales. Proponemos comprobar si señales continuas permiten distinguirlos».

### Slide 3 — «La decisión depende también del aprendizaje y del costo»

**Pregunta:** ¿Qué otras explicaciones debemos distinguir?

**Diseño:** figura nueva 03 en dos tercios de la slide; recuadro lateral con penalizaciones de referencia A3 = 0,0403 y A4 = 0,0500. Mantener la nota que distingue episodio seleccionado 940 y cierre 1000.

**Texto para la diapositiva:**

- El retorno varía poco al cierre, pero el acuerdo de decisiones es **85 %**, inferior al criterio del 95 %.
- El checkpoint seleccionado tiene **75 %** de acuerdo con el anterior; A3 cambia entre checkpoints.
- El costo influye, pero A3 ya gana después del descuento en las diez oportunidades auditadas.

**Guion:** «Todavía no atribuimos toda la brecha a una sola causa. La política no cumplió su criterio de estabilidad. Además, premiamos desempeño con costo, de modo que no queremos usar A3 siempre: queremos reconocer cuándo compensa».

### Slide 4 — «Pruebas propuestas para separar las causas»

**Pregunta:** ¿Cómo transformamos el diagnóstico en el siguiente avance?

**Diseño:** tabla de tres filas, con el rótulo **«Próximos experimentos: aún no ejecutados»**.

| Prueba | Comparación controlada | Evidencia esperada |
| --- | --- | --- |
| Información del estado | Original vs AUC continuo y tendencia; después añadir historia de mantenimiento. | Menos observaciones ambiguas y mejor identificación de oportunidades, sin aumentar innecesariamente el costo. |
| Aprendizaje | Registrar cobertura estado/acción, márgenes Q y decisiones por checkpoint. | Distinguir falta de ejemplos de falta de estabilidad; observar selección consistente en validación. |
| Objetivo | Componentes AUC/tiempo por alternativa, sensibilidad al costo y continuación a varios días. | Distinguir penalización excesiva de decisiones con beneficio posterior. |

**Guion:** «Las pruebas mantendrán comparables datos, calendario y presupuesto. La mejora se medirá por retorno y costo, estabilidad y aprovechamiento de oportunidades; un mayor conteo de A3 por sí solo no será éxito».

### Slide de respaldo — «Tres decisiones que ejemplifican la brecha»

| Caso | Elegida | Utilidad elegida | Utilidad A3 | Qué ilustra |
| --- | --- | ---: | ---: | --- |
| Concept día 7 | A1 | 0,810063 | 0,875907 | Deterioro dentro del mismo estado discretizado. |
| Combinado día 9 | A2 | 0,837144 | 0,883169 | Estado compartido con covariate día 9, donde A2 sí conviene. |
| Combinado día 13 | A4 | 0,876802 | 0,886176 | A3 ofrece mejor utilidad inmediata, aunque DQN prefiere actualizar todo. |

Nota al pie: «Comparación de un paso; no demuestra el óptimo secuencial». Tener disponible la tabla completa de diez oportunidades para preguntas.

**Frase breve para comunicar el avance:** «Realizamos una auditoría posterior de v4 y reconstruimos la información que recibió el DQN. Identificamos oportunidades de A3 omitidas y estados que confunden situaciones distintas, y definimos pruebas controladas para separar representación, aprendizaje y costo».

## 6. Evidencia, verificaciones y archivos

| Fuente | Cobertura revisada | Uso |
| --- | --- | --- |
| [actions.csv](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/outputs/actions.csv) | 420 decisiones; 60 DQN | Acciones ejecutadas, métricas previas/siguientes, costo, reward, Q y fallbacks. |
| [action_audit.csv](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/outputs/action_audit.csv) | 60 decisiones DQN | Utilidades contrafactuales inmediatas y regret. |
| [daily_metrics.csv](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/outputs/daily_metrics.csv) | 448 filas = 4 × 7 × 16 | Trayectorias diarias, AUC y saturación. |
| [policy_comparison.csv](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/outputs/policy_comparison.csv) | 28 filas = 4 × 7 | Retorno, AUC media, costo y conteos por política. |
| [training_history.csv](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/outputs/training_history.csv) | 1 000 episodios; 100 validaciones | Exploración, estabilidad y checkpoint seleccionado. |
| [calibration.csv](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/outputs/calibration.csv) | 6 candidatos | Funcionalidad de reparación y tasas seleccionadas. |
| [stage_calendar.csv](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/outputs/stage_calendar.csv) | 64 filas = 4 × 16 | Inicio y evolución del drift de prueba. |
| [drift_dia1_vs_dia16.csv](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/outputs/drift_dia1_vs_dia16.csv) | 2 filas, valores redondeados | Comparación suplementaria de distribuciones y relación con la etiqueta. |
| [run_summary.json](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/outputs/run_summary.json) | Ejecución v4, seed 42 | Configuración, costo de referencia, gates y episodio 940. |
| [main.ipynb](C:/Users/jeffr/GitHub/tesis-darl/kaggle/dqn-dummy/v4/input/main.ipynb) | `Pipeline.act`, `utility`, `encode_observation`, `BedEnv`, `DQNAgent`, `roll`, `train_agent`, `make_figures` | Reconstrucción de arquitectura y límites de interpretación. |

Se conciliaron rewards con AUC menos costo, métricas previas/siguientes con los días del CSV, conteos y retornos de todas las políticas, utilidades oracle con sus máximos y elecciones DQN con `argmax(Q)`. Los vectores de observación iguales tienen exactamente los mismos Q registrados. Los archivos fuente conservaron sus hashes SHA-256 durante el análisis. Las figuras nuevas se generaron desde los CSV y se inspeccionaron visualmente.

La revisión valida estas relaciones y este diagnóstico de una ejecución; no equivale a reentrenar v4 ni a validar causalmente las hipótesis.

**Material generado:**

- [Decisiones con observación reconstruida](C:/Users/jeffr/GitHub/tesis-darl/outputs/metrics/diagnostico_a3_v4/decisiones_reconstruidas.csv).
- [Agrupación de estados](C:/Users/jeffr/GitHub/tesis-darl/outputs/metrics/diagnostico_a3_v4/estados_observados.csv) y [oportunidades de A3](C:/Users/jeffr/GitHub/tesis-darl/outputs/metrics/diagnostico_a3_v4/oportunidades_a3.csv).
- [Resumen numérico y hashes de fuentes](C:/Users/jeffr/GitHub/tesis-darl/outputs/metrics/diagnostico_a3_v4/resumen.json).
- [Figura 01: brecha de A3](C:/Users/jeffr/GitHub/tesis-darl/outputs/figures/diagnostico_a3_v4/01_brecha_a3.png), [figura 02: observaciones de concepto](C:/Users/jeffr/GitHub/tesis-darl/outputs/figures/diagnostico_a3_v4/02_estado_concepto.png) y [figura 03: estabilidad](C:/Users/jeffr/GitHub/tesis-darl/outputs/figures/diagnostico_a3_v4/03_estabilidad_validacion.png), en formato 16:9.
- [Script de análisis temporal](C:/Users/jeffr/GitHub/tesis-darl/.tmp/diagnostico_a3_v4/analizar.py). Ejecutar desde la raíz con `.venv/Scripts/python.exe .tmp/diagnostico_a3_v4/analizar.py`; no ejecuta el entorno ni entrena la red.
