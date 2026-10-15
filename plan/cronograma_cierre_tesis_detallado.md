# Plan de trabajo y cronograma detallado para cierre de tesis

## Cronograma y responsabilidades

Cada tarea tiene un ID de **T01 a T27**, enlazado a su explicación debajo de la tabla. Cada semana tiene un resultado comprobable. La redacción empieza en la primera semana y las réplicas de las configuraciones cerradas empiezan en la tercera, para no concentrar todo al final.

| Semana                 | Task/Experimento                                                                                                                                                                                                                                                           | Resultados Esperados                                                                                                                                                                           | Responsable      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **1 · 8–14 oct.** | **[T01](#t01)** — Estudiar v4 siguiendo la guía: carga, particiones, etiqueta dummy, replay, pipeline, discretización, Double DQN y gráficas. Seguir una decisión desde la observación hasta el reward y el aprendizaje. | Guía del recorrido del código y un ejemplo de decisión explicado paso a paso. Identificar qué datos y señales recibe el agente y qué cambia al actuar. | Jeffrey |
| **1 · 8–14 oct.** | **[T02](#t02)** — Reproducir una ejecución pequeña en una carpeta nueva y registrar configuración, versiones y diferencias respecto a la corrida presentada. | Ejecución pequeña documentada, archivos de salida identificados y dificultades registradas; resultados v4 conservados. | Jeffrey |
| **1 · 8–14 oct.** | **[T03](#t03)** — Investigar los nueve casos donde el oracle inmediato prefiere A3 y el agente elige otra acción: observación, valores Q, fallbacks, exploración y cobertura durante entrenamiento. | Tabla de hipótesis sobre la decisión del agente y pruebas concretas para comprobarlas en la semana 2. | Jeffrey |
| **1 · 8–14 oct.** | **[T04](#t04)** — Auditar CSV y resumen: conteos, escenarios, días válidos, costos y denominadores. Comparar utilidad A3/A4/A1 en los casos omitidos y revisar correspondencia entre figuras y tablas, incluido el día de acción y reward. | Tabla que muestre cuándo A3 ofrece mayor utilidad inmediata, qué eligió DQN y cuánto se pierde; observaciones sobre consistencia de las gráficas y alcance dummy. | Dayane |
| **1 · 8–14 oct.** | **[T05](#t05)** — Elaborar fichas de PhysioNet, Diabetes e IEEE y una matriz bibliográfica de Double DQN, XGBoost, KS/PSI y drift. | Fichas con objetivo, unidad de registro, identidad, tiempo, faltantes, prevalencia y disponibilidad de etiquetas; referencias que sustenten cada método y sus límites. | Dayane |
| **1 · 8–14 oct.** | **[T06](#t06)** — Revisar juntos el recorrido del código y los casos de A3; presentar diagnóstico inicial a la asesora y confirmar alcance de datasets. | Jeffrey explica cómo decide y aprende el agente; Dayane explica qué muestran los datos y resultados. Diagnóstico inicial y alcance revisados en asesoría. | Jeffrey y Dayane |
| **2 · 15–21 oct.** | **[T07](#t07)** — Comparar observación actual vs. AUC continua+tendencia, peso de costo 0 / 0,025 / 0,05 / 0,10 y tres velocidades de drift sobre train/val. Cambiar un factor por vez y luego combinar los cambios prometedores; mantener fijo el reward de referencia en cada comparación. | Observación seleccionada con validación, curva desempeño–costo y evidencia del efecto de severidad sobre A3/A4. | Jeffrey |
| **2 · 15–21 oct.** | **[T08](#t08)** — Elegir una versión de referencia y trasladar solo lo necesario a los módulos existentes; comprobar equivalencia con el notebook bajo entradas y costos controlados. | Un pequeño replay produce las mismas decisiones y resultados en notebook y módulos. Configuración de referencia documentada para migrar a otros datasets. | Jeffrey |
| **2 · 15–21 oct.** | **[T09](#t09)** — Ejecutar diagnósticos de los inyectores existentes con configuración preparada por Jeffrey: estabilidad de X, prevalencia, cambios de etiquetas y recuperación A1–A4 por severidad. Comparar `logit_reweight` y su variante con recentrado cuando proceda. | Tablas y gráficas que identifiquen qué cambia con cada mecanismo y cuándo conviene A3 frente a A4; límites de cada inyector documentados. | Dayane |
| **2 · 15–21 oct.** | **[T10](#t10)** — Preparar contratos de columnas/categorías para Diabetes e IEEE y redactar datos, drift y justificación de métodos. | Esquema de entrada definido por dataset y borrador de los apartados de datos, inyección y selección de métodos. | Dayane |
| **2 · 15–21 oct.** | **[T11](#t11)** — Integrar el análisis de acciones y las pruebas del agente para cerrar el diagnóstico de A3. | Informe que respalde o descarte cada hipótesis. Las causas abiertas quedan documentadas y limitan los cambios posteriores. | Jeffrey y Dayane |
| **3 · 22–28 oct.** | **[T12](#t12)** — Adaptar Diabetes a ventanas de encuentros, separar pacientes cuando existan identificadores e implementar el pipeline numérico/categórico y A1–A4. Verificar que A2 preserve columnas y semántica de Stage 2; integrar logit con severidad temporal. | Experimento ejecutable con objetivo y tiempo declarados, separación update/eval y compatibilidad de acciones comprobada. | Jeffrey |
| **3 · 22–28 oct.** | **[T13](#t13)** — Ejecutar piloto Double DQN frente a A1–A4, random y heurística en estable, covariate, concept y combined; iniciar réplicas de la configuración validada. | Resultados iniciales por escenario y comienzo de la campaña final de Diabetes. | Jeffrey |
| **3 · 22–28 oct.** | **[T14](#t14)** — Preparar ficha y particiones de Diabetes; confirmar que `readmitted` significa cualquier reingreso frente a ninguno en el notebook actual. Auditar imputación/codificación, estabilidad de X bajo concept y plausibilidad del mecanismo. | Datos y particiones documentados, transformaciones ajustadas solo con datos permitidos y una definición inequívoca del objetivo. Si no hay fechas verificables, el tiempo se declara simulado. | Dayane |
| **3 · 22–28 oct.** | **[T15](#t15)** — Interpretar acciones por severidad, redactar metodología y resultados iniciales y preparar las ventanas IEEE para la semana siguiente. | Explicación de cuándo conviene A3 frente a A4, borrador de resultados Diabetes y datos IEEE preparados. | Dayane |
| **3 · 22–28 oct.** | **[T16](#t16)** — Revisar con la asesora el primer resultado fuera del dummy y la explicación de A3. | Evidencia Diabetes y diagnóstico A3 revisados; observaciones incorporadas a los siguientes experimentos. | Jeffrey y Dayane |
| **4 · 29 oct.–4 nov.** | **[T17](#t17)** — Adaptar IEEE reutilizando agente/evaluador: ordenar por `TransactionDT`, reservar bloques train/val/test y fijar disponibilidad de etiquetas. Ejecutar primero drift sintético controlado y después ventanas observadas sin inyección. | Dos evaluaciones IEEE separadas, con las mismas políticas comparadas y costo de A2/A3/A4 registrado; réplicas de configuraciones cerradas iniciadas. | Jeffrey |
| **4 · 29 oct.–4 nov.** | **[T18](#t18)** — Validar unión por `TransactionID`, faltantes, categorías nuevas y positivos por ventana. Usar `train_transaction` y, si corresponde, `train_identity`, sin depender de etiquetas del test público. | Datos IEEE auditados y ventanas evaluables; reglas de actualización y evaluación documentadas. | Dayane |
| **4 · 29 oct.–4 nov.** | **[T19](#t19)** — Analizar cambios en X, prevalencia y desempeño; revisar curvas contra sus tablas y redactar resultados IEEE y límites de atribución causal. | Informe que distinga perturbación controlada y secuencia observada, sin atribuir automáticamente un cambio de tasa de fraude a concept drift. | Dayane |
| **4 · 29 oct.–4 nov.** | **[T20](#t20)** — Cerrar configuración, presupuesto de cómputo y matriz final de experimentos el 4 de noviembre. | Configuraciones congeladas y campaña final definida para la semana 5. | Jeffrey y Dayane |
| **5 · 5–11 nov.** | **[T21](#t21)** — Completar réplicas autorizadas de configuraciones congeladas; registrar semillas, particiones, checkpoints, costo, casos no evaluables y fallos. Consolidar baselines/ablaciones y revisar estabilidad, regret inmediato y A3/A4. | Resultados crudos en `outputs/`, matriz final trazable y procedimiento de reproducción. Se corrigen solo fallos que invaliden evidencia; test no se usa para nuevas búsquedas. | Jeffrey |
| **5 · 5–11 nov.** | **[T22](#t22)** — Consolidar métricas por corrida y preparar medias, dispersión e IC del 95% con unidad de remuestreo explícita; preparar tablas y figuras. | Comparaciones estadísticas con tamaño efectivo de muestra y límites declarados, sin tratar horas o pacientes correlacionados como réplicas independientes. | Dayane |
| **5 · 5–11 nov.** | **[T23](#t23)** — Completar introducción, antecedentes, metodología, resultados, discusión, conclusiones y referencias; incluir resultados favorables y desfavorables, amenazas a validez y limitaciones. | Borrador completo y argumentos sustentados por los experimentos, sin exigir que DARL gane en todos los escenarios. | Dayane |
| **5 · 5–11 nov.** | **[T24](#t24)** — Revisar fidelidad técnica del manuscrito y enviarlo a la asesora el 11 de noviembre. | Borrador completo revisado por ambos y entregado para recibir correcciones. | Jeffrey y Dayane |
| **6 · 12–18 nov.** | **[T25](#t25)** — Resolver observaciones técnicas, comprobar reproducción de una corrida representativa y coherencia código–metodología; preparar demo y explicaciones de datos, estado, reward, Double DQN y A3. | Correcciones técnicas cerradas, demo preparada y fórmulas/conclusiones consistentes con la versión evaluada. | Jeffrey |
| **6 · 12–18 nov.** | **[T26](#t26)** — Integrar correcciones de redacción, revisar citas, figuras, tablas, formato y anexos; compilar y revisar visualmente el PDF. Preparar diapositivas y preguntas de dominio/evaluación. | Manuscrito y presentación revisados, con resultados y limitaciones comunicados claramente. | Dayane |
| **6 · 12–18 nov.** | **[T27](#t27)** — Cerrar versión final y anexos el 17 de noviembre; reservar el 18 para verificación y dos ensayos conjuntos. Entrega objetivo: 19 de noviembre. | Tesis y anexos listos; cada integrante explica su frente y ambos sustentan contribución, límites y hallazgo sobre A3. | Jeffrey y Dayane |

## Desarrollo de las tareas

Los siguientes apartados describen trabajo pendiente, no experimentos ya ejecutados. Los nombres de entregables son sugeridos: la documentación puede guardarse en `plan/`, los resultados generados en `outputs/` y los scripts temporales en `.tmp/`. Conservar v4 como referencia y registrar cada configuración nueva. La semilla de depuración es 42; las campañas con otras semillas requieren resolver la excepción a la regla vigente del repositorio antes de ejecutarlas.

Jeffrey estudia e implementa cómo el entorno y el agente toman decisiones. Dayane prepara los datos y analiza en qué condiciones las acciones ofrecen mejores resultados. Las tareas conjuntas integran ambas evidencias y cierran decisiones del proyecto.

En estas explicaciones, **oracle inmediato** significa comparar A1–A4 desde el mismo pipeline y medir su utilidad al día siguiente; no equivale a una política óptima para todo el episodio. **Regret inmediato** es la diferencia entre esa mejor utilidad y la de la acción elegida. **Ablación** significa cambiar un componente del experimento para estudiar su efecto, conservando los demás.

<a id="t01"></a>

### T01 — Comprender el recorrido completo de v4

**Responsable:** Jeffrey. **Entrada:** [notebook v4](../kaggle/dqn-dummy/v4/input/main.ipynb).

**De qué trata:** construir una explicación propia de cómo los registros de pacientes se convierten en experiencias de aprendizaje. Debes identificar los componentes y poder seguir un caso concreto entre ellos.

**Cómo llevarla a cabo:**

1. Leer las celdas en este orden: 3 (configuración), 5 y 7 (carga, particiones y etiqueta), 9 (replay), 11 (pipeline y acciones), 13 y 23 (drift y calibración), 15 y 17 (observación y entorno), 19 y 27 (agente y entrenamiento), 25, 29 y 31 (evaluación, figuras y ejecución). Los índices cuentan todas las celdas desde cero, incluidas las Markdown.
2. Para cada bloque, anotar qué recibe, qué transforma y qué devuelve. Explicar cuáles datos son reales, cuáles tiempos son simulados y cómo se crea `y_dummy`.
3. Distinguir XGBoost, que predice la etiqueta, de Double DQN, que elige una intervención. Registrar que índices 0/1/2/3 corresponden a A1/A2/A3/A4.
4. Elegir una fila DQN de `actions.csv` y reconstruir observación → valores Q → acción → pipeline posterior → AUC siguiente → costo → reward → experiencia de aprendizaje.
5. Explicar la discretización diaria de diez valores y el apilamiento de tres días que forma una entrada de treinta valores. Diferenciar reward reportado y centrado de reward dentro de `learn()`.

**Entregable y cierre:** `guia_v4.md`, con mapa de componentes, tabla función/entrada/salida y una decisión explicada. Se cierra cuando puedes explicar el recorrido a Dayane y ubicar cada cálculo en el notebook.

<a id="t02"></a>

### T02 — Ejecutar una prueba pequeña reproducible

**Responsable:** Jeffrey. **Depende de:** [T01](#t01).

**De qué trata:** comprobar que puedes lanzar el experimento y reconocer sus salidas con un costo reducido. Esta prueba verifica funcionamiento y configuración; no demuestra calidad de la política.

**Cómo llevarla a cabo:**

1. Comprobar disponibilidad de `physionet.parquet` y del entorno de paquetes. El notebook admite la ruta explícita `DARL_DUMMY_PARQUET`; en Kaggle, la fuente de datos declarada está en [kernel-metadata.json](../kaggle/dqn-dummy/v4/input/kernel-metadata.json).
2. Antes de ejecutar, establecer `DARL_DUMMY_SMOKE=1` y `DARL_DUMMY_OUTPUT` con una carpeta nueva, por ejemplo `outputs/metrics/darl_cierre/smoke_v4/`. Estas variables deben existir antes de ejecutar la celda de configuración. Conservar el notebook original como referencia.
3. Ejecutar de arriba abajo y registrar plataforma, versiones, semilla, configuración efectiva y duración. Revisar la primera excepción si aparece; no continuar con celdas que dependen de un paso fallido.
4. Verificar dimensiones, particiones disjuntas, forma de observación, al menos una transición y existencia de CSV y resumen. Distinguir errores de ejecución de gates experimentales que una corrida corta puede no superar.
5. Comparar la configuración con v4 completa y explicar qué se redujo. No exigir los mismos conteos o retornos a una prueba de cuatro días.

**Entregable y cierre:** bitácora de reproducción con ruta de salidas, configuración y comprobaciones. Se cierra con ejecución pequeña inspeccionada o con un bloqueo reproducible identificado y resuelto antes de los experimentos dependientes.

<a id="t03"></a>

### T03 — Investigar por qué el agente omite A3

**Responsable:** Jeffrey. **Entradas:** [actions.csv](../kaggle/dqn-dummy/v4/outputs/actions.csv), [action_audit.csv](../kaggle/dqn-dummy/v4/outputs/action_audit.csv) y `training_history.csv`. Usar también la tabla de [T04](#t04).

**De qué trata:** investigar el mecanismo de decisión del agente en los casos donde A3 tiene mayor utilidad inmediata. Dayane cuantifica esos casos; tú buscas posibles causas en observación, entrenamiento y ejecución.

**Cómo llevarla a cabo:**

1. Filtrar la auditoría por `oracle == A3` y `chosen != A3`. Recalcular cuántos casos existen en la versión revisada, en lugar de asumir que siempre serán nueve.
2. Unir cada caso con la fila DQN de `actions.csv` mediante escenario y día. Revisar acción solicitada/ejecutada, razón de fallback, AUC previa, KS, PSI y los cuatro valores Q.
3. Reconstruir las observaciones de los tres días que recibió el agente y comprobar si distintos contextos quedan en la misma representación discretizada.
4. Revisar cómo se entrena A3: frecuencia por escenario, epsilon, selección greedy en validación y experiencias válidas. Los conteos disponibles no describen por sí solos toda la composición del replay; anotar qué registro adicional hace falta.
5. Separar hipótesis: información insuficiente, aprendizaje insuficiente, costos cercanos A3/A4, fallback o ventaja de otra acción a varios días. Proponer una prueba por hipótesis y qué resultado la respaldaría o descartaría.

**Entregable y cierre:** tabla caso/evidencia/hipótesis/prueba. Se cierra con causas candidatas concretas; las intervenciones para comprobarlas pertenecen a [T07](#t07).

<a id="t04"></a>

### T04 — Auditar resultados y comparar la utilidad de las acciones

**Responsable:** Dayane. **Entradas:** CSV y [resumen v4](../kaggle/dqn-dummy/v4/outputs/run_summary.json) guardados.

**De qué trata:** establecer cuándo A3 ofrece un mejor resultado y cuánto cuesta omitirla. Puedes empezar trabajando con las tablas exportadas, sin implementar el agente.

**Cómo llevarla a cabo:**

1. Identificar qué representa una fila de cada archivo: decisión/política/día en `actions.csv`, decisión DQN en `action_audit.csv` y resumen por escenario/política en `policy_comparison.csv`.
2. Filtrar DQN y recalcular conteos A1–A4, decisiones válidas, fallbacks y coincidencia con el oracle. Registrar el denominador de cada porcentaje y contrastarlo con el JSON.
3. Para los casos A3 omitidos, preparar columnas de escenario, día, acción elegida, utilidad A1/A3/A4, utilidad elegida y regret. Ordenar por pérdida y seleccionar dos o tres ejemplos para Jeffrey.
4. Distinguir la utilidad contrafactual desde el mismo pipeline de los resultados de una política que ejecuta A3 todos los días: sus trayectorias son diferentes.
5. Revisar figuras contra sus CSV: escenario, días, leyenda, acción y momento del reward. Registrar cualquier desfase y pedir regeneración desde una fuente identificada.

**Entregable y cierre:** auditoría de resultados y tabla de casos A3 con cálculos verificables. Se cierra cuando cada cifra y figura utilizada puede rastrearse a su archivo y filtro.

<a id="t05"></a>

### T05 — Preparar fichas de datos y sustento bibliográfico

**Responsable:** Dayane. **Entradas:** notebooks de PhysioNet, [Diabetes](../code/src/notebooks/diabetes_drift_injector.ipynb), [IEEE](../code/src/notebooks/ieee_fraud_detection_eda.ipynb), papers y referencias del repositorio.

**De qué trata:** documentar qué problema representa cada dataset y por qué se escogieron los componentes del método. Estas fichas orientan los experimentos posteriores y la redacción académica.

**Cómo llevarla a cabo:**

1. Crear una ficha por dataset con fuente, objetivo, unidad de registro, identificadores, variables, faltantes, prevalencia, reloj disponible y disponibilidad de etiquetas. Distinguir etiqueta original y etiqueta modificada.
2. Señalar qué aspectos requieren verificación: fechas de Diabetes, conservación de identificadores en TableShift, etiquetas IEEE y diferencia entre estancia de PhysioNet y calendario de camas.
3. Preparar una matriz método/problema que resuelve/motivo de uso/referencia primaria/límite. Incluir Double DQN, XGBoost, KS/PSI y taxonomía de drift.
4. Leer las fuentes que respalden cada afirmación y verificar las entradas existentes en `thesis/referencias.bib`. Una guía o resumen sirve para localizar literatura; la referencia final debe corresponder a la fuente original consultada.
5. Revisar contigo los argumentos técnicos de RL y practicar una explicación breve de datos, acciones y evaluación.

**Entregable y cierre:** tres fichas y una matriz bibliográfica con afirmaciones delimitadas. Se cierra cuando las dudas de datos están visibles y cada elección metodológica tiene argumento y fuente.

<a id="t06"></a>

### T06 — Integrar el diagnóstico inicial y revisar el alcance

**Responsables:** Jeffrey y Dayane. **Depende de:** [T01–T05](#t01).

**De qué trata:** reunir comprensión del código y evidencia de resultados para acordar qué investigar primero y qué se podrá terminar en las seis semanas.

**Cómo llevarla a cabo:**

1. Jeffrey presenta el recorrido de una decisión y la prueba pequeña; Dayane presenta los casos A3, las fichas y los límites de los resultados dummy.
2. Separar hechos comprobados, hipótesis y preguntas abiertas. Seleccionar las pruebas de semana 2 con mayor capacidad para explicar A3.
3. Preparar una nota breve para asesoría: avance disponible, problema A3, experimentos propuestos, datasets y alcance temporal real o simulado.
4. Revisar con la asesora prioridades, criterios de evidencia y requisitos de evaluación. Incluir la propuesta de cinco semillas frente a la regla vigente de semilla 42; registrar el acuerdo antes de ejecutar una excepción.
5. Anotar decisiones, responsable y siguiente tarea para cada observación. Si la reunión se retrasa, continuar las tareas independientes y conservar las decisiones que requieren respuesta como pendientes.

**Entregable y cierre:** acta de revisión inicial y lista de prioridades. El diagnóstico queda integrado cuando ambos pueden explicar qué saben, qué falta comprobar y qué prueba resolverá cada duda.

<a id="t07"></a>

### T07 — Comprobar el efecto de observación, costo y severidad

**Responsable:** Jeffrey. **Depende de:** [T03](#t03), [T04](#t04) y prioridades de [T06](#t06).

**De qué trata:** realizar comparaciones controladas para comprobar las hipótesis sobre A3. El objetivo es mejorar decisiones y desempeño con costo justificable, no aumentar artificialmente su frecuencia.

**Cómo llevarla a cabo:**

1. Fijar una configuración base, las particiones train/val y un conjunto común de escenarios. Registrar qué se cambiará y qué permanecerá igual en cada comparación.
2. Comparar bins actuales con AUC continua y tendencia calculada exclusivamente con valores disponibles al decidir. Al cambiar el tamaño de entrada, ajustar la red y reentrenar el agente; no reutilizar pesos incompatibles.
3. Comparar lambda 0, 0,025, 0,05 y 0,10 manteniendo la misma fórmula de reward, salvo el peso intervenido. Reportar AUC y costo en escalas comunes: los retornos con lambdas distintas no miden exactamente la misma preferencia.
4. Seleccionar tres velocidades de drift usando las rejillas/calibración de entrenamiento; no confundir velocidad temporal y severidad alcanzada. Cambiar una familia de parámetros por vez.
5. Registrar retorno, desempeño, costo, frecuencia de acciones, regret y estabilidad en validación. Para distinguir error inmediato de efecto futuro, estudiar continuaciones de varios días desde estados comunes, con misma política posterior y sin dar información futura al agente.
6. Combinar solo los cambios respaldados y seleccionar configuración con validación, antes de evaluar en test.

**Entregable y cierre:** tabla de ablaciones, curvas desempeño–costo y observación elegida con su argumento. Se cierra con evidencia sobre qué hipótesis explica A3 y cuáles permanecen abiertas.

<a id="t08"></a>

### T08 — Consolidar una implementación reutilizable

**Responsable:** Jeffrey. **Depende de:** [T01](#t01), [T02](#t02) y versión seleccionada en [T07](#t07).

**De qué trata:** trasladar la configuración validada a componentes reutilizables sin asumir que el paquete actual reproduce v4. Debe quedar una fuente de verdad para los siguientes experimentos.

**Cómo llevarla a cabo:**

1. Comparar notebook y módulos existentes: representación de datos, transformador, modelo, acciones, reward, disponibilidad de etiquetas y objetivo DQN. Documentar diferencias antes de importar componentes.
2. Revisar en orden `data/`, `drift/`, `pipeline/`, `actions/`, `rl/`, `evaluation/` y `visualization/`. Reutilizar lo compatible y adaptar solo lo necesario para la versión elegida, con comprobación pequeña por componente.
3. Separar variables específicas del dataset de agente, evaluación y gráficos. Definir entradas/salidas mínimas: ventana update/eval, pipeline, observación, acción, transición y registro de métricas.
4. Usar el mismo pequeño replay y las mismas acciones para comparar predicciones, observaciones, índices, reward y terminación. Controlar costos para esta prueba; tiempos reales de ejecución no serán idénticos entre corridas.
5. Verificar por separado el objetivo Double DQN y los registros exportados. Preservar trazabilidad con v4 sin crear un segundo framework completo.

**Entregable y cierre:** configuración de referencia, mapa de componentes y comprobación de equivalencia con tolerancias declaradas. Se cierra cuando la ejecución reutilizable representa fielmente el diseño elegido.

<a id="t09"></a>

### T09 — Validar los escenarios de drift y las acciones

**Responsable:** Dayane. **Apoyo:** Jeffrey prepara la configuración ejecutable. **Entradas:** [compatibilidad de concept drift](../code/src/notebooks/compatibility_concept_drift.ipynb) y [ConceptDriftInjector](../code/src/darl/drift/concept_injector.py).

**De qué trata:** comprobar que las perturbaciones producen el cambio descrito y analizar la respuesta de A1–A4. Es un experimento sobre datos y pipeline que puede realizarse antes de entrenar RL.

**Cómo llevarla a cabo:**

1. Partir de una muestra de entrenamiento y otra de evaluación separadas; fijar variables, mecanismo, severidades y semilla en la configuración preparada.
2. Generar una vez cada escenario y reutilizarlo al comparar acciones. Evitar que cada política reciba etiquetas distintas por repetir sorteos del inyector.
3. Comparar antes/después: igualdad o cambio de X, KS/PSI, prevalencia, proporción de etiquetas modificadas y desempeño del pipeline congelado.
4. Ejecutar A1–A4 desde el mismo pipeline sobre los mismos datos de actualización y evaluar en el conjunto separado. Registrar desempeño, tiempo y utilidad por severidad.
5. Contrastar `logit_reweight` con recentrado solo cuando corresponda. Explicar que el recentrado puede modificar prevalencia y que la estrategia del notebook con regresión logística no demuestra automáticamente el mismo resultado con XGBoost.

**Entregable y cierre:** tablas y gráficos mecanismo/severidad/acción, con configuración y limitaciones. Se cierra cuando puedes explicar qué cambió en los datos y dónde A3/A4 aportan una recuperación comprobable.

<a id="t10"></a>

### T10 — Definir contratos de datos y redactar su metodología

**Responsable:** Dayane. **Depende de:** [T05](#t05) y resultados pertinentes de [T09](#t09). **Revisión técnica:** Jeffrey.

**De qué trata:** especificar qué datos recibirá el entorno para cada dataset. Este contrato evita que la migración introduzca etiquetas ambiguas, columnas desordenadas o transformaciones incompatibles.

**Cómo llevarla a cabo:**

1. Definir etiqueta, identidad, reloj o índice simulado, variables numéricas/categóricas y columnas excluidas. Anotar qué información estaría disponible en el momento de predecir.
2. Especificar formato de ventanas, particiones train/val/test y separación update/eval; declarar cuándo se revelan etiquetas.
3. Definir tratamiento de faltantes, categorías desconocidas y nombres/orden de columnas transformadas. Identificar qué estimaciones se ajustan inicialmente y cuáles pueden cambiar mediante A2/A4.
4. Registrar mínimos de registros/clases para evaluar y actualizar, junto con reglas para ventanas no evaluables. Jeffrey comprueba que el código pueda cumplir el contrato.
5. Redactar los apartados de datos, drift y selección de métodos a partir de ese contrato, distinguiendo decisiones ya comprobadas de decisiones pendientes.

**Entregable y cierre:** un contrato por dataset y borrador metodológico. Se cierra cuando Jeffrey puede implementar el adaptador sin tener que adivinar la etiqueta, el tiempo o el formato esperado.

<a id="t11"></a>

### T11 — Cerrar el diagnóstico de A3

**Responsables:** Jeffrey y Dayane. **Depende de:** [T03](#t03), [T04](#t04), [T07](#t07) y [T09](#t09).

**De qué trata:** integrar las pruebas del agente con los experimentos de acciones para responder a la asesora con evidencia y decidir qué versión continúa.

**Cómo llevarla a cabo:**

1. Retomar cada hipótesis de T03 y asociarla con sus resultados: representación, aprendizaje, costo, compatibilidad o efectos futuros.
2. Jeffrey explica por qué la observación y los Q favorecen ciertas acciones; Dayane explica en qué regiones de datos/severidad A3 tiene mayor utilidad.
3. Clasificar cada hipótesis como respaldada, descartada o pendiente, usando ejemplos y comparaciones, no solo conteos de acciones.
4. Distinguir omisión de A3 con regret apreciable, empate práctico A3/A4 y desacuerdo con el oracle de un paso que puede tener explicación temporal.
5. Registrar configuración seleccionada, cambios aceptados y límites. Una pregunta abierta debe tener impacto y siguiente prueba definidos; no ampliar indefinidamente la investigación.

**Entregable y cierre:** informe diagnóstico A3 y decisión de configuración. Se cierra cuando ambos pueden explicar el uso escaso de A3 hasta donde permite la evidencia, sin atribuir una causa única no comprobada.

<a id="t12"></a>

### T12 — Adaptar el entorno y el pipeline a Diabetes

**Responsable:** Jeffrey. **Depende de:** [T08](#t08), contrato de [T10](#t10) y comprobaciones de [T14](#t14).

**De qué trata:** construir una secuencia de encuentros médicos sobre la que el agente pueda elegir actualizaciones. El tiempo y las unidades de observación deben corresponder al contrato de Diabetes.

**Cómo llevarla a cabo:**

1. Confirmar con Dayane etiqueta, identificadores y columnas disponibles. Si no existen fechas verificables, definir y registrar un orden simulado antes de experimentar.
2. Construir ventanas y separar train/val/test y update/eval, preservando grupos de pacientes cuando la información esté disponible. Comprobar que ninguna etiqueta de evaluación se use para actualizar el modelo.
3. Implementar imputación y codificación numérica/categórica ajustadas con datos permitidos. Verificar nombres, orden y significado de las columnas al aplicar A2 con Stage 2 congelado.
4. Integrar A1–A4 y el inyector validado, con severidad que evolucione por ventana. Conservar una configuración sin inyección para medir el desempeño de referencia.
5. Ejecutar un episodio pequeño con acciones fijadas manualmente. Revisar disponibilidad de clases, tamaño de observación, persistencia del pipeline, costo y evaluación posterior.

**Entregable y cierre:** adaptador Diabetes, configuración y episodio de comprobación. Se cierra cuando las cuatro acciones pueden evaluarse bajo un contrato explícito y A2 no rompe el pipeline.

<a id="t13"></a>

### T13 — Evaluar Double DQN y baselines en Diabetes

**Responsable:** Jeffrey. **Depende de:** [T12](#t12) y validación de datos de [T14](#t14).

**De qué trata:** medir si la política aprendida aporta frente a decisiones simples en Diabetes, usando escenarios y particiones comparables.

**Cómo llevarla a cabo:**

1. Definir episodios estable, covariate, concept y combined y congelar las secuencias que compartirán las políticas. Calibrar inyección y umbrales con entrenamiento.
2. Entrenar Double DQN solo con train y elegir checkpoint/configuración con val. Registrar historia de retorno, acciones y estabilidad; una ejecución que termina no implica que el entrenamiento haya convergido.
3. Evaluar en las mismas secuencias DQN, A1, A2, A3, A4, random y heurística. Reiniciar el mismo pipeline inicial antes de cada política y conservar los datos de actualización/evaluación correspondientes.
4. Guardar resultados por ventana y episodio, desempeño predictivo, costo, decisiones y auditoría inmediata cuando sea viable. Registrar ventanas sin clases suficientes y su tratamiento.
5. Iniciar las réplicas de la configuración cerrada. Mantener semilla 42 salvo excepción acordada; distinguir entrenamiento de agentes nuevos de evaluación de un agente en replays distintos.

**Entregable y cierre:** resultados piloto y matriz de ejecuciones de Diabetes. Se cierra cuando todas las políticas previstas tienen comparaciones trazables o fallos explícitos; no cuando DQN obtiene necesariamente el mejor resultado.

<a id="t14"></a>

### T14 — Verificar etiqueta, particiones y transformaciones de Diabetes

**Responsable:** Dayane. **Entradas:** [notebook de Diabetes](../code/src/notebooks/diabetes_drift_injector.ipynb), datos cargados y contrato de [T10](#t10). Se realiza en coordinación con [T12](#t12).

**De qué trata:** comprobar que el experimento predice el objetivo declarado y que los datos usados para actualizar están separados de los usados para evaluar.

**Cómo llevarla a cabo:**

1. Revisar los valores originales de `readmitted` y la transformación realizada por el cargador. Contrastar frecuencias antes/después; documentar si la clase positiva representa cualquier reingreso o exclusivamente menos de treinta días.
2. Revisar identificadores disponibles, pacientes repetidos y reglas de partición. Si TableShift eliminó información necesaria, documentarlo y acordar con Jeffrey cómo conservarla sin reconstruir identidades ficticias.
3. Preparar una tabla de tamaños, pacientes, positivos, prevalencia y faltantes por partición/rol. Verificar las intersecciones que el contrato exige vacías y qué parte del orden es simulada.
4. Revisar dónde se ajustan imputación/codificación y cómo se manejan categorías nuevas. Pedir a Jeffrey evidencia de nombres/orden y comportamiento de A2.
5. Verificar estabilidad de X bajo concept, cambios de prevalencia/etiquetas y comportamiento por severidad. Señalar qué perturbaciones son controles experimentales y cuáles tienen argumento de dominio.

**Entregable y cierre:** ficha Diabetes actualizada y lista de comprobaciones con evidencia. Se cierra antes de T13 cuando etiqueta, particiones y transformaciones coinciden con la descripción del experimento.

<a id="t15"></a>

### T15 — Interpretar Diabetes y preparar la evaluación IEEE

**Responsable:** Dayane. **Depende de:** [T13](#t13), [T14](#t14) y contrato IEEE de [T10](#t10).

**De qué trata:** convertir las salidas de Diabetes en resultados explicables y preparar los datos del siguiente experimento mientras Jeffrey ejecuta las campañas.

**Cómo llevarla a cabo:**

1. Resumir desempeño, costo y acciones por escenario/severidad; comparar A3 y A4 y localizar pérdidas importantes frente al oracle inmediato.
2. Seleccionar ejemplos de actualización útil, espera razonable y decisión discutible. Explicar qué ocurre en los datos y contrastar la interpretación del agente con Jeffrey.
3. Redactar método y resultados iniciales de Diabetes, dejando explícitos etiqueta, secuencia simulada si procede y límites de las réplicas disponibles.
4. Revisar el [EDA IEEE](../code/src/notebooks/ieee_fraud_detection_eda.ipynb) y preparar rangos de `TransactionDT`, volumen y positivos por ventana. Proponer tamaños viables de ventana con la información de preparación permitida.
5. Entregar a Jeffrey esquema de columnas, propuesta train/val/test y reglas de update/eval para IEEE; fijarlos antes de usar test para evaluar.

**Entregable y cierre:** borrador Diabetes y propuesta de ventanas IEEE. Se cierra cuando los resultados tienen interpretación ligada a tablas y el adaptador IEEE tiene entradas claramente definidas.

<a id="t16"></a>

### T16 — Revisar con la asesora el resultado fuera del dummy

**Responsables:** Jeffrey y Dayane. **Depende de:** [T11](#t11), [T13](#t13), [T14](#t14) y [T15](#t15).

**De qué trata:** contrastar el diagnóstico y la evidencia Diabetes con las expectativas académicas antes de consolidar los experimentos finales.

**Cómo llevarla a cabo:**

1. Preparar una presentación breve con objetivo, arquitectura elegida, respuesta sobre A3 y tabla DQN/baselines en Diabetes.
2. Jeffrey explica observación, reward, aprendizaje y cambios comprobados. Dayane explica etiqueta, particiones, mecanismo de drift, desempeño y costo.
3. Mostrar casos donde DARL funciona y donde la heurística u otra política resulta competitiva. Distinguir evidencia piloto de validación final.
4. Revisar dudas de compatibilidad A2, significado temporal, sustento de algoritmos, réplicas y alcance IEEE observado.
5. Convertir cada observación en una acción con responsable, prioridad y efecto sobre el calendario. Confirmar cualquier reducción de objetivos formales.

**Entregable y cierre:** acta de asesoría y decisiones incorporadas a las configuraciones/redacción. Se cierra cuando las observaciones críticas tienen una resolución o una tarea definida.

<a id="t17"></a>

### T17 — Ejecutar IEEE sintético y temporal observado

**Responsable:** Jeffrey. **Depende de:** [T08](#t08), [T10](#t10), preparación de [T15](#t15) y comprobaciones de [T18](#t18).

**De qué trata:** trasladar el mismo enfoque de mantenimiento a transacciones y evaluar dos condiciones: perturbación controlada y evolución observada sin inyección.

**Cómo llevarla a cabo:**

1. Implementar el adaptador de transacciones, ordenar por `TransactionDT` y fijar bloques temporales train/val/test. Definir cómo se separan update/eval dentro del contrato y cuándo llegan etiquetas.
2. Ajustar pipeline inicial con el bloque permitido y comprobar A1–A4, categorías desconocidas y ventanas insuficientes. Reutilizar agente/evaluador con dimensiones acordes a la observación elegida.
3. Crear la variante sintética con mecanismo y severidad calibrados en train. Conservar las etiquetas originales en el experimento observado, identificando cualquier columna modificada en la variante sintética.
4. Entrenar/seleccionar usando train/val según el protocolo de cada condición y evaluar todas las políticas desde el mismo pipeline inicial sobre secuencias comunes. Si se plantea transferencia de agente entre condiciones, declararla como un experimento separado.
5. Guardar resultados separados de sintético/observado, costo por acción, decisiones, casos no evaluables y configuración. Iniciar réplicas de las versiones cerradas.

**Entregable y cierre:** dos evaluaciones IEEE trazables. Se cierra cuando datos, políticas y condiciones son comparables y no existe mezcla entre etiquetas originales y perturbadas.

<a id="t18"></a>

### T18 — Auditar la unión y las ventanas de IEEE

**Responsable:** Dayane. **Entradas:** `train_transaction`, `train_identity` si se utiliza y contrato IEEE. Se realiza antes y durante [T17](#t17).

**De qué trata:** asegurar que la preparación de transacciones no duplica registros ni introduce información futura y que las ventanas tienen datos suficientes para evaluar.

**Cómo llevarla a cabo:**

1. Verificar unicidad de `TransactionID` en cada tabla, contar coincidencias y revisar cardinalidad de la unión. Comparar filas antes/después; una unión opcional de identidad no debe multiplicar transacciones.
2. Medir faltantes y cobertura de identidad por periodo. Documentar imputación, columnas excluidas y tratamiento de categorías que aparecen después del entrenamiento.
3. Verificar orden de `TransactionDT`, límites de bloques temporales, etiqueta `isFraud` y ausencia de uso de etiquetas del test público no disponibles.
4. Preparar conteos de transacciones, positivos y prevalencia por ventana/rol. Aplicar reglas predefinidas para agregar ventanas o declarar métricas indefinidas; no cambiar tamaños para mejorar resultados de test.
5. Revisar la disponibilidad simulada de etiquetas y las entidades que pueden repetirse entre periodos. Documentar su relación con el caso de despliegue y excluir variables construidas con datos futuros.

**Entregable y cierre:** reporte de calidad IEEE y tabla de ventanas. Se cierra cuando la unión preserva la granularidad y cada ventana tiene evaluación válida o un tratamiento explícito.

<a id="t19"></a>

### T19 — Analizar cambios temporales y resultados IEEE

**Responsable:** Dayane. **Depende de:** [T17](#t17) y [T18](#t18).

**De qué trata:** explicar los resultados en fraude y establecer qué cambios están respaldados por los datos. La condición sin inyección necesita interpretación cuidadosa de su evolución temporal.

**Cómo llevarla a cabo:**

1. Graficar por ventana volumen, prevalencia, señales sobre X y desempeño de las políticas. Comparar periodos con las referencias definidas en el protocolo.
2. Analizar si cambios de distribución, prevalencia y pérdida predictiva aparecen juntos o separados. Una tasa de fraude variable por sí sola no identifica concept drift.
3. Contrastar mantenimiento selectivo frente a no actualizar, actualización completa, random y heurística. Describir cuándo aporta A3 y cuándo A4 compensa su costo.
4. Verificar que ejes temporales, leyendas, métricas y filtros de cada gráfico coincidan con las tablas. Distinguir episodios/corridas y mostrar límites de la evidencia disponible.
5. Redactar resultados sintéticos y observados por separado, señalando qué mecanismo se conoce por construcción y qué causas del cambio observado siguen sin identificarse.

**Entregable y cierre:** borrador de resultados IEEE y figuras verificadas. Se cierra cuando cada conclusión está respaldada por datos y su alcance causal está delimitado.

<a id="t20"></a>

### T20 — Congelar la matriz final de experimentos

**Responsables:** Jeffrey y Dayane. **Depende de:** [T13](#t13), [T17](#t17), auditorías de datos y decisiones de asesoría. **Fecha de cierre:** 4 de noviembre.

**De qué trata:** establecer exactamente qué se evaluará para evitar cambios continuos durante la semana destinada a validar y terminar el manuscrito.

**Cómo llevarla a cabo:**

1. Listar dataset/condición, escenarios, políticas, particiones, observación, reward, costos, hiperparámetros y regla de checkpoint de cada experimento final.
2. Registrar archivos/configuraciones, versión de código y réplicas autorizadas. Comprobar que estas últimas distingan agentes reentrenados y escenarios de evaluación.
3. Estimar tiempo y almacenamiento a partir de los pilotos; ordenar corridas por prioridad y definir cómo reanudar sin mezclar resultados de versiones distintas.
4. Confirmar métricas principales, comparaciones pareadas y exclusiones de ventanas. Separar los resultados usados para elegir diseño de los reservados para evaluar.
5. Fijar que desde este cierre se corrigen fallos que invaliden evidencia y se documentan cambios de versión. No abrir nuevas búsquedas basadas en test ni incorporar agentes/datasets adicionales sin revisar alcance.

**Entregable y cierre:** matriz final, configuraciones congeladas y calendario de cómputo. Se cierra cuando cada corrida prevista tiene objetivo, entrada, responsable y criterio de evaluación.

<a id="t21"></a>

### T21 — Completar las corridas finales y su trazabilidad

**Responsable:** Jeffrey. **Depende de:** [T20](#t20).

**De qué trata:** ejecutar la matriz cerrada, consolidar la evidencia y dejar un procedimiento para reproducirla. Los resultados deben poder vincularse a una configuración específica.

**Cómo llevarla a cabo:**

1. Ejecutar cada corrida con ID, configuración, semilla autorizada, versión de código y carpeta propia en `outputs/`. Conservar checkpoints y registros necesarios sin versionar modelos o logs.
2. Si se acordaron cinco semillas, reentrenar/evaluar conforme a ese protocolo. Si se conserva 42, distinguir réplicas de escenarios de variación entre semillas; no presentar una como si fuera la otra.
3. Comprobar archivos completos, políticas presentes, ventanas válidas y motivos de fallo. No descartar una corrida por obtener mal desempeño; registrar criterios de exclusión técnicos.
4. Consolidar baselines y ablaciones y verificar estabilidad, decisiones A3/A4 y regret inmediato. Entregar a Dayane resultados por corrida, no solo promedios.
5. Documentar preparación, comandos o ejecución del notebook, parámetros y archivos esperados. Si un fallo exige modificar implementación, registrar qué corridas deben repetirse para mantener comparabilidad.

**Entregable y cierre:** matriz ejecutada con estado de cada corrida, resultados crudos y procedimiento de reproducción. Se cierra con todas las ejecuciones previstas resueltas o con fallos y límites explícitamente reportados.

<a id="t22"></a>

### T22 — Consolidar métricas, incertidumbre, tablas y figuras

**Responsable:** Dayane. **Depende de:** resultados de [T21](#t21) y protocolo de [T20](#t20).

**De qué trata:** cuantificar desempeño y variabilidad sin inflar el tamaño de muestra. Debes distinguir el resultado medio de cuán estable es esa diferencia frente a los baselines.

**Cómo llevarla a cabo:**

1. Organizar una tabla por dataset/condición/escenario/política/corrida, conservando denominadores, ventanas evaluables, métricas predictivas, retorno, costo y decisiones.
2. Comprobar unidades y definiciones. AUROC, AP, retorno y regret responden a preguntas distintas; no mezclar retornos de rewards con escalas diferentes.
3. Calcular diferencias pareadas DQN–baseline en las mismas secuencias. Definir la unidad independiente: agente/semilla si se reentrenó, o escenario independiente según el diseño; horas y días correlacionados no aumentan por sí solos el número de réplicas.
4. Calcular media, dispersión e IC del 95% con método declarado y adecuado al diseño. Con pocas réplicas, explicar incertidumbre y límites; un IC sobre escenarios no mide estabilidad entre semillas.
5. Construir tablas/figuras desde resultados crudos, revisar filtros y leyendas y exportar primero a `outputs/`. Pasar los artefactos finales a las carpetas generadas de `thesis/`, con trazabilidad de origen.

**Entregable y cierre:** resumen estadístico, figuras revisadas y notas del cálculo. Se cierra cuando otra persona puede reconstruir cifras e intervalos a partir de las mismas corridas.

<a id="t23"></a>

### T23 — Completar el manuscrito y su argumentación

**Responsable:** Dayane. **Revisión técnica:** Jeffrey. **Entradas:** borradores previos, [T11](#t11), [T19](#t19), [T20](#t20) y resultados de [T22](#t22).

**De qué trata:** construir una tesis completa que conecte problema, método, experimentos y conclusiones. Cada afirmación debe representar el diseño y la evidencia efectivamente obtenidos.

**Cómo llevarla a cabo:**

1. Revisar objetivos y organizar capítulos: problema/antecedentes, fundamentos, metodología, resultados, discusión, conclusiones y referencias.
2. Integrar fichas y contratos, pipeline, acciones, observación, reward, entrenamiento y protocolo estadístico. Pedir a Jeffrey revisión de RL y de las diferencias respecto a v4.
3. Separar prueba dummy, escenarios sintéticos con objetivos de dataset y secuencia IEEE observada. Identificar qué evidencia responde a cada objetivo.
4. Describir resultados favorables y desfavorables y explicar límites: simulación, etiquetas, compatibilidad, costo medido, estabilidad, réplicas y comparación con oracle inmediato.
5. Formular conclusiones proporcionales a los datos y revisar citas. En LaTeX, toda fórmula añadida debe seguir fórmula → `Donde:` → símbolos → explicación conceptual.

**Entregable y cierre:** borrador completo, sin secciones centrales vacías, con tablas y referencias integradas. Se cierra cuando existe una respuesta sustentada para cada objetivo, incluidos resultados limitados o negativos.

<a id="t24"></a>

### T24 — Revisar y entregar el borrador a la asesora

**Responsables:** Jeffrey y Dayane. **Depende de:** [T21](#t21), [T22](#t22) y [T23](#t23). **Fecha objetivo:** 11 de noviembre.

**De qué trata:** comprobar consistencia técnica y narrativa antes de solicitar correcciones finales. Esta tarea describe una entrega futura; no envía mensajes desde este documento.

**Cómo llevarla a cabo:**

1. Jeffrey contrasta metodología con configuración final: dimensiones, Double DQN, reward, costos, hiperparámetros, etiquetas, particiones y selección de checkpoint.
2. Dayane contrasta cifras y argumentos con tablas, figuras y bibliografía. Revisar que cada resultado permita identificar su experimento.
3. Ambos revisan objetivos, resultados y conclusiones y corrigen contradicciones o afirmaciones sin evidencia.
4. Generar un PDF legible y preparar una lista breve de preguntas específicas para la asesora. Señalar explícitamente cualquier pendiente relevante.
5. Realizar la entrega por el canal habitual y registrar la versión enviada y las observaciones recibidas para planificar [T25](#t25) y [T26](#t26).

**Entregable y cierre:** borrador revisado y registro de entrega. Se cierra cuando la asesora dispone de una versión completa identificable y las correcciones posteriores pueden asociarse a ella.

<a id="t25"></a>

### T25 — Resolver correcciones técnicas y preparar la demo

**Responsable:** Jeffrey. **Depende de:** [T24](#t24) y observaciones técnicas recibidas.

**De qué trata:** cerrar problemas de implementación, reproducción y sustento técnico antes de la entrega, y preparar una demostración breve del recorrido del sistema.

**Cómo llevarla a cabo:**

1. Clasificar observaciones técnicas por impacto: error que invalida evidencia, inconsistencia de descripción o aclaración. Priorizar las dos primeras y registrar qué se corrige.
2. Reproducir una corrida representativa con el procedimiento de T21, verificando entrada, configuración, predicciones y archivos esperados. Declarar tolerancias y variación de tiempos; si no se reentrena completamente, explicar el alcance de la comprobación.
3. Revisar fórmulas, reward, discreción/continuidad del estado, información disponible y algoritmo realmente implementado. Confirmar que las conclusiones sigan válidas tras cada corrección.
4. Preparar una demo con una ventana, observación, Q, acción y resultado posterior; acompañarla de salidas guardadas para poder mostrarla sin depender de un entrenamiento largo en vivo.
5. Ensayar respuestas sobre preparación de datos, motivo de Double DQN, costo, A3 y límites de la evaluación. Revisarlas con Dayane.

**Entregable y cierre:** lista de correcciones técnicas resueltas, evidencia de reproducción y demo preparada. Se cierra cuando implementación y explicación de la tesis son coherentes y puedes mostrar un caso verificable.

<a id="t26"></a>

### T26 — Revisar el PDF final y preparar la presentación

**Responsable:** Dayane. **Depende de:** [T24](#t24) y coordinación con cambios de [T25](#t25).

**De qué trata:** integrar correcciones académicas y comprobar que la tesis y sus gráficos se entienden en el documento renderizado. La presentación debe destacar problema, evidencia y contribución.

**Cómo llevarla a cabo:**

1. Registrar observaciones de redacción y resolverlas con sus fuentes. Coordinar cualquier cambio de contenido técnico con Jeffrey para evitar descripciones desactualizadas.
2. Revisar citas/BibTeX, referencias cruzadas, numeración, anexos y correspondencia entre tablas y resultados. Conservar redacción académica en español.
3. Compilar con el procedimiento vigente del proyecto y revisar el PDF página por página: cortes, tablas fuera de márgenes, símbolos, legibilidad, figuras y referencias. Corregir las fuentes y regenerar artefactos; no editar archivos generados manualmente.
4. Preparar diapositivas de problema, datasets, pipeline, diseño experimental, resultados, diagnóstico A3 y limitaciones. Usar figuras verificadas y títulos que expresen lo que la evidencia permite afirmar.
5. Preparar respuestas sobre etiquetas, tiempo real/simulado, drift, métricas y comparaciones. Practicar una explicación básica del agente con Jeffrey.

**Entregable y cierre:** PDF revisado visualmente, lista de correcciones y diapositivas completas. Se cierra cuando el documento compila y los elementos principales son legibles y consistentes.

<a id="t27"></a>

### T27 — Cerrar entrega y ensayar la sustentación

**Responsables:** Jeffrey y Dayane. **Depende de:** [T25](#t25) y [T26](#t26). **Hitos:** versión final el 17 de noviembre, verificación/ensayos el 18 y entrega objetivo el 19.

**De qué trata:** cerrar una versión identificable de la tesis y comprobar que ambos pueden sustentar el proyecto completo, además de su frente de trabajo.

**Cómo llevarla a cabo:**

1. El 17, cerrar manuscrito, referencias, figuras, anexos y documentación de reproducción; registrar versión y comprobar los requisitos formales de entrega vigentes.
2. El 18, abrir PDF y presentación en el entorno donde se usarán, verificar archivos, enlaces y demo y disponer de copia de respaldo. Evitar cambios metodológicos nuevos en este cierre.
3. Realizar un ensayo completo con tiempo medido y reparto de intervenciones: Jeffrey explica entorno/agente; Dayane explica datos, experimentos y resultados. Ambos presentan contribución y límites.
4. Realizar un segundo ensayo centrado en preguntas: A3, elección de algoritmos, simulación, fuga de información, calidad de baselines, incertidumbre y resultados negativos. Corregir respuestas o diapositivas confusas.
5. Revisar lista de entrega y realizarla por el canal académico indicado el 19; conservar constancia de versión entregada.

**Entregable y cierre:** paquete final verificado, dos ensayos realizados y entrega registrada. Ambos pueden explicar la contribución, el diagnóstico A3 y qué conclusiones permite la evidencia.
