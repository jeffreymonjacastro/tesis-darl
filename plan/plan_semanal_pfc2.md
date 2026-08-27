# Tablero de seguimiento semanal y cierre de brechas de DARL

**Proyecto:** DARL: *Drift-Aware Reinforcement Learning for Selective Updating of Two-Stage Tabular ML Pipelines*  
**Horizonte:** 13 semanas, desde la semana 3 hasta la semana 15  
**Equipo:** Jeffrey Monja y Brigitte Rojas  
**Fecha de diagnóstico:** 25 de agosto de 2026

## 1. Diagnóstico ejecutivo

El proyecto tiene una base conceptual y una demostración técnica útiles para PFC1, pero todavía no cuenta con evidencia experimental suficiente para sostener las conclusiones centrales de PFC2. La tesis compila correctamente en un PDF de 89 páginas y el paquete `code/src/darl` pasa compilación estructural; además, el *smoke test* de PPO se ejecuta. Sin embargo, ese entorno usa escenarios sintéticos cuyas recuperaciones de AUC fueron prefijadas para favorecer determinadas acciones, no transiciones producidas por ejecuciones reales del pipeline. En consecuencia, los gráficos actuales prueban que el flujo de software funciona, pero no que DARL aprenda una política superior ni que generalice.

La asesoría de la semana 3 establece un cambio de enfoque: **primero se necesita una solución mínima que aprenda y produzca evidencia, antes de optimizar el algoritmo, ampliar el monitor o perfeccionar la ingeniería**. El trabajo se dividirá en dos frentes desacoplados: Jeffrey liderará el entorno de RL, la recompensa y las acciones; Brigitte liderará la generación de datos, la inyección de drift y el monitoreo. Ambos frentes usarán interfaces mínimas para que el bloqueo de uno no detenga al otro.

Los bloqueos prioritarios son los siguientes:

1. resolver como máximo en la semana 4 si `Update features` es compatible, debe redefinirse o debe retirarse;
2. demostrar de inmediato que un agente DQN puede aprender en un entorno mínimo con una recompensa simple y datos disponibles;
3. desarrollar en paralelo un entorno secuencial y una generación de drift condicionada por dataset, sin hacer que un frente bloquee al otro;
4. empezar el monitoreo con PSI y añadir detectores solo si la evidencia demuestra que PSI es insuficiente;
5. validar la política contra una tabla de decisión empírica y un baseline de umbral fijo antes de comparar familias complejas de RL;
6. posponer la comparación PPO–DQN, la generalización amplia, la reproducibilidad exhaustiva y la edición final de la tesis hasta obtener resultados mínimos interpretables.

El principio de gestión para las 13 semanas será: **velocidad de aprendizaje experimental con evidencia mínima verificable**. Durante las primeras semanas bastan scripts exploratorios, configuraciones simples y salidas guardadas; el empaquetado, la automatización completa y la limpieza del código se realizarán después de confirmar que el enfoque produce resultados útiles.

### Decisiones de prioridad derivadas de la asesoría W3

- **P0 — Semanas 3–4:** cerrar la compatibilidad Stage 1–Stage 2 y presentar una demo en la que DQN realmente entrene; la recompensa, el modelo y el entorno pueden ser deliberadamente simples.
- **P1 — Semanas 5–10:** mejorar en paralelo el entorno secuencial y la generación de datos, integrar acciones reales y contrastar DQN contra baselines simples.
- **P2 — Semanas 11–13:** comparar PPO, ampliar datasets y endurecer reproducibilidad únicamente si el núcleo ya genera resultados interpretables; de lo contrario, usar estas semanas para corregir entorno, recompensa o datos.
- **P3 — Semanas 14–15:** consolidar resultados, redacción, demo y defensa.

Los objetivos asociados al tablero deben expresar resultados, no pasos: obtener un módulo de monitoreo evaluado; formular una política selectiva con un espacio de acciones revisable y una utilidad interpretable; y contrastar la política aprendida contra una tabla empírica y un umbral fijo. No se fijará el número de acciones en el objetivo de tesis hasta cerrar el gate de compatibilidad.

## 2. Evidencia revisada en el repositorio

- `asesorias/Asesoria_w3_transcripcion.md` registra las prioridades y restricciones acordadas con Ariana para la planificación de PFC2.
- `thesis/main.tex` incluye introducción, revisión de literatura, marco teórico, metodología, resultados preliminares y conclusiones.
- `thesis/secciones/capitulo4.tex` define el pipeline, los inyectores, el monitor, las acciones, la recompensa y la demo PPO.
- `code/src/darl/drift/injector.py` implementa drift numérico/categórico y concept drift mediante inversión aleatoria de etiquetas.
- `code/src/darl/actions/selective_update.py` implementa A1, A2, A2c, A3 y A4 principalmente para regresión logística y variables numéricas.
- `code/src/darl/rl/scenarios.py` genera recuperaciones sintéticas usando factores fijos por tipo de drift.
- `code/src/darl/rl/env.py` avanza a una fila independiente después de cada acción; la acción no modifica el escenario siguiente.
- `code/src/darl/rl/training.py` reporta `best_action_share` como la frecuencia de la acción más usada, no como exactitud frente a la acción óptima.
- `outputs/metrics/` no contiene métricas crudas de experimentos; `code/tests/` todavía no contiene pruebas automatizadas.
- Existe una segunda implementación en `code/drift_framework/`; antes de PFC2 debe definirse `code/src/darl/` como única fuente de verdad y migrar solo lo que se valide.

### Comprobaciones realizadas durante esta revisión

- `python -m compileall -q code/src/darl`: **PASS**.
- *Smoke test* PPO de 128 pasos: **PASS**, con 108 escenarios sintéticos; este resultado valida ejecución, no eficacia empírica.
- `latexmk -pdf -outdir=build main.tex`: **PASS**, 89 páginas y sin referencias indefinidas; permanecen tres advertencias menores de `Overfull hbox`.

## 3. Gaps prioritarios y cómo resolverlos

### Gap 1 — Compatibilidad entre la actualización de features y el modelo predictivo

**Problema.** El pipeline se define como `h(x) = g_theta(f_phi(x))`. Si se reemplaza `f_phi` por `f_phi'` y se conserva `g_theta`, no basta con que la nueva matriz tenga el mismo número de columnas: Stage 2 fue entrenado con una escala, orden, codificación y semántica concretas. Un nuevo `QuantileTransformer`, `StandardScaler` o `TargetEncoder` puede mover una observación a otra región del espacio intermedio y cambiar las decisiones de un modelo congelado. La literatura de adaptación de dominio advierte que obtener representaciones alineadas o invariantes no garantiza por sí mismo bajo error en origen y destino, especialmente cuando también existe cambio condicional ([Zhao et al., 2019](https://proceedings.mlr.press/v97/zhao19a.html)). La documentación de `QuantileTransformer` también señala que la transformación es no lineal y puede distorsionar correlaciones ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)).

**Solución propuesta.** Ejecutar un *spike* con fecha límite al cierre de la semana 4 e implementar un `FeatureContract` mínimo para Stage 1 con cuatro validaciones:

1. contrato estructural: mismos nombres, orden, dimensión, tipos y tratamiento de categorías desconocidas;
2. contrato semántico: cada columna transformada conserva la misma interpretación para Stage 2;
3. prueba de estabilidad sobre un conjunto ancla: comparar `g_theta(f_phi(X_anchor))` con `g_theta(f_phi'(X_anchor))` mediante acuerdo de predicción, correlación de scores y cambio de AUC/log-loss;
4. prueba de utilidad en destino: A2 solo se considera válida si mejora el pipeline completo frente a A1 y no presenta degradación material en el conjunto ancla.

Se compararán primero las variantes indispensables: **A2-refit** (reajustar Stage 1) y **A2c-correctiva** (mapear el dominio actual hacia la referencia manteniendo congelado el pipeline). *Importance weighting* quedará como alternativa si ninguna de las dos resulta razonable bajo covariate shift ([Sugiyama et al., 2007](https://www.jmlr.org/beta/papers/v8/sugiyama07a.html)). Al terminar la semana 4 debe existir una decisión binaria y documentada: mantener `Update features` con contrato, redefinirla como corrección compatible o retirarla. El resto del entorno no asumirá desde ahora un número fijo de acciones.

**Evidencia de cierre.** Matriz mínima por transformador y modelo con compatibilidad estructural, cambio de scores, AUC, log-loss y costo; decisión sobre la acción registrada antes de iniciar la semana 5.

### Gap 2 — Inyector de concept drift poco realista y metodología inconsistente

**Problema.** El código final aplica *label flipping* simétrico con probabilidad `0.45 x severity`. Eso introduce ruido uniforme, no un cambio estructurado y explicable en `P(Y|X)`. A la vez, la metodología escrita describe una inversión de covariables alrededor de la mediana manteniendo las etiquetas, mecanismo que también puede cambiar `P(X)` y por tanto no garantiza concept drift puro. Código y tesis describen experimentos distintos.

**Solución propuesta.** Brigitte desarrollará este frente en paralelo al entorno de RL de Jeffrey, empezando por un solo mecanismo plausible en `diabetes_readmission` antes de ampliar combinaciones. Se diseñarán perfiles por dataset:

- **Hospital Readmission:** modificar el riesgo condicional por subgrupos plausibles definidos por fuente de admisión, historial de hospitalizaciones, utilización de emergencia y variables clínicas seleccionadas mediante literatura de dominio.
- **PhysioNet/Sepsis:** modificar gradualmente la relación entre variables fisiológicas/tiempo de estancia y el riesgo, respetando rangos clínicos y dependencias entre variables.
- generar cambios abruptos, graduales, recurrentes y combinados;
- mantener una variante de ruido de etiquetas únicamente como control negativo, con ese nombre explícito;
- validar por separado `P(X)`, prevalencia de `Y`, caída de desempeño y cambio condicional estimado.

Una opción controlable es modificar el logit del riesgo base mediante una función de variables relevantes y una intensidad temporal, en lugar de invertir etiquetas al azar. Cada mecanismo debe tener un *dataset card* que explique qué relación cambia y por qué es plausible. La taxonomía debe conservar la distinción entre cambio real de `P(Y|X)`, covariate shift y label shift descrita por la literatura de concept drift ([Gama et al., 2014](https://dl.acm.org/doi/10.1145/2523813)).

**Evidencia de cierre.** En el primer gate, una secuencia de `diabetes_readmission` presenta degradación controlable y visualizable; posteriormente, para concept drift puro, los detectores que usan solo `X` permanecen dentro de sus límites nulos mientras las métricas supervisadas muestran degradación ordenada por severidad.

### Gap 3 — El entorno actual no es genuinamente secuencial

**Problema.** En el entorno actual, una acción produce una recompensa y luego el cursor pasa a otra fila permutada. Diferir, actualizar Stage 1 o reentrenar todo no cambia el próximo escenario. Por eso el retorno es esencialmente una suma de decisiones inmediatas y PPO no demuestra una ventaja temporal frente a un clasificador o una tabla de decisión.

**Solución propuesta.** Trabajar en dos versiones. La **v0** debe estar lista en la semana 4 y solo demostrar que DQN puede aprender con el entorno, recompensa y datos disponibles, aunque la dinámica sea simplificada. La **v1** incorporará después un estado latente de salud del pipeline con severidad covariada, severidad conceptual, edad de Stage 1, edad de Stage 2, tiempo desde la última acción, costo acumulado y disponibilidad de etiquetas. La transición v1 debe cumplir reglas comprobables:

- `Defer` permite persistencia o crecimiento del drift y acumula pérdida de servicio;
- `Update features` reduce solo el componente que Stage 1 puede corregir y puede introducir penalización por incompatibilidad;
- `Update model` reduce degradación conceptual cuando hay etiquetas suficientes;
- `Retrain all` restablece ambos componentes con mayor costo y posible tiempo de indisponibilidad.

El agente observará métricas ruidosas y/o retrasadas, no el estado latente. Se evaluarán historia apilada o política recurrente solo después de validar la dinámica básica. Si la acción no mejora la predicción de estados futuros o si `gamma` no cambia las decisiones, el problema debe reformularse honestamente como *contextual bandit* o clasificación de acciones.

**Evidencia de cierre.** Primero, una demo DQN v0 que aprende una regla controlada y supera una política aleatoria en episodios separados; después, pruebas de transición, trayectorias visualizables y comparación donde una política miope y una política secuencial enfrenten exactamente las mismas secuencias. La demo v0 es un sanity check, no evidencia final de eficacia de DARL.

### Gap 4 — Recompensa, política y métricas de decisión no validadas

**Problema.** La recompensa actual resta costos normalizados a una diferencia absoluta de AUC sin demostrar que ambas escalas o los pesos `lambda_A` y `lambda_C` representan preferencias justificadas. Además, la tabla llama “recuperación” a `AUC_estrategia / AUC_baseline`, que en realidad es rendimiento retenido. Finalmente, `best_action_share` mide concentración de acciones, no decisiones correctas.

**Solución propuesta.** Empezar con una recompensa v0 deliberadamente simple que combine AUC y costo temporal inverso, con escalas y protección numérica explícitas. No se hará una búsqueda amplia de pesos hasta comprobar que el agente puede aprender. Una vez superado ese sanity check, se separarán cuatro resultados:

1. **pérdida de AUC recuperada:** `(AUC_postaccion - AUC_drift) / (AUC_baseline - AUC_drift)`;
2. **calidad de decisión:** `optimal_action_rate` y `mean_regret` frente al oráculo que maximiza la misma utilidad;
3. **desempeño secuencial:** retorno medio, pérdida acumulada y costo acumulado por episodio;
4. **eficiencia:** tiempo, RAM, actualizaciones completas evitadas y costo relativo.

La recuperación debe manejar denominadores cercanos a cero y reportarse junto con AUC absoluto. La primera comparación obligatoria será DQN frente a una tabla de decisión empírica, un umbral fijo y una política aleatoria. La grilla de pesos, Pareto, oráculo completo y baselines adicionales se incorporarán solo después de obtener una señal de aprendizaje estable. Los diagnósticos internos de DQN o PPO no cuentan como evidencia de recuperación del pipeline.

**Evidencia de cierre.** Un mismo archivo de evaluación calcula utilidad, acción óptima y regret para todas las políticas sobre los mismos episodios y semillas; se reportan resultados positivos o negativos sin cambiar la métrica después de observarlos.

### Gap 5 — La selección de PPO no está demostrada, pero su comparación no es prioritaria

**Problema.** PPO es un candidato razonable, pero todavía no puede presentarse como el mejor algoritmo para DARL. El entorno actual tiene acciones discretas —cuatro en el prototipo, sujetas al gate de compatibilidad— y un vector de observación continuo: esa combinación admite tanto una política Actor-Crítico como un aproximador de valores. PPO reutiliza cada lote durante varias épocas, pero sigue siendo *on-policy* ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)); DQN es *off-policy* y reutiliza transiciones mediante *experience replay* y una red objetivo, lo que puede ahorrar interacciones cuando cada transición exige ejecutar o reentrenar un pipeline real ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)). La guía oficial de Stable-Baselines3 recomienda probar DQN para acciones discretas en un solo proceso por su eficiencia muestral, y PPO/A2C cuando se puede paralelizar la recolección y se prioriza el tiempo de pared ([Stable-Baselines3](https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html)). Por tanto, la decisión depende del costo del simulador, la estabilidad y la generalización, no solo del tamaño del espacio de acciones.

Q-learning tabular no será el candidato principal: sus garantías clásicas requieren representar y visitar repetidamente los pares estado–acción ([Watkins y Dayan, 1992](https://doi.org/10.1007/BF00992698)), mientras DARL observa variables continuas; discretizarlas puede perder información o producir una tabla poco cubierta. Se conservará como baseline interpretable sobre un estado agregado. DQN sí acepta observaciones continuas y acciones discretas, pero puede sobreestimar valores; si aparece ese fallo se evaluará Double DQN, que fue diseñado para mitigarlo ([van Hasselt et al., 2016](https://doi.org/10.1609/aaai.v30i1.10295)). Además, ni `MlpPolicy` de PPO ni DQN feed-forward resuelven por sí solos la observabilidad parcial: con etiquetas retrasadas se debe comparar historia apilada y, solo si aporta valor, una variante recurrente como DRQN o RecurrentPPO ([Hausknecht y Stone, 2015](https://arxiv.org/abs/1507.06527)).

**Solución propuesta.** Usar **DQN como algoritmo de arranque y sanity check**, porque el espacio actual es discreto y una dificultad de convergencia de PPO podría confundirse con un fallo del entorno. No se comparará PPO contra DQN durante el gate inicial. La selección algorítmica solo se abrirá después de que DQN aprenda, las acciones produzcan resultados reales y el entorno v1 demuestre transiciones dependientes de la acción. En ese gate posterior, el protocolo podrá comparar:

1. políticas no-RL: acciones fijas, umbral, clasificador supervisado, *contextual bandit* y oráculo;
2. Q-learning tabular con discretización predefinida del estado, como baseline de simplicidad e interpretabilidad;
3. DQN y PPO con presupuestos equivalentes de interacción y ajuste, sobre exactamente los mismos episodios;
4. historia apilada para ambos algoritmos cuando exista retraso; una variante recurrente solo si la ablación confirma que la memoria es necesaria.

La comparación, si se activa, medirá utilidad/retorno, pérdida de AUC recuperada, regret, costo acumulado, decisiones inválidas, interacciones hasta alcanzar un nivel de desempeño, tiempo de pared, RAM y estabilidad entre réplicas. PPO se conservará como candidato motivado por una posible extensión futura a acciones continuas, no como supuesto algoritmo principal. Si DQN no supera a los baselines simples, el equipo corregirá primero entorno, recompensa y datos en lugar de añadir otro algoritmo. La comparación debe estandarizar implementación, búsqueda de hiperparámetros y semillas porque la varianza de deep RL puede distorsionar conclusiones entre algoritmos ([Henderson et al., 2018](https://doi.org/10.1609/aaai.v32i1.11694)).

**Evidencia de cierre.** Gate inicial: DQN aprende en la demo controlada y se contrasta con baselines simples. Gate opcional: archivo `algorithm_selection.csv` con curvas por interacción y métricas por réplica; si este archivo no se produce por priorización, PPO queda explícitamente como trabajo futuro y no como superioridad demostrada.

### Gap 6 — PSI, KS, C2ST y AUC no identifican por sí solos la causa del drift

**Problema.** PSI y KS son principalmente univariados; C2ST detecta separabilidad multivariada entre dos muestras de `X`, pero no prueba concept drift; AUC y log-loss requieren etiquetas y muestran degradación sin identificar automáticamente la causa. Además, `has_covariate_signal` y `has_concept_signal` se construyen actualmente a partir del tipo verdadero del escenario, lo que filtra la respuesta correcta al agente.

**Solución propuesta.** Brigitte empezará con **PSI como único diagnóstico de data drift** para el MVP. Se medirá si separa `no drift` de los escenarios de covariate shift relevantes para la política. Solo si falla ese criterio se añadirá, de una en una y con una hipótesis explícita, una técnica de la siguiente lista:

- numéricas: KS más tamaño de efecto o Wasserstein;
- categóricas: chi-cuadrado, Jensen-Shannon o Hellinger;
- multivariadas: C2ST y una alternativa como MMD, cuyo test compara distribuciones mediante *maximum mean discrepancy* ([Gretton et al., 2012](https://www.jmlr.org/papers/v13/gretton12a.html));
- supervisadas: AUC, PR-AUC, log-loss, Brier y métricas por subgrupo;
- secuenciales: ADWIN sobre error o log-loss, evaluado por falsos positivos y retraso de detección ([Bifet y Gavaldà, 2007](https://epubs.siam.org/doi/10.1137/1.9781611972771.42)).

C2ST se mantendrá como test aprendido, pero con validación fuera de muestra, permutaciones y reporte de incertidumbre, acorde con su formulación original ([Lopez-Paz y Oquab, 2017](https://arxiv.org/abs/1610.06545)). Los umbrales se calibrarán con ventanas sin drift; se controlará el problema de múltiples pruebas. El estado de DARL usará scores observables, disponibilidad de etiquetas e incertidumbre, nunca el tipo real del inyector.

**Evidencia de cierre.** Gate inicial: curva y umbral de PSI con falsos positivos en `no drift` y sensibilidad por severidad. Gate ampliado, solo si es necesario: tabla de FPR, potencia, retraso, costo y redundancia de los detectores añadidos y selección del vector mediante ablación.

### Gap 7 — Etiquetas retrasadas y factibilidad de las acciones

**Problema.** La tesis reconoce que las etiquetas pueden llegar tarde, pero el entorno ofrece señales conceptuales y recuperaciones como si siempre estuvieran disponibles. Tampoco se representa si hay suficientes nuevas etiquetas para actualizar el modelo.

**Solución propuesta.** En el MVP se asumirá disponibilidad inmediata de etiquetas y se declarará esa simplificación. Después de validar el entorno básico, se añadirán `label_availability`, edad de las etiquetas, tamaño de muestra etiquetada y máscara de acciones factibles. Recién entonces se evaluarán retrasos de 0, 1, 3 y 5 ventanas.

**Evidencia de cierre.** Experimentos por nivel de retraso con utilidad, regret, falsas alarmas y decisiones inválidas; ninguna acción usa etiquetas futuras.

### Gap 8 — Generalización limitada a dos datasets y mezcla de shift real con drift sintético

**Problema.** Hospital Readmission y PhysioNet son insuficientes para sostener generalización amplia. Además, TableShift ofrece particiones de dominio reales, mientras que el estudio actual mezcla esas particiones con inyección sintética sin separar claramente qué fuente de shift se evalúa.

**Solución propuesta.** Activar esta ampliación solo cuando exista un resultado positivo o un resultado negativo estable y explicado en el primer dataset. Entonces se definirán dos ejes:

- **shift natural:** usar `train`, `id_test` y `ood_test` de TableShift sin inyección;
- **drift controlado:** construir secuencias sintéticas sobre una partición fijada, con mecanismo y severidad conocidos.

El tercer dataset recomendado es `college_scorecard`: es público, pertenece a educación, tiene alrededor de 125 mil observaciones y un shift por tipo de institución, por lo que añade diversidad de dominio y tamaño sin el costo de los datasets ACS de más de un millón de filas. TableShift documenta los identificadores y splits oficiales, incluidos `diabetes_readmission`, `physionet` y `college_scorecard` ([repositorio oficial de TableShift](https://github.com/mlfoundations/tableshift)). La evaluación final será *leave-one-dataset-out*: entrenar la política con dos datasets y evaluar sin ajuste en el tercero.

**Evidencia de cierre.** Resultados separados por dataset, tipo de shift y protocolo; caída de generalización con intervalos de confianza, incluso si DARL no supera los baselines.

### Gap 9 — Falta de reproducibilidad experimental y fuente única de código

**Problema.** No hay pruebas automatizadas ni métricas crudas en `outputs/metrics/`. Conviven `code/drift_framework` y `code/src/darl`, con loaders, pipelines y monitores diferentes. Los notebooks contienen simulaciones y no deben ser la fuente de los resultados finales.

**Solución propuesta.** Durante la exploración, exigir solo trazabilidad mínima: semilla 42, configuración junto al resultado y salidas fuera del notebook. No se invertirá tiempo todavía en CI, refactors amplios ni una API definitiva. Cuando el protocolo produzca resultados interpretables, se declarará `code/src/darl` como paquete oficial, se migrará únicamente lo validado y se crearán runners bajo `code/experiments` con artefactos primero en `outputs/`.

**Evidencia de cierre.** Un comando reproduce un experimento pequeño desde cero; CI ejecuta pruebas unitarias y de integración; cada tabla o figura de tesis apunta a CSV/JSON y configuración de origen.

### Gap 10 — Incoherencias y afirmaciones prematuras en la tesis

**Problema.** Se detectaron los siguientes puntos:

- `Resumen`, `Abstract`, `Recomendaciones` y `Anexos` conservan texto de plantilla.
- El Capítulo IV afirma que XGBoost realiza una “normalización interna”; la documentación del *tree booster* describe árboles y umbrales, no un preprocesamiento automático que justifique A3 ([documentación oficial de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)).
- La metodología escrita para concept drift no coincide con `DriftInjector`.
- Se llama “recuperación” a una razón de AUC que representa rendimiento retenido.
- La tabla de literatura marca todas las capacidades de DARL como cumplidas aunque varias solo están diseñadas o simuladas.
- El POMDP escrito asume un *belief state* normal sin que la implementación estime dicho belief.
- La entrada BibTeX de TableShift registra 2024, mientras el repositorio y la publicación oficial la sitúan en NeurIPS 2023; se debe verificar y unificar el metadato.
- `main.tex` solo incluye a Jeffrey como autor; debe confirmarse con la asesora si Brigitte será coautora formal antes de cambiar la portada.

**Solución propuesta.** No priorizar capítulos incompletos durante los gates experimentales, salvo correcciones necesarias para un checkpoint solicitado. Mantener desde ahora tres etiquetas epistemológicas —**implementado**, **validado en simulación** y **validado empíricamente**— y ejecutar la actualización integral de resultados, abstract y conclusiones después de congelar la campaña experimental.

**Evidencia de cierre.** Tesis sin placeholders, afirmaciones trazables a resultados, bibliografía consistente y `latexmk` exitoso sin referencias indefinidas.

## 4. Tablero semanal

| Semana | Task | Experiment | Resultados Esperados | Done |
|---|---|---|---|---|
| Semana 3 | **Jeffrey:** preparar el *spike* de compatibilidad Stage 1–Stage 2 y una demo DQN v0 con recompensa simple.<br>**Brigitte:** construir una secuencia mínima de `diabetes_readmission`, documentar el inyector actual y calcular PSI como señal inicial. | Entrenar DQN en un caso controlado con la recompensa AUC–costo y compararlo con una política aleatoria; ejecutar una prueba ancla con Stage 1 congelado vs reajustado y una ventana `no drift` vs covariate shift. | Curva/log de entrenamiento real, protocolo de compatibilidad con fecha límite W4, primer flujo de ventanas y PSI; separación explícita entre sanity check y evidencia final. | DQN completa entrenamiento y evaluación en episodios separados; existen criterios medibles para decidir A2 y Brigitte entrega datos/PSI consumibles por el entorno. |
| Semana 4 | **Jeffrey:** cerrar el contrato Stage 1–Stage 2 y conectar DQN con resultados mínimos del pipeline.<br>**Brigitte:** completar la evidencia técnica y bibliográfica sobre transformaciones compatibles y estabilizar la interfaz de ventanas/PSI. | Comparar A1, A2-refit y A2c-correctiva con Logistic Regression y XGBoost; entrenar DQN sobre el entorno v0 usando outcomes del pipeline o una tabla empírica trazable. | Decisión **mantener, redefinir o retirar** `Update features`; demo de un agente que aprende; espacio de acciones configurable y no fijado en los objetivos. | La decisión sobre A2 queda registrada y DQN supera la política aleatoria en el caso controlado; si no aprende, el gate sigue abierto y desplaza las tareas opcionales posteriores. |
| Semana 5 | **Jeffrey:** iniciar el entorno secuencial v1 con transiciones dependientes de la acción.<br>**Brigitte:** implementar un primer concept drift condicionado por variables de `diabetes_readmission`. | En paralelo, probar que `Defer` acumula degradación y que una actualización cambia el estado futuro; generar drift abrupto y gradual con dos severidades y un control de ruido de etiquetas. | Trayectorias mínimas del entorno y *dataset card* del inyector con mecanismo, supuestos y variables afectadas. | Cada frente ejecuta sin depender de la implementación final del otro; las transiciones y severidades pueden inspeccionarse con semillas fijas. |
| Semana 6 | **Jeffrey:** integrar el entorno v1 con la interfaz de datos y retirar señales oráculo del estado observado.<br>**Brigitte:** calibrar PSI en `no drift` y covariate shift; validar por separado el concept drift. | Ejecutar secuencias compartidas con estado latente, observaciones ruidosas y acciones simuladas; medir falsos positivos de PSI y degradación de AUC por severidad. | MVP integrado de un dataset, JSON por ventana y evidencia de qué información observa realmente el agente. | El agente no recibe el tipo verdadero de drift; PSI tiene un umbral inicial y concept drift muestra degradación sin depender de una señal fabricada en `X`. |
| Semana 7 | **Jeffrey:** implementar el conjunto de acciones aprobado en W4 y medir recuperación/costo reales.<br>**Brigitte:** auditar que ventanas, etiquetas y métricas no usen información futura. | Ejecutar cada acción disponible sobre los mismos escenarios de `diabetes_readmission`, registrando AUC antes, bajo drift y postacción, además de tiempo y RAM. | Tabla empírica acción–escenario que servirá como referencia y alimentará el entorno. | Todas las acciones válidas producen outcomes reales y comparables; las inválidas están retiradas o enmascaradas y cada fila conserva trazabilidad mínima. |
| Semana 8 | **Jeffrey:** validar la recompensa simple y entrenar DQN con los outcomes reales.<br>**Brigitte:** construir el baseline de umbral fijo y revisar la interpretación de recuperación y costo. | Comparar DQN contra tabla de decisión empírica, umbral fijo y política aleatoria; probar pocos valores predefinidos del peso de costo solo si la recompensa v0 aprende. | AUC recuperada, utilidad, retorno, costo y regret sobre episodios comunes; primera evidencia positiva o negativa interpretable. | La evaluación usa una sola definición de utilidad; `best_action_share` no se presenta como exactitud y no se hace una búsqueda extensa antes de superar el sanity check. |
| Semana 9 | **Jeffrey:** reforzar la dinámica secuencial y añadir máscaras básicas de factibilidad.<br>**Brigitte:** decidir con evidencia si PSI es suficiente; añadir un solo detector adicional únicamente si falla. | Comparar política miope vs DQN cuando el drift persiste o crece; medir el aporte de PSI y, si corresponde, PSI + KS/Wasserstein o C2ST. | Evidencia de necesidad —o no— de decisiones temporales y decisión documentada sobre el vector mínimo de monitoreo. | Diferir cambia la trayectoria acumulada; el monitor se congela con PSI si basta o la incorporación de otra métrica demuestra una mejora medible. |
| Semana 10 | **Jeffrey:** ejecutar robustez y ablaciones mínimas de estado, recompensa y transición.<br>**Brigitte:** repetir severidades/semillas y auditar falsos positivos, resultados negativos y supuestos de datos. | Repetir DQN y baselines en cinco flujos derivados de la semilla 42; retirar una familia de señales a la vez y contrastar entorno inmediato vs secuencial. | IC 95 %, dispersión, tasas de decisión óptima, regret y diagnóstico de qué componentes aportan valor. | Existe un resultado estable, positivo o negativo, que permite decidir si el núcleo está listo para ampliar alcance. |
| Semana 11 | **Jeffrey:** aplicar el gate condicional: si el núcleo funciona, comparar PPO con DQN; si no, corregir la principal causa del fallo.<br>**Brigitte:** auditar equidad de la comparación o ejecutar el experimento correctivo de datos/monitor. | Con gate positivo, usar los mismos episodios y presupuesto para PPO y DQN; con gate negativo, repetir el experimento que aísla entorno, recompensa o generación de datos. | `algorithm_selection.csv` o informe de causa raíz con nueva evidencia; nunca una comparación añadida solo por completar el cronograma. | PPO solo se conserva si la comparación fue habilitada y trazable; si no se habilita, queda declarado como trabajo posterior sin afirmar superioridad. |
| Semana 12 | **Jeffrey:** adaptar la configuración al segundo dataset y, solo como *stretch*, a un tercero.<br>**Brigitte:** preparar mecanismos de drift y ficha de dominio para PhysioNet; asumir `college_scorecard` únicamente si hay capacidad. | Separar shift natural TableShift de drift controlado y probar transferencia de la política sin ajuste entre los datasets disponibles. | Matriz de generalización con al menos dos datasets si el gate W10 fue superado; en caso contrario, una réplica confirmatoria en el dataset principal. | No se afirma generalización con un solo dataset y ningún dataset adicional desplaza la corrección de un fallo central. |
| Semana 13 | **Jeffrey:** congelar el runner y ejecutar la campaña final aprobada.<br>**Brigitte:** auditar calidad, faltantes y trazabilidad, y consolidar literatura directamente relacionada con los resultados. | Repetir la matriz final de dataset, drift, severidad, política y semilla; reejecutar únicamente combinaciones fallidas justificadas. | CSV/JSON crudos, agregados, configuraciones, figuras candidatas y registro de hardware/software. | Toda tabla agregada se rastrea a resultados crudos; el código se limpia y automatiza solo hasta el nivel necesario para reproducir la campaña aprobada. |
| Semana 14 | **Jeffrey:** actualizar metodología, resultados, limitaciones y conclusiones con la evidencia congelada.<br>**Brigitte:** actualizar estado del arte, citas y discusión de monitoreo; realizar revisión cruzada. | Regenerar tablas/figuras desde `outputs/`, verificar BibTeX y compilar LaTeX completo. | Borrador PFC2 sin afirmaciones prematuras, con resultados negativos incluidos y distinción entre diseño, simulación y validación empírica. | `latexmk` termina sin errores ni referencias indefinidas; cada cifra tiene fuente y cada afirmación bibliográfica tiene cita válida. |
| Semana 15 | **Jeffrey:** preparar demo y narrativa de defensa.<br>**Brigitte:** preparar síntesis de resultados, riesgos, preguntas esperables y control final de entregables. | Reejecutar un caso end-to-end y ensayar la defensa centrada en compatibilidad, entorno, generación de datos, recuperación/costo y límites de generalización. | Release candidata de PFC2, informe final, presentación, demo corta y backlog residual. | Una tercera persona reproduce el caso mínimo y el equipo responde cada pregunta central mostrando evidencia y limitaciones, no solo métricas internas de RL. |

## 5. Reglas operativas del tablero

### Reparto de responsabilidades

- **Jeffrey** mantiene propiedad sobre el frente de RL: arquitectura, contrato entre etapas, espacio de acciones, entorno, recompensa y DQN; la comparación con PPO es condicional.
- **Brigitte** mantiene propiedad sobre el frente de datos: generación e inyección de drift, PSI/monitoreo, fichas de datasets, estado del arte y auditoría de resultados.
- Ambos frentes acuerdan una interfaz mínima de ventanas, observaciones, acciones y outcomes; no esperan a que el otro alcance su versión final para avanzar.
- Ambos revisan el entregable de la otra persona antes de marcar `Done`; la propiedad no elimina la revisión cruzada.

### Definición transversal de Done

Una semana solo se cierra si:

1. existe un artefacto inspeccionable —script, CSV/JSON, curva, matriz o nota de decisión— aunque todavía no esté empaquetado;
2. se registra el procedimiento mínimo para repetirlo localmente, sin exigir automatización completa durante la exploración;
3. el resultado incluye una validación proporcional al riesgo;
4. se registran resultados negativos y limitaciones;
5. el avance puede resumirse en una oración y hasta tres bullets para la reunión semanal.

### Cadencia ágil recomendada

- **Inicio de semana, 30 minutos:** seleccionar máximo dos entregables críticos y confirmar dependencias.
- **Mitad de semana, 20 minutos:** demostrar el artefacto parcial; si no existe evidencia ejecutable, reducir alcance.
- **Fin de semana, 45 minutos:** revisión cruzada, cierre de `Done` y preparación de respuestas “Where were we last week?” y “What progress has been made?”.
- Limitar trabajo en progreso a una tarea técnica y una tarea de investigación por persona.
- Para reuniones con expertos, formular preguntas específicas sobre recompensa, entorno, datos reales/sintéticos y criterios de acción; primero confirmar si su experiencia de RL es comparable con DARL.

## 6. Métricas por fase para PFC2

### MVP de semanas 3–4

- Entrenamiento completado, retorno de evaluación y comparación DQN vs política aleatoria.
- AUC antes y después de la acción, costo temporal y recompensa simple.
- Acuerdo de predicción/AUC en el conjunto ancla para compatibilidad y PSI en `no drift` vs covariate shift.

Las métricas siguientes se incorporan gradualmente después de superar el MVP; no deben bloquear la primera demo de aprendizaje.

### Resultado predictivo

- AUC-ROC, PR-AUC, F1, log-loss y Brier.
- AUC bajo drift, AUC postacción y pérdida de AUC recuperada.
- Métricas por subgrupo cuando el dataset tenga dominios o grupos relevantes.

### Calidad de decisión

- `optimal_action_rate`.
- `mean_regret` y regret acumulado.
- Matriz de confusión entre acción elegida y acción oráculo.
- Frecuencia de acciones inválidas o tomadas sin etiquetas suficientes.

### Secuencialidad

- Retorno medio y dispersión por semilla.
- Pérdida predictiva acumulada por episodio.
- Costo acumulado y número de reentrenamientos completos evitados.
- Diferencia frente a política miope y contextual bandit.

### Monitoreo

- Tasa de falsos positivos en `no drift`.
- Potencia por tipo y severidad.
- Retraso de detección y tasa de falsas alarmas.
- Tiempo/RAM del detector y aporte incremental al desempeño de la política.

### Reproducibilidad

- Semilla maestra 42 para toda la reproducibilidad; cinco flujos hijos deterministas derivados de ella para las réplicas finales y el mismo flujo 42 para *smoke tests*.
- Intervalos de confianza del 95 % y tamaños de efecto.
- Versiones de datos, dependencias, configuración y hardware.
- Artefactos crudos antes de tablas o figuras de tesis.

## 7. Orden recomendado para cerrar los gaps

1. **Gate 0 — Compatibilidad, cierre W4:** mantener, redefinir o retirar `Update features`; no congelar todavía el número de acciones.
2. **Gate 1 — Aprendibilidad mínima, cierre W4:** DQN entrena con recompensa simple y supera una política aleatoria en un caso controlado.
3. **Gate 2 — Dos frentes paralelos, W5–W6:** entorno secuencial v1 y drift condicionado por dataset se ejecutan independientemente y luego se integran.
4. **Gate 3 — Evidencia central, W7–W9:** acciones reales, PSI mínimo y comparación DQN contra tabla empírica y umbral fijo.
5. **Gate 4 — Robustez, W10:** varias semillas y ablaciones determinan si existe un resultado estable, incluso si es negativo.
6. **Gate 5 — Ampliación condicional, W11–W12:** PPO y generalización se abren solo si no desplazan la corrección de un fallo central.
7. **Gate 6 — Consolidación PFC2, W13–W15:** reproducibilidad del protocolo aprobado, campaña final, redacción y defensa.

## 8. Fuentes nuevas recomendadas para la revisión bibliográfica

- Zhao, H. et al. (2019). [On Learning Invariant Representations for Domain Adaptation](https://proceedings.mlr.press/v97/zhao19a.html). Fundamenta por qué alinear representaciones no garantiza compatibilidad predictiva.
- Sugiyama, M. et al. (2007). [Covariate Shift Adaptation by Importance Weighted Cross Validation](https://www.jmlr.org/beta/papers/v8/sugiyama07a.html). Alternativa principiada a reajustar transformadores bajo covariate shift.
- Gretton, A. et al. (2012). [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html). Base para evaluar MMD como detector multivariado.
- Lopez-Paz, D. y Oquab, M. (2017). [Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545). Base formal de C2ST y su validación fuera de muestra.
- Bifet, A. y Gavaldà, R. (2007). [Learning from Time-Changing Data with Adaptive Windowing](https://epubs.siam.org/doi/10.1137/1.9781611972771.42). Base de ADWIN y evaluación en secuencias.
- Gardner, J. et al. (2023). [TableShift](https://github.com/mlfoundations/tableshift). Fuente de datasets, identificadores y splits reales de dominio.
- Watkins, C. J. C. H. y Dayan, P. (1992). [Q-learning](https://doi.org/10.1007/BF00992698). Garantías y supuestos del algoritmo tabular que delimitan su uso como baseline en un estado continuo.
- Mnih, V. et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). Base de DQN, *experience replay* y red objetivo para acciones discretas.
- van Hasselt, H., Guez, A. y Silver, D. (2016). [Deep Reinforcement Learning with Double Q-Learning](https://doi.org/10.1609/aaai.v30i1.10295). Alternativa para reducir la sobreestimación de valores de DQN.
- Hausknecht, M. y Stone, P. (2015). [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527). Referencia para evaluar memoria cuando las métricas o etiquetas llegan con retraso.
- Henderson, P. et al. (2018). [Deep Reinforcement Learning That Matters](https://doi.org/10.1609/aaai.v32i1.11694). Sustento para múltiples semillas y comparación estandarizada de algoritmos deep RL.

Estas referencias deben agregarse a `thesis/referencias.bib` solo cuando se incorporen afirmaciones concretas a la tesis y después de verificar los metadatos completos.
