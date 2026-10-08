# Investigación para la revisión de tesis — DARL

**Respuestas a los tres frentes que pidió Osman (reunión del 2 de septiembre)**

Contexto: DARL propone un agente PPO que, sobre un pipeline tabular de dos etapas
(fϕ = preprocesamiento, gθ = modelo XGBoost), elige entre cuatro acciones —Defer,
Update features, Update model, Retrain all— ponderando recuperación de AUC contra
costo (tiempo + memoria). La novedad de la Tabla 1.1 es la columna "política
aprendida (RL)" combinada con la granularidad de dos etapas. Los tres frentes
apuntan, cada uno, a proteger esa novedad.

---

## Frente 2 — Vertex AI / Azure ML / SageMaker (la amenaza más seria a la novedad)

**Lo que las tres plataformas SÍ hacen (idéntico patrón):**

- **Detectan** skew (entrenamiento vs. servicio) y drift por feature, con métricas
  estadísticas por columna, y clasifican por *tipo de señal* (calidad de datos,
  calidad de modelo/predicción, y en SageMaker/Azure también drift de la
  importancia/atribución de features).
- **Alertan** cuando una métrica cruza un **umbral fijo** que tú configuras
  (Cloud Monitoring en GCP, CloudWatch en AWS, Event Grid en Azure).
- **Pueden disparar** un reentrenamiento, pero solo si tú cableas la orquestación
  (Cloud Functions + Vertex Pipelines; EventBridge/Lambda/Step Functions +
  SageMaker Pipelines; Event Grid + Azure ML Pipelines).

**Lo que NINGUNA hace (= tu gap):**

- No **aprenden** una política de decisión. La "lógica de decisión" es una función
  que el ingeniero escribe a mano (umbrales, reglas if/else).
- El reentrenamiento que disparan es **completo** (reajustan y redespliegan el
  pipeline entero), no una elección granular entre actualizar solo fϕ, solo gθ, o
  todo.
- No ponderan **costo de cómputo** dentro de la decisión de qué acción tomar.
- No adaptan la frontera de decisión según *tipo × severidad* del drift.

**Evidencia clave (citable):**

- "Estas herramientas automatizan monitoreo, alertas y, en algunos casos,
  reentrenamiento. Las decisiones de reentrenamiento se guían por umbrales."
  (Logz.io, *AI Model Drift*, dic-2025).
- "Vertex Model Monitoring detecta drift. No reentrena. El reentrenamiento
  automático corre por tu cuenta: la alerta dispara un Pipeline solo si lo
  cableaste así." (análisis de patrones de producción de Vertex, may-2026).
- Azure: la remediación programática vía Event Grid consiste en "correr un pipeline
  de ML para reentrenar el modelo y redesplegarlo" — es decir, reentrenamiento
  completo (Microsoft Learn, *Monitor model performance in production*, v2).
- SageMaker Model Monitor: cuatro tipos de monitoreo (data quality, model quality,
  bias drift, feature-attribution drift); alertas por umbral vía CloudWatch;
  acciones remediadoras (retrain, actualizar preprocesamiento) orquestadas por el
  usuario con Step Functions/Lambda (AWS docs y whitepapers).

**Nota de vigencia:** El antiguo "Data drift (preview)" de Azure ML (SDK v1) se
retiró el 1-sep-2025 y fue reemplazado por Model Monitor (v2). Conviene citar la
versión v2 para no quedar desactualizado.

**Frase de defensa (para cuando el jurado pregunte "¿por qué no usar Vertex?"):**

> Las plataformas gestionadas (Vertex AI, Azure ML, SageMaker) detectan drift por
> feature y disparan, por umbral fijo, un reentrenamiento completo del pipeline que
> el propio ingeniero debe orquestar. DARL se diferencia en tres ejes: (1) aprende
> la política de decisión mediante RL en lugar de reglas de umbral fijas;
> (2) decide a nivel de *etapa* del pipeline (actualizar solo preprocesamiento,
> solo modelo, o todo), no como bloque monolítico; y (3) internaliza el trade-off
> rendimiento–costo dentro de la propia decisión. Es el gap de la Tabla 1.1
> trasladado de la academia a la industria.

**Recomendación operativa de Osman:** que Brigitte, desde el correo de su startup,
haga una consulta general de preventa a un representante de Google Cloud en Lima
("estamos evaluando migrar a Vertex; ¿cómo detecta drift, qué recomienda ante
drift, y con qué volumen/frecuencia conviene?"). Sirve para tener una respuesta con
fuente directa si tus datos internos no son públicos.

---

## Frente 3 — ¿PPO otorga explicabilidad? (lo que Osman pidió traer mañana)

**Respuesta corta: no, PPO no da atribución nativa de features.**

- PPO es un método de *policy gradient*: optimiza los parámetros de una red contra
  una recompensa. No produce ninguna descomposición de la decisión por variable de
  entrada (no hay `feature_importances_` ni equivalente).
- En Stable-Baselines3, para un espacio de observación 1D como el tuyo, la
  `MlpPolicy` por defecto es una red totalmente conectada de **2 capas de 64
  unidades con activación tanh**, con redes separadas para actor (pi) y crítico
  (vf). (SB3 docs, *Policy Networks / Custom Policy*).
- Tu intuición en la reunión fue correcta: como la política moderna es un MLP, la
  pregunta se reduce a "¿una red neuronal da atribución?", y la respuesta es no de
  forma nativa.

**Matiz a tu favor (que conviene explotar en el Capítulo IV):** tu vector de
observación tiene ~6 dimensiones con nombre semántico (PSI, KS, C2ST, severity
score S_X, ΔAUC, Δloss), no es un tensor de píxeles. Con pocas entradas
interpretables y cuatro acciones, la atribución **post-hoc** es barata y muy
convincente.

**Herramientas post-hoc citables:**

1. **Mapa de decisión de la política** (la más convincente para un jurado). Fijas
   cuatro métricas en su valor de referencia, barres dos (p. ej. PSI y ΔAUC) en una
   malla, y coloreas por argmax de la acción. Sale la frontera literal entre
   "actualizar features" y "actualizar modelo". Responde de forma visual la
   pregunta de Osman (¿pesa más PSI o C2ST?). Es exactamente lo que hacen los
   trabajos de XRL en entornos de baja dimensión (p. ej. mapas de regiones de
   acción sobre el espacio de estado en CartPole).
2. **Importancia por permutación** sobre la observación: permutas una métrica (p.
   ej. PSI) en los episodios de evaluación y mides cuánto cambia la distribución de
   acciones.
3. **Destilación de política** a un árbol de decisión superficial ajustado a los
   pares (o_t, a_t) de la política entrenada. Da reglas explícitas del tipo
   "si PSI > x y ΔAUC < y → Update model". Literatura:
   - **VIPER** — Bastani, Pu, Solar-Lezama, *Verifiable Reinforcement Learning via
     Policy Extraction*, NeurIPS 2018 (arXiv:1805.08328). [Referencia verificada.]
   - **DTPO** — *Optimizing Interpretable Decision Tree Policies for RL*, 2024
     (arXiv:2408.11632). Optimiza árboles interpretables directamente y compara
     contra VIPER y PPO.
4. **Atribución tipo SHAP / Shapley por feature de observación:**
   - **SVERL-P** — Beechey et al. 2023: descompone el *rendimiento* del agente en
     contribuciones por feature de estado. Responde literalmente "¿de qué features
     depende más el agente?".
   - *From Explainability to Interpretability* (arXiv:2501.09858, 2025): extrae
     políticas interpretables a partir de valores Shapley para RL on-policy (PPO) y
     off-policy.
   - Survey de referencia: *Explainable Deep Reinforcement Learning: State of the
     Art and Challenges*, ACM Computing Surveys, 2022.

**Cómo escribirlo (siguiendo la instrucción de Osman), repartido en dos lugares:**

- En **2.7 (PPO)**, la frase seca, sin herramientas externas:
  > PPO parametriza el actor y el crítico mediante redes neuronales densas y, por su
  > naturaleza de método de policy gradient, no provee atribución nativa de las
  > variables de observación; a cambio, ofrece buen desempeño en la mayoría de
  > escenarios de decisión secuencial.
- La atribución **post-hoc** va en el **Capítulo metodológico / IV**, que es donde
  corresponde, como figura (el mapa de decisión) y opcionalmente como trabajo
  citando VIPER/SVERL-P. Así cumples lo que pidió Osman ("di si lo hace o no, sin
  mencionar herramientas externas" en 2.7) sin dejar el hueco.

---

## Frente 1 — El censo (concept drift vs. covariate shift en datos tabulares)

**Hallazgo:** el concept drift real y *verificado* en datasets tabulares públicos
es escaso. La mayoría de lo que la literatura llama "datasets de concept drift" son
(a) generadores sintéticos, o (b) datasets reales donde el drift se asume pero nunca
se verificó su momento ni su tipo.

**El caso emblemático — Electricity (Elec2):**
- Žliobaitė (2013), *How good is the Electricity benchmark for evaluating concept
  drift adaptation* (arXiv:1301.3524): un predictor ingenuo que repite la última
  etiqueta alcanza ~85% de accuracy, no por adaptación al drift sino por
  autocorrelación temporal de las etiquetas (picos cada 48 instancias = 24 h).
- *Evaluation methods... for classification of streaming data with temporal
  dependence* (Machine Learning Journal, 2014): de 18 accuracies publicadas sobre
  Electricity, solo 6 superaron ese baseline ingenuo ("Persistent classifier").
- Bifet et al., *Classifier Concept Drift Detection and the Illusion of Progress*:
  refuerza que buena parte del "progreso" reportado era ilusorio por ignorar la
  dependencia temporal.

**El survey canónico:** Lu et al., *Learning under Concept Drift: A Review*
(arXiv:2004.05785). Distingue datasets sintéticos (reglas predefinidas, drift
conocido) de reales, y enumera las **dos limitaciones de los datasets reales**:
(1) no se conoce el inicio/fin exacto del drift, y (2) muchos mezclan varios tipos
de drift. Esto es exactamente el "no está tan demarcado" de Osman.

**La práctica estándar del campo = la tuya.** Los benchmarks recientes reconocen la
sobredependencia de generadores sintéticos y adoptan el enfoque **semi-sintético**:
inyectar cambios distribucionales *controlados y de punto conocido* sobre datasets
reales, para poder evaluar de forma supervisada preservando la complejidad real
(p. ej. *A Framework for Evaluating and Benchmarking Concept Drift Detection
Methods*, 2026, arXiv:2606.07789; y la revisión unificada arXiv:2505.17902). Eso es
precisamente lo que hace DARL (mezcla Beta para drift numérico, remuestreo marginal
para categórico, inversión de relación para concept drift).

**Sobre TableShift (base de tus dos datasets):** Gardner, Popović, Schmidt,
*Benchmarking Distribution Shift in Tabular Data with TableShift*, NeurIPS 2023
(arXiv:2312.07577). Son 15 tareas de clasificación binaria con un *domain/subpopulation
shift* asociado (finanzas, salud, educación, etc.), NO concept drift temporal. Es
decir: tus datasets traen un shift real *por dominio*, pero el **concept drift**
propiamente dicho es lo que inyectas sintéticamente. Por eso inyectar drift
sintético controlado no es una debilidad: es la única forma de tener puntos de drift
con ground truth conocido y de barrer severidad de manera reproducible.

**Cómo defenderlo (sirve en los dos resultados posibles del censo):**
- Si el censo confirma que hay covariate shift abundante pero concept drift real
  escaso/no verificado → tu justificación de por qué inyectas concept drift
  sintético controlado.
- Si aparece algún dataset con concept drift real → lo usas como validación externa.
  El censo te sirve en cualquier caso; solo cambia qué defiendes con él.

**Para la Tabla que pidió Osman** (tipo "de N datasets, X tienen covariate shift,
Y tienen concept drift, y de esos Z no está demarcado / es estacionario"): conviene
construirla ejecutando tu pipeline completo y reportando, por dataset, la
distribución general de los datos (no solo de tus sub-particiones). Osman recordó
que la *presentación de datos* es un criterio de evaluación con puntaje propio
(~3–4 puntos).

---

## Lista de referencias (para el marco teórico y la defensa)

**PPO / interpretabilidad de políticas:**
- Bastani, Pu, Solar-Lezama (2018). *Verifiable Reinforcement Learning via Policy
  Extraction* (VIPER). NeurIPS 2018. arXiv:1805.08328.
- *Optimizing Interpretable Decision Tree Policies for RL* (DTPO, 2024).
  arXiv:2408.11632.
- Beechey et al. (2023). SVERL-P — Shapley values para performance de RL por feature.
- *From Explainability to Interpretability* (2025). arXiv:2501.09858.
- *Explainable Deep Reinforcement Learning: State of the Art and Challenges*.
  ACM Computing Surveys, 2022.
- Stable-Baselines3 — documentación de *Policy Networks* (arquitectura por defecto
  MlpPolicy: 2×64, tanh).

**Concept drift / census:**
- Žliobaitė (2013). *How good is the Electricity benchmark...* arXiv:1301.3524.
- *Evaluation methods... temporal dependence*. Machine Learning Journal, 2014.
- Lu et al. (2019). *Learning under Concept Drift: A Review*. arXiv:2004.05785.
- *A Framework for Evaluating and Benchmarking Concept Drift Detection Methods*
  (2026). arXiv:2606.07789.
- Gardner, Popović, Schmidt (2023). *Benchmarking Distribution Shift in Tabular
  Data with TableShift*. NeurIPS 2023. arXiv:2312.07577.

**Plataformas cloud (para el planteamiento del problema / diferenciación):**
- Google Cloud — *Monitor feature skew and drift* (Vertex AI Model Monitoring docs).
- Microsoft Learn — *Model monitoring in production* / *Monitor model performance
  in production* (Azure ML v2).
- AWS — *SageMaker Model Monitor* (docs y AWS ML Blog); *ML Lens — Evaluate data
  drift* (Well-Architected).

---

## Cosas a verificar antes de citar (honestidad epistémica)

- Las capacidades de las nubes cambian rápido; conviene re-chequear la doc oficial
  de Vertex/SageMaker/Azure el mismo día de la defensa por si añadieron algo nuevo.
- La intuición de Osman ("Vertex recomienda qué acción tomar", 60% de confianza) no
  quedó respaldada en la documentación revisada: Vertex detecta y alerta, pero la
  recomendación/decisión de acción la implementa el usuario. Vale más confirmarlo
  tú con fuente antes de que lo confirme Osman.
- Verifica la cita exacta de VIPER en tu gestor bibliográfico (páginas 2499–2509 de
  NeurIPS 2018).
