# TableShift: investigación a fondo para DARL

**Referencia base:** Gardner, J., Popović, Z. & Schmidt, L. (2023). *Benchmarking Distribution Shift in Tabular Data with TableShift*. NeurIPS 2023 Datasets & Benchmarks Track. arXiv:2312.07577. Repos: github.com/mlfoundations/tableshift · tableshift.org. Afiliaciones: University of Washington + Allen Institute for AI (AI2).

---

## 1. Qué es y por qué existe

TableShift es un **benchmark de distribution shift para datos tabulares**: 15 tareas de clasificación binaria, cada una con un shift asociado, accesibles vía una API de Python. Nace porque, a diferencia de visión y lenguaje —donde abundan benchmarks de shift (WILDS, DomainBed)—, **no existía un benchmark de calidad para shift en datos tabulares**, pese a que lo tabular es dominante en aplicaciones reales (finanzas, salud, política pública) y usa modelos distintos (árboles boosteados vs. redes). Motivos que hacen especial al caso tabular, según los autores: (a) sigue en debate si el deep learning supera a XGBoost/LightGBM/CatBoost incluso sin shift; (b) las features son estructuradas y heterogéneas (numéricas + categóricas + faltantes), no señales crudas; (c) el preprocesamiento es distinto y su impacto poco entendido; (d) los datasets tabulares de calidad son difíciles de acceder (datos personales, no "scrapeables").

**Criterios de selección de las tareas** (§3.1): open source con diccionario de datos; datos reales (nada simulado); ≥3 features y ≥1000 observaciones; features de tipos mixtos; tarea binaria significativa; y —clave— que un baseline tabular bien tuneado muestre un **shift gap estadísticamente significativo** (ΔAcc ≠ 0). Es decir, TableShift selecciona *a propósito* datasets donde el shift degrada el rendimiento.

---

## 2. Las 15 tareas

El **shift gap** ΔAcc = Acc(OOD) − Acc(ID) es el que mide un XGBoost/LightGBM tuneado (negativo = cae en OOD). "Domain Generalization" ✓ = hay ≥2 subdominios de entrenamiento (permite métodos que requieren etiqueta de dominio). 10 de las 15 son domain-generalization; 5 no.

| Tarea | Objetivo (target) | Variable de shift (dominio) | Dom. Gen. | Baseline gap ΔAcc |
|---|---|---|:---:|---:|
| ASSISTments | Respuesta correcta siguiente | Escuela | ✓ | −34.49 % |
| College Scorecard | Baja tasa de graduación | Tipo de institución (Carnegie) | ✓ | −11.16 % |
| ICU Hospital Mortality | Muerte del paciente en el hospital | Tipo de seguro | ✓ | −6.30 % |
| **Hospital Readmission** | **Readmisión ≤30 días (diabéticos)** | **Fuente de admisión** | ✓ | **−5.94 %** |
| Diabetes | Diagnóstico de diabetes | Raza/etnia | ✓ | −4.48 % |
| ICU Length of Stay | Estancia ≥3 días en UCI | Tipo de seguro | ✓ | −3.39 % |
| Voting | Votó en elección presidencial EE.UU. | Región geográfica | ✓ | −2.58 % |
| Food Stamps | Recibió cupones de alimentos | Región geográfica | ✓ | −2.39 % |
| Unemployment | Desempleo (adultos no jubilables) | Nivel educativo | ✓ | −1.28 % |
| Income | Ingreso ≥ 56k | Región geográfica | ✓ | −1.25 % |
| FICO HELOC | Repago de línea de crédito hipotecaria | Nivel de riesgo de terceros | | −22.58 % |
| Public Health Insurance | Cobertura pública (bajos ingresos) | Situación de discapacidad | | −14.46 % |
| **Sepsis** | **Aparición de sepsis en próximas 6 h** | **Duración de estancia (LOS)** | | **−6.05 %** |
| Childhood Lead | Plomo en sangre sobre referencia CDC | Nivel de pobreza (PIR) | | −5.12 % |
| Hypertension | Diagnóstico de hipertensión (50+) | Categoría de IMC | | −4.36 % |

Dominios cubiertos: finanzas, educación, política pública, salud, participación cívica. Fuentes crudas: encuestas (ACS, BRFSS, NHANES, ANES), registros clínicos (MIMIC-III, PhysioNet), UCI, Kaggle, FICO.

---

## 3. Las dos tareas que usas en DARL (en detalle)

Tu Tabla 3.1 elige estas dos precisamente para cubrir **dos regímenes tabulares distintos** (categórico vs. numérico), lo cual es una decisión metodológica sólida y defendible.

### 3.1. "diabetes readmission" = **Hospital Readmission** (régimen categórico)
- **Objetivo:** predecir si un paciente diabético es readmitido dentro de los 30 días de su alta (tu binario "<30 días" vs. "NO").
- **Fuente:** dataset UCI *Diabetes 130-US hospitals* (Strack et al., 2014): 10 años (1999–2008) de atención clínica en 130 centros de EE.UU., >50 features (raza, sexo, edad, tipo de admisión, tiempo en hospital, especialidad, número de pruebas de laboratorio, resultado HbA1c, diagnósticos, medicaciones, visitas previas, etc.). Predomina lo categórico/discreto → encaja con tu régimen "categóricas y discretas".
- **Cómo se define el shift:** *domain split por "fuente de admisión"* (21 fuentes distintas: derivación médica, transferencia de hospital, etc.). Tras barrer valores, usan **"emergency room" (sala de emergencias) como dominio held-out (OOD)**, dejando las otras 20 fuentes como entrenamiento. Es el mayor |D_train| del benchmark (20 subdominios).
- **Magnitud:** shift gap ΔAcc = **−5.94 %**. Métricas de shift: covariate Δx (OTDD) = 42.37; concepto Δy|x (FDD) = 1.30; etiqueta Δy = 0.0060.
- **Acceso:** público (UCI), sin credencial.

### 3.2. "physionet" = **Sepsis** (régimen numérico)
- **Objetivo:** predecir, a partir de datos finos de UCI (mediciones de laboratorio, sensores, demografía), si el paciente desarrollará **sepsis dentro de las próximas 6 horas**.
- **Fuente:** *PhysioNet/Computing in Cardiology Challenge 2019* (Reyna et al., 2019): registros de UCI de **>60,000 pacientes** de dos hospitales, hasta **40 variables clínicas por hora** de estancia. Predomina lo numérico fisiológico → encaja con tu régimen "numéricas fisiológicas". El dataset se arma desde **>40,000 archivos**.
- **Cómo se define el shift:** probaron partir por *hospital* pero **NO producía shift gap** en los baselines tuneados (dato relevante para ti). En su lugar usan **"duración de estancia" (length of stay)**: entrenan con pacientes con ≤47 h en UCI y evalúan con >47 h (47 h = percentil 80). Simula un modelo entrenado en estancias cortas y aplicado a estancias largas.
- **Magnitud:** shift gap ΔAcc = **−6.05 %**. Métricas de shift: covariate Δx (OTDD) = 6609.73; concepto Δy|x (FDD) = 8.44; etiqueta Δy = 0.0040.
- **Acceso:** requiere **credencialización** (PhysioNet). NO es tarea de domain-generalization (un solo dominio de entrenamiento).

---

## 4. Cómo se construye el shift (metodología) — y por qué NO es concept drift temporal

**Modelo formal (§2.1):** cada ejemplo es (x, y, d) donde d es el **dominio**. Los datos son una mezcla de dominios; el entrenamiento y el test usan **conjuntos de dominios distintos** (D_train ≠ D_test), y ahí surge el shift: P_train(x,y) ≠ P_test(x,y). Ese shift se descompone en covariate (cambia p(x)), label (cambia p(y)) y concept (cambia p(y|x)). En la práctica es una **mezcla desconocida de los tres**.

**El mecanismo es un "domain split", no una línea de tiempo.** Se elige una variable categórica (fuente de admisión, LOS, región, raza, tipo de seguro…) y se aparta uno o varios de sus valores como dominio OOD (held-out), entrenando con el resto. Cuando no hay un split obvio, barren cada valor (train en D∖d, test en d) y eligen el que induce mayor gap real y con relevancia del mundo real (§E.1). **Importante para DARL:** esto es **domain/subpopulation shift estático**, NO un stream temporal con concept drift de inicio/tipo conocido. TableShift *no* provee ground truth del momento del drift ni un régimen de concept drift controlado.

**Particiones:** dentro de cada dominio hay validación y test propios; el tuning de hiperparámetros usa accuracy de validación **in-domain**; se reporta accuracy de test ID y OOD (nunca accuracy de train). 100 trials de HyperOpt por modelo.

---

## 5. Cómo TableShift mide covariate / concept / label shift

Advertencia central de los autores (§E.2), textual: *"It is not possible to measure the true shifts for any given dataset… from a finite sample."* Por eso proponen tres **aproximaciones**:

- **Covariate shift Δx = OTDD** (Optimal Transport Dataset Distance, aproximación gaussiana; Alvarez-Melis & Fusi 2020). Distancia entre los conjuntos ID y OOD en el espacio de features.
- **Concept shift Δy|x = FDD** (*Frechet Dataset Distance*, métrica que ellos introducen, inspirada en la Frechet Inception Distance): entrena un clasificador en el dominio fuente, toma las activaciones intermedias de un MLP y calcula la distancia Wasserstein-2 entre distribuciones de esas activaciones.
- **Label shift Δy** = diferencia L2 de las tasas base (proporción de positivos) entre dominios.

**Hallazgos sobre el shift (§5, muy útiles para tu defensa):**
- El **shift gap se correlaciona con el label shift Δy** (ρ ≈ 0.71–0.73). Una regresión de accuracy OOD sobre [accuracy ID, Δy] da R² = 0.996. Los cuatro outliers de mayor gap (Public Coverage, HELOC, ASSISTments, College Scorecard) son los de mayor label shift.
- **Covariate Δx y concept Δy|x están fuertemente correlacionados entre sí** (ρ = 0.99), y los cambios en las predicciones OOD se relacionan con cambios en p(x) → sugiere que buena parte del gap no explicado por [ID, Δy] es covariate, no concept.
- Es decir: **el propio benchmark líder de shift tabular trata covariate/label/concept como entrelazados y difíciles de separar desde muestras finitas.** Esto respalda que DARL inyecte cada tipo por separado, con ground truth conocido.

---

## 6. Modelos evaluados y hallazgos principales

**19 métodos en 5 familias (§4.1):**
- *Baselines:* MLP, **XGBoost, LightGBM, CatBoost** (los boosteados siguen siendo estado del arte tabular).
- *Redes tabulares:* SAINT, TabTransformer, NODE, FT-Transformer, ResNet tabular.
- *Robustez de dominio:* DRO (χ² y CVaR), Group DRO.
- *Robustez a label shift:* Label Group DRO, Adversarial Label DRO.
- *Domain generalization:* DANN, IRM, MixUp, VREx, DeepCORAL, MMD (requieren etiqueta de dominio y ≥2 subdominios → solo aplicables a las 10 tareas domain-gen).

**Tres hallazgos (§5):**
1. **Accuracy ID y OOD están linealmente correlacionadas** (ρ = 0.81) a través de tareas y modelos. Mejorar ID tiende a mejorar OOD. (Antes se sabía en visión/QA, no en tabular.)
2. **Ningún modelo supera consistentemente a los baselines** XGBoost/LightGBM/CatBoost, ni en rendimiento general ni en robustez.
3. **Los métodos de robustez/domain-generalization reducen el gap solo bajando el accuracy ID, no subiendo el OOD** (dos rectas paralelas desplazadas). Y el gap se explica sobre todo por **label shift**, que los métodos de robustez a label shift *no* corrigen (a veces empeoran ID y OOD).

---

## 7. API y acceso a los datos

- **Paquete de Python** que construye cada tarea desde sus fuentes públicas crudas en pocas líneas; documenta cada feature y su codificación; incluye preprocesamiento estándar (one-hot y label encoding, escalado/binning de numéricas, manejo de faltantes). Salidas nativas: **PyTorch DataLoaders, Pandas DataFrames y Ray Datasets**. Se recomienda **imagen Docker** por las dependencias.
- **TableShift NO aloja los datos**: cada fuente es pública, pero algunas requieren **acceso credencializado** (rápido y abierto al público). De tus dos: **Hospital Readmission = acceso público (UCI)**; **Sepsis = acceso credencializado (PhysioNet)**. Otras credencializadas: MIMIC (ICU LOS, ICU Mortality), ANES (Voting), FICO (HELOC).

---

## 8. Implicaciones y limitaciones para DARL

**Por qué TableShift encaja como fuente, pero necesitas inyectar drift sintético encima:**
- TableShift aporta **datasets reales de alta calidad, documentados y con shift real** por dominio → buena base empírica y citable.
- Pero su shift es **estático y por subpoblación/dominio**, NO un **concept drift temporal con inicio/tipo/severidad conocidos**. TableShift no ofrece: (a) un régimen de concept drift controlado, (b) ground truth del momento del drift, ni (c) acciones de mantenimiento. Para evaluar de forma **supervisada** si tu agente PPO elige la acción correcta (defer / actualizar preprocesamiento / actualizar modelo / reentrenar todo), necesitas conocer el tipo y severidad del drift → de ahí tu **inyección sintética controlada** (mezcla Beta numérica, remuestreo marginal categórico, inversión de relación para concept drift). Esto no es un parche: es lo que exige la evaluación supervisada (ver también el censo del turno anterior).
- **Las dos tareas cubren dos regímenes** (Hospital Readmission = categórico/discreto; Sepsis = numérico fisiológico), lo que te permite comprobar si las acciones se comportan distinto según el drift afecte frecuencias categóricas o distribuciones numéricas.

**Matices que conviene declarar en la tesis (honestidad epistémica):**
- El shift nativo de tus dos tareas es principalmente **covariate/label**, no concept: los valores FDD (concepto) son bajos (Hospital Readmission 1.30; Sepsis 8.44) frente a OTDD (covariate) grandes. Por eso el **concept drift lo aportas tú sintéticamente**.
- En Sepsis, partir por *hospital* no generaba gap; el shift efectivo viene de la duración de estancia. Si tu inyección de drift ignora la partición de dominio nativa, decláralo.
- TableShift advierte que **no se puede medir el shift verdadero desde muestra finita**; sus métricas (OTDD/FDD/Δy) son aproximaciones exploratorias, no causales. Cítalo como respaldo de por qué el drift sintético con ground truth es metodológicamente preferible para evaluar.

**Benchmarks relacionados (para contextualizar la elección):**
- **WILDS** (Koh et al., ICML 2021): domain generalization + subpopulation shift, mayormente imagen/texto.
- **Wild-Tab** (Kolesnikov 2023): OOD en **regresión** tabular.
- **Shifts / Shifts 2.0** (Malinin et al.): shift real multimodal; 2 de 5 tareas tabulares.
- **Wild-Time**: shift **temporal** (más cercano a concept drift que TableShift, por si quieres citar la diferencia).
- TableShift ha sido usado/citado por trabajos de adaptación tabular como *Drift-Resilient TabPFN*, *AdapTable* (test-time adaptation) y *TabFSBench* (feature shift), útiles como antecedentes.

---

### Cita sugerida
> Gardner, J., Popović, Z., & Schmidt, L. (2023). Benchmarking Distribution Shift in Tabular Data with TableShift. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track.* arXiv:2312.07577.

Fuentes originales de tus dos tareas (cítalas también):
> Strack, B. et al. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. *BioMed Research International.* (UCI *Diabetes 130-US hospitals 1999–2008*.)
> Reyna, M. A. et al. (2019). Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Computing in Cardiology (CinC).*
