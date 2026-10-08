# Guion de presentación — DARL

**Bloques:** Problemática · Objetivos · Diseño de la solución actual · Limitaciones
**Duración sugerida:** 15–18 min · **16 diapositivas** + referencias
**Fuente:** tesis completa (`thesis/secciones/*.tex`, `thesis/tables/*.tex`,
`thesis/referencias.bib`) y notas de `plan/TableShift_investigacion_DARL.md`.

> Convención de citas: numeración IEEE **propia del PPT** (ver lista al final).
> Cada dato numérico lleva su cita en la misma diapositiva, en pie de lámina,
> con letra de 10–12 pt. Los resultados propios se citan como
> *"Elaboración propia — Tabla X de la tesis"*.

---

## BLOQUE 1 — PROBLEMÁTICA (diapositivas 1–5)

### Diapositiva 1 — Portada
- **Título:** DARL: Aprendizaje por Refuerzo con Diagnóstico de Drift para la
  Actualización Selectiva de Pipelines Tabulares de ML de Dos Etapas
- Autores, asesora (Ariana Villegas), UTEC – Ciencia de la Computación, 2026.
- Visual: `thesis/figures/generated/darl_infografia.png` en pequeño, como fondo/lateral.

---

### Diapositiva 2 — Gancho: los modelos se degradan en producción
**Texto en lámina (frase grande + 3 cifras):**

> "Un modelo que funcionaba bien en entrenamiento puede fallar en producción
> **sin que cambie una sola línea de su código**." [7]

| Cifra | Qué significa | Cita |
|---|---|---|
| **15 / 15** tareas tabulares | En las 15 tareas de TableShift, el rendimiento cae al pasar de datos "conocidos" a datos fuera de distribución | [1] |
| **18 modelos** evaluados | XGBoost, LightGBM, redes profundas, transformers tabulares… **ninguno cierra la brecha** de forma consistente | [1] |
| **hasta −34.5 %** de accuracy | Caída máxima observada (ASSISTments); en finanzas −22.6 % (FICO HELOC) | [1] |

**Visual:** gráfico de barras horizontal con el *shift gap* ΔAcc de las 15
tareas de TableShift, resaltando en color las dos usadas en DARL
(Hospital Readmission −5.94 %, Sepsis −6.05 %). Datos en
`plan/TableShift_investigacion_DARL.md` §2, originales en [1].

**Notas del orador:** los modelos asumen que la distribución de datos es
estable; en la realidad cambian demografía, comportamiento y contexto [2],[7].
Esto se llama *distribution shift*.

> 💡 **Si quieres una cifra tipo "el 58 % de…":** la tesis no tiene una
> estadística global de ese tipo. La más usada en la literatura es
> **"91 % de los modelos de ML se degradan con el tiempo"** de Vela et al.
> (2022), *Scientific Reports* [14]. **No está en `referencias.bib`**:
> verifícala en el paper y agrégala a la bibliografía antes de usarla.

---

### Diapositiva 3 — ¿Por qué importa? Dominios críticos
**Texto en lámina:**
- Los pipelines tabulares se usan en **salud, finanzas y políticas públicas** [1].
- **Sepsis (UCI):** un modelo entrenado con estancias ≤ 47 h pierde **6.05 %**
  de accuracy al aplicarse a estancias largas [1]. Datos de **> 60 000
  pacientes** de UCI [12].
- **Readmisión hospitalaria (diabetes):** pierde **5.94 %** al cambiar la
  fuente de admisión a sala de emergencias [1]. 10 años de datos de
  **130 hospitales** de EE. UU. [13].
- Un modelo de mortalidad que se degrada en silencio **compromete la asignación
  de recursos clínicos** (Introducción de la tesis).

**Visual:** dos tarjetas (Sepsis / Readmisión) con ícono, tamaño del dataset y caída.

---

### Diapositiva 4 — El problema real: ¿QUÉ actualizar?
**Texto en lámina:**
- Un pipeline tabular tiene **dos etapas**: preprocesamiento *f_φ* → modelo *g_θ*.
- Dos tipos de drift dañan etapas distintas [3],[7]:
  - **Covariate shift** — cambia P(X) → se desactualiza el **preprocesamiento**.
  - **Concept drift** — cambia P(Y|X) → se desactualiza el **modelo**.
- La respuesta típica es **reentrenar todo**, lo cual es costoso [2].
- **Pero:** si solo falló una etapa, actualizar esa etapa puede bastar a una
  fracción del costo.

**Visual:** diagrama `Datos → [Stage 1: preprocesamiento] → [Stage 2: XGBoost] → Predicción`,
con un rayo rojo sobre cada etapa etiquetado "covariate" / "concept".
Ecuación pequeña: P(X,Y) = P(X)·P(Y|X).

**Dato de apoyo citable:** Shakhovska y Pukach muestran que solo
transformar con cuantiles (sin reentrenar) baja el estadístico KS de
**0.0559 a 0.0072** [5] → una acción parcial *sí* puede corregir drift.

---

### Diapositiva 5 — Brecha en la literatura
**Visual principal:** la Tabla de comparación (`thesis/tables/comparacion.tex`) con ✓/✗:

| Trabajo | Tipo drift | Severidad | 2 etapas | Recuperación | Costo | RL |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| DISDE [3] | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| SHIFT [4] | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| TableShift [1] | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| CARA [2] | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| Severity-Aware [5] | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ |
| Self-Healing [6] | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| **DARL** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Mensaje en una línea:** *Diagnostican el drift o deciden cuándo reentrenar,
pero todos tratan el pipeline como una sola pieza.*

**Notas del orador (frases citables textuales):**
- Shakhovska y Pukach reconocen como trabajo futuro "adaptar no solo a la
  severidad sino también al tipo de drift" [5].
- Tanna propone como dirección futura integrar **aprendizaje por refuerzo** en
  la remediación [6].
- CARA usa reglas de umbral y es agnóstico al tipo de drift [2].

---

### Diapositiva 6 — Pregunta de investigación
> ¿Es posible aprender, mediante **aprendizaje por refuerzo**, una política que
> a partir de señales de monitoreo elija la **actualización selectiva** que
> **maximiza la recuperación** del rendimiento al **menor costo computacional**?

**Visual:** las 4 acciones como íconos — ⏸ Defer · 🔧 Update features ·
🧠 Update model · 🔄 Retrain all.

---

## BLOQUE 2 — OBJETIVOS (diapositivas 7–8)

### Diapositiva 7 — Objetivo general
**Diseñar, implementar y evaluar DARL**, un framework de actualización selectiva
para pipelines tabulares de dos etapas que aprende una política mediante RL
sobre un **POMDP** [8] formulado a partir del **tipo y severidad** del drift, y
cuantifica la **recuperación** y el **costo** de cada acción.

### Diapositiva 8 — Objetivos específicos (con estado actual)
| # | Objetivo | Herramienta | Estado |
|---|---|---|---|
| OE1 | Módulo de monitoreo que detecta/clasifica drift y su severidad | PSI, KS, C2ST + severity score [5] | 🟡 Parcial |
| OE2 | Formalizar la decisión como POMDP con 4 acciones y recompensa AUC–costo | POMDP [8] | 🟢 Hecho |
| OE3 | Entrenar agente PPO sobre episodios de drift sintético en 2 datasets de TableShift | PPO [9], [1] | 🟡 Demo |
| OE4 | Evaluar la política vs. tabla de decisión empírica | AUC-ROC, costo relativo | 🟡 Solo PhysioNet |

> Ajuste sugerido por la asesora (asesoría semana 3): redactar OE1 sin la frase
> "constituyen el vector de observación del agente", porque es una
> consecuencia y no parte del objetivo.

---

## BLOQUE 3 — DISEÑO DE LA SOLUCIÓN ACTUAL (diapositivas 9–13)

### Diapositiva 9 — Arquitectura DARL: ciclo de 6 módulos
**Visual:** `thesis/figures/manual/metodologia_darl.tex` (compílala a PNG) o
`darl_infografia.png`.

1. Datos TableShift + pipeline base (XGBoost [11], regresión logística como línea base)
2. Inyección sintética de drift (covariate / concept / combinado, severidad γ ∈ [0,1])
3. Monitoreo (PSI, KS, C2ST, ΔAUC, Δℓ)
4. Agente RL observa *o_t* y elige acción
5. Se ejecuta la acción y se mide AUC, tiempo y memoria
6. La recompensa entrena a PPO

---

### Diapositiva 10 — Datos e inyección de drift
| Dataset | Rol | Variables dominantes | Cita |
|---|---|---|---|
| `diabetes_readmission` | Readmisión ≤ 30 días | Categóricas y discretas | [1],[13] |
| `physionet` | Sepsis en próximas 6 h | Numéricas fisiológicas | [1],[12] |

**Mecanismos de drift (una línea cada uno):**
- **Numérico:** mezcla Beta–Bernoulli → cada valor se reemplaza con prob. γ.
- **Categórico:** p_γ(c) = (1−γ)·p₀(c) + γ·q(c) → remuestreo por frecuencias.
- **Concept (PhysioNet):** inversión parcial alrededor de la mediana, intensidad γ.
- Severidades usadas: **leve 0.25 · moderada 0.60 · severa 0.90**.

---

### Diapositiva 10b — Investigación: ¿cómo inyectar concept drift *real*?
**Mensaje:** iteramos el inyector hasta que cambiara P(Y|X) sin tocar P(X).
*Elaboración propia — notebooks `physionet_drift_injector`, `diabetes_drift_injector`.*

| Versión | Qué hace | Problema / resultado |
|---|---|---|
| v1 Label flipping (p = 0.45·γ) | Voltea Y al azar | Es **ruido**, no un cambio de regla; AUC cae a ≈0.50 ya con γ = 0.1 |
| v2 Inversión por mediana (la que describe la tesis) | Modifica X | En realidad mueve P(X): es covariate disfrazado |
| v3 `logit_shift` | Amplifica la relación aprendida | **No degrada** el AUC (0.622 → 0.625): un modelo que ya conoce la relación sigue ordenando bien |
| v4 `logit_reweight` | Rota los pesos: γ = 0.5 sin relación, γ = 1 invertida | ✅ Diabetes: AUC 0.650 → 0.537 de forma ordenada |
| v4b `…_recentered` | + desplaza el intercepto (eventos raros) | ✅ Necesario en sepsis (1.2 % de prevalencia) |

Validación: **KS = PSI = 0** en todas las severidades → el drift es puramente conceptual.

**Hallazgo con acciones (concept puro, PhysioNet, γ = 1.2):** A1, A2 y A2c dan
**0.346** (idénticas), mientras que A3 y A4 dan **0.663**. Esto confirma de
forma empírica la taxonomía [7]: corregir X no sirve contra concept drift.

**Drift combinado:** en PhysioNet, **A3 sola le gana a A4** (0.672 frente a 0.550 con γ = 1.2).
En Diabetes **no se replica**: todas las acciones quedan a ~1 punto de A1.
La acción ideal depende del dataset, y eso motiva aprenderla (RL) en lugar de fijarla con una regla.

---

### Diapositiva 11 — El agente: POMDP + PPO
**Texto en lámina:**
- **Parcialmente observable:** el agente no ve P(X) ni P(Y|X), solo métricas
  estimadas sobre ventanas finitas [8].
- **Observación *o_t*:** severidad estimada, PSI/KS/C2ST, caída de AUC/F1, costos previos.
- **Acciones:** Defer · Update features (*f_φ*) · Update model (*g_θ*) · Retrain all.
- **Recompensa:**
  r = λ_A·(AUC_acción − AUC_drift) − λ_C·(½·tiempo_rel + ½·memoria_rel)
  con λ_A = 1.00, λ_C = 0.25.
- **¿Por qué PPO?** espacio discreto pequeño (|A| = 4), estable gracias al
  *clipping*, reutiliza cada rollout [9],[10].

**Config. demo:** SB3 `MlpPolicy`, γ = 0.95, rollout 64, batch 32, 5 épocas,
episodios de 100 pasos, semilla 42 (Tabla de hiperparámetros de la tesis).

---

### Diapositiva 12 — Resultado 1: reentrenar todo NO siempre es lo óptimo
**Visual:** tabla de recuperación de AUC (PhysioNet, XGBoost, AUC base = 0.7836)
— *Elaboración propia, Tabla "Baseline empírico XGBoost" de la tesis*.

| Condición | Sev. | A1 Defer | A2 Feat. | A3 Model | A4 All | Óptima |
|---|---|---|---|---|---|---|
| Covariate | Leve | 96.6 | 95.4 | 98.0 | 98.1 | **A1** |
| Covariate | Moderada | 90.6 | 91.8 | 96.6 | 96.7 | **A3** |
| Covariate | Severa | 84.9 | 89.7 | 96.1 | 95.7 | **A3** |
| Concept | Severa | 86.3 | 82.6 | 100.0 | 99.7 | **A3** |
| Cov.+Concept | Severa | 80.8 | 88.4 | 95.7 | 96.2 | **A3** |

Costos relativos: A1 = 0.00x · A2 = 0.12x · A3 = 0.45x · A4 = 1.00x.

**Mensaje grande:** *Covariate moderado: A3 recupera 96.6 % vs. 96.7 % de A4,
pero con **55 % menos costo** (0.45x vs 1.00x).*

---

### Diapositiva 13 — Resultado 2: demo PPO
**Visual:** `ppo_reward_episodio.pdf` + `action_distribution.pdf` (o
`cumulative_step_reward_summary.pdf`).

- Evaluación: **50 episodios × 100 pasos = 5 000 decisiones**.
- Reward medio/paso **0.0334**; reward medio/episodio **3.339 ± 0.416**.
- Política final: **Update features 59.86 %**, **Update model 40.14 %**,
  Defer 0 %, Retrain all 0 %.
- Lectura: con la recompensa definida, el agente **prefiere acciones parciales**
  frente a los extremos.

*Elaboración propia — Cap. "Resultados preliminares" de la tesis.*

---

## BLOQUE 4 — LIMITACIONES (diapositivas 14–15)

Separa **limitaciones de alcance** (decisiones de diseño) de **limitaciones que
encontramos** (lo que salió en la práctica). El jurado valora esa honestidad.

### Diapositiva 14 — Limitaciones de alcance (por diseño)
- Solo **clasificación binaria tabular en modo batch**; no streaming ni drift gradual.
- **2 datasets** y **1 modelo principal** (XGBoost) → la generalización requiere más validación.
- Se asume **disponibilidad de etiquetas** en el dominio objetivo; en la
  realidad puede haber *delayed feedback* [7].
- Drift **sintético y controlado**; TableShift ofrece shift de dominio estático,
  no temporal [1].
- Sin despliegue en producción ni integración con Airflow/MLflow.

### Diapositiva 15 — Limitaciones que tuvimos (hallazgos)
| Limitación | Evidencia en la tesis |
|---|---|
| Demo PPO con escenarios **sintéticos, deterministas y livianos**: valida el flujo, no la convergencia ni la superioridad | Cap. Resultados, "Lectura preliminar" |
| Los pasos del episodio son **independientes** → se parece más a un bandido contextual; falta dependencia temporal que justifique RL frente a un clasificador supervisado | Conclusiones, "Trabajo futuro" |
| PPO **nunca eligió Defer ni Retrain all** → posible sesgo de la recompensa (λ_C) | Cap. Resultados |
| **Update features (A2) no siempre ayuda**: cambia la representación y desalinea al modelo | Tabla baseline (Concept moderado: A2 = 82.8 vs A1 = 92.7) |
| Resultados empíricos **solo en PhysioNet**; falta `diabetes_readmission` | Cap. Resultados |
| El concept drift fue difícil de simular: el label flipping es ruido y la amplificación no degrada. Se resolvió con `logit_reweight`, pero **la tesis aún describe la versión anterior** | Notebooks `*_drift_injector`; asesoría semana 3 |
| En Diabetes, bajo drift combinado, **ninguna acción recupera** (todas a ~1 punto de A1) | `compatibility_combined_drift_diabetes` |
| Bugs encontrados: `DriftInjector` mueve variables con muchos ceros incluso con γ = 0; `make_logreg` da AUC ≈ 0.51 sin drift | Notebooks de compatibilidad |
| Falta comparar con baselines simples (siempre reentrenar, umbrales, clasificador supervisado) | Conclusiones |

### Diapositiva 16 — Próximos pasos
1. Correr el flujo completo en ambos datasets, con varias semillas.
2. Comparar contra baselines: siempre reentrenar, siempre diferir, umbrales, clasificador supervisado.
3. Sanity check con **DQN** antes de PPO (recomendación de la asesora).
4. Entorno secuencial: historial de acciones, costos acumulados, persistencia del drift.
5. Gráficos costo agregado vs. desempeño agregado.

---

### Diapositiva 17 — Referencias (IEEE, numeración del PPT)

[1] J. Gardner, Z. Popović y L. Schmidt, "Benchmarking Distribution Shift in Tabular Data with TableShift," en *Advances in Neural Information Processing Systems*, 2024. doi: 10.48550/arXiv.2312.07577.
[2] A. Mahadevan y M. Mathioudakis, "Cost-aware retraining for machine learning," *Knowledge-Based Systems*, vol. 293, p. 111610, 2024. doi: 10.1016/j.knosys.2024.111610.
[3] T. T. Cai, H. Namkoong y S. Yadlowsky, "Diagnosing Model Performance Under Distribution Shift," arXiv:2303.02011, 2023.
[4] H. Singh *et al.*, "Who experiences large model decay and why? A Hierarchical Framework for Diagnosing Heterogeneous Performance Drift," en *Proc. 42nd ICML*, 2025.
[5] K. Shakhovska y P. Pukach, "Severity-Aware Drift Adaptation for Cost-Efficient Model Maintenance," *AI*, vol. 6, no. 11, art. 279, 2025. doi: 10.3390/ai6110279.
[6] R. K. Tanna, "Self-Healing ML Pipelines: Automating Drift Detection and Remediation in Production Systems," *Preprints.org*, 2025 (sin revisión por pares). doi: 10.20944/preprints202510.2522.v1.
[7] J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy y A. Bouchachia, "A survey on concept drift adaptation," *ACM Computing Surveys*, vol. 46, no. 4, 2014. doi: 10.1145/2523813.
[8] L. P. Kaelbling, M. L. Littman y A. R. Cassandra, "Planning and acting in partially observable stochastic domains," *Artificial Intelligence*, vol. 101, pp. 99–134, 1998.
[9] J. Schulman, F. Wolski, P. Dhariwal, A. Radford y O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
[10] R. S. Sutton y A. G. Barto, *Reinforcement Learning: An Introduction*, 2.ª ed. MIT Press, 2018.
[11] T. Chen y C. Guestrin, "XGBoost: A Scalable Tree Boosting System," en *Proc. 22nd ACM SIGKDD*, 2016, pp. 785–794.
[12] M. A. Reyna *et al.*, "Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019," *Critical Care Medicine*, vol. 48, no. 2, pp. 210–217, 2020. **⚠ No está en `referencias.bib`: verificar y agregar.**
[13] B. Strack *et al.*, "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records," *BioMed Research International*, 2014. **⚠ No está en `referencias.bib`: verificar y agregar.**
[14] D. Vela *et al.*, "Temporal quality degradation in AI models," *Scientific Reports*, vol. 12, 11654, 2022. **⚠ Opcional, no está en `referencias.bib`: verificar y agregar.**

---

## ANEXO — Inconsistencias de la tesis que conviene corregir antes de exponer

El jurado puede preguntar por estos puntos:

1. **Resumen, Abstract y Recomendaciones están vacíos** (`resumen.tex`,
   `abstract.tex` sin texto; `recomendaciones.tex` = "Lorem ipsum").
2. **Autoría:** `main.tex` lista solo a Jeffrey Monja; el plan PFC2 incluye a
   Brigitte Dayane Rojas. Verificar quién figura en la portada.
3. **Explicación de A2 vs A3 (Cap. Resultados):** dice que "XGBoost realiza una
   normalización interna de los datos". XGBoost no normaliza; los árboles son
   invariantes a transformaciones monótonas por feature, y por eso reescalar
   (A2) aporta poco. Conviene reformular.
4. **El capítulo de metodología describe un concept drift desactualizado.** La
   tesis usa la inversión por mediana, que modifica X; el código ya usa
   `ConceptDriftInjector` (`logit_reweight`), que modifica solo Y. Hay que
   reescribir esa subsección con la nueva ecuación β_i(γ) = β_i·(1 − 2γ).
5. **Selección de la acción óptima en la tabla:** en *Concept moderado* se elige
   A1 (92.7) sobre A3 (96.6, costo 0.45x), y en *Concept leve* A2 (100.0). El
   valor de **λ_B** no se reporta; sin él, el jurado no puede reproducir la
   columna "Acción óptima". Reportarlo.
6. **Falta la fila "Covariate + Concept / Leve"** en la tabla baseline.
7. **OE3 y OE4 prometen ambos datasets y "generalizar a escenarios no vistos"**;
   los resultados actuales cubren solo PhysioNet y no hay prueba de
   generalización. En la PPT, presentarlo como "en progreso".
8. **Severidad:** el severity score [5] combina KS, Wasserstein y Jensen–Shannon;
   en la tesis se adapta como PSI + KS + C2ST. Aclarar que es una adaptación propia.
