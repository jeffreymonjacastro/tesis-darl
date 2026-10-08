# Censo de datasets tabulares públicos por tipo de distribution shift: covariate shift, concept drift real y datos estacionarios/autocorrelados

## TL;DR
- El concept drift **real y verificado** (cambio confirmado de P(Y|X) con punto y tipo conocidos) es **escaso** en datasets tabulares públicos: prácticamente ningún dataset "real" trae ground truth del momento/tipo de drift, por lo que la práctica estándar del campo es el enfoque **semi-sintético** (inyectar drift controlado sobre datos reales o usar generadores sintéticos) — esto justifica directamente que DARL inyecte drift sintético controlado.
- Los benchmarks tabulares grandes (TableShift, WILDS, Wild-Tab) miden mayoritariamente **domain/subpopulation shift** (dominado por covariate y label shift), NO concept drift temporal; los repositorios de streams (MOA/River, USP de Souza) aportan el drift verificable, pero solo en sus versiones sintéticas o semi-sintéticas.
- Varios datasets etiquetados como "concept drift" (Electricity/Elec2, Covertype, Poker Hand) son en realidad **autocorrelados / con dependencia temporal**, no drift verificado: en Electricity un predictor de persistencia ingenuo logra 85% y **solo 6 de 18** accuracies publicadas lo superaron (Žliobaitė et al. 2015).

## Key Findings

1. **La clasificación en la literatura es inconsistente y no permite un conteo "limpio".** No existe un censo canónico; hay que reconstruirlo cruzando surveys (Lu et al. 2019, IEEE TKDE; Gama et al. 2014, ACM CSUR), repositorios (MOA/River; USP de Souza et al. 2020) y benchmarks OOD (TableShift, WILDS, Wild-Tab). Reporto conteos por categoría con la advertencia explícita de que muchas atribuciones son heredadas de la costumbre, no verificadas.

2. **Sintéticos = donde el drift está verificado.** Los ~10 generadores sintéticos canónicos (SEA, SINE, Agrawal, Hyperplane, STAGGER, RandomRBF, RandomTree, Mixed, LED, Waveform) tienen punto y tipo de drift conocidos por construcción. La mayoría inyecta drift de tipo Source II (cambio de P(Y|X), "actual/real drift" en la taxonomía de Lu et al.); algunos (Hyperplane, RandomRBF, Circle) mezclan Source I+II.

3. **Reales = drift asumido, no verificado.** Los datasets reales frecuentes (Electricity, Covertype, Poker, Airlines, NOAA Weather, Sensor, Usenet, Spam, KDDCup99, GasSensor, Rialto, Posture, Insects) carecen de ground truth del punto/tipo de drift. Lu et al. (2019) lo declaran textualmente: *"the ground truth of precise start and end time of drifts is unknown"* y *"some real datasets may include mixed drift types"*.

4. **La excepción parcial: Insects (Souza et al. 2020).** Es el único conjunto real donde el drift tiene ground truth aproximado, porque se indujo controlando una variable oculta —la temperatura del trampa, ligada al ritmo circadiano de mosquitos *Culex quinquefasciatus*, medido por un sensor óptico de vuelo. Aún así, solo las variantes con drift **abrupto** tienen periodos estacionarios que permiten métricas de detección estándar.

5. **Los benchmarks OOD tabulares son covariate/label shift, no concept drift temporal.** TableShift (15 tareas binarias, 10 de domain generalization) usa splits por dominio (geografía, hospital, raza, etc.); su propio análisis reporta que *"change in the label distribution ∆y is correlated with shift gap"*, es decir el gap ID→OOD se relaciona sobre todo con **label shift**, no con concept shift. Wild-Tab (regresión) y WILDS son domain/subpopulation shift.

## Details

### 1. Conceptos formales y diagnóstico

**Descomposición canónica.** P(X,Y) = P(X)·P(Y|X) = P(Y)·P(X|Y). Sobre esto (Moreno-Torres et al. 2012, *Pattern Recognition*; Gama et al. 2014; Lu et al. 2019):

- **Covariate shift (virtual drift, Source I de Lu et al.):** cambia P(X), se mantiene P(Y|X). No mueve la frontera de decisión óptima. Formalmente Pₜ(X) ≠ Pₜ₊₁(X) con Pₜ(Y|X)=Pₜ₊₁(Y|X). Moreno-Torres añade dos condiciones: que la etiqueta esté causalmente determinada por los covariates y que P(Y|X) se mantenga.
- **Label / prior shift:** cambia P(Y) (y P(X|Y) se mantiene). Es "robusto" ante reducción de covariates; permite corrección por matriz de confusión / adjusted count (Saerens, Forman).
- **Concept drift real (actual drift, Source II):** cambia P(Y|X) — mueve la frontera de decisión y degrada al clasificador. Es el "concept shift" propiamente dicho.
- **Mixto (Source III):** cambian P(X) y P(Y|X) simultáneamente; es lo más común en datos reales.
- **Estacionariedad (sentido de series de tiempo):** propiedades estadísticas constantes en el tiempo. Un dataset puede NO ser estacionario por **autocorrelación / dependencia temporal de etiquetas** sin que haya drift real en P(Y|X) — este es el caso trampa de Electricity.

**Nota terminológica clave (razón central de que el conteo no sea limpio):** Lu et al. adoptan la convención (creciente en la literatura) de llamar "concept drift" a *cualquier* cambio en P(X,Y), incluyendo Source I (covariate). Gama et al. son estrictos: "real drift" = cambio en P(Y|X); "virtual drift" = cambio solo en P(X). Según qué convención se use, el mismo dataset se clasifica distinto.

**Diagnóstico en la práctica:**
- *Covariate shift:* tests de dos muestras sobre P(X) — Kolmogorov-Smirnov univariado por feature, MMD (Maximum Mean Discrepancy), Friedman-Rafsky, energy test, classifier two-sample tests (Lopez-Paz & Oquab), deep-kernel MMD (Liu et al. 2020); para alta dimensión, reducción previa (BBSD de Lipton et al. 2018). **No requieren etiquetas.**
- *Concept drift real:* requiere señal de error del clasificador — DDM (Gama et al. 2004), EDDM, ADWIN, Page-Hinkley, HDDM. **Requiere etiquetas verdaderas, a menudo con retardo.**
- *Por qué es difícil verificar concept drift real:* los métodos error-rate necesitan etiquetas inmediatas; en datos reales no hay ground truth del momento/tipo de drift; la detección por ventanas es "ill-posed" (el drift percibido es en parte artefacto del windowing — "Window Dilemma", Springer 2024); y no se puede medir el shift verdadero desde una muestra finita (TableShift lo dice explícitamente: *"It is not possible to measure the true shifts for any given dataset"*).

### 2. El problema de la autocorrelación: Electricity/Elec2 y otros

Žliobaitė (2013, arXiv:1301.3524) mostró que Electricity (45.312 instancias, 6–8 atributos, predicción UP/DOWN del precio en New South Wales, Australia) tiene **etiquetas fuertemente autocorreladas**. Cita verbatim: *"if the data was distributed independently, such a predictor would achieve 51% accuracy. However, if we test this naive approach on the Electricity dataset it gives much higher 85% accuracy… Autocorrelation peaks at every 48 instances (24 hours)."* El clasificador de clase mayoritaria (siempre DOWN) da 58%. En su tabla de MOA, solo LeveragingBag (88.6%) y AdaHoeffdingOptionTree (86.7%) superaron el baseline de persistencia (85.3%).

En el seguimiento (Žliobaitė et al. 2015, *Machine Learning* J., DOI:10.1007/s10994-014-5441-4) la evidencia es contundente, verbatim: *"Only 6 out of 18 reported accuracies outperformed a naive baseline Persistent classifier. This suggests that current evaluation and benchmarking practices need to be revised."* (Tabla 5).

Souza et al. (2020, *Data Mining and Knowledge Discovery*; arXiv:2005.00113) extienden el diagnóstico: Covertype (*"data are probably organized according to the geographical location"*) y Poker Hand también tienen dependencia temporal, y la versión de MOA de Poker está mal normalizada y reordenada respecto a la de UCI. El consenso del campo (Souza et al.; Pesaranghader & Viktor; Bifet & Gavaldà) es que *la ubicación y/o presencia de concept drift en Electricity, Forest Covertype y Poker-Hand es desconocida*.

**Implicación:** en estos datasets, un detector de cambio "inútil" que dispara alarmas al azar puede inflar la accuracy vía reinicio del modelo, por lo que alta accuracy NO evidencia buena adaptación a drift. Recomiendan siempre comparar contra el baseline de persistencia.

### 3. Censo / inventario consolidado de datasets

**Tabla A — Datasets SINTÉTICOS (drift verificado por construcción; Lu et al. 2019 Tabla 3, MOA/River).** #Insts. "Custom" = generado por parámetros.

| Dataset | #Insts. | #Attrs. | #Cls. | Tipo de drift | Source (P(X) vs P(Y\|X)) | ¿Verificado? |
|---|---|---|---|---|---|---|
| STAGGER | Custom | 3 | 2 | Súbito | II (P(Y\|X)) | Sí |
| SEA | Custom | 3 | 2 | Súbito | II | Sí |
| Rotating Hyperplane | Custom | 10 | 2 | Gradual; Incremental | II | Sí |
| Random RBF | Custom | Custom | Custom | Súbito; Gradual; Incremental | III (mixto) | Sí |
| Random Tree | Custom | Custom | Custom | Súbito; Reocurrente (o "None") | II | Sí |
| LED | Custom | 24 | 10 | Súbito | II | Sí |
| Waveform | Custom | 40 | 3 | Súbito | II | Sí |
| SINE | Custom | 2 | 2 | Súbito | II | Sí |
| Circle/Circles | Custom | 2 | 2 | Gradual | III | Sí |
| Rotating chessboard | Custom | 2 | 2 | Gradual | II | Sí |
| Agrawal (River/MOA) | Custom | 9 | 2 | Súbito/Gradual | II | Sí |
| Mixed | Custom | 4 | 2 | Súbito | II | Sí |

Notas: RandomTree/RTG a veces se usa como control sin drift ("None"). En River hay además Friedman/FriedmanDrift (regresión), STAGGER, Sine, LEDDrift, RandomRBFDrift, ConceptDriftStream.

**Tabla B — Datasets REALES (drift asumido, no verificado salvo indicación; Lu et al. 2019 Tabla 4, USP de Souza, usos frecuentes).**

| Dataset | #Insts. | #Attrs. | #Cls. | Dominio | Tipo de drift atribuido | ¿Verificado? | Estacionario/autocorrelado |
|---|---|---|---|---|---|---|---|
| Electricity/Elec2 | 45.312 | 8 | 2 | Precio eléctrico (NSW) | Temporal/estacional (asumido) | No | Sí, autocorrelado (persistencia=85%) |
| Covertype | 581.012 | 54 | 7 | Cobertura forestal | Geográfico (asumido) | No | Autocorrelado (orden geográfico) |
| Poker-Hand | 1.025.010 | 10 | 10 | Cartas | Categórico (asumido) | No | Dependiente del orden |
| Airlines | 539.384 | 7 | 2 | Retrasos de vuelos | Temporal-anual (asumido) | No | Probable |
| NOAA Weather | 18.159 | 8 | 2 | Meteorología | Temporal-anual (asumido) | No | Estacional |
| Sensor | 2.219.803 | 5 | 54 | Sensores interiores | Temporal-diario (asumido) | No | Sí |
| KDDCup'99 | 494.021 | 41 | 23 | Intrusión de red | Novel class (asumido) | No | Mixto |
| Usenet1 | 1.500 | 99 | 2 | Texto/newsgroups | Recurrente (por diseño) | Parcial (diseño) | — |
| Usenet2 | 1.500 | 99 | 2 | Texto/newsgroups | Recurrente (por diseño) | Parcial (diseño) | — |
| Email data | 1.500 | 913 | 2 | Texto/spam | Súbito (por diseño) | Parcial | — |
| Spam data | 9.324 | 499 | 2 | Filtrado spam | Gradual (asumido) | No | — |
| Spam Assassin corpus | 9.324 | 39.916 | 2 | Filtrado spam | Gradual (asumido) | No | — |
| ECUE drift 1 | 10.983 | 287.034 | 2 | Filtrado spam | (asumido) | No | — |
| ECUE drift 2 | 11.905 | 166.047 | 2 | Filtrado spam | Novel class (asumido) | No | — |
| Insects – Abrupt | 52.848 | 33 | 6 | Sensor óptico insectos | Abrupto (semi-inducido) | **Sí (variable oculta=temperatura)** | Periodos estacionarios entre drifts |
| Insects – Incremental | 57.018 | 33 | 6 | Sensor óptico insectos | Incremental | **Sí (semi)** | Sin periodos estacionarios |
| Insects – IncGradual | 24.150 | 33 | 6 | Sensor óptico insectos | Incremental-gradual | **Sí (semi)** | Sin periodos estacionarios |
| Insects – IncRecurrent | 79.986 | 33 | 6 | Sensor óptico insectos | Incremental-reocurrente | **Sí (semi)** | Sin periodos estacionarios |
| GasSensor | ~13.910 | 128 | 6 | Sensores de gas | Temporal (asumido) | No | Sí |
| Rialto | ~82.250 | 27 | 10 | Imágenes/edificios | Temporal (asumido) | No | Sí |
| Posture (Kaluža) | ~164.860 | ~5 | 11 | Sensores de postura | Temporal (asumido) | No | Sí |
| Powersupply | ~29.928 | 2 | 24 | Suministro eléctrico | Temporal-diario (asumido) | No | Sí |
| Luxembourg | ~1.900 | 32 | 2 | Encuesta social | (asumido) | No | — |

*El repositorio USP DS (Souza et al. 2020) contiene ~27 data streams en total (11 introducidos en el paper); las cuatro variantes Insects son las de ground truth semi-verificado.*

**Tabla C — Benchmarks OOD tabulares "modernos" (domain/subpopulation shift, NO concept drift temporal).**

| Benchmark | #Tareas/datasets | Tipo de shift | Naturaleza |
|---|---|---|---|
| TableShift (Gardner, Popović, Schmidt, NeurIPS 2023, arXiv:2312.07577) | 15 tareas binarias (10 de domain generalization) | Covariate + label + concept mezclados vía splits por dominio | Domain shift, no temporal; shift gap correlacionado con label shift ∆y |
| WILDS (Koh et al., ICML 2021, arXiv:2012.07421) | 10 datasets (mayormente imagen/texto; tabular: CivilComments es subpop.) | Domain generalization + subpopulation shift | Real, no tabular-céntrico |
| Wild-Tab (Kolesnikov 2023, arXiv:2312.01792) | 3 datasets industriales (clima, consumo eléctrico) | OOD en regresión tabular | Domain shift |
| Shifts / Shifts 2.0 (Malinin et al.) | incluye Weather tabular (Yandex) | Distribution shift + incertidumbre | Real |

Tareas de TableShift (todas con split por dominio, no por tiempo): Food Stamps (región geográfica), Income/ACS, Public Health Insurance (discapacidad), ACS Unemployment, ANES Voting, Diabetes (raza), Hypertension (pobreza), Hospital Readmission (fuente de admisión), Childhood Lead (pobreza/PIR), Sepsis (PhysioNet), ICU Length-of-Stay (edad), ICU Mortality (tipo de seguro), FICO HELOC, College Scorecard, ASSISTments.

### 4. La distribución cuantitativa (lectura del censo)

Con la advertencia de que **la literatura no permite un conteo limpio** (terminología inconsistente, atribuciones heredadas, tipos mezclados, ausencia de denominador canónico), la lectura consolidada es:

**(a) Datasets con drift verificado (punto/tipo conocido):** casi exclusivamente los **~10–12 generadores sintéticos** (SEA, SINE, Agrawal, Hyperplane, STAGGER, RandomRBF, RandomTree, Mixed, LED, Waveform, Circles) más el conjunto **Insects** semi-inducido (~4 variantes de ground truth).

**(b) Concept drift real (P(Y|X)) verificado en datos reales SIN inducción:** **≈ 0 datasets con ground truth de punto/tipo.** Lu et al. (2019) y la literatura de streams (Souza et al. 2020) coinciden: no hay dataset real público con las ubicaciones de drift claramente identificadas, y hay consenso de que la ubicación/presencia de drift en Electricity, Covertype y Poker es desconocida.

**(c) Concept drift asumido pero no verificado / estacional / autocorrelado:** la **mayoría** de los ~14–20 datasets reales del catálogo de streams (Electricity, Covertype, Poker, Airlines, NOAA, Sensor, Spam, Usenet, KDDCup99, GasSensor, Rialto, Posture, Powersupply, Luxembourg). Al menos 3 (Electricity, Covertype, Poker) tienen autocorrelación documentada que confunde la evaluación.

**(d) Covariate/feature shift + label shift (benchmarks OOD modernos):** las **15 tareas de TableShift** + Wild-Tab (3) + WILDS. Son domain/subpopulation shift, no drift temporal; TableShift reporta que el shift gap se relaciona fuertemente con label shift, no con concept shift.

**Lectura del "universo" habitual de la literatura de concept drift tabular (~30–35 datasets frecuentemente citados):** aproximadamente **40% sintéticos con drift inyectado verificado**, **~45–50% reales con drift asumido/no verificado (buena parte estacionaria o autocorrelada)**, y **efectivamente 1 conjunto (Insects) con drift real semi-verificado**. En paralelo existe el bloque de **~15 tareas OOD (TableShift) que son covariate/label shift por diseño, no concept drift temporal**. Estos porcentajes son cualitativo-cuantitativos, no un censo cerrado.

### 5. Por qué la práctica estándar es semi-sintética

Como la evaluación supervisada de detección/adaptación requiere conocer *cuándo* y *de qué tipo* es el drift, y los datos reales no lo proveen, el campo recurre a **inyección controlada**:
- **Generadores puramente sintéticos** (MOA/River): control total de tipo/punto/magnitud, pero no replican relaciones reales.
- **Semi-sintético:** tomar un dataset real, (opcionalmente) barajarlo para eliminar la autocorrelación previa, e inyectar drift en puntos conocidos vía transformaciones (permutación de features, swap de etiquetas, cambio de prior de clase, filtrado de features). Frameworks recientes (Cerqueira et al., KDD'26, arXiv:2606.07789) formalizan esto con Monte Carlo y protocolo "leave-one-dataset-out", benchmarkeando 14 detectores sobre 7 datasets reales con 4 tipos de drift inyectado (class prior, label swap, feature permutation, feature filtering), en variantes abrupta y gradual. Motivación explícita: *"naturally occurring drift in production data lacks ground truth labels and timing information."*

## Recommendations

1. **Adoptar el paradigma semi-sintético como diseño primario de evaluación de DARL.** La escasez de concept drift real verificado NO es una limitación de DARL sino una restricción del campo entero; inyectar drift sintético controlado (tipo/punto/magnitud conocidos) sobre datasets reales es la única vía para evaluar de forma **supervisada** si el agente PPO elige la acción correcta (defer / actualizar preprocesamiento / actualizar modelo / reentrenar todo). Documentarlo citando Lu et al. (2019, IEEE TKDE) y Cerqueira et al. (2026, arXiv:2606.07789).

2. **Separar explícitamente los tres regímenes en los experimentos:** (i) *covariate shift puro* (inyectar cambio en P(X): p.ej. shift de medias/varianzas de features) → debería favorecer "actualizar preprocesamiento"; (ii) *concept drift real* (swap/rotación de P(Y|X), estilo SEA/STAGGER/label-swap) → debería favorecer "actualizar/reentrenar modelo"; (iii) *estacionario/autocorrelado sin drift real* → debería favorecer "defer". Esto demuestra que el agente distingue tipos de shift, no solo detecta cualquier cambio.

3. **Usar Electricity/Covertype/Poker solo como *casos de control negativo/estacionario*, nunca como evidencia de manejo de concept drift**, y siempre reportar el baseline de persistencia. Si DARL "acierta" en Electricity, verificar que no sea por el artefacto de autocorrelación (Žliobaitė 2013/2015).

4. **Incluir Insects (Souza et al. 2020) como el único caso real con ground truth aproximado** para validación externa, reconociendo que solo las variantes abruptas (INSECTS-Abr, 52.848 inst.) tienen periodos estacionarios medibles.

5. **Si se quiere cubrir covariate/label shift realista, incorporar 2–3 tareas de TableShift** (p.ej. Hospital Readmission, Diabetes, Food Stamps), dejando claro que son domain/subpopulation shift, no drift temporal.

**Umbrales que cambiarían estas recomendaciones:** si apareciera un dataset tabular real con ground truth publicado y validado del punto/tipo de concept drift (más allá de Insects), convendría promoverlo a evaluación primaria. Si el objetivo de la tesis migrara de "manejo de drift temporal en streams" a "robustez OOD estática", entonces TableShift/Wild-Tab pasarían a ser el benchmark primario.

## Caveats

- **La terminología es inconsistente entre fuentes** (Lu et al. usan "concept drift" para cualquier cambio de P(X,Y) incluyendo covariate; Gama et al. reservan "real drift" para P(Y|X)). Los conteos dependen de qué convención se adopte; se explicitó en cada tabla.
- **Los porcentajes de la sección 4 son aproximados y dependen del "universo" de datasets elegido.** No existe un denominador canónico; distintos surveys listan distintos subconjuntos. Se ofrecen como lectura cualitativa-cuantitativa, no como censo cerrado.
- **Las cifras de #instancias/#atributos** provienen de Lu et al. Tabla 4 (arXiv:2004.05785) y de papers de uso; algunas versiones difieren: p.ej. Poker en MOA vs UCI tiene distinto tamaño y orden, lo que ya es fuente de error en la literatura.
- **Insects es "semi-verificado":** el drift se indujo controlando una variable oculta (temperatura, vía ritmo circadiano de *Culex quinquefasciatus* medido por sensor óptico), no es drift espontáneo etiquetado; y solo las variantes abruptas permiten métricas de detección estándar.
- **TableShift no mide drift temporal:** sus splits son por dominio; su documentación advierte que no se puede medir el shift verdadero desde muestra finita y que sus métricas de covariate/concept/label shift son solo aproximaciones (OTDD para covariate, FDD para concept, proporciones de etiqueta para label).

---

### Referencias clave para citar en la tesis
- Lu, Liu, Dong, Gu, Gama & Zhang, "Learning under Concept Drift: A Review", **IEEE TKDE** 31(12):2346–2363, 2019 (arXiv:2004.05785) — Tabla 3 (sintéticos) y Tabla 4 (reales); cita de limitaciones de datos reales.
- Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia, "A Survey on Concept Drift Adaptation", **ACM Computing Surveys** 46(4):44, 2014 (DOI:10.1145/2523813) — definición real vs virtual drift.
- Žliobaitė, "How good is the Electricity benchmark for evaluating concept drift adaptation", 2013 (arXiv:1301.3524) — autocorrelación, persistencia=85%.
- Žliobaitė, Bifet, Read, Pfahringer & Holmes, "Evaluation methods and decision theory for classification of streaming data with temporal dependence", **Machine Learning** 98(3):455–482, 2015 (DOI:10.1007/s10994-014-5441-4) — "6 de 18 accuracies superaron persistencia".
- Souza, dos Reis, Maletzke & Batista, "Challenges in Benchmarking Stream Learning Algorithms with Real-world Data", **Data Mining and Knowledge Discovery** 34:1805–1858, 2020 (arXiv:2005.00113; USP DS Repository) — Insects, dependencia temporal en Electricity/Covertype/Poker.
- Gardner, Popović & Schmidt, "Benchmarking Distribution Shift in Tabular Data with TableShift", **NeurIPS 2023 Datasets & Benchmarks** (arXiv:2312.07577) — 15 tareas, domain shift, shift gap ~ label shift.
- Koh et al., "WILDS: A Benchmark of in-the-Wild Distribution Shifts", **ICML 2021** (arXiv:2012.07421) — domain generalization + subpopulation shift.
- Kolesnikov, "Wild-Tab: A Benchmark For OOD Generalization In Tabular Regression", 2023 (arXiv:2312.01792).
- Cerqueira et al., "A Framework for Evaluating and Benchmarking Concept Drift Detection Methods", **KDD 2026** (arXiv:2606.07789) — metodología semi-sintética de inyección de drift.
- Moreno-Torres et al., "A unifying view on dataset shift in classification", **Pattern Recognition** 45(1):521–530, 2012 — taxonomía covariate/prior/concept.