# DDQN v6: protocolo de observaciones continuas y reward AUC/AP–costo

## Diseño fijo

Base histórica: notebook local de v5 con weighted label flipping, SHA256
`2260355ff7401ad071071961644bca9e3a8ad447b84730493578850642acfbe3`.
v5 y sus outputs se conservan intactos. Los resultados históricos de v5 usan
otro inyector conceptual y no son un control experimental directo de v6.

Kernel privado `jeffreyamc/darl-dqn-dummy-v6`, T4, internet deshabilitado,
dataset `jeffreyamc/physionet-sepsis-processed`. Notebook autónomo en
`kaggle/dqn-dummy/v6/input/main.ipynb`.

Semilla 42 en todos los generadores. Replay completo: 200 camas, 24 registros
por cama y día, 30 días, 144000 registros. Los pacientes se reemplazan al acabar
su historia y no se repite paciente/ICULOS dentro de un replay. A y B tienen
particiones independientes y disjuntas por paciente, 60/20/20 estratificadas.
El calendario es simulado; no son fechas hospitalarias absolutas.

XGBoost conserva las ocho combinaciones depth={3,5}, learning_rate={0.05,0.1},
trees={60,120}; comprobación temporal 20+10 en A-train, ajuste completo sobre
sus 30 días y selección con A-validation por AUC/AP/parsimonia. A-test no
interviene en la selección. XGBoost usa CPU y dos hilos. La GPU entrena DDQN.

### Concept drift temporal por paciente — temporal_random_onset_v1

Se conserva `make_replay` y la perturbación covariable de v5 mediante IQR y
escala. El XOR ponderado por registro se sustituye por un único inicio
positivo por paciente. `SepsisLabel` permanece intacto; `scenario_label` es
un objetivo semisintético, sin interpretación de diagnóstico clínico validado.

- Configuración congelada: semilla 42, desplazamiento máximo ±24 horas y
  probabilidad individual máxima 0.10 de convertir negativos. No es una cuota
  exacta por cohorte. La calibración no modifica estos límites.
- Construir metadatos de la historia completa: horas observadas, primer/último
  ICULOS, positividad original y primer positivo. Un replay truncado no define
  si el paciente era originalmente negativo.
- Solo modificar nuevos ingresos desde el inicio conceptual. Los pacientes
  ya ingresados conservan todas sus etiquetas originales. Combined retrasa
  el inicio conceptual dos días respecto a covariate.
- Congelar al ingreso `g=clip(concept_level_ingreso/3,0,1)`. Para positivos,
  sortear uniformemente entre horas observadas a una distancia máxima
  `floor(24*g)` de su inicio original; cero es válido. Se conserva el evento
  dentro de la historia completa, aunque no necesariamente dentro del replay.
- Para negativos, convertir con Bernoulli `p=0.10*g`. Si se convierte,
  sortear el inicio entre las horas observadas posteriores al primer registro.
  Una historia de una sola fila no es elegible. No sortear de nuevo por día.
- Las trayectorias generadas cumplen `0...0,1...1` o permanecen completamente
  negativas. Un inicio fuera del replay se registra como censurado; no se
  desplaza para hacerlo visible.
- Sorteos SHA256 separados por propósito, derivados de semilla, partición,
  paciente e ingreso; independientes de política, alfa y lambda. Los metadatos
  futuros solo construyen el mundo simulado: el modelo recibe VITALS y el
  agente conserva sus 15 observaciones de etiquetas ya disponibles.

`inject()` sigue devolviendo un DataFrame. Recibe `patient_metadata` y
`TemporalConceptConfig`; sus atributos inmutables contienen la auditoría y
`label_schedule_id`. Se exportan estos identificadores junto con las métricas
y acciones. El reward y las nueve configuraciones permanecen iguales.

Los directorios por defecto terminan en `temporal_random_onset_v1`, también
en Kaggle. Una ruta explícita DARL_DUMMY_OUTPUT debe estar libre de resultados
previos. No mezclar los outputs históricos del XOR con este inyector.

La calibración se realiza una vez sobre B-train con los mismos criterios y
respaldo de v5. Si ningún candidato cumple, seleccionar mayor caída final
(empate: menor intensidad), congelar las intensidades y registrar
`calibration_qualified=False`; no afirmar validación de degradación/recuperación.

## Reward, estado y causalidad

Nueve combinaciones cartesianas: alfa={0,0.5,1}; lambda={0.01,0.025,0.05}.

\[
r_t=(1-\alpha)AUC_{t+1}+\alpha AP_{t+1}-\lambda C_t,
\quad C_t=s_t/s_{A4}.
\]

Donde: AP es average precision positiva sin interpolación; s_t mide solamente
Pipeline.act, igual que v5. s_A4 es la mediana de cinco mediciones iniciales
A4 sobre los primeros tres días etiquetados de B-train, compartida y fija
durante el barrido. C_t no está acotado a 1. Se exportan tiempo, referencia,
costo relativo sin ponderar, penalización, componentes predictivos y reward.
Sin ambas clases, la contribución predictiva es cero y queda solamente el costo.

Cada bloque observado es [KS, PSI/(1+PSI), (DeltaAUC+1)/2, AP, validez].
DeltaAUC=AUC_reference-AUC. KS y AP ya están en [0,1]. La transformación PSI
es monótona continua de [0,infinito) a [0,1), sin recorte ni ajuste de máximos.
La transformación de DeltaAUC conserva su signo respecto al centro 0.5.
No se estiman scalers con test ni se transforman las métricas del reward.
No se eliminan diferencias pequeñas de AP mediante bins o redondeo.

Referencias AUC/AP: medias diarias válidas en A-validation del modelo elegido.
Con métricas inválidas se conserva el último par válido; inicialmente se usan
las referencias, siempre con validez=0. La red no recibe NaN. Se concatenan
tres bloques del más antiguo al más reciente: 15 entradas.

Días 1–3: warmup A1. Al decidir en d se conocen exclusivamente las etiquetas
de días anteriores; las tres últimas ventanas etiquetadas permiten A2/A3/A4.
Se predice d con el pipeline actualizado y se cierra su recompensa al recibir
las etiquetas en d+1, utilizando las probabilidades originales guardadas.
Hay 27 decisiones (4–30); d=31 solamente cierra la última recompensa.
A3/A4 se enmascaran si la ventana de ajuste no tiene ambas clases.

## Entrenamiento y selección

Un solo notebook prepara XGBoost, los replays, las referencias y el manifiesto
de 1000 episodios antes del barrido. Cada configuración reinicia red, target,
optimizador, buffer y generadores con semilla 42. Se comparte el manifiesto:
cuatro replays B-train, ciclo stable/covariate/concept/combined/covariate/
concept/combined, onset 3–7 y factores {0.75,1,1.25}.

DDQN 15–128–128–4, ReLU, Adam 5e-4→5e-5, gamma=0.99, batch=64,
replay FIFO=50000, Huber, clip de gradientes por valor 10 y copia target
cada 50 gradientes. Epsilon 1, decaimiento 0.99 por episodio, mínimo 0.05.
Exactamente 1000 episodios por configuración; no hay parada temprana.
Se exporta Huber medio por episodio y también el último minibatch.

B-validation greedy cada diez episodios, con cuatro escenarios fijos y pesos
del controlador congelados. Utilidad común:

\[
u_t=0.5 AUC_{t+1}+0.5 AP_{t+1}-0.025 C_t.
\]

Seleccionar cada checkpoint por media de tres validaciones consecutivas de
utilidad común, dando igual peso a cada escenario. Desempates: AP media,
AUC media, menor tiempo y checkpoint anterior. Configuración final: misma
media común; desempates AP, AUC, tiempo e identificador. Congelar selección
antes de construir/evaluar los escenarios B-test. Los diagnósticos de
estabilidad no alteran episodios ni se usan como prueba de superioridad.

B-test evalúa los nueve agentes congelados. El pipeline sí se adapta usando
pasado de ese replay. Ejecutar cada baseline A1–A4/Random/heurística una vez
por escenario y puntuar sus mismos resultados/times con los nueve rewards.
La heurística conserva umbrales crudos; estos no discretizan el estado DDQN.

La auditoría inmediata evalúa alternativas desde el mismo pipeline/pasado.
La acción elegida reutiliza resultado y tiempo reales. Guardar utilidad nativa
y común, AUC, AP, tiempo y regret; no es un oracle óptimo de largo plazo y no
produce transiciones, etiquetas o acciones para el entrenamiento.

## Métricas y archivos

Seleccionar una vez el umbral que maximiza F1 positivo en A-validation del
modelo inicial; en empate usar el mayor. Congelarlo para todas las políticas.
No reajustar umbrales con B-test. Guardar ROC-AUC, AP positiva, PR-AUC
trapezoidal, F1 positivo/macro, precision, recall, balanced accuracy,
prevalencia, soporte y validez contra ambas etiquetas. Las métricas globales
(pooled) se distinguen de las medias diarias. `mean_daily_ap` es la media
aritmética de AP diaria válida; no una nueva métrica multiclase.

CSV comunes: reward_configs, normalization, observation_reference,
patient_partitions, partition_counts, episode_manifest, xgboost_search,
a_metrics, a_daily_metrics, f1_threshold, initial_transform, calibration,
cost_calibration, drift_calendar, drift_samples, concept_config,
patient_history_metadata, concept_patients y concept_timelines.
`concept_audits/` conserva CSV por paciente y trayectoria única en
calibración/train/validation/test, compartidos entre configuraciones.
Contienen ingreso, intensidad, inicios, desplazamiento, sorteos,
probabilidad, conversión, motivo y visibilidad/censura del evento.
`label_schedule_id` vincula métricas y acciones a estas auditorías.

CSV largos: training_history, train_daily_metrics, train_actions,
validation_history, validation_daily_metrics, validation_actions,
checkpoint_history, checkpoint_selection, sweep_summary, selection_freeze,
baseline_daily_metrics, baseline_actions, daily_metrics, actions,
policy_comparisons, oracle_audit y controller_freeze.
Incluyen config_id, alfa, lambda, fase, episodio, escenario y día según alcance.

Guardar predicciones por fila de A-test y B-test, no de los 9000 episodios
de entrenamiento. Modelos inicial/seleccionados permiten verificar/restaurar
los controladores. CSV, PNG, modelos y logs son resultados locales ignorados
por Git; se versionan código, protocolo y README.

## Figuras y ejecución

Modo `plots`: leer solamente CSV, sin parquet ni modelos y sin entrenar.
Regenera las familias originales de figuras y ocho diagnósticos temporales
(cuatro por escenario concept/combined):
drift día 1/30 y relación conceptual; retornos nativo/común comparados con
ventana 100; retorno por escenario con ventana 15 apariciones; monitoreo
crudo/normalizado por escenario; barras de acciones train/validation y medias;
AUC diaria por seis políticas; acciones y oracle inmediato; DDQN frente A1.
Además se conserva diagnóstico epsilon/Huber. La banda es ±1 desviación
estándar temporal, no un intervalo de confianza. Media, std y varianza se
exportan en training_smoothing/scenario_smoothing; action_averages y
figure_manifest describen agregaciones y archivos. Los CSV de drift permiten
regenerar histogramas y curvas por decil.

No ejecutar Kaggle ni entrenamiento local sin nueva autorización.
La validación de este cambio se limita a historias ficticias y gráficos CSV.
Antes de un futuro full autorizado: contratos locales y smoke remoto con nueve
configuraciones, 50 camas, ocho días B, seis A y 24 episodios por agente.
Validar causalidad, dimensión, extremos del reward, máscaras, cierre final,
contratos temporales del inyector, target DDQN, selección de tres checkpoints, predicciones
y métricas reconstruidas. Inspeccionar PNG del smoke antes de full.

Full: 9000 episodios y 27 decisiones por episodio. Persistir por etapa,
checkpoint y configuración. El guard de presupuesto es 39600 segundos por
defecto: ante agotamiento guardar estado interno fallido y resultados parciales,
sin reducir episodios ni declarar éxito. Rehusar directorios de salida con CSV
anteriores para impedir mezclar ejecuciones. Descargar cada versión en una
carpeta separada y reportar estado remoto e interno independientemente.

Integridad no implica mejor rendimiento, cuatro acciones utilizadas ni curvas
siempre crecientes. Con una sola semilla, las conclusiones son descriptivas de
esta ejecución y no estiman robustez entre inicializaciones independientes.

## Comprobación aislada del cambio temporal

Se verifican etiquetas binarias/monótonas, conservación de eventos positivos,
desplazamientos factibles, censura por replay, inicios en horas realmente
observadas, intensidad congelada, ingresos anteriores protegidos, estabilidad
y covariate sin cambios, retraso combinado, sorteos reproducibles y exclusión
de metadatos futuros de las características y observaciones.

Las figuras nuevas se regeneran con `render_temporal_concept_from_csv(root)`
desde concept_patients/concept_timelines/drift_calendar: trayectorias, histograma
de desplazamientos, prevalencia diaria y conteos disjuntos por paciente.
`concept_plot_summary.csv` y `concept_figure_manifest.csv` registran agregados
y rutas. El modo plots general también incorpora estos diagnósticos. Los
outputs históricos sin tablas temporales conservan su regeneración anterior.

La prueba ficticia comprueba implementación y visualización; no entrenamiento
ni recuperación del AUC. El cambio aleatorio independiente de los vitales
puede añadir ruido irreducible. El fallo histórico de reconstrucción de
macro-F1 del smoke remoto no queda resuelto ni validado por este cambio.
