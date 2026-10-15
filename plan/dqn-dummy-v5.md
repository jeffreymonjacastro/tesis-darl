# DARL DQN v5: entrenamiento en A y adaptación temporal en B

## Alcance y ejecución

Implementación autónoma en `kaggle/dqn-dummy/v5/input/main.ipynb`, derivada
del experimento v4. Kernel privado `jeffreyamc/darl-dqn-dummy-v5`, dataset
`jeffreyamc/physionet-sepsis-processed`, T4, internet deshabilitado. XGBoost
usa CPU y dos hilos; DDQN usa CUDA si está disponible. Semilla base 42 para
particiones, colas, exploración, replay y ajustes.

La ejecución inicial smoke usa 50 camas, 8 días en B, 6 días en A (4+2),
24 episodios y las ocho configuraciones completas de XGBoost. Sus outputs
se descargan en `.tmp/`. Tras revisar integridad, ejecutar el modo completo:
200 camas, 30 días, preparación temporal 20+10 y 500–1000 episodios.
Los resultados completos se descargan en `kaggle/dqn-dummy/v5/outputs/`.

## 1. Pacientes, calendario y disponibilidad

A y B se dividen independientemente por paciente en 60% train, 20%
validation y 20% test, estratificando por presencia de sepsis. Son conjuntos
disjuntos. Cada calendario mezcla una cola reproducible de pacientes dentro
de su partición, consume los registros en orden ICULOS y reemplaza al paciente
en la siguiente hora cuando acaba su historia. Cada hora tiene 200 registros
y cada replay consume 144 000 filas. Un paciente puede continuar varios días
y cada pareja paciente/ICULOS se consume una sola vez por replay.

No existe una subdivisión permanente update/evaluación. Las mismas filas
que reciben predicciones aportan monitoreo. El entrenamiento de mantenimiento
usa su historial etiquetado una vez que está disponible. Se registran pacientes
disponibles, utilizados y filas consumidas. La cola se valida antes del ajuste
de A y del entrenamiento del agente; su agotamiento produce un error explícito.
No se reutiliza un paciente dentro del mismo replay.

El dataset no proporciona fechas absolutas hospitalarias. Los 30 días son un
reloj simulado; ICULOS mantiene el tiempo relativo de cada paciente. Forward
fill ocurre exclusivamente dentro de cada historia, en orden horario.

## 2. Preparación independiente en A

Usar HR, O2Sat, Temp, SBP, MAP, DBP y Resp. Objetivo `SepsisLabel`.
Transformador: imputación por mediana, normalización robusta mediana/IQR y
recorte a [-2,2], ajustados exclusivamente con las filas de entrenamiento.
Columnas sin observaciones usan 0/1 inicialmente y conservan sus parámetros
anteriores en un reajuste; IQR mínimo 1e-6.

Para cada combinación depth {3,5}, learning_rate {0.05,0.1} y trees {60,120}:

1. Ajustar con días 1–20 de A-train; medir días 21–30 sin reajustar.
2. Ajustar un modelo nuevo con los 30 días de A-train.
3. Medir A-validation y seleccionar por AUC agregado, luego average precision,
   luego menor cantidad de árboles, profundidad y learning rate.

Balance de clases calculado exclusivamente con las etiquetas de cada ajuste.
El modelo seleccionado se mide en A-test una vez; sus resultados no deciden
selección, calibración ni aprendizaje. La referencia de rendimiento es la media
de los AUC diarios válidos del modelo final en A-validation.

## 3. Deriva artificial y calibración

Stable conserva signos y etiquetas. Covariate conserva la rampa de v4:
desplazamiento direccional en IQR y cambio de escala gradual, desde el día 5.
Concept adelanta el primer positivo de la historia completa del paciente,
gradualmente y con un máximo adicional de 12 horas. Pacientes sin positivos
permanecen negativos. Combined combina medición y concepto; el componente
conceptual empieza dos días después, el día 7.

`SepsisLabel` se conserva; el objetivo experimental va en `scenario_label`.
Concept/combined son cambios artificiales del objetivo, no evidencia de un
cambio clínico real. Reentrenamiento y recompensa usan `scenario_label`.
Se reportan AUC y average precision contra ambas etiquetas.

Calibración solo en B-train: covariate {0.25,0.35,0.5} IQR/día;
concept {0.25,0.5,1} horas/día. Comparar, en el primer día de deriva y en
el último, el modelo inicial con el calendario estable del mismo grupo.
Medir recuperación inmediata en el último día después de ajustar A2 para
covariate o A3 para concept con los tres días anteriores etiquetados.
Elegir la menor velocidad con caída inicial ≤0.05, caída final ≥0.10 y
recuperación ≥0.05. Si ninguna cumple, elegir mayor caída final (empate:
menor velocidad), registrar calibración no cumplida y conservar ese protocolo.
Durante entrenamiento variar inicio entre días 3–7 y velocidad por factores
{0.75,1,1.25}; validation/test mantienen inicio y velocidades calibradas.

## 4. Ciclo diario causal

Al empezar d llegan las etiquetas de d-1. El AUC se calcula con las probabilidades
guardadas cuando se predijo. Cerrar recompensa de la acción vigente en esas
predicciones; codificar estado con métricas de d-3,d-2,d-1. Elegir acción,
ajustar exclusivamente con esos últimos tres días disponibles y predecir d.
Se guardan probabilidad, versión del pipeline y fecha de disponibilidad de etiqueta.

El pipeline permanece fijo durante cada día, por lo que el cálculo vectorizado
diario equivale a inferir sucesivamente sus horas sin ajustes intermedios.
La cola del entorno avanza hasta la recepción siguiente; ningún estado o ajuste
consulta etiquetas del día que se va a predecir. Días 1–3: calentamiento A1.
Decisiones desde día 4. Día 31: solo recepción final, sin predicciones nuevas.

- A1: conservar pipeline.
- A2: reajustar mediana/IQR y referencia KS/PSI, conservar clasificador.
- A3: reentrenar XGBoost, conservar transformador/referencia.
- A4: reajustar ambos y referencia.

A3/A4 usan hiperparámetros seleccionados en A. Si la ventana pasada tiene
una sola clase, ambas quedan enmascaradas. Políticas fijas que piden una acción
deshabilitada ejecutan A1 y registran el motivo. KS/PSI se calculan por signo
sobre exactamente las filas predichas; se usa su máximo para codificar el estado.
Referencia inicial A-train; deciles para PSI y muestra determinista de hasta
1000 valores para KS. Las referencias cambian exclusivamente en A2/A4.

Recompensa: AUC diario − 0.05 × segundos/COST_UNIT. COST_UNIT es la mediana de
cinco mediciones A4 sobre los tres primeros días etiquetados de B-train.
Sin ambas clases, AUC vacío y flag falso: la recompensa es solamente el costo.
No se recalculan predicciones históricas después de un ajuste.

## 5. Estado, aprendizaje y evaluación

Tres bloques de diez entradas: KS onehot de tres categorías (cortes 0.10/0.30),
PSI onehot (0.10/0.25), caída AUC onehot (<0.03, [0.03,0.10), ≥0.10), validez.
Caída relativa a A-validation. Sin AUC, usar último AUC válido y flag falso;
si no existe, usar referencia de A-validation. Total: 30 entradas.

DDQN 30→128→128→4, ReLU, Adam 5e-4→5e-5, Huber, batch 64, gamma 0.99,
replay FIFO 50 000, target cada 50 gradientes. Epsilon inicial 1, decaimiento
0.99 por episodio y mínimo 0.05. Transiciones guardan máscara del estado siguiente;
selección online y evaluación target excluyen acciones deshabilitadas.
La recompensa almacenada se usa directamente en el objetivo DDQN.

Cada episodio reinicia el pipeline seleccionado de A. Solo B-train produce
gradientes. Se emplean cuatro colas de entrenamiento reproducibles; las filas
pueden reaparecer en episodios distintos, pero una sola vez dentro de cada replay.
Evaluación greedy en cuatro escenarios B-validation cada diez episodios.
Seleccionar checkpoint por media de tres validaciones completas consecutivas,
sin seleccionar a partir de una o dos. Parada a partir de episodio 500 si cinco
validaciones tienen variación relativa ≤5% y acuerdo de acciones ≥95%; límite 1000.

Reportar estabilidad en las tres validaciones asociadas al checkpoint seleccionado,
incluyendo variación, acuerdo y acuerdo con tolerancia Q de 0.005. Esta última
métrica evalúa todos los checkpoints sobre estados y máscaras comunes de
la trayectoria de validación del checkpoint seleccionado. Si no es
estable, declararlo. B-test evalúa una política congelada; hashes comprueban
que validation/test no modifican sus pesos.

Comparadores: A1–A4, random y heurística KS≥0.30 o PSI≥0.25, caída AUC≥0.03.
El oracle inmediato examina cuatro alternativas disponibles desde el mismo
pipeline, ajusta solo con pasado y consulta etiquetas del día futuro solamente
para auditar su utilidad. No aporta estados, acciones ni recompensa al aprendizaje.
Los tiempos son mediciones reales; pueden variar entre ejecuciones aun con semilla fija.

## 6. Artefactos y aceptación

CSV: configuración/progreso/error, búsqueda XGBoost, métricas A, particiones,
conteos/capacidad, calibración de deriva/costo, entrenamiento, selección,
acciones, métricas diarias, comparaciones, calendario de deriva, oracle,
control de congelamiento y controles de integridad.
Predicciones de A-test y cada escenario/política B-test se escriben incrementalmente.
Incluyen paciente, ICULOS, hora/día simulado, etiquetas original/escenario,
probabilidad, acción, versión y momento de disponibilidad de etiqueta.
Figuras solo PNG: preparación A, ocupación, deriva, monitoreo, entrenamiento,
acciones, AUC y retorno por política, regret y costo.

Los CSV voluminosos de predicciones y archivos temporales/logs quedan fuera
del versionado. Las tablas agregadas y PNG documentan el resultado en el README.

Antes del modo completo: controles de particiones disjuntas, capacidad horaria,
filas únicas, orden intrapaciente, causalidad del estado/ajuste/recompensa,
warmup, cierre final, ausencia de AUC, máscaras, columnas vacías,
preservación del objetivo original y límite del desplazamiento. Reconstruir
AUC desde CSV; comprobar congelamiento y exclusión del oracle del aprendizaje.
Smoke completo en Kaggle y revisión visual PNG. Después ejecutar completo,
comprobar versión/modo descargados y repetir auditoría de exportaciones.

Integridad y rendimiento son distintos. No se exige superar baselines ni usar
las cuatro acciones. No cambiar diseño ni selección usando B-test. Reportar
estado remoto de Kaggle y estado interno `run_summary.csv` por separado.
