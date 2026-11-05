# Feature engineering de XGBoost en DARL

Oct 29, 2026 · @Brigitte Rojas

Agregar variables mejora el AUROC de XGBoost en unos 0,03 puntos, pero borra la diferencia entre acciones de mantenimiento que DARL estudia; se recomienda mantener los 7 signos vitales y corregir el entorno.

## 1. Objetivo

Evaluar si ampliar las variables de entrada del XGBoost (feature engineering) mejora su desempeño en la predicción de sepsis y cómo ese cambio afecta al entorno de DARL: las acciones de mantenimiento A1–A4 y el drift inyectado.

Todos los experimentos usan los datos del PhysioNet/CinC Challenge 2019 (hospitales A y B) y la semilla 42.

## 2. Hipótesis

No se plantea. El trabajo compara conjuntos de variables y configuraciones del entorno; no propone un método nuevo.

## 3. Experimento 1: ICULOS y la estructura de SepsisLabel

### Objetivo

Decidir si ICULOS (horas desde el ingreso a la UCI) debe entrar como variable del modelo.

### Setup

- Datos: 20.336 pacientes del hospital A y 20.000 del hospital B, un registro por hora.
- Definición oficial del challenge: SepsisLabel = 1 desde t\_sepsis − 6 horas.
- Se analizó la prevalencia por tramo de ICULOS y se usó ICULOS sola como puntaje de riesgo.

### Resultados

- Ningún paciente séptico tiene más de 9 horas de registro después de su primera hora positiva: las horas positivas siempre están al final de la estancia.
- La prevalencia sube de 1,2 % (24–48 h) a 14 % (72–120 h) en A, y de 0,8 % a 8,9 % en B.
- ICULOS sola alcanza un AUROC de 0,641 en A y 0,628 en B, igual al modelo de la tesis (M\_ref = 0,642).
- Con el protocolo v5, ICULOS sola tiene un AP diario de 0,112 en B-test, contra 0,044 del modelo con 7 signos.
- ICULOS se registra distinto en cada hospital: 37 % de los pacientes de A no empieza en la hora 1, contra 6,5 % en B.

### Discusión

El poder predictivo de ICULOS viene de cómo se construyó el dataset, no de la fisiología. Se excluye como variable del modelo y se mantiene solo como marca de tiempo del replay y del drift de concepto.

## 4. Experimento 2: conjuntos de variables con el protocolo v5

### Objetivo

Evaluar XGBoost con cinco conjuntos de variables en A-val, A-test y B-test para establecer un baseline sin ICULOS.

### Setup

- Protocolo v5: particiones 60/20/20 por paciente, replay de 200 camas × 30 días, imputación hacia adelante por paciente, transformador mediana/RIC con recorte a \[−2, 2\].
- XGBoost entrenado con los 30 días de A-train y búsqueda de 8 combinaciones de hiperparámetros elegidas en A-val.
- Métrica: AUROC diario medio, la misma de M\_ref. Semilla 42, una repetición.
- Conjuntos acumulativos: S0 = 7 signos; S1 = S0 + edad y sexo; S2 = S1 + shock index, shock index modificado y presión de pulso; S3 = S2 + promedio, desviación y cambio en 6 h; S4 = S3 + 8 laboratorios de rutina + BUN/creatinina (42 variables).
- Los 8 laboratorios (Glucose, Potassium, Hct, Hgb, BUN, WBC, Creatinine, Platelets) se eligieron porque más del 93 % de los pacientes tiene al menos una medición.

### Resultados

| Conjunto | A-val | A-test | B-test |
| --- | --- | --- | --- |
| S0: 7 signos | 0,641 | 0,655 | 0,676 |
| S1: + edad, sexo | 0,594 | 0,639 | 0,653 |
| S2: + aritméticas | 0,599 | 0,643 | 0,666 |
| S3: + ventanas 6 h | 0,625 | 0,670 | 0,658 |
| S4: + laboratorios | 0,667 | 0,688 | 0,709 |
| ICULOS sola (referencia) | 0,613 | 0,638 | 0,614 |

### Discusión

S0 reproduce M\_ref (0,641 vs. 0,642), lo que valida el protocolo. Sin ICULOS, la mejor mejora es de unos +0,03 AUROC y viene de los laboratorios; edad y sexo empeoran el modelo y las ventanas de 6 h no se transfieren bien de A a B.

## 5. Experimento 3: ablation por grupos de variables en el hospital A

### Objetivo

Identificar qué grupos de variables aportan señal y cuáles añaden ruido, usando solo el hospital A para no contaminar B-test.

### Setup

- Dos direcciones: agregar un grupo a S0 y quitar un grupo de S4.
- Hiperparámetros fijos de la tesis (profundidad 3, tasa 0,1, 60 árboles) para aislar el efecto de las variables.
- Entrenamiento con 7 y con 30 días de A-train; 3 semillas de replay (42, 43, 44).
- Métrica: AUROC diario medio en A-val, confirmado en A-test. Variación entre semillas: ±0,005 con 30 días y ±0,02 con 7 días.

### Resultados (30 días de entrenamiento)

| Grupo | Agregar a S0: Δ A-val / A-test | Quitar de S4: Δ A-val / A-test | Veredicto |
| --- | --- | --- | --- |
| Edad, sexo | −0,029 / −0,020 | +0,013 / −0,001 | Ruido: quitar |
| Laboratorios | +0,034 / +0,001 | −0,031 / −0,016 | Útil |
| Desv. estándar 6 h | +0,009 / +0,017 | −0,005 / −0,012 | Útil |
| Promedio 6 h | +0,005 / +0,009 | −0,006 / −0,011 | Útil |
| Cambio 6 h | +0,010 / +0,009 | −0,002 / +0,001 | Marginal |
| Aritméticas | +0,002 / 0 | −0,001 / +0,002 | Neutro |
| BUN/creatinina | +0,003 / +0,001 | −0,001 / −0,003 | Neutro |

Con 7 días de entrenamiento, todos los conjuntos quedan entre 0,57 y 0,61 en A-val y ninguno supera claramente a S0 (0,599).

### Discusión

Edad y sexo añaden ruido; los laboratorios y las ventanas de promedio y desviación aportan señal. Las variables extra solo ayudan con suficientes datos: con 7 días el modelo no las aprovecha. Conjunto propuesto: 7 signos + 8 laboratorios + promedio y desviación 6 h de cada signo (29 variables), aún no evaluado.

## 6. Experimento 4: modelo congelado vs. acciones A2–A4 con ventanas de 7 días

### Objetivo

Comparar en igualdad de datos el modelo congelado y las acciones de mantenimiento, para S0 y S4, en los cuatro escenarios de drift.

### Setup

- Modelo congelado (A1) entrenado con los días 1–7 de A-train, con la búsqueda de 8 combinaciones rehecha; se reporta también el congelado con 30 días.
- En B-test, desde el día 8, A2 (reajusta f\_φ), A3 (reentrena XGBoost) y A4 (ambos) usan los 7 días etiquetados anteriores.
- Escenarios v5: stable, covariate (rate 0,35), concept (0,5 h/día) y combined; inicio del drift en el día 5. Las variables derivadas se recalculan después de inyectar el drift.
- Semilla 42, un replay de B-test. Métrica: AUROC diario medio de los días 21–30.

### Resultados (AUROC diario medio, días 21–30)

| Escenario | Conjunto | A1 30 d | A1 7 d | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- | --- |
| Stable | S0 | 0,637 | 0,595 | 0,621 | 0,662 | 0,660 |
| Stable | S4 | 0,678 | 0,557 | 0,557 | 0,657 | 0,653 |
| Covariate | S0 | 0,495 | 0,505 | 0,628 | 0,515 | 0,630 |
| Covariate | S4 | 0,548 | 0,482 | 0,535 | 0,634 | 0,666 |
| Concept | S0 | 0,624 | 0,592 | 0,626 | 0,641 | 0,642 |
| Concept | S4 | 0,682 | 0,555 | 0,556 | 0,672 | 0,672 |
| Combined | S0 | 0,501 | 0,511 | 0,611 | 0,505 | 0,625 |
| Combined | S4 | 0,551 | 0,464 | 0,519 | 0,632 | 0,653 |

Tiempo medio por acción: A4 tarda 0,13 s con S0 y 0,39 s con S4; A2 tarda 0,026 s con S0.

### Discusión

- Con 7 días, reentrenar ya supera al congelado de 30 días incluso sin drift (S0 stable: 0,662 vs. 0,637); con 3 días, en cambio, siempre perdía. Se recomienda subir UPDATE\_DAYS de 3 a 7.
- Con S0 cada acción tiene un papel claro: A2 corrige el drift covariado a 1/5 del costo de A4, A3 y A4 el de concepto, y A4 el combinado.
- Con S4 ese patrón se pierde: A3 ≈ A4 en todos los escenarios y el congelado de 7 días se transfiere mal a B, así que A2 solo puede volver a un nivel bajo.

## 7. Experimento 5: diagnóstico del entorno de 200 camas

### Objetivo

Verificar si el entorno de DARL permite medir con precisión las diferencias entre acciones.

### Setup

- Replays v5 de A-val y B-test (200 camas × 30 días, 4.800 filas por día).
- Se midieron: pacientes sépticos por día, variación del AUROC diario del modelo congelado sin drift, pacientes repetidos entre la ventana de 7 días y el día predicho, y saturación del recorte bajo el drift covariado.

### Resultados

| Medida | A-val | B-test |
| --- | --- | --- |
| Pacientes por día | 316 | 319 |
| Pacientes sépticos por día (mín.–máx.) | 14 (6–21) | 9 (2–15) |
| Horas positivas por día | 102 | 62 |
| Pacientes del día ya vistos en los 7 días previos | 61 % | 61 % |

- Desviación estándar del AUROC diario sin drift en B-test: 0,08 con S0 (rango 0,50–0,84) y 0,10 con S4.
- Drift covariado con rate 0,25: el desplazamiento llega a 6,5 RIC el día 30 (HR +143 lpm, temperatura +5,9 °C). El 94 % de los valores está saturado en ±2 desde el día 21; con rate 0,5, el 83 % ya el día 10.

### Discusión

- El ruido de un día (0,08) es mayor que los umbrales de DARL (caída de 0,03, costo λ = 0,05) y que varias diferencias entre acciones; esto dificulta que DDQN y PPO aprendan una política estable.
- Los pacientes repetidos pueden favorecer a A3 y A4: el modelo reentrenado podría reconocer a pacientes cuya sepsis ya se reveló. Falta verificarlo.
- El drift covariado no tiene techo: en los últimos días borra la información en vez de simular un sensor descalibrado, y por eso las tres intensidades del grid dan el mismo resultado.

## 8. Conclusiones y próximos pasos

Se recomienda mantener los 7 signos vitales en el pipeline de DARL y corregir el entorno antes de volver a entrenar a los agentes. Las variables extra suben el AUROC unos 0,03 puntos, pero hacen que A3 y A4 se comporten igual y que A2 pierda su papel, que es justo lo que la tesis estudia.

**Limitaciones:** los experimentos 2, 4 y 5 usan una sola semilla y un solo replay de B-test, y las intensidades de drift del experimento 4 no se calibraron.

**Próximos pasos:**

1. Medir el ruido del AUROC diario con bootstrap por paciente y compararlo con los umbrales y con λ; evaluar una métrica sobre más pacientes (ventana móvil de 3 días o mismo peso por paciente).
2. Subir UPDATE\_DAYS de 3 a 7 días.
3. Poner un techo al drift covariado (0,5–1 RIC: HR +11 a +22 lpm, temperatura +0,5 a +0,9 °C) y agregar a la calibración en B-train un límite de caída y de saturación.
4. Separar el AUROC del día en pacientes nuevos, continuos sin sepsis conocida y continuos con sepsis ya revelada; si se confirma el efecto, excluir de la recompensa a estos últimos.
5. Repetir los experimentos 2 y 4 con varias semillas de replay.
6. Documentar en la tesis la exclusión de ICULOS y la estructura de la etiqueta como limitaciones.
