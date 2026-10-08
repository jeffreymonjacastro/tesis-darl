# Guion de exposición (10 min) — DARL, PG-2

Basado en el deck `PFC2 - PG2` (25 láminas), con las láminas 22–24 reemplazadas
por `DARL_3_slides.pptx`. Ritmo: ~130 palabras por minuto. Reparto sugerido:
**Dayane** abre y cierra (1–10 y 22–25), **Jeffrey** expone la metodología y los
resultados (11–21).

| Bloque | Láminas | Tiempo |
|---|---|---|
| Problemática | 1–7 | 2:35 |
| Objetivos | 8 | 0:30 |
| Diseño de la solución | 9–15 | 3:15 |
| Resultados | 16–21 | 2:15 |
| Limitaciones, pasos, conclusiones | 22–25 | 1:25 |
| **Total** | | **≈ 10:00** |

---

**1 · Portada (0:15)**
Buenos días. Somos Jeffrey Monja y Dayane Rojas, y presentamos DARL: aprendizaje
por refuerzo para decidir qué parte de un pipeline de machine learning
actualizar cuando los datos cambian.

**2 · Agenda (0:10)**
Veremos la problemática, los objetivos, el diseño de la solución, los
resultados, y cerraremos con limitaciones y siguientes pasos.

**3 · Los modelos tabulares se degradan (0:25)**
Un modelo entrenado hoy asume que los datos de mañana se parecerán a los de
hoy, y eso casi nunca pasa. En TableShift (es un benchmark: un conjunto estándar de datasets y pruebas que sirve para medir qué tan bien aguantan los modelos de ML con datos tabulares cuando los datos cambian), el benchmark de NeurIPS 2024, las 15
tareas tabulares pierden rendimiento fuera de distribución, y ninguno de los 18
modelos evaluados cierra la brecha. La caída llega a 34.5 puntos.

**4 · En salud, esa caída afecta decisiones clínicas (0:25)**
Nuestros dos casos son clínicos. En sepsis (emergencia médica potencialmente mortal que ocurre cuando el cuerpo tiene una respuesta extrema y desregulada ante una infección, dañando sus propios tejidos y órganos), el modelo pierde 6 puntos al pasar a
estancias largas en UCI; en readmisión hospitalaria, casi 6 al cambiar la fuente
de admisión. Un modelo de riesgo que se degrada en silencio compromete
decisiones sobre pacientes.

**5 · El problema real: qué actualizar (0:30)**
Un pipeline tabular tiene dos etapas: el preprocesamiento y el modelo. Si cambia
la distribución de las variables, el covariate shift, se desactualiza el
preprocesamiento. Si cambia la relación entre variables y etiqueta, el concept
drift, el modelo deja de ser válido. Hoy la práctica es reentrenar todo, que es
caro, cuando a veces basta con actualizar una sola etapa.

**6 · Brecha en la literatura (0:25)**
Los trabajos previos diagnostican el tipo de drift o deciden cuándo reentrenar,
pero todos tratan el pipeline como una sola pieza. Ninguno decide qué etapa
actualizar ni aprende esa decisión. Los propios autores lo dejan como trabajo
futuro.

**7 · Pregunta de investigación (0:20)**
Nuestra pregunta es: ¿puede una política de aprendizaje por refuerzo elegir,
con señales de monitoreo, la actualización que más rendimiento recupera al
menor costo? El agente tiene cuatro acciones: no hacer nada, actualizar
features, actualizar el modelo o reentrenar todo.

**8 · Objetivos (0:30)**
El objetivo general es diseñar, implementar y evaluar DARL. Para eso:
construimos un módulo de monitoreo con PSI, KS y AUC; formalizamos la decisión
como un POMDP con cuatro acciones y una recompensa que equilibra AUC y costo;
entrenamos un agente DDQN con drift sintético; y evaluamos la política frente al
costo de ignorar el drift.

**9 · Simular concept drift fue un reto (0:25)**
Simular concept drift realista no fue trivial. Voltear etiquetas al azar solo
mete ruido; amplificar la relación no degrada al modelo. La versión que
funcionó rota los pesos de la relación: el AUC cae de forma ordenada y la
distribución de X no cambia, así que es concept drift puro.

**10 · Ante concept drift, corregir X no sirve (0:20)**
Con ese inyector comprobamos algo clave: bajo concept drift, actualizar features
da exactamente lo mismo que no hacer nada, 0.346 de AUC. Reentrenar el modelo
recupera hasta 0.663. la respuesta  depende del tipo de drift.

**11 · EDA de PhysioNet (0:20)** — *cambia a Jeffrey*
PhysioNet tiene siete signos vitales por hora y por paciente. Dos retos: no
todas las variables se miden cada hora, y cada paciente pasa un tiempo distinto
en UCI.

**12 · Simulación temporal (0:30)**
Para que el problema sea secuencial de verdad, simulamos una UCI con 200 camas.
Cada hora llegan los signos vitales; cuando un paciente sale, otro ocupa su
cama. Cada día el modelo predice, y cuando llega la etiqueta calculamos el AUC.
Así, la decisión de hoy afecta al modelo de mañana.

**13 · Inyección de covariate shift (0:20)**
Sobre esa simulación inyectamos covariate shift: al día 16 los vitales se
desplazan hasta 3 rangos intercuartílicos, con KS cercano a 0.9.

**14 · Inyección de concept drift (0:25)**
Para concept drift, los vitales se ven igual, con KS cercano a 0.04, pero la
relación con la sepsis se invierte: en frecuencia cardiaca, por ejemplo, el
riesgo pasa de subir a bajar con el valor. PSI y KS no pueden ver este cambio.

**15 · Metodología: DARL con DDQN (0:55)**
Cada día el agente observa el máximo de PSI y KS entre variables, más el AUC.
Lo discretizamos en tres niveles y apilamos los últimos tres días: 30 entradas
binarias. La recompensa es el AUC del día siguiente menos el costo de la acción,
normalizado contra reentrenar todo. Entrenamos con Double DQN, que usa una red
objetivo para estabilizar el aprendizaje y una política epsilon-greedy para
explorar.

**16 · Resultados: monitoreo diario (0:20)**
Este es el monitoreo en el escenario combinado. Al día 5 KS y PSI suben y cruzan
sus umbrales, y el AUC empieza a caer: el monitor sí detecta el cambio.

**17 · Curva de entrenamiento (0:15)**
Durante 1000 episodios el retorno medio sube de 13.37 a 13.59. Aprende, pero la
mejora es modesta.

**18 · AUC por política (0:25)**
Sin actuar, el AUC cae a 0.59; actualizar solo features es aún peor, 0.50. El
DQN se mantiene entre 0.85 y 0.93, al nivel de reentrenar todo y de la
heurística.

**19 · Retorno acumulado (0:20)**
Al sumar AUC menos costo, el DQN queda primero, empatado con la heurística y
apenas por encima de random y de reentrenar todo.

**20 · Oráculo inmediato (0:20)**
Comparado con la mejor acción de cada día, el agente acierta en unos 9 de 15
días. Falla sobre todo cuando la mejor opción era actualizar solo el modelo,
que nunca elige.

**21 · DQN frente a no actuar (0:25)**
En los cuatro escenarios: estable, AUC igual a no actuar, sin pagar costo
innecesario. Con covariate, 0.93 contra 0.80. Con concept, 0.90 contra 0.77,
aunque reacciona unos tres días tarde. Con el combinado, 0.91 contra 0.79.

**22 · Limitaciones (0:35)** — *cambia a Dayane*
Por diseño, evaluamos un dataset, 16 días y un drift bastante extremo, y
asumimos que la etiqueta llega al día siguiente. En los resultados encontramos
cuatro límites: el agente empata con una heurística, así que aún no demostramos
la ventaja del RL; reacciona tarde al concept drift porque PSI y KS no lo ven;
nunca usa la acción de actualizar solo el modelo; y el costo pesa poco en la
recompensa.

**23 · Siguientes pasos (0:25)**
Primero llevaremos la simulación a Diabetes y a fraude, con drift sintético y
luego real. Después agregaremos una señal de concept drift a la observación y
calibraremos el costo. Por último validaremos con cinco semillas e intervalos
de confianza.

**24 · Conclusiones (0:20)**
Monitorear y actuar recupera hasta 13 puntos de AUC; el agente no interviene si
no hace falta; el tipo de drift determina qué actualizar; y el reto que queda es
que el RL supere a una regla simple.

**25 · Referencias (0:05)**
Estas son nuestras referencias. Muchas gracias; quedamos atentos a sus
preguntas.

---

## Preguntas probables del jurado

- **"Si empata con la heurística, ¿para qué RL?"** → La heurística se diseñó
  conociendo el escenario; el RL la aprende sin reglas. La validación con
  severidades variadas y drift real (pasos 3, 5 y 6) es donde una regla fija
  debería fallar.
- **"¿Por qué nunca elige A3?"** → Con 3 niveles de discretización y costo
  0.05, A2 y A4 dominan la recompensa; calibrar el costo y agregar una señal
  de concept drift (pasos 4 y 5) es la hipótesis para corregirlo.
- **"¿El drift de 3 IQR no es demasiado fácil?"** → Sí, es extremo a propósito
  para validar el flujo; el paso 5 incluye severidades menores.
