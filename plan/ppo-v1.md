# DARL PPO v1: mismo entorno de v5, otro algoritmo

## Objetivo

Comparar **PPO** con el **DDQN de v5** en el mismo problema de mantenimiento,
cambiando **solo el agente**. Entorno, datos, particiones, deriva, estado,
acciones, recompensa, selección de checkpoint y evaluación quedan idénticos a
`plan/dqn-dummy-v5.md`.

La pregunta no es si PPO "arregla" el problema, sino si el resultado depende del
algoritmo o del problema. Interesan tres cosas:

1. **Desempeño en B-test:** retorno, AUC, average precision, costo y *regret*
   frente al oracle inmediato.
2. **Estabilidad de la política.** En v5, el acuerdo de acciones en la ventana del
   checkpoint elegido fue solo 0,69.
3. **Eficiencia:** pasos del entorno necesarios para alcanzar el mejor retorno de
   validación.

**Resultado esperado.** Como PPO recibe la misma observación de 30 entradas, se
espera un desempeño parecido al de DDQN. En v5, DDQN ya iguala a la mejor acción
fija (A2 en covariate/combined, A1 en stable/concept). Un empate bien medido es
un resultado válido.

## Qué se conserva de v5 (sin cambios)

- **Datos y particiones:** PhysioNet, 7 vitales y `SepsisLabel`. A y B divididos
  60/20/20 por paciente; XGBoost seleccionado en A.
- **Replay:** 200 camas, 30 días y etiquetas disponibles al día siguiente.
  Calentamiento A1 en los días 1–3; decisiones del día 4 al 30 (27 por episodio).
- **Deriva y calibración:**
  - stable, covariate, concept y combined;
  - mismas velocidades calibradas en B-train;
  - en entrenamiento, inicio sorteado en los días 3–7 y factor de velocidad
    {0,75; 1; 1,25}.
- **Acciones:** A1 conservar · A2 reajustar transformador · A3 reentrenar XGBoost ·
  A4 ambos. A3 y A4 se enmascaran si la ventana pasada tiene una sola clase.
- **Estado:** 3 días × 10 entradas (KS, PSI y caída de AUC en tres categorías, más
  validez) = 30 entradas.
- **Recompensa:** AUC diario − 0,05 × segundos / COST_UNIT.
- **Selección y evaluación:**
  - solo B-train produce gradientes;
  - B-validation cada 10 episodios;
  - checkpoint por media de tres validaciones consecutivas;
  - B-test con pesos congelados, verificados con hash.
- **Comparadores:** A1–A4, random, heurística y oracle inmediato (solo
  auditoría). Además, **DDQN v5** con sus outputs ya publicados, sobre el mismo
  replay de B-test (semilla 42).

Para no duplicar código, el notebook de PPO parte de una copia de
`kaggle/dqn-dummy/v5/input/main.ipynb`. Solo se reemplazan las celdas del agente,
el bucle de entrenamiento y la selección greedy.

## Qué cambia: el agente PPO

**Arquitectura.** Dos redes del mismo tamaño que la de DDQN, sin compartir pesos:

- **Actor:** 30 → 128 → 128 → 4 (*logits*), ReLU. Las acciones deshabilitadas
  reciben *logit* −∞ antes del *softmax*.
- **Crítico:** 30 → 128 → 128 → 1 (valor del estado), ReLU.

**Recolección.** El agente juega con su política estocástica y guarda, en cada
paso, estado, máscara, acción, log-probabilidad, recompensa y valor. Se actualiza
cada **4 episodios** (≈ 108 decisiones), recorriendo el ciclo de escenarios de v5.

**Ventaja (GAE).**

```
δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
Â_t = δ_t + γ · λ · (1 − done_t) · Â_{t+1}
```

Donde:

- `r_t`: recompensa de v5 en la decisión *t*.
- `V(s)`: valor estimado por el crítico.
- `γ = 0,99`: descuento, igual que en DDQN.
- `λ = 0,5`: cuánto se promedian las diferencias de varios pasos. Con 0,95 y 0,8 el
  agente no aprendía en una prueba aislada: el estado no incluye el día del episodio y
  el retorno pendiente agrega mucho ruido. λ = 0,5 aprende casi igual que λ = 0 (el
  objetivo de un paso que usa DDQN) y conserva crédito a varios días.
- `done_t`: 1 en la última decisión del episodio.

La ventaja `Â_t` mide cuánto mejor fue la acción tomada que lo esperado en ese
estado. Se normaliza (media 0, desviación 1) dentro de cada actualización.

**Objetivo recortado de PPO.**

```
L = E[ min( ρ_t · Â_t , clip(ρ_t, 1 − ε, 1 + ε) · Â_t ) ] − c_v · L_valor + c_e · H(π)
ρ_t = π_nueva(a_t | s_t) / π_vieja(a_t | s_t)
```

Donde:

- `ρ_t`: cuánto cambió la probabilidad de la acción tomada entre la política
  vieja y la nueva.
- `ε = 0,2`: límite del cambio permitido por actualización.
- `L_valor`: error cuadrático del crítico frente al retorno observado.
- `H(π)`: entropía de la política, que favorece la exploración.
- `c_v = 0,5` y `c_e = 0,01`; `c_e` baja linealmente a 0,001.

El recorte impide que una sola actualización cambie demasiado la política. Esa es
la diferencia práctica con DDQN, que puede cambiar la acción elegida de golpe
cuando dos valores Q se cruzan.

**Hiperparámetros iniciales:**

| Parámetro | Valor |
|---|---|
| Optimizador | Adam, 3e-4 → 3e-5 lineal |
| Épocas por actualización | 4 |
| Minibatch | 32 |
| Recorte de gradiente | norma 0,5 |
| Pérdida del crítico | MSE con retorno GAE estandarizado (media y desviación móviles) |
| Política en validación/test | greedy (acción de mayor probabilidad entre las habilitadas) |

## Presupuesto y semillas

- **Presupuesto justo:** el mismo número de episodios que DDQN (500–1000, igual
  número de pasos del entorno) y la misma regla de parada temprana de v5.
- **Semillas:** la semilla 42 primero, para comparar directamente con v5. Si el
  tiempo de Kaggle lo permite, se agregan 43 y 44 para PPO **y** para DDQN. Sin
  varias semillas no se afirma que un algoritmo supere al otro.

## Ejecución

1. **Local (smoke):** 50 camas, 8 días en B, 6 en A y 24 episodios, como el smoke
   de v5. Usa `data/processed/physionet/physionet.parquet`. Comprueba que corre de
   punta a punta y mide el tiempo por episodio.
2. **Kaggle (completo):** kernel privado, dataset
   `jeffreyamc/physionet-sepsis-processed` e internet deshabilitado. Se decide
   CPU o GPU según el tiempo medido en el smoke; la red es pequeña y el costo
   principal es XGBoost.

## Salidas

En `kaggle/ppo-dummy/v1/outputs/`, con los mismos nombres de v5 cuando aplique:

- `training_history.csv`: pérdida del actor y del crítico, entropía, retorno y
  validación.
- `checkpoint_selection.csv`, `actions.csv`, `daily_metrics.csv`,
  `policy_comparisons.csv`, `oracle_audit.csv` y `run_summary.csv`.
- `ppo_vs_ddqn.csv`: por escenario, retorno, AUC, AP, costo, acciones, *regret* y
  estabilidad de PPO frente a DDQN v5.
- Figuras PNG: entrenamiento, acciones por escenario y la comparación PPO vs DDQN.

## Criterios de aceptación

- **Integridad:** los mismos controles de v5 (particiones disjuntas, causalidad,
  máscaras, pesos congelados, AUC reconstruido desde CSV).
- **Control de que el entorno no cambió:** en B-test, las políticas fijas A1–A4,
  random y heurística dan **exactamente las mismas AUC diarias y acciones que en
  v5**. Los retornos solo pueden diferir por el tiempo medido de cada acción.
- **Sin exigencias de rendimiento:** no se exige que PPO supere a DDQN ni a las
  políticas fijas. No se ajustan hiperparámetros mirando B-test.

## Decisiones pendientes (con el profesor y Jeffrey)

1. **Cuenta del kernel:** `bikaxd/darl-ppo-dummy-v1` o bajo la cuenta de Jeffrey,
   junto a v5.
2. **Semillas:** solo la 42, o 42–44 para ambos algoritmos (triplica el tiempo).
3. **Alcance:** mantener la observación de v5 para aislar el algoritmo, y probar
   una observación continua solo en un experimento posterior.

## v2: configuración estándar de PPO

**Por qué.** PPO v1 (kernel `bikaxd/darl-ppo-dummy-v1`, versión 2) terminó íntegro, pero:

- En covariate y combined no usó nunca A2: retorno 15,16 y 15,65, frente a 17,89 y
  17,76 de DDQN.
- La fracción de recorte fue ≤ 0,012 durante todo el entrenamiento, así que la
  política casi no cambiaba en cada actualización.
- Su horizonte (≈ 108 decisiones por actualización) estaba muy por debajo del
  rango habitual de PPO (512–2048).

**Cambios (solo del agente; el entorno sigue idéntico a v5):**

| Parámetro | v1 | v2 |
|---|---|---|
| Episodios por actualización | 4 (≈ 108 decisiones) | 20 (≈ 540 decisiones) |
| Épocas K | 4 | 10 |
| Minibatch | 32 | 64 |
| λ (GAE) | 0,5 | 0,95 (valor estándar) |
| Parada temprana | desde el episodio 500 | ninguna (1000 episodios) |
| Diagnóstico | pérdidas, entropía, fracción de recorte | además, KL aproximada y ratio medio/mín/máx; figura `ppo_diagnostics.png` |

**λ = 0,95.** Con el horizonte de 540 decisiones, la prueba aislada aprende con
λ = 0,95 (probabilidad 0,82 para la acción óptima) casi igual que con λ = 0,5, así
que se usa el valor estándar.

Kernel: `bikaxd/darl-ppo-dummy-v2`; salidas en `kaggle/ppo-dummy/v2/outputs/`.

## v3: inicialización conservadora del actor

**Por qué.** En PPO v2 ningún episodio de entrenamiento terminó sin reentrenar (0 de
1000). El primer A3/A4 degrada el pipeline de forma irreversible y PPO, que aprende de
su propia política, nunca observó la alternativa "no reentrenar".

**Cambio (único frente a v2).** La última capa del actor empieza con pesos × 0,01 y
sesgo tal que P(A1) = 0,9 y 0,033 para A2–A4. Así ≈ 15 % de los episodios iniciales no
reentrena (0,933^27).

**Resultado** (kernel `bikaxd/darl-ppo-dummy-v3`, 1000 episodios; entorno idéntico a
v5: 720 filas, ΔAUC = 0 y acciones iguales al 100 %). Retorno con costo fijo común
(`kaggle/ppo-dummy/v3/outputs/ppo_vs_ddqn.csv`):

| Escenario | DDQN v5 | PPO v1 | PPO v2 | PPO v3 | Acciones PPO v3 (A1/A2/A3/A4) |
|---|---|---|---|---|---|
| stable | 17,94 | 17,94 | 14,87 | **17,94** | 27/0/0/0 |
| covariate | **17,89** | 15,16 | 14,22 | 14,38 | 19/0/8/0 |
| concept | 17,72 | 17,72 | 14,84 | **17,72** | 27/0/0/0 |
| combined | **17,76** | 15,65 | 14,23 | 14,47 | 19/0/8/0 |

- **Se confirma en parte la hipótesis de exploración.** Al observar episodios sin
  reentrenar, PPO aprende A1 en stable y concept (igual que DDQN). La correlación entre
  retorno y número de reentrenamientos pasa a −0,34/−0,38 en esos escenarios (v2:
  −0,03 a −0,10; DDQN: −0,46 a −0,54).
- **No descubre A2.** En covariate y combined usa A1 y luego A3 (8 veces); nunca A2,
  que es la mejor acción fija. En entrenamiento A2 se probó apenas 1–4 veces por
  episodio y su ventaja no superó al ruido del retorno.
- **La política se cerró pronto.** El checkpoint elegido es el último (episodio
  1000, validación 16,26), con KL ≈ 0,001 y fracción de recorte ≈ 0,01 al final.

**Lectura.** La diferencia con DDQN no es de configuración de PPO sino de cómo se
aprende: DDQN reutiliza transiciones pasadas (off-policy) y compara las cuatro acciones
en el mismo estado; PPO solo mejora lo que su política ya hace con frecuencia.

## Réplicas con semillas 43 y 44 (DDQN v5 y PPO v3)

**Diseño.** Solo cambia la semilla del agente (pesos iniciales, exploración, minibatches
y sorteo de escenarios de entrenamiento). Pacientes, XGBoost, replays y B-test siguen
con la semilla 42; el entorno es idéntico a v5 en las 4 corridas (ΔAUC = 0, acciones
iguales al 100 %). Kernels `bikaxd/darl-{ddqn-v5,ppo-v3}-seed-{43,44}`; salidas en
`kaggle/ppo-dummy/seeds/` y tabla en `kaggle/ppo-dummy/analisis/semillas_ddqn_vs_ppo.csv`.

Retorno en B-test con costo fijo común (media de los 4 escenarios):

| Agente | Semilla 42 | Semilla 43 | Semilla 44 | Media | Acuerdo de la ventana elegida |
|---|---|---|---|---|---|
| DDQN v5 | 17,83 | 17,70 | 15,27 | 16,93 | 0,69 / 0,07 / 0,13 |
| PPO v3 | 16,13 | **17,78** | 17,66 | **17,19** | 1,00 / 0,96 / 1,00 |

- **Esto corrige la lectura de la sección v3.** Con las semillas 43 y 44, PPO v3 sí
  descubre A2 en covariate y combined (17 y 5–6 veces) y queda a menos de 0,2 del
  mejor DDQN. El fallo de la semilla 42 fue de exploración, no un límite del método.
- **El DDQN es menos robusto de lo que parecía.** Con la semilla 43 usa A2 los 27
  días en todos los escenarios (pierde poco). Con la 44 casi no usa A2 y reentrena,
  y cae a 14,0–14,6 en covariate y combined.
- **PPO es más estable.** En stable y concept elige A1 en las 3 semillas (retorno
  17,93 y 17,72, siempre igual), y su política apenas cambia entre checkpoints
  seguidos (acuerdo 0,96–1,00 frente a 0,07–0,69).
- **Ninguno domina.** Cada agente tiene una semilla mala de tres; con 3 semillas no
  se puede afirmar que uno supere al otro. La diferencia útil es de estabilidad,
  a favor de PPO v3.

## Réplicas con 8 semillas (42–49)

Semillas 45–49 en CPU (límite de Kaggle: 5 sesiones CPU y 2 GPU); 43–44 en GPU. El
entorno es idéntico a v5 en las 16 corridas. Tablas en
`kaggle/ppo-dummy/analisis/semillas_ddqn_vs_ppo.csv` (por escenario) y
`semillas_resumen.csv` (por semilla); figura `semillas_ddqn_vs_ppo.png`.

Retorno medio en B-test (4 escenarios, costo fijo común). "Falla" = algún escenario
más de 1,0 por debajo de la mejor acción fija.

| Agente | Media | Desv. est. | Mediana | Mín–máx | Fallas | Acuerdo medio |
|---|---|---|---|---|---|---|
| DDQN v5 | 16,26 | 1,40 | 16,06 | 14,62–17,83 | 5 de 8 | 0,24 |
| PPO v3 | **17,16** | **0,74** | **17,49** | 16,11–17,81 | 4 de 8 | **0,98** |

- **Diferencia no significativa.** PPO − DDQN = +0,90; IC 95 % bootstrap [−0,12; 1,92];
  Mann-Whitney p = 0,27. Con 8 semillas no se afirma que un agente supere al otro.
- **Los fallos son de distinto tipo.** PPO falla solo en covariate/combined (no usa A2
  y conserva el modelo); en stable y concept elige A1 en las 8 semillas. DDQN, cuando
  falla, reentrena también en stable y concept, donde no hace falta.
- **PPO es más predecible.** Menor dispersión entre semillas (0,74 vs 1,40) y política
  casi fija entre checkpoints (0,98 vs 0,24).
