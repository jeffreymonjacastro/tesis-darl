# DARL DQN Dummy v3: cambios respecto a v2

Entorno, datos, etapas y recompensa son idénticos a v2. El objetivo es cumplir
`validation_converged`: acuerdo ≥95% entre los dos últimos checkpoints y
variación ≤5%. En v2 el acuerdo fue 78%.

| Parámetro | v2 | v3 |
|---|---|---|
| Episodios (mín/máx) | 200/400 | 500/1000 |
| Learning rate Adam | 1e-3 constante | 5e-4 → 5e-5 lineal por episodio |
| Objetivo Bellman | DQN | Double DQN |
| Pérdida | MSE | Huber |
| Decaimiento epsilon por episodio | 0,985 | 0,99 |
| Selección de checkpoint | mejor retorno de validación | mejor media móvil de 3 validaciones |
| Diagnóstico | — | acuerdo con tolerancia Q 0,005 (no es gate) |

Kernel `jeffreyamc/darl-dqn-dummy-v3`. Comparación generada localmente con
`.tmp/dqn-dummy/compare_v2_v3.py` en `kaggle/dqn-dummy/v3/outputs/comparison/`.
