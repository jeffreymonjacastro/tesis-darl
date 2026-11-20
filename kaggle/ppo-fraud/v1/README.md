# PPO fraude v1 — contraparte de DDQN fraude v2

Mismo diseño que Jeff con PhysioNet (DDQN vs PPO): el entorno es idéntico y **solo cambia el agente**.

- Entorno: el de [DDQN fraude v2](../../ddqn-fraud/v2/README.md) (IEEE-CIS, 40 días, 33 decisiones, entrada continua de 136 valores,
  1000 episodios, semilla 42, selección por validación con ventana de tres checkpoints).
- Reward: α=0,96724, β=0,10531, η=0,92157, λ=0,10062 (los de DDQN v11 B y PPO v4 de Jeff).
- Agente: el PPO de [`ppo-dummy/v3`](../../ppo-dummy/v3/input/main.ipynb) sin cambiar hiperparámetros: actor y crítico 128-128, GAE γ=0,99
  λ=0,95, recorte 0,2, 10 épocas, minibatch 64, actualización cada 20 episodios, Adam 3e-4→3e-5, entropía 0,01→0,001,
  arranque con P(A1)=0,9.

**Limitación:** el notebook de PPO v4 de Jeff (el que usa estos pesos de reward) no está en el repositorio, solo sus figuras.
Por eso se usan los hiperparámetros de v3. Si v4 los cambió, hay que ajustarlos para que la comparación sea exacta.

La política se llama `dqn` en los CSV internos por compatibilidad con el entorno; en las figuras aparece como `ppo`.

```powershell
kaggle kernels push -p . --accelerator NvidiaTeslaT4 --timeout 43200
```
