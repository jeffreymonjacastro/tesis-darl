# DDQN fraude v2 — protocolo v11 B sobre IEEE-CIS Fraud Detection

Parte del notebook de [fraude v1](../README.md) y le incorpora lo de [DDQN v11 B](../../dqn-dummy/v11/README.md).
Datos, inyector de drift, f_φ con categóricas, pipeline, baselines y calibración son los de fraude v1.

## Qué cambia respecto a fraude v1

| Pieza | Fraude v1 | Fraude v2 |
|---|---|---|
| Entrada | 15 (KS máx., PSI máx., AUC, AP, válido × 3 días) | 136: KS y PSI de cada una de las 21 numéricas + AUC, AP, válido (45 por día) × 3 días + horizonte restante |
| Reward | `α=β=η=1`, λ=0,025 | α=0,96724, β=0,10531, η=0,92157, λ=0,10062 (η común para ΔAUC y ΔAP) |
| TD | sin diagnóstico | `td_diagnostics.csv` y verificación de objetivo terminal |
| Episodios | 1000 | 1000 |

La entrada de PhysioNet (52 = 3×17+1) tenía 7 signos vitales; aquí hay 21 numéricas, por eso 136.
Una característica sin valores ese día se reporta con KS=PSI=0 y su bandera `monitor_valid_<x>`.
Sin búsqueda bayesiana ni restricción del 40 %: la selección de checkpoint es la de v1 (mejor ventana de
tres validaciones por reward). `max_action_share` se exporta solo como diagnóstico.

## Gráficas

`12_reward_entrenamiento`, `13_reward_por_escenario` y `14_acciones_validacion` replican el estilo de v11 B (validación greedy del checkpoint elegido). Además `01_aprendizaje`, `02_retorno_por_escenario`, `03_acciones`, `03b_distribucion_acciones_dqn` y 04–11 como en v1.
`03b_distribucion_acciones_dqn` (DDQN global, por escenario y en entrenamiento) y 04–11 como en v1.

## Ejecución

```powershell
kaggle kernels push -p . --accelerator NvidiaTeslaT4 --timeout 43200
```

Smoke local (3 episodios, CPU, solo verifica el código): `DARL_DUMMY_MODE=smoke`.
