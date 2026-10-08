# DARL PoC - Resumen de Implementación

He completado las 5 fases del plan de implementación para la Prueba de Concepto de DARL usando DQN.

## Fase 1: Pipeline XGBoost desacoplado + contrato A2
- Se creó `code/src/darl/pipeline/xgb_stage.py` que implementa `XGBStage2`, un modelo XGBoost desacoplado compatible con el preprocesamiento existente.
- Se implementó `code/src/darl/pipeline/compatibility.py` con `check_stage1_compatibility` usando distancia de Wasserstein para evaluar si el drift introducido es recuperable.
- Se actualizaron las funciones de `selective_update.py` (`run_a1_xgb`, `run_a2_xgb`, `run_a3_xgb`, `run_a4_xgb`, `run_a2_with_contract`), incorporando el cálculo de tiempo de ejecución y uso de RAM usando `tracemalloc`. `run_a2_with_contract` aplica el mapeo robusto (Mediana-IQR) y verifica la divergencia de Wasserstein como fallback (a A3/A4).
- Se modificó `pyproject.toml` para arreglar el nombre a `darl`, fijar python `>=3.10` y añadir `scipy`.
- Pruebas unitarias correspondientes aprobadas.

## Fase 2: Drift controlado
- Se actualizó el `DriftInjector` en `code/src/darl/drift/injector.py`.
- Se corrigió el bug de mutación en `transform()` mediante la clonación correcta del estado (`copy.deepcopy(meta)`).
- Se implementó la selección de variables por `selection_strategy` (`all`, `random`, `domain`, `important`).
- Se introdujo `concept_drift_method` con opciones de inyección de ruido de etiqueta: `label_flip_control` y un framework básico para `boundary_shift`.
- Pruebas unitarias correspondientes aprobadas.

## Fase 3: Monitor calibrado
- Se corrigió `psi_numeric` en `code/src/darl/monitoring/drift_metrics.py` para utilizar un rango de histograma unificado (`np.linspace(min, max)`) que contemple valores extremos en la ventana objetivo (outliers), solucionando los problemas de out-of-bounds del NumPy histogram.
- Se creó `c2st_score` (Classifier Two-Sample Test) utilizando un Random Forest para detectar divergencias multivariadas entre las ventanas objetivo y referencia.
- Se definieron `DriftReport` y `DriftMonitor` que centralizan la medición (PSI univariado, KS, y C2ST global) y exportan el estado para el MDP.
- Pruebas unitarias correspondientes aprobadas.

## Fase 4: Windowing temporal + entorno secuencial + DQN
- Se implementó el fraccionamiento en lotes consecutivos en `code/src/darl/data/window.py`, ya que TableShift descarta marcas temporales precisas.
- Se construyó `TransitionTableBuilder` en `code/src/darl/rl/transition_table.py` para generar recompensas empíricas bajo demanda, resolviendo la lentitud del reentrenamiento en vivo de XGBoost dentro del loop del DQN.
- Se reescribió por completo `DarlUpdateEnv` en `code/src/darl/rl/env.py` para leer los datos del entorno desde la tabla de transiciones pre-computada.
- Se reemplazó el uso de PPO por **DQN** en `code/src/darl/rl/training.py` según la instrucción explícita del usuario, configurando hiperparámetros como `exploration_fraction` e inicializaciones `eps`.
- El test de humo validó exitosamente que DQN logra aprender la acción óptima (`update_features`) bajo el entorno simulado.

## Fase 5: Evaluación y baselines
- Se programaron métodos deterministas en `code/src/darl/evaluation/baselines.py` (`Always A1`, `Always A4`, `Reactive C2ST`, `Random`).
- Se añadió la rutina `compare_policies()` para generar métricas consolidadas comparando DARL(DQN) vs los baselines considerando costo y AUC.
- Pruebas unitarias correspondientes superadas satisfactoriamente.

Todos los subsistemas se encuentran listos para una corrida de experimentación real (e2e) usando el generador de la tabla de transiciones sobre PhysioNet y un entorno DQN entrenado.
