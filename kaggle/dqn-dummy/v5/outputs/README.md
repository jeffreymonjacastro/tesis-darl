# Outputs DARL DQN v5

Resultados del kernel privado `jeffreyamc/darl-dqn-dummy-v5`.
Versión remota 4: `COMPLETE`; estado interno `complete`, modo `full`.
1000 episodios, checkpoint 350. Controles de integridad aprobados;
calibración y estabilidad del checkpoint no cumplidas.
La ejecución smoke se conserva en `.tmp/dqn-v5/kaggle-smoke/`;
esta carpeta contiene la ejecución completa y su auditoría de descarga.

## Archivos

| Archivo | Contenido |
|---|---|
| `run_summary.csv` | Versión, modo, configuración, dependencias, avance, estado interno y errores |
| `xgboost_search.csv` | Ocho configuraciones: comprobación temporal y selección A-validation |
| `a_metrics.csv`, `a_daily_metrics.csv` | Rendimiento inicial de A, agregado y diario |
| `patient_partitions.csv` | Asignación disjunta por paciente y presencia de sepsis |
| `partition_counts.csv` | Pacientes disponibles/utilizados y filas consumidas por replay |
| `calibration.csv`, `cost_calibration.csv` | Velocidades candidatas y cinco tiempos A4 |
| `training_history.csv`, `checkpoint_selection.csv` | Entrenamiento, validación greedy, checkpoint y estabilidad |
| `daily_metrics.csv`, `actions.csv` | AUC de ambas etiquetas, KS/PSI, acciones, máscaras y estados pasados |
| `policy_comparisons.csv` | Comparación de DDQN, A1–A4, random y heurística |
| `drift_calendar.csv` | Intensidad de medición y adelanto sintético por día |
| `oracle_audit.csv` | Alternativas inmediatas, utilidad, elección y regret offline |
| `controller_freeze.csv` | Hash del agente y ausencia de aprendizaje en validation/test |
| `integrity_checks.csv` | Controles realizados dentro del experimento |
| `download_validation.csv` | Auditoría independiente de predicciones descargadas |
| `kaggle_execution.csv` | Estado remoto, versión del kernel y evidencia de descarga |
| `experimental_assessment.csv` | Diferencias frente a baselines, regret y coincidencia de acción con oracle |

`predictions/` contiene A-test y los cuatro escenarios B-test para las siete
políticas. Son 29 CSV escritos incrementalmente, con paciente, ICULOS,
hora/día simulado, ambas etiquetas, probabilidad, versión y acción del pipeline,
y disponibilidad de etiqueta. Estos archivos por fila están ignorados por Git.

`figures/` contiene exclusivamente PNG. AUC vacío representa días sin ambas
clases, cuya recompensa solo penaliza el costo. Las etiquetas concept/combined
son cambios artificiales del objetivo: usar AUC original para distinguirlo.

La integridad no implica superioridad de DDQN. Consultar resultados e
interpretación en [README del experimento](../../README.md) y el
[protocolo](../../../../plan/dqn-dummy-v5.md).
