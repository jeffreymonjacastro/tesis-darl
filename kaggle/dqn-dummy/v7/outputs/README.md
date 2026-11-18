# Resultados DDQN v7 — ejecución FULL completa

**Estado remoto: COMPLETE. Estado interno: complete. Integridad: aprobada.
Calibración del drift: no cualificada.**

[Hallazgos y galería de gráficas](analysis-20261027/README.md) ·
[Resultados originales](download-20261027/ddqn-v7/) ·
[Recibo de descarga](download-20261027/download_receipt.json) ·
[Inventario](download-20261027/download_inventory.csv)

## Ejecución y descarga

- Kernel real: [`jeffreyamc/darl-ddqn-v7-full-auc-ap`](https://www.kaggle.com/code/jeffreyamc/darl-ddqn-v7-full-auc-ap?scriptVersionId=361248980).
- Versión remota 1, scriptVersionId 361248980, kernel id 140267881.
- T4/CUDA, FULL, 200 camas, 30 días, semilla 42; XGBoost CPU en Kaggle.
- Duración interna: 26037.2 segundos, aproximadamente 7 h 14 min.
- Nueve agentes de 1000 episodios: **9000 episodios y 243 000 decisiones**.
- Descarga del 27 de octubre de 2026: **439 archivos remotos**,
  245 CSV, 155 PNG; 1,892,213,407 bytes (1.89 GB).
- No hicieron falta v8/v9 ni otro entrenamiento. No se reentrenó localmente.

Kaggle generó el slug desde el título inicial. Este es el job realmente
analizado; el slug previsto `darl-dqn-dummy-v7` no identifica esta ejecución.
La metadata local conserva el título corregido para eventuales envíos futuros.

## Selección y hallazgos

Seleccionada exclusivamente en B-validation: **α=0.75, β=0.5, λ=0.025**;
checkpoint 890, después de entrenar sus 1000 episodios.
Utilidad común de selección: 8.857755; segundo lugar 8.850070.
La diferencia es pequeña y no demuestra superioridad estadística entre pesos.

| Escenario B-test | AUC diario A1 | AUC diario DDQN |
|---|---:|---:|
| Estable | 0.6655 | 0.6655 |
| Covariate | 0.5398 | 0.6711 |
| Concept | 0.6157 | 0.6157 |
| Combinado | 0.5565 | 0.6245 |

La política seleccionada concentra las actualizaciones en A2. No demuestra
adaptación conceptual mediante reentrenamiento. Frente a A2 constante, DDQN
logra mayor utilidad común con AUC ligeramente inferior y aproximadamente
63.7 % menos tiempo de mantenimiento medido por Pipeline.act().

## Gráficas legibles

Los 155 PNG originales tienen algunos títulos con caracteres dañados.
Se preservan. `analysis-20261027/figures/` contiene 27 copias legibles desde
CSV —las ocho familias para el seleccionado y comparación de nueve agentes—
y una figura adicional de validación greedy. No se cargaron modelos.

![Validación greedy](analysis-20261027/figures/18_validacion_greedy.png)

![AUC DDQN frente a A1](analysis-20261027/figures/a0p75_b0p50_l0p025/09_auc_dqn_vs_a1.png)

## Verificación y límites

74 comprobaciones remotas aprobadas. En el análisis se reconstruyeron los
nueve criterios de selección, los rewards/resúmenes de test y los 9000
retornos desde las 243 000 decisiones de entrenamiento. Los 155 PNG remotos
son archivos válidos; se revisaron visualmente las ocho familias del agente
seleccionado, sin inspeccionar individualmente todos los gráficos restantes.

Calibración no cualificada; shift covariable extremo al final, incluida MAP
sintética negativa; modelo inicial A-test AUC=0.66184, AP=0.04270. Una semilla
y una trayectoria B-test por escenario. No extrapolar el ganador a otros
experimentos ni interpretar las bandas temporales como intervalos entre semillas.

Resultados, snapshots, logs y modelos permanecen fuera de Git; solo se
versiona la documentación autorizada. v5 y v6 se conservaron.
