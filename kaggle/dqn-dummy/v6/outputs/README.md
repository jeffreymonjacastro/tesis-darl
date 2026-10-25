# Resultados DDQN v6

## Cambio temporal del inyector

El código actual usa `temporal_random_onset_v1`. Los resultados de entrenamiento
documentados más abajo pertenecen al inyector anterior y no validan este cambio.
La nueva implementación solo se comprobó mediante historias ficticias y
regeneración CSV; no se ejecutaron modelos ni Kaggle.

[Prueba temporal y gráficos](concept-temporal-check/README.md).
Las futuras ejecuciones guardarán sus resultados en carpetas separadas.

## Evidencia histórica disponible

Smoke local en CPU, semilla 42, 50 camas, ocho días B y 24 episodios por
configuración: **completo**. Los resultados del código final están en
`local-smoke-final/`; la corrida local previa se conserva en `local-smoke/`.

| Comprobación | Resultado |
|---|---|
| Agentes y episodios | 9 agentes, 216 episodios, 5 decisiones por episodio |
| Contratos focalizados | 11 aprobados; `local-contracts.log` |
| Integridad de CSV, causalidad, máscaras y rewards | Aprobada por la auditoría interna |
| Predicciones exportadas | 61 archivos |
| Figuras | 147 PNG, incluidas las ocho familias solicitadas |
| Regeneración sin datasets/modelos | Aprobada; CSV experimentales sin cambios; `local-plots-check.log` |
| Calibración | No cualificada: se utilizó el respaldo previsto |

Estos resultados comprueban el funcionamiento del protocolo pequeño. No son
evidencia de rendimiento del full ni de ejecución en T4. Las comparaciones
de medias diarias y de días con decisión tienen columnas distintas.

## Ejecución remota

Kernel privado: [darl-dqn-dummy-v6](https://www.kaggle.com/code/jeffreyamc/darl-dqn-dummy-v6),
identidad numérica `140191279`. La versión 2 contiene el smoke con código
actualizado, T4, internet deshabilitado y timeout de 1200 segundos. Se reutilizó
la misma identidad para corregir el título y slug de la primera versión.

Última consulta: smoke en `QUEUED`, sin logs. **El full todavía no se ha
ejecutado ni se declara completo.** La consulta de cuota al 22/10/2026 informó
11078 segundos de GPU disponibles aproximadamente; reinicio el 24/10/2026
a las 00:00 UTC. La capacidad efectiva se revisará antes del full.

Configuración esperada: nueve combinaciones alfa={0,0.5,1},
lambda={0.01,0.025,0.05}, 1000 episodios cada una, 200 camas y 30 días.

Consultar `run_summary.csv` para estado interno, modo, etapa, errores,
configuraciones completadas, referencia de costo y calibración. Ver
`sweep_summary.csv` y `selection_freeze.csv` para la selección de validación;
`policy_comparisons.csv` contiene los resultados independientes de B-test.

`figure_manifest.csv` lista las figuras. `training_smoothing.csv`,
`scenario_smoothing.csv`, `action_averages.csv`, `drift_histograms.csv` y
`concept_decile_relationship.csv` conservan los datos de las visualizaciones.

Se conservan los CSV/predicciones/modelos/PNG locales sin versionarlos en Git.
La corrida final local incluye la columna explícita `phase=train` y separa
las medias diarias de las correspondientes a los días con decisión.
