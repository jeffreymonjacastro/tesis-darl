# Estado de DDQN v7

**Ejecución FULL enviada, versión remota 1; último estado comprobado RUNNING.**
No hay todavía resultados descargados ni ganador.

Kaggle creó inicialmente el slug desde el título:
[`jeffreyamc/darl-ddqn-v7-full-auc-ap`](https://www.kaggle.com/code/jeffreyamc/darl-ddqn-v7-full-auc-ap).
ID remoto: `140267881`. Se conserva esta ejecución activa; no se lanzó una
segunda para corregir el nombre. Metadata local corregida a título
`DARL DQN Dummy v7`, compatible con el slug previsto. La consolidación o
continuación deberá montar la ejecución real; no repetir entrenamiento.

Notebook enviado SHA256:
`002e687bc12d57abb4072760f4e19d31ad763ec14ffaaccfd620d8ca72818eaf`.
Metadata remota descargada en `.tmp/ddqn-v7/remote-initial/` confirma T4,
GPU habilitada, privado e internet deshabilitado. Estado interno y avance
de episodios pendientes de evidencia remota; RUNNING no prueba aprendizaje.

Seguimiento de la ejecución en `session-v1/remote_status.csv`; observador
local solo consulta estado y descargará outputs cuando el run sea terminal.
No ejecuta simulación, entrenamiento ni relanzamientos. Una consulta fallida
se registra como error de observación, sin asumir que Kaggle haya terminado.

Objetivo: nueve agentes de 1000 episodios, 200 camas y 30 días;
α/β={0.5,0.75,1}, λ=0.025. 9000 episodios y 243 000 decisiones totales.

Comprobaciones actuales: JSON/AST y compilación de once celdas de código,
sin ejecutar notebook, dataset, simulación ni entrenamiento local.

Tras ejecutar, revisar por separado:

- Estado remoto y versión exacta del slug real indicado arriba.
- `ddqn-v7/run_summary.csv`: estado interno, configuración y episodio activos.
- `remote_contracts.csv`, `initial_export_contracts.csv`, `injector_contracts.csv`.
- `sweep_summary.csv`, `training_history.csv`, `integrity_checks.csv`.
- `calibration.csv` y `calibration_qualified`, incluso si no cumple.
- `figure_manifest.csv` y revisión visual de los PNG.

`partial` requiere continuación. `failed` no valida los resultados, aunque
Kaggle permita descargar los archivos escritos. No presentar un ganador
antes de completar todos los agentes y congelar selección en B-validation.

Resultados, snapshots, logs y modelos están excluidos de Git; solo esta
documentación se versiona. No se sobrescriben los outputs v5/v6.
