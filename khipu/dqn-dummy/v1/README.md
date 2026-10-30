# DDQN Khipu v1 — beneficio de actualizar sobre los mismos registros

Base: `kaggle/dqn-dummy/v7/input/main.ipynb`. Se conservan el inyector temporal,
el preprocessing robusto, XGBoost CPU y Double DQN 15–128–128–4. El protocolo
experimental cambia a pacientes mezclados A+B y reward con ganancia pre/post.

## Protocolo

- Semilla 42; split por paciente 60/20/20 estratificado por clínica y sepsis.
- Cuatro replays de entrenamiento; uno fijo de validación y uno de test.
- 300 camas, registros horarios, 40 días. Días 1–7 ajustan ambas etapas;
  DDQN decide días 8–40: 33 decisiones por episodio, 33 000 en 1000 episodios.
- Selección XGBoost: ocho candidatos de v7, solo primeros siete días de un
  holdout interno de pacientes de train. Umbral F1 y referencias congelados.
- Cada replay tiene una base propia ajustada en sus siete primeros días.
  Se reutiliza su copia inicial al comenzar otro episodio, sin arrastrar ajustes.
- Ventana de actualización: siete días anteriores; etiquetas llegan en t+1.
  Historia DDQN: tres bloques, inicialmente neutros con validez cero. Los días
  de ajuste no se presentan como evaluación prequential.
- Deriva desde día 8–12 en train y día 8 en validation/test; combinado conserva
  el retraso conceptual de dos días. Se mantienen las intensidades candidatas
  y reglas de cualificación de v7; la calibración usa exclusivamente train.
- Validación cada diez episodios. Selección por retorno medio de los cuatro
  escenarios y tres checkpoints consecutivos; desempate por AP, AUC, menor
  tiempo y checkpoint más temprano. Test se evalúa después de congelar selección.
- Baselines independientes: A1, A2, A3, A4, aleatoria y heurística.

## Reward

`reward = alpha*AUC_post + beta*AP_post + gain - lambda*cost_relative`

`gain = eta_auc*(AUC_post-AUC_pre) + eta_ap*(AP_post-AP_pre)`

α=β=ηAUC=ηAP=1 y λ=0.025. Pre y post predicen exactamente los mismos registros
del día t; ambas métricas se calculan al recibir sus etiquetas en t+1. La
última liberación ocurre en día 41. AP es average precision positiva.

A1 reutiliza las probabilidades. A2 cambia preprocessing, A3 clasificador y A4
ambas etapas. Solo se ejecuta la acción elegida; no hay oracle contrafactual
ni estimación de regret. Las políticas baseline son episodios separados.

El costo es tiempo de la acción dividido entre la mediana de cinco A4 sobre
siete días de train. Las inferencias pre/post se cronometran aparte. La base
inicial y la calibración tampoco se suman al costo de cada decisión. Días sin
ambas clases reciben solo `-lambda*cost_relative`, con ganancia NaN y validez
explícita. El retorno puede ser positivo incluso con ganancia negativa.

## Ejecución en Khipu

Desde la raíz del checkout en `feature/ddqn-experiments`:

```bash
bash khipu/dqn-dummy/v1/input/prepare.sh
mkdir -p .tmp
sbatch --job-name=darl-v1-pilot --time=01:00:00 \
  --export=ALL,DARL_DUMMY_MODE=pilot,DARL_DUMMY_BUDGET_SECONDS=3000 \
  khipu/dqn-dummy/v1/input/run.sbatch
# FULL, después de aprobar el piloto:
sbatch khipu/dqn-dummy/v1/input/run.sbatch
```

El piloto conserva 300 camas, 40 días, datos y reward reales; usa 30 episodios
para cubrir tres validaciones y selección/test. No acredita el FULL de 1000.
El sbatch solicita 4 CPU, 16 GiB, un shard RTX A6000 y hasta ocho horas.
Ejecuta Python directamente en el batch: la prueba de `srun` con shards en
este clúster asignó recursos pero falló al crear el paso de ejecución.

El entorno dedicado usa Python 3.13, PyTorch 2.15.0 y CUDA 13.2. Las versiones
directas y transitivas están fijadas en `requirements.lock`; `environment.txt`
registra las instaladas. No se necesita resolver todas
las dependencias del monorepo para ejecutar este notebook.

El presupuesto de FULL es 7,5 horas, con snapshots por episodio completo y
por política de test. Si termina `partial`, reenviar el mismo sbatch reanuda
en el mismo directorio. El snapshot restaura optimizador, replay, RNG y límites
de CSV y rechaza cambios de notebook, dataset, dependencias principales o modo.
Un fallo requiere diagnóstico antes de reenviar. Nunca ejecutar dos jobs
contra el mismo directorio de outputs al mismo tiempo.

Logs Slurm: `.tmp/darl-ddqn-v1-JOBID.log`. Resultados: `outputs/pilot` y
`outputs/full`. `run_summary.csv` distingue `partial`, `failed`,
`pilot_complete` y `complete`. La ganancia predictiva no está garantizada.

## CSV y reproducción local

`train_actions.csv` y `train_daily_metrics.csv` conservan una fila por decisión
y episodio. `validation_*` conserva episodios de validación; `actions.csv`,
`daily_metrics.csv` y `policy_comparisons.csv` corresponden al test. Los alias
`auc`, `average_precision` y `probability` significan post; no reemplazan los
campos explícitos pre/post. `common_reward` coincide con reward en este
experimento de configuración única.

Columnas principales: `auc_pre/post`, `average_precision_pre/post`,
`delta_auc/ap`, `gain_auc_component`, `gain_ap_component`, `gain`,
`auc_component`, `ap_component`, `time_s`, `cost_unit_s`, `cost_relative`,
`cost_penalty`, `reward`, `gain_evaluable`, días de entrenamiento y recepción
de etiquetas, acción solicitada/ejecutada, máscaras, estado y versiones.

`predictions/test_ESCENARIO_POLITICA.csv` conserva claves de registro,
etiquetas original/sintética, `probability_pre/post` y versiones. Las parejas
de train/validation permanecen en memoria hasta liberar el reward, sin
exportar cientos de millones de filas. Los CSV de particiones, calendario,
calibración, configuración y drift permiten auditar y regenerar las figuras.

Tras descargar el directorio de resultados, ejecutar desde la raíz local:

```powershell
$env:DARL_DUMMY_MODE = 'plots'
$env:DARL_DUMMY_OUTPUT = 'khipu/dqn-dummy/v1/outputs/full'
.venv/Scripts/python.exe -c "import nbformat; n=nbformat.read('khipu/dqn-dummy/v1/input/main.ipynb',as_version=4); exec(compile('\n'.join(c.source for c in n.cells if c.cell_type=='code'),'<notebook>','exec'))"
```

Este modo no lee el Parquet ni ajusta modelos. Exporta aprendizaje con media,
DE temporal y varianza de ventana 100; retorno por escenario; acciones;
comparación de políticas; AUC pre/post; componentes del reward; monitoreo;
deriva; epsilon/Huber; distribuciones de vitales y DDQN frente a A1.
Las bandas temporales de una semilla no son incertidumbre entre semillas.

Los resultados generados y modelos quedan excluidos de Git. Los archivos
importantes requieren una copia externa: los discos Khipu no ofrecen backup.
