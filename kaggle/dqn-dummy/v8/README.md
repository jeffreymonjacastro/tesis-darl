# DDQN v8 — ganancia pre/post

Experimento con pacientes de ambas clínicas, un DDQN y recompensa por desempeño
posterior, mejora sobre las mismas observaciones y costo temporal de la acción.
La ejecución completa comprende 1000 episodios de 33 decisiones. Los primeros
siete días ajustan el pipeline y no se consideran evaluación.

El notebook requiere el dataset privado `jeffreyamc/physionet-sepsis-processed`
y CUDA en Kaggle. Usa XGBoost en CPU. Las probabilidades de entrenamiento y
validación solo se conservan hasta calcular la recompensa; en test se exportan
las probabilidades pre/post con identificadores y versiones del pipeline.

## Ejecución

Desde `input/`, verificar la CLI y ejecutar el kernel privado en T4:

```powershell
kaggle --version
kaggle kernels push -p . --accelerator NvidiaTeslaT4 --timeout 43200
kaggle kernels status jeffreyamc/darl-dqn-dummy-v8
```

El estado remoto `COMPLETE` solo indica que terminó el kernel: consultar también
`run_summary.json`. El estado interno `complete` exige las comprobaciones y el
test; `partial` requiere continuar y `failed` requiere diagnóstico.

El presupuesto interno es de diez horas. Para continuar un segmento `partial`,
esperar a que termine, crear otro kernel privado con el mismo notebook y dataset,
un identificador diferente y **únicamente** el kernel anterior en
`kernel_sources`. No iniciar continuaciones simultáneas. El snapshot contiene
agente, optimizador, replay, generadores aleatorios y límites de los CSV;
se rechazan cambios de identidad o de versiones de dependencias.

Descargar cada segmento en un directorio vacío separado bajo `outputs/`:

```powershell
kaggle kernels output <owner/kernel-segmento> -p ..\outputs\<segmento> --force
```

Mantener los resultados, checkpoints y logs fuera de Git. No compartir las
predicciones clínicas sin comprobar las condiciones de uso del dataset.

## Tablas y gráficas

`patient_partitions.csv` registra la partición por paciente; `partition_counts.csv`
registra capacidad de cada replay. `train_daily_metrics.csv`,
`validation_daily_metrics.csv` y `daily_metrics.csv` contienen métricas y
componentes por día. En `actions.csv` figuran acción, ventanas de entrenamiento,
tiempos, referencias de costo, versiones del pipeline y día de liberación de
etiquetas. `predictions/` contiene los pares pre/post de test;
`integrity_checks.csv` documenta las verificaciones ejecutadas.

Para reconstruir figuras desde CSV en un entorno Python con las dependencias del
notebook, establecer `DARL_DUMMY_MODE=plots` y `DARL_DUMMY_OUTPUT` con la ruta
absoluta del segmento completo, y ejecutar el notebook. Este modo no carga el
dataset ni entrena modelos.
