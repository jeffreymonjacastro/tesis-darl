# DDQN v6 — barrido AUC/AP y observaciones continuas

[Protocolo](../../../plan/dqn-dummy-v6.md) · [Notebook](input/main.ipynb) ·
[Resultados](outputs/README.md)

Base: v5 local; en v6 el concepto usa inicios temporales por paciente
(`temporal_random_onset_v1`). v5 y sus resultados se conservan.
Kernel: `jeffreyamc/darl-dqn-dummy-v6`, privado, T4, internet deshabilitado.

## Configuración

- 200 camas, 30 días, 27 decisiones; nueve agentes de 1000 episodios.
- Alfa={0,0.5,1}, lambda={0.01,0.025,0.05}.
- Reward `(1-alfa)*AUC + alfa*AP - lambda*(segundos/referencia_A4)`.
- Estado: tres días de `[KS, PSI/(1+PSI), (DeltaAUC+1)/2, AP, validez]`.
- La red recibe 15 entradas. Reward y diagnósticos conservan métricas originales.
- B-validation selecciona con `0.5*AUC + 0.5*AP - 0.025*C`; B-test no decide pesos.
- Umbral F1 elegido una vez en A-validation y congelado.
- `mean_daily_ap` significa promedio diario de AP positiva válida.

## Modos

El valor de `RUN_MODE` del notebook es `full`. `DARL_DUMMY_MODE` permite
seleccionar `smoke` o `plots`; `DARL_DUMMY_OUTPUT` fija la carpeta de resultados.
El smoke usa las nueve configuraciones, 50 camas, ocho días B y 24 episodios.

Para regenerar todas las figuras localmente, ejecutar el notebook con
`DARL_DUMMY_MODE=plots` y `DARL_DUMMY_OUTPUT` apuntando a los CSV descargados.
No carga PhysioNet, no abre modelos ni modifica los CSV experimentales;
reescribe las figuras y sus tablas derivadas. Requiere las dependencias de
importación del notebook, aunque no utiliza Torch/XGBoost para ajustar modelos.

## Interpretación

Los retornos nativos de diferentes alfa/lambda expresan objetivos distintos.
Usar la utilidad común y AUC/AP/costo separados para comparar agentes.
Las bandas son desviaciones temporales, no variación entre semillas.
La calibración puede no cumplir; se conserva el respaldo de v5 y se informa
explícitamente. Esto no obliga a usar A3/A4 ni demuestra superioridad del agente.

Los archivos pesados, modelos, logs y resultados generados se conservan en
outputs y están excluidos de Git. El estado remoto COMPLETE no basta: revisar
también `run_summary.csv`, `integrity_checks.csv` y el número de episodios.

## Concept drift temporal

Solo nuevos ingresos desde el inicio conceptual: positivos con inicio
desplazado hasta ±24 h sin eliminar su evento; negativos con probabilidad
individual de conversión máxima 10 %. La intensidad se congela al ingreso.
Las etiquetas sintéticas siguen `0...0,1...1`; SepsisLabel no se modifica.
Los metadatos completos son exclusivos del simulador y no entran en la red.

Auditorías por trayectoria en `concept_audits/`; tablas `concept_patients`,
`concept_timelines` y `drift_calendar`. Las figuras 13–16 muestran trayectorias,
desplazamientos, prevalencia y conteos por escenario. Se regeneran desde CSV
sin ajustar modelos. Los outputs por defecto usan la subcarpeta versionada
`temporal_random_onset_v1`; los anteriores corresponden al inyector histórico.

Esta modificación se comprobó con pacientes ficticios, sin entrenar modelos
ni ejecutar Kaggle. [Prueba y gráficos](outputs/concept-temporal-check/README.md).
No demuestra mejor aprendizaje del DDQN ni resuelve el fallo histórico de
auditoría macro-F1 de la ejecución remota anterior.
