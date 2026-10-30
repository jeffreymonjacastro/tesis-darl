# DDQN v7 — α y β independientes, λ=0.025

[Notebook](input/main.ipynb) · [Protocolo](../../../plan/dqn-dummy-v7.md) ·
[Estado y resultados](outputs/README.md)

Base: v6 local, conservado intacto. Ejecución FULL completa: nueve agentes,
9000 episodios. Kernel real privado: `jeffreyamc/darl-ddqn-v7-full-auc-ap`, T4
e internet deshabilitado. [Resultados y gráficas](outputs/analysis-20261027/README.md).
La metadata local usa el título corregido para el slug previsto.

## Experimento

- Nueve agentes: producto de α={0.5,0.75,1} y β={0.5,0.75,1}.
- Reward `α*AUC + β*AP - 0.025*(segundos/referencia_A4)`.
- Cada agente: 1000 episodios, 200 camas, 30 días, 27 decisiones por episodio.
- Estado continuo: tres días de `[KS, PSI/(1+PSI), (AUC_ref-AUC+1)/2, AP, validez]`.
- XGBoost CPU con dos hilos dentro de Kaggle; DDQN obligatoriamente CUDA.
- Selección B-validation: `0.5*AUC + 0.5*AP - 0.025*C`, media de tres
  checkpoints consecutivos y cuatro escenarios con igual peso. Test reservado.
- Concept drift: adelantos de positivos hasta 24 h y conversión de negativos
  hasta probabilidad 10 %, intensidad congelada al ingreso; originales intactos.
- Sin estratificación adicional del replay ni garantía de positivos diarios.

## Ejecución

El notebook rechaza entrenamiento fuera de Kaggle, sin GPU o en smoke.
Ejecutar únicamente `full`; `plots` sirve para regenerar figuras desde CSV.
No hay ajuste local de modelos ni simulación local del experimento.

```powershell
cd kaggle/dqn-dummy/v7/input
kaggle kernels push -p . --accelerator NvidiaTeslaT4 --timeout 43200
```

Presupuesto interno de diez horas. Si termina como `partial`, la versión
siguiente debe montar únicamente este kernel como `kernel_sources` y conservar
el mismo código experimental, cambiando `SEGMENT_VERSION`, slug y título.
Detecta automáticamente `resume_manifest.json` en el input del predecesor.
No crear continuación mientras el kernel anterior siga activo.

El snapshot conserva optimizador, replay buffer, todos los RNG y límites de
CSV. Los activos preparados conservan también muestras y histogramas KS/PSI.
Un predecesor `failed` exige diagnóstico; no se reanuda automáticamente.
Una parada controlada no reinicia los 1000 episodios ni acredita completitud.

## Resultados y gráficos

CSV largos incluyen α/β/λ y componentes del reward. AP positiva,
PR-AUC trapezoidal y `mean_daily_ap` tienen nombres y significados distintos.
Se exportan F1 positivo/macro, precision, recall, balanced accuracy, prevalencia
y soportes, contra objetivo sintético y etiqueta original.

Ocho familias: drift de vitales/concepto, comparación de rewards con ventana
100, retorno por escenario con ventana 15, monitoreo, barras de acciones,
AUC por política, decisiones/auditoría inmediata y DDQN frente a A1.
Figuras adicionales muestran trayectorias temporales del inyector y Huber.
`training_smoothing.csv` conserva media, desviación estándar y varianza.
Las bandas ±1 DE son temporales y no variación entre semillas.

`DARL_DUMMY_MODE=plots` reconstruye figuras desde los CSV de
`DARL_DUMMY_OUTPUT`, sin dataset, modelos ni entrenamiento. En este proyecto,
la regeneración de esta ejecución se realizará también en Kaggle.

## Validación

Compilación/AST local no prueba entrenamiento. Contratos numéricos y
reconstrucción del A-test se comprueban dentro del run FULL antes de DDQN.
El barrido solo estará completo con nueve agentes, 9000 episodios,
243 000 decisiones, exportaciones verificadas y ocho familias renderizadas.
Kaggle COMPLETE, estado interno, calibración e integridad se informan separados.
No se exige superioridad de DDQN ni curvas crecientes para aprobar integridad.
