# DARL DQN Dummy v1: plan de implementación revisado

## 1. Propósito y alcance

Demostración académica de aprendizaje por refuerzo para el mantenimiento de un
pipeline tabular. Se utilizarán mediciones existentes de PhysioNet como entradas
numéricas y una etiqueta binaria artificial, independiente de la etiqueta clínica.
El objetivo es verificar cambios de distribución, recuperación de AUC, costo de
actualizaciones y decisiones del agente mediante resultados reproducibles.

La demostración no valida predicción clínica, no modifica pacientes ni constituye
un procedimiento biológico. Los cambios sintéticos son transformaciones numéricas
de copias del dataset dentro del experimento. Las únicas operaciones externas serán
las necesarias para ejecutar el notebook privado y recuperar sus resultados en la
cuenta Kaggle autorizada del usuario, mediante su CLI oficial.

Esta revisión aclara el alcance; no busca ocultar el contenido ni eludir controles.
Los avisos de comprobación adicional no prueban una infracción y la redacción no
garantiza que desaparezcan. Fuente oficial:
https://help.openai.com/en/articles/20001326-additional-safety-checks-for-biological-and-cybersecurity-requests-in-chatgpt-codex-and-the-api

### Resultado esperado

- Notebook independiente, explicado en español, ejecutado de principio a fin.
- Cuatro acciones funcionales y auditables, con costo real en segundos.
- Gráficos de reward, distribución, AUC, acciones y regret, revisados visualmente.
- Comparación con baselines sobre pacientes reservados.
- Informe explícito de criterios cumplidos e incumplidos, sin fabricar convergencia.

## 2. Entregables y límites de cambios

```text
kaggle/dqn-dummy/
├── README.md
└── v1/
    ├── input/
    │   ├── main.ipynb
    │   └── kernel-metadata.json
    └── outputs/
        ├── main.executed.ipynb
        ├── run_summary.json
        ├── training_history.csv
        ├── daily_metrics.csv
        ├── actions.csv
        ├── action_audit.csv
        ├── policy_comparison.csv
        └── figures/
```

Toda lógica experimental estará dentro del notebook, con Markdown antes de cada
bloque y sanity checks incrementales. No importar `darl` ni adjuntar `darl-package`.
No modificar `code/src/darl/`, experimentos anteriores, tesis ni notebook con cambios
pendientes del usuario. Las pruebas temporales irán en `.tmp/dqn-dummy/`.
Agregar únicamente regla de exclusión necesaria para outputs de la nueva demo.
No realizar commit o push. No guardar credenciales ni registros individuales del
dataset en entregables para presentación; usar conteos y métricas agregadas.

## 3. Datos y simulación

### 3.1 Carga y validación

Fuente: `jeffreyamc/physionet-sepsis-processed`, ya autorizada y accesible en Kaggle.
Resolver `physionet.parquet` desde `/kaggle/input/`; admitir ruta local para pruebas.
Leer únicamente `patient_id`, `source_set`, `ICULOS`, `SepsisLabel`, `HR`, `O2Sat`,
`Temp`, `SBP`, `MAP`, `DBP` y `Resp`.
Comprobar esquema, duplicados paciente–ICULOS, orden temporal y ambas clínicas.
Registrar versiones y dispositivos. `ICULOS` no será predictor.

### 3.2 Particiones

Con semilla 42, separar pacientes completos, estratificados por presencia de sepsis
en su historia: A 70/15/15 para entrenamiento inicial/validación/prueba y B 60/20/20
para entrenamiento DQN/validación/prueba. Dividir cada partición B en pacientes de
actualización y evaluación al 50%. `SepsisLabel` se conserva y solo se utiliza para
esta estratificación; el predictor se entrenará con `y_dummy`.
Comprobar ausencia de pacientes compartidos entre particiones y entre roles.

### 3.3 Replay horario

Simular 16 días con 200 camas, 100 por rol. Recorrer registros de cada paciente en
orden de ICULOS. Tras consumir su última observación, incorporar siguiente paciente
en siguiente hora. No reutilizar pacientes dentro de una simulación ni interpolar
horas faltantes. Si se agota cola de pacientes, detener y registrar diagnóstico.
Cada día completo: 4.800 filas y 2.400 filas de evaluación. Inferir sobre las 200
camas; AUC principal se calcula en pacientes de evaluación.
Los días y recambios son dispositivos de replay, no calendario ni altas reales.

### 3.4 Faltantes y etiqueta artificial

Forward fill causal dentro del paciente; imputar faltantes restantes mediante
medianas ajustadas únicamente sobre datos autorizados. Fijar coordenadas robustas
con mediana/IQR de A-train. Antes de aplicar cambios de medición, calcular score con
pesos `[1.2, -1.1, 0.3, -0.2, -0.7, 0.2, 0.8]` en orden de los siete vitales.
Usar mediana del score de A-train como umbral e invertir 5% de etiquetas con RNG 42.
Guardar `y_dummy` separada; conservar `SepsisLabel` intacta. Identificadores,
etiquetas y escenario no serán features.

## 4. Pipeline y cambios estadísticos controlados

### 4.1 Transformación y predictor

Transformar cada variable mediante
`clip((x_imputed - median) / max(iqr, 1e-6), -2, 2)`.
XGBoost inicial: A-train, 60 árboles, profundidad 3, learning rate 0,1,
`tree_method="hist"`, CPU, dos hilos y semilla 42.

| Acción | Transformador | XGBoost |
|---|---|---|
| A1 defer | Conservar | Conservar |
| A2 update features | Reajustar | Conservar |
| A3 update model | Conservar | Reentrenar |
| A4 retrain all | Reajustar | Reentrenar |

Actualizar con filas del día actual de pacientes de actualización. Conservar
pipeline entre decisiones. Medir duración efectiva con `perf_counter`; registrar
fracción de variables saturadas por clipping. Verificar invariantes de A1–A4.

### 4.2 Escenarios

- Estable: sin transformación de mediciones ni etiquetas.
- Covariate: cambiar ubicación y escala de mediciones; conservar etiqueta dummy.
- Concept: conservar mediciones; invertir etiqueta dummy, `1 - y_dummy`.
- Combinado: aplicar ambas modificaciones.

Transformación: `m_ref + scale * (x - m_ref) + offset * iqr_ref`.
Mantener faltantes y registrar parámetros. Esta transformación opera sobre copias
numéricas y no describe una intervención en pacientes.

Seleccionar severidad únicamente con B-train: offsets 3, 4 y 5 IQR, escalas 1,4 y
1,8. Elegir combinación menos intensa que produzca degradación y recuperación
medibles. Si ninguna lo logra, registrar criterio incumplido. Congelar parámetros
antes de B-test; no reajustar mirando prueba final ni modificar rewards para
favorecer acciones.
Entrenamiento: inicio del cambio entre días 5–8 y severidades variables.
Prueba: días 1–4 estables; días 5–16 con cambio persistente.

## 5. Monitoreo y entorno DQN

### 5.1 Observaciones

Calcular KS y PSI por vital, agregados mediante máximo, y AUC previa a acción.
PSI: deciles de referencia congelados, límites abiertos y suavizado numérico.
Registrar también prevalencia artificial y saturación.

| Métrica | Categoría 1 | Categoría 2 | Categoría 3 |
|---|---|---|---|
| KS | <0,10 | 0,10–0,30 | ≥0,30 |
| PSI | <0,10 | 0,10–0,25 | ≥0,25 |
| AUC | <0,65 | 0,65–0,85 | ≥0,85 |

Codificar tres métricas con one-hot, agregar indicador de AUC válida y concatenar
tres días: 30 entradas. Los umbrales son operativos para esta demo. Las categorías
son representaciones observables, no estado verdadero del POMDP. Régimen y estado
interno del pipeline permanecen ocultos al agente.

### 5.2 Secuencia temporal y reward

Inferir día d, disponer de etiquetas al cierre, construir observación, elegir
acción y ajustar con pacientes de actualización. Evaluar resultado en día d+1.
Al cierre de d+1, insertar `(obs, action, reward, next_obs, done)` en replay.

`reward = auc_next_day - 0.1 * action_time_seconds`

Si solo existe una clase, registrar AUC inválida, conservar última AUC válida para
observación y excluir transición del aprendizaje. No inventar AUC medida.
Documentar disponibilidad de etiquetas al cierre como simplificación de demo.

### 5.3 Agente de referencia de clase

Clases dentro del notebook: `QNetwork`, `ReplayBuffer`, `DQNAgent` y entorno con
`reset()`/`step(action)`. Dos capas de 128 neuronas, ReLU y cuatro salidas sin
activación. Replay FIFO 50.000, batch 64, Adam 0,001, gamma 0,99.
Epsilon inicia en 1,0, decae por actualización mediante 0,995 y mínimo 0,01.
Target explícitamente congelada y fuera del optimizador; hard update cada 50
actualizaciones. Un paso de gradiente por transición, clipping como en notebook
de clase y bootstrap cero para terminales.

Entrenar 200 episodios balanceados entre escenarios. Evaluar greedy en validación
cada diez; guardar mejor checkpoint por retorno. Continuar hasta máximo 400 si
validación no cumple criterios. No incluir oracle ni selección supervisada en
decisiones del agente.

## 6. Pruebas, gráficos y criterios de éxito

### 6.1 Evaluación congelada

Comparar DQN, A1–A4 permanentes, random y heurística sobre mismo replay B-test.
Heurística: A1 estable, A2 alerta covariate, A3 AUC baja, A4 ambas.
Alerta: KS ≥0,30 o PSI ≥0,25; AUC baja <0,85.
Medir AUC, retorno, segundos y número/tipo de actualizaciones.

Para cada decisión DQN, clonar pipeline previo y evaluar cuatro acciones sobre
mismos datos autorizados y mismo día siguiente. Calcular oracle inmediato,
regret y fracción a ≤0,01 de máxima utilidad. Oracle es exclusivamente diagnóstico,
no política óptima de largo plazo. Separar interiores y fronteras del régimen.

### 6.2 Sanity checks

Comprobar particiones, ocupación, recambio, causalidad del relleno, conservación
de features/etiquetas por escenario, invariantes A1–A4, aritmética de reward,
capacidad y shapes de replay, congelamiento de target, copia hard y terminales.
Validar formato notebook y compilar celdas antes de ejecución.

### 6.3 Figuras PNG/SVG 16:9

1. Ocupación y recambio, con identificadores de replay anónimos.
2. Calendario de cambios estadísticos.
3. KS, PSI y AUC con umbrales.
4. Retorno por episodio, media móvil, evaluación greedy y epsilon.
5. AUC y reward acumulado por política.
6. Línea temporal de acciones DQN.
7. Utilidad de cuatro acciones, decisiones y regret.
8. Saturación y recuperación tras actualización de features.

Mostrar datos crudos y suavizados por separado. Revisar imágenes renderizadas:
signos, unidades, series, leyendas, escalas, cortes, tamaño y correspondencia con
CSV. Gráficos significativos requieren resultados medidos, no curvas prescritas.

### 6.4 Gates explícitos

- Ejecución completa y cuatro acciones verificadas.
- Retorno DQN positivo en prueba.
- Utilidad DQN superior a A1 y random en escenarios con cambio.
- ≥80% de decisiones próximas al oracle en interiores del régimen.
- A1–A4 seleccionadas durante evaluación; reportar cualquier ausencia.
- Variación relativa ≤5% en últimos cinco checkpoints greedy y política estable.

Si falla un gate, conservar evidencias y marcar `needs_improvement`; no afirmar
convergencia por reward positivo solamente. Definir variación relativa como rango
de retornos medios dividido por valor absoluto de su media, con piso 1e-6; definir
estabilidad de política como ≥95% de coincidencia entre acciones de últimos dos
checkpoints sobre replay de validación fijo.

## 7. Kaggle y entrega

Kernel privado `jeffreyamc/darl-dqn-dummy-v1`, notebook, dataset único
`jeffreyamc/physionet-sepsis-processed`, T4 solicitada. DQN usa CUDA si disponible;
XGBoost usa CPU. Registrar dispositivos y dependencia del costo respecto a hardware.

1. Ejecutar sanity check local reducido.
2. Subir notebook mediante skill Kaggle y CLI oficial desde `v1/input/`.
3. Monitorear ejecución y persistir progreso después de etapas costosas.
4. Descargar outputs; confirmar `COMPLETE` remoto y estado interno de ejecución.
5. Obtener notebook ejecutado con tablas y figuras.
6. Recalcular métricas desde CSV e inspeccionar visualmente ocho figuras.
7. Escribir README con resultados observados, imágenes y límites de interpretación.

Ante comprobación adicional de ChatGPT/Codex, conservar trabajo y seguir opciones
oficiales del producto. Si bloqueo repetido impide tarea autorizada, reportar
mensaje exacto y recomendar feedback/soporte con descripción redactada, sin
credenciales ni datos sensibles. No intentar desactivar o sortear controles.

La entrega tendrá enlace Kaggle, notebook explicado y ejecutado, gráficos revisados
y tabla de gates con evidencia. Ningún éxito se declarará sin comprobar archivos,
ejecución y correspondencia de resultados.
