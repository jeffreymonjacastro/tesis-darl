# DDQN fraude v1 — protocolo v8 sobre IEEE-CIS Fraud Detection

Copia del notebook [DDQN v8](../dqn-dummy/v8/README.md) de PhysioNet para el segundo caso de la
tesis. El notebook se generó desde el de v8: las celdas de utilidades, persistencia, DDQN, entorno,
entrenamiento, selección, figuras y verificaciones son las mismas. Solo cambian los datos, el
inyector de concepto, la codificación de categóricas en f_φ y un tope en el drift covariable.

## Qué se mantiene igual que v8

- Cuatro escenarios (`stable`, `covariate`, `concept`, `combined`) y cuatro acciones:
  A1 nada, A2 reajustar f_φ y la referencia del monitor, A3 reentrenar XGBoost con el f_φ vigente,
  A4 ambas. El pipeline es persistente.
- Episodios de 40 días: días 1–7 de ajuste, 33 decisiones (días 8–40), ventana de 7 días previos
  y etiquetas con 1 día de retraso. Drift desde el día 8 (test) o entre 8 y 12 (train).
- Reward `AUC_post + AP_post + (ΔAUC + ΔAP) − 0,025 · tiempo/tiempo_A4`.
- Estado de 15 entradas: 3 días × [KS, PSI/(1+PSI), (AUC_ref − AUC + 1)/2, AP, validez].
- DDQN 15–128–128–4, Adam 5e-4→5e-5, γ = 0,99, 1000 episodios, ciclo de escenarios de v8,
  selección por B-validation con media de tres checkpoints. Baselines A1–A4, aleatoria y heurística.
- f_φ numérico `(x − mediana)/IQR` recortado a [−2, 2], faltantes a la mediana.
- Búsqueda de 8 combinaciones de XGBoost y calibración del drift con las mismas grillas y criterios.

## Qué cambia y por qué

| Pieza | v8 (PhysioNet) | Fraude v1 |
|---|---|---|
| Datos | 7 signos vitales por hora | Top 30 por \|SHAP\| (21 numéricas, 9 categóricas) de la base final: sin reloj, sin D1–D15, sin IDs de cliente, clientes hasta su primer fraude |
| Partición | Por paciente 60/20/20 | Por tiempo: train en ventanas que empiezan los días 1, 14, 27 y 40 (hasta el 79); validation 81–120; test 143–182 |
| Episodio | Replay de 300 camas | 40 días reales consecutivos |
| Holdout inicial | Pacientes de train | Días 8–14 (ajuste en 1–7) |
| f_φ categóricas | No hay | Codificación por frecuencia, que A2 reajusta |
| IQR = 0 | Se fuerza a 1e-6 | Se usa la desviación estándar (muchos conteos tienen IQR 0) |
| Covariate | Desplazamiento de `cov·IQR` sin tope y escala hasta ×1,4 | Igual, con **tope de 3 IQR** para que la ventana de 7 días alcance al drift |
| Concepto | Adelanto del inicio de sepsis y conversión de negativos | Target shift: cada día una fracción `0,3·intensidad` de los fraudes fuera del segmento pasa a legítima, y el mismo número de legítimas del segmento (W, débito, gmail, monto bajo la mediana) pasa a fraude. Prevalencia constante y X intacta |
| Etiqueta original | `SepsisLabel` | `isFraud` |

Prueba local previa (políticas fijas, episodio de test, sin DDQN): con el tope, A2 recupera el
covariate (AP 0,13 → 0,50) y A3 no (0,23); concept se resuelve con A3. Con λ = 0,025, A3/A4
siguen ganando por retorno en stable y covariate, igual que en PhysioNet.

## Ejecución

Dataset privado `bikaxd/ieee-fraud-darl` (`ieee_fraud_darl.parquet`). Desde `input/`:

```powershell
kaggle kernels push -p . --accelerator NvidiaTeslaT4 --timeout 43200
kaggle kernels status bikaxd/darl-ddqn-fraud-v1
```

Las continuaciones `partial` se hacen como en v8. `DARL_DUMMY_MODE=smoke` corre 3 episodios en CPU
local solo para comprobar el código (no son resultados). Los outputs se descargan a
`outputs/<segmento>/` y no se versionan, salvo su README.
