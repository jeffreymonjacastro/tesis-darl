# DDQN v9: horizonte observable y monitoreo por signo vital

V9 conserva el entorno y el reward del FULL de v8, y añade una representación
explícita del horizonte y diagnósticos del aprendizaje DDQN. El protocolo completo
está en [el plan](../../../plan/dqn-dummy-v9.md).

## Reward y datos compartidos

`r = AUC_post + AP_post + G - 0.025 × C`, donde
`G = (AUC_post - AUC_pre) + (AP_post - AP_pre)` y `C = segundos / referencia_A4`.
Se conserva el signo positivo de G confirmado por el usuario. Un día sin ambas
clases recibe únicamente la penalización de costo y G queda no evaluable.

Se monta el kernel privado completo `jeffreyamc/darl-dqn-dummy-v8` para reutilizar
su preparación: particiones, replays, pipelines iniciales, referencias, umbral F1,
calibración, calendario y denominador del costo. Se exige coincidencia del código
base, hash del dataset y versiones de dependencias. Se inicia un DDQN nuevo: no
se cargan pesos, optimizador ni buffer del controlador v8.

Se conservan 300 camas, 40 días, siete días iniciales y siete días de datos
etiquetados para actualizar. Son 33 decisiones por episodio, 1000 episodios,
semilla 42, DDQN CUDA en T4 y XGBoost CPU de dos hilos. La calibración heredada
puede no cumplir sus criterios y se reporta independientemente de la integridad.

## Observaciones

El modo predeterminado `per_vital` concatena, del día más antiguo al más reciente,
tres bloques de 17 entradas: KS de HR/O2Sat/Temp/SBP/MAP/DBP/Resp, PSI de esos
mismos vitales, caída de AUC respecto a la referencia, AP positiva y validez
predictiva. Cada PSI se transforma como `PSI/(1+PSI)`; la caída AUC como
`(AUC_ref-AUC+1)/2`. KS/AP/validez conservan sus rangos naturales.

Se añade una única entrada de horizonte: `h=(40-día+1)/33`. Vale 1 antes de la
decisión del día 8, 1/33 antes de la decisión del día 40 y 0 en el siguiente estado
terminal, el día 41. La entrada de la red tiene **52 valores**.

La variable `DARL_DUMMY_OBSERVATION_MODE=aggregate` conserva los máximos KS/PSI,
tres bloques de cinco entradas y el mismo horizonte: **16 valores**. Cada modo
necesita un entrenamiento independiente; esta ejecución usa `per_vital`, no
demuestra el efecto aislado de añadir el horizonte ni evalúa el modo agregado.

El orden es fijo. Se exporta la validez del monitoreo de cada vital. En modo por
variable, una referencia o muestra diaria ausente produce un error explícito en
vez de introducir un cero que pueda interpretarse como ausencia de drift. Los
tres bloques artificiales iniciales están marcados con validez predictiva cero.

## Diagnósticos, persistencia y figuras

- Q para las cuatro acciones en entrenamiento, validación y test; estado,
  horizonte y marcas `terminated`/`truncated` por decisión.
- `td_diagnostics.csv`: cada 100 gradientes, resúmenes ponderados de muestras
  terminales y no terminales con Q, objetivo, error TD firmado/absoluto, RMSE y
  Huber. Se comprueba que objetivos terminales sean su recompensa.
- Acumuladores y límites de los CSV dentro del snapshot reanudable; al finalizar,
  se comprueba `muestras TD = actualizaciones × 64`.
- Once familias heredadas de figuras y dos nuevas: Q/TD y monitoreo por vital.
  Todas se reconstruyen exclusivamente desde CSV con `DARL_DUMMY_MODE=plots`.
- Selección de checkpoint con tres validaciones consecutivas y cuatro escenarios
  ponderados por igual; test posterior con controlador congelado.

## Ejecución autorizada

Solo se permite `full` o `plots`; no hay piloto/smoke ni entrenamiento local.
Desde `input/`, con Kaggle CLI autenticada:

```powershell
kaggle kernels push -p . --accelerator NvidiaTeslaT4 --timeout 43200
kaggle kernels status jeffreyamc/darl-dqn-dummy-v9
```

El kernel es privado, con internet deshabilitado. Se conserva la imagen Docker
de v8. El presupuesto interno es de diez horas con snapshot persistente.
Una continuación solo restaura snapshots del protocolo v9 y el mismo esquema;
el montaje de v8 completo nunca se interpreta como un snapshot para continuar.

Los outputs se descargan en una carpeta nueva bajo `outputs/`. Los estados
remoto, interno, integridad, calibración y rendimiento se verifican por separado.
No se exige que v9 supere v8 ni que utilice todas las acciones para considerar
correcta la implementación.

## Estado

Notebook ejecutado en [Kaggle, versión 1](https://www.kaggle.com/code/jeffreyamc/darl-dqn-dummy-v9).
El FULL terminó con estado remoto e interno **complete**, en Tesla T4, tras
9127 segundos (2 h 32 min). Se completaron **1000 episodios y 33 000 decisiones**,
28 evaluaciones de políticas en test, 97 comprobaciones de integridad y 16
contratos numéricos. No hubo entrenamiento ni inferencia local.

Se descargaron y verificaron los 213 archivos remotos, aproximadamente 1.58 GB,
en [el bundle original](outputs/full-20261105/ddqn-v9/). El recibo de descarga
incluye tamaños y SHA-256; se comprobó el CRC del snapshot sin deserializar modelos.
Las 109 comprobaciones independientes verificaron estados/Q finitos, horizonte,
reward, muestras TD, preparación idéntica a v8 y las identidades/etiquetas de los
28 archivos de predicciones de test (237 600 registros cada uno).

Se seleccionó el checkpoint **740** exclusivamente en validación, frente al 940
de v8. La misma fórmula permite comparar directamente sus retornos nativos:

| Indicador | v8 | v9 |
|---|---:|---:|
| Mejor media de tres validaciones | 21.6845 | 21.9155 |
| Última media de tres validaciones | 21.1929 | 21.8171 |
| Brecha mejor/última ventana | 0.4916 | 0.0984 |
| Desviación temporal, últimos diez checkpoints | 0.2662 | 0.2009 |
| Media de \|Q − reward terminal\|, últimas diez validaciones | 59.5247 | 0.4042 |
| Retorno medio de test | 21.4488 | 21.4840 |
| AUC diaria media de test | 0.5918 | 0.5941 |
| AP diaria media de test | 0.0568 | 0.0577 |

La mejora principal está en la coherencia de las estimaciones Q y en el menor
retroceso al final de validación. La mejora predictiva de test es pequeña:
el retorno aumenta 0.0351, AUC 0.00233 y AP 0.00088. No es uniforme: el retorno
covariate aumenta de 20.5586 a 21.0116, mientras combined baja de 21.9440 a 21.5485.
Las acciones de test son A1=6, A2=14, A3=96 y A4=16 (132 decisiones).

El [informe comparativo](outputs/comparison-v8-v9/README.md) contiene tablas,
CSV y tres figuras de comparación. Se revisaron visualmente las 13 figuras
[regeneradas desde CSV](outputs/figures-regenerated-20261105/figures/) y las
figuras comparativas. El bundle remoto se conserva intacto.

**Calibración:** `calibration_qualified=False` en ambas versiones. Se conserva
como limitación independiente de la integridad. Una semilla y dos cambios de
observación simultáneos no permiten atribuir la mejora solo al horizonte ni
afirmar superioridad estadística general. La comparación Q/retorno inicial usa
una trayectoria observada por política, no valores Q verdaderos. No se probó una
interrupción/reanudación deliberada; sí se verificaron la implementación de la
persistencia, el manifiesto final y la integridad del snapshot descargado.

El CLI de Windows recodificó únicamente las etiquetas de figuras del código
remoto. Se comprobó coincidencia del AST algorítmico y se conserva un recibo de
ambos hashes. Las figuras locales se regeneraron con el texto correcto, sin
repetir entrenamiento. Los cortes de red durante la descarga se resolvieron
reanudando transferencias; no se lanzó otra ejecución en Kaggle.
