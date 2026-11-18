# DDQN v10: optimización bayesiana de cuatro pesos del reward

**Objetivo:** implementar v10 desde el notebook local de v9 y enviar una ejecución
privada FULL a Kaggle T4. El usuario solicita detener el seguimiento después del
envío; no esperar la finalización ni descargar resultados hasta su aviso.

**Arquitectura:** conservar el simulador, la preparación compartida y el DDQN de
v9. Entrenar agentes independientes para puntos sugeridos secuencialmente por
un optimizador bayesiano. Exportar CSV detallados solo de la configuración
ganadora y gráficos comparativos de todas las configuraciones completadas.

**Tecnologías:** PyTorch CUDA, XGBoost CPU de dos hilos, pandas, NumPy, SciPy y
GaussianProcessRegressor de scikit-learn; no requiere instalar un optimizador
externo ni habilitar internet en Kaggle.

## Especificación y restricciones

- Base: `kaggle/dqn-dummy/v9/input/main.ipynb`. Conservar todas las versiones anteriores.
- 300 camas, 40 días, warmup de siete días, 33 decisiones y 1000 episodios por agente, semilla 42.
- Entrada exactamente igual al FULL de v9: tres bloques de 17 métricas más horizonte, 52 valores; misma normalización y orden. No optimizar la arquitectura ni el estado.
- Reward: `alpha*AUC_post + beta*AP_post + eta*(delta_auc+delta_ap) - lambda*C`, con `C=seconds/reference_A4`.
- Cuatro parámetros continuos independientes, cada uno en `[0.1, 1]`. Una sola eta, aplicada a ambas diferencias. No imponer `alpha+beta=1`.
- Conservar mismo manejo de días inválidos, referencias, máscaras, particiones, inyectores, replays, calendario, hiperparámetros DDQN y selección por tres validaciones consecutivas.
- Entrenar solo en Kaggle con GPU; comprobaciones locales JSON/AST. No smoke ni entrenamiento local.
- JSON UTF-8 con escapes Unicode al serializar el notebook para evitar la recodificación cp1252 observada en Kaggle CLI de Windows. Verificar ida/vuelta del código enviado antes de afirmar que coincide.
- Preservar estados parciales al límite de sesión; nunca reducir episodios ni declarar ganador final de una búsqueda incompleta.

## Decisiones confirmadas por el usuario

1. **Presupuesto confirmado:** cuatro configuraciones FULL de 1000 episodios. Dos puntos iniciales y dos propuestas adaptativas GP/EI. V9 tardó 9127 segundos: cuatro equivalen a 36 508 segundos sin evaluación final; hay que permitir estado parcial si se alcanza el límite de sesión, sin reducir la búsqueda.
2. **Objetivo aprobado:** maximizar `mean(train_return[901:1000]) - mean(train_return[1:100])` sobre los 1000 episodios completos. Es diferencia de retornos nativos, no utilidad fija de v9. En empate exacto se conserva el primer trial completado. Los pesos afectan la escala de este criterio; no equivale a superioridad predictiva absoluta.
3. **Checkpoint dentro de un trial:** conservar v9: pesos del último checkpoint de la mejor ventana de tres validaciones consecutivas, con sus desempates AP/AUC/tiempo. La elección entre los cuatro trials usa exclusivamente la diferencia aprobada de entrenamiento; test no interviene.

## Tareas de implementación y evidencias de aceptación

- [x] Crear `kaggle/dqn-dummy/v10/input/main.ipynb`, metadata y README; conservar celdas temáticas y docstrings.
- [x] Introducir configuración explícita alpha/beta/eta/lambda; parametrizar entorno, evaluación, CSV, verificador y títulos. Distinguir retorno nativo de las métricas de referencia heredadas; BO y selección global usan solo el criterio aprobado de incremento del retorno nativo.
- [x] Implementar búsqueda bayesiana secuencial mediante proceso gaussiano Matérn y adquisición Expected Improvement. Inicialización reproducible, candidatos dentro de límites, exclusión de duplicados y sugerencias posteriores condicionadas a las puntuaciones de validación anteriores. Documentar cantidad de puntos iniciales según presupuesto confirmado.
- [x] Reiniciar agente, optimizador, buffer y generadores por configuración; compartir preparación, replays, calendario, denominador A4 y trayectorias de validación. No consultar test durante la búsqueda.
- [x] Persistir estado del optimizador bayesiano, ensayos completados y trial activo; conservar snapshots DDQN y offsets para una continuación consistente. Separar elección final de un mejor provisional.
- [x] Congelar configuración y checkpoint con validación; evaluar solo la ganadora en test contra A1–A4, Random y heurística, sobre los mismos pacientes/etiquetas.
- [x] Generar comparaciones de curvas nativas y del criterio fijo, parámetros/puntuaciones y evolución del mejor valor; etiquetar escalas y bandas temporales sin afirmar variación entre semillas.
- [x] Conservar gráficas del ganador heredadas de v9 y añadir acciones por episodio, promedios, secuencia diaria y comparación con políticas de acción fija.
- [x] Guardar exclusivamente CSV detallados del ganador. Mantener solo un resumen compacto de búsqueda y series derivadas de comparación, sin tablas por decisión ni predicciones de agentes perdedores. Documentar estas excepciones de comparación en el README.
- [x] Implementar contratos de aritmética en límites/punto interior, eta compartida, estados y persistencia. Programar su ejecución dentro del FULL remoto: aprobación numérica aún no observada. La selección global usa la diferencia de entrenamiento aprobada; test queda aislado.
- [x] Validar localmente JSON/AST, referencias de archivos, dependencias disponibles, UTF-8, metadata privada T4 y ausencia de código de entrenamiento local.
- [x] Enviar `jeffreyamc/darl-ddqn-v10-bayesian-reward` y comprobar aceptación del envío/identidad del código. **No esperar a que termine.** Entregar enlace, configuración lanzada y estado observado.

## Organización de outputs

```text
outputs/
├── README.md
├── figures/
├── predictions/
├── models/
├── checkpoints/
├── analysis/
│   ├── training/
│   ├── validation/
│   ├── test/
│   ├── comparisons/
│   └── plot_data/
└── metadata/
    ├── configuration/
    ├── preparation/
    └── checks/
```

Evitar CSV redundantes: en v9 `DailyEnv.daily` y `DailyEnv.actions` reciben el
mismo registro de decisión. Mantener una sola tabla canónica por fase y que
figuras/verificación la consulten. Guardar medias móviles, bandas y varianzas
en `analysis/plot_data/`; consolidar auditorías de concept drift por paciente
en `metadata/preparation/` y contratos/manifiestos en `metadata/checks/`.

Los checkpoints de búsquedas parciales pueden contener el estado transitorio
necesario para continuar; los outputs finales completos conservarán únicamente
los detalles del ganador y las comparaciones compactas de la búsqueda.

## Riesgos concretos detectados al revisar v9

- La persistencia actual recorre solo `OUT/*.csv` y `predictions/*.csv`. Debe adaptarse a rutas anidadas con un inventario explícito o recursivo: restaurar offsets/tablas y detectar faltantes en todas las carpetas nuevas. Probar este contrato dentro de Kaggle antes del entrenamiento.
- No borrar detalles de un mejor provisional mientras siga siendo candidato a ganador. Conservar su historial, métricas, tiempos medidos y checkpoint seleccionado; reemplazarlos solo cuando una configuración completada lo supere. Los pesos seleccionados no permiten reconstruir el historial de aprendizaje ni sus tiempos originales.
- En una búsqueda parcial, identificar el mejor como provisional. No presentar como ganador final hasta completar todos los ensayos presupuestados y congelar selección antes del test.
- Vincular atómicamente estado BO, sugerencia activa, trial completado, pesos y offsets. BO y selección global usarán la misma diferencia de promedios de entrenamiento aprobada, conservando el primer trial en empate. La validación solo selecciona pesos dentro de cada trial.
- Forzar al menos una sugerencia adaptativa; con cuatro trials se fijan dos iniciales. Normalizar parámetros para el GP, añadir jitter/ruido, rechazar puntuaciones no finitas y candidatos duplicados. El tiempo medido sigue siendo ruidoso aunque la semilla sea fija.

## Envío realizado

Versión 1, kernel ID 144497826, estado observado RUNNING. Fuente remota idéntica a la local y metadata privada T4 sin internet. Recibo: kaggle/dqn-dummy/v10/outputs/submission.json. Sin entrenamiento local, sin smoke, sin esperar ni descargar resultados. Los contratos numéricos y PNG reales siguen sujetos a la ejecución FULL; no se afirman aprobados todavía.
