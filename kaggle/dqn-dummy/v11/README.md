# DDQN v11: dos búsquedas paralelas con restricción de acciones

Base: v10, mismos 52 inputs, red y simulación (300 camas, 40 días, siete días iniciales, 33 decisiones; 1000 episodios por agente). Reward y parámetros independientes [0.1,1] intactos. Semilla42 para entrenamiento.

## Dos notebooks

- `a/input/main.ipynb`: worker0, cuda:0, diez configuraciones nuevas.
- `b/input/main.ipynb`: worker1, cuda:1, diez configuraciones nuevas.

Kaggle asigna T4 ×2 a cada sesión NvidiaTeslaT4; son máquinas separadas. El índice CUDA es local y se valida sin fallback silencioso. No comparten una pareja de tarjetas. No se entrena XGBoost en GPU: conserva CPU con dos hilos.

Cada worker realiza una búsqueda bayesiana independiente con cuatro puntos iniciales y seis adaptativos. Se divide un pool determinista común de candidatos por paridad; se excluyen los cuatro ensayos de v10. Esto evita duplicados y produce veinte configuraciones nuevas si ambas búsquedas terminan. No es un optimizador distribuido con observaciones compartidas.

## Selección acordada

Objetivo: promedio del reward nativo901–1000 menos promedio1–100, sujeto a que ninguna acción del checkpoint elegido supere40% de las132 decisiones greedy de validación (cuatro escenarios por33 decisiones). El usuario confirmó que se usa validación para seleccionar, y test únicamente para comprobar después.

Dentro de un trial se selecciona el último checkpoint de la mejor ventana de tres validaciones entre checkpoints factibles. Si no existe uno factible, se guarda un diagnóstico con la menor concentración, explícitamente no factible. BO usa GP de objetivo y de margen0.4−máxima_proporción; EI×probabilidad de factibilidad, priorizando hallar factibilidad al inicio. Entre trials solo candidatos factibles pueden ganar; empate conserva el primero.

No se cambian acciones para fabricar proporciones. El límite40% no obliga a que aparezcan todas las acciones: 40/40/20/0 cumple. Si ninguna configuración es factible tras diez trials, se publica complete_no_feasible sin ganadora ni selección usando test. Un candidato puede incumplir el40% en test aunque cumpla en validación: se exporta ese resultado sin volver a seleccionar. El límite se aplica al agregado de cuatro escenarios; no se exige equilibrio por escenario estable.

La referencia es el ganador3 de v10 con sus coeficientes exactos guardados, crecimiento6.285418898506938 y checkpoint930; se reutilizan sus métricas y acciones, sin volver a entrenarlo. La referencia no obtiene excepción al límite. Ambos workers publican resúmenes con identidad para poder reunir los20 ensayos después; los ganadores globales se compararán usando el mismo objetivo y factibilidad de validación, nunca el test.

## Outputs y duración

Cada worker conserva figures/, predictions/, models/, checkpoints/, analysis/{training,validation,test,comparisons,plot_data}/ y metadata/{configuration,preparation,checks}/. Los detalles se conservan solo para su ganadora local; series compactas de todos los ensayos y referencia permiten reproducir comparaciones. Nuevas figuras21/22 muestran curvas v10/nuevas y crecimiento frente al límite40%.

Presupuesto11h por sesión y timeout12h. V10 tardó5h10 para cuatro trials: diez podrían superar una sesión. Guardar partial con pesos/buffer/RNG/offsets, estadoBO y trial activo; no reducir episodios ni declarar diez completados. Las continuaciones requieren montar el partial correspondiente junto a v8 y v10, y se lanzarán solo cuando el usuario lo solicite. Una búsqueda completa representa10000 episodios/330000 decisiones por worker; ambas,20000/660000.

Estado de preparación: JSON/AST y fuente revisados sin entrenamiento local ni smoke. Contratos numéricos y resultados reales pendientes del FULL en Kaggle. La calibración heredada sigue siendo una limitación separada de la integridad. Una semilla, cuatro pesos y diez puntos por worker no demuestran un óptimo global; el crecimiento del reward también cambia con la escala de los pesos.

El usuario pidió solo lanzar ambos notebooks, sin esperar a que terminen ni descargar outputs. Enlaces y recibos se registrarán al enviar.

## Envíos realizados

Versión2 vigente de ambos notebooks. A observado RUNNING; B observado QUEUED, todavía sin iniciar. Los envíos se solicitaron en paralelo; el comienzo efectivo depende de Kaggle. No se esperará a que terminen ni se lanzarán versiones adicionales para sortear la cola.

La versión1 de A falló al generar la gráfica inicial por dtype object tras concatenar un DataFrame vacío. Se reprodujo el error con los CSV existentes; la corrección elimina el concat vacío y fuerza arrays numéricos. Pruebas gráficas de referencia sola y con trial completado pasaron sin modelos locales. B también terminó temprano; se actualizó al mismo arreglo. No se completó ningún FULL con esas primeras versiones. Los recibos conservan esta incidencia y las identidades de la fuente vigente.
