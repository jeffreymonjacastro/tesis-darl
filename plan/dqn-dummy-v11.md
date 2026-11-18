# DDQN v11: veinte configuraciones nuevas en dos búsquedas paralelas

## Requisitos confirmados

- Base v10 y outputs organizados sin modificar versiones anteriores.
- Dos notebooks Kaggle privados T4: A usa cuda:0, B cuda:1. Son índices locales de máquinas independientes. La opción NvidiaTeslaT4 reserva T4 ×2 por sesión; no una GPU compartida entre notebooks. Validar explícitamente dispositivo disponible.
- Diez configuraciones nuevas FULL por notebook, 1000 episodios cada una, 300 camas, 40 días, 33 decisiones, red e input52 intactos. Semilla de entrenamiento42.
- Alpha/beta/eta/lambda independientes en [.1,1], igual reward y costo de v10. Comparar con los valores exactos del ganador3 de v10, no con sus redondeos.
- Dos procesos bayesianos independientes, cuatro puntos iniciales y seis adaptativos. Pool común determinista dividido por paridad de índice evita duplicados entre procesos; excluir los cuatro puntos ya ensayados en v10. No es un único optimizador distribuido que intercambie observaciones.
- Maximizar media reward nativo901:1000 menos1:100, sujeto a máximo40% de una acción en validación greedy (cuatro escenarios, 132 decisiones agregadas) en el checkpoint seleccionado. El usuario confirmó validación para seleccionar y test para comprobar independientemente.
- El máximo40% no garantiza que todas las acciones tengan proporción positiva: no añadir cuotas mínimas ni forzar acciones. Si no hay candidato factible, informar sin ganadora; no relajar el límite.
- GP de objetivo y GP de margen de factibilidad: Expected Improvement × probabilidad de factibilidad, buscando factibilidad mientras no haya ninguna observación factible.
- Mantener selección de checkpoint por ventana de tres validaciones entre checkpoints factibles. Guardar diagnóstico explícitamente no factible si ninguno cumple.
- Test nunca interviene en selección. Exportar cumplimiento del40% en test aunque falle; no reintentar selección con test.
- Conservar datos detallados solo del ganador local; comparar ambos ganadores al descargar luego con el mismo criterio y sin test. Evidencia compacta de los20 trials y referencia v10 permite combinar resultados.
- Dos sesiones, presupuesto11h y timeout12h cada una. El FULL anterior tardó5h10 para4 trials; diez pueden exceder una sesión. Persistir progreso íntegro y reportar partial, sin reducir episodios ni declarar completados10. No lanzar continuaciones automáticamente.
- Solo lanzar ambos notebooks y comprobar aceptación/código; no esperar finalización ni descargar resultados hasta aviso.

## Entregables y comprobaciones

- [x] v11/a/input y v11/b/input con notebook UTF-8, JSON protegido contra recodificación y metadata privada T4.
- [x] Estado y red idénticos a v10 por AST; únicamente dispositivo y restricciones nuevas.
- [x] Búsqueda, checkpoint, referencia v10 y persistencia con identidad de worker/pool/límite. Reanudación conserva generadores/offsets y no pisa estado local nuevo.
- [x] Contratos numéricos remotos: límite40%, denominador132, selección factible, excepción sin factibles, pools disjuntos/exclusiones, igualdad de propuestas tras restore, eta/costo y cierre causal.
- [x] Gráficas comparativas de incremento/nivel, factibilidad, acciones y referencia v10, con CSV derivados reproducibles.
- [x] Revisión independiente y JSON/AST local sin entrenamiento ni simulación.
- [x] Envío único de cada worker; verificar fuente remota y estado observado. Reportar enlaces, capacidades y límites sin afirmar resultados.

## Evidencia de envío

Fuentes A/B versión2 idénticas local/remoto; metadata privada T4, internet off; A RUNNING y B QUEUED. No se esperó el final. Validación local JSON/AST y regresión gráfica exclusivamente CSV tras reproducir el fallo inicial de versión1; sin entrenamiento local. Contratos numéricos finales y resultados permanecen pendientes de la ejecución. Los checks marcados corresponden a implementación/envío, no a aprobación de resultados aún no observados.
