# Transcripción de la asesoría — Semana 3

**Audio:** `Asesoria_w3.mp3`  
**Participantes:** Jeffrey, Ariana y Dayane  
**Duración:** 29:50

> Los fragmentos que no pudieron distinguirse con suficiente claridad están marcados como `[inaudible]`.

00:00 - 00:08 [Ariana]: "Bien, perfecto. Pueden empezar compartiendo su pantalla y vamos conversando."

00:15 - 00:27 [Jeffrey]: "Ok. Bueno, hicimos esta presentación siguiendo los avances semanales. La agenda es esta."

00:28 - 01:11 [Jeffrey]: "Empezando con la recapitulación: la semana pasada Dayane no tenía mucho contexto, entonces fue una semana de explicación y también estuvimos pensando en los datos mientras hacíamos la presentación grupal. Por eso hicimos la presentación grupal y un pequeño roadmap que hemos estado conversando durante el ciclo."

01:14 - 01:51 [Jeffrey]: "Primero quería recordar los objetivos para preguntarles si les parecen bien. Nuestro objetivo general es implementar este framework de toma de decisiones, que se centra en actualizar selectivamente los pipelines con base en la detección del drift y la acción correcta, de forma que pueda mejorar o restablecer el desempeño del modelo."

01:53 - 02:18 [Jeffrey]: "Como mencionamos, el objetivo específico era implementar el módulo que detecta el tipo de drift, de modo que podamos tenerlo mapeado, y luego aprender una política; es decir, definir bien el entorno donde el agente va a actuar y las cuatro acciones que se van a realizar para que el agente pueda aprender."

02:21 - 02:51 [Jeffrey]: "También queremos evaluar las políticas aprendidas, los retornos y las métricas de reinforcement learning, como el valor, el action-value, la cantidad de veces que se eligen las acciones, el costo de acción y el regret, sobre todo. Entonces, ¿está de acuerdo, profe, con estos objetivos?"

02:51 - 02:53 [Ariana]: "Sí, sí, parecen coherentes."

02:53 - 02:57 [Jeffrey]: "Creo que al menos el ciclo pasado la profesora me dijo que los afine un poco. Más o menos."

03:03 - 03:04 [Ariana]: "Un segundo."

03:06 - 03:08 [Ariana]: "¿Y qué comentario te dio?"

03:09 - 03:28 [Jeffrey]: "Lo que pasa es que en el póster que había colocado los objetivos eran muy generales; o sea, eran más técnicos, y me dijo si podía afinarlos para esto."

03:32 - 03:37 [Jeffrey]: "A ver. A ver."

03:38 - 03:50 [Ariana]: "[Lee parte del contenido de la diapositiva; inaudible]. Aquí el problema con los objetivos es que se sienten más como pasos."

03:51 - 03:52 [Jeffrey]: "Ok."

03:54 - 04:17 [Ariana]: "Por ejemplo, en el primer objetivo colocas implementar el módulo de monitoreo que detecta y clasifica covariate shift y concept drift, y cuantifica su severidad. Punto; ahí estamos. Que sus salidas formen el vector de observación del agente es algo que necesitamos, pero no es parte del objetivo en particular: es una consecuencia de haber cumplido ese objetivo."

04:22 - 04:40 [Ariana]: "En cuanto al siguiente, formalizar la actualización selectiva como cuatro acciones, tendría cuidado con mencionar el número. Es algo que todavía no tenemos claro y probablemente quisiéramos que sea variable; no quisiéramos que esté ligado a un número particular."

04:42 - 05:05 [Ariana]: "Ok. Entonces, mide recuperación y costo. Perfecto. El tercero es medir la política aprendida bajo drift y severidad conocidos, contrastándola contra una tabla de decisión empírica y el baseline de un umbral fijo. Usualmente los objetivos tienen que estar relacionados con un resultado, con algo interpretable."

05:08 - 05:25 [Ariana]: "Por ejemplo, la implementación nos va a dar un módulo como resultado. De esta formalización esperamos alguna formulación matemática que nos lleve o nos permita tener convergencia."

05:26 - 05:55 [Ariana]: "Pero cuando dices ‘medir’, no tenemos ninguna hipótesis de resultado, porque podrías medir de manera positiva o negativa. Podríamos reestructurar el objetivo, por ejemplo: contrastar la política aprendida bajo drift y severidad conocidos contra una tabla de decisión empírica. No nos interesa solamente medirlo, ¿cierto? Nos interesa comparar o contrastar."

06:07 - 06:34 [Jeffrey]: "Bueno, en el aspecto de gaps y soluciones, he estado conversando un poco sobre lo que colocamos al final de los next steps de la diapositiva grupal y, consultándolo con la IA, más o menos quiero comentar estos gaps."

06:36 - 07:25 [Jeffrey]: "El primero es la compatibilidad de Stage 1 y Stage 2. Nos mencionaste que un modelo aprende una distribución X de datos; si se actualiza a X prima, ¿la función va a tener el mismo desempeño o va a seguir funcionando? Es algo que vamos a investigar y ver cómo solucionarlo. Por ejemplo, se podría implementar algún contrato que verifique la semántica y se asegure de que el modelo reciba cierto tipo de input, y que al actualizarlo mediante el feature update no se vuelva incompatible. Igual es un gap que tenemos que investigar más."

07:31 - 07:50 [Ariana]: "En este punto les recomiendo verificar si es posible e invertir, al menos hasta la semana 4, en resolverlo. Si no es posible o nos damos cuenta de que no hay una forma razonable de hacerlo, podemos deshacernos de esa acción, ¿ok?"

07:57 - 08:06 [Ariana]: "Vale la pena que tengan un timeline interno para saber hasta cuándo vamos a buscar que esto sea razonable. ¿De acuerdo?"

08:07 - 08:13 [Jeffrey]: "Ok, sí. Justo eso vamos a conversar durante esa semana."

08:15 - 08:17 [Ariana]: "Ok, perfecto. Continúa."

08:18 - 09:05 [Jeffrey]: "El siguiente gap es el inyector de concept drift. Hemos visto que, para la experimentación, cada dataset es independiente. Si queremos hacer experimentaciones realistas, tenemos que aplicar concept drift con base en el dataset. Por ejemplo, en diabetes readmission, cierto conjunto de variables puede hacer que un caso sea considerado uno en lugar de cero. Entonces, el concept drift debería adaptarse mejor a cada dataset para enriquecer la experimentación. ¿De acuerdo?"

09:07 - 09:43 [Jeffrey]: "Luego está el entorno no secuencial. El semestre pasado hice una simulación inyectando drift y severidades aleatorias para ver cómo actuaba el agente. Ahora buscamos un entorno más secuencial, donde el drift aumente si no se actúa a tiempo, y evaluar esa condición. ¿De acuerdo?"

09:44 - 10:17 [Ariana]: "Les recomiendo fuertemente que la inyección de concept drift y el entorno secuencial se trabajen en paralelo, porque necesitamos un entorno de evaluación. Incluso si todavía no tenemos esa inyección de concept drift, ya deberíamos poder entrenar. ¿Por qué no llegamos a hacer todo en el semestre pasado? Teníamos el ambiente y un agente de RL inicial, pero todavía no tenemos un MVP o una demo de cómo este entrenamiento afecta el resultado."

10:18 - 10:35 [Ariana]: "Yo priorizaría tener esa demo primero. Incluso antes de la inyección y del entorno secuencial podemos entrenar el modelo con nuestras circunstancias actuales. Tenemos una recompensa, o un entorno, que permite maximizar la recompensa inicial."

10:39 - 10:50 [Ariana]: "Aunque sea un algoritmo muy tonto o las soluciones no sean las mejores, necesitamos tener un sanity check de estos factores."

10:52 - 10:55 [Jeffrey]: "Un sanity check. Ya, perfecto."

10:59 - 11:00 [Jeffrey]: "¿Continúo?"

11:01 - 11:53 [Jeffrey]: "Ya, es buena idea. Lo siguiente es la recompensa y las métricas. Aunque experimenté el semestre pasado con ciertas fórmulas de recompensa, con más experimentación nos daremos cuenta de si realmente satisfacen el entorno. Esto va de la mano con comprobar si esas fórmulas son adecuadas y si realmente recuperan el AUC. Capaz con más experimentación podremos definir una mejor forma."

11:56 - 12:21 [Ariana]: "Una vez más, les aconsejo utilizar la recompensa más sencilla. La que tú mencionas, el AUC directamente, puede que no sea suficiente, pero por ahora nos va a servir como evidencia. Una recompensa simple podría ser el AUC multiplicado por la inversa del costo de tiempo, con un factor de uno."

12:22 - 12:35 [Jeffrey]: "Ok. Sí, se parece bastante a la recompensa que habíamos propuesto el semestre pasado. Vamos a ver qué tal le va y sería cuestión de variar algunos parámetros."

12:36 - 12:48 [Ariana]: "Recuérdame algo. Si bien ya lo habíamos conversado y lo teníamos más o menos alineado, aún no habíamos entrenado, ¿cierto? ¿O ya teníamos algún entrenamiento?"

12:50 - 12:58 [Jeffrey]: "No, no. Los resultados fueron simulados, entonces el entrenamiento no era representativo."

13:00 - 13:17 [Ariana]: "Ok, prioricemos eso. De hecho, yo lo pondría después de compatibilidad a nivel de prioridad. Necesitamos asegurarnos de que la versión mínima funcione."

13:19 - 13:25 [Jeffrey]: "Perfecto, sí. Ahorita lo ajusto. Ok."

13:26 - 14:11 [Jeffrey]: "Este siguiente gap se me ocurrió porque estoy llevando un curso de reinforcement learning y nos han presentado distintos tipos de aprendizaje de agentes. Me entró la curiosidad de si Proximal Policy Optimization será la técnica ideal. Estuve investigando y, al menos para un entorno como el planteado en la tesis, no está demostrado que PPO sea ideal. Entonces surgió la pregunta de qué pasaría si utilizamos algún método off-policy."

14:12 - 14:44 [Jeffrey]: "Capaz sea más barato; [inaudible]. Es algo más investigativo, para research, pero estaba planteando empezar comparando DQN con PPO. Capaz uno tenga algunas ventajas frente al otro. ¿Debería empezar con uno, ignorar la comparación y justificar por qué se escogió solamente ese algoritmo?"

14:45 - 14:58 [Ariana]: "Eso es importante, pero no es el foco de nuestra tesis ahora. Es algo que nos gustaría hacer, pero para mí es un adicional; si nos da tiempo, sería interesante."

15:00 - 15:27 [Ariana]: "Seguro ya lo verás en clase: PPO tiene la desventaja de que su convergencia es más difícil de alcanzar y requerirá más episodios. Bajo esta idea, quizá sería interesante iniciar con DQN, dado que también trabaja en entornos discretos. La razón por la cual elegimos PPO es que esperamos que en algún momento nuestras acciones puedan estar en un entorno continuo, aunque todavía no sea el caso."

15:28 - 15:52 [Ariana]: "Entonces podríamos iniciar con DQN, que puede tener un mejor grado de convergencia y ser útil en esta primera etapa. Yo creo que sí: utilicemos DQN solo para asegurarnos de que todo esté en orden y que el algoritmo funcione, porque podríamos tener problemas con la convergencia de PPO que no estén relacionados con nuestro entorno."

15:54 - 16:17 [Ariana]: "Pero me saltaría la comparación. En este punto, donde todavía necesitamos validar si nuestro algoritmo aprende, es más importante tener un ambiente razonable que permita que aprenda, más allá de comparar cuál es más eficiente. En pocas palabras, necesitamos una solución antes de optimizar la solución."

16:17 - 16:25 [Jeffrey]: "Ok. Claro. Ya, sorry. Ok."

16:26 - 17:22 [Jeffrey]: "Esos son cinco gaps. Hay otros cinco que no mencionamos en la evaluación grupal, pero teníamos la duda. Yo había propuesto que el POMDP evaluara métricas generales de las distribuciones de datos. La pregunta es si esas métricas son suficientes y si puedo agregar alguna más representativa que ayude al agente. Esto también está alineado con la experimentación, porque podremos comprobar si agregar o quitar algún valor del environment que necesita el agente mejora o no el resultado. ¿Qué opinas?"

17:24 - 17:51 [Ariana]: "Esto es interesante, pero, una vez más, tenemos que pensar en la forma más simple de resolver el problema. Ve con el diagnóstico más tonto: PSI. Si funciona, perfecto, no necesitamos seguir intentando. Si vemos que no es capaz de capturar la diferencia, repensamos el problema, pero no añadamos complejidad antes de que sea necesaria."

17:53 - 18:17 [Jeffrey]: "Sí, tiene razón. Luego, el siguiente gap [inaudible]. No, espera, creo que se relaciona mucho con otro gap. A ver, el gap siete."

18:21 - 19:10 [Jeffrey]: "Este se relaciona con el gap de continuidad, de tener entornos secuenciales. El siguiente gap es la generalización. El semestre pasado fue bastante útil explorar datasets adecuados para la experimentación. Uno tiene mayormente variables categóricas y otro, variables numéricas. Si queremos tener establecidos esos experimentos, un gap sería llevarlos a más datasets como un trabajo posterior, cuando ya estemos estables con los demás puntos."

19:14 - 19:20 [Dayane]: "De acuerdo, yo voy a abarcar tanto generalización como reproducibilidad."

19:20 - 19:26 [Ariana]: "Esas dos son interesantes cuando ya tengamos resultados positivos. Primero tengamos resultados."

19:27 - 20:11 [Jeffrey]: "Sí. El siguiente gap salió porque el agente que tiene acceso a mi código me dijo que lo centralice un poco. El semestre pasado tengo módulos y scripts Python separados: un módulo de extracción de datos y una clase Environment para el agente. La idea sería terminar de establecer esos códigos para empezar con la experimentación y hacerla más reproducible; es decir, definir una secuencia de pasos, ejecutar un solo archivo y que corran los experimentos."

20:14 - 20:39 [Ariana]: "Eso está muy bien y son buenas prácticas de ingeniería de software, pero aquí necesitamos velocidad de experimentación. Si el código es feo o va en contra de todas las buenas prácticas que nos han enseñado durante la carrera, pero nos ayuda a obtener resultados ahora, está bien."

20:42 - 21:28 [Ariana]: "Vamos a empezar a preocuparnos por la reproducibilidad y por mantener la calidad del código cuando ya tengamos resultados positivos. Eso nos va a quitar tiempo y ahora necesitamos ser rápidos. Mucho del código que hagan, optimicen y empaqueten quizá ni siquiera lo necesitemos porque vamos a estar iterando y muchas soluciones iniciales van a ser malas. Mi recomendación es que no ocupemos mucho tiempo haciendo código reproducible o siguiendo esas buenas prácticas en esta etapa; hasta que tengamos resultados positivos, no es una prioridad."

21:31 - 21:56 [Jeffrey]: "Ya, perfecto. El décimo gap dice que la tesis está incompleta porque falta redactar el abstract y, mejor dicho, la parte de resultados. Es algo que se hará después. Incluso creo que ya pasaste las fechas para ir enviando los escritos y los informes de estado de la tesis."

21:57 - 22:18 [Ariana]: "Exacto. Está incompleta porque todavía no tenemos resultados y va a seguir incompleta. No se estresen demasiado por eso ahora, pero quiero que sean conscientes de que, justamente para completarla, tenemos que ser veloces. Hoy también les envié un anuncio."

22:20 - 22:50 [Ariana]: "Conversando con sus compañeros, decidí que la entrega sea opcional. No quiero presionarlos porque me interesa que se enfoquen en la experimentación, pero tendremos ciertos checkpoints donde revisaré capítulos particulares por si quieren actualizarlos. En algunas tesis requiere menos esfuerzo porque es parecido a lo que ya tienen escrito, pero no se sientan presionados a hacerlo para cada fecha. Si no llegan o están priorizando experimentos, prioricen los experimentos; al final del semestre haremos una revisión completa."

22:52 - 22:57 [Jeffrey]: "Ya, perfecto: experimentos. Ok."

23:03 - 23:45 [Jeffrey]: "Como tuvimos poco tiempo el fin de semana, el plan lo generó la IA, pero vamos a aprovechar lo que queda. Tenemos miércoles, jueves y viernes para plantear mejor la estructura del roadmap, los gaps y la forma de abordarlos. Justo, como mencionas, primero iría el contrato Stage 1–Stage 2 y luego las acciones; con el feedback que nos has dado vamos a replantearlo porque hay prioridades diferentes."

23:48 - 23:51 [Ariana]: "Bien, perfecto. Entonces sigamos."

23:51 - 25:01 [Jeffrey]: "Finalmente, esta semana tendremos un mejor plan para dividirnos las ideas. Osman tenía unos problemas con trámites y postergó la reunión, lo que nos da más tiempo para plantear las preguntas. Voy a conversar el horario; quizá el próximo jueves a las cinco nos quede bien. Queremos preguntarle por su experiencia trabajando con modelos de reinforcement learning en la industria para que nos cuente su experiencia y podamos mejorar nuestro diseño. Estas preguntas serán mejor abordadas después de establecer bien el plan de acción durante esta semana."

25:03 - 25:32 [Ariana]: "Aprovechen y sean específicos. Preguntar solamente cuál es su experiencia con reinforcement learning en la industria no les va a dar mucha información. Podrían preguntar: ¿cómo diseñamos una buena función de recompensa?, ¿qué tipo de entorno han utilizado?, ¿cómo diseñan la data?, ¿es data real o sintética? Sean específicos e intenten obtener información que también ayude a minimizar nuestras propias incertidumbres."

25:36 - 25:55 [Jeffrey]: "Sí, tiene razón: que las preguntas estén más relacionadas. En mis clases de reinforcement learning mencionan que una de las cosas más complicadas es definir el entorno, además de la política y la estrategia."

25:59 - 26:24 [Ariana]: "Preocúpense también por entender exactamente qué está haciendo o cómo ha utilizado reinforcement learning. Tengo la impresión de que quizá no lo ha utilizado en un entorno similar al nuestro, por lo que vi de su trabajo, pero puedo estar equivocada. Por el propósito educativo de la reunión, probablemente, si me uno, lo haga como oyente, a menos que vea necesaria mi intervención."

26:24 - 26:32 [Ariana]: "Los voy a dejar liderar bastante esta reunión. Prepárense bien y lo discutimos la próxima semana, también antes, para que estén listos, ¿ok?"

26:33 - 26:37 [Jeffrey]: "Ya, perfecto. Igual está buenazo recibir todo el feedback posible."

26:38 - 26:43 [Ariana]: "Perfecto. Muy bien, ¿qué vamos a hacer para la próxima semana, chicos? Cuéntame, ¿cuáles van a ser nuestras prioridades?"

26:45 - 27:04 [Jeffrey]: "Como mencioné, primero vamos a investigar la compatibilidad y, de paso, vamos a dividirnos las tareas de cada uno de los gaps que hemos hablado. Nuestra idea es llegar mañana con una propuesta de solución para el primero."

27:07 - 27:11 [Ariana]: "Ok, aquí les tengo una recomendación adicional: resolvamos uno e intentemos tener dos, ¿ok?"

27:13 - 27:30 [Ariana]: "Hagamos todo el esfuerzo por tener un ambiente donde el modelo entrene. La recompensa puede ser básica, el modelo puede ser básico y pueden utilizar las librerías que requieran, pero necesitamos un modelo aprendiendo para la próxima semana."

27:34 - 27:58 [Ariana]: "Este proyecto tiene dos partes bastante grandes, así que es bueno que sean dos personas y que cada uno pueda obsesionarse con una. Una es el entorno de aprendizaje por refuerzo: cómo elegimos el algoritmo, la función de recompensa y las entradas. El otro gran punto es la generación de datos."

27:59 - 28:17 [Ariana]: "¿Cómo hacemos para identificar los drifts? Tenemos el drift sintético, el concept drift y el otro, el feature drift; es decir, el covariate shift. Perfecto."

28:18 - 28:39 [Ariana]: "La idea es que cada uno se concentre en un problema. Por ejemplo, se puede avanzar en la parte de reinforcement learning sin importar todavía qué tan buena sea la parte de covariate shift. Así podrán avanzar de manera independiente y el progreso de uno impactará positivamente al otro, pero el problema de uno no lo afectará negativamente."

28:40 - 28:57 [Ariana]: "Eso quiere decir que al menos ya tienen una base sobre la cual trabajar. Discutan con cuidado cómo pueden paralelizar esto para aprovecharse al máximo entre los dos y utilizar al máximo sus capacidades. Lo conversamos más la próxima semana. ¿Les parece?"

28:57 - 28:59 [Jeffrey]: "Sí."

28:59 - 29:14 [Jeffrey]: "Creo que vamos exactamente hacia donde nos faltaba: ver en qué se enfocará cada uno. Con tu consejo ahora sí creo que estamos más encaminados. Gracias."

29:16 - 29:37 [Ariana]: "Ok, perfecto. Cerramos con eso. Para la próxima semana espero que podamos deshacernos de la incertidumbre sobre la acción dos y que vengan al menos con una simulación. Como les digo, puede ser estúpidamente básica, pero necesitamos algo aprendiendo. ¿Listo?"

29:39 - 29:40 [Jeffrey]: "Sí, está bien."

29:41 - 29:44 [Ariana]: "Ok, gracias a ambos, chicos."

29:44 - 29:46 [Jeffrey]: "Gracias a ti. Hasta luego."

29:47 - 29:48 [Ariana]: "Perfecto, nos vemos. Hasta luego."
