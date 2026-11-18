# Conexión y recuperación

Fuente: https://docs.khipu.utec.edu.pe/primeros-pasos/acceder-a-khipu/ y https://docs.khipu.utec.edu.pe/ondemand/ .

- Khipu documenta SSH a `khipu.utec.edu.pe` con una cuenta activa. Usa un perfil SSH existente o `ssh <usuario>@khipu.utec.edu.pe`; el usuario es un marcador, nunca una credencial embebida. Verifica por un canal confiable la huella del servidor antes de aceptar una clave nueva; si cambió, detente e investiga.
- Usa llaves configuradas o SSH agent cuando estén disponibles. No imprimas, busques masivamente ni persistas contraseñas en archivos, variables, historial o logs. Si se requiere interacción secreta, deja que el usuario la introduzca de forma segura. No desactives la comprobación de host.
- Para un timeout, comprueba DNS, conectividad TCP al puerto 22, red y anuncios de mantenimiento; prueba una red estable. La guía indica contactar la mesa de ayuda si ocurre en campus o fuera de Perú. No diagnostiques un timeout como contraseña incorrecta.
- `Permission denied`: detén los intentos y revisa usuario, llave y método; no pruebes múltiples contraseñas. Khipu advierte que más de tres fallos consecutivos pueden bloquear la IP. `Connection refused` puede reflejar bloqueo u otra causa; no lo atribuyas automáticamente. Pide asistencia o desbloqueo mediante https://servicedesk.utec.edu.pe/ o khipu@utec.edu.pe.
- Alternativa web: https://ood.khipu.utec.edu.pe/ ofrece shell, gestor de trabajos, escritorio y Jupyter; usa las mismas credenciales y Slurm. El primer acceso documentado debe ser por SSH para crear el home; si no existe, Open OnDemand muestra “Home directory not found”. No lo presentes como solución que evita todas las fallas de cuenta o red.
- Transferencias: `scp` o `rsync` por SSH; para archivos grandes, usa rsync cuando exista. Khipu indica que sus discos no tienen backup: copia resultados valiosos fuera del clúster. Consulta https://docs.khipu.utec.edu.pe/primeros-pasos/transferir-archivos/ .
