---
name: khipu
description: Conectar y trabajar en el clúster HPC Khipu de UTEC; diagnosticar acceso SSH u Open OnDemand, seleccionar recursos CPU/GPU y gestionar trabajos con Slurm, especialmente cuando urge conseguir GPU.
---

# Khipu

Skill interna de operaciones para el clúster Khipu. La documentación puede cambiar: verifica las políticas y límites actuales, y usa el estado real de Slurm como fuente de verdad. Las credenciales nunca se incluyen en esta skill, scripts, registros ni comandos; usa autenticación existente autorizada, preferiblemente llaves SSH o un agente de credenciales. No afirmes tener acceso a secretos o memoria si no están disponibles. No compartas contraseñas ni llaves con otros. Las políticas prohíben utilizar credenciales de otra persona; si no está claro que el acceso automatizado con la cuenta cumple las reglas, solicita aclaración a soporte.

1. Identifica objetivo, fecha límite, tamaño del trabajo, CPU/RAM, VRAM, compatibilidad GPU y tiempo estimado. Si falta información, estima con un piloto pequeño; no inventes disponibilidad.
2. Conecta según [conexión y diagnóstico](references/conexion.md). Nunca hagas intentos repetidos de contraseña. Si no hay credenciales o sesión disponibles, explica la limitación y cómo establecer una sesión segura.
3. Antes de reservar, consulta límites de cuenta, particiones, nodos, GRES y cola. Sigue [recursos y Slurm](references/recursos-slurm.md), en particular la ruta rápida para GPU. Distingue información observada de estimaciones; ninguna estimación de cola garantiza inicio o finalización.
4. Ejecuta el trabajo mediante Slurm y dentro del alcance solicitado. El nodo de acceso es para preparación, transferencia y gestión, no cómputo intensivo. Prefiere batch para trabajos que deban sobrevivir desconexiones. Guarda resultados y revisa estado, salida y métricas.
5. Antes de informar, comprueba que la selección cumple VRAM, RAM, CPU, tiempo y límites; explica estado real, motivo de espera, alternativas y riesgos relevantes. No afirmes haber ejecutado o probado nada que no hayas verificado.
