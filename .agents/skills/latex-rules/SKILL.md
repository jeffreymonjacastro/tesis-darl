---
name: latex-rules
description: Editar, revisar, estructurar y compilar automáticamente el informe de LaTeX en español académico. Usar cuando Antigravity deba modificar secciones, estilo argumentativo, numeración, referencias cruzadas, figuras, tablas o consistencia formal del informe. Se compilará el documento usando latexmk de forma obligatoria tras cada cambio.
---

# latex-rules

## Alcance

Trabajar solo sobre fuentes LaTeX versionables dentro del repositorio, especialmente en la carpeta `thesis`, en las subcarpetas `thesis/main.tex`, `thesis/sections/`, `thesis/tables/`, `thesis/figures` e `thesis/images/`.

No editar manualmente archivos auxiliares generados en `build/`. Tras cada cambio en los archivos fuente del informe o `main.tex`, se debe ejecutar obligatoriamente la compilación con latexmk para validar que el documento compila sin errores y mantener el PDF actualizado en `build/`.

## Flujo

1. Revisar `git status --short` antes de mover, borrar o sobrescribir contenido.
2. Leer el archivo `.tex` y el contexto cercano antes de proponer cambios.
3. Mantener redacción académica en español, con tono formal, preciso y directo.
4. Preservar la terminología del proyecto: MLOps, Amazon Bedrock, AWS Lambda, AWS Step Functions, Amazon Athena, Amazon QuickSight, SPICE, Dataset QuickSight, Gini, Population Stability Index (PSI), volumetría, calidad de datos, data drift, reproceso inteligente, auto-sanación (self-healing), Time To Recovery (TTR), Centro de Excelencia de Analítica (ACoE), Interbank.
5. Evitar reescrituras amplias si el usuario pide modificar una sección puntual.
6. Mantener etiquetas, citas y comandos LaTeX existentes salvo que estén claramente rotos.

## Estilo del informe

- El documento utiliza la clase de documento `article`. Por lo tanto, no se usan capítulos (`\chapter`), sino secciones (`\section`), subsecciones (`\subsection`), subsubsecciones (`\subsubsection`), etc.
- Usar párrafos con tesis clara, evidencia y cierre conceptual.
- Evitar promesas empíricas que aún no estén implementadas o evaluadas.
- Distinguir claramente entre el diagnóstico de drift/vacíos en Athena, la orquestación del reproceso mediante Step Functions, la actualización del dataset en QuickSight, y la fase de notificación.
- Cuando se mencione una contribución o justificación, conectarla con la eficiencia operativa (ej. ahorro de horas-hombre, reducción del TTR) y el mandato regulatorio (SBS).

## Figuras y tablas

- Para insertar imágenes o figuras, **usar el comando personalizado correspondiente** (ej. `\figura` o el macro definido en la tesis):
  `\figura{nombre_archivo}{descripción_caption}{etiqueta_label}`
- **Organización de archivos y carpetas obligatoria**:
  - Las **imágenes y figuras manuales/diagramas** deben guardarse en `thesis/figures/manual/`.
  - Las **figuras generadas por código/experimentos** deben guardarse primero en `outputs/figures/` y luego exportarse a `thesis/figures/generated/`.
  - Las **tablas generadas por código/experimentos** deben guardarse primero en `outputs/tables/` y luego exportarse a `thesis/tables/generated/`.
  - No colocar archivos de imágenes, figuras o tablas sueltos en la raíz de `thesis/` o en carpetas no autorizadas.
- Para diagramas o flujogramas complejos, usar TikZ. Asegurarse de cargar e incluir la librería babel (`\usetikzlibrary{babel}`) si se usan nodos de texto en español para evitar colisiones de caracteres especiales.
- No inventar resultados numéricos. Si faltan datos o KPIs, dejar texto metodológico o un marcador explícito para el usuario.

## Compilación y Validación

- Tras cada modificación del informe o de `main.tex`, **ejecutar obligatoriamente el siguiente comando** para compilar el documento y verificar que no hay errores de sintaxis o referencias rotas:
  `latexmk -pdf -outdir=build main.tex`
- Asegurarse de que el PDF y los metadatos auxiliares se escriban en la carpeta `build/` sin generar archivos auxiliares huérfanos en la raíz.
- Si la compilación falla (por ejemplo, por paquetes o archivos faltantes), se debe corregir el problema de inmediato y reportarlo.
