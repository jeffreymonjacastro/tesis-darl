# Estructura LaTeX vigente

Consultar esta referencia antes de crear secciones, modificar inclusiones, insertar tablas o figuras, o compilar.

## Documento principal

- El punto de entrada es `thesis/main.tex` y usa la clase `tesisutec`, no `article`.
- La tesis usa `\frontmatter`, `\mainmatter` y capítulos con `\chapter`; no sustituirlos por `\section` sin una decisión explícita de estructura.
- `main.tex` incluye `encabezados/dedicatoria`, `encabezados/agradecimientos`, `secciones/resumen`, `secciones/abstract`, `secciones/introduccion`, `secciones/capitulo1`, `secciones/capitulo2`, `secciones/capitulo4`, `secciones/capitulo5`, `secciones/conclusiones`, `secciones/recomendaciones` y `secciones/anexos`. `capitulo3` está desactivado como opcional.
- La bibliografía usa `\bibliographystyle{IEEEtran}` y `\bibliography{referencias}`; las claves viven en `thesis/referencias.bib`.

## Recursos editoriales

- Capítulos y secciones: `thesis/secciones/`.
- Encabezados iniciales: `thesis/encabezados/`.
- Tablas LaTeX curadas: `thesis/tables/`; las provenientes de código van en `thesis/tables/generated/` tras generarse primero en `outputs/tables/`.
- Figuras manuales o diagramas: `thesis/figures/manual/`.
- Figuras experimentales: generar primero en `outputs/figures/` y exportar la versión curada a `thesis/figures/generated/`.
- Imágenes institucionales: `thesis/images/`.
- `thesis/build/` contiene productos auxiliares y PDF generados: no editarlo manualmente.

La tesis ya usa tanto `\includegraphics` como `\input` para figuras y tablas. Mantener el patrón local del capítulo que se modifica; no asumir que existe un macro de figura distinto.

## Inclusión y referencias

- Usar rutas relativas a `thesis/` en los archivos `.tex`.
- Incluir tablas reutilizables con `\input{tables/<archivo>}` cuando el capítulo ya siga ese patrón.
- Usar prefijos consistentes y únicos: `fig:`, `tab:`, `eq:`, `sec:`. Referenciar con `\ref` o `\eqref`; no escribir números manualmente.
- Usar captions descriptivos con unidades, condiciones experimentales y alcance cuando correspondan.

## Compilación

Desde `thesis/`:

```powershell
latexmk -pdf -outdir=build main.tex
```

Revisar el log y confirmar que el PDF y auxiliares se escribieron en `thesis/build/`. No editar `.aux`, `.bbl`, `.blg`, `.fls`, `.log`, `.out`, `.toc`, `.lof` o `.lot` para corregir el resultado.
