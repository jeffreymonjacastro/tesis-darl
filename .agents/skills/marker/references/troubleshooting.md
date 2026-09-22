# Diagnóstico de Marker

Consultar esta guía cuando una conversión no arranque, tenga mala calidad o agote recursos. Recoger primero: comando, tipo de documento, páginas afectadas, modo (`fast` o `balanced`), sistema/dispositivo y el error completo. No cambiar varias causas posibles a la vez.

## El comando o módulo no se encuentra

Ejecutar Marker dentro del entorno del proyecto:

```powershell
uv run marker_single --help
uv run python -c "import marker; print('Marker disponible')"
```

Si falla el segundo comando, verificar que `marker-pdf` esté en `pyproject.toml` y sincronizar con `uv sync`. No instalarlo globalmente para ocultar un problema del entorno del proyecto.

## Un formato no-PDF no se procesa

La instalación básica de `marker-pdf` es para PDF. Para DOCX, PPTX, XLSX, HTML o EPUB se requiere el extra `full`. Añadirlo únicamente cuando esos formatos entren en el alcance del proyecto y luego volver a sincronizar el entorno:

```powershell
uv add "marker-pdf[full]"
```

## El OCR o el servidor de inferencia no inicia

- En NVIDIA, comprobar Docker, NVIDIA Container Toolkit y el backend vLLM.
- En CPU o Apple Silicon, instalar y hacer accesible `llama-server` de llama.cpp; usar modo `fast`.
- Si hay un servidor ya administrado, configurar `SURYA_INFERENCE_URL` con su endpoint compatible.
- Si solo se necesita texto embebido de un PDF digital, probar `--disable_ocr`: no arranca el servidor, pero no sirve para escaneos ni para extraer matemáticas con OCR.

No asumir que una falla del servidor es un error de extracción. Corregir la infraestructura o usar deliberadamente la ruta sin OCR.

## Texto corrupto, PDF escaneado o ecuaciones mal extraídas

1. Revisar unas pocas páginas con `--page_range`.
2. Quitar `--disable_ocr` para que Marker pueda decidir por página cuándo usar OCR.
3. Si la capa de texto existente es engañosa, agregar `--force_ocr`.
4. Para fórmulas o documentos complejos con GPU, evaluar `--mode balanced`.
5. Si el PDF contiene OCR previo defectuoso, probar `--strip_existing_ocr` para regenerarlo.

No usar `--force_ocr` como ajuste predeterminado para una colección de PDFs digitales: aumenta el coste y el tiempo.

## Tablas, formularios o resultados complejos

Primero inspeccionar la salida Markdown/HTML de páginas representativas. El modo LLM (`--use_llm`) puede ayudar a fusionar tablas, tratar formularios y mejorar matemáticas, pero implica un proveedor configurado y potencial coste/transferencia de datos. Obtener autorización antes de activarlo y nunca escribir claves en código, notebooks ni el repositorio.

Para extraer solo tablas desde Python, usar `marker.converters.table.TableConverter`; sus resultados se emiten como bloques HTML o JSON. Para OCR exclusivo, usar `marker.converters.ocr.OCRConverter`.

## Memoria insuficiente o rendimiento pobre

- Reducir `--workers` o procesar un subconjunto con `--max_files`.
- Usar `fast` y deshabilitar extracción de imágenes si estas no son necesarias.
- Dividir documentos excepcionalmente largos antes de un lote grande, conservando una relación trazable entre partes y original.
- Usar `--skip_existing` para reanudar en vez de reprocesar resultados correctos.

No aumentar la concurrencia hasta confirmar la capacidad del servidor de inferencia: más workers pueden sobrecargar CPU, RAM o VRAM sin mejorar el rendimiento.

## Depuración reproducible

Activar `--debug` (CLI) o `{"debug": True, "output_dir": "outputs/marker/debug"}` (Python) sobre un documento pequeño. Conservar el comando, versión de `marker-pdf`, modo y página afectada junto con los artefactos de `outputs/`; esa evidencia permite distinguir un problema de entrada, OCR, layout o renderer.

Fuentes: documentación oficial de Marker y código actual de `config/parser.py`, `scripts/convert.py`, `providers/pdf.py` y `builders/line.py` en https://github.com/datalab-to/marker.
