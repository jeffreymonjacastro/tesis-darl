---
name: marker
description: "Convierte documentos con la biblioteca Python Marker: PDF a Markdown, JSON, HTML o chunks, y diagnostica problemas de OCR e inferencia. Usar para extracción estructurada de documentos; no para sintetizar su contenido en una skill."
---

# Marker

Usar Marker cuando se necesite extraer contenido y estructura de documentos. Marker produce una representación estructurada; no interpreta marcos conceptuales ni crea una skill de conocimiento. Para ese último propósito, usar `book-to-skill` después de la conversión.

## Preparación y alcance

- Usar el entorno del repositorio: `uv run marker_single ...` o `uv run python ...`. No asumir que los ejecutables estén instalados globalmente.
- Confirmar el tipo de documento, las páginas de interés, el formato de salida y si el OCR es realmente necesario. `marker-pdf` base cubre PDF; para DOCX, PPTX, XLSX, HTML o EPUB se necesita el extra `marker-pdf[full]`.
- Escribir resultados generados en `outputs/` con un subdirectorio descriptivo. No alterar el documento de entrada ni versionar los resultados.
- La primera ejecución que requiere OCR puede descargar modelos e iniciar un servidor de inferencia. Informar antes de lanzar una conversión completa; no usar una conversión de un documento grande como prueba trivial.

## Elegir la ruta

- **PDF digital, rápido y sin OCR:** `--mode fast --disable_ocr`. Conserva la capa de texto del PDF; no recupera bien escaneos ni ecuaciones como LaTeX.
- **PDF mixto, escaneado o con texto corrupto:** usar OCR (omitir `--disable_ocr`); añadir `--force_ocr` si Marker conserva una capa de texto defectuosa.
- **Ecuaciones o fidelidad máxima con GPU:** `--mode balanced`. Es más costoso que `fast`.
- **Lote:** usar `marker <carpeta>`, empezar con la concurrencia predeterminada y usar `--skip_existing` para reanudar. No incrementar `--workers` sin observar memoria y capacidad de inferencia.
- **Salida para RAG:** `--output_format chunks`; para inspección humana, Markdown; para integración programática, JSON o HTML.

## CLI

Conversión de un PDF digital:

```powershell
uv run marker_single .\literature\papers\articulo.pdf `
  --output_dir .\outputs\marker\articulo `
  --mode fast `
  --disable_ocr
```

Conversión con OCR de páginas concretas:

```powershell
uv run marker_single .\literature\papers\escaneo.pdf `
  --output_dir .\outputs\marker\escaneo `
  --mode fast `
  --page_range "0,5-10" `
  --force_ocr
```

Para un directorio:

```powershell
uv run marker .\literature\papers `
  --output_dir .\outputs\marker\lote `
  --output_format markdown `
  --skip_existing
```

## API de Python

Usar la API cuando el resultado debe pasar directamente al código, no mediante archivos intermedios:

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(artifact_dict=create_model_dict())
rendered = converter("ruta/al/documento.pdf")
markdown, _, images = text_from_rendered(rendered)
```

Configurar opciones mediante `config` al crear el convertidor. Por ejemplo, para investigar una conversión, usar `{"debug": True, "output_dir": "outputs/marker/debug"}` y revisar `rendered.metadata` para la ruta de datos de depuración.

## OCR, hardware y LLM

- El OCR/inferencia de Marker usa Surya. En NVIDIA requiere Docker más NVIDIA Container Toolkit y el backend vLLM; en CPU o Apple Silicon requiere el binario `llama-server` de llama.cpp. Si ya existe un servidor compatible, se puede usar `SURYA_INFERENCE_URL`.
- Para CPU, preferir `fast`; `--disable_ocr` evita iniciar el servidor de inferencia, pero sacrifica OCR y reconocimiento de ecuaciones.
- `--use_llm` puede mejorar tablas, formularios y matemáticas, pero requiere configurar un proveedor y credenciales. No activarlo ni introducir claves sin autorización explícita.

## Diagnóstico

Para fallos de calidad, servidor, memoria, extras de formatos no-PDF y depuración, leer [troubleshooting.md](references/troubleshooting.md) antes de cambiar configuración o dependencias.

## Fuentes vigentes

- Repositorio y documentación: https://github.com/datalab-to/marker
- Opciones instaladas: `uv run marker_single --help` y `uv run marker --help`
