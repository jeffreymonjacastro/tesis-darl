---
name: transcribe-asesoria
description: >-
  Orquesta el flujo completo de transcripción offline de audios de asesorías DARL en Kaggle GPU.
  Versiona el código localmente, actualiza el dataset privado de audios, ejecuta el kernel de
  faster-whisper-large-v3 y recupera el reporte final en Markdown.
---

# transcribe-asesoria

## Overview
Esta skill instruction-only guía al agente para procesar un nuevo audio de asesoría sin intervención manual. La skill actualiza el audio en un Kaggle Dataset privado, clona la versión más reciente del script de transcripción de la GPU en una nueva carpeta versionada (`vN`), y delega a Kaggle CLI la ejecución y recuperación del reporte Markdown transcrito de la reunión.

## Dependencies
- **`kaggle`**: Necesaria para todos los comandos CLI subyacentes (`datasets version`, `kernels push`, `kernels status`, `kernels output`). Las mejores prácticas y sintaxis estricta se definen allí.

## Workflow

Sigue estos pasos en orden estricto utilizando comandos de PowerShell:

### 1. Preparar Dataset de Audio
- El usuario debe proporcionar la ruta al audio a transcribir (ej. `asesorias/audio/asesoria_w5.m4a`).
- Limpia el directorio de staging eliminando cualquier archivo de audio previo en `.tmp/audio-to-script/dataset_staging/`. **Nota**: No elimines el archivo `dataset-metadata.json`.
- Copia el nuevo archivo de audio a la carpeta `.tmp/audio-to-script/dataset_staging/`.
- Actualiza el dataset remoto en Kaggle ejecutando:
  `kaggle datasets version -p .tmp/audio-to-script/dataset_staging/ -m "Nuevo audio de asesoria"`

### 2. Crear Nueva Versión Local
- Explora la carpeta `kaggle/audio_to_script/` para determinar cuál es la última versión (ej. `v1`, `v2`).
- Crea la siguiente versión sumando 1 (ej. si la última es `v1`, crea `v2`).
- Crea la estructura de carpetas `kaggle/audio_to_script/v{N}/input/` y `kaggle/audio_to_script/v{N}/outputs/`.
- Copia los archivos `kernel-metadata.json` y `transcribe_gpu.py` desde la carpeta `input/` de la versión anterior a la nueva carpeta `input/`.

### 3. Ejecutar Kernel en Kaggle
- Navega a la nueva carpeta `input/`: `cd kaggle/audio_to_script/v{N}/input/`
- Empuja el kernel solicitando una GPU T4:
  `kaggle kernels push -p . --accelerator NvidiaTeslaT4`
- Lee la salida del comando para obtener el slug del kernel ejecutado (ej. `jeffreyamc/darl-audio-transcribe`).

### 4. Monitorear Ejecución
- Utiliza un loop o chequeos recurrentes para verificar el estado del kernel:
  `kaggle kernels status jeffreyamc/darl-audio-transcribe`
- Espera hasta que el estado sea `COMPLETE`. Si el estado es `ERROR`, `FAILED` o `CANCELLED`, detén el flujo, extrae los logs (`kaggle kernels logs`) y notifica al usuario del error.

### 5. Descargar Outputs y Finalizar
- Descarga los resultados remotamente forzando la sobreescritura si es necesario:
  `kaggle kernels output jeffreyamc/darl-audio-transcribe -p ../outputs --force`
- Copia el archivo `.md` generado (que contiene la transcripción) desde la carpeta `outputs/` hacia la carpeta raíz `asesorias/`.
- Notifica al usuario el fin de la transcripción y bríndale la ruta local del archivo Markdown final en `asesorias/`.

## Common Mistakes
- **Borrar dataset-metadata.json:** Al limpiar la carpeta de staging, a veces los agentes borran todos los archivos. Asegúrate de borrar solo los archivos de audio (`*.m4a`, `*.wav`, `*.mp3`) y conservar `dataset-metadata.json`.
- **Olvidar acelerador GPU:** El flag `--accelerator NvidiaTeslaT4` es obligatorio en el comando `push`, de lo contrario Kaggle lo ejecutará por defecto en CPU, lo cual es ineficientemente lento o causará timeout.
- **Rutas de descarga erróneas:** Asegúrate de ejecutar el comando `output` de Kaggle desde el directorio correcto, o proporcionar una ruta relativa correcta (ej. `kaggle/audio_to_script/v{N}/outputs`) para no mezclar outputs de versiones pasadas.
