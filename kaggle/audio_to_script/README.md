# Kaggle Transcription Workflow Completo (Actualizado)

El workflow para transcribir audios de asesorías usando GPU T4 de Kaggle ha sido configurado y ejecutado exitosamente. Ahora con ejecución totalmente offline mediante datasets.

## Cambios realizados

1. **Dataset Staging:** Se creó una carpeta temporal `.tmp/audio-to-script/dataset_staging/` para subir el audio.
2. **Kaggle Dataset (Audio):** Se creó el dataset privado `darl-asesoria-audio` en Kaggle.
3. **Kaggle Dataset (Modelo):** Se descargó localmente en `outputs/models/faster-whisper-large-v3` el modelo pre-entrenado y se subió como Dataset a Kaggle (`jeffreyamc/faster-whisper-large-v3`). Esto permite ejecuciones reproducibles y rápidas sin internet.
4. **Script Adaptado:** Se creó `kaggle/audio_to_script/v1/input/transcribe_gpu.py` adaptado para:
   - Descargar e instalar `faster-whisper`.
   - Utilizar el modelo `large-v3` cargándolo desde el Dataset adjunto (`/kaggle/input/faster-whisper-large-v3`) con inferencia FP16 sobre la GPU CUDA.
   - Encontrar el audio recursivamente en `/kaggle/input/`.
   - Asignar los participantes correctos ("Jeffrey, Ariana y Dayane" o "Jeffrey, Dayane y Osman") detectando el nombre del archivo.
   - Generar la salida Markdown y JSON (sin SRT como solicitado).
5. **Kaggle Kernel:** Se preparó la metadata del script local (`kernel-metadata.json`) ligando ambos datasets.

El workflow está ahora estandarizado y listo para correr audios presentes y futuros de forma consistente.
