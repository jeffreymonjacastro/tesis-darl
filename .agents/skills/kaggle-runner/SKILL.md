---
name: kaggle-runner
description: Preparar, subir, ejecutar y recuperar notebooks o scripts de DARL en Kaggle usando Kaggle CLI. Usar cuando Codex deba organizar un objetivo en carpetas kaggle por objetivo con kaggle-metadata.json y un .ipynb o .py, ejecutar el kernel remoto, y descargar resultados a output/results o outputs/kaggle/.
---

# kaggle-runner

## Estructura

Usar esta forma por objetivo:

```text
kaggle/
└── <objetivo>/
    ├── input/
    │   ├── kaggle-metadata.json
    │   └── <notebook>.ipynb o <script>.py
    └── output/
        └── results/
```

## Reglas

- No guardar `kaggle.json`, tokens ni credenciales en el repo.
- No versionar datasets pesados ni resultados temporales.
- Mantener un objetivo por carpeta.
- Preferir scripts/notebooks reproducibles con semillas en `42`.
- Escribir resultados finales en `output/results/` y, si corresponde, consolidar en `outputs/kaggle/`.

## Metadata minima

Crear o revisar `input/kaggle-metadata.json` con campos compatibles con Kaggle CLI:

```json
{
  "id": "<usuario>/<slug>",
  "title": "<titulo>",
  "code_file": "<archivo.ipynb-o-py>",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

Usar `kernel_type: "script"` si se ejecuta un `.py`.

## Flujo CLI

1. Verificar que la CLI exista con `kaggle --version`.
2. Verificar autenticacion sin mostrar secretos.
3. Validar que `input/` contenga exactamente metadata y archivo principal.
4. Ejecutar desde `kaggle/<objetivo>/input/`.
5. Subir o actualizar kernel con Kaggle CLI segun el caso.
6. Esperar estado remoto hasta finalizacion.
7. Descargar outputs a `../output/results/`.
8. Reportar rutas y estado final.

## Validacion

Confirmar que el kernel termino sin error y que `output/results/` contiene los archivos esperados. Si Kaggle CLI falla por auth, pedir al usuario reparar credenciales sin solicitar ni imprimir tokens.
