from pathlib import Path


def find_project_root(start=Path.cwd()):
    """Busca la raíz del repositorio basándose en la existencia de pyproject.toml."""
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("No se encontró la raíz del repositorio")
