import logging
from tableshift import get_dataset
from tableshift.core.features import PreprocessorConfig
from darl.utils import find_project_root

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def passthrough_preprocessor() -> PreprocessorConfig:
    return PreprocessorConfig(
        categorical_features="passthrough",
        numeric_features="passthrough",
        dropna="all",
    )


def load_dataset(dataset_name: str):
    """Load a TableShift dataset by name and return it."""
    project_root = find_project_root()
    cache_dir = project_root / "data" / "raw" / "tableshift_cache"

    logger.info(f"Project root: {project_root}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Cargando dataset {dataset_name}...")

    return get_dataset(
        name=dataset_name,
        cache_dir=str(cache_dir),
        preprocessor_config=passthrough_preprocessor(),
    )
