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
    """
    Load a TableShift dataset by name, caching the processed result to disk
    so subsequent calls skip re-reading and re-featurizing the raw source
    files (e.g. Physionet's ~40k per-patient .psv files).

    input:
    - dataset_name: str, name of the dataset to load

    output:
    - dataset: TableShift dataset object
    """
    project_root = find_project_root()
    cache_dir = project_root / "data" / "raw" / "tableshift_cache"

    logger.info(f"Project root: {project_root}")
    logger.info(f"Cache dir: {cache_dir}")

    common_kwargs = dict(
        name=dataset_name,
        cache_dir=str(cache_dir),
        preprocessor_config=passthrough_preprocessor(),
    )

    # Cheap probe: build the dataset object without running the (slow)
    # feature pipeline, just to check whether a processed cache already
    # exists on disk for this name/splitter combination.
    probe = get_dataset(**common_kwargs, initialize_data=False)

    if probe.is_cached():
        logger.info(f"Cache procesada encontrada para {dataset_name}; cargando desde disco...")
        return get_dataset(**common_kwargs, use_cached=True)

    logger.info(
        f"No hay cache procesada para {dataset_name}; procesando desde los "
        "archivos raw (puede tardar varios minutos) y guardando el resultado..."
    )
    dset = get_dataset(**common_kwargs)
    dset.to_sharded(file_type="csv")
    logger.info(f"Cache procesada guardada en {dset.base_dir}")
    return dset
