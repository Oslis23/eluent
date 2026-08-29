"""Tools for loading and writing reusable package data."""

import os

from platformdirs import user_cache_dir

from .. import app_name, __version__


CACHE_DIR = user_cache_dir(
    app_name,
    version=__version__,
    appauthor=False,
)

ELUENT_CACHE = "ELUENT_CACHE"
DEFAULT_CACHE = CACHE_DIR


def resolve_cache(cache: str | None = None) -> str:
    cache_dir = cache or os.environ.get(ELUENT_CACHE, DEFAULT_CACHE)
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_data_path(
    filename: str,
    env_key: str = ELUENT_CACHE,
    default: str = DEFAULT_CACHE,
) -> str:
    cache_dir = resolve_cache(
        os.environ.get(env_key, default)
    )
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    return cache_dir, os.path.join(cache_dir, filename)
