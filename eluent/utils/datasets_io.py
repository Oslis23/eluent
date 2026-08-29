
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Union
from functools import partial
import hashlib
import os
import tempfile

from carabiner import print_err

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from pandas import DataFrame
else:
    Dataset, DataFrame, IterableDataset = Any, Any, Any

from .package_data import resolve_cache, configure_hf_cache


DATASETS_PREFIX: str = "hf://datasets/"

def hasher(s: str, n: int = 16) -> str:
    return hashlib.sha256(s).hexdigest()[:n]

def _lock_path(
    key: str,
    cache_dir: str | None = None
) -> str:
    cache_dir = resolve_cache(cache_dir)
    locks_dir = os.path.join(cache_dir, ".locks")
    os.makedirs(locks_dir, exist_ok=True)
    h = hasher(key.encode("utf-8"))
    return os.path.join(locks_dir, f"{h}.lock")


def _load_from_file(
    filename: str, 
    cache: str | None = None
) -> Dataset:

    cache, datasets_cache, _ = configure_hf_cache(cache)
    from datasets import load_dataset, Dataset, DatasetDict
    from filelock import FileLock

    if filename.removesuffix(".gz").endswith((".csv", ".tsv", ".txt")):
        read_f = partial(
            load_dataset,
            path="csv",
            data_files=filename,
            cache_dir=datasets_cache,
            sep="," if filename.endswith((".csv", ".csv.gz")) else "\t",
        )
        lock_key = ("csv", read_f.keywords.get('sep'))
    elif filename.endswith((".arrow", ".hd5", ".json", ".parquet", ".xml")):
        _, ext = os.path.splitext(filename)
        protocol = ext.lstrip(".")
        read_f = partial(
            load_dataset,
            path=protocol,
            data_files=filename,
            cache_dir=datasets_cache,
        )
        lock_key = (protocol, "")
    elif filename.endswith(".hf"):
        read_f = partial(
            Dataset.load_from_disk, 
            dataset_path=filename,
        )
        lock_key = ("hf", "")
    else:
        raise IOError(f"Could not infer how to open '{filename}' from its extension.")

    # Cross-task lock on the shared filesystem
    lockfile = _lock_path(
        key="_".join(lock_key),
        cache=cache, 
    )
    with FileLock(lockfile, timeout=60. * 60.):
        ds = read_f()
    if isinstance(ds, DatasetDict):
        return ds["train"]
    else:
        return ds


def _load_from_dataframe(
    cls,
    dataframe: Union[DataFrame, Mapping[str, ArrayLike], Iterable[Mapping[str, ArrayLike]]],
    cache: str | None = None
) -> Dataset:
    from pandas import DataFrame

    cache = resolve_cache(cache)

    if not isinstance(dataframe, DataFrame):
        dataframe = DataFrame(dataframe)

    hash_name = hasher(dataframe.to_string())
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filename = os.path.join(tmpdir, f"{hash_name}.csv.gz")
        dataframe.to_csv(csv_filename, index=False)
        ds = _load_from_file(
            csv_filename, 
            cache=cache,
        )
    return ds


def _get_ref_chunk(
    s, 
    sep: Optional[str] = None, 
    all_seps: str = "@~:"
) -> str:
    if sep is not None:
        if sep in s:
            s = s.rpartition(sep)[-1]
        else:
            return None
    for _sep in all_seps:
        s = s.partition(_sep)[0]
    return s


def _resolve_hf_hub_dataset(
    ref: str, 
    cache: str | None = None
) -> Dataset:

    cache, datasets_cache, _ = configure_hf_cache(cache)
    from datasets import concatenate_datasets, load_dataset, DatasetDict
    from filelock import FileLock

    ref = ref.removeprefix(DATASETS_PREFIX).removeprefix("hf://")
    seps = "@~:"
    repo = _get_ref_chunk(ref, all_seps=seps)
    ver = _get_ref_chunk(ref, "@", all_seps=seps)
    split = _get_ref_chunk(ref, ":", all_seps=seps)
    config = _get_ref_chunk(ref, "~", all_seps=seps)
    
    lock_key = "::".join([
        "hf",
        repo,
        config or "",
        ver or "",
    ])
    lockfile = _lock_path(cache, lock_key)

    with FileLock(
        lockfile,
        timeout=60 * 60,
    ):
        ds = load_dataset(
            path=repo, 
            name=config, 
            split=split, 
            revision=ver, 
            cache_dir=datasets_cache,
        )
    if isinstance(ds, DatasetDict):
        ds = concatenate_datasets([v for key, v in ds.items()])
    return ds


class AutoDataset:

    def __init__(self, dataset):
        self._dataset = dataset

    @classmethod
    def load(
        cls, 
        data: str | DataFrame, 
        cache: str | None = None
    ) -> Union[Dataset, IterableDataset]:
        from datasets import load_dataset, Dataset, IterableDataset
        from pandas import DataFrame

        cache = resolve_cache(cache)

        if isinstance(data, (Dataset, IterableDataset)):
            dataset = data
        elif isinstance(data, (DataFrame, Mapping)):
            dataset = _load_from_dataframe(
                data, 
                cache=cache,
            )
        elif isinstance(data, str):
            if data.startswith("hf://"):
                dataset = _resolve_hf_hub_dataset(
                    data,
                    cache=cache,
                )
            elif os.path.exists(data):
                dataset = _load_from_file(
                    data,
                    cache=cache,
                )
            else:
                raise ValueError(
                    f"""
                    If `data` is a string, it must start with "{DATASETS_PREFIX}" or a path to an existing file. 
                    It was "{data}".
                    """
                )
        else:
            raise ValueError(
                """
                Data must be a string, Dataset, dictionary, or Pandas DataFrame.
                """
            )
        return cls(dataset)
