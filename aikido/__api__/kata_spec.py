from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class KataSpec:
    batch_preprocess: bool = field(
        default=True,
        metadata={"desc": "Whether the preprocessor should run a batch processing"}
    )
    batch_preprocess_size: int = field(
        default=1000,
        metadata={"desc": "The size of batch used by the preprocessor"}
    )
    distributed: bool = field(
        default=False,
        metadata={"desc": "Set to True if you are running in a distributed evn, e.g. using DistributedDataParallel"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"desc": "Path to the caching directory"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"desc": "Maximum length of a text input tensor"}
    )
    remove_columns: Optional[List[str]] = field(
        default=None,
        metadata={"desc": "Columns to remove from the exercise dict"}
    )
    dataloader_drop_last: bool = field(
        default=False, metadata={"desc": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=0, metadata={"desc": "Number of subprocesses to use for data loading."},
    )