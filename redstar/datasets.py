from typing import Callable, cast

from datasets import DatasetDict, load_dataset

from redstar.types import Records

DatasetRegistry: dict[str, Callable[..., Records]] = {}


def register_dataset(name: str):
    def wrapper(func):
        DatasetRegistry[name] = func
        return func

    return wrapper


@register_dataset('gsm8k')
def load_gsm8k_dataset(split: str = 'test') -> Records:
    dataset_dict = load_dataset('gsm8k', 'main')
    dataset_dict = cast(DatasetDict, dataset_dict)
    records = [i for i in dataset_dict[split]]
    records = cast(Records, records)
    return records
