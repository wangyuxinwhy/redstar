from enum import Enum
from typing import Callable, ClassVar, Sequence

from redstar.types import Record, Records


class ProcessorType(str, Enum):
    single = 'single'
    group = 'group'


class BaseProcessor:
    processor_type: ClassVar[ProcessorType]

    def process(self, records: Records) -> Records:
        raise NotImplementedError


class BaseSingleProcessor(BaseProcessor):
    processor_type: ClassVar[ProcessorType] = ProcessorType.single

    def _process(self, record: Record) -> Record:
        raise NotImplementedError

    def process(self, records: Records) -> Records:
        return [self._process(record) for record in records]


class BaseGroupProcessor(BaseProcessor):
    processor_type: ClassVar[ProcessorType] = ProcessorType.group

    def _process(self, records: Records) -> Records:
        raise NotImplementedError

    def group(self, records: Records) -> Sequence[Records]:
        raise NotImplementedError

    def process(self, records: Records) -> Records:
        return [record for grouped_records in self.group(records) for record in self._process(grouped_records)]


class LambdaProcessor(BaseSingleProcessor):
    def __init__(self, func: Callable[[Record], Record]):
        self.func = func

    def _process(self, record: Record) -> Record:
        return self.func(record)


class SelectKeysProcessor(BaseSingleProcessor):
    def __init__(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def _process(self, records: Record) -> Record:
        return {key: records[key] for key in self.keys}
