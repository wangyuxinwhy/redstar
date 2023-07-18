from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from redstar.datasets import DatasetRegistry
from redstar.pipeline import EvaluationPipeline
from redstar.types import Records

TaskRegistry: dict[str, 'Task'] = {}
FixturesDir = Path(__file__).parent / 'fixtures'


@dataclass
class Task:
    task_name: str
    dataset_name: str
    pipeline: EvaluationPipeline
    load_records_function: Callable[[], Records] | None = None
    tags: set[str] = field(default_factory=set)

    def __post_init__(self):
        TaskRegistry[self.task_name] = self
        self._records = None

    @property
    def records(self):
        if self._records is None:
            if self.load_records_function is None:
                self._records = DatasetRegistry[self.dataset_name]()
            else:
                self._records = self.load_records_function()
        return self._records
