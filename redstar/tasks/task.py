from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lmclient import LMClient

from redstar.pipeline import EvaluationPipeline
from redstar.types import Records

TaskRegistry: dict[str, 'Task'] = {}
FixturesDir = Path(__file__).parent / 'fixtures'


@dataclass
class Task:
    task_name: str
    load_dataset_func: Callable[[], Records]
    create_pipeline_func: Callable[[LMClient], EvaluationPipeline]

    def __post_init__(self):
        TaskRegistry[self.task_name] = self
